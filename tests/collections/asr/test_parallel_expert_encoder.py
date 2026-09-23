# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import tarfile

import pytest
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.modules.parallel_expert_encoder import (
    CTCTimestampInputs,
    ParallelExpertEncoder,
    ParallelExpertEncoderPT,
    _clone_config,
    _default_dtype,
    _disable_dist_feature_sync,
)
from nemo.collections.speechlm2.parts.ctc_timestamp_utils import (
    CTCTimestampArtifact,
    MultiSpeakerSOTWordTimestampAligner,
    TransformerCTCDecoder,
    _disable_max_seq_length_sync,
    get_ctc_timestamp_aligner,
)

# ``@experimental`` wraps the class in a wrapt proxy, so ``__new__`` (used to build
# bare instances that skip the heavy real ``__init__``) must target the underlying
# class. Attribute access / isinstance still go through the proxy name.
_PEE = getattr(ParallelExpertEncoder, "__wrapped__", ParallelExpertEncoder)


# ----------------------------------------------------------------------------- #
# Module-level context managers / helpers
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
def test_clone_config_is_deep_and_handles_none():
    cfg = OmegaConf.create({"a": {"b": 1}})
    clone = _clone_config(cfg)
    assert clone == cfg
    clone.a.b = 2
    assert cfg.a.b == 1  # original untouched
    assert _clone_config(None) is None


@pytest.mark.unit
def test_ctc_timestamp_head_runs_only_when_generated_text_is_aligned(monkeypatch):
    encoder = _PEE.__new__(_PEE)
    nn.Module.__init__(encoder)
    encoder.ctc_timestamp_model_path = "/tmp/timestamp.pt"
    captured = {}

    class _Decoder(nn.Module):
        def forward(self, encoder_output, encoded_lengths):
            captured["decoder_args"] = (encoder_output, encoded_lengths)
            return torch.zeros(encoder_output.shape[0], encoder_output.shape[2], 3)

    class _Extractor:
        ctc_decoder = _Decoder()

        def extract_from_outputs_batch(self, **kwargs):
            captured["extract_kwargs"] = kwargs
            return [{"timestamps": []}]

    def get_extractor(actual_encoder, model_path, device):
        captured["loader_args"] = (actual_encoder, model_path, device)
        return _Extractor()

    monkeypatch.setattr(
        "nemo.collections.speechlm2.parts.ctc_timestamp_utils.get_ctc_timestamp_aligner",
        get_extractor,
    )
    asr_encoded = torch.zeros(1, 16, 4)
    asr_lengths = torch.tensor([4])
    speaker_probs = torch.zeros(1, 4, 2)
    diarization_labels = torch.zeros(1, 8, 40, dtype=torch.bool)
    diarization_labels[:, 0, 5:15] = True
    timestamp_inputs = CTCTimestampInputs(
        asr_encoded,
        asr_lengths,
        speaker_probs,
        asr_lengths,
        diarization_labels=diarization_labels,
        diarization_lengths=torch.tensor([30]),
    )
    assert "decoder_args" not in captured

    result = encoder.generate_ctc_timestamps(
        timestamp_inputs=timestamp_inputs,
        sot_transcripts=["<spk:0> hello"],
        audio_durations=[1.0],
    )

    assert result == [{"timestamps": []}]
    assert captured["loader_args"] == (
        encoder,
        "/tmp/timestamp.pt",
        torch.device("cpu"),
    )
    decoder_states, decoder_lengths = captured["decoder_args"]
    assert decoder_states is asr_encoded
    assert decoder_lengths is asr_lengths
    assert captured["extract_kwargs"]["sot_transcripts"] == ["<spk:0> hello"]
    assert captured["extract_kwargs"]["audio_durations"] == [1.0]
    assert captured["extract_kwargs"]["ctc_log_probs"].shape == (1, 4, 3)
    assert captured["extract_kwargs"]["diarization_labels"] is diarization_labels
    assert captured["extract_kwargs"]["diarization_lengths"].tolist() == [30]
    assert captured["extract_kwargs"]["diarization_frame_seconds"] == 0.01
    assert "_ctc_timestamp_capture_state" not in encoder.__dict__


@pytest.mark.unit
def test_generate_ctc_timestamps_requires_request_owned_inputs():
    encoder = _PEE.__new__(_PEE)
    nn.Module.__init__(encoder)

    with pytest.raises(TypeError, match="CTCTimestampInputs"):
        encoder.generate_ctc_timestamps(
            timestamp_inputs=None,
            sot_transcripts=["hello"],
            audio_durations=[1.0],
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_dtype", "storage_dtype", "shares_storage"),
    [
        (torch.float32, torch.bfloat16, False),
        (torch.bfloat16, torch.bfloat16, True),
        (torch.float16, torch.float16, True),
    ],
)
def test_ctc_timestamp_inputs_use_detached_half_precision_encoder_states(input_dtype, storage_dtype, shares_storage):
    encoder = _PEE.__new__(_PEE)
    nn.Module.__init__(encoder)
    states = torch.randn(1, 16, 4, dtype=input_dtype, requires_grad=True)
    speaker_probs = torch.randn(1, 4, 2, requires_grad=True)
    diarization_probs = torch.tensor(
        [[[0.9, 0.1], [0.8, 0.7], [0.2, 0.6], [0.1, 0.2]]],
        requires_grad=True,
    )
    encoder.max_speaker_count = 8
    encoder.frame_shift_seconds = 0.01

    timestamp_inputs = encoder._build_ctc_timestamp_inputs(
        states,
        torch.tensor([4]),
        speaker_probs,
        torch.tensor([4]),
        diarization_probs,
        torch.tensor([4]),
    )

    assert timestamp_inputs.asr_encoded.dtype == storage_dtype
    assert (timestamp_inputs.asr_encoded.data_ptr() == states.data_ptr()) is shares_storage
    assert timestamp_inputs.sortformer_sigmoids.data_ptr() == speaker_probs.data_ptr()
    assert not timestamp_inputs.asr_encoded.requires_grad
    assert timestamp_inputs.asr_encoded.grad_fn is None
    assert not timestamp_inputs.sortformer_sigmoids.requires_grad
    assert timestamp_inputs.sortformer_sigmoids.grad_fn is None
    assert timestamp_inputs.diarization_labels.dtype == torch.bool
    assert timestamp_inputs.diarization_labels.shape == (1, 8, 4)
    assert timestamp_inputs.diarization_labels[0, :2].tolist() == [
        [True, True, False, False],
        [False, True, True, False],
    ]
    assert not timestamp_inputs.diarization_labels[:, 2:].any()
    assert not timestamp_inputs.diarization_labels.requires_grad
    assert timestamp_inputs.diarization_lengths.tolist() == [4]
    assert timestamp_inputs.diarization_frame_seconds == 0.01


@pytest.mark.unit
def test_deferred_ctc_head_uses_bounded_encoder_state_windows():
    encoder = _PEE.__new__(_PEE)
    nn.Module.__init__(encoder)
    encoder.online_inference_length = 3
    encoder.chunk_left_context = 1
    encoder.chunk_right_context = 1
    decoder = _FakeCTCDecoder()
    states = torch.zeros(2, 16, 7)
    timestamp_inputs = CTCTimestampInputs(
        asr_encoded=states,
        asr_encoded_lengths=torch.tensor([7, 5]),
        sortformer_sigmoids=torch.zeros(2, 7, 2),
        sortformer_lengths=torch.tensor([7, 5]),
    )

    ctc_log_probs = encoder._decode_ctc_timestamp_inputs(decoder, timestamp_inputs)

    assert ctc_log_probs.shape == (2, 7, 3)
    assert [length.tolist() for length in decoder.calls] == [[4, 4], [5, 3], [2, 0]]


@pytest.mark.unit
def test_ctc_timestamp_loader_disables_distributed_length_sync(monkeypatch, tmp_path):
    adapter_path = tmp_path / "adapter.pt"
    adapter_path.touch()
    decoder = nn.Linear(4, 4)
    decoder.sync_max_audio_length = True
    adapter = CTCTimestampArtifact(decoder=decoder, tokenizer=object(), decoder_config={})
    monkeypatch.setattr(
        "nemo.collections.speechlm2.parts.ctc_timestamp_utils.load_ctc_timestamp_artifact",
        lambda *args, **kwargs: adapter,
    )
    encoder = _PEE.__new__(_PEE)
    nn.Module.__init__(encoder)

    extractor = get_ctc_timestamp_aligner(encoder, str(adapter_path), torch.device("cpu"))

    assert extractor.ctc_decoder is decoder
    assert not decoder.sync_max_audio_length


@pytest.mark.unit
def test_inference_ctc_decoder_skips_collective_when_distributed_is_initialized(
    monkeypatch,
):
    decoder = TransformerCTCDecoder(
        feat_in=32,
        num_classes=5,
        use_transformer=True,
        n_heads=2,
        n_layers=1,
        drop_rate=0.0,
        ff_expansion=0.5,
        self_attention_model="rope",
        sync_max_audio_length=True,
    ).eval()
    _disable_max_seq_length_sync(decoder)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def unexpected_collective(*args, **kwargs):
        raise AssertionError("inference-only CTC decoding must not enter a distributed collective")

    monkeypatch.setattr(torch.distributed, "all_reduce", unexpected_collective)
    decoder.transformer.update_max_seq_length(7, torch.device("cpu"))

    assert not decoder.transformer.sync_max_audio_length


@pytest.mark.unit
@pytest.mark.parametrize("target_dtype", [torch.float64, torch.float16])
def test_default_dtype_sets_and_restores(target_dtype):
    prev = torch.get_default_dtype()
    with _default_dtype(target_dtype):
        assert torch.get_default_dtype() == target_dtype
    assert torch.get_default_dtype() == prev


@pytest.mark.unit
@pytest.mark.parametrize("noop_dtype", [torch.get_default_dtype(), torch.int32])
def test_default_dtype_noop_paths(noop_dtype):
    # Same-dtype and non-floating dtype are both no-ops.
    prev = torch.get_default_dtype()
    with _default_dtype(noop_dtype):
        assert torch.get_default_dtype() == prev
    assert torch.get_default_dtype() == prev


@pytest.mark.unit
def test_disable_dist_feature_sync_noop_when_uninitialized():
    assert not dist.is_initialized()
    orig = dist.is_initialized
    with _disable_dist_feature_sync():
        pass
    assert dist.is_initialized is orig  # nothing patched when dist is down


@pytest.mark.unit
@pytest.mark.parametrize("use_transformer", [True, False])
def test_transformer_ctc_decoder_modes(use_transformer):
    torch.manual_seed(7)
    decoder = TransformerCTCDecoder(
        feat_in=32,
        num_classes=5,
        use_transformer=use_transformer,
        n_heads=2,
        n_layers=1,
        drop_rate=0.0,
        ff_expansion=0.5,
        self_attention_model="rope",
    ).eval()
    lengths = torch.tensor([7, 4])
    states = torch.randn(2, 32, 7)

    with torch.no_grad():
        log_probs = decoder(encoder_output=states, encoded_lengths=lengths if use_transformer else None)

    assert log_probs.shape == (2, 7, 6)
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(2, 7), atol=1e-5)
    assert decoder.requires_encoded_lengths is use_transformer
    assert (decoder.transformer is not None) is use_transformer
    assert isinstance(decoder.decoder_layers[0], nn.Conv1d)

    if use_transformer:
        changed_padded_states = states.clone()
        changed_padded_states[1, :, 4:] = torch.randn_like(changed_padded_states[1, :, 4:]) * 100
        with torch.no_grad():
            reference = decoder(encoder_output=states, encoded_lengths=lengths)
            actual = decoder(encoder_output=changed_padded_states, encoded_lengths=lengths)
        assert torch.allclose(reference[1, :4], actual[1, :4], atol=1e-6)

        with pytest.raises(ValueError, match="requires encoded_lengths"):
            decoder(encoder_output=states, encoded_lengths=None)


@pytest.mark.unit
def test_parse_sot_words_retains_turns_and_assigns_untagged_single_speaker():
    words = MultiSpeakerSOTWordTimestampAligner.parse_sot_words("<spk:0> hello <spk:1> yes <spk:0> again")
    assert [(word["speaker_tag"], word["turn_index"]) for word in words] == [
        (0, 0),
        (1, 1),
        (0, 2),
    ]

    untagged = MultiSpeakerSOTWordTimestampAligner.parse_sot_words("hello world")
    assert [word["speaker_tag"] for word in untagged] == [0, 0]


@pytest.mark.unit
def test_timestamp_extractor_batches_all_speakers_in_parallel(monkeypatch):
    blank_id = 2
    token_ids = {"a": 0, "b": 1}

    def tokenize_words(words, blank):
        assert blank == blank_id
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    labels = [blank_id, 0, blank_id, 1, blank_id]
    logits = torch.full((len(labels), blank_id + 1), -12.0)
    for frame, label in enumerate(labels):
        logits[frame, label] = 12.0

    extractor = MultiSpeakerSOTWordTimestampAligner(blank_id=blank_id)
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    calls = []
    original = extractor._ctc_forced_align_batched

    def record_batch(*args, **kwargs):
        call_labels = args[1] if args else kwargs["labels"]
        calls.append(call_labels.shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(extractor, "_ctc_forced_align_batched", record_batch)
    result = extractor.extract_from_outputs_batch(
        ctc_log_probs=torch.log_softmax(logits, dim=-1).unsqueeze(0),
        sortformer_sigmoids=None,
        sot_transcripts=["<spk:0> a <spk:1> b"],
    )[0]

    # Without speaker probabilities, the CTC-only paths are rendered directly.
    assert calls == [2]
    assert result["alignment_mode"] == "parallel"
    assert set(result["speaker_word_timestamps"]) == {0, 1}


@pytest.mark.unit
def test_timestamp_extractor_runs_untagged_single_speaker_in_parallel(monkeypatch):
    blank_id = 1

    def tokenize_words(words, blank):
        assert blank == blank_id
        return [dict(word, token_ids=[0]) for word in words]

    logits = torch.full((5, 2), -12.0)
    for frame, label in enumerate([blank_id, 0, blank_id, 0, blank_id]):
        logits[frame, label] = 12.0
    extractor = MultiSpeakerSOTWordTimestampAligner(blank_id=blank_id)
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    calls = []
    original = extractor._ctc_forced_align_batched

    def record_batch(*args, **kwargs):
        call_labels = args[1] if args else kwargs["labels"]
        calls.append(call_labels.shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(extractor, "_ctc_forced_align_batched", record_batch)

    result = extractor.extract_from_outputs_batch(
        ctc_log_probs=torch.log_softmax(logits, dim=-1).unsqueeze(0),
        sortformer_sigmoids=None,
        sot_transcripts=["hello world"],
    )[0]

    assert result["alignment_mode"] == "parallel"
    assert calls == [1]
    assert [row["word"] for row in result["speaker_word_timestamps"][0]] == [
        "hello",
        "world",
    ]


@pytest.mark.unit
def test_compact_batched_ctc_alignment_handles_repeated_tokens():
    blank_id = 1
    target = torch.tensor([[blank_id, 0, blank_id, 0, blank_id]])
    frame_labels = [blank_id, 0, blank_id, 0, blank_id]
    logits = torch.full((len(frame_labels), 2), -12.0)
    for frame, label in enumerate(frame_labels):
        logits[frame, label] = 12.0

    paths, scores = MultiSpeakerSOTWordTimestampAligner()._ctc_forced_align_batched(
        torch.log_softmax(logits, dim=-1),
        target,
        torch.tensor([target.shape[1]]),
        blank_id,
        torch.full_like(target, -1),
        None,
        0.0,
    )

    assert target[0, paths[0]].tolist() == frame_labels
    assert len(scores) == 1


@pytest.mark.unit
@pytest.mark.parametrize(("speaker_weight", "expected_dp_calls"), [(0.0, 1), (0.25, 2)])
def test_timestamp_extractor_maps_speakers_from_preliminary_ctc_paths(monkeypatch, speaker_weight, expected_dp_calls):
    blank_id = 2
    token_ids = {"a": 0, "b": 1}

    def tokenize_words(words, blank):
        assert blank == blank_id
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    frame_labels = [blank_id, 0, 0, blank_id, 1, 1, blank_id]
    logits = torch.full((len(frame_labels), blank_id + 1), -12.0)
    for frame, label in enumerate(frame_labels):
        logits[frame, label] = 12.0
    speaker_probs = torch.tensor(
        [
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.5, 0.5],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
        ]
    )
    extractor = MultiSpeakerSOTWordTimestampAligner(blank_id=blank_id, speaker_logprob_weight=speaker_weight)
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    calls = 0
    original = extractor._ctc_forced_align_batched

    def record_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(extractor, "_ctc_forced_align_batched", record_call)

    result = extractor.extract_from_outputs_batch(
        ctc_log_probs=torch.log_softmax(logits, dim=-1).unsqueeze(0),
        sortformer_sigmoids=speaker_probs.unsqueeze(0),
        sot_transcripts=["<spk:0> a <spk:1> b"],
    )[0]

    assert result["speaker_tag_to_sortformer_column"] == {0: 1, 1: 0}
    assert result["alignment_mode"] == "parallel"
    assert calls == expected_dp_calls


@pytest.mark.unit
def test_timestamp_extractor_keeps_parallel_mode_when_speakers_exceed_columns(
    monkeypatch,
):
    blank_id = 2
    token_ids = {"a": 0, "b": 1}

    def tokenize_words(words, blank):
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    frame_labels = [blank_id, 0, blank_id, 1, blank_id]
    logits = torch.full((len(frame_labels), blank_id + 1), -12.0)
    for frame, label in enumerate(frame_labels):
        logits[frame, label] = 12.0
    extractor = MultiSpeakerSOTWordTimestampAligner(blank_id=blank_id)
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)

    result = extractor.extract_from_outputs_batch(
        ctc_log_probs=torch.log_softmax(logits, dim=-1).unsqueeze(0),
        sortformer_sigmoids=torch.ones(1, len(frame_labels), 1),
        sot_transcripts=["<spk:0> a <spk:1> b"],
    )[0]

    assert result["alignment_mode"] == "parallel"
    assert result["speaker_tag_to_sortformer_column"] == {0: None, 1: None}


@pytest.mark.unit
def test_timestamp_extractor_batch_honors_record_lengths(monkeypatch):
    blank_id = 2
    token_ids = {"a": 0, "b": 1}

    def tokenize_words(words, blank):
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    def log_probs(frame_labels, padded_frames=7):
        logits = torch.full((padded_frames, blank_id + 1), -12.0)
        for frame, label in enumerate(frame_labels):
            logits[frame, label] = 12.0
        return torch.log_softmax(logits, dim=-1)

    extractor = MultiSpeakerSOTWordTimestampAligner(blank_id=blank_id)
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    dp_calls = []
    original = extractor._ctc_forced_align_batched

    def record_dp_batch(*args, **kwargs):
        dp_calls.append((args[0].shape, kwargs["frame_lengths"].tolist()))
        return original(*args, **kwargs)

    monkeypatch.setattr(extractor, "_ctc_forced_align_batched", record_dp_batch)
    diarization_labels = torch.zeros(2, 8, 7, dtype=torch.bool)
    diarization_labels[0, 0, :2] = True
    diarization_labels[1, 1, 1:4] = True
    results = extractor.extract_from_outputs_batch(
        ctc_log_probs=torch.stack(
            [
                log_probs([blank_id, 0, 0, blank_id]),
                log_probs([blank_id, 1, 1, blank_id, blank_id]),
            ]
        ),
        sortformer_sigmoids=None,
        sot_transcripts=["<spk:0> a", "<spk:1> b"],
        ctc_lengths=torch.tensor([4, 5]),
        diarization_labels=diarization_labels,
        diarization_lengths=torch.tensor([4, 5]),
        diarization_frame_seconds=0.01,
    )

    assert [result["num_ctc_frames"] for result in results] == [4, 5]
    assert [result["alignment_mode"] for result in results] == ["parallel", "parallel"]
    assert dp_calls == [(torch.Size([2, 7, 3]), [4, 5])]
    assert results[0]["diarization_timestamps"] == [{"speaker": 0, "start": 0.0, "end": 0.02}]
    assert results[1]["diarization_timestamps"] == [{"speaker": 1, "start": 0.01, "end": 0.04}]
    assert [result["diarization_frame_seconds"] for result in results] == [0.01, 0.01]
    assert [result["diarization_max_speaker_count"] for result in results] == [8, 8]
    assert [result["diarization_activity_threshold"] for result in results] == [0.5, 0.5]


@pytest.mark.unit
def test_timestamp_extractor_rejects_mismatched_batch_metadata():
    extractor = MultiSpeakerSOTWordTimestampAligner(blank_id=2)
    with pytest.raises(ValueError, match="one string per batch item"):
        extractor.extract_from_outputs_batch(
            ctc_log_probs=torch.zeros(2, 4, 3),
            sortformer_sigmoids=None,
            sot_transcripts=["only one"],
        )


@pytest.mark.unit
def test_transformer_ctc_decoder_rejects_unsupported_modes():
    with pytest.raises(ValueError, match="residual connections require use_transformer=True"):
        TransformerCTCDecoder(feat_in=32, num_classes=5, use_transformer=False, residual=True)

    with pytest.raises(ValueError, match="d_model to equal feat_in"):
        TransformerCTCDecoder(feat_in=32, num_classes=5, d_model=16)


# ----------------------------------------------------------------------------- #
# Static pure helpers on ParallelExpertEncoder
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("max_pos, dim", [(4, 8), (1, 16), (10, 4)])
def test_build_sinusoid_position_encoding(max_pos, dim):
    pe = ParallelExpertEncoder._build_sinusoid_position_encoding(max_pos, dim)
    assert pe.shape == (max_pos, dim)
    # row 0: sin(0)=0 on even indices, cos(0)=1 on odd indices
    assert torch.allclose(pe[0, 0::2], torch.zeros(dim // 2))
    assert torch.allclose(pe[0, 1::2], torch.ones(dim // 2))


@pytest.mark.unit
@pytest.mark.parametrize(
    "cur_len, target_len",
    [(3, 6), (6, 3), (5, 5), (1, 4)],
)
def test_align_diar_frames_length_and_padding(cur_len, target_len):
    n_spk = 3
    diar = torch.arange(cur_len * n_spk, dtype=torch.float32).reshape(1, cur_len, n_spk)
    out = ParallelExpertEncoder._align_diar_frames(diar, target_len)
    assert out.shape == (1, target_len, n_spk)
    if target_len <= cur_len:
        # truncation keeps the leading frames unchanged
        assert torch.equal(out, diar[:, :target_len, :])
    else:
        # padding repeats the last frame
        assert torch.equal(out[:, :cur_len, :], diar)
        for t in range(cur_len, target_len):
            assert torch.equal(out[:, t, :], diar[:, -1, :])


@pytest.mark.unit
@pytest.mark.parametrize("param_dtype", [torch.float64, torch.float16])
def test_match_module_io_casts_to_param_dtype(param_dtype):
    module = nn.Linear(4, 4).to(param_dtype)
    tensor = torch.zeros(2, 4, dtype=torch.float32)
    out = ParallelExpertEncoder._match_module_io(tensor, module)
    assert out.dtype == param_dtype


@pytest.mark.unit
def test_match_module_io_paramless_module_unchanged():
    module = nn.Identity()  # no parameters
    tensor = torch.zeros(2, 4, dtype=torch.float32)
    out = ParallelExpertEncoder._match_module_io(tensor, module)
    assert out.dtype == torch.float32
    assert out is tensor


# ----------------------------------------------------------------------------- #
# forward() offline/online dispatch
# ----------------------------------------------------------------------------- #
def dispatch_stub(online_inference_length, chunk_feat_len, training):
    """Build a bare ParallelExpertEncoder with stubbed branch methods."""
    enc = _PEE.__new__(_PEE)
    nn.Module.__init__(enc)
    enc.online_inference_length = online_inference_length
    enc.chunk_feat_len = chunk_feat_len
    enc.training = training
    enc._forward = lambda **kw: "offline"
    enc._forward_online = lambda **kw: "online"
    return enc


@pytest.mark.unit
@pytest.mark.parametrize(
    "online_len, chunk_feat_len, training, n_frames, expected",
    [
        (500, 100, False, 200, "online"),  # eval + long enough -> online
        (500, 100, False, 50, "offline"),  # eval but shorter than one window
        (500, 100, True, 200, "offline"),  # training always offline
        (0, 100, False, 200, "offline"),  # online disabled
        (500, 100, False, 100, "offline"),  # exactly one window (not strictly greater)
    ],
)
def test_forward_dispatch(online_len, chunk_feat_len, training, n_frames, expected):
    enc = dispatch_stub(online_len, chunk_feat_len, training)
    audio = torch.zeros(1, 8, n_frames)
    length = torch.tensor([n_frames])
    assert enc.forward(audio, length) == expected


# ----------------------------------------------------------------------------- #
# _forward_online orchestration (stubbed ASR encoder, provided spk_targets)
# ----------------------------------------------------------------------------- #
class _FakeASR(nn.Module):
    """Minimal stand-in for the wrapped ConformerEncoder."""

    def __init__(self, d_model: int, sf: int):
        super().__init__()
        self.subsampling_factor = sf
        self.d_model = d_model
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, audio_signal, length):
        b, _, t = audio_signal.shape
        # generous frame count so the trim logic never clamps
        t_out = (t + self.subsampling_factor - 1) // self.subsampling_factor + 8
        out = torch.randn(b, self.d_model, t_out)
        return out, length // self.subsampling_factor


class _FakeCTCDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, encoder_output, encoded_lengths):
        self.calls.append(encoded_lengths.clone())
        batch_size, _, num_frames = encoder_output.shape
        return torch.full(
            (batch_size, num_frames, 3),
            float(len(self.calls)),
            device=encoder_output.device,
        )


def online_stub(d_model, n_spk, sf, win, lc, rc):
    enc = _PEE.__new__(_PEE)
    nn.Module.__init__(enc)
    enc.asr_encoder = _FakeASR(d_model, sf)
    enc.asr_normalize_type = None
    enc.online_inference_length = win
    enc.chunk_left_context = lc
    enc.chunk_right_context = rc
    enc.chunk_feat_len = win * sf
    enc.left_ctx_feat_len = lc * sf
    enc.right_ctx_feat_len = rc * sf
    enc.freeze_asr = True
    enc.freeze_diar = False  # The stub has no `diarization_model`, so `freeze_diar` must be False to keep
    enc.n_spk = n_spk
    enc.max_speaker_count = 8
    enc.frame_shift_seconds = 0.01
    enc.speaker_feature_mode = "continuous"
    enc.speaker_activity_threshold = None
    enc.asr_norm = nn.LayerNorm(d_model)
    enc.diar_norm = nn.LayerNorm(n_spk)
    enc.register_buffer("diar_kernel", torch.randn(n_spk, d_model))
    enc._suppress_online_pbar = True

    def run_diarization(audio_signal, length, *, return_native_resolution=False):
        native = torch.zeros(audio_signal.shape[0], audio_signal.shape[-1], n_spk)
        native[..., 0] = 1.0
        if return_native_resolution:
            return None, native, length.clone()
        return native

    enc._run_diarization = run_diarization
    enc.eval()
    return enc


@pytest.mark.unit
@pytest.mark.parametrize(
    "sf, win, lc, rc, n_frames",
    [
        (8, 10, 2, 2, 240),  # 3 full chunks
        (8, 10, 0, 0, 200),  # partial last chunk, no context
        (4, 5, 1, 1, 64),  # 4 chunks, small subsampling
        (8, 50, 5, 5, 160),  # single chunk (n_frames < window)
    ],
)
def test_forward_online_output_length_telescopes(sf, win, lc, rc, n_frames):
    d_model, n_spk, b = 16, 4, 2
    enc = online_stub(d_model, n_spk, sf, win, lc, rc)

    mels = torch.randn(b, 80, n_frames)
    length = torch.tensor([n_frames] * b)
    spk_targets = torch.rand(b, 5, n_spk)  # arbitrary; aligned internally

    outputs, encoded_len = enc._forward_online(audio_signal=mels, length=length, spk_targets=spk_targets)

    expected_t = round(n_frames / sf)
    assert outputs.shape == (b, d_model, expected_t)
    assert encoded_len.tolist() == [expected_t] * b


@pytest.mark.unit
def test_forward_online_skips_ctc_head_when_capture_is_disabled():
    enc = online_stub(d_model=16, n_spk=4, sf=8, win=10, lc=2, rc=2)
    decoder = _FakeCTCDecoder()
    extractor = type("Extractor", (), {"ctc_decoder": decoder})()
    enc.__dict__["_ctc_timestamp_extractor_cache"] = ("/tmp/timestamp.pt", extractor)

    enc._forward_online(
        audio_signal=torch.randn(1, 80, 200),
        length=torch.tensor([200]),
        spk_targets=torch.rand(1, 25, 4),
    )

    assert decoder.calls == []


@pytest.mark.unit
def test_forward_online_returns_detached_encoder_states_without_running_ctc_head():
    enc = online_stub(d_model=16, n_spk=4, sf=8, win=10, lc=2, rc=2)
    decoder = _FakeCTCDecoder()
    extractor = type("Extractor", (), {"ctc_decoder": decoder})()
    enc.__dict__["_ctc_timestamp_extractor_cache"] = ("/tmp/timestamp.pt", extractor)

    _, encoded_len, timestamp_inputs = enc._forward_online(
        audio_signal=torch.randn(2, 80, 200),
        length=torch.tensor([200, 160]),
        spk_targets=torch.rand(2, 25, 4),
        return_ctc_timestamp_inputs=True,
    )

    assert decoder.calls == []
    assert timestamp_inputs.asr_encoded.shape == (2, 16, 25)
    assert timestamp_inputs.asr_encoded_lengths.tolist() == encoded_len.tolist() == [25, 20]
    assert timestamp_inputs.sortformer_sigmoids.shape == (2, 25, 4)
    assert timestamp_inputs.sortformer_lengths.tolist() == [25, 20]
    assert not timestamp_inputs.asr_encoded.requires_grad
    assert timestamp_inputs.asr_encoded.grad_fn is None
    assert "_ctc_timestamp_capture_state" not in enc.__dict__


@pytest.mark.unit
def test_forward_offline_returns_detached_encoder_states_without_second_asr_pass():
    enc = online_stub(d_model=16, n_spk=4, sf=8, win=10, lc=2, rc=2)
    decoder = _FakeCTCDecoder()
    extractor = type("Extractor", (), {"ctc_decoder": decoder})()
    enc.__dict__["_ctc_timestamp_extractor_cache"] = ("/tmp/timestamp.pt", extractor)
    asr_calls = 0
    original_asr_forward = enc.asr_encoder.forward

    def count_asr_calls(*args, **kwargs):
        nonlocal asr_calls
        asr_calls += 1
        return original_asr_forward(*args, **kwargs)

    enc.asr_encoder.forward = count_asr_calls
    _, encoded_len, timestamp_inputs = enc._forward(
        audio_signal=torch.randn(1, 80, 64),
        length=torch.tensor([64]),
        spk_targets=torch.rand(1, 8, 4),
        return_ctc_timestamp_inputs=True,
    )

    assert asr_calls == 1
    assert decoder.calls == []
    assert timestamp_inputs.asr_encoded.shape == (1, 16, 16)
    assert timestamp_inputs.asr_encoded_lengths.tolist() == encoded_len.tolist() == [8]
    assert not timestamp_inputs.asr_encoded.requires_grad
    assert timestamp_inputs.asr_encoded.grad_fn is None


@pytest.mark.unit
def test_deferred_timestamp_retention_is_twenty_four_times_smaller_than_ctc_logits():
    states = torch.empty(1, 1280, 100, dtype=torch.bfloat16)
    timestamp_inputs = CTCTimestampInputs(
        asr_encoded=states,
        asr_encoded_lengths=torch.tensor([100]),
        sortformer_sigmoids=torch.empty(1, 100, 4),
        sortformer_lengths=torch.tensor([100]),
        diarization_labels=torch.empty(1, 8, 800, dtype=torch.bool),
        diarization_lengths=torch.tensor([800]),
    )
    retained_state_bytes = sum(
        value.numel() * value.element_size()
        for value in timestamp_inputs.__dict__.values()
        if isinstance(value, torch.Tensor)
    )
    hypothetical_ctc_bytes = 100 * 32769 * states.element_size()

    assert hypothetical_ctc_bytes / retained_state_bytes > 24


# ----------------------------------------------------------------------------- #
# ParallelExpertEncoderPT.is_pe_nemo
# ----------------------------------------------------------------------------- #
def write_nemo(path, *, target=None, include_cfg=True):
    with tarfile.open(path, "w") as tf:
        if include_cfg:
            data = (f"target: {target}\n" if target is not None else "foo: bar\n").encode()
            info = tarfile.TarInfo(name="model_config.yaml")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        else:
            data = b"not a config"
            info = tarfile.TarInfo(name="weights.ckpt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


@pytest.mark.unit
@pytest.mark.parametrize(
    "target, expected",
    [
        (
            "nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT",
            True,
        ),
        ("ParallelExpertEncoderPT", True),
        ("nemo.collections.asr.models.SomethingElse", False),
        (None, False),  # model_config.yaml present but no `target`
    ],
)
def test_is_pe_nemo_by_target(tmp_path, target, expected):
    nemo_path = str(tmp_path / "bundle.nemo")
    write_nemo(nemo_path, target=target)
    assert ParallelExpertEncoderPT.is_pe_nemo(nemo_path) is expected


@pytest.mark.unit
def test_is_pe_nemo_without_model_config(tmp_path):
    nemo_path = str(tmp_path / "no_cfg.nemo")
    write_nemo(nemo_path, include_cfg=False)
    assert ParallelExpertEncoderPT.is_pe_nemo(nemo_path) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_path",
    [None, 123, "missing.nemo", "not_a_nemo.txt"],
)
def test_is_pe_nemo_rejects_bad_paths(tmp_path, bad_path):
    # a real-but-non-.nemo file to exercise the suffix check
    if bad_path == "not_a_nemo.txt":
        p = tmp_path / "not_a_nemo.txt"
        p.write_text("hello")
        bad_path = str(p)
    assert ParallelExpertEncoderPT.is_pe_nemo(bad_path) is False


# ----------------------------------------------------------------------------- #
# ParallelExpertEncoderPT.save_to_nemo guard rails
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
def test_save_to_nemo_rejects_non_encoder(tmp_path):
    with pytest.raises(TypeError):
        ParallelExpertEncoderPT.save_to_nemo(
            nn.Linear(2, 2),
            str(tmp_path / "out.nemo"),
            template_bundle_path=str(tmp_path / "tpl.nemo"),
        )


@pytest.mark.unit
def test_save_to_nemo_missing_template(tmp_path):
    # __new__ produces a real ParallelExpertEncoder instance (passes isinstance)
    # without running the heavy __init__, so we reach the template existence check.
    fake_encoder = _PEE.__new__(_PEE)
    with pytest.raises(FileNotFoundError):
        ParallelExpertEncoderPT.save_to_nemo(
            fake_encoder,
            str(tmp_path / "out.nemo"),
            template_bundle_path=str(tmp_path / "does_not_exist.nemo"),
        )


# ----------------------------------------------------------------------------- #
# End-to-end fusion with real toy encoders
#
# ParallelExpertEncoder loads two real sub-encoders and fuses them:
#   * an ASR ConformerEncoder (cf. tests/collections/asr/test_conformer_encoder.py)
#   * a Sortformer diarizer    (cf. tests/collections/speaker_tasks/test_diar_sortformer_models.py)
# These tests build tiny-but-real instances of both and run the wrapper end to end.
# ----------------------------------------------------------------------------- #
_MEL_FEATURES = 128
_ASR_D_MODEL = 32
_DIAR_FC_D_MODEL = 32
_DIAR_TF_D_MODEL = 16
_N_SPK = 4
_SUBSAMPLING_FACTOR = 8


def toy_asr_encoder_cfg() -> DictConfig:
    """Tiny ConformerEncoder config the PE encoder mounts as its ASR branch."""
    return DictConfig(
        {
            "_target_": "nemo.collections.asr.modules.ConformerEncoder",
            "feat_in": _MEL_FEATURES,
            "feat_out": -1,
            "n_layers": 1,
            "d_model": _ASR_D_MODEL,
            "subsampling": "dw_striding",
            "subsampling_factor": _SUBSAMPLING_FACTOR,
            "subsampling_conv_channels": 16,
            "ff_expansion_factor": 4,
            "self_attention_model": "rel_pos",
            "n_heads": 4,
            "att_context_size": [-1, -1],
            "conv_kernel_size": 9,
            "dropout": 0.0,
            "dropout_pre_encoder": 0.0,
            "dropout_emb": 0.0,
            "dropout_att": 0.0,
        }
    )


def toy_diarization_model_cfg() -> DictConfig:
    """Tiny SortformerEncLabelModel config the PE encoder mounts as its diar branch."""
    model_defaults = {"fc_d_model": _DIAR_FC_D_MODEL, "tf_d_model": _DIAR_TF_D_MODEL}
    return DictConfig(
        {
            "target": "nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel",
            "sample_rate": 16000,
            "pil_weight": 0.5,
            "ats_weight": 0.5,
            "max_num_of_spks": _N_SPK,
            "streaming_mode": False,
            "async_streaming": False,
            "model_defaults": DictConfig(model_defaults),
            "preprocessor": DictConfig(
                {
                    "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                    "normalize": "per_feature",
                    "window_size": 0.025,
                    "sample_rate": 16000,
                    "window_stride": 0.01,
                    "window": "hann",
                    "features": _MEL_FEATURES,
                    "n_fft": 512,
                    "frame_splicing": 1,
                    "dither": 0.00001,
                }
            ),
            "encoder": DictConfig(
                {
                    "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                    "feat_in": _MEL_FEATURES,
                    "feat_out": -1,
                    "n_layers": 1,
                    "d_model": _DIAR_FC_D_MODEL,
                    "subsampling": "dw_striding",
                    "subsampling_factor": _SUBSAMPLING_FACTOR,
                    "subsampling_conv_channels": 16,
                    "causal_downsampling": False,
                    "ff_expansion_factor": 4,
                    "self_attention_model": "rel_pos",
                    "n_heads": 4,
                    "att_context_size": [-1, -1],
                    "conv_kernel_size": 9,
                    "conv_norm_type": "batch_norm",
                    "dropout": 0.0,
                    "dropout_pre_encoder": 0.0,
                    "dropout_emb": 0.0,
                    "dropout_att": 0.0,
                }
            ),
            "transformer_encoder": DictConfig(
                {
                    "_target_": "nemo.collections.asr.modules.transformer.transformer_encoders.TransformerEncoder",
                    "num_layers": 1,
                    "hidden_size": _DIAR_TF_D_MODEL,
                    "inner_size": 32,
                    "num_attention_heads": 4,
                    "attn_score_dropout": 0.0,
                    "attn_layer_dropout": 0.0,
                    "ffn_dropout": 0.0,
                    "hidden_act": "relu",
                    "pre_ln": False,
                    "pre_ln_final_layer_norm": True,
                }
            ),
            "sortformer_modules": DictConfig(
                {
                    "_target_": "nemo.collections.asr.modules.sortformer_modules.SortformerModules",
                    "num_spks": _N_SPK,
                    "dropout_rate": 0.0,
                    "fc_d_model": _DIAR_FC_D_MODEL,
                    "tf_d_model": _DIAR_TF_D_MODEL,
                }
            ),
            "loss": DictConfig(
                {
                    "_target_": "nemo.collections.asr.losses.bce_loss.BCELoss",
                    "weight": None,
                    "reduction": "mean",
                }
            ),
        }
    )


def build_toy_pe_encoder(**overrides) -> ParallelExpertEncoder:
    """Construct a real ParallelExpertEncoder from the tiny ASR + diar configs."""
    kwargs = dict(
        asr_encoder_cfg=toy_asr_encoder_cfg(),
        diarization_model_cfg=toy_diarization_model_cfg(),
        asr_normalize_type="per_feature",
        # Keep the input far below one window so forward() stays on the offline path.
        online_inference_length=500,
    )
    kwargs.update(overrides)
    return ParallelExpertEncoder(**kwargs)


@pytest.mark.unit
def test_pe_encoder_builds_and_wires_both_real_encoders():
    enc = build_toy_pe_encoder()
    # The two fused sub-encoders are the real classes, not stubs.
    assert isinstance(enc.asr_encoder, ConformerEncoder)
    assert isinstance(enc.diarization_model, SortformerEncLabelModel)
    # ConformerEncoder-compatible drop-in properties come from the ASR branch.
    assert enc.d_model == _ASR_D_MODEL
    assert enc.subsampling_factor == _SUBSAMPLING_FACTOR
    # Speaker count + fusion kernel come from the diar branch.
    assert enc.n_spk == _N_SPK
    assert enc.max_speaker_count == 8
    assert enc.diar_kernel.shape == (_N_SPK, _ASR_D_MODEL)
    # freeze_diar defaults to True -> diar params are frozen, ASR params remain trainable.
    assert all(not p.requires_grad for p in enc.diarization_model.parameters())
    assert any(p.requires_grad for p in enc.asr_encoder.parameters())


@pytest.mark.unit
@pytest.mark.parametrize(
    "high_resolution, requested_diar_subsampling_factor, expected_asr_aligned_factor",
    [(True, 1, _SUBSAMPLING_FACTOR)],
)
def test_pe_encoder_matches_diar_output_resolution_to_asr_encoder(
    high_resolution, requested_diar_subsampling_factor, expected_asr_aligned_factor
):
    diar_cfg = toy_diarization_model_cfg()
    diar_cfg.high_resolution = high_resolution
    diar_cfg.output_subsampling_factor = requested_diar_subsampling_factor

    enc = build_toy_pe_encoder(diarization_model_cfg=diar_cfg)

    assert (
        enc.diarization_model.output_subsampling_factor
        == enc.asr_encoder.subsampling_factor
        == expected_asr_aligned_factor
    )
    assert (
        enc.diarization_model._cfg.output_subsampling_factor
        == enc.asr_encoder.subsampling_factor
        == expected_asr_aligned_factor
    )


@pytest.mark.unit
def test_pe_encoder_retains_native_10ms_diarization_labels_during_online_timestamp_capture():
    diar_cfg = toy_diarization_model_cfg()
    diar_cfg.high_resolution = True
    diar_cfg.output_subsampling_factor = 1
    enc = build_toy_pe_encoder(
        diarization_model_cfg=diar_cfg,
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
    ).eval()
    enc._suppress_online_pbar = True
    mels = torch.randn(2, _MEL_FEATURES, 160)
    lengths = torch.tensor([160, 120])

    _, encoded_lengths, timestamp_inputs = enc._forward_online(
        audio_signal=mels,
        length=lengths,
        return_ctc_timestamp_inputs=True,
    )

    assert encoded_lengths.tolist() == [20, 15]
    assert timestamp_inputs.diarization_labels.dtype == torch.bool
    assert timestamp_inputs.diarization_labels.shape == (2, 8, 160)
    assert timestamp_inputs.diarization_lengths.tolist() == [160, 120]
    assert not timestamp_inputs.diarization_labels[:, _N_SPK:].any()


@pytest.mark.unit
@pytest.mark.parametrize(
    "asr_subsampling_factor, error_match",
    [(4, "requires the diarization output subsampling factor")],
)
def test_pe_encoder_rejects_incompatible_diar_output_resolution(asr_subsampling_factor, error_match):
    asr_cfg = toy_asr_encoder_cfg()
    asr_cfg.subsampling_factor = asr_subsampling_factor

    with pytest.raises(ValueError, match=error_match):
        build_toy_pe_encoder(asr_encoder_cfg=asr_cfg)


@pytest.mark.unit
@pytest.mark.parametrize("batch_size, n_frames", [(1, 160), (2, 200)])
def test_pe_encoder_offline_forward_runs_internal_diarizer(batch_size, n_frames):
    enc = build_toy_pe_encoder().eval()
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)  # spk_targets=None -> Sortformer runs internally

    expected_t = int(encoded_len[0].item())
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()
    assert encoded_len.tolist() == [expected_t] * batch_size


@pytest.mark.unit
def test_pe_encoder_offline_forward_accepts_diar_override_and_fuses_it():
    enc = build_toy_pe_encoder().eval()
    batch_size, n_frames = 2, 160
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)

    # Arbitrary diar frame count: PE aligns it to the ASR frame count internally.
    dp1 = torch.rand(batch_size, 7, _N_SPK)
    dp2 = torch.rand(batch_size, 7, _N_SPK)

    with torch.no_grad():
        out1, len1 = enc(mels, length, spk_targets=dp1)
        out2, len2 = enc(mels, length, spk_targets=dp2)

    expected_t = int(len1[0].item())
    assert out1.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert torch.equal(len1, len2)
    assert torch.isfinite(out1).all()
    # Same audio + same (dropout-free, eval) ASR branch, but different speaker
    # predictions must change the fused output -> proves the diar branch is fused in.
    assert not torch.allclose(out1, out2)


@pytest.mark.unit
def test_pe_encoder_online_forward_matches_conformer_io_with_real_encoders():
    # Small window so a modest input crosses onto the long-form online path.
    enc = build_toy_pe_encoder(
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    ).eval()
    enc._suppress_online_pbar = True

    batch_size, n_frames = (
        1,
        320,
    )  # > online_inference_length * subsampling_factor (=80)
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames)
    length = torch.full((batch_size,), n_frames, dtype=torch.long)

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)

    expected_t = int(encoded_len[0].item())
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()


# ----------------------------------------------------------------------------- #
# GPU end-to-end fusion with real toy encoders
#
# These mirror the CPU end-to-end tests but run on CUDA. They additionally
# exercise the device/dtype-bridging machinery the wrapper exists for: fp32 mels
# fed into (optionally) bf16 experts on the GPU, handled by `_match_module_io`
# (offline) and `_default_dtype` / `_disable_dist_feature_sync` (online).
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE GPU test requires CUDA")
@pytest.mark.parametrize("batch_size, n_frames", [(1, 160), (2, 200)])
def test_pe_encoder_offline_forward_on_gpu(batch_size, n_frames):
    enc = build_toy_pe_encoder().eval().cuda()
    # Mels arrive un-normalised in fp32 (the SALM perception contract).
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)  # spk_targets=None -> Sortformer runs internally

    expected_t = int(encoded_len[0].item())
    assert outputs.is_cuda
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()
    assert encoded_len.tolist() == [expected_t] * batch_size


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="PEE bf16 GPU test requires CUDA with bf16 support",
)
def test_pe_encoder_offline_forward_bf16_experts_on_gpu():
    # Experts run in bf16 while mels stay fp32 -> exercises `_match_module_io`
    # device/dtype bridging on both branches before their conv subsampling.
    enc = build_toy_pe_encoder().eval().cuda().to(torch.bfloat16)
    batch_size, n_frames = 2, 200
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)

    expected_t = int(encoded_len[0].item())
    assert outputs.is_cuda
    assert outputs.dtype == torch.bfloat16
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert torch.isfinite(outputs).all()


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE GPU test requires CUDA")
def test_pe_encoder_offline_forward_accepts_diar_override_on_gpu():
    enc = build_toy_pe_encoder().eval().cuda()
    batch_size, n_frames = 2, 160
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    dp1 = torch.rand(batch_size, 7, _N_SPK, device="cuda")
    dp2 = torch.rand(batch_size, 7, _N_SPK, device="cuda")

    with torch.no_grad():
        out1, len1 = enc(mels, length, spk_targets=dp1)
        out2, len2 = enc(mels, length, spk_targets=dp2)

    expected_t = int(len1[0].item())
    assert out1.is_cuda
    assert out1.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert torch.equal(len1, len2)
    assert torch.isfinite(out1).all()
    # Different speaker predictions must change the fused output.
    assert not torch.allclose(out1, out2)


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="PEE GPU test requires CUDA")
def test_pe_encoder_online_forward_on_gpu():
    enc = (
        build_toy_pe_encoder(
            online_inference_length=10,
            chunk_left_context=2,
            chunk_right_context=2,
            diar_fifo_len=10,
            diar_spkcache_update_period=20,
            diar_spkcache_len=20,
        )
        .eval()
        .cuda()
    )
    enc._suppress_online_pbar = True

    batch_size, n_frames = (
        1,
        320,
    )  # > online_inference_length * subsampling_factor (=80)
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)

    expected_t = int(encoded_len[0].item())
    assert outputs.is_cuda
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()
