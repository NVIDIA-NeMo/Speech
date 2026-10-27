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

import gc
import importlib.util
import io
import tarfile
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.modules import online_encoder_graphs as online_graphs_module
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.modules.online_encoder_graphs import OnlineGraphs
from nemo.collections.asr.modules.parallel_expert_encoder import (
    CTCTimestampInputs,
    ParallelExpertEncoder,
    ParallelExpertEncoderPT,
    _clone_config,
    _default_dtype,
    _disable_dist_feature_sync,
)
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output, unpack_encoder_output
from nemo.collections.asr.parts.preprocessing.features import normalize_batch, normalize_packed_batch
from nemo.collections.common.parts.optional_cuda_graphs import capture_cuda_graph
from nemo.collections.speechlm2.parts import ctc_timestamp_utils
from nemo.collections.speechlm2.parts.ctc_timestamp_utils import (
    CTCTimestampArtifact,
    MultiSpeakerSOTWordTimestampAligner,
    TransformerCTCDecoder,
    _disable_max_seq_length_sync,
    align_prepared_batch,
    get_ctc_timestamp_aligner,
    merge_prepared_batches,
)
from tests.collections.asr.ctc_timestamp_test_utils import search_streams

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
def test_encoder_hands_generated_text_to_its_ctc_timestamp_aligner(monkeypatch):
    encoder = _PEE.__new__(_PEE)
    nn.Module.__init__(encoder)
    encoder.ctc_timestamp_model_path = "/tmp/timestamp.pt"
    captured = {}

    class _Aligner:
        def generate_from_inputs(self, timestamp_inputs, sot_transcripts, audio_durations):
            captured["generate_args"] = (timestamp_inputs, sot_transcripts, audio_durations)
            return [{"timestamps": []}]

    def get_aligner(actual_encoder, model_path, device):
        captured["loader_args"] = (actual_encoder, model_path, device)
        return _Aligner()

    monkeypatch.setattr(
        "nemo.collections.speechlm2.parts.ctc_timestamp_utils.get_ctc_timestamp_aligner",
        get_aligner,
    )
    timestamp_inputs = CTCTimestampInputs(
        torch.zeros(1, 16, 4), torch.tensor([4]), torch.zeros(1, 4, 2), torch.tensor([4])
    )

    result = encoder.generate_ctc_timestamps(
        timestamp_inputs=timestamp_inputs,
        sot_transcripts=["<spk:0> hello"],
        audio_durations=[1.0],
    )

    assert result == [{"timestamps": []}]
    assert captured["loader_args"] == (encoder, "/tmp/timestamp.pt", torch.device("cpu"))
    inputs, transcripts, durations = captured["generate_args"]
    assert inputs is timestamp_inputs and transcripts == ["<spk:0> hello"] and durations == [1.0]
    assert "_ctc_timestamp_capture_state" not in encoder.__dict__


@pytest.mark.unit
def test_deferred_ctc_head_runs_only_when_generated_text_is_aligned(monkeypatch):
    captured = {}

    class _Decoder(nn.Module):
        num_classes_with_blank = 3

        def forward(self, encoder_output, encoded_lengths):
            captured["decoder_args"] = (encoder_output, encoded_lengths)
            return torch.arange(12.0).reshape(1, 4, 3)

    def prepare(ctc_shape, read_columns, sortformer_sigmoids, sot_transcripts, **kwargs):
        captured["prepare_args"] = (ctc_shape, sortformer_sigmoids, sot_transcripts, kwargs)
        # Transcripts become targets before the head runs.
        assert "decoder_args" not in captured
        return {"records": read_columns([[2, 0]], [3])}

    aligner = MultiSpeakerSOTWordTimestampAligner(ctc_decoder=_Decoder())
    monkeypatch.setattr(aligner, "_prepare", prepare)
    asr_encoded = torch.zeros(1, 16, 4)
    asr_lengths = torch.tensor([4])
    speaker_probs = torch.zeros(1, 6, 2)
    diarization_labels = torch.zeros(1, 8, 40, dtype=torch.bool)
    diarization_labels[:, 0, 5:15] = True
    timestamp_inputs = CTCTimestampInputs(
        asr_encoded,
        asr_lengths,
        speaker_probs,
        torch.tensor([6]),
        diarization_labels=diarization_labels,
        diarization_lengths=torch.tensor([30]),
    )
    assert "decoder_args" not in captured

    prepared = aligner.prepare_from_inputs(timestamp_inputs, ["<spk:0> hello"], [1.0])

    # Columns 2 and 0 of the record's three valid frames, not the head's whole output.
    (log_probs,) = prepared["records"]
    assert torch.equal(log_probs, torch.tensor([[2.0, 0.0], [5.0, 3.0], [8.0, 6.0]]))
    decoder_states, decoder_lengths = captured["decoder_args"]
    torch.testing.assert_close(decoder_states, asr_encoded)
    torch.testing.assert_close(decoder_lengths, asr_lengths)
    ctc_shape, sortformer_sigmoids, transcripts, kwargs = captured["prepare_args"]
    assert tuple(ctc_shape) == (1, 4, 3)
    assert sortformer_sigmoids is speaker_probs and transcripts == ["<spk:0> hello"]
    assert kwargs["ctc_lengths"].tolist() == [4]
    # Speaker frames past the four CTC frames are never aligned.
    assert kwargs["sortformer_lengths"].tolist() == [4]
    assert kwargs["audio_durations"] == [1.0]
    assert kwargs["diarization_labels"] is diarization_labels
    assert kwargs["diarization_lengths"].tolist() == [30]
    assert kwargs["diarization_frame_seconds"] == 0.01


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
    decoder = _FakeCTCDecoder()
    aligner = MultiSpeakerSOTWordTimestampAligner(
        ctc_decoder=decoder,
        online_inference_length=3,
        chunk_left_context=1,
        chunk_right_context=1,
    )
    states = torch.zeros(2, 16, 7)
    timestamp_inputs = CTCTimestampInputs(
        asr_encoded=states,
        asr_encoded_lengths=torch.tensor([7, 5]),
        sortformer_sigmoids=torch.zeros(2, 7, 2),
        sortformer_lengths=torch.tensor([7, 5]),
    )

    ctc_log_probs = aligner.ctc_log_probs_from_inputs(timestamp_inputs)

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
def test_ctc_timestamp_loader_copies_the_encoder_windows_on_every_call(monkeypatch, tmp_path):
    adapter_path = tmp_path / "adapter.pt"
    adapter_path.touch()
    adapter = CTCTimestampArtifact(decoder=nn.Linear(4, 4), tokenizer=object(), decoder_config={})
    monkeypatch.setattr(
        "nemo.collections.speechlm2.parts.ctc_timestamp_utils.load_ctc_timestamp_artifact",
        lambda *args, **kwargs: adapter,
    )
    encoder = _PEE.__new__(_PEE)
    nn.Module.__init__(encoder)
    encoder.online_inference_length, encoder.chunk_left_context, encoder.chunk_right_context = 500, 50, 25

    aligner = get_ctc_timestamp_aligner(encoder, str(adapter_path), torch.device("cpu"))

    assert (aligner.online_inference_length, aligner.chunk_left_context, aligner.chunk_right_context) == (500, 50, 25)
    assert all(value is not encoder for value in vars(aligner).values())

    encoder.online_inference_length, encoder.chunk_left_context, encoder.chunk_right_context = 250, 40, 10

    assert get_ctc_timestamp_aligner(encoder, str(adapter_path), torch.device("cpu")) is aligner
    assert (aligner.online_inference_length, aligner.chunk_left_context, aligner.chunk_right_context) == (250, 40, 10)

    encoder.ctc_timestamp_alignment = {"max_word_frames": 6}
    updated = get_ctc_timestamp_aligner(encoder, str(adapter_path), torch.device("cpu"))
    assert updated is not aligner
    assert updated.ctc_decoder is aligner.ctc_decoder
    assert updated.max_word_frames == 6
    assert get_ctc_timestamp_aligner(encoder, str(adapter_path), torch.device("cpu")) is updated
    encoder.ctc_timestamp_alignment = {}
    restored = get_ctc_timestamp_aligner(encoder, str(adapter_path), torch.device("cpu"))
    assert restored.speaker_logprob_weight == 1.0 and restored.max_word_frames == 12
    assert restored.ctc_decoder is updated.ctc_decoder


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
def test_timestamp_extractor_searches_each_speaker_once(monkeypatch):
    blank_id = 2
    token_ids = {"a": 0, "b": 1}

    def tokenize_words(words, blank):
        assert blank == blank_id
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    labels = [blank_id, 0, blank_id, 1, blank_id]
    logits = torch.full((len(labels), blank_id + 1), -12.0)
    for frame, label in enumerate(labels):
        logits[frame, label] = 12.0

    extractor = MultiSpeakerSOTWordTimestampAligner(blank_id=blank_id, ctc_frame_seconds=0.01)
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    calls = []
    original = ctc_timestamp_utils._banded_ctc_search

    def record_batch(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ctc_timestamp_utils, "_banded_ctc_search", record_batch)
    result = extractor.extract_from_outputs_batch(
        ctc_log_probs=torch.log_softmax(logits, dim=-1).unsqueeze(0),
        sortformer_sigmoids=None,
        sot_transcripts=["<spk:0> a <spk:1> b"],
    )[0]

    # Without speaker probabilities, the CTC-only paths are rendered directly.
    assert calls == [1, 1]
    assert result["alignment_mode"] == "banded"
    assert set(result["speaker_word_timestamps"]) == {0, 1}


@pytest.mark.unit
def test_timestamp_extractor_searches_untagged_single_speaker_once(monkeypatch):
    blank_id = 1

    def tokenize_words(words, blank):
        assert blank == blank_id
        return [dict(word, token_ids=[0]) for word in words]

    logits = torch.full((5, 2), -12.0)
    for frame, label in enumerate([blank_id, 0, blank_id, 0, blank_id]):
        logits[frame, label] = 12.0
    extractor = MultiSpeakerSOTWordTimestampAligner(blank_id=blank_id, ctc_frame_seconds=0.01)
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    calls = []
    original = ctc_timestamp_utils._banded_ctc_search

    def record_batch(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ctc_timestamp_utils, "_banded_ctc_search", record_batch)

    result = extractor.extract_from_outputs_batch(
        ctc_log_probs=torch.log_softmax(logits, dim=-1).unsqueeze(0),
        sortformer_sigmoids=None,
        sot_transcripts=["hello world"],
    )[0]

    assert result["alignment_mode"] == "banded"
    assert calls == [1]
    assert [row["word"] for row in result["speaker_word_timestamps"][0]] == [
        "hello",
        "world",
    ]


def _tokenize_by_table(token_ids):
    def tokenize_words(words, blank):
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    return tokenize_words


@pytest.mark.unit
def test_prepared_alignment_keeps_only_the_columns_its_transcript_uses(monkeypatch):
    blank_id = 5
    extractor = MultiSpeakerSOTWordTimestampAligner(blank_id=blank_id, ctc_frame_seconds=0.01)
    monkeypatch.setattr(extractor, "_tokenize_words", _tokenize_by_table({"a": 3, "b": 1}))
    log_probs = torch.log_softmax(torch.randn(1, 6, blank_id + 1), dim=-1)

    prepared = extractor.prepare_alignment(log_probs, None, ["<spk:0> a <spk:1> b a"])

    (record,) = prepared["records"]
    # Blank first, then the transcript's tokens in vocabulary order.
    assert torch.equal(record["ctc_log_probs"], log_probs[0][:, [blank_id, 1, 3]])
    assert [stream["labels"] for stream in record["streams"]] == [[0, 2, 0], [0, 1, 0, 2, 0]]


@pytest.mark.unit
def test_prepared_alignment_needs_an_audio_duration_or_ctc_frame_seconds():
    aligner = MultiSpeakerSOTWordTimestampAligner(blank_id=2)

    with pytest.raises(ValueError, match="Record 0 has no audio duration"):
        aligner.prepare_alignment(torch.zeros(1, 4, 3), None, ["a"])


@pytest.mark.unit
def test_clean_sot_transcript_drops_special_tokens_but_keeps_speaker_tags():
    special_tokens = ["<s>", "</s>", "<|im_end|>", "<spk:0>", "<SPK:1>"]

    cleaned = MultiSpeakerSOTWordTimestampAligner.clean_sot_transcript(
        "<s><spk:0> hello  <|im_end|>world <SPK:1> yes</s>", special_tokens
    )

    assert cleaned == "<spk:0> hello world <SPK:1> yes"


@pytest.mark.unit
@pytest.mark.parametrize("speaker_weight", [0.0, 0.5])
def test_dp_over_the_gathered_columns_matches_the_dp_over_the_full_output(speaker_weight):
    torch.manual_seed(0)
    blank_id = 9
    full = torch.log_softmax(torch.randn(2, 12, blank_id + 1), dim=-1)
    labels = torch.tensor([[9, 4, 9, 4, 9, 7, 9], [9, 2, 9, 9, 9, 9, 9]])
    state_lengths = torch.tensor([7, 3])
    speaker_columns = torch.tensor([[-1, 0, -1, 0, -1, 0, -1], [-1, 1, -1, -1, -1, -1, -1]])
    speaker_probs = torch.rand(2, 12, 2)
    columns = [blank_id, 2, 4, 7]
    gathered_labels = labels.clone().apply_(columns.index)

    full_paths, full_scores = search_streams(
        full, labels, state_lengths, blank_id, speaker_columns, speaker_probs, speaker_weight
    )
    paths, scores = search_streams(
        full[:, :, columns], gathered_labels, state_lengths, 0, speaker_columns, speaker_probs, speaker_weight
    )

    assert [path.tolist() for path in paths] == [path.tolist() for path in full_paths]
    assert scores == full_scores


@pytest.mark.unit
def test_a_prepared_batch_aligns_without_the_aligner_that_prepared_it(monkeypatch):
    blank_id = 2
    extractor = MultiSpeakerSOTWordTimestampAligner(
        blank_id=blank_id, speaker_logprob_weight=0.5, maximum_token_len=0.5
    )
    monkeypatch.setattr(extractor, "_tokenize_words", _tokenize_by_table({"a": 0, "b": 1}))
    frames = [blank_id, 0, blank_id, 1, blank_id, 0, blank_id]
    logits = torch.full((len(frames), blank_id + 1), -12.0)
    for frame, label in enumerate(frames):
        logits[frame, label] = 12.0
    speaker_probs = torch.zeros(1, len(frames), 2)
    speaker_probs[0, :3, 0] = 0.9
    speaker_probs[0, 3:, 1] = 0.9
    inputs = dict(
        ctc_log_probs=torch.log_softmax(logits, dim=-1).unsqueeze(0),
        sortformer_sigmoids=speaker_probs,
        sot_transcripts=["<spk:0> a <spk:1> b a"],
        audio_durations=[0.7],
    )

    prepared = extractor.prepare_alignment(**inputs)

    assert align_prepared_batch(prepared) == extractor.extract_from_outputs_batch(**inputs)


@pytest.mark.unit
def test_merged_prepared_batches_align_like_separate_batches(monkeypatch):
    torch.manual_seed(0)
    blank_id = 2
    extractor = MultiSpeakerSOTWordTimestampAligner(blank_id=blank_id, ctc_frame_seconds=0.01)
    monkeypatch.setattr(extractor, "_tokenize_words", _tokenize_by_table({"a": 0, "b": 1}))
    short = extractor.prepare_alignment(torch.log_softmax(torch.randn(1, 5, blank_id + 1), dim=-1), None, ["a b"])
    long = extractor.prepare_alignment(torch.log_softmax(torch.randn(1, 9, blank_id + 1), dim=-1), None, ["b a b"])

    merged = merge_prepared_batches([short, long])

    assert align_prepared_batch(merged) == align_prepared_batch(short) + align_prepared_batch(long)
    stricter = {**long, "config": {**long["config"], "maximum_token_len": 0.5}}
    assert merge_prepared_batches([short, stricter]) is None


class _WindowProbeDecoder(nn.Module):
    """A CTC head whose calls record how many of its earlier outputs are still alive."""

    num_classes_with_blank = 4

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(16, 4)
        self.outputs = []
        self.live = []

    def forward(self, encoder_output, encoded_lengths):
        self.live.append(sum(output() is not None for output in self.outputs))
        log_probs = torch.log_softmax(self.projection(encoder_output.transpose(1, 2)), dim=-1)
        self.outputs.append(weakref.ref(log_probs))
        return log_probs


def _same(left, right):
    """Equality for prepared batches: tensors must match in dtype and every value."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(map(_same, left, right))
    return left == right


@pytest.mark.unit
def test_prepared_inputs_match_the_whole_head_output_one_window_at_a_time(monkeypatch):
    torch.manual_seed(0)
    decoder = _WindowProbeDecoder()
    aligner = MultiSpeakerSOTWordTimestampAligner(
        ctc_decoder=decoder, online_inference_length=3, chunk_left_context=1, chunk_right_context=1
    )
    monkeypatch.setattr(aligner, "_tokenize_words", _tokenize_by_table({"a": 0, "b": 2}))
    diarization_labels = torch.zeros(2, 4, 56, dtype=torch.bool)
    diarization_labels[0, 1, 10:30] = True
    timestamp_inputs = CTCTimestampInputs(
        asr_encoded=torch.randn(2, 16, 7),
        asr_encoded_lengths=torch.tensor([7, 5]),
        sortformer_sigmoids=torch.rand(2, 7, 2),
        sortformer_lengths=torch.tensor([7, 5]),
        diarization_labels=diarization_labels,
        diarization_lengths=torch.tensor([56, 40]),
    )
    transcripts = ["<spk:0> a b <spk:1> b", "a"]

    prepared = aligner.prepare_from_inputs(timestamp_inputs, transcripts, [0.56, 0.4])

    # No earlier window was alive when the next one was decoded.
    assert decoder.live == [0, 0, 0]
    expected = aligner.prepare_alignment(
        aligner.ctc_log_probs_from_inputs(timestamp_inputs),
        timestamp_inputs.sortformer_sigmoids,
        transcripts,
        ctc_lengths=timestamp_inputs.asr_encoded_lengths,
        sortformer_lengths=timestamp_inputs.sortformer_lengths,
        audio_durations=[0.56, 0.4],
        diarization_labels=diarization_labels,
        diarization_lengths=timestamp_inputs.diarization_lengths,
    )
    assert decoder.live == [0, 0, 0, 0, 0, 0]
    assert _same(prepared, expected)


@pytest.mark.unit
def test_compact_batched_ctc_alignment_handles_repeated_tokens():
    blank_id = 1
    target = torch.tensor([[blank_id, 0, blank_id, 0, blank_id]])
    frame_labels = [blank_id, 0, blank_id, 0, blank_id]
    logits = torch.full((len(frame_labels), 2), -12.0)
    for frame, label in enumerate(frame_labels):
        logits[frame, label] = 12.0

    paths, scores = search_streams(
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
@pytest.mark.parametrize(("speaker_weight", "expected_dp_calls"), [(0.0, 2), (0.25, 2), (1.0, 2)])
def test_timestamp_extractor_maps_speakers_without_preliminary_dp(monkeypatch, speaker_weight, expected_dp_calls):
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
    extractor = MultiSpeakerSOTWordTimestampAligner(
        blank_id=blank_id, speaker_logprob_weight=speaker_weight, ctc_frame_seconds=0.01
    )
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    calls = 0
    original = ctc_timestamp_utils._banded_ctc_search

    def record_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ctc_timestamp_utils, "_banded_ctc_search", record_call)

    result = extractor.extract_from_outputs_batch(
        ctc_log_probs=torch.log_softmax(logits, dim=-1).unsqueeze(0),
        sortformer_sigmoids=speaker_probs.unsqueeze(0),
        sot_transcripts=["<spk:0> a <spk:1> b"],
    )[0]

    assert result["speaker_tag_to_sortformer_column"] == {0: 1, 1: 0}
    assert result["alignment_mode"] == "banded"
    assert calls == expected_dp_calls


@pytest.mark.unit
def test_timestamp_extractor_reports_unmapped_speaker_without_legacy_fallback(
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
    extractor = MultiSpeakerSOTWordTimestampAligner(blank_id=blank_id, ctc_frame_seconds=0.01)
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)

    result = extractor.extract_from_outputs_batch(
        ctc_log_probs=torch.log_softmax(logits, dim=-1).unsqueeze(0),
        sortformer_sigmoids=torch.ones(1, len(frame_labels), 1),
        sot_transcripts=["<spk:0> a <spk:1> b"],
    )[0]

    assert result["alignment_mode"] == "banded"
    assert result["speaker_tag_to_sortformer_column"] == {0: 0, 1: None}
    assert [f["speaker"] for f in result["alignment_diagnostics"]["failed_streams"]] == [1]


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

    extractor = MultiSpeakerSOTWordTimestampAligner(blank_id=blank_id, ctc_frame_seconds=0.01)
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    dp_calls = []
    original = ctc_timestamp_utils._banded_ctc_search

    def record_dp_batch(*args, **kwargs):
        dp_calls.append((args[0].shape, len(args[2])))
        return original(*args, **kwargs)

    monkeypatch.setattr(ctc_timestamp_utils, "_banded_ctc_search", record_dp_batch)
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
    assert [result["alignment_mode"] for result in results] == ["banded", "banded"]
    # Each stream searches only its own valid frames, without cross-record padding.
    assert dp_calls == [((4, 2), 4), ((5, 2), 5)]
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

    _, encoded_len, timestamp_inputs = enc._forward_online_with_ctc_timestamp_inputs(
        audio_signal=torch.randn(2, 80, 200),
        length=torch.tensor([200, 160]),
        spk_targets=torch.rand(2, 25, 4),
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
    _, encoded_len, timestamp_inputs = enc._forward_with_ctc_timestamp_inputs(
        audio_signal=torch.randn(1, 80, 64),
        length=torch.tensor([64]),
        spk_targets=torch.rand(1, 8, 4),
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


def toy_transformer_asr_encoder_cfg() -> DictConfig:
    return DictConfig(
        {
            "_target_": "nemo.collections.asr.modules.transformer_encoder.TransformerEncoder",
            "feat_in": _MEL_FEATURES,
            "d_model": _ASR_D_MODEL,
            "n_heads": 2,
            "n_layers": 1,
            "subsampling": "feature_stacking",
            "subsampling_factor": _SUBSAMPLING_FACTOR,
            "drop_rate": 0.0,
            "dropout_pre_encoder": 0.0,
            "dropout_emb": 0.0,
            "qkv_bias": False,
            "qk_norm": True,
            "ff_expansion": 2.0,
            "pre_block_norm": True,
            "self_attention_model": "rope",
            "attn_mode": "full",
            "sync_max_audio_length": False,
        }
    )


def toy_packed_diarization_model_cfg() -> DictConfig:
    cfg = toy_diarization_model_cfg()
    cfg.encoder = toy_transformer_asr_encoder_cfg()
    cfg.encoder.d_model = _DIAR_FC_D_MODEL
    cfg.encoder.qk_norm = False
    cfg.transformer_encoder.num_layers = 0
    cfg.transformer_encoder.pre_ln = False
    return cfg


def build_toy_packed_pe_encoder(**overrides) -> ParallelExpertEncoder:
    """Construct the canonical PEE with native packed-capable branch encoders."""
    kwargs = {
        "asr_encoder_type": "transformer",
        "asr_encoder_cfg": toy_transformer_asr_encoder_cfg(),
        "diarization_model_cfg": toy_packed_diarization_model_cfg(),
    }
    kwargs.update(overrides)
    return build_toy_pe_encoder(**kwargs)


def bundle_config(**overrides) -> DictConfig:
    config = {
        "target": "nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT",
        "asr_encoder_cfg": toy_asr_encoder_cfg(),
        "diarization_model_cfg": toy_diarization_model_cfg(),
        "asr_normalize_type": "per_feature",
        "speaker_feature_config_version": 1,
        "speaker_feature_mode": "thresholded",
        "speaker_activity_threshold": 0.5,
    }
    config.update(overrides)
    return OmegaConf.create(config)


def write_bundle(path, config, state=None):
    config_bytes = OmegaConf.to_yaml(config).encode()
    weights = io.BytesIO()
    torch.save(state or {}, weights)
    with tarfile.open(path, "w") as archive:
        config_info = tarfile.TarInfo("model_config.yaml")
        config_info.size = len(config_bytes)
        archive.addfile(config_info, io.BytesIO(config_bytes))
        weight_bytes = weights.getvalue()
        weight_info = tarfile.TarInfo("model_weights.ckpt")
        weight_info.size = len(weight_bytes)
        archive.addfile(weight_info, io.BytesIO(weight_bytes))


@pytest.mark.unit
def test_pe_encoder_builds_and_wires_both_real_encoders():
    enc = build_toy_pe_encoder()
    # The two fused sub-encoders are the real classes, not stubs.
    assert isinstance(enc.asr_encoder, ConformerEncoder)
    assert enc.asr_encoder_type == "fastconformer"
    assert isinstance(enc.diarization_model, SortformerEncLabelModel)
    assert enc.diar_normalize_type == "per_feature"
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
    # train() leaves the frozen diar branch in eval mode.
    enc.train()
    assert enc.asr_encoder.training
    assert not enc.diarization_model.training


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

    _, encoded_lengths, timestamp_inputs = enc._forward_online_with_ctc_timestamp_inputs(
        audio_signal=mels,
        length=lengths,
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

    targets = torch.zeros(batch_size, expected_t, _N_SPK)
    targets[:, :, 0] = 1.0
    with torch.no_grad():
        external_outputs, external_len = enc(mels, length, spk_targets=targets)
    assert external_outputs.shape == outputs.shape
    assert torch.equal(external_len, encoded_len)


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


# ----------------------------------------------------------------------------- #
# Branch construction, packed training, online inference and portable bundles
# ----------------------------------------------------------------------------- #
@pytest.mark.unit
def test_pe_encoder_selects_native_transformer_asr_branch():
    encoder = build_toy_pe_encoder(
        asr_encoder_type="transformer",
        asr_encoder_cfg=toy_transformer_asr_encoder_cfg(),
    )
    assert isinstance(encoder.asr_encoder, TransformerEncoder)
    assert encoder.asr_encoder_type == "transformer"
    assert encoder.d_model == _ASR_D_MODEL
    assert encoder.subsampling_factor == _SUBSAMPLING_FACTOR


@pytest.mark.unit
@pytest.mark.parametrize(
    ("asr_encoder_type", "asr_encoder_cfg", "expected_class"),
    [
        ("fastconformer", toy_transformer_asr_encoder_cfg, "ConformerEncoder"),
        ("transformer", toy_asr_encoder_cfg, "TransformerEncoder"),
    ],
)
def test_pe_encoder_rejects_asr_encoder_type_config_mismatch(asr_encoder_type, asr_encoder_cfg, expected_class):
    with pytest.raises(TypeError, match=rf"requires .*{expected_class}"):
        build_toy_pe_encoder(
            asr_encoder_type=asr_encoder_type,
            asr_encoder_cfg=asr_encoder_cfg(),
        )


@pytest.mark.unit
def test_pe_encoder_rejects_unknown_asr_encoder_type():
    with pytest.raises(ValueError, match="asr_encoder_type must be one of"):
        build_toy_pe_encoder(asr_encoder_type="auto")


@pytest.mark.unit
def test_freeze_asr_keeps_both_frozen_branches_in_eval():
    encoder = build_toy_pe_encoder(freeze_asr=True)
    encoder.train()
    assert not encoder.asr_encoder.training
    assert not encoder.diarization_model.training
    assert all(not parameter.requires_grad for parameter in encoder.asr_encoder.parameters())


@pytest.mark.unit
def test_pe_encoder_rejects_incompatible_branch_frame_rates():
    diarization_config = toy_diarization_model_cfg()
    diarization_config.encoder.subsampling_factor = 4
    with pytest.raises(ValueError, match="embedded diarization encoder subsampling factor"):
        build_toy_pe_encoder(diarization_model_cfg=diarization_config)


@pytest.mark.unit
def test_pe_encoder_offline_forward_runs_transformer_asr_branch_on_unequal_lengths():
    encoder = build_toy_pe_encoder(
        asr_encoder_type="transformer",
        asr_encoder_cfg=toy_transformer_asr_encoder_cfg(),
    ).eval()
    mels = torch.randn(2, _MEL_FEATURES, 160)
    lengths = torch.tensor([160, 137])
    with torch.no_grad():
        output, output_lengths = encoder(mels, lengths)
    assert output.shape[:2] == (2, _ASR_D_MODEL)
    assert output.shape[-1] == int(output_lengths.max())
    assert torch.isfinite(output).all()


@pytest.mark.unit
def test_offline_sortformer_receives_per_feature_normalized_mels(monkeypatch):
    encoder = build_toy_pe_encoder().eval()
    mels = 4.0 * torch.randn(2, _MEL_FEATURES, 80) + 17.0
    lengths = torch.tensor([80, 53])
    observed = []
    original_frontend = encoder.diarization_model.frontend_encoder

    def tracked_frontend(**kwargs):
        observed.append(kwargs["processed_signal"].detach().clone())
        return original_frontend(**kwargs)

    monkeypatch.setattr(encoder.diarization_model, "frontend_encoder", tracked_frontend)
    with torch.no_grad():
        encoder._run_diarization(mels, lengths)

    expected, _, _ = normalize_batch(mels, lengths, normalize_type="per_feature")
    torch.testing.assert_close(observed[0], expected)


@pytest.mark.unit
def test_mixed_missing_rttm_rows_use_sortformer_predictions(monkeypatch):
    encoder = build_toy_pe_encoder().eval()
    mels = torch.randn(3, _MEL_FEATURES, 80)
    lengths = torch.tensor([80, 72, 64])
    diarization = torch.rand(3, 10, _N_SPK)
    asr_states = torch.randn(3, _ASR_D_MODEL, 10)
    asr_lengths = torch.tensor([10, 9, 8])
    monkeypatch.setattr(encoder, "_run_diarization", lambda *_: diarization)
    monkeypatch.setattr(encoder, "_run_asr", lambda *_: (asr_states, asr_lengths))

    targets = torch.zeros(3, 10, _N_SPK)
    targets[0, :, 0] = 1.0
    targets[1] = encoder.missing_rttm_target
    targets[2, :, 2] = 1.0
    expected = encoder._fuse_diar_and_asr(
        asr_states,
        targets,
        diarization_preds=diarization,
        use_diarization=torch.tensor([False, True, False]),
    )
    actual, actual_lengths = encoder(mels, lengths, spk_targets=targets)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_lengths, asr_lengths)


@pytest.mark.unit
@pytest.mark.parametrize(("training", "world_size"), [(True, 1), (False, 2)])
def test_all_rttm_rows_still_run_sortformer_in_collective_safe_paths(monkeypatch, training, world_size):
    encoder = build_toy_pe_encoder().train(training)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: world_size > 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
    mels = torch.randn(2, _MEL_FEATURES, 80)
    lengths = torch.tensor([80, 64])
    diarization = torch.rand(2, 10, _N_SPK)
    asr_states = torch.randn(2, _ASR_D_MODEL, 10)
    asr_lengths = torch.tensor([10, 8])
    diarization_calls = 0

    def run_diarization(*_):
        nonlocal diarization_calls
        diarization_calls += 1
        return diarization

    monkeypatch.setattr(encoder, "_run_diarization", run_diarization)
    monkeypatch.setattr(encoder, "_run_asr", lambda *_: (asr_states, asr_lengths))

    targets = torch.zeros(2, 10, _N_SPK)
    targets[0, :, 0] = 1.0
    targets[1, :, 1] = 1.0
    expected = encoder._fuse_diar_and_asr(asr_states, targets)
    actual, actual_lengths = encoder(mels, lengths, spk_targets=targets)

    assert diarization_calls == 1
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_lengths, asr_lengths)


@pytest.mark.unit
def test_all_rttm_rows_can_skip_sortformer_in_single_process_eval(monkeypatch):
    encoder = build_toy_pe_encoder().eval()
    mels = torch.randn(2, _MEL_FEATURES, 80)
    lengths = torch.tensor([80, 64])
    asr_states = torch.randn(2, _ASR_D_MODEL, 10)
    asr_lengths = torch.tensor([10, 8])
    monkeypatch.setattr(
        encoder,
        "_run_diarization",
        lambda *_: (_ for _ in ()).throw(AssertionError("single-process eval unexpectedly ran Sortformer")),
    )
    monkeypatch.setattr(encoder, "_run_asr", lambda *_: (asr_states, asr_lengths))

    targets = torch.zeros(2, 10, _N_SPK)
    targets[0, :, 0] = 1.0
    targets[1, :, 1] = 1.0
    expected = encoder._fuse_diar_and_asr(asr_states, targets)
    actual, actual_lengths = encoder(mels, lengths, spk_targets=targets)

    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_lengths, asr_lengths)


@pytest.mark.unit
def test_all_rttm_rows_still_run_sortformer_in_packed_training_path(monkeypatch):
    encoder = build_toy_pe_encoder().train()
    lengths = torch.tensor([80, 64])
    packed_input = pack_encoder_output(torch.randn(2, 80, _MEL_FEATURES), lengths)
    output_lengths = torch.tensor([10, 8])
    asr_states = pack_encoder_output(torch.randn(2, 10, _ASR_D_MODEL), output_lengths)
    diarization = pack_encoder_output(torch.rand(2, 10, _N_SPK), output_lengths)
    diarization_calls = 0

    def run_diarization(_):
        nonlocal diarization_calls
        diarization_calls += 1
        return diarization

    monkeypatch.setattr(encoder, "_run_diarization_packed", run_diarization)
    monkeypatch.setattr(encoder, "_run_asr_packed", lambda _: asr_states)

    targets = torch.zeros(2, 10, _N_SPK)
    targets[0, :, 0] = 1.0
    targets[1, :, 1] = 1.0
    expected = encoder._fuse_diar_and_asr_packed(asr_states, targets)
    actual = encoder.forward_sequence_packed(packed_input, spk_targets=targets)

    assert diarization_calls == 1
    torch.testing.assert_close(actual.data, expected.data)
    assert torch.equal(actual.lengths, expected.lengths)


@pytest.mark.unit
def test_speaker_threshold_and_kernel_scale_are_preserved():
    encoder = build_toy_pe_encoder(speaker_activity_threshold=0.5, spk_kernel_scale=0.25).eval()
    asr_states = torch.randn(1, _ASR_D_MODEL, 3)
    targets = torch.full((1, 3, _N_SPK), 0.5)
    targets[:, 1, 0] = 0.5001
    fused = encoder._fuse_diar_and_asr(asr_states, targets)

    normalized = encoder.asr_norm(asr_states.transpose(1, 2))
    binary = (targets > 0.5).to(normalized.dtype)
    infusion = encoder.diar_norm(binary) @ encoder.diar_kernel
    expected = (normalized + 0.25 * infusion).transpose(1, 2)
    torch.testing.assert_close(fused, expected)


@pytest.mark.unit
def test_high_resolution_diarization_is_pooled_to_asr_grid():
    diarization_config = toy_diarization_model_cfg()
    diarization_config.high_resolution = True
    diarization_config.output_subsampling_factor = 1
    encoder = build_toy_pe_encoder(diarization_model_cfg=diarization_config).eval()
    lengths = torch.tensor([80, 53])
    mels = torch.randn(2, _MEL_FEATURES, 80)
    packed_input = pack_encoder_output(mels.transpose(1, 2), lengths)

    with torch.no_grad():
        padded = encoder._run_diarization(mels, lengths)
        packed = encoder._run_diarization_packed(packed_input)
        asr = encoder._run_asr_packed(packed_input)

    assert torch.equal(packed.lengths, asr.lengths)
    assert padded.shape[1] == asr.max_seqlen
    restored = unpack_encoder_output(packed, total_length=padded.shape[1])
    valid = torch.arange(padded.shape[1])[None, :] < packed.lengths[:, None]
    torch.testing.assert_close(restored[valid], padded[valid], rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_high_resolution_fusion_pools_offline_but_not_aligned_online_predictions():
    diarization_config = toy_diarization_model_cfg()
    diarization_config.high_resolution = True
    diarization_config.output_subsampling_factor = 1
    encoder = build_toy_pe_encoder(diarization_model_cfg=diarization_config).eval()
    asr_states = torch.randn(1, _ASR_D_MODEL, 3)
    fine_predictions = torch.sigmoid(torch.arange(24 * _N_SPK).reshape(1, 24, _N_SPK).float() / 17.0 - 2.0)
    aligned_predictions = encoder.diarization_model.sortformer_modules.downsample_preds(
        fine_predictions, _SUBSAMPLING_FACTOR
    )

    offline_fused = encoder._fuse_diar_and_asr(asr_states, fine_predictions)
    online_fused = encoder._fuse_diar_and_asr(asr_states, aligned_predictions)

    assert aligned_predictions.shape[1] == asr_states.shape[-1]
    torch.testing.assert_close(offline_fused, online_fused)


@pytest.mark.unit
def test_packed_diarization_supports_optional_post_encoder():
    diarization_config = toy_packed_diarization_model_cfg()
    diarization_config.transformer_encoder = None
    encoder = build_toy_pe_encoder(diarization_model_cfg=diarization_config).eval()
    lengths = torch.tensor([80, 53])
    mels = torch.randn(2, _MEL_FEATURES, 80)

    with torch.no_grad():
        predictions = encoder._run_diarization_packed(pack_encoder_output(mels.transpose(1, 2), lengths))

    assert predictions.lengths.tolist() == [10, 7]
    assert torch.isfinite(predictions.data).all()


@pytest.mark.unit
def test_packed_fallback_matches_padded_forward_for_dense_and_packed_inputs():
    torch.manual_seed(0)
    encoder = build_toy_pe_encoder().eval()
    lengths = torch.tensor([80, 53])
    mels = torch.randn(2, _MEL_FEATURES, 80)
    mels[1, :, 53:] = 0.0
    targets = torch.zeros(2, 10, _N_SPK)
    targets[0, :, 0] = 1.0
    targets[1] = -1.0

    with torch.no_grad():
        padded, output_lengths = encoder(mels, lengths, spk_targets=targets)
        packed_from_dense = encoder.forward_sequence_packed(mels, lengths, spk_targets=targets)
        packed_input = pack_encoder_output(mels.transpose(1, 2), lengths)
        packed_from_packed = encoder.forward_sequence_packed(packed_input, spk_targets=targets)

    restored = unpack_encoder_output(packed_from_dense, total_length=padded.shape[-1])
    valid = torch.arange(padded.shape[-1])[None, :] < output_lengths[:, None]
    torch.testing.assert_close(restored[valid], padded.transpose(1, 2)[valid], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(packed_from_packed.data, packed_from_dense.data, rtol=1e-5, atol=1e-6)
    assert torch.equal(packed_from_packed.lengths, packed_from_dense.lengths)


@pytest.mark.unit
def test_native_packed_path_normalizes_diar_and_asr_independently_without_unpack(
    monkeypatch,
):
    encoder = build_toy_packed_pe_encoder().eval()
    lengths = torch.tensor([80, 53])
    mels = torch.randn(2, _MEL_FEATURES, 80)
    packed = pack_encoder_output(mels.transpose(1, 2), lengths)

    calls = []
    original_forward = encoder._forward_packed_branch

    def tracked_forward(branch, features, chunk_size_seconds):
        calls.append((branch, features.data.detach().clone(), chunk_size_seconds))
        return original_forward(branch, features, chunk_size_seconds)

    monkeypatch.setattr(encoder, "_forward_packed_branch", tracked_forward)
    monkeypatch.setattr(
        "nemo.collections.asr.modules.parallel_expert_encoder.unpack_encoder_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("native packed path unpacked")),
    )
    with torch.no_grad():
        output = encoder.forward_sequence_packed(packed)

    assert [branch for branch, _, _ in calls] == [
        encoder.diarization_model.encoder,
        encoder.asr_encoder,
    ]
    expected_diar = normalize_packed_batch(packed, "per_feature")
    torch.testing.assert_close(calls[0][1], expected_diar.data)
    torch.testing.assert_close(calls[1][1], expected_diar.data)
    assert torch.equal(output.lengths, torch.tensor([10, 7]))
    assert torch.isfinite(output.data).all()


@pytest.mark.unit
def test_native_packed_output_matches_padded_two_branch_path():
    torch.manual_seed(0)
    encoder = build_toy_packed_pe_encoder().eval()
    lengths = torch.tensor([80, 53])
    mels = torch.randn(2, _MEL_FEATURES, 80)
    mels[1, :, 53:] = 0.0
    with torch.no_grad():
        padded, padded_lengths = encoder(mels, lengths)
        packed = encoder.forward_sequence_packed(mels, lengths)
    restored = unpack_encoder_output(packed, total_length=padded.shape[-1])
    valid = torch.arange(padded.shape[-1])[None, :] < padded_lengths[:, None]
    torch.testing.assert_close(restored[valid], padded.transpose(1, 2)[valid], rtol=1e-4, atol=1e-5)


@pytest.mark.unit
def test_native_packed_branches_share_chunk_size_after_feature_stacking(monkeypatch):
    encoder = build_toy_packed_pe_encoder(
        frame_shift_seconds=0.01,
        chunk_size_seconds=0.16,
    ).eval()
    packed = pack_encoder_output(torch.randn(2, 80, _MEL_FEATURES), torch.tensor([80, 65]))
    diar_calls = []
    asr_calls = []
    diar_forward = encoder.diarization_model.encoder.forward_sequence_packed
    asr_forward = encoder.asr_encoder.forward_sequence_packed

    def tracked_diar(audio_signal, length, *args, **kwargs):
        diar_calls.append(
            (
                audio_signal.lengths.detach().cpu().tolist(),
                kwargs.get("bypass_pre_encode", False),
            )
        )
        return diar_forward(audio_signal, length, *args, **kwargs)

    def tracked_asr(audio_signal, length, *args, **kwargs):
        asr_calls.append(
            (
                audio_signal.lengths.detach().cpu().tolist(),
                kwargs.get("bypass_pre_encode", False),
            )
        )
        return asr_forward(audio_signal, length, *args, **kwargs)

    monkeypatch.setattr(encoder.diarization_model.encoder, "forward_sequence_packed", tracked_diar)
    monkeypatch.setattr(encoder.asr_encoder, "forward_sequence_packed", tracked_asr)
    with torch.no_grad():
        output = encoder.forward_sequence_packed(packed)

    expected_chunks = [([2, 2, 2, 2, 2, 2, 2, 2, 2, 1], True)]
    assert diar_calls == expected_chunks
    assert asr_calls == expected_chunks
    assert output.lengths.tolist() == [10, 9]


@pytest.mark.unit
def test_packed_fallback_rejects_online_scope():
    encoder = build_toy_pe_encoder().eval()
    with encoder.online_inference(), pytest.raises(RuntimeError, match="offline API"):
        encoder.forward_sequence_packed(torch.randn(1, _MEL_FEATURES, 32), torch.tensor([32]))


@pytest.mark.unit
def test_activation_checkpointing_wraps_trainable_asr_layers_and_packed_backward():
    encoder = build_toy_packed_pe_encoder().train()
    encoder.set_activation_checkpointing(True)
    encoder.set_activation_checkpointing(True)

    assert getattr(encoder.asr_encoder.pre_encode, "_checkpoint_wrapped_module", None) is not None
    assert all(getattr(layer, "_checkpoint_wrapped_module", None) is not None for layer in encoder.asr_encoder.layers)
    assert all(
        getattr(layer, "_checkpoint_wrapped_module", None) is None
        for layer in encoder.diarization_model.encoder.layers
    )

    mels = torch.randn(1, 64, _MEL_FEATURES, requires_grad=True)
    packed = pack_encoder_output(mels, torch.tensor([64]))
    output = encoder._run_asr_packed(packed)
    output.data.square().mean().backward()
    assert mels.grad is not None
    assert torch.isfinite(mels.grad).all()


def scoped_dispatch_stub(enabled):
    encoder = _PEE.__new__(_PEE)
    nn.Module.__init__(encoder)
    encoder.online_inference_length = 10
    encoder.online_inference_enabled = enabled
    encoder._forward = lambda **kwargs: ("offline", None)
    encoder._forward_online = lambda **kwargs: ("online", None)
    return encoder


@pytest.mark.unit
def test_online_inference_context_controls_dispatch_and_restores_state():
    encoder = scoped_dispatch_stub(False)
    audio = torch.zeros(1, 8, 20)
    length = torch.tensor([20])
    assert encoder(audio, length)[0] == "offline"
    with encoder.online_inference():
        assert encoder(audio, length)[0] == "online"
    assert encoder(audio, length)[0] == "offline"


@pytest.mark.unit
def test_online_inference_passes_normalized_time_major_mels_to_sortformer(monkeypatch):
    """Sortformer receives normalized time-major features before its pre-encoder."""
    encoder = build_toy_pe_encoder(
        diarization_model_cfg=toy_packed_diarization_model_cfg(),
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    ).eval()
    encoder._suppress_online_pbar = True
    original = encoder.diarization_model.forward_streaming_step
    observed_signals = []

    def checked_forward_streaming_step(**kwargs):
        processed_signal = kwargs["processed_signal"]
        observed_signals.append(processed_signal.detach().clone())
        assert processed_signal.shape[2] == _MEL_FEATURES
        return original(**kwargs)

    monkeypatch.setattr(
        encoder.diarization_model,
        "forward_streaming_step",
        checked_forward_streaming_step,
    )
    mels = 4.0 * torch.randn(1, _MEL_FEATURES, 160) + 17.0
    lengths = torch.tensor([160])

    with torch.no_grad(), encoder.online_inference():
        encoder(mels, lengths)

    expected, _, _ = normalize_batch(mels, lengths, normalize_type="per_feature")
    assert observed_signals
    expected_first_chunk = expected[:, :, : observed_signals[0].shape[1]].transpose(1, 2)
    torch.testing.assert_close(observed_signals[0], expected_first_chunk)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_online_inference_uses_activation_device_after_nested_parent_move():
    """A nested Lightning Sortformer must not retain its construction device."""
    parent = nn.Module()
    parent.encoder = build_toy_pe_encoder(
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    ).eval()
    parent.encoder._suppress_online_pbar = True
    parent.to("cuda")

    mels = torch.randn(1, _MEL_FEATURES, 40, device="cuda")
    lengths = torch.tensor([40], device="cuda")
    with torch.no_grad(), parent.encoder.online_inference():
        output, output_lengths = parent.encoder(mels, lengths)

    assert output.device.type == "cuda"
    assert output_lengths.device.type == "cuda"
    assert torch.isfinite(output).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("asr_encoder_type", "asr_encoder_cfg"),
    [
        ("fastconformer", toy_asr_encoder_cfg),
        ("transformer", toy_transformer_asr_encoder_cfg),
    ],
)
def test_online_inference_matches_independent_valid_prefixes_for_unequal_audio(asr_encoder_type, asr_encoder_cfg):
    encoder = build_toy_pe_encoder(
        asr_encoder_type=asr_encoder_type,
        asr_encoder_cfg=asr_encoder_cfg(),
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    ).eval()
    encoder._suppress_online_pbar = True
    mels = torch.randn(2, _MEL_FEATURES, 321)
    lengths = torch.tensor([321, 173])

    with torch.no_grad(), encoder.online_inference():
        batched_output, batched_lengths = encoder(mels, lengths)
        first_output, first_length = encoder(mels[:1, :, :321], lengths[:1])
        second_output, second_length = encoder(mels[1:2, :, :173], lengths[1:])

    expected_lengths = torch.cat([first_length, second_length])
    assert torch.equal(batched_lengths, expected_lengths)
    assert batched_output.shape == (2, _ASR_D_MODEL, int(expected_lengths.max()))
    torch.testing.assert_close(batched_output[0, :, : first_length[0]], first_output[0], rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(batched_output[1, :, : second_length[0]], second_output[0], rtol=1e-5, atol=2e-5)


@pytest.mark.unit
@pytest.mark.parametrize("asr_encoder_type", ["fastconformer", "transformer"])
def test_asr_output_frame_boundary_on_tensors_matches_ints(asr_encoder_type):
    encoder = SimpleNamespace(asr_encoder_type=asr_encoder_type, subsampling_factor=8)
    boundaries = torch.arange(0, 400)
    expected = [ParallelExpertEncoder._asr_output_frame_boundary(encoder, int(b)) for b in boundaries]
    actual = ParallelExpertEncoder._asr_output_frame_boundary(encoder, boundaries)
    assert actual.dtype == boundaries.dtype
    assert actual.tolist() == expected


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize(
    "asr_encoder_type, asr_encoder_cfg",
    [
        ("fastconformer", toy_asr_encoder_cfg),
        ("transformer", toy_transformer_asr_encoder_cfg),
    ],
)
def test_single_window_online_inference_can_be_captured_in_a_cuda_graph(asr_encoder_type, asr_encoder_cfg):
    """A graph-captured single-window online forward over padded frames matches eager on the valid frames."""
    encoder = (
        build_toy_pe_encoder(
            asr_encoder_type=asr_encoder_type,
            asr_encoder_cfg=asr_encoder_cfg(),
            online_inference_length=40,
        )
        .cuda()
        .eval()
    )
    encoder._suppress_online_pbar = True
    lengths = torch.tensor([200, 137], device="cuda")
    mels = torch.randn(2, _MEL_FEATURES, 200, device="cuda")
    static_mels = torch.zeros(2, _MEL_FEATURES, 256, device="cuda")
    static_mels[:, :, :200] = mels
    static_lengths = lengths.clone()

    with torch.no_grad(), encoder.online_inference():
        expected, expected_lengths = encoder(mels, lengths)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            encoder(static_mels, static_lengths)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, out_lengths = encoder(static_mels, static_lengths)
    graph.replay()

    assert torch.equal(out_lengths, expected_lengths)
    for i, n in enumerate(expected_lengths.tolist()):
        torch.testing.assert_close(out[i, :, :n], expected[i, :, :n], rtol=1e-5, atol=2e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "asr_encoder_type,asr_encoder_cfg",
    [("fastconformer", toy_asr_encoder_cfg), ("transformer", toy_transformer_asr_encoder_cfg)],
)
def test_online_inference_extra_padding_does_not_change_outputs(asr_encoder_type, asr_encoder_cfg):
    """Online windows cover the padded width; padding past every row's length, even across windows, is masked."""
    torch.manual_seed(0)
    encoder = build_toy_pe_encoder(
        asr_encoder_type=asr_encoder_type,
        asr_encoder_cfg=asr_encoder_cfg(),
        online_inference_length=40,
    ).eval()
    encoder._suppress_online_pbar = True
    lengths = torch.tensor([200, 137])
    mels = torch.randn(2, _MEL_FEATURES, 200)
    padded = torch.cat([mels, torch.zeros(2, _MEL_FEATURES, 3 * encoder.chunk_feat_len)], dim=-1)

    with torch.no_grad(), encoder.online_inference():
        expected, expected_lengths = encoder(mels, lengths)
        out, out_lengths = encoder(padded, lengths)

    assert torch.equal(out_lengths, expected_lengths)
    for i, n in enumerate(expected_lengths.tolist()):
        torch.testing.assert_close(out[i, :, :n], expected[i, :, :n])


@pytest.mark.unit
def test_online_inference_extra_padding_does_not_change_high_resolution_outputs():
    """Padded hidden states must not reach valid frames through the high-resolution upsampler's convolution."""
    torch.manual_seed(0)
    diarization_config = toy_diarization_model_cfg()
    diarization_config.high_resolution = True
    encoder = build_toy_pe_encoder(
        diarization_model_cfg=diarization_config,
        online_inference_length=40,
        speaker_feature_mode="continuous",
        speaker_activity_threshold=None,
    ).eval()
    encoder._suppress_online_pbar = True
    # A trained upsampler reads its right-hand neighbour; the default initialization does not.
    upsample = encoder.diarization_model.sortformer_modules.subpixel_upsample
    with torch.no_grad():
        upsample.weight.normal_()
    lengths = torch.tensor([200, 137])
    mels = torch.randn(2, _MEL_FEATURES, 200)
    padded = torch.cat([mels, torch.randn(2, _MEL_FEATURES, 56)], dim=-1)

    with torch.no_grad(), encoder.online_inference():
        expected, expected_lengths = encoder(mels, lengths)
        out, out_lengths = encoder(padded, lengths)

    assert torch.equal(out_lengths, expected_lengths)
    for i, n in enumerate(expected_lengths.tolist()):
        torch.testing.assert_close(out[i, :, :n], expected[i, :, :n])


@pytest.mark.unit
@pytest.mark.parametrize(
    "asr_encoder_type,asr_encoder_cfg",
    [("fastconformer", toy_asr_encoder_cfg), ("transformer", toy_transformer_asr_encoder_cfg)],
)
def test_online_inference_batches_asr_windows_like_one_window_per_forward(asr_encoder_type, asr_encoder_cfg):
    """Encoding several ASR windows per forward returns what one window per forward does."""
    torch.manual_seed(0)
    encoder = build_toy_pe_encoder(
        asr_encoder_type=asr_encoder_type,
        asr_encoder_cfg=asr_encoder_cfg(),
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
    ).eval()
    encoder._suppress_online_pbar = True
    lengths = torch.tensor([7 * encoder.chunk_feat_len - 13, 3 * encoder.chunk_feat_len + 5])
    mels = torch.randn(2, _MEL_FEATURES, int(lengths.max()))

    with torch.no_grad(), encoder.online_inference():
        batched, batched_lengths = encoder(mels, lengths)
        encoder._ASR_WINDOW_ROWS = 1
        expected, expected_lengths = encoder(mels, lengths)

    assert torch.equal(batched_lengths, expected_lengths)
    for i, n in enumerate(expected_lengths.tolist()):
        torch.testing.assert_close(batched[i, :, :n], expected[i, :, :n])


@pytest.mark.unit
@pytest.mark.parametrize(
    "asr_encoder_type,asr_encoder_cfg",
    [("fastconformer", toy_asr_encoder_cfg), ("transformer", toy_transformer_asr_encoder_cfg)],
)
def test_online_cuda_graphs_replay_streaming_diarization_steps(asr_encoder_type, asr_encoder_cfg):
    """Steady-state streaming Sortformer steps replayed from CUDA graphs give the eager outputs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    encoder = (
        build_toy_pe_encoder(
            asr_encoder_type=asr_encoder_type,
            asr_encoder_cfg=asr_encoder_cfg(),
            online_inference_length=10,
            chunk_left_context=2,
            chunk_right_context=2,
            diar_fifo_len=10,
            diar_spkcache_update_period=20,
            diar_spkcache_len=20,
        )
        .cuda()
        .eval()
    )
    encoder._suppress_online_pbar = True
    lengths = torch.tensor([12 * encoder.chunk_feat_len - 13, 9 * encoder.chunk_feat_len + 5], device="cuda")
    mels = torch.randn(2, _MEL_FEATURES, int(lengths.max()), device="cuda")

    with torch.no_grad(), encoder.online_inference():
        expected, expected_lengths = encoder(mels, lengths)
        encoder.online_graphs = OnlineGraphs()
        out, out_lengths = encoder(mels, lengths)
        replayed, _ = encoder(mels, lengths)

    assert encoder.online_graphs._diar_steps.graphs, "no streaming step was captured"
    assert torch.equal(out_lengths, expected_lengths)
    for i, n in enumerate(expected_lengths.tolist()):
        torch.testing.assert_close(out[i, :, :n], expected[i, :, :n])
        torch.testing.assert_close(replayed[i, :, :n], expected[i, :, :n])


@pytest.mark.unit
def test_streaming_step_graphs_fall_back_to_eager_when_capture_fails(monkeypatch):
    """A failed capture leaves the step eager for good instead of breaking inference."""
    from types import SimpleNamespace

    def failing_capture(fn, pool=None):
        raise RuntimeError("synthetic capture failure")

    monkeypatch.setattr(online_graphs_module, "capture_cuda_graph", failing_capture)
    monkeypatch.setattr(online_graphs_module.torch.cuda, "graph_pool_handle", lambda: None)
    calls = []

    def step(state, chunk, chunk_length):
        calls.append(chunk_length.tolist())
        return state, chunk * 2

    graphs = online_graphs_module._StreamingStepGraphs()
    state = SimpleNamespace(spkcache=torch.zeros(1, 4, 2), spkcache_compressed=True)
    chunk, chunk_length = torch.ones(1, 3, 2), torch.tensor([3])
    for _ in range(3):
        _, preds = graphs(step, state, chunk, chunk_length, extra_key=())
        assert torch.equal(preds, chunk * 2)
    assert len(calls) == 3
    assert not graphs.enabled and not graphs.graphs


@pytest.mark.unit
def test_transformer_online_inference_preserves_partial_feature_stack():
    encoder = build_toy_pe_encoder(
        asr_encoder_type="transformer",
        asr_encoder_cfg=toy_transformer_asr_encoder_cfg(),
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    ).eval()
    encoder._suppress_online_pbar = True
    mels = torch.randn(1, _MEL_FEATURES, 161)
    lengths = torch.tensor([161])
    targets = torch.zeros(1, 21, _N_SPK)
    targets[:, :, 0] = 1.0

    with torch.no_grad():
        _, expected_lengths = encoder._run_asr(mels, lengths)
        with encoder.online_inference():
            output, output_lengths = encoder(mels, lengths, spk_targets=targets)

    assert torch.equal(output_lengths, expected_lengths)
    assert output.shape == (1, _ASR_D_MODEL, int(expected_lengths[0]))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("asr_encoder_type", "asr_encoder_cfg", "write_type_flag"),
    [
        ("fastconformer", toy_asr_encoder_cfg, False),
        ("transformer", toy_transformer_asr_encoder_cfg, True),
    ],
)
def test_strict_two_branch_bundle_loading(tmp_path, asr_encoder_type, asr_encoder_cfg, write_type_flag):
    source = build_toy_pe_encoder(
        asr_encoder_type=asr_encoder_type,
        asr_encoder_cfg=asr_encoder_cfg(),
    ).eval()
    state = {f"encoder.{key}": value for key, value in source.state_dict().items()}
    archive = tmp_path / "two_branch.nemo"
    config_overrides = {"asr_encoder_cfg": asr_encoder_cfg()}
    if write_type_flag:
        config_overrides["asr_encoder_type"] = asr_encoder_type
    write_bundle(archive, bundle_config(**config_overrides), state)

    restored = ParallelExpertEncoderPT.load_from_nemo(str(archive), strict=True).eval()
    assert restored.asr_encoder_type == asr_encoder_type
    assert set(restored.state_dict()) == set(source.state_dict())
    for key, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


@pytest.mark.unit
def test_inline_config_reconstructs_architecture_without_standalone_weights():
    config = bundle_config(chunk_size_seconds=30.0)

    restored = ParallelExpertEncoderPT.from_inline_config(config).eval()

    assert restored.chunk_size_seconds == 30.0
    assert restored.diar_normalize_type == "per_feature"
    assert restored._bundle_config.diar_normalize_type == "per_feature"
    assert restored.speaker_feature_mode == "thresholded"
    assert restored.speaker_activity_threshold == 0.5
    assert restored._bundle_config.speaker_feature_config_version == 1
    assert restored._bundle_config.speaker_feature_mode == "thresholded"
    assert restored._bundle_config.speaker_activity_threshold == 0.5
    assert restored._bundle_config.target == config.target


@pytest.mark.unit
def test_legacy_canonical_bundle_requires_explicit_speaker_contract(tmp_path):
    config = bundle_config()
    del config.speaker_feature_config_version
    del config.speaker_feature_mode
    del config.speaker_activity_threshold
    with pytest.raises(ValueError, match="cannot be inferred safely"):
        ParallelExpertEncoderPT.from_inline_config(config)

    source = build_toy_pe_encoder().eval()
    state = {f"encoder.{key}": value for key, value in source.state_dict().items()}
    archive = tmp_path / "legacy-canonical.nemo"
    write_bundle(archive, config, state)
    overrides = {
        "speaker_feature_config_version": 1,
        "speaker_feature_mode": "continuous",
        "speaker_activity_threshold": None,
        "sync_max_audio_length": False,
    }
    restored = ParallelExpertEncoderPT.load_from_nemo(
        str(archive),
        strict=True,
        config_overrides=overrides,
    ).eval()

    assert restored.speaker_feature_mode == "continuous"
    assert restored.speaker_activity_threshold is None
    assert restored._bundle_config.speaker_feature_config_version == 1
    assert restored._bundle_config.speaker_feature_mode == "continuous"
    assert restored._bundle_config.speaker_activity_threshold is None
    assert restored._bundle_config.sync_max_audio_length is False


@pytest.mark.unit
def test_sync_max_audio_length_is_disabled_by_portable_config():
    config = bundle_config(sync_max_audio_length=False)
    config.asr_encoder_cfg.sync_max_audio_length = True
    config.diarization_model_cfg.encoder.sync_max_audio_length = True

    restored = ParallelExpertEncoderPT.from_inline_config(config).eval()

    assert restored.sync_max_audio_length is False
    assert restored.asr_encoder.sync_max_audio_length is False
    assert restored.diarization_model.encoder.sync_max_audio_length is False
    assert restored._bundle_config.sync_max_audio_length is False


@pytest.mark.unit
def test_explicit_portable_speaker_contract_is_independent_of_target_name():
    config = bundle_config(
        target="nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT",
        speaker_feature_config_version=1,
        speaker_feature_mode="thresholded",
        speaker_activity_threshold=0.5,
    )

    restored = ParallelExpertEncoderPT.from_inline_config(config).eval()

    assert restored.speaker_feature_mode == "thresholded"
    assert restored.speaker_activity_threshold == 0.5


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"speaker_feature_config_version": 1},
            "requires an explicit speaker_feature_mode",
        ),
        (
            {
                "speaker_feature_config_version": 1,
                "speaker_feature_mode": "continuous",
                "speaker_activity_threshold": 0.5,
            },
            "continuous.*speaker_activity_threshold=None",
        ),
        (
            {
                "speaker_feature_config_version": 1,
                "speaker_feature_mode": "thresholded",
                "speaker_activity_threshold": None,
            },
            "thresholded.*non-null",
        ),
        (
            {
                "speaker_feature_config_version": 2,
                "speaker_feature_mode": "thresholded",
                "speaker_activity_threshold": 0.5,
            },
            "Unsupported speaker_feature_config_version",
        ),
    ],
)
def test_invalid_speaker_feature_contract_is_rejected(overrides, match):
    config = bundle_config()
    for key in ("speaker_feature_config_version", "speaker_feature_mode", "speaker_activity_threshold"):
        if key not in overrides:
            del config[key]
    config = OmegaConf.merge(config, overrides)
    with pytest.raises(ValueError, match=match):
        ParallelExpertEncoderPT.from_inline_config(config)


@pytest.mark.unit
def test_current_pee_loader_constructs_canonical_inline_schema():
    from nemo.collections.asr.modules.parallel_expert_encoder import (
        ParallelExpertEncoderPT as CurrentParallelExpertEncoderPT,
    )

    restored = CurrentParallelExpertEncoderPT.from_inline_config(bundle_config()).eval()

    assert isinstance(restored, ParallelExpertEncoder)
    assert restored.asr_encoder_type == "fastconformer"


@pytest.mark.unit
def test_current_pee_loader_rejects_obsolete_three_expert_inline_schema():
    from nemo.collections.asr.modules.parallel_expert_encoder import (
        ParallelExpertEncoderPT as CurrentParallelExpertEncoderPT,
    )

    config = OmegaConf.create(
        {
            "target": "nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT",
            "speech_expert_cfg": {"_target_": "example.Speech"},
            "speaker_expert_cfg": {"_target_": "example.Speaker"},
            "sound_expert_cfg": {"_target_": "example.Sound"},
            "sortformer_modules_cfg": {"_target_": "example.Sortformer"},
        }
    )
    with pytest.raises(ValueError, match="missing required config sections"):
        CurrentParallelExpertEncoderPT.from_inline_config(config)


@pytest.mark.unit
def test_exported_inline_config_round_trips_consolidated_weights():
    to_hf_path = Path(__file__).parents[3] / "examples" / "speechlm2" / "to_hf.py"
    spec = importlib.util.spec_from_file_location("to_hf_for_pee_roundtrip", to_hf_path)
    to_hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(to_hf)

    source = ParallelExpertEncoderPT(bundle_config()).encoder.eval()
    source.chunk_size_seconds = 30.0
    root = SimpleNamespace(
        cfg={"pe_encoder_path": "/models/placeholderParallelExpertEncoder.nemo"},
        perception=SimpleNamespace(encoder=source),
    )

    exported = to_hf._hf_export_config(root, "bfloat16")
    restored = ParallelExpertEncoderPT.from_inline_config(exported["pe_encoder_config"]).eval()
    restored.load_state_dict(source.state_dict(), strict=True)

    assert set(restored.state_dict()) == set(source.state_dict())
    assert restored.chunk_size_seconds == 30.0
    assert restored.diar_normalize_type == "per_feature"
    assert restored.speaker_feature_mode == "thresholded"
    assert restored.speaker_activity_threshold == 0.5
    assert exported["pe_encoder_config"]["speaker_feature_config_version"] == 1
    assert exported["pe_encoder_config"]["speaker_feature_mode"] == "thresholded"
    assert exported["pe_encoder_config"]["speaker_activity_threshold"] == 0.5
    assert exported["pe_encoder_config"]["sync_max_audio_length"] is False
    mels = torch.randn(1, _MEL_FEATURES, 161)
    lengths = torch.tensor([161])
    with torch.no_grad():
        expected, expected_lengths = source(mels, lengths)
        actual, actual_lengths = restored(mels, lengths)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_lengths, expected_lengths)


@pytest.mark.unit
def test_legacy_three_expert_bundle_is_rejected(tmp_path):
    archive = tmp_path / "legacy.nemo"
    legacy = OmegaConf.create(
        {
            "target": "nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT",
            "speech_expert_cfg": {"_target_": "legacy.Speech"},
            "speaker_expert_cfg": {"_target_": "legacy.Speaker"},
            "sound_expert_cfg": {"_target_": "legacy.Sound"},
            "sortformer_modules_cfg": {"_target_": "legacy.Sortformer"},
        }
    )
    write_bundle(archive, legacy)
    with pytest.raises(ValueError, match="missing required config sections"):
        ParallelExpertEncoderPT.load_from_nemo(str(archive), strict=True)


@pytest.mark.unit
def test_save_to_nemo_persists_shared_chunk_size(tmp_path, monkeypatch):
    source = build_toy_pe_encoder(chunk_size_seconds=30.0).eval()
    template = tmp_path / "template.nemo"
    write_bundle(template, bundle_config())
    captured = {}

    def capture_config(shell, output_path):
        captured["output_path"] = output_path
        captured["config"] = OmegaConf.to_container(shell._cfg, resolve=True)

    monkeypatch.setattr(ParallelExpertEncoderPT, "save_to", capture_config)
    output = tmp_path / "output.nemo"
    ParallelExpertEncoderPT.save_to_nemo(source, str(output), template_bundle_path=str(template))

    assert captured["output_path"] == str(output)
    assert captured["config"]["chunk_size_seconds"] == 30.0


class _FlatMelPerception(nn.Module):
    """Perception stand-in around a real encoder: the 'audio' is a flattened mel spectrogram ``(B, T * F)``."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def maybe_preprocess_audio(self, input_signal, input_signal_length, processed_signal, processed_signal_length):
        if processed_signal is not None:
            return processed_signal, processed_signal_length
        b, n = input_signal.shape
        mels = input_signal.view(b, n // _MEL_FEATURES, _MEL_FEATURES).transpose(1, 2)
        return mels, input_signal_length // _MEL_FEATURES

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        return_ctc_timestamp_inputs=False,
    ):
        x, x_len = self.maybe_preprocess_audio(
            input_signal, input_signal_length, processed_signal, processed_signal_length
        )
        out = self.encoder(audio_signal=x, length=x_len, return_ctc_timestamp_inputs=return_ctc_timestamp_inputs)
        return (out[0].transpose(1, 2), *out[1:])


@pytest.mark.unit
def test_encoder_graphs_return_eager_ctc_timestamp_inputs():
    """Graphs captured with CTC timestamp inputs replay the eager embeddings and alignment inputs of every row."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from nemo.collections.speechlm2.vllm.salm.encoder_graphs import EncoderGraphRunner

    torch.manual_seed(0)
    diarization_config = toy_diarization_model_cfg()
    diarization_config.high_resolution = True
    encoder = build_toy_pe_encoder(diarization_model_cfg=diarization_config, online_inference_length=40)
    encoder._suppress_online_pbar = True
    perception = _FlatMelPerception(encoder).cuda().eval()
    runner = EncoderGraphRunner(perception, batch_sizes=(2, 4), frame_bucket=64, max_windows=2)
    runner.ctc_timestamp_inputs = True
    frames = torch.tensor([2 * encoder.chunk_feat_len - 30, 150, encoder.chunk_feat_len + 7], device="cuda")
    audio = torch.randn(3, int(frames.max()) * _MEL_FEATURES, device="cuda")
    lengths = frames * _MEL_FEATURES

    embeds, states = runner(audio, lengths, return_ctc_timestamp_inputs=True)
    assert runner.graphs
    with torch.no_grad(), encoder.online_inference():
        ref_embs, ref_lens, ref_states = perception(
            input_signal=audio, input_signal_length=lengths, return_ctc_timestamp_inputs=True
        )

    for i, n in enumerate(ref_lens.tolist()):
        torch.testing.assert_close(embeds[i], ref_embs[i, :n])
    for name, time_axis, lengths_name in (
        ("asr_encoded", 2, "asr_encoded_lengths"),
        ("sortformer_sigmoids", 1, "sortformer_lengths"),
        ("diarization_labels", 2, "diarization_lengths"),
    ):
        got_lengths, ref_lengths = getattr(states, lengths_name), getattr(ref_states, lengths_name)
        assert torch.equal(got_lengths, ref_lengths), name
        for i, n in enumerate(ref_lengths.tolist()):
            got, ref = getattr(states, name)[i], getattr(ref_states, name)[i]
            torch.testing.assert_close(got.narrow(time_axis - 1, 0, n), ref.narrow(time_axis - 1, 0, n), msg=name)
    assert runner(audio, lengths) is not None  # the same graphs also serve calls without the inputs


def _long_audio_runner(ctc_timestamp_inputs=False, long_audio=True):
    from nemo.collections.speechlm2.vllm.salm.encoder_graphs import EncoderGraphRunner

    torch.manual_seed(0)
    diarization_config = toy_diarization_model_cfg()
    diarization_config.high_resolution = True
    encoder = build_toy_pe_encoder(
        diarization_model_cfg=diarization_config,
        online_inference_length=10,
        chunk_left_context=2,
        chunk_right_context=2,
        diar_fifo_len=10,
        diar_spkcache_update_period=20,
        diar_spkcache_len=20,
    )
    encoder._suppress_online_pbar = True
    perception = _FlatMelPerception(encoder).cuda().eval()
    runner = EncoderGraphRunner(
        perception, batch_sizes=(1, 2, 4), frame_bucket=16, max_windows=1, last_window_bucket=32, long_audio=long_audio
    )
    runner.ctc_timestamp_inputs = ctc_timestamp_inputs
    return runner, perception, encoder


def _eager(perception, encoder, audio, lengths, **kwargs):
    with encoder.online_graphs.paused(), torch.no_grad(), encoder.online_inference():
        return perception(input_signal=audio, input_signal_length=lengths, **kwargs)


def _eager_window_pieces(encoder):
    caches = (encoder.online_graphs._asr_windows, encoder.online_graphs._diar_steps)
    return sum(len(cache.seen) for cache in caches if cache is not None)


@pytest.mark.unit
def test_encoder_graphs_serve_audio_of_any_length_window_by_window():
    """Audio past the whole-forward graphs replays window graphs captured at start-up and matches eager inference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner, perception, encoder = _long_audio_runner()
    window = encoder.chunk_feat_len
    runner(torch.randn(1, 50 * _MEL_FEATURES, device="cuda"), torch.tensor([50 * _MEL_FEATURES], device="cuda"))
    assert runner.long_graph_count() > 0
    eager_pieces = _eager_window_pieces(encoder)

    for frames in ([3 * window - 5], [window + 1, 5 * window + 37, 2 * window], [12 * window + 3] * 3 + [40] * 3):
        frames = torch.tensor(frames, device="cuda")
        audio = torch.randn(len(frames), int(frames.max()) * _MEL_FEATURES, device="cuda")
        lengths = frames * _MEL_FEATURES
        got = runner(audio, lengths)
        ref_embs, ref_lens = _eager(perception, encoder, audio, lengths)
        assert len(got) == len(frames)
        for i, n in enumerate(ref_lens.tolist()):
            torch.testing.assert_close(got[i], ref_embs[i, :n], rtol=1e-5, atol=2e-5)
    # Every window of every length replayed a graph captured at start-up.
    assert _eager_window_pieces(encoder) == eager_pieces


def _assert_states_match(states, ref_states):
    """Every row of ``CTCTimestampInputs`` equals the reference over its valid frames (padding may differ)."""
    for name, time_axis, lengths_name in (
        ("asr_encoded", 2, "asr_encoded_lengths"),
        ("sortformer_sigmoids", 1, "sortformer_lengths"),
        ("diarization_labels", 2, "diarization_lengths"),
    ):
        got_lengths, ref_lengths = getattr(states, lengths_name), getattr(ref_states, lengths_name)
        assert torch.equal(got_lengths, ref_lengths), name
        for i, n in enumerate(ref_lengths.tolist()):
            got, ref = getattr(states, name)[i], getattr(ref_states, name)[i]
            # Batch shapes can change FP32 results across a BF16/FP16 rounding boundary.
            # Allow one storage-precision epsilon, keeping FP32 states and boolean labels strict.
            rtol = torch.finfo(got.dtype).eps if got.dtype in (torch.bfloat16, torch.float16) else 1e-5
            torch.testing.assert_close(
                got.narrow(time_axis - 1, 0, n), ref.narrow(time_axis - 1, 0, n), rtol=rtol, atol=2e-5, msg=name
            )


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype,delta,matches",
    [(torch.bfloat16, 1 / 256, True), (torch.bfloat16, 1 / 32, False), (torch.float32, 1 / 256, False)],
)
def test_timestamp_state_comparison_respects_storage_precision(dtype, delta, matches):
    def states(value):
        return CTCTimestampInputs(
            asr_encoded=torch.full((1, 1, 1), value, dtype=dtype),
            asr_encoded_lengths=torch.tensor([1]),
            sortformer_sigmoids=torch.zeros(1, 1, 1),
            sortformer_lengths=torch.tensor([1]),
            diarization_labels=torch.zeros(1, 1, 1, dtype=torch.bool),
            diarization_lengths=torch.tensor([1]),
        )

    # Adjacent BF16 values can straddle a rounding boundary after an FP32 batch-shape change.
    if matches:
        _assert_states_match(states(0.75), states(0.75 + delta))
    else:
        with pytest.raises(AssertionError, match="asr_encoded"):
            _assert_states_match(states(0.75), states(0.75 + delta))


@pytest.mark.unit
def test_capture_survives_garbage_collection_of_leaked_graphs():
    """Cyclic garbage holding an old CUDAGraph must not be destroyed, and so invalidate the capture, mid-capture."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.zeros(4, device="cuda")
    leaked_graph, _ = capture_cuda_graph(lambda: x + 1)
    leaked = [leaked_graph]
    leaked.append(leaked)
    del leaked_graph, leaked

    def churn_allocations_while_capturing():
        if torch.cuda.is_current_stream_capturing():
            assert not gc.isenabled()
            for _ in range(100_000):
                cycle = []
                cycle.append(cycle)
        return x * 2

    graph, out = capture_cuda_graph(churn_allocations_while_capturing)
    assert gc.isenabled()
    graph.replay()
    torch.testing.assert_close(out, x * 2)


@pytest.mark.unit
def test_graphs_share_their_pool_under_an_allocator_split_limit():
    """vLLM 0.32 profiles, and so captures, under max_split_size_mb:20; graphs must still reuse their shared pool."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pool, graphs, seen = torch.cuda.graph_pool_handle(), [], []
    before = torch._C._accelerator_getAllocatorSettings()
    torch._C._accelerator_setAllocatorSettings("garbage_collection_threshold:0.9,max_split_size_mb:20")
    try:
        # Like the encoder's per-window activations: temporaries just under 20 MiB, a little smaller per graph.
        for k in range(8):
            x = torch.zeros(int((19.9 - 0.15 * k) * 2**20) // 2, device="cuda", dtype=torch.bfloat16)

            def run(x=x):
                if torch.cuda.is_current_stream_capturing():
                    seen.append(torch._C._accelerator_getAllocatorSettings())
                t = x
                for _ in range(20):
                    t = t + 1
                return t.sum()

            graphs.append(capture_cuda_graph(run, pool=pool))
        restored = torch._C._accelerator_getAllocatorSettings()
        pool_bytes = sum(
            segment["total_size"]
            for segment in torch.cuda.memory_snapshot()
            if tuple(segment.get("segment_pool_id", (0, 0))) == tuple(pool)
        )
    finally:
        graphs.clear()
        torch._C._accelerator_setAllocatorSettings(before)
    # Under the limit every temporary takes a new 20 MiB segment: about 3 GiB for these graphs.
    assert pool_bytes < 256 * 2**20
    assert seen == ["garbage_collection_threshold:0.9"] * 8
    assert restored == "garbage_collection_threshold:0.9,max_split_size_mb:20"


@pytest.mark.unit
def test_group_by_length_joins_rows_within_the_length_ratio_and_caps_the_group_size():
    from nemo.collections.speechlm2.vllm.salm.encoder_graphs import _group_by_length

    lengths = [100, 400, 90, 130, 410, 1000]
    # Longest first; a row joins only if within 1.5x of its group's longest row.
    assert _group_by_length(list(range(6)), lengths, max_rows=8) == [[5], [4, 1], [3, 0, 2]]
    assert _group_by_length(list(range(6)), lengths, max_rows=2) == [[5], [4, 1], [3, 0], [2]]
    assert _group_by_length([], lengths, max_rows=8) == []


@pytest.mark.unit
def test_cat_and_take_rows_of_ctc_timestamp_inputs_pad_and_reorder():
    from nemo.collections.asr.modules.parallel_expert_encoder import CTCTimestampInputs
    from nemo.collections.speechlm2.vllm.salm.encoder_graphs import _cat_states, _take_rows

    def inputs(rows, frames, fill):
        return CTCTimestampInputs(
            asr_encoded=torch.full((rows, 3, frames), float(fill)),
            asr_encoded_lengths=torch.full((rows,), frames),
            sortformer_sigmoids=torch.full((rows, frames, 2), float(fill)),
            sortformer_lengths=torch.full((rows,), frames),
        )

    joined = _cat_states([inputs(1, 4, 1), inputs(2, 6, 2)])
    assert joined.asr_encoded.shape == (3, 3, 6) and joined.sortformer_sigmoids.shape == (3, 6, 2)
    # The shorter piece is zero padded on its trailing dimensions, the longer one is untouched.
    assert joined.asr_encoded[0, :, 4:].eq(0).all() and joined.asr_encoded[0, :, :4].eq(1).all()
    assert joined.asr_encoded[1:].eq(2).all()

    swapped = _take_rows(joined, torch.tensor([2, 0, 1]))
    assert swapped.asr_encoded_lengths.tolist() == [6, 4, 6]
    assert _cat_states([joined]) is joined


@pytest.mark.unit
def test_online_graphs_switches_restore_their_previous_value_and_clear_drops_graphs():
    graphs = OnlineGraphs()
    assert graphs.enabled and not graphs.capture_now and graphs.count() == 0
    with graphs.paused():
        assert not graphs.enabled
        with graphs.capturing():
            assert graphs.capture_now
        assert not graphs.capture_now
    assert graphs.enabled

    with pytest.raises(RuntimeError):
        with graphs.paused():
            raise RuntimeError
    assert graphs.enabled  # restored even when the block raises

    graphs._asr_windows = graphs._diar_steps = None
    graphs.clear()
    assert graphs.count() == 0


@pytest.mark.unit
def test_encoder_graphs_route_each_row_by_its_own_length():
    """Short rows replay the whole-forward graphs and long rows run in groups of similar length, in request order."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner, perception, encoder = _long_audio_runner(ctc_timestamp_inputs=True)
    window = encoder.chunk_feat_len
    long_calls = []
    original = runner._call_long

    def spy(processed, processed_len, return_ctc_timestamp_inputs):
        long_calls.append(processed_len.tolist())
        return original(processed, processed_len, return_ctc_timestamp_inputs)

    runner._call_long = spy
    frames = [5 * window + 3, 30, 20 * window, 4 * window + 5, 40, 2 * window - 1]
    audio = torch.randn(len(frames), max(frames) * _MEL_FEATURES, device="cuda")
    lengths = torch.tensor(frames, device="cuda") * _MEL_FEATURES

    embeds, states = runner(audio, lengths, return_ctc_timestamp_inputs=True)
    ref_embs, ref_lens, ref_states = _eager(perception, encoder, audio, lengths, return_ctc_timestamp_inputs=True)

    # No short row ran through the windowed path; the long rows were grouped by length (the 5- and 4-window rows
    # together), and the results came back in request order.
    assert long_calls == [[20 * window], [5 * window + 3, 4 * window + 5], [2 * window - 1]]
    assert len(embeds) == len(frames)
    for i, n in enumerate(ref_lens.tolist()):
        torch.testing.assert_close(embeds[i], ref_embs[i, :n], rtol=1e-5, atol=2e-5)
    _assert_states_match(states, ref_states)


@pytest.mark.unit
def test_encoder_graphs_keep_request_order_within_one_long_group():
    """Rows of one group run longest first; each embedding must still go back to its own row."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner, perception, encoder = _long_audio_runner()
    window = encoder.chunk_feat_len
    frames = torch.tensor([4 * window + 7, 5 * window + 1, 4 * window + 40, 5 * window + 30], device="cuda")
    audio = torch.randn(len(frames), int(frames.max()) * _MEL_FEATURES, device="cuda")
    lengths = frames * _MEL_FEATURES

    got = runner(audio, lengths)
    ref_embs, ref_lens = _eager(perception, encoder, audio, lengths)
    assert [len(g) for g in got] == ref_lens.tolist()
    for i, n in enumerate(ref_lens.tolist()):
        torch.testing.assert_close(got[i], ref_embs[i, :n], rtol=1e-5, atol=2e-5)


@pytest.mark.unit
def test_encoder_graphs_leave_long_rows_to_the_eager_path_without_long_audio():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner, _, encoder = _long_audio_runner(long_audio=False)
    frames = torch.tensor([30, 3 * encoder.chunk_feat_len], device="cuda")
    audio = torch.randn(2, int(frames.max()) * _MEL_FEATURES, device="cuda")
    assert runner(audio, frames * _MEL_FEATURES) is None


@pytest.mark.unit
def test_encoder_graphs_return_ctc_timestamp_inputs_for_long_audio():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner, perception, encoder = _long_audio_runner(ctc_timestamp_inputs=True)
    window = encoder.chunk_feat_len
    frames = torch.tensor([4 * window + 9, 2 * window - 3], device="cuda")
    audio = torch.randn(2, int(frames.max()) * _MEL_FEATURES, device="cuda")
    lengths = frames * _MEL_FEATURES

    embeds, states = runner(audio, lengths, return_ctc_timestamp_inputs=True)
    ref_embs, ref_lens, ref_states = _eager(perception, encoder, audio, lengths, return_ctc_timestamp_inputs=True)
    for i, n in enumerate(ref_lens.tolist()):
        torch.testing.assert_close(embeds[i], ref_embs[i, :n], rtol=1e-5, atol=2e-5)
    _assert_states_match(states, ref_states)


@pytest.mark.unit
def test_asr_window_graphs_pad_rows_to_graph_sizes_and_match_eager():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    asr = ConformerEncoder(**{k: v for k, v in toy_asr_encoder_cfg().items() if k != "_target_"}).cuda().eval()
    graphs = online_graphs_module._ASRWindowGraphs(max_graphs=8)

    def split(rows, frames):
        return [4] * (rows // 4) + ([2] if rows % 4 else [])

    rows = torch.randn(6, _MEL_FEATURES, 96, device="cuda")
    lengths = torch.tensor([96, 90, 96, 40, 96, 77], device="cuda")
    with torch.no_grad():
        expected, _ = asr(audio_signal=rows, length=lengths)
        graphs(asr, rows, lengths, split, capture_now=True)
        got = graphs(asr, rows, lengths, split)
    assert set(graphs.graphs) == {
        (4, (_MEL_FEATURES, 96), rows.dtype, lengths.dtype),
        (2, (_MEL_FEATURES, 96), rows.dtype, lengths.dtype),
    }
    out_lengths = (lengths + _SUBSAMPLING_FACTOR - 1) // _SUBSAMPLING_FACTOR
    for i, n in enumerate(out_lengths.tolist()):
        torch.testing.assert_close(got[i, :, :n], expected[i, :, :n], rtol=1e-5, atol=2e-5)


@pytest.mark.unit
def test_streaming_step_graphs_pad_rows_to_the_bucket_and_match_eager():
    """Diarization steps of 3 rows replay graphs captured for 4 rows and return the eager outputs of the 3."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    encoder = (
        build_toy_pe_encoder(
            online_inference_length=10,
            chunk_left_context=2,
            chunk_right_context=2,
            diar_fifo_len=10,
            diar_spkcache_update_period=20,
            diar_spkcache_len=20,
        )
        .cuda()
        .eval()
    )
    encoder._suppress_online_pbar = True
    lengths = torch.tensor(
        [7 * encoder.chunk_feat_len, 5 * encoder.chunk_feat_len + 3, 6 * encoder.chunk_feat_len], device="cuda"
    )
    mels = torch.randn(3, _MEL_FEATURES, int(lengths.max()), device="cuda")

    with torch.no_grad(), encoder.online_inference():
        expected, expected_lengths = encoder(mels, lengths)
        encoder.online_graphs = OnlineGraphs(bucket=lambda rows: 4)
        with encoder.online_graphs.capturing():
            encoder(mels, lengths)
        seen = len(encoder.online_graphs._diar_steps.seen)
        out, out_lengths = encoder(mels, lengths)

    steps = encoder.online_graphs._diar_steps
    assert steps.graphs and all(
        key[2][0][1][0] == 4 for key in steps.graphs
    ), "steps were not graphed at the bucket size"
    assert len(steps.seen) == seen
    assert torch.equal(out_lengths, expected_lengths)
    for i, n in enumerate(expected_lengths.tolist()):
        torch.testing.assert_close(out[i, :, :n], expected[i, :, :n], rtol=1e-5, atol=2e-5)


@pytest.mark.unit
def test_failed_long_audio_capture_keeps_the_whole_forward_graphs(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner, perception, encoder = _long_audio_runner()

    def fail(n_feat, dtype):
        raise torch.OutOfMemoryError("synthetic")

    monkeypatch.setattr(runner, "_capture_long", fail)
    short = torch.randn(1, 50 * _MEL_FEATURES, device="cuda")
    assert runner(short, torch.tensor([short.shape[1]], device="cuda")) is not None
    assert runner.graphs and runner.enabled and not runner.long_audio
    frames = 4 * encoder.chunk_feat_len
    long = torch.randn(1, frames * _MEL_FEATURES, device="cuda")
    assert runner(long, torch.tensor([long.shape[1]], device="cuda")) is None  # left to the eager path


@pytest.mark.unit
@pytest.mark.parametrize("cause", ["long_audio_off", "capture_failed"])
def test_eager_long_audio_never_captures_window_graphs(monkeypatch, cause):
    """Without long-audio graphs, the eager fallback must not capture window graphs lazily while serving."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    runner, perception, encoder = _long_audio_runner(long_audio=cause != "long_audio_off")
    if cause == "capture_failed":

        def fail(n_feat, dtype):
            raise torch.OutOfMemoryError("synthetic")

        monkeypatch.setattr(runner, "_capture_all", fail)
    short = torch.randn(1, 50 * _MEL_FEATURES, device="cuda")
    runner(short, torch.tensor([short.shape[1]], device="cuda"))
    assert encoder.online_graphs is None
    long = torch.randn(2, 4 * encoder.chunk_feat_len * _MEL_FEATURES, device="cuda")
    lengths = torch.tensor([long.shape[1]] * 2, device="cuda")
    for _ in range(3):
        assert runner(long, lengths) is None
        with torch.no_grad(), encoder.online_inference():
            perception(input_signal=long, input_signal_length=lengths)
    assert runner.long_graph_count() == 0
