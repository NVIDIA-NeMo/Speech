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
    PEETransformerCTCTimestampExtractor,
    ParallelExpertEncoder,
    ParallelExpertEncoderPT,
    TransformerCTCDecoder,
    _clone_config,
    _default_dtype,
    _disable_dist_feature_sync,
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
def test_timestamp_extractor_batches_multiple_records_through_shared_dp(monkeypatch):
    """Unequal recording lengths and speakers share each preliminary/final DP."""
    blank_id = 4
    token_ids = {"a": 0, "b": 1, "c": 2}

    def tokenize_words(words, blank, *, alignment_mode):
        assert blank == blank_id
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    def make_log_probs(labels, padded_frames=8):
        logits = torch.full((padded_frames, blank_id + 1), -12.0)
        for frame_index, label in enumerate(labels):
            logits[frame_index, label] = 12.0
        return torch.log_softmax(logits, dim=-1)

    # Each record has two independent t-SOT speaker streams.  The second CTC
    # row is padded to the first row's width, and its declared length excludes
    # the deliberately unrelated tail.
    ctc_log_probs = torch.stack(
        [
            make_log_probs([blank_id, 0, 0, blank_id, 1, 1, blank_id, blank_id]),
            make_log_probs([blank_id, 2, 2, blank_id, 0, blank_id]),
        ]
    )
    transcripts = ["<spk:0> a <spk:1> b", "<spk:3> c <spk:4> a"]
    ctc_lengths = torch.tensor([8, 6])

    extractor = PEETransformerCTCTimestampExtractor(blank_id=blank_id, alignment_mode="parallel")
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    original_align = extractor._ctc_viterbi_align_batched
    stream_batch_calls = []

    def record_stream_batch(**kwargs):
        stream_batch_calls.append((kwargs["labels"].shape[0], kwargs["use_coarse_alignment"]))
        return original_align(**kwargs)

    monkeypatch.setattr(extractor, "_ctc_viterbi_align_batched", record_stream_batch)
    batched_results = extractor.extract_from_outputs_batch(
        ctc_log_probs=ctc_log_probs,
        sortformer_sigmoids=None,
        sot_transcripts=transcripts,
        ctc_lengths=ctc_lengths,
        alignment_mode="parallel",
    )

    sequential_extractor = PEETransformerCTCTimestampExtractor(blank_id=blank_id, alignment_mode="parallel")
    monkeypatch.setattr(sequential_extractor, "_tokenize_words", tokenize_words)
    sequential_results = [
        sequential_extractor.extract_from_outputs(
            ctc_log_probs=ctc_log_probs[index, :length],
            sortformer_sigmoids=None,
            sot_transcript=transcripts[index],
            alignment_mode="parallel",
        )
        for index, length in enumerate(ctc_lengths.tolist())
    ]

    assert batched_results == sequential_results
    # Two speakers in each of two recordings: all four paths are packed both
    # for exact speaker-assignment evidence and for the final alignment.
    assert stream_batch_calls == [(4, False), (4, True)]


@pytest.mark.unit
def test_timestamp_extractor_batches_repeated_speaker_turn_fences(monkeypatch):
    """A shared serialized guide fences repeated turns in every batch record."""
    blank_id = 4
    token_ids = {"a": 0, "b": 1, "c": 2}

    def tokenize_words(words, blank, *, alignment_mode):
        assert blank == blank_id
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    def make_log_probs(labels, padded_frames=10):
        logits = torch.full((padded_frames, blank_id + 1), -12.0)
        for frame_index, label in enumerate(labels):
            logits[frame_index, label] = 12.0
        return torch.log_softmax(logits, dim=-1)

    # In each record the first and last words belong to the same speaker, with
    # a different speaker's t-SOT turn between them. The CTC paths make all
    # boundaries explicit, so a leaked first/last token is immediately visible.
    ctc_log_probs = torch.stack(
        [
            make_log_probs([blank_id, 0, 0, blank_id, 1, 1, blank_id, 2, 2, blank_id]),
            make_log_probs([blank_id, 2, 2, blank_id, 0, 0, blank_id, 1, 1]),
        ]
    )
    transcripts = [
        "<spk:0> a <spk:1> b <spk:0> c",
        "<spk:3> c <spk:4> a <spk:3> b",
    ]
    ctc_lengths = torch.tensor([10, 9])

    extractor = PEETransformerCTCTimestampExtractor(blank_id=blank_id, alignment_mode="parallel")
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    original_align = extractor._ctc_viterbi_align_batched
    stream_batch_calls = []

    def record_stream_batch(**kwargs):
        stream_batch_calls.append((kwargs["labels"].shape[0], kwargs["use_coarse_alignment"]))
        return original_align(**kwargs)

    monkeypatch.setattr(extractor, "_ctc_viterbi_align_batched", record_stream_batch)
    batched_results = extractor.extract_from_outputs_batch(
        ctc_log_probs=ctc_log_probs,
        sortformer_sigmoids=None,
        sot_transcripts=transcripts,
        ctc_lengths=ctc_lengths,
        alignment_mode="parallel",
    )

    sequential_extractor = PEETransformerCTCTimestampExtractor(blank_id=blank_id, alignment_mode="parallel")
    monkeypatch.setattr(sequential_extractor, "_tokenize_words", tokenize_words)
    sequential_results = [
        sequential_extractor.extract_from_outputs(
            ctc_log_probs=ctc_log_probs[index, :length],
            sortformer_sigmoids=None,
            sot_transcript=transcripts[index],
            alignment_mode="parallel",
        )
        for index, length in enumerate(ctc_lengths.tolist())
    ]

    assert batched_results == sequential_results
    for result in batched_results:
        rows = [row for speaker_rows in result["speaker_word_timestamps"].values() for row in speaker_rows]
        rows = sorted(rows, key=lambda row: row["word_index"])
        assert [row["start_frame"] for row in rows] == [1, 4, 7]
        assert [row["turn_index"] for row in rows] == [0, 1, 2]
    # Four independent per-speaker preliminary paths, two serialized turn
    # guides, then four final fenced speaker paths are each one DP batch.
    assert stream_batch_calls == [(4, False), (2, False), (4, True)]


@pytest.mark.unit
def test_timestamp_extractor_batch_rejects_mismatched_lengths():
    ctc = torch.log_softmax(torch.zeros((2, 4, 3)), dim=-1)
    extractor = PEETransformerCTCTimestampExtractor(blank_id=2)
    with pytest.raises(ValueError, match="ctc_lengths"):
        extractor.extract_from_outputs_batch(
            ctc_log_probs=ctc,
            sortformer_sigmoids=None,
            sot_transcripts=["a", "b"],
            ctc_lengths=torch.tensor([4]),
        )


@pytest.mark.unit
def test_parse_sot_words_retains_each_tag_occurrence_as_a_turn():
    words = PEETransformerCTCTimestampExtractor.parse_sot_words(
        "<spk:0> first turn <spk:1> yes <spk:0> second turn"
    )

    assert [word["speaker_tag"] for word in words] == [0, 0, 1, 0, 0]
    assert [word["turn_index"] for word in words] == [0, 0, 1, 2, 2]


@pytest.mark.unit
def test_parallel_turn_fences_prevent_same_speaker_turn_leakage():
    # Speaker zero appears three times. The serialized anchor puts `did`, the
    # middle sentence, and the later `i` in separate intervals. The source
    # bounds intentionally overlap with speaker one, but never with an adjacent
    # turn of speaker zero.
    words = [
        {"word": "did", "word_index": 0, "speaker_tag": 0, "turn_index": 0},
        {"word": "yeah", "word_index": 1, "speaker_tag": 1, "turn_index": 1},
        {"word": "he", "word_index": 2, "speaker_tag": 0, "turn_index": 2},
        {"word": "book", "word_index": 3, "speaker_tag": 0, "turn_index": 2},
        {"word": "yes", "word_index": 4, "speaker_tag": 1, "turn_index": 3},
        {"word": "i", "word_index": 5, "speaker_tag": 0, "turn_index": 4},
    ]
    anchor_rows = [
        {"word_index": 0, "start_frame": 8, "end_frame": 9},
        {"word_index": 1, "start_frame": 10, "end_frame": 11},
        {"word_index": 2, "start_frame": 15, "end_frame": 16},
        {"word_index": 3, "start_frame": 20, "end_frame": 21},
        {"word_index": 4, "start_frame": 23, "end_frame": 24},
        {"word_index": 5, "start_frame": 30, "end_frame": 31},
    ]

    bounds, diagnostics = PEETransformerCTCTimestampExtractor._build_parallel_turn_frame_bounds(
        tokenized_words=words,
        serialized_anchor_rows=anchor_rows,
        ctc_num_frames=40,
    )

    # Same-speaker cuts are (9 + 15) // 2 = 12 and (21 + 30) // 2 = 25.
    assert bounds[0] == (0, 12)
    assert bounds[2] == bounds[3] == (13, 25)
    assert bounds[5] == (26, 39)
    assert diagnostics[0][1]["turn_index"] == 2
    assert diagnostics[0][1]["min_source_frame"] == 13


@pytest.mark.unit
def test_batched_ctc_viterbi_honors_per_token_source_frame_bounds():
    blank_id = 3
    labels = torch.tensor([[blank_id, 0, blank_id, 1, blank_id, 2, blank_id]], dtype=torch.long)
    # The unbounded logits favor token 1 at frame 6 and token 2 at frame 7.
    # Bounds force the middle token into frames 3--5 and the final token after it.
    logits = torch.full((9, blank_id + 1), -12.0)
    for frame, label in enumerate([blank_id, 0, 0, blank_id, 1, blank_id, 1, 2, blank_id]):
        logits[frame, label] = 12.0
    log_probs = torch.log_softmax(logits, dim=-1)
    state_min = torch.tensor([[0, 0, 0, 3, 0, 6, 0]], dtype=torch.long)
    state_max = torch.tensor([[8, 2, 8, 5, 8, 8, 8]], dtype=torch.long)

    paths, _, _ = PEETransformerCTCTimestampExtractor()._ctc_viterbi_align_batched(
        ctc_log_probs=log_probs,
        labels=labels,
        state_lengths=torch.tensor([labels.shape[1]]),
        blank_id=blank_id,
        state_speaker_columns=torch.full_like(labels, -1),
        speaker_probs=None,
        speaker_logprob_weight=0.0,
        state_min_source_frames=state_min,
        state_max_source_frames=state_max,
    )

    path = paths[0]
    for frame, state in enumerate(path.tolist()):
        if labels[0, state].item() != blank_id:
            assert state_min[0, state].item() <= frame <= state_max[0, state].item()


@pytest.mark.unit
def test_coarse_ctc_viterbi_band_matches_dense_alignment_with_safe_narrow_band():
    blank_id = 4
    labels = [blank_id, 0, blank_id, 1, blank_id, 1, blank_id]
    state_path = [0, 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6]
    logits = torch.full((len(state_path), blank_id + 1), -12.0)
    for frame_index, state in enumerate(state_path):
        logits[frame_index, labels[state]] = 4.0
    ctc_log_probs = torch.log_softmax(logits, dim=-1)
    kwargs = {
        "ctc_log_probs": ctc_log_probs,
        "labels": labels,
        "blank_id": blank_id,
        "state_speaker_columns": [None] * len(labels),
        "speaker_probs": None,
        "speaker_logprob_weight": 0.0,
    }

    dense_path, dense_score, dense_info = PEETransformerCTCTimestampExtractor()._ctc_viterbi_align(**kwargs)
    banded_path, banded_score, banded_info = PEETransformerCTCTimestampExtractor(
        coarse_alignment_band_size=3
    )._ctc_viterbi_align(**kwargs)

    assert torch.equal(banded_path, dense_path)
    assert banded_score == pytest.approx(dense_score)
    assert dense_info["requested_band_size"] is None
    assert banded_info["requested_band_size"] == 3
    assert banded_info["coarse_num_frames"] is not None
    assert banded_info["used_coarse_band"] is True


@pytest.mark.unit
def test_coarse_ctc_viterbi_band_falls_back_when_compact_timeline_cannot_be_compressed():
    blank_id = 3
    labels = torch.tensor([[blank_id, 0, blank_id, 1, blank_id]], dtype=torch.long)
    source_frames = torch.tensor([[0, 1, -1, 4, 5]], dtype=torch.long)
    logits = torch.full((6, blank_id + 1), -12.0)
    for frame_index, label in enumerate([blank_id, 0, blank_id, blank_id, 1, blank_id]):
        logits[frame_index, label] = 4.0
    ctc_log_probs = torch.log_softmax(logits, dim=-1)
    separator_states = torch.tensor([[True, False, True, False, True]])

    paths, scores, diagnostics = PEETransformerCTCTimestampExtractor(
        coarse_alignment_band_size=4
    )._ctc_viterbi_align_batched(
        ctc_log_probs=ctc_log_probs,
        labels=labels,
        state_lengths=torch.tensor([labels.shape[1]]),
        blank_id=blank_id,
        state_speaker_columns=torch.full_like(labels, -1),
        speaker_probs=None,
        speaker_logprob_weight=0.0,
        source_frame_indices=source_frames,
        time_lengths=torch.tensor([source_frames.shape[1]]),
        separator_state_mask=separator_states,
    )

    assert len(paths) == len(scores) == len(diagnostics) == 1
    assert labels[0, paths[0][2]].item() == blank_id
    assert diagnostics[0]["used_coarse_band"] is False
    assert diagnostics[0]["fallback_reason"] == "insufficient_coarse_compression"


@pytest.mark.unit
def test_target_aware_coarse_groups_preserve_regions_and_virtual_separators():
    # Acoustic regions [0, 7) and [8, 16), with a virtual separator at 7.
    # Four interior pairs reduce 15 acoustic frames to the requested 11 while
    # retaining each region endpoint as an individual CTC frame.
    virtual_time_mask = torch.tensor([False] * 7 + [True] + [False] * 8, dtype=torch.bool)
    groups = PEETransformerCTCTimestampExtractor._coarse_time_groups_target_aware(
        num_frames=virtual_time_mask.numel(),
        target_acoustic_groups=11,
        virtual_time_mask=virtual_time_mask,
    )

    assert groups[0] == (0, 1)
    assert (6, 7) in groups
    assert (7, 8) in groups
    assert (8, 9) in groups
    assert groups[-1] == (15, 16)
    assert sum(not bool(virtual_time_mask[start].item()) for start, _ in groups) == 11

    cursor = 0
    for start, end in groups:
        assert start == cursor
        assert start < end <= virtual_time_mask.numel()
        if bool(virtual_time_mask[start].item()):
            assert end - start == 1
        else:
            assert 1 <= end - start <= 2
            assert not virtual_time_mask[start:end].any()
        cursor = end
    assert cursor == virtual_time_mask.numel()


@pytest.mark.unit
def test_target_aware_coarse_groups_avoid_stride_two_fallback():
    # With 29 acoustic frames, a target needing 20 frames cannot use the
    # existing endpoint-preserving uniform stride-two grouping (16 groups).
    # The target-aware mixed 1/2-frame grouping keeps 28 acoustic coarse frames
    # (20 required + 8-frame slack) and enables the band.
    blank_id = 40
    target = [blank_id]
    for token in range(20):
        target.extend([token, blank_id])
    state_path = [0] + [2 * index + 1 for index in range(20)] + [40] * 8
    logits = torch.full((len(state_path), blank_id + 1), -20.0)
    for frame_index, state in enumerate(state_path):
        logits[frame_index, target[state]] = 20.0
    ctc_log_probs = torch.log_softmax(logits, dim=-1)
    kwargs = {
        "ctc_log_probs": ctc_log_probs,
        "labels": torch.tensor([target], dtype=torch.long),
        "state_lengths": torch.tensor([len(target)]),
        "blank_id": blank_id,
        "state_speaker_columns": torch.full((1, len(target)), -1, dtype=torch.long),
        "speaker_probs": None,
        "speaker_logprob_weight": 0.0,
        "source_frame_indices": torch.arange(len(state_path), dtype=torch.long).unsqueeze(0),
        "time_lengths": torch.tensor([len(state_path)]),
        "separator_state_mask": torch.zeros((1, len(target)), dtype=torch.bool),
    }

    dense_paths, dense_scores, _ = PEETransformerCTCTimestampExtractor()._ctc_viterbi_align_batched(**kwargs)
    paths, scores, diagnostics = PEETransformerCTCTimestampExtractor(
        coarse_alignment_band_size=8
    )._ctc_viterbi_align_batched(**kwargs)

    assert torch.equal(paths[0], dense_paths[0])
    assert scores[0] == pytest.approx(dense_scores[0])
    assert diagnostics[0]["used_coarse_band"] is True
    assert diagnostics[0]["fallback_reason"] is None
    assert diagnostics[0]["coarse_stride"] is None
    assert diagnostics[0]["coarse_frame_selection"] == "max_pool"
    assert diagnostics[0]["coarse_hard_speaker_gate"] == "not_requested"
    assert diagnostics[0]["coarse_grouping_mode"] == "target_aware_mixed_1_2"
    assert diagnostics[0]["coarse_target_acoustic_frames"] == 28


@pytest.mark.unit
def test_coarse_guide_disables_hard_gate_and_fine_path_enforces_it():
    # The target-aware groups have one paired region, (26, 28). Only frame 27
    # is speaker-active. A hard-gated representative coarse guide at frame 26
    # has no legal path, while the soft max-pooled guide remains viable. The
    # final fine DP still must enforce the hard gate and use frame 27.
    blank_id = 40
    target = [blank_id]
    for token in range(20):
        target.extend([token, blank_id])
    state_path = [0] + [2 * index + 1 for index in range(19)] + [38] * 7 + [39, 40]
    ctc_log_probs = torch.full((len(state_path), blank_id + 1), -float("inf"))
    for frame_index, state in enumerate(state_path):
        ctc_log_probs[frame_index, target[state]] = 0.0
    speaker_probs = torch.ones((len(state_path), 1))
    speaker_probs[26, 0] = 0.0
    speaker_probs[28, 0] = 0.0
    labels = torch.tensor([target], dtype=torch.long)
    state_columns = torch.tensor(
        [[0 if state_index % 2 else -1 for state_index in range(len(target))]], dtype=torch.long
    )
    kwargs = {
        "ctc_log_probs": ctc_log_probs,
        "labels": labels,
        "state_lengths": torch.tensor([len(target)]),
        "blank_id": blank_id,
        "state_speaker_columns": state_columns,
        "speaker_probs": speaker_probs,
        "speaker_logprob_weight": 0.0,
        "speaker_gate_threshold": 0.5,
        "source_frame_indices": torch.arange(len(state_path), dtype=torch.long).unsqueeze(0),
        "time_lengths": torch.tensor([len(state_path)]),
        "separator_state_mask": torch.zeros((1, len(target)), dtype=torch.bool),
    }

    dense_paths, dense_scores, _ = PEETransformerCTCTimestampExtractor()._ctc_viterbi_align_batched(**kwargs)
    extractor = PEETransformerCTCTimestampExtractor(coarse_alignment_band_size=8)
    paths, scores, diagnostics = extractor._ctc_viterbi_align_batched(**kwargs)

    assert torch.equal(paths[0], dense_paths[0])
    assert scores[0] == pytest.approx(dense_scores[0])
    assert diagnostics[0]["used_coarse_band"] is True
    assert diagnostics[0]["coarse_frame_selection"] == "max_pool"
    assert diagnostics[0]["coarse_hard_speaker_gate"] == "disabled_for_guide"
    assert paths[0][27].item() == 39

    groups = extractor._coarse_time_groups_target_aware(
        num_frames=len(state_path),
        target_acoustic_groups=28,
        virtual_time_mask=torch.zeros(len(state_path), dtype=torch.bool),
    )
    assert [group for group in groups if group[1] - group[0] > 1] == [(26, 28)]
    representative_sources = torch.tensor(
        [[start if end - start == 1 else (start + end - 1) // 2 for start, end in groups]],
        dtype=torch.long,
    )
    representative_states = torch.arange(len(target), dtype=torch.long).view(1, 1, -1).expand(
        1, len(groups), -1
    )
    representative_emissions = extractor._gather_ctc_emissions_for_states(
        log_probs=ctc_log_probs,
        labels=labels,
        state_lengths=torch.tensor([len(target)]),
        source_frame_indices=representative_sources,
        time_lengths=torch.tensor([len(groups)]),
        state_speaker_columns=state_columns,
        speaker_probs=speaker_probs,
        speaker_logprob_weight=0.0,
        speaker_gate_threshold=0.5,
        separator_state_mask=torch.zeros((1, len(target)), dtype=torch.bool),
        state_indices=representative_states,
    )[0]
    with pytest.raises(ValueError, match="No valid CTC Viterbi path"):
        extractor._ctc_viterbi_dp_dense(
            emissions=representative_emissions,
            labels=labels[0],
            blank_id=blank_id,
        )


@pytest.mark.unit
def test_coarse_ctc_viterbi_can_be_forced_dense_for_speaker_mapping():
    blank_id = 3
    labels = [blank_id, 0, blank_id, 1, blank_id]
    logits = torch.full((10, blank_id + 1), -10.0)
    for frame_index, state in enumerate([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]):
        logits[frame_index, labels[state]] = 5.0
    kwargs = {
        "ctc_log_probs": torch.log_softmax(logits, dim=-1),
        "labels": labels,
        "blank_id": blank_id,
        "state_speaker_columns": [None] * len(labels),
        "speaker_probs": None,
        "speaker_logprob_weight": 0.0,
    }
    extractor = PEETransformerCTCTimestampExtractor(coarse_alignment_band_size=2)
    dense_path, dense_score, dense_info = extractor._ctc_viterbi_align(
        **kwargs,
        use_coarse_alignment=False,
    )
    default_path, default_score, default_info = extractor._ctc_viterbi_align(**kwargs)

    assert torch.equal(dense_path, default_path)
    assert dense_score == pytest.approx(default_score)
    assert dense_info["requested_band_size"] is None
    assert dense_info["used_coarse_band"] is False
    assert default_info["requested_band_size"] == 2


@pytest.mark.unit
def test_batched_coarse_ctc_viterbi_prunes_fine_target_states():
    """The parallel fine pass must use its N-sized target corridor, not S."""
    blank_id = 10
    target = [blank_id]
    for token in range(9):
        target.extend([token, blank_id])
    num_states = len(target)
    num_frames = 100
    labels = torch.tensor([target, target], dtype=torch.long)
    logits = torch.full((num_frames, blank_id + 1), -15.0)
    state_path = torch.floor(torch.arange(num_frames) * (num_states - 1) / (num_frames - 1)).long()
    for frame_index, state in enumerate(state_path.tolist()):
        logits[frame_index, target[state]] = 15.0
    ctc_log_probs = torch.log_softmax(logits, dim=-1)
    source_frames = torch.arange(num_frames, dtype=torch.long).unsqueeze(0).expand(2, -1)
    common_kwargs = {
        "ctc_log_probs": ctc_log_probs,
        "labels": labels,
        "state_lengths": torch.tensor([num_states, num_states]),
        "blank_id": blank_id,
        "state_speaker_columns": torch.full_like(labels, -1),
        "speaker_probs": None,
        "speaker_logprob_weight": 0.0,
        "source_frame_indices": source_frames,
        "time_lengths": torch.tensor([num_frames, num_frames - 5]),
        "separator_state_mask": torch.zeros_like(labels, dtype=torch.bool),
    }
    dense_paths, dense_scores, _ = PEETransformerCTCTimestampExtractor()._ctc_viterbi_align_batched(
        **common_kwargs
    )
    banded_extractor = PEETransformerCTCTimestampExtractor(coarse_alignment_band_size=3)
    forced_dense_paths, forced_dense_scores, forced_dense_diagnostics = banded_extractor._ctc_viterbi_align_batched(
        **common_kwargs,
        use_coarse_alignment=False,
    )
    band_paths, band_scores, diagnostics = banded_extractor._ctc_viterbi_align_batched(**common_kwargs)

    for stream_index in range(2):
        assert torch.equal(forced_dense_paths[stream_index], dense_paths[stream_index])
        assert forced_dense_scores[stream_index] == pytest.approx(dense_scores[stream_index])
        assert forced_dense_diagnostics[stream_index]["requested_band_size"] is None
        assert torch.equal(band_paths[stream_index], dense_paths[stream_index])
        assert band_scores[stream_index] == pytest.approx(dense_scores[stream_index])
        assert diagnostics[stream_index]["used_coarse_band"] is True
        assert diagnostics[stream_index]["fine_band_max_states"] <= 2 * 3 + 1
        assert diagnostics[stream_index]["fine_band_max_states"] < num_states


@pytest.mark.unit
@pytest.mark.parametrize("band_size", [-1, True, 1.5])
def test_coarse_ctc_viterbi_band_size_validation(band_size):
    with pytest.raises((TypeError, ValueError), match="coarse_alignment_band_size"):
        PEETransformerCTCTimestampExtractor(coarse_alignment_band_size=band_size)


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
    enc.speaker_feature_mode = "continuous"
    enc.speaker_activity_threshold = None
    enc.asr_norm = nn.LayerNorm(d_model)
    enc.diar_norm = nn.LayerNorm(n_spk)
    enc.register_buffer("diar_kernel", torch.randn(n_spk, d_model))
    enc._suppress_online_pbar = True
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
        ("nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT", True),
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
            nn.Linear(2, 2), str(tmp_path / "out.nemo"), template_bundle_path=str(tmp_path / "tpl.nemo")
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
            '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
            'feat_in': _MEL_FEATURES,
            'feat_out': -1,
            'n_layers': 1,
            'd_model': _ASR_D_MODEL,
            'subsampling': 'dw_striding',
            'subsampling_factor': _SUBSAMPLING_FACTOR,
            'subsampling_conv_channels': 16,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rel_pos',
            'n_heads': 4,
            'att_context_size': [-1, -1],
            'conv_kernel_size': 9,
            'dropout': 0.0,
            'dropout_pre_encoder': 0.0,
            'dropout_emb': 0.0,
            'dropout_att': 0.0,
        }
    )


def toy_diarization_model_cfg() -> DictConfig:
    """Tiny SortformerEncLabelModel config the PE encoder mounts as its diar branch."""
    model_defaults = {'fc_d_model': _DIAR_FC_D_MODEL, 'tf_d_model': _DIAR_TF_D_MODEL}
    return DictConfig(
        {
            'target': 'nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel',
            'sample_rate': 16000,
            'pil_weight': 0.5,
            'ats_weight': 0.5,
            'max_num_of_spks': _N_SPK,
            'streaming_mode': False,
            'async_streaming': False,
            'model_defaults': DictConfig(model_defaults),
            'preprocessor': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
                    'normalize': 'per_feature',
                    'window_size': 0.025,
                    'sample_rate': 16000,
                    'window_stride': 0.01,
                    'window': 'hann',
                    'features': _MEL_FEATURES,
                    'n_fft': 512,
                    'frame_splicing': 1,
                    'dither': 0.00001,
                }
            ),
            'encoder': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
                    'feat_in': _MEL_FEATURES,
                    'feat_out': -1,
                    'n_layers': 1,
                    'd_model': _DIAR_FC_D_MODEL,
                    'subsampling': 'dw_striding',
                    'subsampling_factor': _SUBSAMPLING_FACTOR,
                    'subsampling_conv_channels': 16,
                    'causal_downsampling': False,
                    'ff_expansion_factor': 4,
                    'self_attention_model': 'rel_pos',
                    'n_heads': 4,
                    'att_context_size': [-1, -1],
                    'conv_kernel_size': 9,
                    'conv_norm_type': 'batch_norm',
                    'dropout': 0.0,
                    'dropout_pre_encoder': 0.0,
                    'dropout_emb': 0.0,
                    'dropout_att': 0.0,
                }
            ),
            'transformer_encoder': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.transformer.transformer_encoders.TransformerEncoder',
                    'num_layers': 1,
                    'hidden_size': _DIAR_TF_D_MODEL,
                    'inner_size': 32,
                    'num_attention_heads': 4,
                    'attn_score_dropout': 0.0,
                    'attn_layer_dropout': 0.0,
                    'ffn_dropout': 0.0,
                    'hidden_act': 'relu',
                    'pre_ln': False,
                    'pre_ln_final_layer_norm': True,
                }
            ),
            'sortformer_modules': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.sortformer_modules.SortformerModules',
                    'num_spks': _N_SPK,
                    'dropout_rate': 0.0,
                    'fc_d_model': _DIAR_FC_D_MODEL,
                    'tf_d_model': _DIAR_TF_D_MODEL,
                }
            ),
            'loss': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.losses.bce_loss.BCELoss',
                    'weight': None,
                    'reduction': 'mean',
                }
            ),
        }
    )


def build_toy_pe_encoder(**overrides) -> ParallelExpertEncoder:
    """Construct a real ParallelExpertEncoder from the tiny ASR + diar configs."""
    kwargs = dict(
        asr_encoder_cfg=toy_asr_encoder_cfg(),
        diarization_model_cfg=toy_diarization_model_cfg(),
        asr_normalize_type='per_feature',
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

    batch_size, n_frames = 1, 320  # > online_inference_length * subsampling_factor (=80)
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
@pytest.mark.run_only_on('GPU')
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
@pytest.mark.run_only_on('GPU')
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
@pytest.mark.run_only_on('GPU')
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
@pytest.mark.run_only_on('GPU')
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

    batch_size, n_frames = 1, 320  # > online_inference_length * subsampling_factor (=80)
    mels = torch.randn(batch_size, _MEL_FEATURES, n_frames, device="cuda", dtype=torch.float32)
    length = torch.full((batch_size,), n_frames, dtype=torch.long, device="cuda")

    with torch.no_grad():
        outputs, encoded_len = enc(mels, length)

    expected_t = int(encoded_len[0].item())
    assert outputs.is_cuda
    assert outputs.shape == (batch_size, _ASR_D_MODEL, expected_t)
    assert expected_t > 0
    assert torch.isfinite(outputs).all()


@pytest.mark.unit
def test_timestamp_extractor_serializes_when_sot_exceeds_active_sortformer_columns(monkeypatch):
    """t-SOT remains intact when Sortformer has too few active output streams."""
    blank_id = 3
    token_ids = {"a": 0, "b": 1}

    def tokenize_words(words, blank, *, alignment_mode):
        assert blank == blank_id
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    logits = torch.full((7, blank_id + 1), -12.0)
    for frame_index, label in enumerate([blank_id, 0, 0, blank_id, 1, 1, blank_id]):
        logits[frame_index, label] = 12.0

    extractor = PEETransformerCTCTimestampExtractor(blank_id=blank_id, alignment_mode="parallel")
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    result = extractor.extract_from_outputs(
        ctc_log_probs=torch.log_softmax(logits, dim=-1),
        sortformer_sigmoids=torch.tensor(
            [[0.9, 0.1, 0.1]] * 7,
            dtype=torch.float32,
        ),
        sot_transcript="<spk:0> a <spk:1> b",
        alignment_mode="parallel",
        parallel_speaker_gate_threshold=None,
    )

    assert result["requested_alignment_mode"] == "parallel"
    assert result["alignment_mode"] == "serialized"
    assert result["speaker_tag_to_sortformer_column"] == {0: None, 1: None}
    assert result["alignment_diagnostics"]["speaker_count_policy"]["reason"] == (
        "sot_speakers_exceed_active_sortformer_columns"
    )
    rows = sorted(
        (row for speaker_rows in result["speaker_word_timestamps"].values() for row in speaker_rows),
        key=lambda row: row["word_index"],
    )
    assert [(row["word"], row["speaker_tag"]) for row in rows] == [("a", 0), ("b", 1)]


@pytest.mark.unit
def test_timestamp_extractor_ignores_least_active_extra_sortformer_column(monkeypatch):
    """Optimal mapping excludes low-total-activity Sortformer columns first."""
    blank_id = 3
    token_ids = {"a": 0, "b": 1}

    def tokenize_words(words, blank, *, alignment_mode):
        assert blank == blank_id
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    logits = torch.full((7, blank_id + 1), -12.0)
    for frame_index, label in enumerate([blank_id, 0, 0, blank_id, 1, 1, blank_id]):
        logits[frame_index, label] = 12.0
    # Column 0 has strong local evidence for the first word but the lowest
    # total speech mass. The policy must remove it before optimal assignment.
    sortformer = torch.tensor(
        [
            [0.99, 0.80, 0.70],
            [0.99, 0.80, 0.70],
            [0.01, 0.80, 0.70],
            [0.01, 0.80, 0.70],
            [0.01, 0.80, 0.70],
            [0.01, 0.80, 0.70],
            [0.01, 0.80, 0.70],
        ]
    )

    extractor = PEETransformerCTCTimestampExtractor(blank_id=blank_id, alignment_mode="parallel")
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    result = extractor.extract_from_outputs(
        ctc_log_probs=torch.log_softmax(logits, dim=-1),
        sortformer_sigmoids=sortformer,
        sot_transcript="<spk:0> a <spk:1> b",
        alignment_mode="parallel",
        speaker_logprob_weight=0.0,
        parallel_speaker_gate_threshold=None,
    )

    policy = result["alignment_diagnostics"]["speaker_count_policy"]
    assert result["alignment_mode"] == "parallel"
    assert policy["selected_sortformer_columns"] == [1, 2]
    assert policy["ignored_sortformer_columns"] == [0]
    assert set(result["speaker_tag_to_sortformer_column"].values()) == {1, 2}


@pytest.mark.unit
def test_timestamp_extractor_batch_applies_speaker_count_policy_per_record(monkeypatch):
    """One padded batch can contain serialized-fallback and parallel records."""
    blank_id = 3
    token_ids = {"a": 0, "b": 1, "c": 2}

    def tokenize_words(words, blank, *, alignment_mode):
        assert blank == blank_id
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    def make_log_probs(labels, padded_frames=10):
        logits = torch.full((padded_frames, blank_id + 1), -12.0)
        for frame_index, label in enumerate(labels):
            logits[frame_index, label] = 12.0
        return torch.log_softmax(logits, dim=-1)

    ctc_log_probs = torch.stack(
        [
            make_log_probs([blank_id, 0, 0, blank_id, 1, 1, blank_id, 2, 2, blank_id]),
            make_log_probs([blank_id, 0, 0, blank_id]),
        ]
    )
    sortformer_sigmoids = torch.tensor(
        [
            [[0.8, 0.7]] * 10,
            [[0.1, 0.9]] * 4 + [[0.0, 0.0]] * 6,
        ]
    )

    extractor = PEETransformerCTCTimestampExtractor(blank_id=blank_id, alignment_mode="parallel")
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    results = extractor.extract_from_outputs_batch(
        ctc_log_probs=ctc_log_probs,
        sortformer_sigmoids=sortformer_sigmoids,
        sot_transcripts=["<spk:0> a <spk:1> b <spk:2> c", "<spk:4> a"],
        ctc_lengths=torch.tensor([10, 4]),
        sortformer_lengths=torch.tensor([10, 4]),
        alignment_mode="parallel",
        speaker_logprob_weight=0.0,
        parallel_speaker_gate_threshold=None,
    )

    assert [result["alignment_mode"] for result in results] == ["serialized", "parallel"]
    assert results[0]["speaker_tag_to_sortformer_column"] == {0: None, 1: None, 2: None}
    assert results[1]["speaker_tag_to_sortformer_column"] == {4: 1}


@pytest.mark.unit
def test_timestamp_extractor_retries_lower_parallel_gate_before_serializing(monkeypatch):
    """A capacity-limited speaker retries at a lower gate and stays parallel."""
    blank_id = 2

    def tokenize_words(words, blank, *, alignment_mode):
        assert blank == blank_id
        return [dict(word, token_ids=[0]) for word in words]

    logits = torch.full((5, blank_id + 1), -12.0)
    for frame_index, label in enumerate([blank_id, 0, blank_id, 0, blank_id]):
        logits[frame_index, label] = 12.0
    extractor = PEETransformerCTCTimestampExtractor(
        blank_id=blank_id,
        alignment_mode="parallel",
        speaker_logprob_weight=0.0,
        parallel_speaker_gate_threshold=0.5,
        parallel_speaker_gate_min_threshold=0.2,
        parallel_active_region_padding_seconds=0.0,
        parallel_active_region_merge_gap_seconds=0.0,
    )
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)

    result = extractor.extract_from_outputs(
        ctc_log_probs=torch.log_softmax(logits, dim=-1),
        sortformer_sigmoids=torch.tensor([[0.9], [0.9], [0.4], [0.4], [0.0]]),
        sot_transcript="<spk:0> a a",
        alignment_mode="parallel",
    )

    assert result["requested_alignment_mode"] == "parallel"
    assert result["alignment_mode"] == "parallel"
    assert result["speaker_tag_to_sortformer_column"] == {0: 0}
    retry = result["alignment_diagnostics"]["parallel_active_regions"][0]["adaptive_gate_retry"]
    assert retry["attempted_thresholds"] == [0.5, 0.4]
    assert retry["selected_threshold"] == 0.4
    assert retry["gate_floor"] == 0.2
    assert result["alignment_diagnostics"]["alignment_fallback"] is None
    assert [row["word"] for row in result["speaker_word_timestamps"][0]] == ["a", "a"]


@pytest.mark.unit
def test_timestamp_extractor_serializes_after_parallel_gate_floor(monkeypatch):
    """No full parallel timeline is opened when a speaker still has no CTC path at the floor."""
    blank_id = 2
    token_ids = {"a": 0, "b": 1}

    def tokenize_words(words, blank, *, alignment_mode):
        assert blank == blank_id
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    # The active timeline contains only the first two CTC frames at every gate
    # in [0.5, 0.4, 0.3, 0.25, 0.2], so token b has no valid parallel path.
    # The complete serialized transcript can reach b at frame three.
    ctc_log_probs = torch.full((5, blank_id + 1), float("-inf"))
    for frame_index, label in enumerate([blank_id, 0, 0, 1, blank_id]):
        ctc_log_probs[frame_index, label] = 0.0
    extractor = PEETransformerCTCTimestampExtractor(
        blank_id=blank_id,
        alignment_mode="parallel",
        speaker_logprob_weight=0.0,
        parallel_speaker_gate_threshold=0.5,
        parallel_speaker_gate_min_threshold=0.2,
        parallel_active_region_padding_seconds=0.0,
        parallel_active_region_merge_gap_seconds=0.0,
    )
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)

    result = extractor.extract_from_outputs(
        ctc_log_probs=ctc_log_probs,
        sortformer_sigmoids=torch.tensor([[0.9], [0.9], [0.1], [0.1], [0.1]]),
        sot_transcript="<spk:0> a b",
        alignment_mode="parallel",
    )

    assert result["requested_alignment_mode"] == "parallel"
    assert result["alignment_mode"] == "serialized"
    assert result["speaker_tag_to_sortformer_column"] == {0: None}
    fallback = result["alignment_diagnostics"]["alignment_fallback"]
    assert fallback["from_alignment_mode"] == "parallel"
    assert fallback["to_alignment_mode"] == "serialized"
    assert fallback["reason"] == "no_valid_ctc_viterbi_path_in_active_regions"
    assert fallback["attempted_gate_thresholds"] == {"0": [0.5, 0.4, 0.3, 0.25, 0.2]}
    assert fallback["gate_floor"] == 0.2
    assert [row["word"] for row in result["speaker_word_timestamps"][0]] == ["a", "b"]


@pytest.mark.unit
def test_timestamp_extractor_batch_retries_or_serializes_per_record(monkeypatch):
    """A terminal fallback in one batch item leaves the healthy item parallel."""
    blank_id = 2
    token_ids = {"a": 0, "b": 1}

    def tokenize_words(words, blank, *, alignment_mode):
        assert blank == blank_id
        return [dict(word, token_ids=[token_ids[word["word"]]]) for word in words]

    retry_logits = torch.full((5, blank_id + 1), -12.0)
    for frame_index, label in enumerate([blank_id, 0, blank_id, 0, blank_id]):
        retry_logits[frame_index, label] = 12.0
    floor_logits = torch.full((5, blank_id + 1), float("-inf"))
    for frame_index, label in enumerate([blank_id, 0, 0, 1, blank_id]):
        floor_logits[frame_index, label] = 0.0

    extractor = PEETransformerCTCTimestampExtractor(
        blank_id=blank_id,
        alignment_mode="parallel",
        speaker_logprob_weight=0.0,
        parallel_speaker_gate_threshold=0.5,
        parallel_speaker_gate_min_threshold=0.2,
        parallel_active_region_padding_seconds=0.0,
        parallel_active_region_merge_gap_seconds=0.0,
    )
    monkeypatch.setattr(extractor, "_tokenize_words", tokenize_words)
    results = extractor.extract_from_outputs_batch(
        ctc_log_probs=torch.stack([torch.log_softmax(retry_logits, dim=-1), floor_logits]),
        sortformer_sigmoids=torch.tensor(
            [
                [[0.9], [0.9], [0.4], [0.4], [0.0]],
                [[0.9], [0.9], [0.1], [0.1], [0.1]],
            ]
        ),
        sot_transcripts=["<spk:0> a a", "<spk:0> a b"],
        ctc_lengths=torch.tensor([5, 5]),
        sortformer_lengths=torch.tensor([5, 5]),
        alignment_mode="parallel",
    )

    assert [result["alignment_mode"] for result in results] == ["parallel", "serialized"]
    assert results[0]["alignment_diagnostics"]["parallel_active_regions"][0]["adaptive_gate_retry"][
        "selected_threshold"
    ] == 0.4
    assert results[1]["alignment_diagnostics"]["alignment_fallback"]["to_alignment_mode"] == "serialized"
    assert [row["word"] for row in results[1]["speaker_word_timestamps"][0]] == ["a", "b"]
