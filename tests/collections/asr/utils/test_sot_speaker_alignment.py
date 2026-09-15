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

from itertools import permutations

import numpy as np
import pytest
import torch
from lhotse.testing.dummies import dummy_cut, dummy_recording

import nemo.collections.asr.parts.utils.sot_speaker_alignment as sot_alignment
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_hidden_length_from_sample_length


@pytest.mark.unit
def test_parse_speaker_tokens_handles_multi_digit_speakers():
    assert sot_alignment.parse_speaker_tokens("<spk:10> hello <spk:1> world") == [10, 1]


@pytest.mark.unit
def test_sl_and_wl_sot_have_same_speaker_sequence():
    sl_text = "<spk:0> hello world <spk:1> yes"
    wl_text = sot_alignment.sl_to_wl_sot(sl_text)

    assert wl_text == "<spk:0> hello <spk:0> world <spk:1> yes"
    assert sot_alignment.parse_speaker_tokens(sl_text) == sot_alignment.parse_speaker_tokens(wl_text)


@pytest.mark.unit
def test_ensure_single_speaker_sot_prefixes_no_token_text():
    text, spk_idx, changed = sot_alignment.ensure_single_speaker_sot("hello world")

    assert text == "<spk:0> hello world"
    assert spk_idx == 0
    assert changed


@pytest.mark.unit
def test_ensure_single_speaker_sot_keeps_existing_tokens():
    text, spk_idx, changed = sot_alignment.ensure_single_speaker_sot("<spk:2> hello")

    assert text == "<spk:2> hello"
    assert spk_idx == -1
    assert not changed


@pytest.mark.unit
def test_get_speaker_token_index_map_requires_canonical_in_range_tokens():
    assert sot_alignment.get_speaker_token_index_map("<spk:1> one <spk:0> zero <spk:1> again", 2) == {
        "<spk:1>": 1,
        "<spk:0>": 0,
    }
    assert sot_alignment.get_speaker_token_index_map("<spk:01> one", 2) is None
    assert sot_alignment.get_speaker_token_index_map("<spk:2> two", 2) is None


@pytest.mark.unit
def test_speaker_activity_from_cut_uses_exact_pr_rttm_labels(tmp_path):
    cut = dummy_cut(0, duration=0.04, recording=dummy_recording(0, duration=0.04, with_data=True))
    rttm_path = tmp_path / "permutation_resolved.rttm"
    rttm_path.write_text(
        "\n".join(
            [
                f"SPEAKER {cut.recording_id} 0 0.000 0.020 <NA> <NA> <spk:1> <NA> <NA>",
                f"SPEAKER {cut.recording_id} 0 0.020 0.020 <NA> <NA> <spk:0> <NA> <NA>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cut.custom = {"rttm_filepath": str(rttm_path)}
    kwargs = {
        "num_speakers": 2,
        "num_sample_per_mel_frame": 160,
        "num_mel_frame_per_target_frame": 1,
    }

    arrival_order_activity = sot_alignment.speaker_activity_from_cut(cut, **kwargs)
    resolved_activity, is_permutation_resolved = sot_alignment.speaker_activity_from_cut(
        cut,
        text="<spk:0> zero <spk:1> one",
        return_permutation_resolved=True,
        **kwargs,
    )

    assert is_permutation_resolved
    assert torch.equal(resolved_activity, arrival_order_activity[:, [1, 0]])

    incomplete_activity, is_permutation_resolved = sot_alignment.speaker_activity_from_cut(
        cut,
        text="<spk:0> zero only",
        return_permutation_resolved=True,
        **kwargs,
    )
    assert not is_permutation_resolved
    assert torch.equal(incomplete_activity, arrival_order_activity)


@pytest.mark.unit
def test_speaker_activity_from_cut_preserves_eight_pr_rttm_columns(tmp_path):
    cut = dummy_cut(0, duration=0.08, recording=dummy_recording(0, duration=0.08, with_data=True))
    rttm_path = tmp_path / "eight_speakers.rttm"
    rttm_path.write_text(
        "\n".join(
            f"SPEAKER {cut.recording_id} 0 {(7 - speaker) * 0.01:.3f} 0.010 <NA> <NA> " f"<spk:{speaker}> <NA> <NA>"
            for speaker in range(8)
        )
        + "\n",
        encoding="utf-8",
    )
    cut.custom = {"rttm_filepath": str(rttm_path)}
    text = " ".join(f"<spk:{speaker}> utterance-{speaker}" for speaker in range(8))

    activity, is_permutation_resolved = sot_alignment.speaker_activity_from_cut(
        cut,
        num_speakers=8,
        num_sample_per_mel_frame=160,
        num_mel_frame_per_target_frame=1,
        text=text,
        return_permutation_resolved=True,
    )

    assert is_permutation_resolved
    assert activity.shape == (8, 8)
    assert torch.equal(activity.sum(dim=0), torch.ones(8))


@pytest.mark.unit
def test_speaker_activity_from_cut_keeps_legacy_mapping_when_labels_do_not_match(tmp_path):
    cut = dummy_cut(0, duration=0.04, recording=dummy_recording(0, duration=0.04, with_data=True))
    rttm_path = tmp_path / "legacy.rttm"
    rttm_path.write_text(
        "\n".join(
            [
                f"SPEAKER {cut.recording_id} 0 0.000 0.020 <NA> <NA> speaker_b <NA> <NA>",
                f"SPEAKER {cut.recording_id} 0 0.020 0.020 <NA> <NA> speaker_a <NA> <NA>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cut.custom = {"rttm_filepath": str(rttm_path)}
    kwargs = {
        "num_speakers": 2,
        "num_sample_per_mel_frame": 160,
        "num_mel_frame_per_target_frame": 1,
    }

    legacy_activity = sot_alignment.speaker_activity_from_cut(cut, **kwargs)
    detected_activity, is_permutation_resolved = sot_alignment.speaker_activity_from_cut(
        cut,
        text="<spk:0> zero <spk:1> one",
        return_permutation_resolved=True,
        **kwargs,
    )

    assert not is_permutation_resolved
    assert torch.equal(detected_activity, legacy_activity)


@pytest.mark.unit
def test_speaker_activity_from_cut_rejects_more_speakers_than_configured(tmp_path):
    cut = dummy_cut(0, duration=0.05, recording=dummy_recording(0, duration=0.05, with_data=True))
    rttm_path = tmp_path / "five_speakers.rttm"
    rttm_path.write_text(
        "\n".join(
            f"SPEAKER {cut.recording_id} 0 {speaker * 0.01:.3f} 0.010 <NA> <NA> <spk:{speaker}> <NA> <NA>"
            for speaker in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    cut.custom = {"rttm_filepath": str(rttm_path)}

    with pytest.raises(ValueError, match="contains 5 speakers.*num_speakers=4"):
        sot_alignment.speaker_activity_from_cut(
            cut,
            num_speakers=4,
            num_sample_per_mel_frame=160,
            num_mel_frame_per_target_frame=1,
        )


@pytest.mark.unit
def test_fix_speaker_activity_swaps_simple_two_speaker_permutation():
    activity = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    fixed = sot_alignment.fix_speaker_activity("<spk:0> hello world <spk:1> yes now", activity, num_speakers=2)

    expected = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    assert torch.equal(fixed, expected)


@pytest.mark.unit
def test_fix_speaker_activity_empty_text_is_noop():
    activity = torch.tensor([[1.0, 0.0]])

    fixed = sot_alignment.fix_speaker_activity("", activity, num_speakers=2)

    assert fixed is activity


@pytest.mark.unit
def test_dtw_cost_batch_streaming_matches_naive_reference():
    activity = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.bool_,
    )
    spk_seq = np.array([0, 0, 2, 1], dtype=np.intp)
    perm_batch = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=np.intp)
    token_weights = np.array([0.5, 1.5, 2.0, 0.75], dtype=np.float32)

    actual = sot_alignment.dtw_cost_batch(activity, spk_seq, perm_batch, num_speakers=3, token_weights=token_weights)

    expected = []
    activity_sum = np.maximum(np.count_nonzero(activity, axis=1), 1).astype(np.float32)
    for perm in perm_batch:
        permuted = activity[:, perm]
        local = 1.0 - permuted[:, spk_seq].T.astype(np.float32) / activity_sum
        local *= token_weights[:, np.newaxis]
        costs = np.full(local.shape, np.inf, dtype=np.float32)
        costs[0] = np.cumsum(local[0], dtype=np.float32)
        for token_idx in range(1, local.shape[0]):
            costs[token_idx, 0] = costs[token_idx - 1, 0] + local[token_idx, 0]
            for frame_idx in range(1, local.shape[1]):
                costs[token_idx, frame_idx] = local[token_idx, frame_idx] + min(
                    costs[token_idx - 1, frame_idx],
                    costs[token_idx - 1, frame_idx - 1],
                    costs[token_idx, frame_idx - 1],
                )
        expected.append(costs[-1, -1] / sum(local.shape))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_coarsen_activity_uses_binary_majority_bins():
    activity = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.bool_,
    )

    coarse = sot_alignment._coarsen_activity_for_alignment(activity, max_frames=3)

    assert coarse.dtype == np.bool_
    np.testing.assert_array_equal(coarse, [[False, False], [False, True], [True, False]])
    # Sequences at or below the bound avoid both a copy and any resolution loss.
    assert sot_alignment._coarsen_activity_for_alignment(activity, max_frames=6) is activity


@pytest.mark.unit
def test_default_alignment_grid_has_80ms_floor_and_no_redundant_frames():
    max_frames = sot_alignment._DEFAULT_MAX_ALIGNMENT_FRAMES
    at_80ms_floor = np.zeros((max_frames, 2), dtype=np.bool_)

    # 1,200 * 80 ms = 96 seconds: do not upsample or manufacture extra bins.
    assert sot_alignment._coarsen_activity_for_alignment(at_80ms_floor, max_frames) is at_80ms_floor

    # Immediately above the threshold, produce exactly 1,200 bins. Since there
    # are 1,201 source frames, integer proportional boundaries are strictly
    # increasing and each output bin consumes one or two real source frames.
    just_above_floor = np.zeros((max_frames + 1, 2), dtype=np.bool_)
    coarse = sot_alignment._coarsen_activity_for_alignment(just_above_floor, max_frames)
    assert coarse.shape == (max_frames, 2)


@pytest.mark.unit
def test_fix_speaker_activity_caps_one_hour_alignment_at_1200_frames(monkeypatch):
    base_frame_seconds = 0.08
    duration_seconds = 3600
    num_frames = round(duration_seconds / base_frame_seconds)
    activity = torch.zeros(num_frames, 2)
    activity[: num_frames // 2, 1] = 1.0
    activity[num_frames // 2 :, 0] = 1.0

    observed = {}
    original_dtw_cost_batch = sot_alignment.dtw_cost_batch

    def capture_alignment_shape(activity, *args, **kwargs):
        observed["num_frames"] = activity.shape[0]
        return original_dtw_cost_batch(activity, *args, **kwargs)

    monkeypatch.setattr(sot_alignment, "dtw_cost_batch", capture_alignment_shape)
    fixed = sot_alignment.fix_speaker_activity("<spk:0> hello world <spk:1> yes now", activity, num_speakers=2)

    assert observed["num_frames"] == 1200
    assert duration_seconds / observed["num_frames"] == pytest.approx(3.0)
    # Coarsening is alignment-only: the returned training target keeps all 45,000 frames.
    assert fixed.shape == activity.shape
    assert torch.equal(fixed, activity[:, [1, 0]])


@pytest.mark.unit
def test_fix_speaker_activity_keeps_exhaustive_search_through_six_speakers(monkeypatch):
    num_speakers = 6
    frames_per_speaker = 4
    activity = torch.zeros(num_speakers * frames_per_speaker, num_speakers)
    text_parts = []
    for speaker_idx in range(num_speakers):
        activity[speaker_idx * frames_per_speaker : (speaker_idx + 1) * frames_per_speaker, speaker_idx] = 1
        text_parts.append(f"<spk:{speaker_idx}> word")

    observed = {}
    original_dtw_cost_batch = sot_alignment.dtw_cost_batch

    def capture_permutation_count(activity, spk_seq_arr, perm_batch, *args, **kwargs):
        observed["num_permutations"] = perm_batch.shape[0]
        return original_dtw_cost_batch(activity, spk_seq_arr, perm_batch, *args, **kwargs)

    monkeypatch.setattr(sot_alignment, "dtw_cost_batch", capture_permutation_count)
    fixed = sot_alignment.fix_speaker_activity(" ".join(text_parts), activity, num_speakers=num_speakers)

    assert observed["num_permutations"] == 720
    assert torch.equal(fixed, activity)


@pytest.mark.unit
def test_fix_speaker_activity_bounds_eight_speaker_search_and_recovers_temporal_mapping(monkeypatch):
    num_speakers = 8
    frames_per_speaker = 40
    source_column_for_text_speaker = [7, 5, 3, 1, 6, 4, 2, 0]
    activity = torch.zeros(num_speakers * frames_per_speaker, num_speakers)
    expected = torch.zeros_like(activity)
    text_parts = []
    for text_speaker, source_column in enumerate(source_column_for_text_speaker):
        frame_slice = slice(text_speaker * frames_per_speaker, (text_speaker + 1) * frames_per_speaker)
        activity[frame_slice, source_column] = 1
        expected[frame_slice, text_speaker] = 1
        text_parts.append(f"<spk:{text_speaker}> " + " ".join([f"word{text_speaker}"] * 10))

    observed = {}
    original_dtw_cost_batch = sot_alignment.dtw_cost_batch

    def capture_permutation_count(activity, spk_seq_arr, perm_batch, *args, **kwargs):
        observed["num_permutations"] = perm_batch.shape[0]
        return original_dtw_cost_batch(activity, spk_seq_arr, perm_batch, *args, **kwargs)

    monkeypatch.setattr(sot_alignment, "dtw_cost_batch", capture_permutation_count)
    fixed = sot_alignment.fix_speaker_activity(" ".join(text_parts), activity, num_speakers=num_speakers)

    assert observed["num_permutations"] == 720
    assert torch.equal(fixed, expected)


@pytest.mark.unit
def test_fix_speaker_activity_exactly_collapses_text_unused_permutation_classes(monkeypatch):
    num_speakers = 8
    activity = torch.zeros(80, num_speakers)
    for speaker_idx in range(num_speakers):
        activity[speaker_idx * 10 : (speaker_idx + 1) * 10, speaker_idx] = 1
    text = "<spk:0> alpha beta <spk:2> gamma delta"

    observed = {"num_permutations": []}
    original_dtw_cost_batch = sot_alignment.dtw_cost_batch

    def capture_permutation_count(activity, spk_seq_arr, perm_batch, *args, **kwargs):
        observed["num_permutations"].append(perm_batch.shape[0])
        return original_dtw_cost_batch(activity, spk_seq_arr, perm_batch, *args, **kwargs)

    monkeypatch.setattr(sot_alignment, "dtw_cost_batch", capture_permutation_count)
    bounded = sot_alignment.fix_speaker_activity(text, activity, num_speakers=num_speakers)

    spk_seq_arr = np.asarray(sot_alignment.parse_speaker_tokens(text), dtype=np.intp)
    perm_batch = np.asarray(list(permutations(range(num_speakers))), dtype=np.intp)
    token_counts = np.maximum(np.bincount(spk_seq_arr, minlength=num_speakers), 1.0)
    token_weights = (spk_seq_arr.size / token_counts)[spk_seq_arr]
    activity_np = activity.numpy().astype(np.bool_)
    text_freq = sot_alignment.get_text_speaker_char_counts(text, num_speakers)
    rttm_freq = activity_np.sum(axis=0).astype(np.float32)
    rttm_freq /= rttm_freq.sum()
    exhaustive_costs = original_dtw_cost_batch(
        activity_np,
        spk_seq_arr,
        perm_batch,
        num_speakers,
        token_weights,
    ) + sot_alignment.speaker_freq_cost_batch(text_freq, rttm_freq, perm_batch)
    exhaustive = activity[:, perm_batch[int(np.argmin(exhaustive_costs))]].clone()
    exhaustive[:, [speaker_idx for speaker_idx in range(num_speakers) if speaker_idx not in {0, 2}]] = 0

    # Two text-used slots can be injectively assigned to eight active RTTM
    # columns in P(8, 2) = 56 distinct ways.  Reordering the remaining six
    # columns cannot affect DTW and those output columns are zeroed afterward.
    assert observed["num_permutations"] == [56]
    assert torch.equal(bounded, exhaustive)


@pytest.mark.unit
@pytest.mark.parametrize(
    "n_spk_in, num_speakers",
    [
        (4, 4),  # speaker dim == num_speakers -> unchanged
        (2, 4),  # speaker dim < num_speakers -> zero-pad missing columns
    ],
)
def test_collate_speaker_activity_targets_normalizes_speaker_axis(n_spk_in, num_speakers):
    num_frames = 2
    # Distinct per-element values so padding is observable.
    activity = torch.arange(1, num_frames * n_spk_in + 1, dtype=torch.float32).reshape(num_frames, n_spk_in)

    targets, target_length = sot_alignment.collate_speaker_activity_targets(
        [activity],
        audio_lens=torch.tensor([2560]),
        num_speakers=num_speakers,
        num_sample_per_mel_frame=160,
        num_mel_frame_per_target_frame=8,
        dtype=torch.float32,
    )

    # Zero-pad missing columns so the speaker axis is always num_speakers wide.
    expected = torch.zeros(num_frames, num_speakers)
    expected[:, :n_spk_in] = activity

    assert targets.shape == (1, num_frames, num_speakers)
    assert torch.equal(targets[0], expected)
    assert target_length.tolist() == [get_hidden_length_from_sample_length(2560, 160, 8)]


@pytest.mark.unit
def test_collate_speaker_activity_targets_rejects_speaker_truncation():
    with pytest.raises(ValueError, match="8 columns.*num_speakers=4"):
        sot_alignment.collate_speaker_activity_targets(
            [torch.ones(2, 8)],
            audio_lens=torch.tensor([2560]),
            num_speakers=4,
            num_sample_per_mel_frame=160,
            num_mel_frame_per_target_frame=8,
            dtype=torch.float32,
        )


@pytest.mark.unit
def test_collate_speaker_activity_targets_mixed_speaker_counts_and_lengths():
    # Batch mixing different speaker counts AND time lengths must not crash inside
    # collate_matrices: the speaker axis is normalized first, then the time axis is
    # zero-padded to the batch max.
    four_spk = torch.ones(2, 4)
    two_spk = torch.full((3, 2), 2.0)  # (T=3, N=2) -> padded to (3, 4)

    targets, target_length = sot_alignment.collate_speaker_activity_targets(
        [four_spk, two_spk],
        audio_lens=torch.tensor([2560, 3840]),
        num_speakers=4,
        num_sample_per_mel_frame=160,
        num_mel_frame_per_target_frame=8,
        dtype=torch.float16,
    )

    assert targets.shape == (2, 3, 4)  # B=2, T_max=3, num_speakers=4
    assert targets.dtype == torch.float16
    # Four-speaker example is unchanged, and its 3rd time-step is zero-padded.
    assert torch.equal(targets[0, :2], torch.ones(2, 4, dtype=torch.float16))
    assert torch.equal(targets[0, 2], torch.zeros(4, dtype=torch.float16))
    # Padded example: cols 2-3 are zero.
    assert torch.equal(targets[1, :, 2:], torch.zeros(3, 2, dtype=torch.float16))
    assert torch.equal(targets[1, :, :2], torch.full((3, 2), 2.0, dtype=torch.float16))
    assert target_length.tolist() == [
        get_hidden_length_from_sample_length(2560, 160, 8),
        get_hidden_length_from_sample_length(3840, 160, 8),
    ]


@pytest.mark.unit
def test_collate_speaker_activity_targets_uses_generated_frame_lengths():
    activities = [torch.ones(3, 2), torch.ones(5, 2)]

    _, target_length = sot_alignment.collate_speaker_activity_targets(
        activities,
        # These loaded-audio-derived lengths map to 4 and 3 target frames,
        # intentionally disagreeing with the generated target tensors.
        audio_lens=torch.tensor([5120, 3840]),
        num_speakers=2,
        num_sample_per_mel_frame=160,
        num_mel_frame_per_target_frame=8,
        dtype=torch.float32,
    )

    assert target_length.tolist() == [3, 5]
