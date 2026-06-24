# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_hidden_length_from_sample_length
from nemo.collections.asr.parts.utils.sot_speaker_alignment import (
    collate_speaker_activity_targets,
    ensure_single_speaker_sot,
    fix_speaker_activity,
    parse_speaker_tokens,
    sl_to_wl_sot,
)


@pytest.mark.unit
def test_parse_speaker_tokens_handles_multi_digit_speakers():
    assert parse_speaker_tokens("<spk:10> hello <spk:1> world") == [10, 1]


@pytest.mark.unit
def test_sl_and_wl_sot_have_same_speaker_sequence():
    sl_text = "<spk:0> hello world <spk:1> yes"
    wl_text = sl_to_wl_sot(sl_text)

    assert wl_text == "<spk:0> hello <spk:0> world <spk:1> yes"
    assert parse_speaker_tokens(sl_text) == parse_speaker_tokens(wl_text)


@pytest.mark.unit
def test_ensure_single_speaker_sot_prefixes_no_token_text():
    text, spk_idx, changed = ensure_single_speaker_sot("hello world")

    assert text == "<spk:0> hello world"
    assert spk_idx == 0
    assert changed


@pytest.mark.unit
def test_ensure_single_speaker_sot_keeps_existing_tokens():
    text, spk_idx, changed = ensure_single_speaker_sot("<spk:2> hello")

    assert text == "<spk:2> hello"
    assert spk_idx == -1
    assert not changed


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

    fixed = fix_speaker_activity("<spk:0> hello world <spk:1> yes now", activity, num_speakers=2)

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

    fixed = fix_speaker_activity("", activity, num_speakers=2)

    assert fixed is activity


@pytest.mark.unit
@pytest.mark.parametrize(
    "n_spk_in, num_speakers",
    [
        (5, 4),  # speaker dim > num_speakers -> truncate extra columns
        (4, 4),  # speaker dim == num_speakers -> unchanged
        (2, 4),  # speaker dim < num_speakers -> zero-pad missing columns
    ],
)
def test_collate_speaker_activity_targets_normalizes_speaker_axis(n_spk_in, num_speakers):
    num_frames = 2
    # Distinct per-element values so truncation/padding is observable.
    activity = torch.arange(1, num_frames * n_spk_in + 1, dtype=torch.float32).reshape(num_frames, n_spk_in)

    targets, target_length = collate_speaker_activity_targets(
        [activity],
        audio_lens=torch.tensor([2560]),
        num_speakers=num_speakers,
        num_sample_per_mel_frame=160,
        num_mel_frame_per_target_frame=8,
        dtype=torch.float32,
    )

    # Truncate extras / zero-pad missing so the speaker axis is always num_speakers wide.
    expected = torch.zeros(num_frames, num_speakers)
    keep = min(n_spk_in, num_speakers)
    expected[:, :keep] = activity[:, :keep]

    assert targets.shape == (1, num_frames, num_speakers)
    assert torch.equal(targets[0], expected)
    assert target_length.tolist() == [get_hidden_length_from_sample_length(2560, 160, 8)]


@pytest.mark.unit
def test_collate_speaker_activity_targets_mixed_speaker_counts_and_lengths():
    # Batch mixing different speaker counts AND time lengths must not crash inside
    # collate_matrices: the speaker axis is normalized first, then the time axis is
    # zero-padded to the batch max.
    five_spk = torch.ones(2, 5)  # (T=2, N=5) -> truncated to (2, 4)
    two_spk = torch.full((3, 2), 2.0)  # (T=3, N=2) -> padded to (3, 4)

    targets, target_length = collate_speaker_activity_targets(
        [five_spk, two_spk],
        audio_lens=torch.tensor([2560, 3840]),
        num_speakers=4,
        num_sample_per_mel_frame=160,
        num_mel_frame_per_target_frame=8,
        dtype=torch.float16,
    )

    assert targets.shape == (2, 3, 4)  # B=2, T_max=3, num_speakers=4
    assert targets.dtype == torch.float16
    # Truncated example: only first 4 cols kept, and its 3rd time-step is zero-padded.
    assert torch.equal(targets[0, :2], torch.ones(2, 4, dtype=torch.float16))
    assert torch.equal(targets[0, 2], torch.zeros(4, dtype=torch.float16))
    # Padded example: cols 2-3 are zero.
    assert torch.equal(targets[1, :, 2:], torch.zeros(3, 2, dtype=torch.float16))
    assert torch.equal(targets[1, :, :2], torch.full((3, 2), 2.0, dtype=torch.float16))
    assert target_length.tolist() == [
        get_hidden_length_from_sample_length(2560, 160, 8),
        get_hidden_length_from_sample_length(3840, 160, 8),
    ]
