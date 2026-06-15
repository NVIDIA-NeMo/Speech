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

from nemo.collections.asr.parts.utils.sot_speaker_alignment import (
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
