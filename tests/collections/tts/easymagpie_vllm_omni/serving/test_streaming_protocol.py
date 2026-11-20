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
from __future__ import annotations

import pytest
from easymagpie_vllm_omni.easymagpie import _merge_streaming_text_chunk


def test_absolute_streaming_chunks_append_once_across_lookahead_repeats():
    tokens, appended = _merge_streaming_text_chunk([], [10], 0)
    assert tokens == [10]
    assert appended

    tokens, appended = _merge_streaming_text_chunk(tokens, [11, 12], 1)
    assert tokens == [10, 11, 12]
    assert appended

    # The async runner may expose this same segment again on later decode steps.
    tokens, appended = _merge_streaming_text_chunk(tokens, [11, 12], 1)
    assert tokens == [10, 11, 12]
    assert not appended

    # A partially overlapping repeat is also safe and appends only the new tail.
    tokens, appended = _merge_streaming_text_chunk(tokens, [12, 13], 2)
    assert tokens == [10, 11, 12, 13]
    assert appended


@pytest.mark.parametrize(
    ("incoming", "start", "message"),
    [
        ([11], None, "require text_token_start"),
        ([11], 2, "Invalid text_token_start"),
        ([99], 0, "Conflicting streaming text chunk"),
    ],
)
def test_absolute_streaming_chunks_reject_missing_or_inconsistent_positions(incoming, start, message):
    with pytest.raises(ValueError, match=message):
        _merge_streaming_text_chunk([10], incoming, start)
