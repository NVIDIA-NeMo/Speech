# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.inference.utils.lcs_merge import longest_common_substring


class TestLCSMerge:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "buffer, data, expected_start1, expected_start2, expected_length",
        [
            ([1, 2, 3, 4, 5], [3, 4, 5, 6, 7], 2, 0, 3),
            ([1, 2], [1], 0, 0, 1),
            ([1], [1, 2], 0, 0, 1),
            (
                [1, 2, 3, 11, 12, 13, 4, 5, 6],
                [1, 2, 3, 4, 5, 6, 11, 12, 13],
                6,
                3,
                3,
            ),  # need to return rightmost lcs in the first list
            ([1, 2, 3, 11, 12, 13, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 11, 12, 13], 6, 3, 4),
            ([1, 2, 3], [4, 5, 6], -1, -1, 0),
            ([1, 2, 3, 1, 2, 3], [1, 2, 3], 3, 0, 3),
            ([], [], -1, -1, 0),
            ([1, 2, 3], [], -1, -1, 0),
            ([1, 1, 1, 1, 1], [1, 1], 3, 0, 2),
        ],
    )
    def test_longest_common_substring(self, buffer, data, expected_start1, expected_start2, expected_length):
        start1, start2, length = longest_common_substring(buffer, data)
        assert start1 == expected_start1
        assert start2 == expected_start2
        assert length == expected_length
