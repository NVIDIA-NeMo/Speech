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

from nemo.collections.asr.inference.utils.lcs_merge import MergingStrategy, lcs_merge, longest_common_substring


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
            ),
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
        assert (start1, start2, length) == (expected_start1, expected_start2, expected_length)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "buffer, data, search_size, min_lcs_length, merging_strategy, expected_result",
        [
            ([1, 2, 3, 4, 5], [3, 4, 5, 6, 7], 5, 1, MergingStrategy.LCSUBSTR, [1, 2, 3, 4, 5, 6, 7]),
            ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], 5, 1, MergingStrategy.LCSUBSTR, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            ([1, 2, 3, 4, 5], [3, 4, 5, 6, 7, 8, 9], 5, 1, MergingStrategy.LCSUBSTR, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ([1, 2, 3, 4, 9], [3, 4, 5, 6, 7], 5, 1, MergingStrategy.LCS, [1, 2, 3, 4, 5, 6, 7]),
            ([1, 2, 3, 4, 5], [3, 4, 5, 6, 7], 1, 2, MergingStrategy.LCSUBSTR, [1, 2, 3, 4, 5, 3, 4, 5, 6, 7]),
        ],
    )
    def test_lcs_merge(self, buffer, data, search_size, min_lcs_length, merging_strategy, expected_result):
        result = lcs_merge(
            buffer,
            data,
            search_size=search_size,
            min_lcs_length=min_lcs_length,
            merging_strategy=merging_strategy,
            sep_id=None,
        )
        assert result == expected_result

    @pytest.mark.unit
    def test_lcs_merge_empty_buffer(self):
        """Test that empty buffer returns just the data."""
        result = lcs_merge([], [1, 2, 3], search_size=5, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR)
        assert result == [1, 2, 3]

    @pytest.mark.unit
    def test_lcs_merge_empty_data(self):
        """Test that empty data returns the buffer unchanged."""
        result = lcs_merge([1, 2, 3], [], search_size=5, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR)
        assert result == [1, 2, 3]

    @pytest.mark.unit
    def test_lcs_merge_both_empty(self):
        """Test merging two empty lists."""
        result = lcs_merge([], [], search_size=5, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR)
        assert result == []

    @pytest.mark.unit
    @pytest.mark.parametrize("search_size", [0, -1, -10])
    def test_lcs_merge_invalid_search_size(self, search_size):
        """Test that search_size <= 0 results in simple concatenation with separator."""
        buffer = [1, 2, 3]
        data = [4, 5, 6]
        result = lcs_merge(
            buffer, data, search_size=search_size, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR
        )
        assert result == [1, 2, 3, 4, 5, 6]

    @pytest.mark.unit
    def test_lcs_merge_with_sep_id_no_overlap(self):
        """Test separator is inserted when no LCS is found."""
        buffer = [1, 2, 3]
        data = [7, 8, 9]
        sep_id = [100]
        result = lcs_merge(
            buffer, data, search_size=3, sep_id=sep_id, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR
        )
        assert result == [1, 2, 3, 100, 7, 8, 9]

    @pytest.mark.unit
    def test_lcs_merge_with_multi_token_sep_id(self):
        """Test separator with multiple tokens is inserted correctly."""
        buffer = [1, 2, 3]
        data = [7, 8, 9]
        sep_id = [100, 101, 102]
        result = lcs_merge(
            buffer, data, search_size=3, sep_id=sep_id, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR
        )
        assert result == [1, 2, 3, 100, 101, 102, 7, 8, 9]

    @pytest.mark.unit
    def test_lcs_merge_with_sep_id_when_overlap_exists(self):
        """Test separator is NOT inserted when LCS is found."""
        buffer = [1, 2, 3, 4, 5]
        data = [4, 5, 6, 7]
        sep_id = [100]
        result = lcs_merge(
            buffer, data, search_size=5, sep_id=sep_id, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR
        )
        assert result == [1, 2, 3, 4, 5, 6, 7]
        assert 100 not in result

    @pytest.mark.unit
    def test_lcs_merge_search_size_larger_than_buffer(self):
        """Test that search_size larger than buffer still works correctly."""
        buffer = [1, 2, 3]
        data = [2, 3, 4, 5]
        result = lcs_merge(buffer, data, search_size=100, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR)
        assert result == [1, 2, 3, 4, 5]

    @pytest.mark.unit
    def test_lcs_merge_search_size_limits_overlap_detection(self):
        """Test that search_size limits the overlap detection window."""
        buffer = [1, 2, 3, 4, 5, 6, 7, 8]
        data = [2, 3, 9, 10]  # overlap [2,3] is outside search window of last 3 elements
        result = lcs_merge(buffer, data, search_size=3, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR)
        # No overlap in last 3 elements [6,7,8], so data is appended
        assert result == [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 9, 10]

    @pytest.mark.unit
    def test_lcs_merge_min_lcs_length_threshold(self):
        """Test that LCS shorter than min_lcs_length causes concatenation."""
        buffer = [1, 2, 3, 4, 5]
        data = [5, 6, 7]  # only 1 element overlap
        result = lcs_merge(buffer, data, search_size=5, min_lcs_length=2, merging_strategy=MergingStrategy.LCSUBSTR)
        # LCS length is 1, which is < min_lcs_length=2, so concatenate
        assert result == [1, 2, 3, 4, 5, 5, 6, 7]

    @pytest.mark.unit
    def test_lcs_merge_min_lcs_length_exact_match(self):
        """Test that LCS equal to min_lcs_length triggers merge."""
        buffer = [1, 2, 3, 4, 5]
        data = [4, 5, 6, 7]  # 2 element overlap
        result = lcs_merge(buffer, data, search_size=5, min_lcs_length=2, merging_strategy=MergingStrategy.LCSUBSTR)
        assert result == [1, 2, 3, 4, 5, 6, 7]

    @pytest.mark.unit
    def test_lcs_merge_data_is_subset_of_buffer_end(self):
        """Test when data is entirely contained in the end of buffer."""
        buffer = [1, 2, 3, 4, 5]
        data = [4, 5]
        result = lcs_merge(buffer, data, search_size=5, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR)
        assert result == [1, 2, 3, 4, 5]

    @pytest.mark.unit
    def test_lcs_merge_complete_overlap(self):
        """Test when data starts exactly where buffer search window begins."""
        buffer = [1, 2, 3, 4, 5]
        data = [3, 4, 5, 6, 7, 8]
        result = lcs_merge(buffer, data, search_size=3, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR)
        assert result == [1, 2, 3, 4, 5, 6, 7, 8]

    @pytest.mark.unit
    def test_lcs_merge_lcs_strategy_with_gaps(self):
        """Test LCS strategy handles non-contiguous common subsequences."""
        buffer = [1, 2, 100, 3, 4]  # has 100 inserted
        data = [2, 3, 4, 5, 6]  # continuous 2, 3, 4
        result = lcs_merge(buffer, data, search_size=5, min_lcs_length=1, merging_strategy=MergingStrategy.LCS)
        # LCS finds [2, 3, 4] even with gap
        assert result == [1, 2, 100, 3, 4, 5, 6]

    @pytest.mark.unit
    def test_lcs_merge_single_element_overlap(self):
        """Test merging with single element overlap."""
        buffer = [1, 2, 3]
        data = [3, 4, 5]
        result = lcs_merge(buffer, data, search_size=3, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR)
        assert result == [1, 2, 3, 4, 5]

    @pytest.mark.unit
    def test_lcs_merge_repeated_elements(self):
        """Test merging lists with repeated elements."""
        buffer = [1, 1, 1, 2, 2, 2]
        data = [2, 2, 2, 3, 3, 3]
        result = lcs_merge(buffer, data, search_size=6, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR)
        assert result == [1, 1, 1, 2, 2, 2, 3, 3, 3]

    @pytest.mark.unit
    def test_lcs_merge_long_sequences(self):
        """Test merging longer sequences for performance sanity."""
        buffer = list(range(100))
        data = list(range(90, 150))  # overlap from 90-99
        result = lcs_merge(buffer, data, search_size=20, min_lcs_length=1, merging_strategy=MergingStrategy.LCSUBSTR)
        assert result == list(range(150))


class TestLongestCommonSubstringEdgeCases:
    """Additional edge case tests for longest_common_substring function."""

    @pytest.mark.unit
    def test_single_element_match(self):
        """Test with single element lists that match."""
        start1, start2, length = longest_common_substring([5], [5])
        assert (start1, start2, length) == (0, 0, 1)

    @pytest.mark.unit
    def test_single_element_no_match(self):
        """Test with single element lists that don't match."""
        start1, start2, length = longest_common_substring([5], [6])
        assert (start1, start2, length) == (-1, -1, 0)

    @pytest.mark.unit
    def test_match_at_buffer_start(self):
        """Test when LCS is at the start of buffer."""
        start1, start2, length = longest_common_substring([1, 2, 3, 9, 9], [1, 2, 3, 8, 8])
        assert (start1, start2, length) == (0, 0, 3)

    @pytest.mark.unit
    def test_match_at_data_end(self):
        """Test when LCS is at the end of data."""
        start1, start2, length = longest_common_substring([7, 8, 9], [1, 2, 7, 8, 9])
        assert (start1, start2, length) == (0, 2, 3)

    @pytest.mark.unit
    def test_entire_data_is_substring(self):
        """Test when entire data is a substring of buffer."""
        start1, start2, length = longest_common_substring([1, 2, 3, 4, 5, 6], [3, 4, 5])
        assert (start1, start2, length) == (2, 0, 3)

    @pytest.mark.unit
    def test_entire_buffer_is_substring(self):
        """Test when entire buffer is a substring of data."""
        start1, start2, length = longest_common_substring([3, 4, 5], [1, 2, 3, 4, 5, 6])
        assert (start1, start2, length) == (0, 2, 3)
