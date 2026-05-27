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

from nemo.collections.asr.inference.utils.enums import MergingStrategy
from nemo.collections.asr.parts.utils.streaming_utils import longest_common_subsequence_merge


def longest_common_substring(buffer: list[int], data: list[int]) -> tuple[int, int, int]:
    """
    Find the longest common substring between two lists of integers.
    If there are multiple LCSs, return the rightmost one in the buffer.
    Args:
        buffer: (list[int]) The buffer of tokens.
        data: (list[int]) The new tokens to merge with the buffer.
    Returns:
        A tuple containing - (tuple[int, int, int]):
          - Start index of the longest common substring in the buffer.
          - Start index of the longest common substring in the data.
          - Length of the longest common substring.
    """
    n, m = len(buffer), len(data)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    max_len = 0
    end_i = end_j = -1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if buffer[i - 1] == data[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

                # Logic:
                # 1. If we find a strictly longer substring, take it.
                # 2. If it's the same length, update if this occurrence
                #    ends further right in the buffer (larger i).
                if dp[i][j] > max_len or (dp[i][j] == max_len and i >= end_i):
                    max_len = dp[i][j]
                    end_i = i
                    end_j = j
            else:
                dp[i][j] = 0

    if max_len == 0:
        return -1, -1, 0

    return end_i - max_len, end_j - max_len, max_len


def lcs_merge(
    buffer: list[int],
    data: list[int],
    search_size: int,
    sep_id: list[int] | None = None,
    min_lcs_length: int = 1,
    merging_strategy: MergingStrategy = MergingStrategy.LCSUBSTR,
) -> list[int]:
    """
    Merge the buffer and data using the LCS algorithm.
    Args:
        buffer: (list[int]) The buffer of tokens.
        data: (list[int]) The new tokens to merge with the buffer.
        search_size: (int) The size of the search window in the buffer.
        sep_id: (list[int] | None) The separator token ids. If no LCS is found, separator token is used to merge the buffer and data.
        min_lcs_length: (int) The minimum length of the LCS.
        merging_strategy: (MergingStrategy) The merging strategy to use.
    Returns:
        (list[int]) The merged tokens.
    """

    if len(buffer) == 0:
        buffer += data
        return buffer

    if sep_id is None:
        sep_id = []

    if search_size < 1:
        buffer += sep_id + data
        return buffer

    buffer_slice = buffer[-search_size:]

    if merging_strategy == MergingStrategy.LCSUBSTR:
        i_rel, j_rel, length = longest_common_substring(buffer_slice, data)
    elif merging_strategy == MergingStrategy.LCS:
        (i_rel, j_rel, length), _ = longest_common_subsequence_merge(buffer_slice, data)
    else:
        raise ValueError(
            f"Invalid merging strategy: {merging_strategy!r}. Supported strategies: {[s.name for s in MergingStrategy]}"
        )

    if length < min_lcs_length:
        buffer += sep_id + data
        return buffer

    base = len(buffer) - len(buffer_slice)
    i_abs_start = base + i_rel
    i_abs_end = i_abs_start + length  # end position (exclusive) in `buffer`
    j_after = j_rel + length  # first index after LCS in `data`

    merged = buffer[:i_abs_end] + data[j_after:]
    return merged
