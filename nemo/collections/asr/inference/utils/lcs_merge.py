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


def rightmost_lcs(buffer: list[int], data: list[int]) -> tuple[int, int, int]:
    """
    Find the Rightmost Longest Common Subsequence (LCS) between buffer and data arrays.
    Args:
        buffer: (list[int]) The buffer of tokens.
        data: (list[int]) The new tokens to merge with the buffer.
    Returns:
        tuple[int, int, int]: A tuple containing:
        (int) The start index of the LCS in the buffer.
        (int) The start index of the LCS in the data.
        (int) The length of the LCS.
    """
    n, m = len(buffer), len(data)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    best_len = 0
    best_end_i = -1
    best_end_j = -1

    for i in range(1, n + 1):
        bi = buffer[i - 1]
        for j in range(1, m + 1):
            if bi == data[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

                cur_len = dp[i][j]
                end_i = i - 1
                end_j = j - 1

                # Rightmost LCS selection
                if cur_len > best_len or (
                    cur_len == best_len and (end_i > best_end_i or (end_i == best_end_i and end_j > best_end_j))
                ):
                    best_len = cur_len
                    best_end_i = end_i
                    best_end_j = end_j
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    if best_len == 0:
        return -1, -1, 0

    start_i = best_end_i - best_len + 1
    start_j = best_end_j - best_len + 1

    return start_i, start_j, best_len


def lcs_merge(
    buffer: list[int], data: list[int], search_size: int, sep_id: list[int] | None = None, min_lcs_length: int = 1
) -> list[int]:
    """
    Merge the buffer and data using the LCS algorithm.
    Args:
        buffer: (list[int]) The buffer of tokens.
        data: (list[int]) The new tokens to merge with the buffer.
        search_size: (int) The size of the search window in the buffer.
        sep_id: (list[int] | None) The separator token ids. If no LCS is found, separator token is used to merge the buffer and data.
        min_lcs_length: (int) The minimum length of the LCS.
    Returns:
        (list[int]) The merged tokens.
    """

    if len(buffer) == 0:
        buffer += data
        return buffer

    if search_size < 1:
        buffer += data if sep_id is None else sep_id + data
        return buffer

    buffer_slice = buffer[-search_size:]

    i_rel, j_rel, length = rightmost_lcs(buffer_slice, data)

    if length < min_lcs_length:
        buffer += data if sep_id is None else sep_id + data
        return buffer

    base = len(buffer) - len(buffer_slice)
    i_abs_start = base + i_rel
    i_abs_end = i_abs_start + length  # end position (exclusive) in `buffer`
    j_after = j_rel + length  # first index after LCS in `data`

    merged = buffer[:i_abs_end] + data[j_after:]
    return merged
