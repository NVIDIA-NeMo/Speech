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


from typing import Callable

from nemo.collections.asr.inference.utils.constants import POST_WORD_PUNCTUATION, PRE_WORD_PUNCTUATION
from nemo.collections.asr.inference.utils.text_segment import TextSegment, Word


def merge_timesteps(timesteps1: list, timesteps2: list) -> list:
    """
    Merge two lists of timesteps by preserving the order and ensuring that the timesteps are in increasing order
    Args:
        timesteps1: (list) The first list of timesteps
        timesteps2: (list) The second list of timesteps
    Returns:
        (list) The merged list of timesteps
    """
    # If both lists are empty, return an empty list
    if not timesteps1 and not timesteps2:
        return []

    # If timesteps1 is not empty and the first timestep is negative,
    # shift all the timesteps by the absolute value of the first timestep
    if timesteps1:
        if (first := timesteps1[0]) < 0:  # Assigns and checks in the same line
            for i, t in enumerate(timesteps1):
                timesteps1[i] = t - first

    # If timesteps2 is not empty and the first timestep is negative,
    # shift all the timesteps by the absolute value of the first timestep
    if timesteps2:
        if (first := timesteps2[0]) < 0:
            for i, t in enumerate(timesteps2):
                timesteps2[i] = t - first

    # If the first list is empty, return the second list
    if not timesteps1:
        return timesteps2

    # If the second list is empty, return the first list
    if not timesteps2:
        return timesteps1

    # If the last timestep of the first list is greater than the first timestep of the second list,
    # calculate the gap between the two timesteps and shift all the timesteps of the second list by the gap
    if (gap := timesteps2[0] - timesteps1[-1]) <= 0:
        return timesteps1 + [t + abs(gap) + 1 for t in timesteps2]
    return timesteps1 + timesteps2


def merge_segment_tail(
    segment_head: TextSegment,
    segment_tail: TextSegment,
    conf_aggregator: Callable = None,
) -> TextSegment:
    """
    Merge the segment_tail into the segment_head
    Args:
        segment_head: (TextSegment) The head segment
        segment_tail: (TextSegment) The tail segment
        conf_aggregator: (Callable) The function to aggregate the confidence
    Returns:
        (TextSegment) The merged segment
    """
    head = segment_head.copy()

    # for models that have built-in punctuation, we need to rm the last punctuation before merging
    if head.text and (last_char := head.text[-1]) and last_char in POST_WORD_PUNCTUATION:
        head.text = head.text.rstrip(last_char)

    # merge the segment_tail text
    head.text += segment_tail.text

    # update the end timestep
    head.end = segment_tail.end

    # update the confidence
    if conf_aggregator is not None:
        head.conf = conf_aggregator([head.conf, segment_tail.conf])

    return head


def merge_word_tail(
    word_head: Word, word_tail: Word, pnc_word_head: Word = None, conf_aggregator: Callable = None
) -> tuple[Word, Word]:
    """
    Merge the word_tail into the word_head
    Args:
        word_head: (Word) The head word
        word_tail: (Word) The tail word
        pnc_word_head: (Word) The head word with punctuation/capitalization
        conf_aggregator: (Callable) The function to aggregate the confidence
    Returns:
        (tuple[Word, Word]) The merged word and the head word with punctuation/capitalization
    """

    head = word_head.copy()
    head_text = head.text

    # for models that have built-in punctuation, we need to rm the last punctuation before merging
    if head_text and (last_char := head_text[-1]) and last_char in POST_WORD_PUNCTUATION:
        head.text = head_text.rstrip(last_char)

    # merge the word_tail text
    head.text += word_tail.text

    # update the end timestep
    head.end = word_tail.end

    # update the confidence
    if conf_aggregator is not None:
        head.conf = conf_aggregator([head.conf, word_tail.conf])

    pnc_head = None
    if pnc_word_head is not None:

        last_char = pnc_word_head.text[-1] if pnc_word_head.text else None
        first_char = pnc_word_head.text[0] if pnc_word_head.text else None

        pnc_head = head.copy()

        if last_char in POST_WORD_PUNCTUATION:
            if pnc_head.text and pnc_head.text[-1] not in POST_WORD_PUNCTUATION:
                pnc_head.text = pnc_head.text + last_char

        if first_char in PRE_WORD_PUNCTUATION:
            if pnc_head.text and pnc_head.text[0] not in PRE_WORD_PUNCTUATION:
                pnc_head.text = first_char + pnc_head.text

        if first_char and first_char.isupper():
            pnc_head.capitalize()

    return head, pnc_head


def find_max_overlap(state_tokens: list, new_tokens: list, limit: int) -> int:
    """
    Finds the maximum overlap between the state_tokens suffix and the new_tokens prefix
    Args:
        state_tokens: (list) The list of state tokens
        new_tokens: (list) The list of new tokens
        limit: (int) The limit on the overlap
    Returns:
        (int) The maximum overlap within the limit
    """
    max_overlap = 0
    for k in range(1, min(len(state_tokens), len(new_tokens), limit) + 1):
        if state_tokens[-k:] == new_tokens[:k]:
            max_overlap = k
    return max_overlap


def detect_overlap(
    state_tokens: list[int],
    state_timesteps: list[float],
    new_tokens: list[int],
    new_timesteps: list[float],
    overlap_search_th: int = 3,
    close_in_time_th: float = 2.0,
) -> int:
    """
    Detect the overlap between state_tokens and new_tokens
    Args:
        state_tokens: (list[int]) The list of state tokens
        state_timesteps: (list[float]) The list of state timesteps
        new_tokens: (list[int]) The list of new tokens
        new_timesteps: (list[float]) The list of new timesteps
        overlap_search_th: (int) The threshold on the overlap
        close_in_time_th: (float) The threshold on the close in time
    Returns:
        (int) The overlap between the state_tokens and the new_tokens
    """
    overlap = 0
    if state_tokens:
        overlap = find_max_overlap(state_tokens, new_tokens, overlap_search_th)
        if overlap > 0:
            close_in_time = (new_timesteps[overlap - 1] - state_timesteps[-overlap]) <= close_in_time_th
            overlap = overlap if close_in_time else 0
    return overlap
