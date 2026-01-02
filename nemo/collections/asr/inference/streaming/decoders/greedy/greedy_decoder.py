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

from nemo.collections.asr.inference.utils.constants import SENTENCEPIECE_UNDERSCORE


class GreedyDecoder:
    """Base class for the greedy decoder"""

    def __init__(self, vocabulary: list[str], conf_func: Callable = None):
        """
        Initialize the GreedyDecoder
        Args:
            vocabulary (list[str]): list of vocabulary tokens
            conf_func (Callable): function to compute confidence
        """

        self.vocabulary = vocabulary
        self.blank_id = len(vocabulary)
        self.conf_func = conf_func
        self.is_start_tokens = [token.startswith(SENTENCEPIECE_UNDERSCORE) for token in vocabulary]

    def count_silent_tokens(self, tokens: list[int], start: int, end: int) -> int:
        """
        Count how many silent tokens appear in [start, end).
        Args:
            tokens (list[int]): list of tokens
            start (int): start index
            end (int): end index
        Returns:
            int: number of silent tokens
        """
        if end <= start or start >= len(tokens):
            return 0
        return sum(self.is_token_silent(tokens[i]) for i in range(start, min(end, len(tokens))))

    def is_token_start_of_word(self, token_id: int) -> bool:
        """
        Check if the token is the start of a word
        Args:
            token_id (int): token id
        Returns:
            bool: True if the token is the start of a word, False otherwise
        """
        return self.is_start_tokens[token_id]

    def is_token_silent(self, token_id: int) -> bool:
        """
        Check if the token is silent
        Args:
            token_id (int): token id
        Returns:
            bool: True if the token is silent, False otherwise
        """
        return token_id == self.blank_id

    def first_non_silent_token(self, tokens: list[int], start: int, end: int) -> int:
        """
        Return the index of the first non-silent token in [start, end).
        If none found, return -1.
        Args:
            tokens (list[int]): list of tokens
            start (int): start index
            end (int): end index
        Returns:
            int: index of the first non-silent token
        """
        for i in range(start, min(end, len(tokens))):
            if not self.is_token_silent(tokens[i]):
                return i
        return -1

    def count_non_silent_tokens(self, tokens: list[int], start: int, end: int) -> int:
        """
        Count how many non-silent tokens appear in [start, end).
        Args:
            tokens (list[int]): list of tokens
            start (int): start index
            end (int): end index
        Returns:
            int: number of non-silent tokens
        """
        if end <= start or start >= len(tokens):
            return 0
        return sum(not self.is_token_silent(tokens[i]) for i in range(start, min(end, len(tokens))))

    def __call__(self, *args, **kwds):
        raise NotImplementedError("Subclass of GreedyDecoder should implement `__call__` method!")
