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


import re
from collections import OrderedDict

from nemo.collections.asr.inference.utils.constants import DEFAULT_SEMIOTIC_CLASS


# Compile regex pattern once at module level for better performance
TOKEN_PATTERN = re.compile(r'tokens \{.*?(?=tokens \{|$)', re.DOTALL)


def get_semiotic_class(tokens: list[OrderedDict]) -> str:
    """
    Returns the semiotic class of the given tokens.
    """
    return list(tokens[0]["tokens"].keys())[0]


def split_text(text: str, sep: str = " ") -> tuple[list, int]:
    """
    Splits the text into words based on the separator.
    Args:
        text: (str) input text
        sep: (str) separator to split the text
    Returns:
        words: (list) list of words
        n_words: (int) number of words
    """
    words = [w for w in text.split(sep) if w]
    return words, len(words)


def find_tokens(text: str) -> list[str]:
    """
    Find the start and end positions of token blocks in the given text.
    Args:
        text: (str) input text containing token blocks
    Returns:
        token_blocks: (list[str]) list of token blocks
    """

    # Use compiled regex to find all token blocks in a single pass
    token_blocks = TOKEN_PATTERN.findall(text)

    # Strip whitespace from each block
    return [block.strip() for block in token_blocks]


def get_trivial_alignment(N: int, i_shift: int = 0, o_shift: int = 0) -> list[tuple]:
    """
    Returns a trivial word alignment for N input words.
    Args:
        N: (int) number of input words
        i_shift: (int) input shift
        o_shift: (int) output shift
    Returns:
        (list) Returns a trivial word alignment
    """
    return [([i + i_shift], [i + o_shift], DEFAULT_SEMIOTIC_CLASS) for i in range(N)]


def fallback_to_trivial_alignment(
    input_words: list[str], i_shift: int = 0, o_shift: int = 0
) -> tuple[list[str], list[str], list[tuple]]:
    """
    Returns a trivial word alignment for the input words.
    Args:
        input_words: (list[str]) list of input words
        i_shift: (int) input shift
        o_shift: (int) output shift
    Returns:
        (tuple) Returns a tuple of input words, output words, and a trivial word alignment
    """
    return input_words, input_words.copy(), get_trivial_alignment(N=len(input_words), i_shift=i_shift, o_shift=o_shift)
