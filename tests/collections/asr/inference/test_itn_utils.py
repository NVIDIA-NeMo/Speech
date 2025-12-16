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
from nemo.collections.asr.inference.utils.itn_utils import (
    fallback_to_trivial_alignment,
    find_tokens,
    get_semiotic_class,
    get_trivial_alignment,
    split_text,
)


class TestItnUtils:

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text, expected_words, expected_n",
        [
            ("hello world    how are you", ["hello", "world", "how", "are", "you"], 5),
            ("hello", ["hello"], 1),
            ("a hello world    b ccc d   e", ["a", "hello", "world", "b", "ccc", "d", "e"], 7),
            (" a hello world    b ccc d   e", ["a", "hello", "world", "b", "ccc", "d", "e"], 7),
            ("a hello world    b ccc d   e ", ["a", "hello", "world", "b", "ccc", "d", "e"], 7),
            (" a hello world    b ccc d   e ", ["a", "hello", "world", "b", "ccc", "d", "e"], 7),
            ("  a hello world    b ccc d   e  ", ["a", "hello", "world", "b", "ccc", "d", "e"], 7),
        ],
    )
    def test_split_text(self, text, expected_words, expected_n):
        words, n = split_text(text)
        assert words == expected_words
        assert n == expected_n

    @pytest.mark.unit
    def test_get_semiotic_class(self):
        tokens = [{"tokens": {"name": "hello"}}]
        semiotic_class = get_semiotic_class(tokens)
        assert semiotic_class == "name"

    @pytest.mark.unit
    def test_find_tokens(self):
        text = "tokens {name: hello}   tokens {name: world} tokens {name: how} tokens {name: are} tokens {name: you}"
        tokens = find_tokens(text)
        assert tokens == [
            "tokens {name: hello}",
            "tokens {name: world}",
            "tokens {name: how}",
            "tokens {name: are}",
            "tokens {name: you}",
        ]

    @pytest.mark.unit
    def test_get_trivial_alignment(self):
        N = 5
        i_shift = 1
        o_shift = 2
        alignment = get_trivial_alignment(N, i_shift, o_shift)
        assert alignment == [
            ([1], [2], "name"),
            ([2], [3], "name"),
            ([3], [4], "name"),
            ([4], [5], "name"),
            ([5], [6], "name"),
        ]

    @pytest.mark.unit
    def test_fallback_to_trivial_alignment(self):
        input_words = ["hello", "world", "how", "are", "you"]
        input_words, output_words, word_alignment = fallback_to_trivial_alignment(input_words)
        assert input_words == ["hello", "world", "how", "are", "you"]
        assert output_words == ["hello", "world", "how", "are", "you"]
        assert word_alignment == [
            ([0], [0], "name"),
            ([1], [1], "name"),
            ([2], [2], "name"),
            ([3], [3], "name"),
            ([4], [4], "name"),
        ]
