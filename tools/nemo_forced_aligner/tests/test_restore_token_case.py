# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.parts.utils.aligner_utils import restore_token_case


@pytest.mark.parametrize(
    "word,word_tokens,expected_word_tokens_cased",
    [
        ("HEY!", ['▁he', 'y', '!'], ['▁HE', 'Y', '!']),
        ("BabABa▁", ['▁b', 'a', 'b', 'a', 'b', 'a'], ['▁B', 'a', 'b', 'A', 'B', 'a']),
        ("BabAB▁a", ['▁b', 'a', 'b', 'a', 'b', '_a'], ['▁B', 'a', 'b', 'A', 'B', '_a']),
        ("Bab▁AB▁a", ['▁b', 'a', 'b', '▁a', 'b', '▁a'], ['▁B', 'a', 'b', '▁A', 'B', '▁a']),
        ("▁Bab▁AB▁a", ['▁b', 'a', 'b', '▁a', 'b', '▁a'], ['▁B', 'a', 'b', '▁A', 'B', '▁a']),
        ("▁Bab▁AB▁▁a", ['▁b', 'a', 'b', '▁a', 'b', '▁a'], ['▁B', 'a', 'b', '▁A', 'B', '▁a']),
        ("▁▁BabAB▁a", ['▁b', 'a', 'b', 'a', 'b', '▁a'], ['▁B', 'a', 'b', 'A', 'B', '▁a']),
        ("m²", ['▁', 'm', '2'], ['▁', 'm', '2']),
        ("²", ['▁', '2'], ['▁', '2']),
    ],
)
def test_restore_token_case(word, word_tokens, expected_word_tokens_cased):
    word_tokens_cased = restore_token_case(word, word_tokens)
    assert word_tokens_cased == expected_word_tokens_cased
