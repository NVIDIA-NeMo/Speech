# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo.collections.tts.g2p.utils import get_heteronym_spans


class TestG2pDataUtils:
    @staticmethod
    def _create_expected_output(words):
        return [([word], False) for word in words]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_get_heteronym_spans(self):
        supported_heteronyms = ["live", "read", "protest", "diffuse", "desert"]
        sentences = [
            "I live in California. I READ a book. Only people who have already gained something are willing to protest."
            " He reads a book!",
            "Yesterday, I read a book.",
            "He read a book last night and pre-diffuse and LivE-post and pre-desert-post.",
            "the soldier deserted the desert in desert.",
        ]

        expected_start_end = [
            [(2, 6), (24, 28), (98, 105)],
            [(13, 17)],
            [(3, 7), (34, 41), (46, 50), (64, 70)],
            [(25, 31), (35, 41)],
        ]
        expected_heteronyms = [
            ["live", "read", "protest"],
            ['read'],
            ['read', 'diffuse', 'live', 'desert'],
            ['desert', 'desert'],
        ]

        out_start_end, out_heteronyms = get_heteronym_spans(sentences, supported_heteronyms)
        assert out_start_end == expected_start_end, "start-end spans do not match"
        assert out_heteronyms == expected_heteronyms, "heteronym spans do not match"
