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
from nemo.collections.common.tokenizers.text_to_speech.tokenizer_utils import (
    any_locale_word_tokenize,
    english_word_tokenize,
    french_text_preprocessing,
)


class TestTokenizerUtils:
    @staticmethod
    def _create_expected_output(words):
        return [([word], False) for word in words]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_word_tokenize(self):
        input_text = "apple banana pear"
        expected_output = self._create_expected_output(["apple", " ", "banana", " ", "pear"])

        output = english_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_word_tokenize_with_punctuation(self):
        input_text = "Hello, world!"
        expected_output = self._create_expected_output(["hello", ", ", "world", "!"])

        output = english_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_word_tokenize_with_contractions(self):
        input_text = "It's a c'ntr'ction."
        expected_output = self._create_expected_output(["it's", " ", "a", " ", "c'ntr'ction", "."])

        output = english_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_word_tokenize_with_compound_words(self):
        input_text = "Forty-two is no run-off-the-mill number."
        expected_output = self._create_expected_output(
            ["forty-two", " ", "is", " ", "no", " ", "run-off-the-mill", " ", "number", "."]
        )

        output = english_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_word_tokenize_with_escaped(self):
        input_text = "Leave |this part UNCHANGED|."
        expected_output = [(["leave"], False), ([" "], False), (["this", "part", "UNCHANGED"], True), (["."], False)]

        output = english_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_any_locale_word_tokenize(self):
        input_text = "apple banana pear"
        expected_output = self._create_expected_output(["apple", " ", "banana", " ", "pear"])

        output = any_locale_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_any_locale_word_tokenize_with_accents(self):
        input_text = "The naïve piñata at the café..."
        expected_output = self._create_expected_output(
            ["The", " ", "naïve", " ", "piñata", " ", "at", " ", "the", " ", "café", "..."]
        )

        output = any_locale_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_any_locale_word_tokenize_with_numbers(self):
        input_text = r"Three times× four^teen ÷divided by [movies] on \slash."
        expected_output = self._create_expected_output(
            [
                "Three",
                " ",
                "times",
                "× ",
                "four",
                "^",
                "teen",
                " ÷",
                "divided",
                " ",
                "by",
                " [",
                "movies",
                "] ",
                "on",
                " \\",
                "slash",
                ".",
            ]
        )

        output = any_locale_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_any_locale_word_tokenize_fr(self):
        input_text = "pomme banane poire"
        expected_output = self._create_expected_output(["pomme", " ", "banane", " ", "poire"])

        output = any_locale_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_any_locale_word_tokenize_with_accents_fr(self):
        input_text = "L’hétérogénéité entre les langues est étonnante."
        expected_output = self._create_expected_output(
            ["L", "’", "hétérogénéité", " ", "entre", " ", "les", " ", "langues", " ", "est", " ", "étonnante", "."]
        )

        output = any_locale_word_tokenize(input_text)
        assert output == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_any_locale_word_tokenize_with_numbers(self):
        input_text = r"Trois fois× quatorze^ et dix ÷ divisé par [films] sur \slash."
        expected_output = self._create_expected_output(
            [
                "Trois",
                " ",
                "fois",
                "× ",
                "quatorze",
                "^ ",
                "et",
                " ",
                "dix",
                " ÷ ",
                "divisé",
                " ",
                "par",
                " [",
                "films",
                "] ",
                "sur",
                " \\",
                "slash",
                ".",
            ]
        )

        output = any_locale_word_tokenize(input_text)
        print(output)
        print(expected_output)
        assert output == expected_output
