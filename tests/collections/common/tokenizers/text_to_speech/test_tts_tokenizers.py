# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import (
    EnglishCharsTokenizer,
    GermanCharsTokenizer,
    IPATokenizer,
    ItalianCharsTokenizer,
    SpanishCharsTokenizer,
)
from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p


class TestTTSTokenizers:
    PHONEME_DICT_DE = {
        "HALLO": ["hˈaloː"],
        "WELT": ["vˈɛlt"],
    }
    PHONEME_DICT_EN = {"HELLO": ["həˈɫoʊ"], "WORLD": ["ˈwɝɫd"], "CAFE": ["kəˈfeɪ"]}
    PHONEME_DICT_ES = {
        "BUENOS": ["bwˈenos"],
        "DÍAS": ["dˈias"],
    }

    @staticmethod
    def _parse_text(tokenizer, text):
        tokens = tokenizer.encode(text)
        chars = tokenizer.decode(tokens)
        chars = chars.replace('|', '')
        return chars, tokens

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_chars_tokenizer(self):
        input_text = "Hello world!"
        expected_output = "hello world!"

        tokenizer = EnglishCharsTokenizer()
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output
        assert len(tokens) == len(input_text)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_chars_tokenizer_unknown_token(self):
        input_text = "Hey 🙂 there"
        expected_output = "hey there"

        tokenizer = EnglishCharsTokenizer()
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output
        assert len(tokens) == len(expected_output)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_english_chars_tokenizer_accented_character(self):
        input_text = "Let's drink at the café."
        expected_output = "let's drink at the cafe."

        tokenizer = EnglishCharsTokenizer()
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output
        assert len(tokens) == len(input_text)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_german_chars_tokenizer(self):
        input_text = "Was ist dein Lieblingsgetränk?"
        expected_output = "Was ist dein Lieblingsgetränk?"

        tokenizer = GermanCharsTokenizer()
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output
        assert len(tokens) == len(input_text)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_italian_chars_tokenizer(self):
        input_text = "Ciao mondo!"
        expected_output = "ciao mondo!"

        tokenizer = ItalianCharsTokenizer()
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output
        assert len(tokens) == len(input_text)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_spanish_chars_tokenizer(self):
        input_text = "¿Cuál es su nombre?"
        expected_output = "¿cuál es su nombre?"

        tokenizer = SpanishCharsTokenizer()
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output
        assert len(tokens) == len(input_text)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_tokenizer(self):
        input_text = "Hello world!"
        expected_output = " həˈɫoʊ ˈwɝɫd! "

        g2p = IpaG2p(phoneme_dict=self.PHONEME_DICT_EN)

        tokenizer = IPATokenizer(g2p=g2p, locale=None, pad_with_space=True)
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_tokenizer_unsupported_locale(self):
        g2p = IpaG2p(phoneme_dict=self.PHONEME_DICT_EN)
        with pytest.raises(ValueError, match="Unsupported locale"):
            IPATokenizer(g2p=g2p, locale="asdf")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_tokenizer_de_de(self):
        input_text = "Hallo welt"
        expected_output = "hˈaloː vˈɛlt"

        g2p = IpaG2p(phoneme_dict=self.PHONEME_DICT_DE, locale="de-DE")
        tokenizer = IPATokenizer(g2p=g2p, locale="de-DE")
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_tokenizer_en_us(self):
        input_text = "Hello café."
        expected_output = "həˈɫoʊ kəˈfeɪ."
        g2p = IpaG2p(phoneme_dict=self.PHONEME_DICT_EN)

        tokenizer = IPATokenizer(g2p=g2p, locale="en-US")
        tokenizer.tokens.extend("CAFE")
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_tokenizer_es_es(self):
        input_text = "¡Buenos días!"
        expected_output = "¡bwˈenos dˈias!"

        g2p = IpaG2p(phoneme_dict=self.PHONEME_DICT_ES, locale="es-ES")
        tokenizer = IPATokenizer(g2p=g2p, locale="es-ES")
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_ipa_tokenizer_fixed_vocab(self):
        phoneme_dict = self.PHONEME_DICT_EN
        phoneme_dict["WOUND"] = ["ˈwaʊnd", "ˈwund"]
        g2p = IpaG2p(phoneme_dict=phoneme_dict)

        assert "WOUND" in g2p.phoneme_dict

        # fmt: off
        symbol_vocab = {
            'H', 'E', 'L', 'L', 'O',
            'W', 'O', 'R', 'L', 'D',
            'C', 'A', 'F', 'E',
            'W', 'O', 'U', 'N', 'D',
            'h', 'ə', 'ˈ', 'ɫ', 'o', 'ʊ',
            'ˈ', 'w', 'ɝ', 'ɫ', 'd',
            'k', 'ə', 'ˈ', 'f', 'e', 'ɪ',
            'ˈ', 'w', 'a', 'ʊ', 'n', 'd',
            'ˈ', 'w', 'u', 'n', 'd',
        }
        # fmt: on
        fixed_vocab = symbol_vocab - {'ʊ', 'F'}
        tokenizer = IPATokenizer(g2p=g2p, locale="en-US", fixed_vocab=fixed_vocab)

        # Make sure phoneme_dict has been updated properly
        assert "HELLO" not in tokenizer.g2p.phoneme_dict
        assert "WORLD" in tokenizer.g2p.phoneme_dict
        assert "CAFE" not in tokenizer.g2p.phoneme_dict
        assert len(tokenizer.g2p.phoneme_dict["WOUND"]) == 1
        assert tokenizer.g2p.phoneme_dict["WOUND"][0] == list("ˈwund")

        chars, tokens = self._parse_text(tokenizer, "Hello, wound")
        expected_output = "HELLO, ˈwund"
        assert chars == expected_output
