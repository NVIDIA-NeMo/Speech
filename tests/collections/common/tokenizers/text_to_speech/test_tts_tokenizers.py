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
    FrenchCharsTokenizer,
    GermanCharsTokenizer,
    HindiCharsTokenizer,
    IPATokenizer,
    ItalianCharsTokenizer,
    JapanesePhonemeTokenizer,
    SpanishCharsTokenizer,
    VietnameseCharsTokenizer,
)
from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p
from nemo.collections.tts.g2p.models.ja_jp_ipa import JapaneseG2p, JapaneseKatakanaAccentG2p


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
    PHONEME_DICT_IT = {
        "CIAO": ["tʃˈao"],
        "MONDO": ["mˈondo"],
    }
    PHONEME_DICT_FR = {
        "BONJOUR": ["bɔ̃ʒˈuʁ"],
        "LE": ["lˈə-"],
        "MONDE": ["mˈɔ̃d"],
    }
    PHONEME_DICT_JA = {
        "ハロー": ["haɾoː"],
        "ワールド": ["wa:ɾdo"],
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
    def test_vietnamese_chars_tokenizer(self):
        input_text = "Xin chào các bạn."
        expected_output = "xin chào các bạn."

        tokenizer = VietnameseCharsTokenizer()
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output
        assert len(tokens) == len(input_text)

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_french_chars_tokenizer(self):
        input_text = "Bon après-midi !"
        expected_output = "bon après-midi !"

        tokenizer = FrenchCharsTokenizer()
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
    def test_ipa_tokenizer_it_it(self):
        input_text = "Ciao mondo"
        expected_output = "tʃˈao mˈondo"

        g2p = IpaG2p(phoneme_dict=self.PHONEME_DICT_IT, locale="it-IT")
        tokenizer = IPATokenizer(g2p=g2p, locale="it-IT")
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
    def test_ipa_tokenizer_fr_fr(self):
        input_text = "Bonjour le monde"
        expected_output = "bɔ̃ʒˈuʁ lˈə- mˈɔ̃d"

        g2p = IpaG2p(phoneme_dict=self.PHONEME_DICT_FR, locale="fr-FR")
        tokenizer = IPATokenizer(g2p=g2p, locale="fr-FR")
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

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_japanese_phoneme_tokenizer(self):
        input_text = "ハロー ワールド."
        expected_output = "haɾoː wa:ɾdo."
        g2p = JapaneseG2p(phoneme_dict=self.PHONEME_DICT_JA, word_segmenter="janome")

        tokenizer = JapanesePhonemeTokenizer(g2p=g2p)
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_japanese_katakana_accent_tokenizer(self):
        input_chopsticks = "箸"
        input_bridge = "橋"

        g2p = JapaneseKatakanaAccentG2p()
        tokenizer = JapanesePhonemeTokenizer(g2p=g2p, punct=False)

        chars_chopsticks, _ = self._parse_text(tokenizer, input_chopsticks)
        chars_bridge, _ = self._parse_text(tokenizer, input_bridge)

        assert chars_chopsticks != chars_bridge, "Homonyms should have different pitch accents"

        # 箸 (1ハ0シ) starts high
        assert '1' in chars_chopsticks[:2]
        # 橋 (0ハ1シ) starts low
        assert '0' in chars_bridge[:2]

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_hindi_chars_tokenizer(self):
        input_text = "नमस्ते दुनिया!"
        expected_output = "नमस्ते दुनिया!"

        tokenizer = HindiCharsTokenizer()
        chars, tokens = self._parse_text(tokenizer, input_text)

        assert chars == expected_output
