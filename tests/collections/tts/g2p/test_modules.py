# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import unicodedata

import pytest

from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p
from nemo.collections.tts.g2p.utils import GRAPHEME_CASE_LOWER, GRAPHEME_CASE_MIXED, GRAPHEME_CASE_UPPER


class TestIpaG2p:

    PHONEME_DICT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phoneme_dict")
    PHONEME_DICT_PATH_DE = os.path.join(PHONEME_DICT_DIR, "test_dict_de.txt")
    PHONEME_DICT_PATH_EN = os.path.join(PHONEME_DICT_DIR, "test_dict_en.txt")
    PHONEME_DICT_PATH_ES = os.path.join(PHONEME_DICT_DIR, "test_dict_es.txt")
    GRAPHEME_PREFIX = "#"

    @staticmethod
    def _create_g2p(
        phoneme_dict=PHONEME_DICT_PATH_EN,
        locale=None,
        apply_to_oov_word=lambda x: x,
        use_chars=False,
        phoneme_probability=None,
        grapheme_case=GRAPHEME_CASE_UPPER,
        grapheme_prefix="",
    ):
        return IpaG2p(
            phoneme_dict,
            locale=locale,
            apply_to_oov_word=apply_to_oov_word,
            use_chars=use_chars,
            phoneme_probability=phoneme_probability,
            grapheme_case=grapheme_case,
            grapheme_prefix=grapheme_prefix,
        )

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_dict_with_phonemes(self):
        # fmt: off
        expected_symbols = {
            'h', 'ə', 'ˈ', 'ɫ', 'o', 'ʊ',
            'ˈ', 'w', 'ɝ', 'ɫ', 'd',
            'ˈ', 'l', 'ɛ', 'd',
            'ˈ', 'l', 'i', 'd',
            'ɛ', 'n', 'ˈ', 'v', 'ɪ', 'd', 'i', 'ə',
            'ˈ', 'd', 'ʒ', 'o', 'ʊ', 'n', 'z',
            'ˈ', 'ɛ', 'ɹ', 'ˌ', 'p', 'ɔ', 'ɹ', 't',
        }
        # fmt: on
        g2p = self._create_g2p()

        assert expected_symbols == g2p.symbols
        assert len(g2p.phoneme_dict["HELLO"]) == 1
        assert len(g2p.phoneme_dict["WORLD"]) == 1
        assert len(g2p.phoneme_dict["LEAD"]) == 2
        assert len(g2p.phoneme_dict["NVIDIA"]) == 1
        assert len(g2p.phoneme_dict["JONES"]) == 1
        assert len(g2p.phoneme_dict["AIRPORT"]) == 1
        assert g2p.phoneme_dict["HELLO"][0] == list("həˈɫoʊ")
        assert g2p.phoneme_dict["WORLD"][0] == list("ˈwɝɫd")
        assert g2p.phoneme_dict["LEAD"] == [list("ˈlɛd"), list("ˈlid")]
        assert g2p.phoneme_dict["NVIDIA"][0] == list("ɛnˈvɪdiə")
        assert g2p.phoneme_dict["JONES"][0] == list("ˈdʒoʊnz")
        assert g2p.phoneme_dict["AIRPORT"][0] == list("ˈɛɹˌpɔɹt")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_normalize_dict_with_graphemes_and_phonemes(self):
        # fmt: off
        expected_symbols = {
            f"{self.GRAPHEME_PREFIX}{char}"
            for char in {
                'H', 'E', 'L', 'L', 'O',
                'W', 'O', 'R', 'L', 'D',
                'L', 'E', 'A', 'D',
                'N', 'V', 'I', 'D', 'I', 'A',
                'J', 'O', 'N', 'E', 'S',
                'A', 'I', 'R', 'P', 'O', 'R', 'T',
            }
        }.union(
                {
                    'h', 'ə', 'ˈ', 'ɫ', 'o', 'ʊ',
                    'ˈ', 'w', 'ɝ', 'ɫ', 'd',
                    'ˈ', 'l', 'ɛ', 'd',
                    'ˈ', 'l', 'i', 'd',
                    'ɛ', 'n', 'ˈ', 'v', 'ɪ', 'd', 'i', 'ə',
                    'ˈ', 'd', 'ʒ', 'o', 'ʊ', 'n', 'z',
                    'ˈ', 'ɛ', 'ɹ', 'ˌ', 'p', 'ɔ', 'ɹ', 't',
                }
            )
        # fmt: on

        g2p = self._create_g2p(use_chars=True, grapheme_prefix=self.GRAPHEME_PREFIX)

        assert expected_symbols == g2p.symbols
        assert len(g2p.phoneme_dict["HELLO"]) == 1
        assert len(g2p.phoneme_dict["WORLD"]) == 1
        assert len(g2p.phoneme_dict["LEAD"]) == 2
        assert len(g2p.phoneme_dict["NVIDIA"]) == 1
        assert len(g2p.phoneme_dict["JONES"]) == 1
        assert len(g2p.phoneme_dict["AIRPORT"]) == 1
        assert g2p.phoneme_dict["HELLO"][0] == list("həˈɫoʊ")
        assert g2p.phoneme_dict["WORLD"][0] == list("ˈwɝɫd")
        assert g2p.phoneme_dict["LEAD"] == [list("ˈlɛd"), list("ˈlid")]
        assert g2p.phoneme_dict["NVIDIA"][0] == list("ɛnˈvɪdiə")
        assert g2p.phoneme_dict["JONES"][0] == list("ˈdʒoʊnz")
        assert g2p.phoneme_dict["AIRPORT"][0] == list("ˈɛɹˌpɔɹt")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_replace_symbols(self):
        g2p = self._create_g2p(use_chars=True, grapheme_prefix=self.GRAPHEME_PREFIX)

        # fmt: off
        # Get full vocab without 'i' (phoneme) and 'J' (grapheme)
        fixed_symbols = {
            f"{self.GRAPHEME_PREFIX}{char}"
            for char in {
                'H', 'E', 'L', 'L', 'O',
                'W', 'O', 'R', 'L', 'D',
                'L', 'E', 'A', 'D',
                'N', 'V', 'I', 'D', 'I', 'A',
                'O', 'N', 'E', 'S',
                'A', 'I', 'R', 'P', 'O', 'R', 'T',
            }
        }.union(
                {
                    'h', 'ə', 'ˈ', 'ɫ', 'o', 'ʊ',
                    'ˈ', 'w', 'ɝ', 'ɫ', 'd',
                    'ˈ', 'l', 'ɛ', 'd',
                    'ˈ', 'l', 'd',
                    'ɛ', 'n', 'ˈ', 'v', 'ɪ', 'd', 'ə',
                    'ˈ', 'd', 'ʒ', 'o', 'ʊ', 'n', 'z',
                    'ˈ', 'ɛ', 'ɹ', 'ˌ', 'p', 'ɔ', 'ɹ', 't',
                }
        )
        # fmt: on

        assert len(g2p.phoneme_dict["LEAD"]) == 2
        assert len(g2p.phoneme_dict["JONES"]) == 1
        assert len(g2p.phoneme_dict["NVIDIA"]) == 1

        # Test with keep_alternate set to True (default)
        g2p.replace_symbols(symbols=fixed_symbols, keep_alternate=True)

        # Check that the alternate pron of "LEAD" was kept
        assert len(g2p.phoneme_dict["LEAD"]) == 1
        assert g2p.phoneme_dict["LEAD"][0] == list("ˈlɛd")
        # Check that filtering was done for unique entries, both grapheme and phoneme
        assert "JONES" not in g2p.phoneme_dict
        assert "NVIDIA" not in g2p.phoneme_dict
        # Check that other words weren't affected
        assert g2p.phoneme_dict["HELLO"][0] == list("həˈɫoʊ")
        assert g2p.phoneme_dict["WORLD"][0] == list("ˈwɝɫd")
        assert g2p.phoneme_dict["AIRPORT"][0] == list("ˈɛɹˌpɔɹt")

        # Test with keep_alternate set to False
        g2p = self._create_g2p(use_chars=True, grapheme_prefix=self.GRAPHEME_PREFIX)
        g2p.replace_symbols(symbols=fixed_symbols, keep_alternate=False)

        # Check that both "LEAD" entries were removed
        assert "LEAD" not in g2p.phoneme_dict
        # Other checks remain the same
        assert "JONES" not in g2p.phoneme_dict
        assert "NVIDIA" not in g2p.phoneme_dict
        assert g2p.phoneme_dict["HELLO"][0] == list("həˈɫoʊ")
        assert g2p.phoneme_dict["WORLD"][0] == list("ˈwɝɫd")
        assert g2p.phoneme_dict["AIRPORT"][0] == list("ˈɛɹˌpɔɹt")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call(self):
        input_text = "Hello world."
        expected_output = [char for char in "həˈɫoʊ ˈwɝɫd."]
        g2p = self._create_g2p()

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_file_or_object_dict_type(self):
        input_text = "Hello world."
        expected_output = [char for char in "həˈɫoʊ ˈwɝɫd."]

        phoneme_dict = {"HELLO": ["həˈɫoʊ"], "WORLD": ["ˈwɝɫd"], "LEAD": ["ˈlɛd", "ˈlid"], "NVIDIA": ["ɛnˈvɪdiə"]}

        g2p_file = self._create_g2p()
        g2p_dict = self._create_g2p(phoneme_dict=phoneme_dict)

        phonemes_file = g2p_file(input_text)
        phonemes_dict = g2p_dict(input_text)
        assert phonemes_dict == expected_output
        assert phonemes_file == phonemes_dict

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_oov_word(self):
        input_text = "Hello Kitty!"
        expected_output = list("həˈɫoʊ") + [" "] + list("KITTY") + ["!"]
        g2p = self._create_g2p()

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_oov_func(self):
        input_text = "Hello Kitty!"
        expected_output = list("həˈɫoʊ") + [" "] + list("test!")
        g2p = self._create_g2p(apply_to_oov_word=lambda x: "test")

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_uppercase_grapheme_only(self):
        input_text = "Hello world."
        expected_output = [self.GRAPHEME_PREFIX + char if char not in " ." else char for char in input_text.upper()]
        g2p = self._create_g2p(
            use_chars=True,
            phoneme_probability=0.0,
            grapheme_case=GRAPHEME_CASE_UPPER,
            grapheme_prefix=self.GRAPHEME_PREFIX,
        )

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_lowercase_grapheme_only(self):
        input_text = "Hello world."
        expected_output = [self.GRAPHEME_PREFIX + char if char not in " ." else char for char in input_text.lower()]
        g2p = self._create_g2p(
            use_chars=True,
            phoneme_probability=0.0,
            grapheme_case=GRAPHEME_CASE_LOWER,
            grapheme_prefix=self.GRAPHEME_PREFIX,
        )

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_mixed_case_grapheme_only(self):
        input_text = "Hello world."
        expected_output = [self.GRAPHEME_PREFIX + char if char not in " ." else char for char in input_text]
        g2p = self._create_g2p(
            use_chars=True,
            phoneme_probability=0.0,
            grapheme_case=GRAPHEME_CASE_MIXED,
            grapheme_prefix=self.GRAPHEME_PREFIX,
        )

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_uppercase_grapheme_and_get_phoneme_only(self):
        input_text = "Hello world."
        expected_output = ["h", "ə", "ˈ", "ɫ", "o", "ʊ", " ", "ˈ", "w", "ɝ", "ɫ", "d", "."]
        g2p = self._create_g2p(use_chars=True, phoneme_probability=1.0, grapheme_case=GRAPHEME_CASE_UPPER)

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_lowercase_grapheme_and_get_phoneme_only(self):
        input_text = "Hello world."
        expected_output = ["h", "ə", "ˈ", "ɫ", "o", "ʊ", " ", "ˈ", "w", "ɝ", "ɫ", "d", "."]
        g2p = self._create_g2p(use_chars=True, phoneme_probability=1.0, grapheme_case=GRAPHEME_CASE_LOWER)

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_mixed_case_grapheme_and_get_phoneme_only(self):
        input_text = "Hello world."
        expected_output = ["h", "ə", "ˈ", "ɫ", "o", "ʊ", " ", "ˈ", "w", "ɝ", "ɫ", "d", "."]
        g2p = self._create_g2p(use_chars=True, phoneme_probability=1.0, grapheme_case=GRAPHEME_CASE_MIXED)

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_with_escaped_characters(self):
        input_text = "Hello |wo rld|."
        expected_output = [
            "h",
            "ə",
            "ˈ",
            "ɫ",
            "o",
            "ʊ",
            " ",
            f"{self.GRAPHEME_PREFIX}wo",
            f"{self.GRAPHEME_PREFIX}rld",
            ".",
        ]
        g2p = self._create_g2p(grapheme_prefix=self.GRAPHEME_PREFIX)

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_instantiate_unsupported_locale(self):
        with pytest.raises(ValueError, match="Unsupported locale"):
            self._create_g2p(locale="en-USA")

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_de_de(self):
        input_text = "Hallo „welt“" + " " + "Weg" + " " + unicodedata.normalize("NFD", "Abendröte!" + " " + "weg")
        expected_output = (
            list("hˈaloː „vˈɛlt“")
            + [" "]
            + list("vˈeːk")
            + [" "]
            + [f"{self.GRAPHEME_PREFIX}{char}" for char in unicodedata.normalize("NFC", "Abendröte")]
            + ["!"]
            + [" "]
            + list("vˈɛk")
        )
        g2p = self._create_g2p(
            use_chars=True,
            phoneme_dict=self.PHONEME_DICT_PATH_DE,
            locale="de-DE",
            grapheme_case=GRAPHEME_CASE_MIXED,
            grapheme_prefix=self.GRAPHEME_PREFIX,
            apply_to_oov_word=None,
        )

        phonemes = g2p(input_text)
        assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_en_us(self):
        input_text = "Hello NVIDIA'S airport's Jones's airports worlds Kitty!"

        g2p_upper = self._create_g2p(locale="en-US", grapheme_case=GRAPHEME_CASE_UPPER)
        expected_output_upper = [char for char in "həˈɫoʊ ɛnˈvɪdiəz ˈɛɹˌpɔɹts ˈdʒoʊnzɪz ˈɛɹˌpɔɹts ˈwɝɫdz KITTY!"]

        g2p_lower = self._create_g2p(
            locale="en-US",
            grapheme_case=GRAPHEME_CASE_LOWER,
            grapheme_prefix=self.GRAPHEME_PREFIX,
            apply_to_oov_word=None,
        )
        expected_output_lower = (
            [char for char in "həˈɫoʊ ɛnˈvɪdiəz ˈɛɹˌpɔɹts ˈdʒoʊnzɪz ˈɛɹˌpɔɹts ˈwɝɫdz"]
            + [" "]
            + [f"{self.GRAPHEME_PREFIX}{char}" for char in "kitty"]
            + ["!"]
        )

        g2p_mixed = self._create_g2p(
            locale="en-US",
            grapheme_case=GRAPHEME_CASE_MIXED,
            grapheme_prefix=self.GRAPHEME_PREFIX,
            apply_to_oov_word=None,
        )
        expected_output_mixed = (
            [char for char in "həˈɫoʊ ɛnˈvɪdiəz ˈɛɹˌpɔɹts ˈdʒoʊnzɪz ˈɛɹˌpɔɹts ˈwɝɫdz"]
            + [" "]
            + [f"{self.GRAPHEME_PREFIX}{char}" for char in "kitty"]
            + ["!"]
        )

        for g2p, expected_output in zip(
            [g2p_upper, g2p_lower, g2p_mixed], [expected_output_upper, expected_output_lower, expected_output_mixed]
        ):
            phonemes = g2p(input_text)
            assert phonemes == expected_output

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_forward_call_es_es(self):
        input_text = "¿Hola mundo, amigo?"
        expected_output = [char for char in "¿ˈola mˈundo, AMIGO?"]
        g2p = self._create_g2p(phoneme_dict=self.PHONEME_DICT_PATH_ES, locale="es-ES")

        phonemes = g2p(input_text)
        assert phonemes == expected_output
