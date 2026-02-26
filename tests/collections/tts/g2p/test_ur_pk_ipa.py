# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Unit tests for UrduIpaG2p (ur_pk_ipa.py)."""

import json
import os
import tempfile

import pytest

from nemo.collections.tts.g2p.models.ur_pk_ipa import UrduIpaG2p, urdu_word_tokenize


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_DICT = {
    "غیر حاضری":    "ɣɛːr hɑːzriː",
    "شوکت خانم ليب": "ʃoːˈkət̪ xɑːˈnəm leːb",
    "مختار احمد":   "mʊxˈt̪aːr ˈæhməd",
    "افسوس ناک":    "əfˈsoːs naːk",
    "پانی":         "pɑːniː",
    "آم":           "ɑːm",
}


@pytest.fixture
def dict_file(tmp_path):
    """Write SAMPLE_DICT to a temp JSON file and return its path."""
    path = tmp_path / "urdu_ipa_dict.json"
    path.write_text(json.dumps(SAMPLE_DICT, ensure_ascii=False), encoding="utf-8")
    return str(path)


@pytest.fixture
def g2p(dict_file):
    return UrduIpaG2p(phoneme_dict=dict_file)


# ---------------------------------------------------------------------------
# urdu_word_tokenize
# ---------------------------------------------------------------------------

class TestUrduWordTokenize:
    def test_pure_urdu_tokens_are_not_unchanged(self):
        result = urdu_word_tokenize("غیر حاضری")
        assert result == [
            (["غیر"], False),
            (["حاضری"], False),
        ]

    def test_digits_are_unchanged(self):
        result = urdu_word_tokenize("123")
        assert result == [(["123"], True)]

    def test_latin_is_unchanged(self):
        result = urdu_word_tokenize("NeMo")
        assert result == [(["NeMo"], True)]

    def test_mixed_sentence(self):
        result = urdu_word_tokenize("پانی 100ml")
        tokens = {tok: flag for (tok_list, flag) in result for tok in tok_list}
        assert tokens["پانی"] is False
        assert tokens["100ml"] is True

    def test_empty_string(self):
        assert urdu_word_tokenize("") == []

    def test_extra_whitespace_ignored(self):
        result = urdu_word_tokenize("  پانی  آم  ")
        words = [tok for (tok_list, _) in result for tok in tok_list]
        assert words == ["پانی", "آم"]


# ---------------------------------------------------------------------------
# UrduIpaG2p — initialisation
# ---------------------------------------------------------------------------

class TestUrduIpaG2pInit:
    def test_load_from_json_file(self, dict_file):
        g2p = UrduIpaG2p(phoneme_dict=dict_file)
        assert len(g2p.phoneme_dict) == len(SAMPLE_DICT)

    def test_load_from_dict_object(self):
        g2p = UrduIpaG2p(phoneme_dict=SAMPLE_DICT)
        assert len(g2p.phoneme_dict) == len(SAMPLE_DICT)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="no valid entries"):
            UrduIpaG2p(phoneme_dict={})

    def test_stress_stripping_at_load(self):
        g2p = UrduIpaG2p(phoneme_dict=SAMPLE_DICT, use_stresses=False)
        for prons in g2p.phoneme_dict.values():
            for pron in prons:
                for token in pron:
                    assert "ˈ" not in token
                    assert "ˌ" not in token

    def test_heteronyms_from_list(self):
        g2p = UrduIpaG2p(phoneme_dict=SAMPLE_DICT, heteronyms=["پانی"])
        assert "پانی" in g2p.heteronyms

    def test_heteronyms_from_file(self, dict_file, tmp_path):
        het_file = tmp_path / "heteronyms.txt"
        het_file.write_text("پانی\nآم\n", encoding="utf-8")
        g2p = UrduIpaG2p(phoneme_dict=dict_file, heteronyms=str(het_file))
        assert "پانی" in g2p.heteronyms
        assert "آم" in g2p.heteronyms


# ---------------------------------------------------------------------------
# UrduIpaG2p — single word lookup
# ---------------------------------------------------------------------------

class TestParseOneWord:
    def test_known_word(self, g2p):
        pron, handled = g2p.parse_one_word("پانی")
        assert handled is True
        assert pron == ["pɑːniː"]

    def test_oov_returns_grapheme_chars(self, g2p):
        pron, handled = g2p.parse_one_word("انجان")
        assert handled is False
        assert pron == list("انجان")

    def test_heteronym_returns_grapheme_chars(self):
        g2p = UrduIpaG2p(phoneme_dict=SAMPLE_DICT, heteronyms=["پانی"])
        pron, handled = g2p.parse_one_word("پانی")
        assert handled is True
        assert pron == list("پانی")

    def test_oov_with_apply_func(self):
        g2p = UrduIpaG2p(
            phoneme_dict=SAMPLE_DICT,
            apply_to_oov_word=lambda w: ["OOV"],
        )
        pron, handled = g2p.parse_one_word("انجان")
        assert pron == ["OOV"]
        assert handled is True


# ---------------------------------------------------------------------------
# UrduIpaG2p — __call__ (full inference)
# ---------------------------------------------------------------------------

class TestCall:
    def test_single_word(self, g2p):
        assert g2p("پانی") == ["pɑːniː"]

    def test_multi_word_phrase_lookup(self, g2p):
        # "غیر حاضری" is a two-word phrase in the dict — should match as one entry
        result = g2p("غیر حاضری")
        assert result == ["ɣɛːr", "hɑːzriː"]

    def test_three_word_phrase_lookup(self, g2p):
        result = g2p("شوکت خانم ليب")
        assert result == ["ʃoːˈkət̪", "xɑːˈnəm", "leːb"]

    def test_sentence_with_oov(self, g2p):
        result = g2p("پانی انجان")
        # پانی is known; انجان is OOV -> grapheme chars
        assert result[:1] == ["pɑːniː"]
        assert result[1:] == list("انجان")

    def test_non_urdu_passthrough(self, g2p):
        result = g2p("NeMo")
        assert result == ["NeMo"]

    def test_mixed_urdu_latin(self, g2p):
        result = g2p("پانی NeMo آم")
        assert "pɑːniː" in result
        assert "NeMo" in result
        assert "ɑːm" in result

    def test_nfc_normalisation(self, g2p):
        # Compose پانی using NFD (decomposed) form — should still match
        import unicodedata
        nfd = unicodedata.normalize("NFD", "پانی")
        result = g2p(nfd)
        assert result == ["pɑːniː"]

    def test_empty_string(self, g2p):
        assert g2p("") == []

    def test_stress_stripped_output(self):
        g2p = UrduIpaG2p(phoneme_dict=SAMPLE_DICT, use_stresses=False)
        result = g2p("شوکت خانم ليب")
        for token in result:
            assert "ˈ" not in token

    def test_is_unique_in_phoneme_dict(self, g2p):
        # All SAMPLE_DICT entries have one variant
        assert g2p.is_unique_in_phoneme_dict("پانی") is True