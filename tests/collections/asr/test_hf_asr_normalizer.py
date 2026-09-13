# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Pins the behaviour of the vendored Open ASR Leaderboard normalizer.

The cases below are the ones where the leaderboard normalizer diverges from Whisper's, so they
would catch a refresh of the vendored copy that silently changes leaderboard WER.
"""

import pytest

from nemo.collections.asr.parts.utils.hf_asr_normalizer import (
    EnglishAcronymNormalizer,
    EnglishTextNormalizer,
    MultilingualNormalizer,
    get_hf_normalizer,
)


@pytest.fixture(scope="module")
def normalizer():
    return EnglishTextNormalizer()


class TestEnglishTextNormalizer:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected",
        [
            # Inherited from Whisper: casing, punctuation, contractions, titles, numbers, spelling.
            ("Mr. Smith paid $1,000.", "mister smith paid $1000"),
            ("She'd gone, ain't that right, y'all?", "she had gone aint that right you all"),
            ("It's twenty-three percent!", "it is 23%"),
            ("colour organisation defence", "color organization defense"),
            ("um, it was [inaudible] (laughs) fine", "it was fine"),
        ],
    )
    def test_whisper_inherited_behaviour(self, normalizer, text, expected):
        assert normalizer(text) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected",
        [
            # Leaderboard-only: acronym collapsing.
            ("The B B C reported it", "the bbc reported it"),
            ("5 G rollout", "5g rollout"),
            # Leaderboard-only: multi-word compound rewrites.
            ("Wi-Fi and e-mail", "wifi and email"),
            ("et cetera", "etc"),
            ("at 3 p.m.", "at 3 pm"),
        ],
    )
    def test_leaderboard_specific_behaviour(self, normalizer, text, expected):
        assert normalizer(text) == expected

    @pytest.mark.unit
    def test_lone_single_letters_are_preserved(self, normalizer):
        # A run of single characters is only collapsed when it is unambiguously an acronym.
        assert normalizer("a big cat") == "a big cat"

    @pytest.mark.unit
    @pytest.mark.parametrize("text", ["", "   ", "[noise]"])
    def test_empty_input(self, normalizer, text):
        assert normalizer(text).strip() == ""


class TestEnglishAcronymNormalizer:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("b b c", "bbc"),
            ("5 g", "5g"),
            ("a big cat", "a big cat"),
            ("i a", "i a"),  # runs containing "a"/"i" need 3+ tokens
            ("i a m", "iam"),
        ],
    )
    def test_collapse(self, text, expected):
        assert EnglishAcronymNormalizer()(text) == expected


class TestGetHfNormalizer:
    @pytest.mark.unit
    def test_english_dispatch(self):
        assert isinstance(get_hf_normalizer("en"), EnglishTextNormalizer)

    @pytest.mark.unit
    def test_non_english_dispatch(self):
        # num2words is an optional dependency, so only the diacritic/punctuation half is asserted here.
        normalizer = MultilingualNormalizer(remove_diacritics=False)
        assert normalizer("Wir haben Häuser, ja!") == "wir haben häuser ja"

    @pytest.mark.unit
    def test_non_english_number_normalization(self):
        pytest.importorskip("num2words")
        assert get_hf_normalizer("de")("Wir haben 10 000 Häuser") == "wir haben zehntausend häuser"
