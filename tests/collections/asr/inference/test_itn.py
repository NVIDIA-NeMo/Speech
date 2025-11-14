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
from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer


@pytest.fixture(scope="module")
def en_itn_model():
    return AlignmentPreservingInverseNormalizer(
        lang="en", input_case=AlignmentPreservingInverseNormalizer.LOWER_CASED, cache_dir=None
    )


@pytest.fixture(scope="module")
def de_itn_model():
    return AlignmentPreservingInverseNormalizer(
        lang="de", input_case=AlignmentPreservingInverseNormalizer.LOWER_CASED, cache_dir=None
    )


@pytest.fixture(scope="module")
def es_itn_model():
    return AlignmentPreservingInverseNormalizer(
        lang="es", input_case=AlignmentPreservingInverseNormalizer.LOWER_CASED, cache_dir=None
    )


class TestAlignmentPreservingInverseNormalizer:

    @pytest.mark.unit
    def test_word_alignment_cardinal_en(self, en_itn_model):
        text = "zzz minus twenty five thousand thirty seven zzz"
        iwords, owords, alignment = en_itn_model.get_word_alignment(text, sep=" ")
        assert iwords == ["zzz", "minus", "twenty", "five", "thousand", "thirty", "seven", "zzz"]
        assert owords == ["zzz", "-25037", "zzz"]
        assert alignment == [([0], [0], "name"), ([1, 2, 3, 4, 5, 6], [1], "cardinal"), ([7], [2], "name")]

    @pytest.mark.unit
    def test_word_alignment_time_en(self, en_itn_model):
        text = "zzz eleven fifty five p m zzz"
        iwords, owords, alignment = en_itn_model.get_word_alignment(text, sep=" ")
        assert iwords == ["zzz", "eleven", "fifty", "five", "p", "m", "zzz"]
        assert owords == ["zzz", "11:55", "p.m.", "zzz"]
        assert alignment == [([0], [0], "name"), ([1, 2, 3, 4, 5], [1, 2], "time"), ([6], [3], "name")]

    @pytest.mark.unit
    def test_word_alignment_money_en(self, en_itn_model):
        text = "zzz two hundred fifty dollars zzz"
        iwords, owords, alignment = en_itn_model.get_word_alignment(text, sep=" ")
        assert iwords == ["zzz", "two", "hundred", "fifty", "dollars", "zzz"]
        assert owords == ["zzz", "$250", "zzz"]
        assert alignment == [([0], [0], "name"), ([1, 2, 3, 4], [1], "money"), ([5], [2], "name")]

    @pytest.mark.unit
    def test_word_alignment_combo_en(self, en_itn_model):
        text = "eleven twenty seven fifty seven october twenty fourth nineteen seventy"
        iwords, owords, alignment = en_itn_model.get_word_alignment(text, sep=" ")
        assert iwords == [
            "eleven",
            "twenty",
            "seven",
            "fifty",
            "seven",
            "october",
            "twenty",
            "fourth",
            "nineteen",
            "seventy",
        ]
        assert owords == ["1120", "07:57", "october", "24", "1970"]
        assert alignment == [([0, 1], [0], "date"), ([2, 3, 4], [1], "time"), ([5, 6, 7, 8, 9], [2, 3, 4], "date")]

    @pytest.mark.unit
    def test_word_alignment_measure_en(self, en_itn_model):
        text = "it is two hundred fifty meters long"
        iwords, owords, alignment = en_itn_model.get_word_alignment(text, sep=" ")
        assert iwords == ["it", "is", "two", "hundred", "fifty", "meters", "long"]
        assert owords == ["it", "is", "250", "m", "long"]
        assert alignment == [
            ([0], [0], "name"),
            ([1], [1], "name"),
            ([2, 3, 4, 5], [2, 3], "measure"),
            ([6], [4], "name"),
        ]

    @pytest.mark.unit
    def test_word_alignment_sterling_en(self, en_itn_model):
        text = "trade turnover of three million pounds sterling"
        iwords, owords, alignment = en_itn_model.get_word_alignment(text, sep=" ")
        assert iwords == ["trade", "turnover", "of", "three", "million", "pounds", "sterling"]
        assert owords == ["trade", "turnover", "of", "£3", "million"]
        assert alignment == [
            ([0], [0], "name"),
            ([1], [1], "name"),
            ([2], [2], "name"),
            ([3, 4, 5, 6], [3, 4], "money"),
        ]

    @pytest.mark.unit
    def test_word_alignment_time_de(self, de_itn_model):
        text = "zzz drei uhr zwanzig zzz"
        iwords, owords, alignment = de_itn_model.get_word_alignment(text, sep=" ")
        assert iwords == ["zzz", "drei", "uhr", "zwanzig", "zzz"]
        assert owords == ['zzz', '03:20', 'Uhr', 'zzz']
        assert alignment == [([0], [0], "name"), ([1, 2, 3], [1, 2], "time"), ([4], [3], "name")]

    @pytest.mark.unit
    def test_word_alignment_money_de(self, de_itn_model):
        text = "zzz zwei hundert fünfzig dollar zzz"
        iwords, owords, alignment = de_itn_model.get_word_alignment(text, sep=" ")
        assert iwords == ["zzz", "zwei", "hundert", "fünfzig", "dollar", "zzz"]
        assert owords == ["zzz", "$250", "zzz"]
        assert alignment == [([0], [0], "name"), ([1, 2, 3, 4], [1], "money"), ([5], [2], "name")]

    @pytest.mark.unit
    def test_word_alignment_cardinal_de(self, de_itn_model):
        text = "zzz minus fünfundzwanzigtausendsiebenunddreißig zzz"
        iwords, owords, alignment = de_itn_model.get_word_alignment(text, sep=" ")
        assert iwords == ["zzz", "minus", "fünfundzwanzigtausendsiebenunddreißig", "zzz"]
        assert owords == ["zzz", "-25037", "zzz"]
        assert alignment == [([0], [0], "name"), ([1, 2], [1], "cardinal"), ([3], [2], "name")]

    @pytest.mark.unit
    def test_word_alignment_measure_de(self, de_itn_model):
        text = "es ist zweihundertfünfzig meter lang"
        iwords, owords, alignment = de_itn_model.get_word_alignment(text, sep=" ")
        assert iwords == ["es", "ist", "zweihundertfünfzig", "meter", "lang"]
        assert owords == ["es", "ist", "250", "m", "lang"]
        assert alignment == [([0], [0], "name"), ([1], [1], "name"), ([2, 3], [2, 3], "measure"), ([4], [4], "name")]

    @pytest.mark.unit
    def test_word_alignment_combo_es(self, es_itn_model):
        text = "un mil intereses al diez por ciento a la semana estándar derecho diez por ciento"
        iwords, owords, alignment = es_itn_model.get_word_alignment(text, sep=" ")
        assert iwords == [
            'un',
            'mil',
            'intereses',
            'al',
            'diez',
            'por',
            'ciento',
            'a',
            'la',
            'semana',
            'estándar',
            'derecho',
            'diez',
            'por',
            'ciento',
        ]
        assert owords == ['1000', 'intereses', 'al', '10', '%', 'a', 'la', 'semana', 'estándar', 'derecho', '10', '%']
        assert alignment == [
            ([0, 1], [0], 'cardinal'),
            ([2], [1], 'name'),
            ([3], [2], 'name'),
            ([4, 5, 6], [3, 4], 'measure'),
            ([7], [5], 'name'),
            ([8], [6], 'name'),
            ([9], [7], 'name'),
            ([10], [8], 'name'),
            ([11], [9], 'name'),
            ([12, 13, 14], [10, 11], 'measure'),
        ]
