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

"""
Tests for utility functions in nemo.collections.speechlm2.data.utils:
  - normalize_numbers (and its sub-expanders)
  - is_mcq_cut_train / is_mcq_cut_val / is_asr_cut
"""

import pytest

from nemo.collections.speechlm2.data.utils import (
    is_asr_cut,
    is_mcq_cut_train,
    is_mcq_cut_val,
    normalize_numbers,
)


# ── normalize_numbers ────────────────────────────────────────────────────────


class TestNormalizeNumbers:
    def test_plain_number(self):
        assert normalize_numbers("I have 3 cats") == "I have three cats"

    def test_large_number(self):
        result = normalize_numbers("There are 100 people")
        assert "one hundred" in result

    def test_comma_separated(self):
        result = normalize_numbers("Population is 1,000,000")
        assert "," not in result
        assert "one million" in result

    def test_dollars_whole(self):
        result = normalize_numbers("It costs $5")
        assert "five dollars" in result

    def test_dollars_with_cents(self):
        result = normalize_numbers("Price is $3.50")
        assert "three dollars" in result
        assert "fifty cents" in result

    def test_dollars_zero(self):
        result = normalize_numbers("$0.00")
        assert "zero dollars" in result

    def test_ordinal_1st(self):
        result = normalize_numbers("He came 1st")
        assert "first" in result

    def test_ordinal_2nd(self):
        result = normalize_numbers("She was 2nd")
        assert "second" in result

    def test_ordinal_3rd(self):
        result = normalize_numbers("3rd place")
        assert "third" in result

    def test_ordinal_11th(self):
        result = normalize_numbers("The 11th hour")
        assert "eleventh" in result

    def test_decimal(self):
        result = normalize_numbers("Pi is 3.14")
        assert "three point one four" in result

    def test_roman_numeral(self):
        result = normalize_numbers("World War II")
        assert "two" in result
        assert "II" not in result

    def test_roman_numeral_iv(self):
        result = normalize_numbers("Chapter IV")
        assert "four" in result

    def test_no_numbers(self):
        assert normalize_numbers("hello world") == "hello world"

    def test_empty_string(self):
        assert normalize_numbers("") == ""

    def test_mixed_text(self):
        result = normalize_numbers("I bought 2 items for $10.50 on the 3rd")
        assert "two" in result
        assert "ten dollars" in result
        assert "fifty cents" in result
        assert "third" in result

    def test_timestamp_tokens_not_present(self):
        """normalize_numbers should NOT be called on text with timestamps,
        but if it were, verify it doesn't crash."""
        result = normalize_numbers("hello world")
        assert result == "hello world"


# ── Cut classifiers ──────────────────────────────────────────────────────────


class _FakeCut:
    """Minimal stand-in for a lhotse Cut with arbitrary attributes."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestIsMcqCutTrain:
    def test_mcq_training_standard(self):
        cut = _FakeCut(shard_origin="MCQ_training_v2")
        assert is_mcq_cut_train(cut) is True

    def test_mcq_training_think(self):
        cut = _FakeCut(shard_origin="MCQ_training_think_v1")
        assert is_mcq_cut_train(cut) is True
        assert is_mcq_cut_train(cut, check_think=True) is True

    def test_mcq_training_no_think_flag(self):
        cut = _FakeCut(shard_origin="MCQ_training_v2")
        assert is_mcq_cut_train(cut, check_think=True) is False

    def test_singleqa_excluded(self):
        cut = _FakeCut(shard_origin="MCQ_training_singleQA_v1")
        assert is_mcq_cut_train(cut) is False

    def test_no_shard_origin(self):
        cut = _FakeCut()
        assert is_mcq_cut_train(cut) is False

    def test_unrelated_shard_origin(self):
        cut = _FakeCut(shard_origin="asr_librispeech")
        assert is_mcq_cut_train(cut) is False


class TestIsMcqCutVal:
    def test_openbookqa(self):
        cut = _FakeCut(shard_origin="openbookqa_val")
        assert is_mcq_cut_val(cut) is True

    def test_mmsu(self):
        cut = _FakeCut(shard_origin="mmsu_test")
        assert is_mcq_cut_val(cut) is True

    def test_bbh(self):
        cut = _FakeCut(shard_origin="bbh_eval")
        assert is_mcq_cut_val(cut) is True

    def test_no_shard_origin(self):
        cut = _FakeCut()
        assert is_mcq_cut_val(cut) is False

    def test_training_origin_not_val(self):
        cut = _FakeCut(shard_origin="MCQ_training_v2")
        assert is_mcq_cut_val(cut) is False


class TestIsAsrCut:
    def test_task_asr(self):
        cut = _FakeCut(task="asr")
        assert is_asr_cut(cut) is True

    def test_formatter_nemo_tarred(self):
        cut = _FakeCut(formatter="nemo_tarred_to_duplex")
        assert is_asr_cut(cut) is True

    def test_both_attributes(self):
        cut = _FakeCut(task="asr", formatter="nemo_tarred_to_duplex")
        assert is_asr_cut(cut) is True

    def test_neither(self):
        cut = _FakeCut(task="s2s_duplex")
        assert is_asr_cut(cut) is False

    def test_no_attributes(self):
        cut = _FakeCut()
        assert is_asr_cut(cut) is False
