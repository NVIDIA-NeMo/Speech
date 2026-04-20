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

"""
Tests for cpWER calculation in nemo.collections.asr.metrics.der.
All expected values are pre-verified against MeetEval (https://github.com/fgnt/meeteval).
"""

from itertools import permutations
import pytest
import torch

from nemo.collections.asr.metrics.der import (
    calculate_session_cpWER,
    calculate_session_cpWER_bruteforce,
    concat_perm_word_error_rate,
)


def assert_cpwer_equals(cpwer_actual, cpwer_expected, tol=1e-6):
    diff = torch.abs(torch.tensor(cpwer_expected - cpwer_actual))
    assert diff <= tol, f"cpWER mismatch: actual={cpwer_actual:.8f}, expected={cpwer_expected:.8f}"


def permuted_input_test(hyp, ref, expected_cpwer):
    """Verify cpWER is stable across all permutations of hypothesis speakers."""
    for hyp_permed in permutations(hyp):
        cpwer, _, _ = calculate_session_cpWER(spk_hypothesis=list(hyp_permed), spk_reference=ref)
        assert_cpwer_equals(cpwer, expected_cpwer)


def assert_cpwer_with_bruteforce(ref, hyp, expected, tol=1e-6):
    """Assert both LSA and brute-force give the expected cpWER."""
    cpwer_lsa, _, _ = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
    cpwer_bf, _, _ = calculate_session_cpWER_bruteforce(spk_hypothesis=hyp, spk_reference=ref)
    assert_cpwer_equals(cpwer_lsa, expected, tol)
    assert_cpwer_equals(cpwer_bf, expected, tol)


class TestCpWERBasic:
    """Basic cpWER tests migrated from test_diar_metrics.py with MeetEval-verified values."""

    @pytest.mark.unit
    def test_cpwer_oneword(self):
        ref = ["oneword"]

        hyp = ["oneword"]
        assert_cpwer_with_bruteforce(ref, hyp, expected=0.0)
        permuted_input_test(hyp, ref, 0.0)

        hyp = ["wrongword"]
        assert_cpwer_with_bruteforce(ref, hyp, expected=1.0)
        permuted_input_test(hyp, ref, 1.0)

    @pytest.mark.unit
    def test_cpwer_perfect(self):
        hyp = ["ff", "aa bb cc", "dd ee"]
        ref = ["aa bb cc", "dd ee", "ff"]
        cpwer, _, _ = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        assert_cpwer_equals(cpwer, 0.0)
        permuted_input_test(hyp, ref, 0.0)

    @pytest.mark.unit
    def test_cpwer_spk_confusion_and_asr_error(self):
        """MeetEval: errors=5, length=11 → cpWER=5/11."""
        hyp = ["aa bb c ff", "dd e ii jj kk", "hi"]
        ref = ["aa bb cc ff", "dd ee gg jj kk", "hh ii"]
        assert_cpwer_with_bruteforce(ref, hyp, expected=5 / 11)
        permuted_input_test(hyp, ref, 5 / 11)

    @pytest.mark.unit
    def test_cpwer_undercount(self):
        """Hyp has fewer speakers than ref (4 vs 6). MeetEval: errors=3, length=11 → cpWER=3/11."""
        hyp = ["aa bb cc", "dd ee gg", "hh ii", "jj kk"]
        ref = ["aa bb cc", "dd ee", "ff", "gg", "hh ii", "jj kk"]
        assert_cpwer_with_bruteforce(ref, hyp, expected=3 / 11)

    @pytest.mark.unit
    def test_cpwer_overcount(self):
        """Hyp has more speakers than ref (3 vs 2). MeetEval: errors=7, length=11 → cpWER=7/11."""
        hyp = ["aa bb cc", "dd ee gg hh", "ii jj kk"]
        ref = ["aa bb cc", "dd ee ff gg hh ii jj kk"]
        assert_cpwer_with_bruteforce(ref, hyp, expected=7 / 11)


class TestCpWERExpectedValues:
    """
    Verify NeMo cpWER on a wide range of inputs.
    All expected values pre-computed with MeetEval's cp_word_error_rate.
    """

    @pytest.mark.unit
    def test_perfect_match_2spk(self):
        cpwer, _, _ = calculate_session_cpWER(["hello world", "good morning"], ["hello world", "good morning"])
        assert_cpwer_equals(cpwer, 0.0)

    @pytest.mark.unit
    def test_errors_2spk(self):
        """MeetEval: errors=4, length=14 → cpWER=4/14."""
        cpwer, _, _ = calculate_session_cpWER(
            ["hey how are you we that's nice", "i'm good yes hi is your sister"],
            ["hi how are you well that's nice", "i'm good yeah how is your sister"],
        )
        assert_cpwer_equals(cpwer, 4 / 14)

    @pytest.mark.unit
    def test_permuted_speakers(self):
        cpwer, _, _ = calculate_session_cpWER(
            ["one two three", "alpha beta gamma"],
            ["alpha beta gamma", "one two three"],
        )
        assert_cpwer_equals(cpwer, 0.0)

    @pytest.mark.unit
    def test_false_alarm_speaker(self):
        """Hyp has an extra speaker. MeetEval: errors=2, length=4 → cpWER=0.5."""
        cpwer, _, _ = calculate_session_cpWER(["a b", "c d", "e f"], ["a b", "c d"])
        assert_cpwer_equals(cpwer, 0.5)

    @pytest.mark.unit
    def test_missed_speaker(self):
        """Ref has a speaker hyp doesn't produce. MeetEval: errors=2, length=4 → cpWER=0.5."""
        cpwer, _, _ = calculate_session_cpWER(["a", "b"], ["a", "b", "c d"])
        assert_cpwer_equals(cpwer, 0.5)

    @pytest.mark.unit
    def test_single_speaker(self):
        """MeetEval: errors=1, length=4 → cpWER=0.25."""
        cpwer, _, _ = calculate_session_cpWER(["the quick brown dog"], ["the quick brown fox"])
        assert_cpwer_equals(cpwer, 0.25)

    @pytest.mark.unit
    def test_cross_boundary(self):
        """Per-pair scoring prevents edits from crossing speaker boundaries.
        Old NeMo concatenation approach would give cpWER=0 here.
        MeetEval: errors=2, length=4 → cpWER=0.5."""
        cpwer, _, _ = calculate_session_cpWER(["the cat sat", "on"], ["the cat", "sat on"])
        assert_cpwer_equals(cpwer, 0.5)

    @pytest.mark.unit
    def test_cross_boundary_repeated_word(self):
        """MeetEval: errors=2, length=4 → cpWER=0.5."""
        cpwer, _, _ = calculate_session_cpWER(["a b b", "c"], ["a b", "b c"])
        assert_cpwer_equals(cpwer, 0.5)

    @pytest.mark.unit
    def test_empty_hyp_speaker(self):
        """MeetEval: errors=2, length=4 → cpWER=0.5."""
        cpwer, _, _ = calculate_session_cpWER(["hello world", ""], ["hello world", "good morning"])
        assert_cpwer_equals(cpwer, 0.5)

    @pytest.mark.unit
    def test_3spk_with_errors(self):
        """MeetEval: errors=2, length=6 → cpWER=2/6."""
        cpwer, _, _ = calculate_session_cpWER(
            ["alpha x", "gamma delta y", "zeta"],
            ["alpha beta", "gamma delta epsilon", "zeta"],
        )
        assert_cpwer_equals(cpwer, 2 / 6)

    @pytest.mark.unit
    def test_3spk_permuted(self):
        cpwer, _, _ = calculate_session_cpWER(["bird", "cat", "dog fish"], ["cat", "dog fish", "bird"])
        assert_cpwer_equals(cpwer, 0.0)

    @pytest.mark.unit
    def test_3hyp_2ref(self):
        """MeetEval: errors=4, length=4 → cpWER=1.0."""
        cpwer, _, _ = calculate_session_cpWER(
            ["hello", "there general", "kenobi"],
            ["hello there", "general kenobi"],
        )
        assert_cpwer_equals(cpwer, 1.0)

    @pytest.mark.unit
    def test_overestimation(self):
        """From MeetEval docstring: 1 ref, 2 hyp. errors=3, length=2 → cpWER=1.5."""
        cpwer, _, _ = calculate_session_cpWER(["z", "a e f"], ["a b"])
        assert_cpwer_equals(cpwer, 1.5)

    @pytest.mark.unit
    def test_asymmetric_word_counts(self):
        """MeetEval: errors=2, length=11 → cpWER=2/11."""
        cpwer, _, _ = calculate_session_cpWER(
            ["z", "b c d e f g h i j x"],
            ["a", "b c d e f g h i j k"],
        )
        assert_cpwer_equals(cpwer, 2 / 11)

    @pytest.mark.unit
    def test_4hyp_3ref(self):
        """MeetEval: errors=1, length=3 → cpWER=1/3."""
        cpwer, _, _ = calculate_session_cpWER(["b", "c", "d", "a"], ["a", "b", "c"])
        assert_cpwer_equals(cpwer, 1 / 3)


class TestCpWERBruteforceAgreement:
    """Verify Hungarian and brute-force give the same cpWER."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ref, hyp",
        [
            (["a b c", "d e f"], ["a b c", "d e f"]),
            (["a b c", "d e f"], ["d e f", "a b c"]),
            (["a b", "c d"], ["a b", "c d", "e f"]),
            (["a", "b", "c d"], ["a", "b"]),
            (["a b c", "d e f g h"], ["a b x", "d y f g h"]),
        ],
    )
    def test_lsa_matches_bruteforce(self, ref, hyp):
        cpwer_lsa, _, _ = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        cpwer_bf, _, _ = calculate_session_cpWER_bruteforce(spk_hypothesis=hyp, spk_reference=ref)
        assert_cpwer_equals(cpwer_lsa, cpwer_bf)


class TestConcatPermWordErrorRate:
    """Test the batch launcher function."""

    @pytest.mark.unit
    def test_multi_session(self):
        """Session 1: 1/6, Session 2: 1/2 (pre-computed with MeetEval)."""
        refs = [["a b c", "d e f"], ["hello world"]]
        hyps = [["a b x", "d e f"], ["hello earth"]]
        cpwer_values, _, _ = concat_perm_word_error_rate(hyps, refs)
        assert len(cpwer_values) == 2
        assert_cpwer_equals(cpwer_values[0], 1 / 6)
        assert_cpwer_equals(cpwer_values[1], 1 / 2)

    @pytest.mark.unit
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            concat_perm_word_error_rate([["a"]], [["a"], ["b"]])
