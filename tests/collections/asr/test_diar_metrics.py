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

from itertools import permutations

import pytest
import torch

from nemo.collections.asr.metrics.der import (
    calculate_session_cpWER,
    calculate_session_cpWER_bruteforce,
    get_online_DER_stats,
    get_partial_ref_labels,
)


def word_count(spk_transcript):
    return sum([len(w.split()) for w in spk_transcript])


def calculate_wer_count(_ins, _del, _sub, ref_word_count):
    return (_ins + _del + _sub) / ref_word_count


def permuted_input_test(hyp, ref, calculated):
    """
    Randomly permute the input to see if evaluation result stays the same.
    """
    for hyp_permed in permutations(hyp):
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp_permed, spk_reference=ref)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6


class TestConcatMinPermWordErrorRate:
    """
    Tests for cpWER calculation.
    """

    @pytest.mark.unit
    def test_cpwer_oneword(self):
        hyp = ["oneword"]
        ref = ["oneword"]
        _ins, _del, _sub = 0, 0, 0
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)
        cpWER_perm, hyp_min_perm, ref_str = calculate_session_cpWER_bruteforce(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

        # Test with a substitution
        hyp = ["wrongword"]
        _ins, _del, _sub = 0, 0, 1
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)
        cpWER_perm, hyp_min_perm, ref_str = calculate_session_cpWER_bruteforce(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_cpwer_perfect(self):
        hyp = ["ff", "aa bb cc", "dd ee"]
        ref = ["aa bb cc", "dd ee", "ff"]
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        calculated = 0
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)

    @pytest.mark.unit
    def test_cpwer_spk_counfusion_and_asr_error(self):
        hyp = ["aa bb c ff", "dd e ii jj kk", "hi"]
        ref = ["aa bb cc ff", "dd ee gg jj kk", "hh ii"]
        _ins, _del, _sub = 0, 1, 4
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)
        cpWER_perm, hyp_min_perm, ref_str = calculate_session_cpWER_bruteforce(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_cpwer_undercount(self):
        hyp = ["aa bb cc", "dd ee gg", "hh ii", "jj kk"]
        ref = ["aa bb cc", "dd ee", "ff", "gg", "hh ii", "jj kk"]
        _ins, _del, _sub = 0, 1, 0
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        cpWER_perm, hyp_min_perm, ref_str = calculate_session_cpWER_bruteforce(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_cpwer_overcount(self):
        hyp = ["aa bb cc", "dd ee gg hh", "ii jj kk"]
        ref = ["aa bb cc", "dd ee ff gg hh ii jj kk"]
        _ins, _del, _sub = 0, 1, 0
        cpWER, hyp_min, ref_str = calculate_session_cpWER(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        cpWER_perm, hyp_min_perm, ref_str = calculate_session_cpWER_bruteforce(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

    @pytest.mark.parametrize(
        "pred_labels, ref_labels, expected_output",
        [
            ([], [], []),
            (["0.0 1.0 speaker1"], [], []),
            (["0.0 1.0 speaker1"], ["0.0 1.5 speaker1"], ["0.0 1.0 speaker1"]),
            (["0.1 0.4 speaker1", "0.5 1.0 speaker2"], ["0.0 1.5 speaker1"], ["0.0 1.0 speaker1"]),
            (
                ["0.5 1.0 speaker2", "0.1 0.4 speaker1"],
                ["0.0 1.5 speaker1"],
                ["0.0 1.0 speaker1"],
            ),  # Order of prediction does not matter
            (
                ["0.1 1.4 speaker1", "0.5 1.0 speaker2"],
                ["0.0 1.5 speaker1"],
                ["0.0 1.4 speaker1"],
            ),  # Overlapping prediction
            (
                ["0.1 0.6 speaker1", "0.2 1.5 speaker2"],
                ["0.5 1.0 speaker1", "1.01 2.0 speaker2"],
                ["0.5 1.0 speaker1", "1.01 1.5 speaker2"],
            ),
            (
                ["0.0 2.0 speaker1"],
                ["0.0 2.0 speaker1", "1.0 3.0 speaker2", "0.0 5.0 speaker3"],
                ["0.0 2.0 speaker1", "1.0 2.0 speaker2", "0.0 2.0 speaker3"],
            ),
        ],
    )
    def test_get_partial_ref_labels(self, pred_labels, ref_labels, expected_output):
        assert get_partial_ref_labels(pred_labels, ref_labels) == expected_output

    @pytest.mark.parametrize(
        "DER, CER, FA, MISS, diar_eval_count, der_stat_dict, deci, expected_der_dict, expected_der_stat_dict",
        [
            (
                0.3,
                0.1,
                0.05,
                0.15,
                1,
                {"cum_DER": 0, "cum_CER": 0, "avg_DER": 0, "avg_CER": 0, "max_DER": 0, "max_CER": 0},
                3,
                {"DER": 30.0, "CER": 10.0, "FA": 5.0, "MISS": 15.0},
                {"cum_DER": 0.3, "cum_CER": 0.1, "avg_DER": 30.0, "avg_CER": 10.0, "max_DER": 30.0, "max_CER": 10.0},
            ),
            (
                0.1,
                0.2,
                0.03,
                0.07,
                2,
                {"cum_DER": 0.3, "cum_CER": 0.3, "avg_DER": 15.0, "avg_CER": 15.0, "max_DER": 30.0, "max_CER": 10.0},
                2,
                {"DER": 10.0, "CER": 20.0, "FA": 3.0, "MISS": 7.0},
                {"cum_DER": 0.4, "cum_CER": 0.5, "avg_DER": 20.0, "avg_CER": 25.0, "max_DER": 30.0, "max_CER": 20.0},
            ),
        ],
    )
    def test_get_online_DER_stats(
        self, DER, CER, FA, MISS, diar_eval_count, der_stat_dict, deci, expected_der_dict, expected_der_stat_dict
    ):
        actual_der_dict, actual_der_stat_dict = get_online_DER_stats(
            DER, CER, FA, MISS, diar_eval_count, der_stat_dict, deci
        )
        assert actual_der_dict == expected_der_dict
        assert actual_der_stat_dict == expected_der_stat_dict
