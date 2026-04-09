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

import pytest

from nemo.collections.asr.metrics.der import get_online_DER_stats, get_partial_ref_labels


class TestDiarMetrics:
    """Tests for DER-related utility functions (cpWER tests are in test_cpwer.py)."""

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
