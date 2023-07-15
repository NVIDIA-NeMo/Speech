# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import math
import tempfile

import numpy as np
import pytest
from scipy.stats import uniform

from nemo.collections.asr.parts.utils.confidence_metrics import (
    auc_nt,
    auc_pr,
    auc_roc,
    auc_yc,
    ece,
    nce,
    save_confidence_hist,
    save_custom_confidence_curve,
    save_nt_curve,
    save_pr_curve,
    save_roc_curve,
)

# set convenient name2metric mapping
name2metric = {
    f.__name__: (f, ans)
    for f, ans in zip((auc_roc, auc_pr, auc_nt, auc_yc, ece, nce), (0.833, 0.917, 0.833, 0.421, 0.232, 0.403))
}
# ece does not have a default value
name2metric_all_correct = {
    f.__name__: (f, ans) for f, ans in zip((auc_roc, auc_pr, auc_nt, auc_yc, nce), (0.5, 1.0, 0.0, 0.0, -math.inf))
}
name2metric_all_incorrect = {
    f.__name__: (f, ans) for f, ans in zip((auc_roc, auc_pr, auc_nt, auc_yc, nce), (0.5, 0.0, 1.0, 0.0, -math.inf))
}

# Initialize data
Y_TRUE = [1, 0, 0, 1, 1]
Y_TRUE_ALL_CORRECT = [1, 1, 1, 1, 1]
Y_TRUE_ALL_INCORRECT = [0, 0, 0, 0, 0]
Y_SCORE = [0.6, 0.7, 0.02, 0.95, 0.8]
Y_TRUE_RANDOM = np.random.choice(2, 1000, p=[0.2, 0.8])
# probability distribution with mean ~= 0.65 and std ~= 0.25
Y_SCORE_RANDOM = uniform.rvs(size=1000, loc=0.5, scale=0.5) - 0.5 * np.random.choice(2, 1000, p=[0.8, 0.2])

TOL_DEGREE = 3
TOL = 1 / math.pow(10, TOL_DEGREE)


class TestConfidenceMetrics:
    @pytest.mark.unit
    @pytest.mark.parametrize('metric_name', name2metric.keys())
    def test_metric_main(self, metric_name):
        metric, ans = name2metric[metric_name]

        assert round(metric(Y_TRUE, Y_SCORE), TOL_DEGREE) == ans

    @pytest.mark.unit
    @pytest.mark.parametrize('metric_name', name2metric_all_correct.keys())
    def test_metric_all_correct(self, metric_name):
        metric, ans = name2metric_all_correct[metric_name]

        assert round(metric(Y_TRUE_ALL_CORRECT, Y_SCORE), TOL_DEGREE) == ans

    @pytest.mark.unit
    @pytest.mark.parametrize('metric_name', name2metric_all_incorrect.keys())
    def test_metric_all_incorrect(self, metric_name):
        metric, ans = name2metric_all_incorrect[metric_name]

        assert round(metric(Y_TRUE_ALL_INCORRECT, Y_SCORE), TOL_DEGREE) == ans

    @pytest.mark.unit
    def test_metric_auc_yc_aux(self):
        n_bins = 10
        result, result_std, result_max, (thresholds, yc_curve) = auc_yc(
            Y_TRUE, Y_SCORE, n_bins=n_bins, return_std_maximum=True, return_curve=True
        )

        assert round(result_std, TOL_DEGREE) == 0.228
        assert round(result_max, TOL_DEGREE) == 0.667
        assert np.allclose(np.array(thresholds), np.array([i / n_bins for i in range(0, n_bins + 1)]), atol=TOL)
        assert np.allclose(
            np.array(yc_curve), np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.167, 0.667, 0.667, 0.333, 0.0]), atol=TOL
        )


class TestSaveConfidencePlot:
    @pytest.mark.unit
    def test_save_confidence_hist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_confidence_hist(Y_SCORE_RANDOM, tmpdir)

    @pytest.mark.unit
    @pytest.mark.parametrize('plot_func', (save_roc_curve, save_pr_curve, save_nt_curve))
    def test_save_simple_confidence_curve(self, plot_func):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_func(Y_TRUE_RANDOM, Y_SCORE_RANDOM, tmpdir)

    @pytest.mark.unit
    def test_save_custom_confidence_curve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ranges = np.arange(0, 1, 0.01)
            save_custom_confidence_curve(ranges, ranges, tmpdir)
