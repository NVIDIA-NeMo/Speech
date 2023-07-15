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

import pytest
import torch

from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    get_confidence_aggregation_bank,
    get_confidence_measure_bank,
)

# Initialize probability vectors
VOCAB_SIZES = (100, 1000, 10000)
ONE_VEC_SET, ZERO_VEC_SET, RAND_VEC_SET, OVERFIT_RAND_VEC_SET = {}, {}, {}, {}
for vocab_size in VOCAB_SIZES:
    # batch size 2 to test different positions of probability one
    ONE_VEC_SET[vocab_size] = torch.nan_to_num(
        torch.cat(
            [
                torch.tensor([[0] + [float('-inf')] * (vocab_size - 1)]),
                torch.tensor([[float('-inf')] * (vocab_size - 3) + [0] + [float('-inf')] * 2]),
            ]
        )
    )
    ZERO_VEC_SET[vocab_size] = torch.nan_to_num(torch.tensor([[math.log(1 / vocab_size)] * vocab_size] * 2))
    # batch size 1
    rand_logit = torch.rand((1, vocab_size))
    rand_logit_overfit = rand_logit.clone()
    rand_logit_overfit[0, 0] += vocab_size
    RAND_VEC_SET[vocab_size] = torch.nan_to_num(torch.nn.functional.log_softmax(rand_logit, -1))
    OVERFIT_RAND_VEC_SET[vocab_size] = torch.nan_to_num(torch.nn.functional.log_softmax(rand_logit_overfit, -1))
AGGREGATION_VEC_SIMPLE = [0.0, 0.5, 1]

TOL_DEGREE = 6
TOL = 1 / math.pow(10, TOL_DEGREE)


def get_measure_parametrize_ranges():
    confidence_measure_bank = {}
    alpha_range = (0.25, 0.5, 1.0)
    bank_exception = None
    try:
        confidence_measure_bank = get_confidence_measure_bank()
    except Exception as e:
        alpha_range = ()
        bank_exception = e
    return confidence_measure_bank, alpha_range, bank_exception


def get_aggregation_parametrize_ranges():
    confidence_aggregation_bank = {}
    bank_exception = None
    try:
        confidence_aggregation_bank = get_confidence_aggregation_bank()
    except Exception as e:
        bank_exception = e
    return confidence_aggregation_bank, bank_exception


class TestConfidenceMeasureBank:
    measure_bank, alphas, bank_build_exception = get_measure_parametrize_ranges()

    @pytest.mark.unit
    def test_measure_bank(self):
        if self.bank_build_exception is not None:
            raise self.bank_build_exception

        assert isinstance(self.measure_bank, dict)
        assert len(self.measure_bank) > 0

    @pytest.mark.unit
    @pytest.mark.parametrize('measure_name', measure_bank.keys())
    @pytest.mark.parametrize('alpha', alphas)
    @pytest.mark.parametrize('vocab_size', VOCAB_SIZES)
    def test_confidence_measures_one(self, measure_name, alpha, vocab_size):
        measure = self.measure_bank[measure_name]

        assert torch.allclose(measure(ONE_VEC_SET[vocab_size], vocab_size, alpha), torch.tensor([1.0, 1.0]), atol=TOL)

    @pytest.mark.unit
    @pytest.mark.parametrize('measure_name', measure_bank.keys())
    @pytest.mark.parametrize('alpha', alphas)
    @pytest.mark.parametrize('vocab_size', VOCAB_SIZES)
    def test_confidence_measures_zero(self, measure_name, alpha, vocab_size):
        measure = self.measure_bank[measure_name]

        assert torch.allclose(measure(ZERO_VEC_SET[vocab_size], vocab_size, alpha), torch.tensor([0.0, 0.0]), atol=TOL)

    @pytest.mark.unit
    @pytest.mark.parametrize('measure_name', measure_bank.keys())
    @pytest.mark.parametrize('alpha', alphas)
    @pytest.mark.parametrize('vocab_size', VOCAB_SIZES)
    def test_confidence_measures_partial_order(self, measure_name, alpha, vocab_size):
        measure = self.measure_bank[measure_name]
        value_normal = round(float(measure(RAND_VEC_SET[vocab_size], vocab_size, alpha)[0]), TOL_DEGREE)
        value_overfit = round(float(measure(OVERFIT_RAND_VEC_SET[vocab_size], vocab_size, alpha)[0]), TOL_DEGREE)

        assert 0 <= value_normal < value_overfit <= 1, (
            measure(RAND_VEC_SET[vocab_size], vocab_size, alpha),
            measure(OVERFIT_RAND_VEC_SET[vocab_size], vocab_size, alpha),
        )


class TestConfidenceAggregationBank:
    aggregation_bank, bank_build_exception = get_aggregation_parametrize_ranges()

    @pytest.mark.unit
    def test_aggregation_bank(self):
        if self.bank_build_exception is not None:
            raise self.bank_build_exception

        assert isinstance(self.aggregation_bank, dict)
        assert len(self.aggregation_bank) > 0

    @pytest.mark.unit
    @pytest.mark.parametrize('aggregation_name', aggregation_bank.keys())
    def test_confidence_agregation_simple(self, aggregation_name):
        # alaptev: would skipif work with parametrize arguments?
        if aggregation_name not in ("mean", "min", "max", "prod"):
            pytest.skip(f"{aggregation_name} is not a simple aggregation")
        aggregation = self.aggregation_bank[aggregation_name]
        if aggregation_name == "mean":
            assert aggregation(AGGREGATION_VEC_SIMPLE) == 0.5
        elif aggregation_name == "min":
            assert aggregation(AGGREGATION_VEC_SIMPLE) == 0.0
        if aggregation_name == "max":
            assert aggregation(AGGREGATION_VEC_SIMPLE) == 1.0
        if aggregation_name == "prod":
            assert aggregation(AGGREGATION_VEC_SIMPLE) == 0.0
