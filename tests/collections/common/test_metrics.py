# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import torch

from nemo.collections.common.metrics.classification_accuracy import TopKClassificationAccuracy
from nemo.collections.common.metrics.punct_er import (
    DatasetPunctuationErrorRate,
    OccurancePunctuationErrorRate,
    punctuation_error_rate,
)

from .loss_inputs import ALL_NUM_MEASUREMENTS_ARE_ZERO, NO_ZERO_NUM_MEASUREMENTS, SOME_NUM_MEASUREMENTS_ARE_ZERO
from .perplexity_inputs import NO_PROBS_NO_LOGITS, ONLY_LOGITS1, ONLY_LOGITS100, ONLY_PROBS, PROBS_AND_LOGITS
from .pl_utils import LossTester, PerplexityTester


class TestCommonMetrics:
    top_k_logits = torch.tensor([[0.1, 0.3, 0.2, 0.0], [0.9, 0.6, 0.2, 0.3], [0.2, 0.1, 0.4, 0.3]],)  # 1  # 0  # 2

    @pytest.mark.unit
    def test_top_1_accuracy(self):
        labels = torch.tensor([0, 0, 2], dtype=torch.long)

        accuracy = TopKClassificationAccuracy(top_k=None)
        acc = accuracy(logits=self.top_k_logits, labels=labels)

        assert accuracy.correct_counts_k.shape == torch.Size([1])
        assert accuracy.total_counts_k.shape == torch.Size([1])
        assert abs(acc[0] - 0.667) < 1e-3

    @pytest.mark.unit
    def test_top_1_2_accuracy(self):
        labels = torch.tensor([0, 1, 0], dtype=torch.long)

        accuracy = TopKClassificationAccuracy(top_k=[1, 2])
        top1_acc, top2_acc = accuracy(logits=self.top_k_logits, labels=labels)

        assert accuracy.correct_counts_k.shape == torch.Size([2])
        assert accuracy.total_counts_k.shape == torch.Size([2])

        assert abs(top1_acc - 0.0) < 1e-3
        assert abs(top2_acc - 0.333) < 1e-3

    @pytest.mark.unit
    def test_top_1_accuracy_distributed(self):
        # Simulate test on 2 process DDP execution
        labels = torch.tensor([[0, 0, 2], [2, 0, 0]], dtype=torch.long)

        accuracy = TopKClassificationAccuracy(top_k=None)
        proc1_acc = accuracy(logits=self.top_k_logits, labels=labels[0])
        correct1, total1 = accuracy.correct_counts_k, accuracy.total_counts_k

        accuracy.reset()
        proc2_acc = accuracy(logits=torch.flip(self.top_k_logits, dims=[1]), labels=labels[1])  # reverse logits
        correct2, total2 = accuracy.correct_counts_k, accuracy.total_counts_k

        correct = torch.stack([correct1, correct2])
        total = torch.stack([total1, total2])

        assert correct.shape == torch.Size([2, 1])
        assert total.shape == torch.Size([2, 1])

        assert abs(proc1_acc[0] - 0.667) < 1e-3  # 2/3
        assert abs(proc2_acc[0] - 0.333) < 1e-3  # 1/3

        accuracy.reset()
        accuracy.correct_counts_k = torch.tensor([correct.sum()])
        accuracy.total_counts_k = torch.tensor([total.sum()])
        acc_topk = accuracy.compute()
        acc_top1 = acc_topk[0]

        assert abs(acc_top1 - 0.5) < 1e-3  # 3/6

    @pytest.mark.unit
    def test_top_1_accuracy_distributed_uneven_batch(self):
        # Simulate test on 2 process DDP execution
        accuracy = TopKClassificationAccuracy(top_k=None)

        proc1_acc = accuracy(logits=self.top_k_logits, labels=torch.tensor([0, 0, 2]))
        correct1, total1 = accuracy.correct_counts_k, accuracy.total_counts_k

        proc2_acc = accuracy(
            logits=torch.flip(self.top_k_logits, dims=[1])[:2, :],  # reverse logits, select first 2 samples
            labels=torch.tensor([2, 0]),
        )  # reduce number of labels
        correct2, total2 = accuracy.correct_counts_k, accuracy.total_counts_k

        correct = torch.stack([correct1, correct2])
        total = torch.stack([total1, total2])

        assert correct.shape == torch.Size([2, 1])
        assert total.shape == torch.Size([2, 1])

        assert abs(proc1_acc[0] - 0.667) < 1e-3  # 2/3
        assert abs(proc2_acc[0] - 0.500) < 1e-3  # 1/2

        accuracy.correct_counts_k = torch.tensor([correct.sum()])
        accuracy.total_counts_k = torch.tensor([total.sum()])
        acc_topk = accuracy.compute()
        acc_top1 = acc_topk[0]

        assert abs(acc_top1 - 0.6) < 1e-3  # 3/5


@pytest.mark.parametrize("ddp", [True, False])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize(
    "probs, logits",
    [
        (ONLY_PROBS.probs, ONLY_PROBS.logits),
        (ONLY_LOGITS1.probs, ONLY_LOGITS1.logits),
        (ONLY_LOGITS100.probs, ONLY_LOGITS100.logits),
        (PROBS_AND_LOGITS.probs, PROBS_AND_LOGITS.logits),
        (NO_PROBS_NO_LOGITS.probs, NO_PROBS_NO_LOGITS.logits),
    ],
)
class TestPerplexity(PerplexityTester):
    def test_perplexity(self, ddp, dist_sync_on_step, probs, logits):
        self.run_class_perplexity_test(
            ddp=ddp, probs=probs, logits=logits, dist_sync_on_step=dist_sync_on_step,
        )


@pytest.mark.parametrize("ddp", [True, False])
@pytest.mark.parametrize("dist_sync_on_step", [True, False])
@pytest.mark.parametrize("take_avg_loss", [True, False])
@pytest.mark.parametrize(
    "loss_sum_or_avg, num_measurements",
    [
        (NO_ZERO_NUM_MEASUREMENTS.loss_sum_or_avg, NO_ZERO_NUM_MEASUREMENTS.num_measurements),
        (SOME_NUM_MEASUREMENTS_ARE_ZERO.loss_sum_or_avg, SOME_NUM_MEASUREMENTS_ARE_ZERO.num_measurements),
        (ALL_NUM_MEASUREMENTS_ARE_ZERO.loss_sum_or_avg, ALL_NUM_MEASUREMENTS_ARE_ZERO.num_measurements),
    ],
)
class TestLoss(LossTester):
    def test_loss(self, ddp, dist_sync_on_step, loss_sum_or_avg, num_measurements, take_avg_loss):
        self.run_class_loss_test(
            ddp=ddp,
            loss_sum_or_avg=loss_sum_or_avg,
            num_measurements=num_measurements,
            dist_sync_on_step=dist_sync_on_step,
            take_avg_loss=take_avg_loss,
        )


class TestPunctuationErrorRate:
    reference = "Hi, dear! Nice to see you. What's"
    hypothesis = "Hi dear! Nice to see you! What's?"
    punctuation_marks = [".", ",", "!", "?"]

    operation_amounts = {
        '.': {'Correct': 0, 'Deletions': 0, 'Insertions': 0, 'Substitutions': 1},
        ',': {'Correct': 0, 'Deletions': 1, 'Insertions': 0, 'Substitutions': 0},
        '!': {'Correct': 1, 'Deletions': 0, 'Insertions': 0, 'Substitutions': 0},
        '?': {'Correct': 0, 'Deletions': 0, 'Insertions': 1, 'Substitutions': 0},
    }
    substitution_amounts = {
        '.': {'.': 0, ',': 0, '!': 1, '?': 0},
        ',': {'.': 0, ',': 0, '!': 0, '?': 0},
        '!': {'.': 0, ',': 0, '!': 0, '?': 0},
        '?': {'.': 0, ',': 0, '!': 0, '?': 0},
    }
    correct_rate = 0.25
    deletions_rate = 0.25
    insertions_rate = 0.25
    substitutions_rate = 0.25
    punct_er = 0.75
    operation_rates = {
        '.': {'Correct': 0.0, 'Deletions': 0.0, 'Insertions': 0.0, 'Substitutions': 1.0},
        ',': {'Correct': 0.0, 'Deletions': 1.0, 'Insertions': 0.0, 'Substitutions': 0.0},
        '!': {'Correct': 1.0, 'Deletions': 0.0, 'Insertions': 0.0, 'Substitutions': 0.0},
        '?': {'Correct': 0.0, 'Deletions': 0.0, 'Insertions': 1.0, 'Substitutions': 0.0},
    }
    substitution_rates = {
        '.': {'.': 0.0, ',': 0.0, '!': 1.0, '?': 0.0},
        ',': {'.': 0.0, ',': 0.0, '!': 0.0, '?': 0.0},
        '!': {'.': 0.0, ',': 0.0, '!': 0.0, '?': 0.0},
        '?': {'.': 0.0, ',': 0.0, '!': 0.0, '?': 0.0},
    }

    @pytest.mark.unit
    def test_punctuation_error_rate(self):
        assert punctuation_error_rate([self.reference], [self.hypothesis], self.punctuation_marks) == self.punct_er

    @pytest.mark.unit
    def test_OccurancePunctuationErrorRate(self):
        oper_obj = OccurancePunctuationErrorRate(self.punctuation_marks)
        operation_amounts, substitution_amounts, punctuation_rates = oper_obj.compute(self.reference, self.hypothesis)

        assert operation_amounts == self.operation_amounts
        assert substitution_amounts == self.substitution_amounts
        assert punctuation_rates.correct_rate == self.correct_rate
        assert punctuation_rates.deletions_rate == self.deletions_rate
        assert punctuation_rates.insertions_rate == self.insertions_rate
        assert punctuation_rates.substitutions_rate == self.substitutions_rate
        assert punctuation_rates.punct_er == self.punct_er
        assert punctuation_rates.operation_rates == self.operation_rates
        assert punctuation_rates.substitution_rates == self.substitution_rates

    @pytest.mark.unit
    def test_DatasetPunctuationErrorRate(self):
        dper_obj = DatasetPunctuationErrorRate([self.reference], [self.hypothesis], self.punctuation_marks)
        dper_obj.compute()

        assert dper_obj.correct_rate == self.correct_rate
        assert dper_obj.deletions_rate == self.deletions_rate
        assert dper_obj.insertions_rate == self.insertions_rate
        assert dper_obj.substitutions_rate == self.substitutions_rate
        assert dper_obj.punct_er == self.punct_er
        assert dper_obj.operation_rates == self.operation_rates
        assert dper_obj.substitution_rates == self.substitution_rates
