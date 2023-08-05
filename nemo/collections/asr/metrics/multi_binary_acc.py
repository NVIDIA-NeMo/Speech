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

import logging

import torch
from torchmetrics import Metric

__all__ = ['MultiBinaryAccuracy']


class MultiBinaryAccuracy(Metric):
    """
    This metric computes accuracies that are needed to evaluate multiple binary outputs.
    For example, if a model returns a set of multiple sigmoid outputs per each sample or at each time step,
    F1 score can be calculated to monitor Type 1 error and Type 2 error together.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            signals, signal_lengths, targets = batch
            preds, _ = self.forward(input_signal=signals,
                                    signal_lengths=signal_lengths,
                                    targets=targets)
            loss = self.loss(logits=preds, labels=targets)
            self._accuracy_valid(preds, targets, signal_lengths)
            f1_acc = self._accuracy.compute()
            self.val_outputs = {'val_loss': loss, 'val_f1_acc': f1_acc}
            return self.val_outputs

        def on_validation_epoch_end(self):
            ...
            val_loss_mean = torch.stack([x['val_loss'] for x in self.val_outputs]).mean()
            correct_counts = torch.stack([x['val_correct_counts'] for x in self.val_outputs]).sum(axis=0)
            total_counts = torch.stack([x['val_total_counts'] for x in self.val_outputs]).sum(axis=0)

            self._accuracy_valid.correct_counts_k = correct_counts
            self._accuracy_valid.total_counts_k = total_counts
            f1_acc = self._accuracy_valid.compute()
            self._accuracy_valid.reset()

            self.log('val_loss', val_loss_mean)
            self.log('val_f1_acc', f1_acc)
            self.val_outputs.clear()  # free memory
            return {'val_loss': val_loss_mean, 'val_f1_acc': f1_acc}

    Args:
        preds (torch.Tensor):
            Predicted values which should be in range of [0, 1].
        targets (torch.Tensor):
            Target values which should be in range of [0, 1].
        signal_lengths (torch.Tensor):
            Length of each sequence in the batch input. signal_lengths values are used to
            filter out zero-padded parts in each sequence.

    Returns:
        f1_score (torch.Tensor):
            F1 score calculated from the predicted value and binarized target values.
    """

    full_state_update = False

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.total_correct_counts = 0
        self.total_sample_counts = 0
        self.true_positive_count = 0
        self.false_positive_count = 0
        self.false_negative_count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, signal_lengths: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            preds_list = [preds[k, : signal_lengths[k], :] for k in range(preds.shape[0])]
            targets_list = [targets[k, : signal_lengths[k], :] for k in range(targets.shape[0])]
            self.preds = torch.cat(preds_list, dim=0)
            self.targets = torch.cat(targets_list, dim=0)

            self.true = self.preds.round().bool() == self.targets.round().bool()
            self.false = self.preds.round().bool() != self.targets.round().bool()
            self.positive = self.preds.round().bool() == 1
            self.negative = self.preds.round().bool() == 0

            self.positive_count = torch.sum(self.preds.round().bool() == True)
            self.true_positive_count += torch.sum(torch.logical_and(self.true, self.positive))
            self.false_positive_count += torch.sum(torch.logical_and(self.false, self.positive))
            self.false_negative_count += torch.sum(torch.logical_and(self.false, self.negative))

            self.total_correct_counts += torch.sum(self.preds.round().bool() == self.targets.round().bool())
            self.total_sample_counts += torch.prod(torch.tensor(self.targets.shape))

    def compute(self):
        """
        Compute F1 score from the accumulated values. Return -1 if the F1 score is NaN.
        """
        self.precision = self.true_positive_count / (self.true_positive_count + self.false_positive_count)
        self.recall = self.true_positive_count / (self.true_positive_count + self.false_negative_count)
        self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        if torch.isnan(self.f1_score):
            logging.warn("self.f1_score contains NaN value. Returning -1 instead of NaN value.")
            self.f1_score = -1
        return self.f1_score
