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

from typing import Any, Dict, Optional, Tuple

import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import METRIC_EPS

__all__ = ['ClassificationReport', 'MultiLabelClassificationReport']


class ClassificationReport(Metric):
    """
    This metric computes the number of True Positive, False Negative, and False Positive examples per class.
    When doing distributed training/evaluation the result of res=ClassificationReport(predictions, labels) calls
    will be all-reduced between all workers using SUM operations.

    If used with PytorchLightning LightningModule, include TPs, FNs, and FPs inside validation_step results.
    Then aggregate them at the end of validation epoch to correctly compute validation precision, recall, f1
    using get_precision_recall_f1().

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            tp, fn, fp, _ = self.classification_report(preds, labels)

            return {'val_loss': val_loss, 'tp': tp, 'fn': fn, 'fp': fp}

        def on_validation_epoch_end(self):
            ...
            # calculate metrics and classification report
            precision, recall, f1, report = self.classification_report.compute()

            logging.info(report)

            self.log('val_loss', avg_loss, prog_bar=True)
            self.log('precision', precision)
            self.log('f1', f1)
            self.log('recall', recall)

    Args:
        num_classes: number of classes in the dataset
        label_ids (optional): label name to label id mapping
        mode: how to compute the average
        dist_sync_on_step: sync across ddp
        process_group: which processes to sync across
    Return:
        aggregated precision, recall, f1, report
    """

    full_state_update = True

    def __init__(
        self,
        num_classes: int,
        label_ids: Dict[str, int] = None,
        mode: str = 'macro',
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.num_classes = num_classes
        if label_ids:
            self.ids_to_labels = {v: k for k, v in label_ids.items()}
        else:
            self.ids_to_labels = None
        self.mode = mode

        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx='sum', persistent=False)
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx='sum', persistent=False)
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx='sum', persistent=False)
        self.add_state(
            "num_examples_per_class", default=torch.zeros(num_classes), dist_reduce_fx='sum', persistent=False
        )

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Updates attributes needed for new classification report (true positive, false negative, false postive, examples per class)

        Args:
            predictions: predicted labels 
            labels: actual labels

        Return:
            None
        """
        TP = []
        FN = []
        FP = []
        for label_id in range(self.num_classes):
            current_label = labels == label_id
            label_predicted = predictions == label_id

            TP.append((label_predicted == current_label)[label_predicted].sum())
            FP.append((label_predicted != current_label)[label_predicted].sum())
            FN.append((label_predicted != current_label)[current_label].sum())

        tp = torch.tensor(TP).to(predictions.device)
        fn = torch.tensor(FN).to(predictions.device)
        fp = torch.tensor(FP).to(predictions.device)
        num_examples_per_class = tp + fn

        self.tp += tp
        self.fn += fn
        self.fp += fp
        self.num_examples_per_class += num_examples_per_class

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aggregates and then calculates logs classification report similar to sklearn.metrics.classification_report.
        Typically used during epoch_end.

        Return:
            aggregated precision, recall, f1, report
        """
        total_examples = torch.sum(self.num_examples_per_class)
        num_non_empty_classes = torch.nonzero(self.num_examples_per_class).size(0)

        precision = torch.true_divide(self.tp * 100, (self.tp + self.fp + METRIC_EPS))
        recall = torch.true_divide(self.tp * 100, (self.tp + self.fn + METRIC_EPS))
        f1 = torch.true_divide(2 * precision * recall, (precision + recall + METRIC_EPS))

        report = '\n{:50s}   {:10s}   {:10s}   {:10s}   {:10s}'.format('label', 'precision', 'recall', 'f1', 'support')
        for i in range(len(self.tp)):
            label = f'label_id: {i}'
            if self.ids_to_labels and i in self.ids_to_labels:
                label = f'{self.ids_to_labels[i]} ({label})'

            report += '\n{:50s}   {:8.2f}   {:8.2f}   {:8.2f}   {:8.0f}'.format(
                label, precision[i], recall[i], f1[i], self.num_examples_per_class[i]
            )

        micro_precision = torch.true_divide(torch.sum(self.tp) * 100, torch.sum(self.tp + self.fp) + METRIC_EPS)
        micro_recall = torch.true_divide(torch.sum(self.tp) * 100, torch.sum(self.tp + self.fn) + METRIC_EPS)
        micro_f1 = torch.true_divide(2 * micro_precision * micro_recall, (micro_precision + micro_recall + METRIC_EPS))

        macro_precision = torch.sum(precision) / num_non_empty_classes
        macro_recall = torch.sum(recall) / num_non_empty_classes
        macro_f1 = torch.sum(f1) / num_non_empty_classes
        weighted_precision = torch.sum(precision * self.num_examples_per_class) / total_examples
        weighted_recall = torch.sum(recall * self.num_examples_per_class) / total_examples
        weighted_f1 = torch.sum(f1 * self.num_examples_per_class) / total_examples

        report += "\n-------------------"

        report += '\n{:50s}   {:8.2f}   {:8.2f}   {:8.2f}   {:8.0f}'.format(
            'micro avg', micro_precision, micro_recall, micro_f1, total_examples
        )

        report += '\n{:50s}   {:8.2f}   {:8.2f}   {:8.2f}   {:8.0f}'.format(
            'macro avg', macro_precision, macro_recall, macro_f1, total_examples
        )
        report += (
            '\n{:50s}   {:8.2f}   {:8.2f}   {:8.2f}   {:8.0f}'.format(
                'weighted avg', weighted_precision, weighted_recall, weighted_f1, total_examples
            )
            + '\n'
        )

        self.total_examples = total_examples

        if self.mode == 'macro':
            return macro_precision, macro_recall, macro_f1, report
        elif self.mode == 'weighted':
            return weighted_precision, weighted_recall, weighted_f1, report
        elif self.mode == 'micro':
            return micro_precision, micro_recall, micro_f1, report
        elif self.mode == 'all':
            return precision, recall, f1, report
        else:
            raise ValueError(
                f'{self.mode} mode is not supported. Choose "macro" to get aggregated numbers \
            or "all" to get values for each class.'
            )


class MultiLabelClassificationReport(ClassificationReport):
    """
    This metric computes the number of True Positive, False Negative, and False Positive examples per class for
    a multi-label dataset.
    
    When doing distributed training/evaluation the result of res=ClassificationReport(predictions, labels) calls
    will be all-reduced between all workers using SUM operations.

    If used with PytorchLightning LightningModule, include TPs, FNs, and FPs inside validation_step results.
    Then aggregate them at the end of validation epoch to correctly compute validation precision, recall, f1
    using get_precision_recall_f1().

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            tp, fn, fp, _ = self.classification_report(preds, labels)

            return {'val_loss': val_loss, 'tp': tp, 'fn': fn, 'fp': fp}

        def on_validation_epoch_end(self):
            ...
            # calculate metrics and classification report
            precision, recall, f1, report = self.classification_report.compute()

            logging.info(report)

            self.log('val_loss', avg_loss, prog_bar=True)
            self.log('precision', precision)
            self.log('f1', f1)
            self.log('recall', recall)

    Args:
        num_classes: number of classes in the dataset
        label_ids (optional): label name to label id mapping
        mode: how to compute the average
        dist_sync_on_step: sync across ddp
        process_group: which processes to sync across
    Return:
        aggregated precision, recall, f1, report
    """

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Updates attributes needed for new classification report (true positive, false negative, false postive, examples per class)

        Args:
            predictions: predicted labels 
            labels: actual labels

        Return:
            None
        """
        predictions = predictions.t()

        TP = []
        FN = []
        FP = []

        for label_id in range(self.num_classes):
            current_label = labels[label_id]
            labels_predicted = predictions[label_id]
            TP.append((labels_predicted == current_label)[labels_predicted == 1].sum())
            FP.append((labels_predicted != current_label)[labels_predicted == 1].sum())
            FN.append((labels_predicted != current_label)[current_label == 1].sum())

        tp = torch.tensor(TP).to(predictions.device)
        fn = torch.tensor(FN).to(predictions.device)
        fp = torch.tensor(FP).to(predictions.device)
        num_examples_per_class = tp + fn

        self.tp += tp
        self.fn += fn
        self.fp += fp
        self.num_examples_per_class += num_examples_per_class
