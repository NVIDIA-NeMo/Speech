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

from typing import Optional

import torch

from nemo.core.classes import Exportable, Loss, NeuralModule, typecheck
from nemo.core.neural_types import LabelsType, LogprobsType, LossType, MaskType, NeuralType

__all__ = ["SmoothedCrossEntropyLoss", "SmoothedNLLLoss"]


class SmoothedCrossEntropyLoss(Loss):
    """
    Calculates Cross-entropy loss with label smoothing for a batch of sequences.

    SmoothedCrossEntropyLoss:
    1) excludes padding tokens from loss calculation
    2) allows to use label smoothing regularization
    3) allows to calculate loss for the desired number of last tokens
    4) per_token_reduction - if False disables reduction per token

    Args:
        label_smoothing (float): label smoothing regularization coefficient
        predict_last_k (int): parameter which sets the number of last tokens to calculate the loss for, for example
            0: (default) calculate loss on the entire sequence (e.g., NMT)
            1: calculate loss on the last token only (e.g., LM evaluation)
            Intermediate values allow to control the trade-off between eval
            time (proportional to the number of batches) and eval performance
            (proportional to the number of context tokens)
        pad_id (int): padding id
        eps (float): the small eps number to avoid division buy zero
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {
            "log_probs": NeuralType(("B", "T", "D"), LogprobsType()),
            "labels": NeuralType(("B", "T"), LabelsType()),
            "output_mask": NeuralType(("B", "T"), MaskType(), optional=True),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        pad_id: Optional[int] = None,
        label_smoothing: Optional[float] = 0.0,
        predict_last_k: Optional[int] = 0,
        eps: float = 1e-6,
        per_token_reduction: bool = True,
    ):
        super().__init__()
        self._pad_id = pad_id
        self._eps = eps
        self._predict_last_k = predict_last_k
        self._label_smoothing = label_smoothing
        self._per_token_reduction = per_token_reduction

    @typecheck()
    def forward(self, log_probs, labels, output_mask=None):
        """
        Args:
            log_probs: float tensor of shape batch_size x seq_len x vocab_size, values should be log probabilities
            labels: int tensor of shape batch_size x seq_len
            output_mask: binary tensor of shape batch_size x seq_len
            eps: epsilon param to avoid divide by zero in loss calculation
        """
        if output_mask is None and self._pad_id is None:
            raise ValueError("Both output_mask and pad_id are None")
        if output_mask is None and self._pad_id is not None:
            output_mask = (labels != self._pad_id).to(log_probs.dtype)

        if output_mask.dtype is not log_probs.dtype:
            output_mask = output_mask.to(log_probs.dtype)

        batch_size, seq_len, vocab_size = log_probs.size()
        smoothing = vocab_size * self._label_smoothing / (vocab_size - 1)
        target_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)

        smoothing_log_probs = log_probs.mean(dim=-1)
        neg_log_likelihood = (1.0 - smoothing) * target_log_probs + smoothing * smoothing_log_probs
        neg_log_likelihood = neg_log_likelihood[:, -self._predict_last_k :]
        output_mask = output_mask[:, -self._predict_last_k :]

        # when False avoid per token reduction
        if self._per_token_reduction:
            neg_log_likelihood = -torch.sum(neg_log_likelihood * output_mask)
            neg_log_likelihood = neg_log_likelihood / (output_mask.sum() + self._eps)
        else:
            neg_log_likelihood = -(neg_log_likelihood * output_mask)

        return neg_log_likelihood


class SmoothedNLLLoss(NeuralModule, Exportable):
    """
    Calculate negative log likelihodd for sequence input, also applies label smoothing (if set).
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {
            "log_probs": NeuralType(("B", "T", "D"), LogprobsType()),
            "labels": NeuralType(("B", "T"), LabelsType()),
            "output_mask": NeuralType(("B", "T"), MaskType(), optional=True),
            "lengths": NeuralType(("B"), LabelsType(), optional=True),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, reduction='mean', label_smoothing=0.0, eps=1e-8, **kwargs):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.nll_loss = torch.nn.NLLLoss(reduction='none', **kwargs)
        self.eps = eps  # small constant to avoid divide by zero

    @typecheck()
    def forward(self, log_probs, labels, output_mask=None, lengths=None):
        """
        Params:
        -   log_probs: BxTxC
        -   labels: B
        -   output_mask: BxT
        -   lengths: B
        """

        if output_mask is None and lengths is None:
            output_mask = torch.ones_like(log_probs).float()
        elif output_mask is None and lengths is not None:
            output_mask = torch.arange(log_probs.size(1), device=log_probs.device)[None, :] < lengths[:, None]
            output_mask = output_mask.float()

        log_probs = log_probs.transpose(1, 2)  # BxTxC -> BxCxT

        loss = output_mask * self.nll_loss(log_probs, labels)
        batch_size = loss.size(0)
        if self.reduction == "mean":
            loss = loss.sum() / (torch.sum(output_mask) + self.eps)
        elif self.reduction == "batchmean":
            loss = loss.sum() / batch_size
        elif self.reduction == "batch":
            loss = loss.reshape(batch_size, -1).sum(1) / (output_mask.reshape(batch_size, -1).sum(1) + self.eps)

        if self.label_smoothing == 0.0:
            return loss
        else:
            # Regularizing Neural Networks by Penalizing Confident Output Distributions.
            # https://arxiv.org/abs/1701.06548
            loss_reg = torch.mean(log_probs, dim=1) * output_mask
            if self.reduction == "mean":
                loss_reg = torch.sum(loss_reg) / torch.sum(output_mask)
            elif self.reduction == "batchmean":
                loss_reg = torch.sum(loss_reg) / labels.shape[0]
            elif self.reduction == "batch":
                loss_reg = loss_reg.sum(1) / output_mask.sum(1)

            return -self.label_smoothing * loss_reg + (1 - self.label_smoothing) * loss
