# ! /usr/bin/python
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from typing import Dict, Optional, Sequence, Union

import torch

from nemo.core.classes import Loss, Typing, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogitsType, LossType, NeuralType, ProbsType

__all__ = ['BCELoss', 'BCEWithLogitsLoss']


class BCELoss(Loss, Typing):
    """Compute Binary Cross Entropy (BCE) from speaker probabilities.

    Frames at or beyond each value in ``target_lens`` are excluded before the
    underlying PyTorch loss is evaluated in FP32.
    """

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Input type definitions for Binary Cross Entropy loss."""
        return {
            "probs": NeuralType(('B', 'T', 'C'), ProbsType()),
            'labels': NeuralType(('B', 'T', 'C'), LabelsType()),
            "target_lens": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Output type definition for Binary Cross Entropy loss."""
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        reduction: str = 'mean',
        weight: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    ) -> None:
        """Initialize the probability-based BCE loss.

        Args:
            reduction: Reduction applied by ``torch.nn.BCELoss``. Supported
                values are ``"none"``, ``"mean"``, and ``"sum"``.
            weight: Optional element-wise weight broadcastable to the
                concatenated valid-frame tensors. List-like values are
                converted to floating-point tensors.
        """
        super().__init__()
        if weight is not None and not torch.is_tensor(weight):
            weight = torch.as_tensor(weight, dtype=torch.float)
        self.loss_f = torch.nn.BCELoss(weight=weight, reduction=reduction)

    @typecheck()
    def forward(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        target_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate probability-based BCE after excluding padded frames.

        Args:
            probs: Speaker probabilities with shape ``(B, T, C)``.
            labels: Binary speaker targets with shape ``(B, T, C)``.
            target_lens: Number of valid temporal frames per batch item with
                shape ``(B,)``.

        Returns:
            The BCE loss with the configured reduction.
        """
        probs = torch.cat([probs[k, : target_lens[k], :] for k in range(probs.shape[0])], dim=0)
        labels = torch.cat([labels[k, : target_lens[k], :] for k in range(labels.shape[0])], dim=0)
        with torch.autocast(device_type=probs.device.type, enabled=False):
            loss = self.loss_f(probs.float(), labels.float())
        return loss


class BCEWithLogitsLoss(Loss, Typing):
    """Compute numerically stable Binary Cross Entropy from speaker logits.

    Frames at or beyond each value in ``target_lens`` are excluded before the
    fused logits-based PyTorch loss is evaluated in FP32.
    """

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Input type definitions for Binary Cross Entropy with logits loss."""
        return {
            "logits": NeuralType(('B', 'T', 'C'), LogitsType()),
            'labels': NeuralType(('B', 'T', 'C'), LabelsType()),
            "target_lens": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Output type definition for Binary Cross Entropy with logits loss."""
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        reduction: str = 'mean',
        pos_weight: Optional[Union[torch.Tensor, float, Sequence[float]]] = None,
    ) -> None:
        """Initialize the fused logits-based BCE loss.

        Args:
            reduction: Reduction applied by ``torch.nn.BCEWithLogitsLoss``.
                Supported values are ``"none"``, ``"mean"``, and ``"sum"``.
            pos_weight: Optional weight applied to positive examples.
                ``None`` is unweighted and adds no persistent state. Non-tensor
                values are converted to floating-point tensors.
        """
        super().__init__()
        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float)
        self.loss_f = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    @typecheck()
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate logits-based BCE after excluding padded frames.

        Args:
            logits: Raw speaker logits with shape ``(B, T, C)``.
            labels: Binary speaker targets with shape ``(B, T, C)``.
            target_lens: Number of valid temporal frames per batch item with
                shape ``(B,)``.

        Returns:
            The fused BCE-with-logits loss with the configured reduction.
        """
        logits = torch.cat([logits[k, : target_lens[k], :] for k in range(logits.shape[0])], dim=0)
        labels = torch.cat([labels[k, : target_lens[k], :] for k in range(labels.shape[0])], dim=0)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            loss = self.loss_f(logits.float(), labels.float())
        return loss
