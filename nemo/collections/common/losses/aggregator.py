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

from typing import List

import torch

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LossType, NeuralType

__all__ = ['AggregatorLoss']


class AggregatorLoss(Loss):
    """
    Sums several losses into one.

    Args:
        num_inputs: number of input losses
        weights: a list of coefficient for merging losses
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        input_types = {}
        for i in range(self._num_losses):
            input_types["loss_" + str(i + 1)] = NeuralType(elements_type=LossType())

        return input_types

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_inputs: int = 2, weights: List[float] = None):
        super().__init__()
        self._num_losses = num_inputs
        if weights is not None and len(weights) != num_inputs:
            raise ValueError("Length of weights should be equal to the number of inputs (num_inputs)")

        self._weights = weights

    @typecheck()
    def forward(self, **kwargs):
        """ """
        values = [kwargs[x] for x in sorted(kwargs.keys())]
        loss = torch.zeros_like(values[0])
        for loss_idx, loss_value in enumerate(values):
            if self._weights is not None:
                loss = loss.add(loss_value, alpha=self._weights[loss_idx])
            else:
                loss = loss.add(loss_value)
        return loss
