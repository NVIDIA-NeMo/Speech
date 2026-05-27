# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.core.classes import Loss
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import LabelsType, LogitsType, LossType, NeuralType
from nemo.utils import logging

__all__ = ['MultiSimilarityLoss']


class MultiSimilarityLoss(Loss):
    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {"logits": NeuralType(('B', 'D'), LogitsType()), "labels": NeuralType(('B'), LabelsType())}

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        scale_pos: Optional[float] = 2.0,  # Params found to work best in our experiments
        scale_neg: Optional[float] = 40.0,
        offset: Optional[float] = 0.5,
        margin: Optional[float] = 0.1,
    ):
        super().__init__()
        self._scale_pos = scale_pos
        self._scale_neg = scale_neg
        self._offset = offset
        self._margin = margin
        self._epsilon = 1e-5

    @typecheck()
    def forward(self, logits, labels):
        cos_sim = torch.matmul(logits, torch.t(logits))
        losses = []

        for i in range(logits.size(0)):
            # mine hard pairs relative to anchor i
            positive_sims = cos_sim[i][labels.eq(labels[i])]
            positive_sims = positive_sims[positive_sims.lt(1 - self._epsilon)]  # omit identical pairs
            negative_sims = cos_sim[i][labels.ne(labels[i])]

            if len(negative_sims) == 0 or len(positive_sims) == 0:
                continue

            # negatives that are more similar than the least-similar positive
            hard_negatives = negative_sims[negative_sims.gt(min(positive_sims) - self._margin)]

            # positives that are less similar than the most-similar negative
            hard_positives = positive_sims[positive_sims.lt(max(negative_sims) + self._margin)]

            if len(hard_negatives) == 0 or len(hard_positives) == 0:
                continue

            pos_term = (
                1.0
                / self._scale_pos
                * torch.log(1 + torch.sum(torch.exp(-self._scale_pos * (hard_positives - self._offset))))
            )
            neg_term = (
                1.0
                / self._scale_neg
                * torch.log(1 + torch.sum(torch.exp(self._scale_neg * (hard_negatives - self._offset))))
            )
            losses.append(pos_term + neg_term)

        if len(losses) == 0:
            loss = torch.zeros([], requires_grad=True).cuda()
            logging.info(f'Encountered zero loss in multisimloss, loss = {loss}. No hard examples found in the batch')
        else:
            loss = torch.sum(torch.stack(losses)) / logits.size(0)

        return loss
