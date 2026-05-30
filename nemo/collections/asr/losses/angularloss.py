# ! /usr/bin/python
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

import torch

from nemo.core.classes import Loss, Typing, typecheck
from nemo.core.neural_types import LabelsType, LogitsType, LossType, NeuralType

__all__ = ['AngularSoftmaxLoss']


class AngularSoftmaxLoss(Loss, Typing):
    """
    Computes ArcFace Angular softmax angle loss
    reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf
    args:
    scale: scale value for cosine angle
    margin: margin value added to cosine angle
    """

    @property
    def input_types(self):
        """Input types definitions for AnguarLoss."""
        return {
            "logits": NeuralType(('B', 'D'), LogitsType()),
            "labels": NeuralType(('B',), LabelsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for AngularLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, scale=20.0, margin=1.35):
        super().__init__()

        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

    @typecheck()
    def forward(self, logits, labels):
        numerator = self.scale * torch.cos(
            torch.acos(torch.clamp(torch.diagonal(logits.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps))
            + self.margin
        )
        excl = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
