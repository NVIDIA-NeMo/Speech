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

from typing import Dict, Optional

import torch

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import NeuralType, SpectrogramType


class MixtureConsistencyProjection(NeuralModule):
    """Ensure estimated sources are consistent with the input mixture.
    Note that the input mixture is assume to be a single-channel signal.

    Args:
        weighting: Optional weighting mode for the consistency constraint.
            If `None`, use uniform weighting. If `power`, use the power of the
            estimated source as the weight.
        eps: Small positive value for regularization

    Reference:
        Wisdom et al, Differentiable consistency constraints for improved deep speech enhancement, 2018
    """

    def __init__(self, weighting: Optional[str] = None, eps: float = 1e-8):
        super().__init__()
        self.weighting = weighting
        self.eps = eps

        if self.weighting not in [None, 'power']:
            raise NotImplementedError(f'Weighting mode {self.weighting} not implemented')

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "mixture": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
            "estimate": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        """Returns definitions of module output ports."""
        return {
            "output": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    @typecheck()
    def forward(self, mixture: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        """Enforce mixture consistency on the estimated sources.
        Args:
            mixture: Single-channel mixture, shape (B, 1, F, N)
            estimate: M estimated sources, shape (B, M, F, N)

        Returns:
            Source estimates consistent with the mixture, shape (B, M, F, N)
        """
        if mixture.size(-3) != 1:
            raise ValueError(f'Mixture must have a single channel, got shape {mixture.shape}')

        # number of sources
        M = estimate.size(-3)
        # estimated mixture based on the estimated sources
        estimated_mixture = torch.sum(estimate, dim=-3, keepdim=True)

        # weighting
        if self.weighting is None:
            weight = 1 / M
        elif self.weighting == 'power':
            weight = estimate.abs().pow(2)
            weight = weight / (weight.sum(dim=-3, keepdim=True) + self.eps)
        else:
            raise NotImplementedError(f'Weighting mode {self.weighting} not implemented')

        # consistent estimate
        consistent_estimate = estimate + weight * (mixture - estimated_mixture)

        return consistent_estimate
