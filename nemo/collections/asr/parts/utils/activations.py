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
import torch.nn as nn

__all__ = ['Swish', 'Snake']


@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    equation for snake activation function: x + (alpha + eps)^-1 * sin(alpha * x)^2
    """
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + eps).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake(nn.Module):
    """
    Snake activation function introduced in 'https://arxiv.org/abs/2006.08195'
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


class Swish(nn.SiLU):
    """
    Swish activation function introduced in 'https://arxiv.org/abs/1710.05941'
    Mathematically identical to SiLU. See note in nn.SiLU for references.
    """
