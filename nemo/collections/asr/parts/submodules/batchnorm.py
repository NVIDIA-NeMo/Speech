# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from __future__ import annotations

from functools import reduce
from typing import List

import torch
import torch.nn as nn


class FusedBatchNorm1d(nn.Module):
    """
    Fused BatchNorm to use in Conformer to improve accuracy in finetuning with TTS scenario
    Drop-in replacement for BatchNorm1d with simple affine projection
    """

    def __init__(self, num_features: int):
        """
        Args:
            num_features: number of channels, see original BatchNorm1d documentation
        """
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    @classmethod
    def from_batchnorm(cls, bn: nn.BatchNorm1d) -> FusedBatchNorm1d:
        """
        Construct FusedBatchNorm1d module from BatchNorm1d
        Args:
            bn: original BatchNorm module

        Returns:
            FusedBatchNorm1d module with initialized params; in eval mode result is equivalent to original BatchNorm
        """
        assert isinstance(bn, nn.BatchNorm1d)
        fused_bn = FusedBatchNorm1d(bn.num_features)
        # init projection params from original batch norm
        # so, for inference mode output is the same
        std = torch.sqrt(bn.running_var.data + bn.eps)
        fused_bn.weight.data = bn.weight.data / std
        fused_bn.bias.data = bn.bias.data - bn.running_mean.data * fused_bn.weight.data
        return fused_bn

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            return x * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1)
        assert x.dim() == 2
        return x * self.weight + self.bias


def _get_module_by_name(module: nn.Module, full_layer_name: str) -> nn.Module:
    names = full_layer_name.split(sep='.')
    return reduce(getattr, names, module)


def replace_bn_with_fused_bn(module: nn.Module, full_layer_name: str):
    """
    Replace BatchNorm1d named `full_layer_name` in nn.Module with FusedBatchNorm1d
    Args:
        module: nn.Module instance, modified inplace
        full_layer_name: name of BatchNorm1d submodule in module to replace
    """
    bn = _get_module_by_name(module, full_layer_name)
    assert isinstance(bn, nn.BatchNorm1d)
    fused_bn = FusedBatchNorm1d.from_batchnorm(bn)
    try:
        parent_name, norm_name = full_layer_name.rsplit(".", maxsplit=1)
        setattr(_get_module_by_name(module, parent_name), norm_name, fused_bn)
    except ValueError:
        norm_name = full_layer_name
        setattr(module, norm_name, fused_bn)


def replace_bn_with_fused_bn_all(model: nn.Module) -> List[str]:
    """
    Replace BatchNorm1d with FusedBatchNorm1d in model
    Args:
        model: nn.Module instance, modified inplace

    Returns:
        list of replaced module names
    """
    replaced_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            replace_bn_with_fused_bn(model, name)
            replaced_module_names.append(name)
    return replaced_module_names
