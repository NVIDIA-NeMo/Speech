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


class MaskedBatchNorm1d(nn.Module):
    """Batch normalization over (B, D, T) features that excludes frames beyond each sequence's length.

    With ``lengths=None`` it behaves as ``torch.nn.BatchNorm1d`` and carries the same state
    (``weight``, ``bias``, ``running_mean``, ``running_var``, ``num_batches_tracked``),
    so checkpoints move between the two.
    Training normalizes with the statistics of the batch's valid frames per feature
    and moves the running statistics towards them;
    evaluation normalizes with the running statistics.
    A batch without valid frames leaves the running statistics unchanged.

    Args:
        num_features: Size of the feature axis D.
        eps: Added to the variance before the square root.
        momentum: Weight of the current batch in the update of the running statistics;
            None gives the cumulative average over all batches, as in ``torch.nn.BatchNorm1d``.
        affine: Whether to learn a scale and a shift per feature.
        track_running_stats: Whether to keep running statistics and normalize with them in evaluation;
            without them the batch statistics are used in evaluation too.
        unbiased_running_var: Whether the running variance is updated with the unbiased estimate,
            as ``torch.nn.BatchNorm1d`` does;
            False updates it with the biased estimate that also normalizes the batch.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        unbiased_running_var: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.unbiased_running_var = unbiased_running_var
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Normalize padded features.

        Args:
            x: Features with shape (B, D, T), or (N, D) as ``torch.nn.BatchNorm1d`` accepts them.
            lengths: Valid frames per sequence with shape (B,); None counts every frame.

        Returns:
            Normalized features with the shape and dtype of ``x``.
        """
        if lengths is None:
            return self._normalize(x, None)
        if x.dim() != 3:
            raise ValueError(f"{self} needs (B, D, T) features with lengths, got shape {tuple(x.shape)}.")
        mask = torch.arange(x.shape[2], device=x.device)[None, :] < lengths[:, None]
        return self._normalize(x, mask.unsqueeze(1))

    def forward_packed(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize token-flat features, one row per valid frame of the batch.

        Args:
            data: Features with shape (N, D).

        Returns:
            Normalized features with the shape and dtype of ``data``.
        """
        return self._normalize(data, None)

    def _normalize(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        reduce_dims = [dim for dim in range(x.dim()) if dim != 1]
        shape = [1, -1] + [1] * (x.dim() - 2)
        # Half and bfloat16 statistics accumulate in float32; wider inputs keep their precision.
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        if self.training or not self.track_running_stats:
            if mask is None:
                count = values.new_full((), float(values.numel() // values.shape[1]))
                mean = values.mean(dim=reduce_dims)
                var = (values - mean.view(shape)).square().mean(dim=reduce_dims)
            else:
                count = mask.sum().to(values.dtype)
                safe_count = count.clamp_min(1.0)
                mean = torch.where(mask, values, 0.0).sum(dim=reduce_dims) / safe_count
                var = torch.where(mask, values - mean.view(shape), 0.0).square().sum(dim=reduce_dims) / safe_count
            if self.training and self.track_running_stats:
                self._update_running_statistics(mean, var, count)
        else:
            mean, var = self.running_mean, self.running_var
        out = (values - mean.view(shape)) * torch.rsqrt(var.view(shape) + self.eps)
        if self.affine:
            out = out * self.weight.view(shape) + self.bias.view(shape)
        return out.to(x.dtype)

    @torch.no_grad()
    def _update_running_statistics(self, mean: torch.Tensor, var: torch.Tensor, count: torch.Tensor) -> None:
        # The unbiased estimate needs two values; one value or none leaves the running statistics as they are.
        # No .item() on the way except for the cumulative average, so the update also runs under CUDA graph capture.
        if self.unbiased_running_var:
            var = var * (count / (count - 1).clamp_min(1.0))
            valid = count > 1
        else:
            valid = count > 0
        self.num_batches_tracked += valid.to(torch.long)
        if self.momentum is None:
            factor = 1.0 / float(self.num_batches_tracked.clamp_min(1))
        else:
            factor = self.momentum
        self.running_mean += torch.where(valid, (mean - self.running_mean) * factor, 0.0)
        self.running_var += torch.where(valid, (var - self.running_var) * factor, 0.0)


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
