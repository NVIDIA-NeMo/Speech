# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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

from functools import lru_cache

import transformer_engine.pytorch as te
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TELinear
from megatron.core.post_training.modelopt.layers import Linear as MTLinear
from transformer_engine.common.recipe import DelayedScaling, Format


@lru_cache
def get_pad_tensor(*, pad_shape, dtype, device):
    """Get cached pad tensor."""
    return torch.zeros(pad_shape, device=device, dtype=dtype)


def pad_to_multiple(x, multiple=8):
    """Pad tensor to make sequence length divisible by multiple."""
    seq_len, b, d = x.shape
    if seq_len % multiple == 0:
        return x
    pad_len = multiple - (seq_len % multiple)
    pad_shape = (pad_len, b, d)
    pad_tensor = get_pad_tensor(pad_shape=pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad_tensor], dim=0)


def set_format_recipe():
    """Set the fp8 format recipe. for Hyena."""
    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    return fp8_recipe


def fp8_padded_forward(cls, self, x):
    """Forward call that ensures proper padding as required by TE layers."""
    L = x.shape[0]
    x = pad_to_multiple(x)
    with te.fp8_autocast(enabled=True, fp8_recipe=set_format_recipe()):
        x, bias = cls.forward(x)
    if x.shape[0] > L:
        x = x[:L, :, :]
    return x, bias


def rmsnorm(self, x):
    """Plain rmsnorm implementation."""
    assert self.eps == 1e-6, self.eps
    norm = x.norm(2, dim=-1, keepdim=True) * self.in_features ** (-0.5) + self.eps
    return self.layer_norm_weight * x / norm


import torch
import torch.nn as nn


class NoTP:
    """
    Mixin to disallow tensor parallelism > 1 for certain classes.
    Checks for parallel_mode or tensor parallel world size > 1 and raises an
    exception.
    """

    def __init__(self, *args, **kwargs):  # pylint: disable=missing-function-docstring
        parallel_mode = str(kwargs.get("parallel_mode"))
        config = kwargs.get("config")
        tp_world_size = getattr(config, "tensor_model_parallel_size", 1) if config else 1

        if parallel_mode != "None" or tp_world_size > 1:
            raise RuntimeError(
                "This class does not support tensor parallelism (TP > 1). "
                "Set tensor_model_parallel_size=1, or don't use "
                "unfused_rmsnorm and plain_row_linear in the Hyena config."
            )
        super().__init__(*args, **kwargs)


class Linear(NoTP, MTLinear):
    """
    Same as TERowParallelLinear, but doesn't use TE or any parallelism, but
    compatible arguments-wise and checkpoints/sharding-wise.
    """

    def forward(self, x):  # pylint: disable=missing-function-docstring
        bias = None if self._return_bias else self.bias
        x = nn.functional.linear(x, self.weight, bias)
        if self._return_bias:
            return x, self.bias.detach()
        return x, None


class RMSNormLinear(NoTP, TELayerNormColumnParallelLinear):
    """
    RMSNorm + Linear that doesn't use TE or fusion, yet reuses checkpoint
    structure/sharding of TELayerNormColumnParallelLinear.
    """

    def forward(self, x):  # pylint: disable=missing-function-docstring
        x = rmsnorm(self, x)
        return nn.functional.linear(x, self.weight, None), None


class TELinearFp8(NoTP, TELinear):
    """TELinear that internally ensures fp8 padding as required by TE."""

    def forward(self, x):  # pylint: disable=missing-function-docstring
        return fp8_padded_forward(super(), self, x)


class TELayerNormColumnParallelLinearFp8(NoTP, TELayerNormColumnParallelLinear):
    """
    TELayerNormColumnParallelLinearFp8 that internally ensures fp8 padding
    as required by TE.
    """

    def forward(self, x):  # pylint: disable=missing-function-docstring
        return fp8_padded_forward(super(), self, x)


class RMSNormTELinearFp8(TELinearFp8):
    """
    PyTorch-compatible drop-in for TELayerNormColumnParallelLinearFp8.
    Ignores parallelization, but signatures and parameter names are compatible.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        gather_output=False,
        **kwargs,
    ):  # pylint: disable=missing-function-docstring
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode=None,
            skip_weight_param_allocation=False,
            **kwargs,
        )
        config = kwargs["config"]
        self.register_parameter("layer_norm_weight", nn.Parameter(torch.empty(input_size, dtype=config.params_dtype)))
        self.eps = config.layernorm_epsilon

    def forward(self, x):  # pylint: disable=missing-function-docstring
        x = rmsnorm(self, x)
        return super().forward(x)
