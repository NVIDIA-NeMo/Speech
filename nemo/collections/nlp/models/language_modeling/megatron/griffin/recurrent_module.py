# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import math
from dataclasses import dataclass
from typing import Union

import einops
import torch
from accelerated_scan.ref import scan
from causal_conv1d import causal_conv1d_fn
from einops import rearrange
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn


# Class copied from https://github.com/google-deepmind/recurrentgemma
class BlockDiagonalLinear(nn.Module):
    """Block-diagonal linear layer."""

    def __init__(
        self, width: int, num_blocks: int, w_init_variance_scale: float = 1.0,
    ):
        """Initializes the BlockDiagonalLinear.

    Args:
      width: The number of dimensions of the input and output.
      num_blocks: The number of diagonal blocks in the layer.
      w_init_variance_scale: A parameters that scales the variance of the
        initialization of the weights.
    """
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.w_init_variance_scale = w_init_variance_scale
        self.block_width = self.width // self.num_blocks

        # Parameters.
        self.w = nn.Parameter(torch.zeros([self.num_blocks, self.block_width, self.block_width]))
        self.b = nn.Parameter(torch.zeros([self.num_blocks, self.block_width]))

        # Initialization.
        self.w_init_(self.w)

    def w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weight `w` of the layer."""
        std = math.sqrt(self.w_init_variance_scale / self.block_width)
        torch.nn.init.normal_(w, mean=0.0, std=std)

    def forward(self, x):
        """Calls the BlockDiagonalLinear."""
        # Split x to blocks.
        x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

        # Linear layer over each block + bias.
        y = torch.einsum("... h i, h i j -> ... h j", x, self.w) + self.b

        # Flatten the output.
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


# Class copied from https://github.com/google-deepmind/recurrentgemma


def rnn_scan(
    x, a, reset, h0,
):
    """Runs the recurrence of a linear RNN.

  Args:
    x: The input sequence.
    a: The diagonal of the recurrence matrix `A`.
    reset: Indicator of document boundaries, e.g. when to reset the hidden
      state of the RNN.
    h0: The initial hidden state.

  Returns:
    The output of the linear recurrence.
  """

    assert x.ndim == 3
    assert a.shape == x.shape[-a.ndim :]
    assert a.dtype == x.dtype
    assert type(a) is type(x)

    # Multiply `a` by the reset.
    a = a * (1 - reset)[..., None]

    if x.shape[1] == 1:
        # Using scan in sampling mode.
        y = a * h0[:, None] + x
    else:
        # Using scan in linear mode.
        x = x.permute(0, 2, 1)
        a = a.permute(0, 2, 1)
        x = x.contiguous()
        a = a.contiguous()
        y = scan(a.float(), x.float()).type_as(x)
        y = y.permute(0, 2, 1)
    return y, None


# Class copied from https://github.com/google-deepmind/recurrentgemma


def rnn_param_init(*, width: int, min_rad: float, max_rad: float, transform: str = "softplus",) -> torch.Tensor:
    """Initializes the `A` parameter of the RG-LRU uniformly on a ring."""
    unif = torch.rand(width)
    # Proportional to area in a ring.
    a_real = 0.5 * torch.log(unif * (max_rad ** 2 - min_rad ** 2) + min_rad ** 2 + 1e-8)

    if transform == "softplus":
        # Inverse transform.
        return torch.log(torch.exp(-a_real) - 1.0)
    else:
        raise NotImplementedError()


# Class copied from https://github.com/google-deepmind/recurrentgemma


class RGLRU(nn.Module):
    """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

    def __init__(
        self, width: int, num_heads: int, w_init_variance_scale: float = 1.0,
    ):
        """Initializes the RG-LRU.

    Args:
      width: The number of dimensions of the input and output.
      num_heads: The number of diagonal blocks in the input and A gate layers.
      w_init_variance_scale: Initialization parameter for the
        BlockDiagonalLinear layers of the gates. See the `BlockDiagonalLinear`
        layer for details.
    """
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.w_init_variance_scale = w_init_variance_scale

        # Parameters and layers.
        self.a_param = nn.Parameter(self.a_param_init)
        self.input_gate = BlockDiagonalLinear(
            width=self.width, num_blocks=self.num_heads, w_init_variance_scale=w_init_variance_scale,
        )
        self.a_gate = BlockDiagonalLinear(
            width=self.width, num_blocks=self.num_heads, w_init_variance_scale=self.w_init_variance_scale
        )

    @property
    def a_param_init(self) -> torch.Tensor:
        """Initializes the `A` parameter of the RG-LRU."""
        return rnn_param_init(width=self.width, min_rad=0.9, max_rad=0.999)

    def __call__(
        self, x, segment_pos, prev_h,
    ):
        """Calls the RG-LRU.

    Args:
      x: Sequence of input activations.
      segment_pos: Position of each token in the sequence.
      prev_h: The previous hidden state of the RG-LRU.

    Returns:
      Output of the block together with the updated hidden state.
    """
        for param in self.parameters():
            param.data_ptr()

        bs, l, d = x.shape
        assert segment_pos.shape == (bs, l)
        reset = (segment_pos == 0).type(torch.int32)
        prev_h = torch.zeros(size=(bs, d)) if prev_h is None else prev_h
        prev_h = prev_h.cuda()
        # Gates for x and a.
        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))

        # Compute the parameter `A` of the recurrence.
        log_a = -8.0 * gate_a * nn.functional.softplus(self.a_param)
        a = torch.exp(log_a)

        # Gate the input.
        gated_x = x * gate_x

        # Apply gamma normalization to the input.
        multiplier = torch.sqrt((1 - torch.exp(2 * log_a)) + 1e-6)
        multiplier = reset[..., None] + (1 - reset)[..., None] * multiplier
        normalized_x = gated_x * multiplier.type(x.dtype)

        y, last_h = rnn_scan(x=normalized_x, a=a, reset=reset, h0=prev_h,)

        return y, last_h


class Conv1D(MegatronModule):
    def __init__(self, config, width, temporal_width):
        super().__init__(config=config)
        self.config = config
        self.width = width
        self.temporal_width = temporal_width
        self.conv_1d = nn.Conv1d(
            in_channels=width,
            out_channels=width,
            bias=True,
            kernel_size=temporal_width,
            groups=width,
            padding=temporal_width - 1,
        )

    def forward(
        self, x, segment_pos=None, prev_x=None,
    ):
        x = x.permute(0, 2, 1)
        output = causal_conv1d_fn(
            x=x, weight=rearrange(self.conv_1d.weight, "d 1 w -> d w"), bias=self.conv_1d.bias, activation=None,
        ).permute(0, 2, 1)
        return output, None


@dataclass
class RecurrentLayerSubmodules:
    linear_in: Union[ModuleSpec, type] = IdentityOp
    linear_out: Union[ModuleSpec, type] = IdentityOp
    conv_1d: Union[ModuleSpec, type] = IdentityOp
    rg_lru: Union[ModuleSpec, type] = IdentityOp


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Returns the GELU activation function with the same approximation as JAX."""
    return nn.functional.gelu(x, approximate="tanh")


class RecurrentLayer(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: RecurrentLayerSubmodules,
        layer_idx=None,
        residual_in_fp32=False,
        **kwargs,
    ):
        """
        Top level Mamba Layer
        """
        super().__init__(config)
        self.config = config
        self.residual_in_fp32 = residual_in_fp32

        self.linear_in = build_module(
            submodules.linear_in,
            self.config.hidden_size,
            self.config.hidden_size * 2,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
        )

        self.linear_out = build_module(
            submodules.linear_out,
            self.config.hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            input_is_parallel=True,
        )

        self.conv_1d = build_module(
            submodules.conv_1d, config=self.config, width=self.config.hidden_size, temporal_width=4
        )

        self.rg_lru = build_module(
            submodules.rg_lru, width=self.config.hidden_size, num_heads=self.config.num_attention_heads
        )

    def forward(self, hidden_states, attention_mask=None, rotary_pos_emb=None):

        segment_pos = torch.arange(hidden_states.shape[0]).unsqueeze(0).repeat(hidden_states.shape[1], 1).cuda()
        in_intermidiate_parallel, in_bias_parallel = self.linear_in(hidden_states)

        x_bias_parallel, y_bias_parallel = in_bias_parallel.chunk(2, dim=-1)
        x_intermidiate_parallel, y_intermidiate_parallel = in_intermidiate_parallel.chunk(2, dim=-1)

        y = bias_gelu_impl(y_intermidiate_parallel, y_bias_parallel)

        x = x_intermidiate_parallel + x_bias_parallel
        x = x.permute(1, 0, 2)

        x, _ = self.conv_1d(x=x, segment_pos=segment_pos, prev_x=None)

        x, _ = self.rg_lru(x=x, segment_pos=segment_pos, prev_h=None,)

        x = x.permute(1, 0, 2)

        x = x * y
        x_intermidiate_parallel, x_bias_parallel = self.linear_out(x)

        return x_intermidiate_parallel, x_bias_parallel
