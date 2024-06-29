# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import nn as nn

from nemo.collections.asr.parts.submodules import multi_head_attention as mha
from nemo.collections.common.parts import adapter_modules
from nemo.core.classes.mixins import adapter_mixin_strategies


class MHAResidualAddAdapterStrategy(adapter_mixin_strategies.ResidualAddAdapterStrategy):
    """
    An implementation of residual addition of an adapter module with its input for the MHA Adapters.
    """

    def forward(self, input: dict, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'):
        """
        A basic strategy, comprising of a residual connection over the input, after forward pass by
        the underlying adapter. Additional work is done to pack and unpack the dictionary of inputs and outputs.

        Note: The `value` tensor is added to the output of the attention adapter as the residual connection.

        Args:
            input: A dictionary of multiple input arguments for the adapter module.

                `query`, `key`, `value`: Original output tensor of the module, or the output of the
                 previous adapter (if more than one adapters are enabled).

                 `mask`: Attention mask.

                 `pos_emb`: Optional positional embedding for relative encoding.

            adapter: The adapter module that is currently required to perform the forward pass.
            module: The calling module, in its entirety. It is a module that implements `AdapterModuleMixin`,
                therefore the strategy can access all other adapters in this module via `module.adapter_layer`.

        Returns:
            The result tensor, after one of the active adapters has finished its forward passes.
        """
        out = self.compute_output(input, adapter, module=module)

        value_name = None
        if 'value' in input:
            value_name = 'value'
        elif 'values' in input:
            value_name = 'values'
        else:
            raise ValueError(
                "Input dictionary must contain 'value' or 'values' key for residual connection. Input "
                f"dictionary keys: {input.keys()}"
            )

        # If not in training mode, or probability of stochastic depth is 0, skip step.
        p = self.stochastic_depth
        if not module.training or p == 0.0:
            pass
        else:
            out = self.apply_stochastic_depth(out, input[value_name], adapter, module=module)

        # Return the residual connection output = input + adapter(input)
        result = input[value_name] + out

        # If l2_lambda is activated, register the loss value
        self.compute_auxiliary_losses(result, input[value_name], adapter, module=module)

        return result

    def compute_output(
        self, input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'
    ) -> torch.Tensor:
        """
        Compute the output of a single adapter to some input.

        Args:
            input: Original output tensor of the module, or the output of the previous adapter (if more than
                one adapters are enabled).
            adapter: The adapter module that is currently required to perform the forward pass.
            module: The calling module, in its entirety. It is a module that implements `AdapterModuleMixin`,
                therefore the strategy can access all other adapters in this module via `module.adapter_layer`.

        Returns:
            The result tensor, after one of the active adapters has finished its forward passes.
        """
        if isinstance(input, (list, tuple)):
            out = adapter(*input)
        elif isinstance(input, dict):
            out = adapter(**input)
        else:
            out = adapter(input)
        return out


@dataclass
class MHAResidualAddAdapterStrategyConfig(adapter_mixin_strategies.ResidualAddAdapterStrategyConfig):
    _target_: str = "{0}.{1}".format(
        MHAResidualAddAdapterStrategy.__module__, MHAResidualAddAdapterStrategy.__name__
    )  # mandatory field


class MultiHeadAttentionAdapter(mha.MultiHeadAttention, adapter_modules.AdapterModuleUtil):
    """Multi-Head Attention layer of Transformer.

    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
        proj_dim (int, optional): Optional integer value for projection before computing attention.
           If None, then there is no projection (equivalent to proj_dim = n_feat).
           If > 0, then will project the n_feat to proj_dim before calculating attention.
           If <0, then will equal n_head, so that each head has a projected dimension of 1.
       adapter_strategy: By default, MHAResidualAddAdapterStrategyConfig. An adapter composition function object.
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        proj_dim: Optional[int] = None,
        adapter_strategy: MHAResidualAddAdapterStrategy = None,
    ):
        super().__init__(n_head=n_head, n_feat=n_feat, dropout_rate=dropout_rate, max_cache_len=0)

        self.pre_norm = nn.LayerNorm(n_feat)

        # Set the projection dim to number of heads automatically
        if proj_dim is not None and proj_dim < 1:
            proj_dim = n_head

        self.proj_dim = proj_dim

        # Recompute weights for projection dim
        if self.proj_dim is not None:
            if self.proj_dim % n_head != 0:
                raise ValueError(f"proj_dim ({proj_dim}) is not divisible by n_head ({n_head})")

            self.d_k = self.proj_dim // n_head
            self.s_d_k = math.sqrt(self.d_k)
            self.linear_q = nn.Linear(n_feat, self.proj_dim)
            self.linear_k = nn.Linear(n_feat, self.proj_dim)
            self.linear_v = nn.Linear(n_feat, self.proj_dim)
            self.linear_out = nn.Linear(self.proj_dim, n_feat)

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # reset parameters for Q to be identity operation
        self.reset_parameters()

    def forward(self, query, key, value, mask, pos_emb=None, cache=None):
        """Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            cache (torch.Tensor) : (batch, time_cache, size)

        returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache  (torch.Tensor) : (batch, time_cache_next, size)
        """
        # Need to perform duplicate computations as at this point the tensors have been
        # separated by the adapter forward
        query = self.pre_norm(query)
        key = self.pre_norm(key)
        value = self.pre_norm(value)

        return super().forward(query, key, value, mask, pos_emb, cache=cache)

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.zeros_(self.linear_out.weight)
            nn.init.zeros_(self.linear_out.bias)

    def get_default_strategy_config(self) -> 'dataclass':
        return MHAResidualAddAdapterStrategyConfig()


@dataclass
class MultiHeadAttentionAdapterConfig:
    n_head: int
    n_feat: int
    dropout_rate: float = 0.0
    proj_dim: Optional[int] = None
    adapter_strategy: Optional[Any] = field(default_factory=lambda: MHAResidualAddAdapterStrategyConfig())
    _target_: str = "{0}.{1}".format(MultiHeadAttentionAdapter.__module__, MultiHeadAttentionAdapter.__name__)


class RelPositionMultiHeadAttentionAdapter(mha.RelPositionMultiHeadAttention, adapter_modules.AdapterModuleUtil):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
        proj_dim (int, optional): Optional integer value for projection before computing attention.
            If None, then there is no projection (equivalent to proj_dim = n_feat).
            If > 0, then will project the n_feat to proj_dim before calculating attention.
            If <0, then will equal n_head, so that each head has a projected dimension of 1.
        adapter_strategy: By default, MHAResidualAddAdapterStrategyConfig. An adapter composition function object.
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        proj_dim: Optional[int] = None,
        adapter_strategy: MHAResidualAddAdapterStrategyConfig = None,
    ):
        super().__init__(
            n_head=n_head, n_feat=n_feat, dropout_rate=dropout_rate, pos_bias_u=None, pos_bias_v=None, max_cache_len=0
        )

        self.pre_norm = nn.LayerNorm(n_feat)

        # Set the projection dim to number of heads automatically
        if proj_dim is not None and proj_dim < 1:
            proj_dim = n_head

        self.proj_dim = proj_dim

        # Recompute weights for projection dim
        if self.proj_dim is not None:
            if self.proj_dim % n_head != 0:
                raise ValueError(f"proj_dim ({proj_dim}) is not divisible by n_head ({n_head})")

            self.d_k = self.proj_dim // n_head
            self.s_d_k = math.sqrt(self.d_k)
            self.linear_q = nn.Linear(n_feat, self.proj_dim)
            self.linear_k = nn.Linear(n_feat, self.proj_dim)
            self.linear_v = nn.Linear(n_feat, self.proj_dim)
            self.linear_out = nn.Linear(self.proj_dim, n_feat)
            self.linear_pos = nn.Linear(n_feat, self.proj_dim, bias=False)
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # reset parameters for Q to be identity operation
        self.reset_parameters()

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
            cache (torch.Tensor) : (batch, time_cache, size)
        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache_next (torch.Tensor) : (batch, time_cache_next, size)
        """
        # Need to perform duplicate computations as at this point the tensors have been
        # separated by the adapter forward
        query = self.pre_norm(query)
        key = self.pre_norm(key)
        value = self.pre_norm(value)

        return super().forward(query, key, value, mask, pos_emb, cache=cache)

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.zeros_(self.linear_out.weight)
            nn.init.zeros_(self.linear_out.bias)

            # NOTE: This exact procedure apparently highly important.
            # Above operation is safe to do as self.linear_out.weight *= 0.0 (similar for bias)
            # However:
            # DO NOT REPLACE BELOW WITH self.pos_bias_u *= 0.0 OR self.pos_bias_v *= 0.0
            # For some reason at init sometimes it will cause the value of the tensor to become NaN
            # All operations to compute matrix_ac and matrix_bd will then fail.
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)

    def get_default_strategy_config(self) -> 'dataclass':
        return MHAResidualAddAdapterStrategyConfig()


@dataclass
class RelPositionMultiHeadAttentionAdapterConfig:
    n_head: int
    n_feat: int
    dropout_rate: float = 0.0
    proj_dim: Optional[int] = None
    adapter_strategy: Optional[Any] = field(default_factory=lambda: MHAResidualAddAdapterStrategyConfig())
    _target_: str = "{0}.{1}".format(
        RelPositionMultiHeadAttentionAdapter.__module__, RelPositionMultiHeadAttentionAdapter.__name__
    )


class PositionalEncodingAdapter(mha.PositionalEncoding, adapter_modules.AdapterModuleUtil):
    """
    Absolute positional embedding adapter.

    .. note::

        Absolute positional embedding value is added to the input tensor *without residual connection* !
        Therefore, the input is changed, if you only require the positional embedding, drop the returned `x` !

    Args:
        d_model (int): The input dimension of x.
        max_len (int): The max sequence length.
        xscale (float): The input scaling factor. Defaults to 1.0.
        adapter_strategy (AbstractAdapterStrategy): By default, ReturnResultAdapterStrategyConfig.
            An adapter composition function object.
            NOTE: Since this is a positional encoding, it will not add a residual !
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        xscale=1.0,
        adapter_strategy: adapter_mixin_strategies.ReturnResultAdapterStrategyConfig = None,
    ):

        super().__init__(
            d_model=d_model,
            dropout_rate=0.0,
            max_len=max_len,
            xscale=xscale,
            dropout_rate_emb=0.0,
        )

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    def get_default_strategy_config(self) -> 'dataclass':
        return adapter_mixin_strategies.ReturnResultAdapterStrategyConfig()


@dataclass
class PositionalEncodingAdapterConfig:
    d_model: int
    max_len: int = 5000
    xscale: float = 1.0
    adapter_strategy: Optional[Any] = field(
        default_factory=lambda: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    )
    _target_: str = "{0}.{1}".format(PositionalEncodingAdapter.__module__, PositionalEncodingAdapter.__name__)


class RelPositionalEncodingAdapter(mha.RelPositionalEncoding, adapter_modules.AdapterModuleUtil):
    """
    Relative positional encoding for TransformerXL's layers
    See : Appendix B in https://arxiv.org/abs/1901.02860

    .. note::

        Relative positional embedding value is **not** added to the input tensor !
        Therefore, the input should be updated changed, if you only require the positional embedding, drop the returned `x` !

    Args:
        d_model (int): embedding dim
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        adapter_strategy: By default, ReturnResultAdapterStrategyConfig. An adapter composition function object.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        xscale=1.0,
        adapter_strategy: adapter_mixin_strategies.ReturnResultAdapterStrategyConfig = None,
    ):
        super().__init__(d_model=d_model, dropout_rate=0.0, max_len=max_len, xscale=xscale, dropout_rate_emb=0.0)

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    def get_default_strategy_config(self) -> 'dataclass':
        return adapter_mixin_strategies.ReturnResultAdapterStrategyConfig()


@dataclass
class RelPositionalEncodingAdapterConfig:
    d_model: int
    max_len: int = 5000
    xscale: float = 1.0
    adapter_strategy: Optional[Any] = field(
        default_factory=lambda: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    )
    _target_: str = "{0}.{1}".format(RelPositionalEncodingAdapter.__module__, RelPositionalEncodingAdapter.__name__)
