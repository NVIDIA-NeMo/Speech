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

from nemo.collections.asr.modules.transformer import transformer_modules
from nemo.collections.asr.parts.submodules.adapters.multi_head_attention_adapter_module import (
    MHAResidualAddAdapterStrategy,
    MHAResidualAddAdapterStrategyConfig,
)
from nemo.collections.common.parts import adapter_modules
from nemo.core.classes.mixins import adapter_mixin_strategies, adapter_mixins


class TransformerMultiHeadAttentionAdapter(transformer_modules.MultiHeadAttention, adapter_modules.AdapterModuleUtil):
    """Multi-Head Attention layer of Transformer Encoder.

    Args:
        hidden_size (int): number of heads
        num_attention_heads (int): size of the features
        attn_score_dropout (float): dropout rate for the attention scores
        attn_layer_dropout (float): dropout rate for the layer
        proj_dim (int, optional): Optional integer value for projection before computing attention.
           If None, then there is no projection (equivalent to proj_dim = n_feat).
           If > 0, then will project the n_feat to proj_dim before calculating attention.
           If <0, then will equal n_head, so that each head has a projected dimension of 1.
       adapter_strategy: By default, MHAResidualAddAdapterStrategyConfig. An adapter composition function object.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        proj_dim: Optional[int] = None,
        adapter_strategy: MHAResidualAddAdapterStrategy = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
        )

        self.pre_norm = nn.LayerNorm(hidden_size)

        # Set the projection dim to number of heads automatically
        if proj_dim is not None and proj_dim < 1:
            proj_dim = num_attention_heads

        self.proj_dim = proj_dim

        # Recompute weights for projection dim
        if self.proj_dim is not None:
            if self.proj_dim % num_attention_heads != 0:
                raise ValueError(f"proj_dim ({proj_dim}) is not divisible by n_head ({num_attention_heads})")

            self.attn_head_size = self.proj_dim // num_attention_heads
            self.attn_scale = math.sqrt(math.sqrt(self.attn_head_size))
            self.query_net = nn.Linear(hidden_size, self.proj_dim)
            self.key_net = nn.Linear(hidden_size, self.proj_dim)
            self.value_net = nn.Linear(hidden_size, self.proj_dim)
            self.out_projection = nn.Linear(self.proj_dim, hidden_size)

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # reset parameters for Q to be identity operation
        self.reset_parameters()

    def forward(self, queries, keys, values, attention_mask):
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
        query = self.pre_norm(queries)
        key = self.pre_norm(keys)
        value = self.pre_norm(values)

        return super().forward(query, key, value, attention_mask)

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.zeros_(self.out_projection.weight)
            nn.init.zeros_(self.out_projection.bias)

    def get_default_strategy_config(self) -> 'dataclass':
        return MHAResidualAddAdapterStrategyConfig()


@dataclass
class TransformerMultiHeadAttentionAdapterConfig:
    hidden_size: int
    num_attention_heads: int
    attn_score_dropout: float = 0.0
    attn_layer_dropout: float = 0.0
    proj_dim: Optional[int] = None
    adapter_strategy: Optional[Any] = field(default_factory=lambda: MHAResidualAddAdapterStrategyConfig())
    _target_: str = "{0}.{1}".format(
        TransformerMultiHeadAttentionAdapter.__module__, TransformerMultiHeadAttentionAdapter.__name__
    )
