# coding=utf-8
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


import enum
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from nemo.collections.common.parts.adapter_modules import AbstractAdapterModule
from nemo.collections.common.parts.utils import activation_registry
from nemo.collections.nlp.modules.common.megatron.utils import init_method_const, init_method_normal
from nemo.core.classes.mixins import adapter_mixin_strategies

try:
    from apex.transformer.tensor_parallel import RowParallelLinear, ColumnParallelLinear
    from apex.normalization.fused_layer_norm import MixedFusedLayerNorm

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False


class AdapterName(str, enum.Enum):
    """
    Names for adapters used in NLP Adapters and IA3. Note: changing this will break backward compatibility. 
    """

    MLP_INFUSED = "mlp_infused_adapter"
    KEY_INFUSED = "key_infused_adapter"
    VALUE_INFUSED = "value_infused_adapter"
    PRE_ATTN_ADAPTER = 'adapter_1'
    POST_ATTN_ADAPTER = 'adapter_2'


class InfusedAdapter(AbstractAdapterModule):
    def __init__(
        self, in_features: int, adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ) -> None:
        super().__init__()
        self.scalers = nn.Parameter(torch.ones(in_features))
        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x):
        x = x * self.scalers[None, None, :]
        return x


class MLPInfusedAdapter(InfusedAdapter):
    """
    MLPInfusedAdapter is basically a clone of InfusedAdapter. We do this to make the adapter_mixin agnostic to adapter names
    and only check adapter class types. 
    """

    pass


@dataclass
class InfusedAdapterConfig:
    in_features: int
    adapter_strategy: Optional[Any] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(InfusedAdapter.__module__, InfusedAdapter.__name__)


@dataclass
class MLPInfusedAdapterConfig(InfusedAdapterConfig):
    _target_: str = "{0}.{1}".format(MLPInfusedAdapter.__module__, MLPInfusedAdapter.__name__)


class ParallelLinearAdapter(AbstractAdapterModule):
    def __init__(
        self,
        in_features: int,
        dim: int,
        activation: str = 'swish',
        norm_position: str = 'post',
        norm_type: str = 'mixedfusedlayernorm',
        column_init_method: str = 'xavier',
        row_init_method: str = 'zero',
        dropout: float = 0.0,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ):
        super().__init__()
        if not HAVE_APEX:
            logging.info("Apex is required to use ParallelLinearAdapters.")
            raise RuntimeError("ParallelLinearAdapter can not run without Apex.")
        self.activation = activation_registry[activation]()
        self.norm_position = norm_position

        if column_init_method == 'xavier':
            self.linear_in = ColumnParallelLinear(in_features, dim, bias=False)
        elif column_init_method == 'normal':
            self.linear_in = ColumnParallelLinear(in_features, dim, bias=False, init_method=init_method_normal(0.2))
        elif column_init_method == 'zero':
            self.linear_in = ColumnParallelLinear(in_features, dim, bias=False, init_method=init_method_const(0.0))
        else:
            raise NotImplementedError("column_init_method should be zero, normal or xavier")

        if row_init_method == 'xavier':
            self.linear_out = RowParallelLinear(dim, in_features, bias=False)
        elif row_init_method == 'normal':
            self.linear_out = RowParallelLinear(dim, in_features, bias=False, init_method=init_method_normal(0.2))
        elif row_init_method == 'zero':
            self.linear_out = RowParallelLinear(dim, in_features, bias=False, init_method=init_method_const(0.0))
        else:
            raise NotImplementedError("row_init_method should be zero, normal or xavier")

        if norm_type == 'mixedfusedlayernorm':
            self.layer_norm = MixedFusedLayerNorm(in_features, 1e-5, sequence_parallel_enbaled=False)
        elif norm_type == 'layernorm':
            self.layer_norm = nn.LayerNorm(in_features)
        else:
            raise NotImplementedError("norm_type should be either mixedfusedlayernorm or layernorm")

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x):

        if self.norm_position == 'pre':
            x = self.layer_norm(x)

        x, _ = self.linear_in(x)  # (@adithyare) ColumnLinear returns output and bias, we are ignoring the bias term.
        x = self.activation(x)
        x, _ = self.linear_out(x)

        if self.norm_position == 'post':
            x = self.layer_norm(x)

        # Add dropout if available
        if self.dropout is not None:
            x = self.dropout(x)

        return x


@dataclass
class ParallelLinearAdapterConfig:
    in_features: int
    dim: int
    activation: str = 'swish'
    norm_position: str = 'post'
    norm_type: str = 'mixedfusedlayernorm'
    column_init_method: str = 'xavier'
    row_init_method: str = 'zero'
    dropout: float = 0.0
    adapter_strategy: Optional[Any] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(ParallelLinearAdapter.__module__, ParallelLinearAdapter.__name__)
