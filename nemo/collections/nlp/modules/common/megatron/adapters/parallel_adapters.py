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
import torch.nn.init as init

from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil
from nemo.collections.common.parts.utils import activation_registry
from nemo.collections.nlp.modules.common.megatron.fused_bias_gelu import fused_bias_gelu
from nemo.collections.nlp.modules.common.megatron.utils import init_method_const, init_method_normal
from nemo.core.classes.mixins import adapter_mixin_strategies

try:
    from apex.normalization.fused_layer_norm import MixedFusedLayerNorm

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class AdapterName(str, enum.Enum):
    """
    Names for adapters used in NLP Adapters and IA3. Note: changing this will break backward compatibility. 
    """

    MLP_INFUSED = "mlp_infused_adapter"
    KEY_INFUSED = "key_infused_adapter"
    VALUE_INFUSED = "value_infused_adapter"
    PRE_ATTN_ADAPTER = 'adapter_1'
    POST_ATTN_ADAPTER = 'adapter_2'
    PTUNING_ADAPTER = "ptuning_adapter"
    LORA_KQV_ADAPTER = "lora_kqv_adapter"
    LORA_KV_ADAPTER = "lora_kv_adapter"
    LORA_Q_ADAPTER = "lora_q_adapter"


class InfusedAdapter(nn.Module, AdapterModuleUtil):
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


class ParallelLinearAdapter(nn.Module, AdapterModuleUtil):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        activation: str = 'swish',
        norm_position: str = 'post',
        norm_type: str = 'mixedfusedlayernorm',
        column_init_method: str = 'xavier',  # TODO: (@adithyare) should rename this to input_init_method to be more precise.
        row_init_method: str = 'zero',  # TODO: (@adithyare) should rename this to output_init_method to be more precise.
        gather_output: bool = True,
        dropout: float = 0.0,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ):
        super().__init__()
        if not HAVE_APEX:
            logging.info("Apex is required to use ParallelLinearAdapters.")
            raise RuntimeError("ParallelLinearAdapter can not run without Apex.")
        if not HAVE_MEGATRON_CORE:
            logging.info("Megatron-core is required to use ParallelLinearAdapters.")
            raise RuntimeError("ParallelLinearAdapter can not run without Megatron-core.")
        self.activation = activation_registry[activation]()
        self.norm_position = norm_position

        self.linear_in = ColumnParallelLinear(
            in_features, dim, bias=False, gather_output=True, init_method=self._get_init_fn(column_init_method)
        )
        if gather_output:
            self.linear_out = RowParallelLinear(
                dim, out_features, bias=False, init_method=self._get_init_fn(row_init_method)
            )
        else:
            # (@adithyare) we use this option to mirror the behavior a column parallel layer with two low-rank column parallel layers
            # if the original column parallel layer uses gather_output=False, then we will use the self.liner_out layer defined below.
            self.linear_out = ColumnParallelLinear(
                dim, out_features, bias=False, gather_output=False, init_method=self._get_init_fn(row_init_method)
            )

        if self.norm_position in ["pre", "post"]:
            ln_features = in_features if self.norm_position == "pre" else out_features
            if norm_type == 'mixedfusedlayernorm':
                self.layer_norm = MixedFusedLayerNorm(ln_features, 1e-5, sequence_parallel_enbaled=False)
            elif norm_type == 'layernorm':
                self.layer_norm = nn.LayerNorm(ln_features)
            else:
                raise NotImplementedError("norm_type should be either mixedfusedlayernorm or layernorm")

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    def _get_init_fn(self, init_method: str):
        if init_method == 'xavier':
            init_fn = init.xavier_normal_
        elif init_method == 'normal':
            init_fn = init_method_normal(0.2)
        elif init_method == "zero":
            init_fn = init_method_const(0.0)
        else:
            raise NotImplementedError("out_init_method should be zero, normal or xavier")
        return init_fn

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
    out_features: int
    dim: int
    activation: str = 'swish'
    norm_position: str = 'post'
    norm_type: str = 'mixedfusedlayernorm'
    column_init_method: str = 'xavier'
    row_init_method: str = 'zero'
    gather_output: bool = True
    dropout: float = 0.0
    adapter_strategy: Optional[Any] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(ParallelLinearAdapter.__module__, ParallelLinearAdapter.__name__)


class LoraKQVAdapter(ParallelLinearAdapter):
    """
    Lora Adapters are the same arch as regualr adapters but with potentially different input and output feature sizes 
    and they do not use an bottleneck activation function
    """

    pass


@dataclass
class LoraKQVAdapterConfig(ParallelLinearAdapterConfig):
    _target_: str = "{0}.{1}".format(LoraKQVAdapter.__module__, LoraKQVAdapter.__name__)


class PromptEncoderAdapter(nn.Module, AdapterModuleUtil):
    """
    The Tensor Parallel MLP prompt encoder network that is used to generate the virtual 
    token embeddings for p-tuning. It only have two layers.
    TODO: (@adithyare) Need to add all the functionality from the PromptEncoder class
    """

    def __init__(
        self,
        virtual_tokens: int,
        bottleneck_dim: int,
        embedding_dim: int,
        init_std: float,
        output_dim: int,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ):
        """
        Initializes the Tensor Model parallel MLP PromptEncoderMLP module.
        Args:
            virtual_tokens: the  number of vitural tokens
            hidden_size: hidden dimension
            output_size:  the output dimension
            init_std: the MLP init std value 
        """
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.virtual_tokens = virtual_tokens
        self.activation = "gelu"

        sequence_parallel = False
        gradient_accumulation_fusion = False
        # (@adithyare) the persistent=False will not pollute the indices into the state_dict of this module.
        self.register_buffer("indices", torch.LongTensor(list(range(self.virtual_tokens))), persistent=False)
        self.embedding = torch.nn.Embedding(self.virtual_tokens, self.embedding_dim)
        self.first = ColumnParallelLinear(
            self.embedding_dim,
            self.bottleneck_dim,
            gather_output=False,
            init_method=init_method_normal(init_std),
            skip_bias_add=True,
            use_cpu_initialization=False,
            bias=True,
            sequence_parallel_enabled=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
        self.second = RowParallelLinear(
            self.bottleneck_dim,
            self.output_dim,
            input_is_parallel=True,
            init_method=init_method_normal(init_std),
            skip_bias_add=True,
            use_cpu_initialization=False,
            bias=True,
            sequence_parallel_enabled=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, batch_size):
        input_embeds = self.embedding(self.indices).unsqueeze(0)
        intermediate_parallel, bias_parallel = self.first(input_embeds)
        intermediate_parallel = fused_bias_gelu(intermediate_parallel, bias_parallel)
        output_embeds, bias_parallel = self.second(intermediate_parallel)
        output_embeds = output_embeds + bias_parallel
        output_embeds = output_embeds.transpose(0, 1)
        output_embeds = output_embeds.expand(self.virtual_tokens, batch_size, self.output_dim)
        return output_embeds


@dataclass
class PromptEncoderAdapterConfig:
    virtual_tokens: int
    bottleneck_dim: int
    embedding_dim: int
    init_std: float
    output_dim: int
    adapter_strategy: Optional[Any] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(PromptEncoderAdapter.__module__, PromptEncoderAdapter.__name__)
