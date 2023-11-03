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
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.init as init

from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil
from nemo.collections.common.parts.utils import activation_registry
from nemo.collections.nlp.modules.common.megatron.fused_bias_gelu import fused_bias_gelu
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults, init_method_const, init_method_normal
from nemo.core.classes.mixins import adapter_mixin_strategies
from nemo.core.classes.mixins.adapter_mixins import AdapterConfig


try:
    from apex.normalization.fused_layer_norm import MixedFusedLayerNorm

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import ModelParallelConfig
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

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
    MM_LINEAR_ADAPTER = "mm_linear_adapter"


class InfusedAdapter(nn.Module, AdapterModuleUtil):
    def __init__(
        self, in_features: int, model_parallel_config: Optional[ModelParallelConfig] = None, **kwargs
    ) -> None:
        super().__init__()

        if model_parallel_config is None:
            model_parallel_config = ModelParallelConfig()

        self.scalers = nn.Parameter(torch.ones(in_features))

        # cast all parameters when using amp O2 training
        if model_parallel_config.bf16:
            self.bfloat16()
        elif model_parallel_config.fp16:
            self.half()

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_mixin_strategies.ReturnResultAdapterStrategy())

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
class InfusedAdapterConfig(AdapterConfig):
    in_features: int
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
        norm_position: Optional[str] = 'post',
        norm_type: Optional[str] = 'mixedfusedlayernorm',
        column_init_method: str = 'xavier',  # TODO: (@adithyare) should rename this to input_init_method to be more precise.
        row_init_method: str = 'zero',  # TODO: (@adithyare) should rename this to output_init_method to be more precise.
        gather_output: bool = True,
        dropout: float = 0.0,
        model_parallel_config: Optional[ModelParallelConfig] = None,
        **kwargs,
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

        # megatron_gpt_peft_models will provide this arg, but deprecated ones do not.
        # in case this arg is not provided, use the dummy default config.
        if model_parallel_config is None:
            model_parallel_config = ModelParallelConfig()

        self.linear_in = ColumnParallelLinear(
            in_features,
            dim,
            config=model_parallel_config,
            bias=False,
            gather_output=True,
            init_method=self._get_init_fn(column_init_method),
        )
        if gather_output:
            self.linear_out = RowParallelLinear(
                dim,
                out_features,
                config=model_parallel_config,
                bias=False,
                init_method=self._get_init_fn(row_init_method),
            )
        else:
            # (@adithyare) we use this option to mirror the behavior a column parallel layer with two low-rank column parallel layers
            # if the original column parallel layer uses gather_output=False, then we will use the self.liner_out layer defined below.
            self.linear_out = ColumnParallelLinear(
                dim,
                out_features,
                config=model_parallel_config,
                bias=False,
                gather_output=False,
                init_method=self._get_init_fn(row_init_method),
            )

        if self.norm_position in ["pre", "post"]:
            ln_features = in_features if self.norm_position == "pre" else out_features
            if norm_type == 'mixedfusedlayernorm':
                self.layer_norm = MixedFusedLayerNorm(ln_features, 1e-5, sequence_parallel_enbaled=False)
            elif norm_type == 'layernorm':
                self.layer_norm = nn.LayerNorm(ln_features)
            else:
                raise NotImplementedError("norm_type should be either mixedfusedlayernorm or layernorm")
        else:
            self.layer_norm = None

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # cast all parameters when using amp O2 training
        if model_parallel_config.bf16:
            self.bfloat16()
        elif model_parallel_config.fp16:
            self.half()

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_mixin_strategies.ReturnResultAdapterStrategy())

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

    def adapter_unfreeze(self,):
        """
        Can be customized to allow for selective training of only some params in the PEFT.
        """
        super().adapter_unfreeze()

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
class ParallelLinearAdapterConfig(AdapterConfig):
    in_features: int
    out_features: int
    dim: int
    activation: str = 'swish'
    norm_position: Optional[str] = 'post'
    norm_type: Optional[str] = 'mixedfusedlayernorm'
    column_init_method: str = 'xavier'
    row_init_method: str = 'zero'
    gather_output: bool = True
    dropout: float = 0.0
    _target_: str = "{0}.{1}".format(ParallelLinearAdapter.__module__, ParallelLinearAdapter.__name__)


class LoraKQVAdapter(ParallelLinearAdapter):
    """
    Lora Adapters are the same arch as regular adapters but with potentially different input and output feature sizes
    and they do not use an bottleneck activation function
    """

    pass


class LoraKVAdapter(ParallelLinearAdapter):
    """
    Lora Adapters are the same arch as regular adapters but with potentially different input and output feature sizes
    and they do not use an bottleneck activation function
    """

    pass


class LoraQAdapter(ParallelLinearAdapter):
    """
    Lora Adapters are the same arch as regular adapters but with potentially different input and output feature sizes
    and they do not use an bottleneck activation function
    """

    pass


@dataclass
class LoraKQVAdapterConfig(ParallelLinearAdapterConfig):
    _target_: str = "{0}.{1}".format(LoraKQVAdapter.__module__, LoraKQVAdapter.__name__)


@dataclass
class LoraQAdapterConfig(ParallelLinearAdapterConfig):
    _target_: str = "{0}.{1}".format(LoraQAdapter.__module__, LoraQAdapter.__name__)


@dataclass
class LoraKVAdapterConfig(ParallelLinearAdapterConfig):
    _target_: str = "{0}.{1}".format(LoraKVAdapter.__module__, LoraKVAdapter.__name__)


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
        model_parallel_config: Optional[ModelParallelConfig] = None,
        **kwargs,
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

        if model_parallel_config is None:
            model_parallel_config = ModelParallelConfig()

        sequence_parallel = False
        gradient_accumulation_fusion = False
        # (@adithyare) the persistent=False will not pollute the indices into the state_dict of this module.
        self.register_buffer("indices", torch.LongTensor(list(range(self.virtual_tokens))), persistent=False)
        self.embedding = torch.nn.Embedding(self.virtual_tokens, self.embedding_dim)
        self.register_buffer("inference_table", torch.Tensor(self.virtual_tokens, self.output_dim), persistent=True)
        self.is_inference_ready = False
        self.first = ColumnParallelLinear(
            self.embedding_dim,
            self.bottleneck_dim,
            config=model_parallel_config,
            gather_output=False,
            init_method=init_method_normal(init_std),
            skip_bias_add=True,
            bias=True,
        )
        self.second = RowParallelLinear(
            self.bottleneck_dim,
            self.output_dim,
            config=model_parallel_config,
            input_is_parallel=True,
            init_method=init_method_normal(init_std),
            skip_bias_add=True,
            bias=True,
        )

        # cast all parameters when using amp O2 training
        if model_parallel_config.bf16:
            self.bfloat16()
        elif model_parallel_config.fp16:
            self.half()

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_mixin_strategies.ReturnResultAdapterStrategy())

    def set_inference_table(self, prompt_representation: torch.Tensor):
        """
        This method caches the output representation from the Encoder and saves it inside `self.inference_table`.
        """
        prompt_representation = prompt_representation.detach().clone()
        self.inference_table.data = prompt_representation
        self.is_inference_ready = True
        return True

    def clear_inference_table(self,):
        self.inference_table.fill_(0.0)
        self.is_inference_ready = False

    def get_inference_table(self,):
        return self.inference_table.data

    def inner_forward(self,):
        input_embeds = self.embedding(self.indices).unsqueeze(0)
        intermediate_parallel, bias_parallel = self.first(input_embeds)
        intermediate_parallel = fused_bias_gelu(intermediate_parallel, bias_parallel)
        output_embeds, bias_parallel = self.second(intermediate_parallel)
        output_embeds = output_embeds + bias_parallel
        output_embeds = output_embeds.transpose(0, 1)
        return output_embeds

    def forward(self, batch_size: int, use_cached_reps: bool = False) -> torch.Tensor:
        """
        Forward pass through the encoder with caching of prompt representations
        """
        if use_cached_reps:
            output_embeds = self.get_inference_table().unsqueeze(1)
        else:
            if self.training:
                if self.is_inference_ready:
                    self.clear_inference_table()
                output_embeds = self.inner_forward()
            else:
                output_embeds = self.inner_forward()
                if not self.is_inference_ready:
                    output_embeds = self.inner_forward()
                    self.set_inference_table(output_embeds.squeeze(1))
                output_embeds = self.get_inference_table().unsqueeze(1)

        output_embeds = output_embeds.expand(self.virtual_tokens, batch_size, self.output_dim)
        return output_embeds


@dataclass
class PromptEncoderAdapterConfig(AdapterConfig):
    virtual_tokens: int
    bottleneck_dim: int
    embedding_dim: int
    init_std: float
    output_dim: int
    _target_: str = "{0}.{1}".format(PromptEncoderAdapter.__module__, PromptEncoderAdapter.__name__)


class ParallelLinearAdapterWeightTying(ParallelLinearAdapter):
    """
    Extends parallel linear adapter for weight tying by providing a position embedding and convenience methods for tying weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        activation: str = 'swish',
        norm_position: Optional[str] = 'post',
        norm_type: Optional[str] = 'mixedfusedlayernorm',
        column_init_method: str = 'xavier',  # TODO: (@adithyare) should rename this to input_init_method to be more precise.
        row_init_method: str = 'zero',  # TODO: (@adithyare) should rename this to output_init_method to be more precise.
        gather_output: bool = True,
        dropout: float = 0.0,
        num_position_embeddings: int = 1,
        dim_position_embeddings: int = 1024,
        position_embedding_strategy: Optional[str] = "add",
        model_parallel_config: Optional[ModelParallelConfig] = None,
        **kwargs,
    ):
        self.position_embeddings = None
        self.mlp = None
        self.position_embedding_strategy = position_embedding_strategy
        assert self.position_embedding_strategy in ["add", "concat", "mlpconcat", "biasadd", None]
        if self.position_embedding_strategy == "concat":
            in_features += dim_position_embeddings
        elif self.position_embedding_strategy == "mlpconcat":
            in_features += dim_position_embeddings
        elif self.position_embedding_strategy == "biasadd":
            assert (
                out_features == dim_position_embeddings
            ), "adapter output feature size should match position emb size to bias add"
        elif self.position_embedding_strategy == "add":
            assert (
                in_features == dim_position_embeddings
            ), "adapter input feature size should match position emb size to add"
        super().__init__(
            in_features,
            out_features,
            dim,
            activation,
            norm_position,
            norm_type,
            column_init_method,
            row_init_method,
            gather_output,
            dropout,
            model_parallel_config,
            **kwargs,
        )
        if self.position_embedding_strategy:
            self.position_embeddings = torch.nn.Embedding(num_position_embeddings, dim_position_embeddings)
            self.position_embeddings.weight.data.fill_(0.0)
        if self.position_embedding_strategy == "mlpconcat":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim_position_embeddings, dim_position_embeddings, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(dim_position_embeddings, dim_position_embeddings, bias=False),
            )
        self.register_buffer("position_id", torch.LongTensor([1]), persistent=False)

    def set_position(self, position_id):
        self.position_id *= position_id

    def tie_weights(self, position_id, adapter):

        self.set_position(position_id)
        if self.linear_in:
            self.linear_in.weight = adapter.linear_in.weight
        if self.linear_out:
            self.linear_out.weight = adapter.linear_out.weight
        if self.layer_norm:
            self.layer_norm.weight = adapter.layer_norm.weight
            self.layer_norm.bias = adapter.layer_norm.bias
        if self.mlp:
            self.mlp[0].weight = adapter.mlp[0].weight
            self.mlp[2].weight = adapter.mlp[2].weight
        if self.position_embeddings:
            self.position_embeddings.weight = adapter.position_embeddings.weight

        return True

    def forward(self, x):

        if self.position_embedding_strategy:
            pos = self.position_embeddings(self.position_id).unsqueeze(0)
            if self.position_embedding_strategy == "add":
                pos = pos.expand_as(x)
                x = x + pos

            elif self.position_embedding_strategy == "concat":
                pos = pos.expand(x.shape[0], x.shape[1], pos.shape[2])
                x = torch.cat((x, pos), dim=2)
            elif self.position_embedding_strategy == "mlpconcat":
                pos = pos.expand(x.shape[0], x.shape[1], pos.shape[2])
                pos = self.mlp(pos)
                x = torch.cat((x, pos), dim=2)

        if self.norm_position == 'pre':
            x = self.layer_norm(x)

        x, _ = self.linear_in(x)  # (@adithyare) ColumnLinear returns output and bias, we are ignoring the bias term.
        x = self.activation(x)
        x, _ = self.linear_out(x)
        if self.norm_position == 'post':
            x = self.layer_norm(x)

        if self.position_embedding_strategy == "biasadd":
            pos = pos.expand_as(x)
            x = x + pos

        # Add dropout if available
        if self.dropout is not None:
            x = self.dropout(x)

        return x


@dataclass
class ParallelLinearAdapterWeightTyingConfig:
    in_features: int
    out_features: int
    dim: int
    activation: str = 'swish'
    norm_position: Optional[str] = 'post'
    norm_type: Optional[str] = 'mixedfusedlayernorm'
    column_init_method: str = 'xavier'
    row_init_method: str = 'zero'
    gather_output: bool = True
    dropout: float = 0.0
    num_position_embeddings: int = 1
    dim_position_embeddings: int = 1024
    position_embedding_strategy: Optional[str] = "concat"
    _target_: str = "{0}.{1}".format(
        ParallelLinearAdapterWeightTying.__module__, ParallelLinearAdapterWeightTying.__name__
    )


class LoraKQVAdapterWeightTying(ParallelLinearAdapterWeightTying):
    """
    TODO
    """

    pass


@dataclass
class LoraKQVAdapterWeightTyingConfig(ParallelLinearAdapterWeightTyingConfig):
    _target_: str = "{0}.{1}".format(LoraKQVAdapterWeightTying.__module__, LoraKQVAdapterWeightTying.__name__)


class MultiModalLinearAdapter(nn.Module, AdapterModuleUtil):
    def __init__(self, in_features: int, out_features: int, bias: bool, **kwargs) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias,)

    def forward(self, x):
        return self.linear(x)


@dataclass
class MultiModalLinearAdapterConfig:
    in_features: int
    out_features: int
    bias: bool
    _target_: str = "{0}.{1}".format(MultiModalLinearAdapter.__module__, MultiModalLinearAdapter.__name__)
