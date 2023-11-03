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
import torch
import torch.nn.functional as F

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    MLPInfusedAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.fused_bias_geglu import fused_bias_geglu
from nemo.collections.nlp.modules.common.megatron.fused_bias_gelu import fused_bias_gelu
from nemo.collections.nlp.modules.common.megatron.fused_layer_norm import get_layer_norm
from nemo.collections.nlp.modules.common.megatron.layer_norm_1p import LayerNorm1P
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults, ApproxGELUActivation, erf_gelu
from nemo.collections.nlp.modules.common.megatron.utils import openai_gelu as openai_gelu_func
from nemo.collections.nlp.modules.common.megatron.utils import squared_relu
from nemo.core import adapter_mixins

try:
    from apex.normalization import MixedFusedRMSNorm
    from apex.transformer import parallel_state, tensor_parallel

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()


try:
    from megatron.core import ModelParallelConfig, parallel_state, tensor_parallel
    from megatron.core.parallel_state import get_tensor_model_parallel_world_size

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False


class ParallelMLP(MegatronModule, adapter_mixins.AdapterModuleMixin):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        init_method,
        output_layer_init_method,
        hidden_size,
        ffn_hidden_size,
        dtype=torch.float32,
        bias_activation_fusion=True,
        openai_gelu=False,
        onnx_safe=False,
        activation='gelu',
        bias=True,
        transformer_block_type='pre_ln',
        normalization='layernorm',
        layernorm_epsilon=1e-5,
        persist_layer_norm=False,
        dropout=0.0,
    ):
        super(ParallelMLP, self).__init__(config=config)
        self.activation = activation
        self.bias = bias
        self.transformer_block_type = transformer_block_type
        self.normalization = normalization
        self.layernorm_epsilon = layernorm_epsilon
        self.persist_layer_norm = persist_layer_norm
        self.activation = activation
        self.dropout = dropout
        self.dtype = dtype
        self.set_accepted_adapter_types([MLPInfusedAdapterConfig._target_])

        supported_activations = [
            'gelu',
            'geglu',
            'reglu',
            'swiglu',
            'squared-relu',
            'fast-geglu',
            'fast-swiglu',
            'fast-reglu',
            'approx-gelu',
        ]

        if activation not in supported_activations:
            raise ValueError(
                f"Activation {activation} not supported. Supported activations are {supported_activations}"
            )

        self.fast_glu_activation = activation in ['fast-geglu', 'fast-swiglu', 'fast-reglu']

        # Project to 4h.
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size * 2
            if self.fast_glu_activation
            else ffn_hidden_size,  # NOTE: When using geglu, divide ffn dim by 2/3 to keep overall params the same.
            config=config,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=bias,
        )

        if activation in ['geglu', 'reglu', 'swiglu']:
            # Separate linear layer for *GLU activations.
            # Source: https://github.com/huggingface/transformers/blob/bee361c6f1f7704f8c688895f2f86f6e5ff84727/src/transformers/models/t5/modeling_t5.py#L292
            self.dense_h_to_4h_2 = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                ffn_hidden_size,  # NOTE: When using *glu, divide ffn dim by 2/3 to keep overall params the same.
                config=config,
                gather_output=False,
                init_method=init_method,
                skip_bias_add=True,
                bias=bias,
            )

        self.glu_activation_family = activation in [
            'geglu',
            'reglu',
            'swiglu',
            'fast-geglu',
            'fast-reglu',
            'fast-swiglu',
        ]
        bias_activation_fusion_unavailable = activation in ['reglu', 'swiglu']

        if bias_activation_fusion_unavailable and bias_activation_fusion:
            raise ValueError(
                f"Cannot use bias_activation_fusion with {activation} activation. Please turn bias gelu fusion off."
            )

        if self.glu_activation_family and onnx_safe and self.bias_activation_fusion:
            raise ValueError(
                f"Cannot use onnx_safe with specificed activation function and bias_activation_fusion : {activation} Please turn onnx safe off."
            )

        if bias_activation_fusion and not bias:
            raise ValueError(
                f"Cannot use bias_activation_fusion without bias terms. Please set bias=True or bias_activation_fusion=False."
            )

        self.bias_activation_fusion = bias_activation_fusion

        # Give openai_gelu precedence over other activations if set, for HF compatibility. Normally this is off and shouldn't affect regular model training.
        if openai_gelu:
            self.activation_func = openai_gelu_func
        elif activation in ["gelu", "geglu", "fast-geglu"]:
            self.activation_func = F.gelu
        elif activation == 'approx-gelu':
            self.activation_func = ApproxGELUActivation
        elif onnx_safe:
            self.activation_func = erf_gelu
        elif activation in ["reglu", "fast-reglu"]:
            self.activation_func = F.relu
        elif activation in ["swiglu", "fast-swiglu"]:
            # SiLU or sigmoid linear unit is the same as swish with beta = 1 (which is what https://arxiv.org/pdf/2002.05202.pdf uses.)
            self.activation_func = F.silu
        elif activation == 'squared-relu':
            self.activation_func = squared_relu

        # Project back to h.
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            config=config,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            bias=bias,
        )

        # Normformer normalization
        if transformer_block_type == 'normformer':
            if normalization == 'layernorm':
                self.normalization = get_layer_norm(
                    ffn_hidden_size // get_tensor_model_parallel_world_size(), layernorm_epsilon, persist_layer_norm
                )
            elif normalization == 'layernorm1p':
                self.normalization = LayerNorm1P(
                    ffn_hidden_size // get_tensor_model_parallel_world_size(),
                    layernorm_epsilon,
                    sequence_parallel_enabled=config.sequence_parallel,
                )
            else:
                self.normalization = MixedFusedRMSNorm(
                    ffn_hidden_size // get_tensor_model_parallel_world_size(), layernorm_epsilon
                )

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.fast_glu_activation:
            intermediate_parallel, intermediate_parallel_2 = torch.chunk(intermediate_parallel, 2, dim=-1)
            if bias_parallel is not None:
                bias_parallel, bias_parallel_2 = torch.chunk(bias_parallel, 2, dim=-1)
        elif self.glu_activation_family and not self.fast_glu_activation:
            intermediate_parallel_2, bias_parallel_2 = self.dense_h_to_4h_2(hidden_states)

        if self.bias_activation_fusion:
            if self.activation == 'gelu':
                intermediate_parallel = fused_bias_gelu(intermediate_parallel, bias_parallel)
            elif self.activation in ['geglu', 'fast-geglu']:
                intermediate_parallel = fused_bias_geglu(
                    intermediate_parallel, bias_parallel, intermediate_parallel_2, bias_parallel_2
                )

        elif self.glu_activation_family and not self.bias_activation_fusion:
            if bias_parallel is not None:
                intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel) * (
                    intermediate_parallel_2 + bias_parallel_2
                )
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel) * intermediate_parallel_2

        else:
            if bias_parallel is not None:
                intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

        if self.dropout > 0:
            intermediate_parallel = F.dropout(intermediate_parallel, p=self.dropout, training=self.training)

        infused_adapter = self.get_adapter_module(AdapterName.MLP_INFUSED)
        if infused_adapter:
            intermediate_parallel = infused_adapter(intermediate_parallel)

        # Normformer normalization
        if self.transformer_block_type == 'normformer':
            intermediate_parallel = self.normalization(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class SwitchMLP(MegatronModule):
    """Top-1 MoE
    
    Curently supports Sinkhorn based expert routing."""

    def __init__(
        self,
        config: ModelParallelConfig,
        num_experts,
        init_method,
        output_layer_init_method,
        hidden_size,
        ffn_hidden_size,
        dtype=torch.float32,
        bias_activation_fusion=True,
        openai_gelu=False,
        onnx_safe=False,
        activation='gelu',
        bias=True,
        transformer_block_type='pre_ln',
        normalization='layernorm',
        layernorm_epsilon=1e-5,
        persist_layer_norm=False,
        sequence_parallel=False,
        dropout=0.0,
    ):
        super(SwitchMLP, self).__init__(config=config)

        self.num_experts = num_experts
        self.route_algo = SwitchMLP.sinkhorn
        self.router = tensor_parallel.RowParallelLinear(
            hidden_size,
            num_experts,
            config=config,
            input_is_parallel=False,
            init_method=init_method,
            skip_bias_add=False,
            bias=bias,
        )

        mlp_args = {
            'config': config,
            'init_method': init_method,
            'output_layer_init_method': output_layer_init_method,
            'hidden_size': hidden_size,
            'ffn_hidden_size': ffn_hidden_size,
            'dtype': dtype,
            'bias_activation_fusion': bias_activation_fusion,
            'openai_gelu': openai_gelu,
            'onnx_safe': onnx_safe,
            'activation': activation,
            'bias': bias,
            'transformer_block_type': transformer_block_type,
            'normalization': normalization,
            'layernorm_epsilon': layernorm_epsilon,
            'persist_layer_norm': persist_layer_norm,
            'dropout': dropout,
        }
        self.experts = torch.nn.ModuleList([ParallelMLP(**mlp_args) for _ in range(num_experts)])

    def forward(self, hidden_states):
        hidden_shape = hidden_states.shape
        route, _ = self.router(hidden_states)
        route = route.view(-1, self.num_experts)
        if self.training:
            with torch.no_grad():
                norm_route = self.route_algo(
                    route.detach().to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, max_ind = torch.max(norm_route, dim=1)
            route = torch.sigmoid(route)
            max_prob = route[torch.arange(route.size(0)), max_ind]
        else:
            route = torch.sigmoid(route)
            max_prob, max_ind = torch.max(route, dim=1)
        max_prob = torch.unsqueeze(max_prob, 1)

        hidden_states = hidden_states.view(-1, hidden_shape[-1])

        local_indices = (max_ind == 0).nonzero()
        hidden = hidden_states[local_indices, :]
        output, output_bias = self.experts[0](hidden)
        output_bias = output_bias.expand_as(output)

        output_total = torch.empty_like(hidden_states, dtype=output.dtype)
        output_bias_total = torch.empty_like(hidden_states, dtype=output_bias.dtype)

        output_total[local_indices, :] = output
        output_bias_total[local_indices, :] = output_bias

        for expert_num, expert in enumerate(self.experts):
            if expert_num == 0:
                continue
            local_indices = (max_ind == expert_num).nonzero()
            hidden = hidden_states[local_indices, :]
            output, output_bias = expert(hidden)
            output_bias = output_bias.expand_as(output)
            output_total[local_indices, :] = output
            output_bias_total[local_indices, :] = output_bias

        output_total = output_total * max_prob
        output_bias_total = output_bias_total * max_prob
        output_total = output_total.view(hidden_shape)
        output_bias_total = output_bias_total.view(hidden_shape)

        return output_total, output_bias_total

    @classmethod
    def sinkhorn(cls, cost, tol=0.0001):
        "Megatron-LMs sinkhorn implementation"

        cost = torch.exp(cost)
        d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
        d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

        eps = 0.00000001
        error = 1e9
        d1_old = d1
        while error > tol:
            d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
            d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
            error = torch.mean(torch.abs(d1_old - d1))
            d1_old = d1
        return d1 * cost * d0.unsqueeze(1)
