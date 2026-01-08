# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass

from nemo.utils.megatron_utils import ApexGuardDefaults

try:
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec, build_module
    from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
    from megatron.core.utils import make_viewless_tensor

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    TransformerConfig = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False

try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@dataclass
class TransformerLayerSubmodulesWithPostLNSupport(TransformerLayerSubmodules):
    """TransformerLayerSubmodules with post layer norm"""

    def __init__(self, post_att_layernorm, post_mlp_layernorm, **kwargs):
        super(TransformerLayerSubmodulesWithPostLNSupport, self).__init__(**kwargs)
        self.post_att_layernorm = post_att_layernorm
        self.post_mlp_layernorm = post_mlp_layernorm


class TransformerLayerWithPostLNSupport(TransformerLayer):
    """TransformerLayer with post layer norm."""

    def __init__(self, *args, **kwargs):
        super(TransformerLayerWithPostLNSupport, self).__init__(*args, **kwargs)
        # [Module add: Post attention LN]
        self.post_att_layernorm = build_module(
            self.submodules_config.post_att_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # [Module add: Post MLP LN]
        self.post_mlp_layernorm = build_module(
            self.submodules_config.post_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        **kwargs,
    ):
        """Copy from megatron/core/transformer/transformer_layer.py with modification of applying
        extra post layer norm if needed."""
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Post-LN after Self Attention
        hidden_states = self.post_att_layernorm(hidden_states)

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Post-LN after MLP
        hidden_states = self.post_mlp_layernorm(hidden_states)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        return output, context


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_bert_layer_with_transformer_engine_spec_postln():
    """Retrieve the Layer Spec when using Transformer Engine"""
    return ModuleSpec(
        module=TransformerLayerWithPostLNSupport,
        submodules=TransformerLayerSubmodulesWithPostLNSupport(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            post_att_layernorm=TENorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            post_mlp_layernorm=TENorm,
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_bert_layer_local_spec_postln():
    """Retrieve the Layer Spec when using MCore Engine"""
    return ModuleSpec(
        module=TransformerLayerWithPostLNSupport,
        submodules=TransformerLayerSubmodulesWithPostLNSupport(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            post_att_layernorm=FusedLayerNorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            post_mlp_layernorm=FusedLayerNorm,
        ),
    )
