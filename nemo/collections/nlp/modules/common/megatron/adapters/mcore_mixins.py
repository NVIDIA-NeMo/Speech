# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from megatron.core.models.gpt.gpt_embedding import GPTEmbedding
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import make_viewless_tensor

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    LoraKQVAdapterConfig,
    ParallelLinearAdapterConfig,
    PromptEncoderAdapterConfig,
)
from nemo.core import adapter_mixins


def swap_mcore_mixin(module, mcore_mixin):
    """
    Casts module to mcore_mixin and register corresponding adapters.
    """
    module.__class__ = mcore_mixin
    module.mcore_register_adapters()


class MCoreAdapterModuleMixin(adapter_mixins.AdapterModuleMixin):
    def mcore_register_adapters(self):
        """
        Performs any necessary setup after swapping class.
        Must use self.set_accepted_adapter_types([<NeMo adapter config>_target_]) to register adapter.
        """
        raise NotImplementedError("Mcore mixins should implement setup_adapters on a subclass of MyBase")


class MCoreSelfAttentionMixin(SelfAttention, MCoreAdapterModuleMixin):
    def mcore_register_adapters(self):
        """
        Setup NeMo LoRA adapter to this MCore layer.
        """
        self.set_accepted_adapter_types([LoraKQVAdapterConfig._target_])  # only self attn (packed qkv) for now
        self.linear_qkv.return_layernorm_output = True  # need layernorm output for lora mlp

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        (mixed_qkv, layernorm_output), _ = self.linear_qkv(hidden_states)

        # LoRA logic
        if self.is_adapter_available():
            lora_kqv_adapter = self.get_adapter_module(AdapterName.LORA_KQV_ADAPTER)
            if lora_kqv_adapter:
                lora_mixed_qkv = lora_kqv_adapter(layernorm_output)
                mixed_qkv = mixed_qkv + lora_mixed_qkv

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query, key, value) = torch.split(
            mixed_qkv,
            [
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ],
            dim=3,
        )
        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        return query, key, value


class MCoreGPTEmbeddingMixin(GPTEmbedding, MCoreAdapterModuleMixin):
    def mcore_register_adapters(self):
        """
        Setup NeMo ptuning adapter to this MCore layer.
        """
        self.set_accepted_adapter_types([PromptEncoderAdapterConfig._target_])

    def forward(self, input_ids, position_ids):
        encoder_input = super().forward(input_ids, position_ids)

        if self.is_adapter_available():
            _sq, _bs, _hs = encoder_input.size()
            ptuning_adapter = self.get_adapter_module(AdapterName.PTUNING_ADAPTER)
            v = ptuning_adapter.virtual_tokens
            if ptuning_adapter and _sq >= v:  # The sequence should be longer the v to insert virtual embeddings.
                virtual_embeddings = ptuning_adapter(_bs)
                encoder_input = encoder_input[
                    v:, :, :
                ]  # the first v tokens are pads so that they can be swapped out with virtual embeddings.
                encoder_input = torch.concat([virtual_embeddings, encoder_input], dim=0)
        return encoder_input


class MCoreTransformerLayerMixin(TransformerLayer, MCoreAdapterModuleMixin):
    def mcore_register_adapters(self):
        self.set_accepted_adapter_types([ParallelLinearAdapterConfig._target_])

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        enc_dec_attn_mask=None,
        inference_params=None,
        rotary_pos_emb=None,
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output_with_bias = self.self_attention(
            layernorm_output, attention_mask, inference_params=inference_params, rotary_pos_emb=rotary_pos_emb,
        )

        # adapter logic
        if self.is_adapter_available():
            adapter_1 = self.get_adapter_module(AdapterName.PRE_ATTN_ADAPTER)
            if adapter_1:
                attention_output, bias = attention_output_with_bias
                attention_output = (
                    adapter_1(attention_output) + attention_output
                )  # simple adapter call with residual connection
                attention_output_with_bias = (attention_output, bias)

        # Residual connection.
        if self.config.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # bias_dropout_add fusion returning fp32 instead of bf16
        with self.bias_dropout_add_exec_handler():
            layernorm_input = self.bias_dropout_add_func(
                attention_output_with_bias, residual, self.config.hidden_dropout
            )

        # Layer norm post the self attention.
        layernorm_output = self.post_self_attn_layernorm(layernorm_input)

        # MLP.
        mlp_output_with_bias = self.mlp(layernorm_output)

        # adapter logic
        if self.is_adapter_available():
            adapter_2 = self.get_adapter_module(AdapterName.POST_ATTN_ADAPTER)
            if adapter_2:
                mlp_output, bias = mlp_output_with_bias
                mlp_output = adapter_2(mlp_output) + mlp_output  # simple adapter call with residual connection
                mlp_output_with_bias = (mlp_output, bias)

        # Second residual connection.
        if self.config.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        with self.bias_dropout_add_exec_handler():
            output = self.bias_dropout_add_func(mlp_output_with_bias, residual, self.config.hidden_dropout)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)

        return output
