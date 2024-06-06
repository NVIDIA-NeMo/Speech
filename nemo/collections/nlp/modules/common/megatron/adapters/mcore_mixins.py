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
import torch.nn.functional as F
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.custom_layers.transformer_engine import (
    SplitAlongDim,
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
)
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import make_viewless_tensor

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    Lora4HtoHAdapterConfig,
    LoraDenseAttentionAdapterConfig,
    LoraHto4HAdapterConfig,
    LoraKQVAdapterConfig,
    LoraUnfusedHto4HAdapterConfig,
    LoraUnfusedKQVAdapterConfig,
    MLPInfusedAdapterConfig,
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
        Setup NeMo LoRA or IA3 adapter to this MCore layer.
        """
        self.set_accepted_adapter_types(
            [
                LoraUnfusedKQVAdapterConfig._target_,
                LoraKQVAdapterConfig._target_,
                LoraDenseAttentionAdapterConfig._target_,
                InfusedAdapterConfig._target_,
            ]
        )
        self.linear_qkv.return_layernorm_output = True  # need layernorm output for lora mlp
        if (
            self.config.sequence_parallel
            and hasattr(self.linear_qkv, "return_layernorm_output_gathered")
            and not self.linear_qkv.ub_overlap_ag
        ):
            # for LoRA SP, TE v1.5 can return layernorm output gathered so there is no need
            # to perform the redundant gather in the adapter module, unless TP communication
            # overlap is used.
            self.linear_qkv.return_layernorm_output_gathered = True

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        linear_qkv_output, _ = self.linear_qkv(hidden_states)
        layernorm_output = None

        # In megatron/core/models/gpt/gpt_layer_specs.py when fused module is used(e.g. TELayerNormColumnParallelLinear)
        # both LN and qkv will be returned.
        # In nemo/collections/nlp/models/language_modeling/megatron/falcon/falcon_spec.py TEColumnParallelLinear(non-fused) is used for linear_qkv,
        # which only returns linear.
        if isinstance(linear_qkv_output, tuple):
            if len(linear_qkv_output) == 2:  # fused module, qkv&LN
                mixed_qkv, layernorm_output = linear_qkv_output
            else:
                raise ValueError(f"Unexpected number of outputs from linear_qkv output: {len(linear_qkv_output)}")
        else:  # for qkv&LN not fused only mixed_qkv
            mixed_qkv = linear_qkv_output

        # LoRA logic
        if self.is_adapter_available():
            lora_kqv_adapter = self.get_adapter_module(AdapterName.LORA_KQV_ADAPTER)
            if lora_kqv_adapter and self.adapter_cfg[AdapterName.LORA_KQV_ADAPTER]['enabled']:
                if layernorm_output is not None:
                    lora_mixed_qkv = lora_kqv_adapter(layernorm_output)
                else:
                    lora_mixed_qkv = lora_kqv_adapter(hidden_states)

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
        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(
                mixed_qkv,
                3,
                split_arg_list,
            )
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(
                mixed_qkv,
                split_arg_list,
                dim=3,
            )

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.is_adapter_available():
            key_infused_adapter = self.get_adapter_module(AdapterName.KEY_INFUSED)
            value_infused_adapter = self.get_adapter_module(AdapterName.VALUE_INFUSED)
            if key_infused_adapter and self.adapter_cfg[AdapterName.KEY_INFUSED]['enabled']:
                assert value_infused_adapter is not None, "Expected value_infused_adapter not found!"
                kls = key.shape
                key = key_infused_adapter(key.reshape(kls[0], kls[1], -1)).reshape(kls).to(query.dtype)
            if value_infused_adapter and self.adapter_cfg[AdapterName.VALUE_INFUSED]['enabled']:
                assert key_infused_adapter is not None, "Expected key_infused_adapter not found!"
                vls = value.shape
                value = value_infused_adapter(value.reshape(vls[0], vls[1], -1)).reshape(vls).to(query.dtype)

            lora_unfused_kqv_adapter = self.get_adapter_module(AdapterName.LORA_UNFUSED_KQV_ADAPTER)
            if lora_unfused_kqv_adapter and self.adapter_cfg[AdapterName.LORA_UNFUSED_KQV_ADAPTER]['enabled']:
                assert lora_kqv_adapter is None
                if layernorm_output is not None:
                    lq, lk, lv = lora_unfused_kqv_adapter(layernorm_output)
                else:
                    lq, lk, lv = lora_unfused_kqv_adapter(hidden_states)
                query, key, value = query + lq, key + lk, value + lv
        return query, key, value

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ):
        # hidden_states: [sq, b, h]

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, key, value, rotary_pos_emb
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)
        # LoRA logic
        if self.is_adapter_available():
            lora_linear_proj_adapter = self.get_adapter_module(AdapterName.LORA_DENSE_ATTENTION_ADAPTER)
            if lora_linear_proj_adapter and self.adapter_cfg[AdapterName.LORA_DENSE_ATTENTION_ADAPTER]['enabled']:
                lora_output = lora_linear_proj_adapter(core_attn_out)
                output = output + lora_output

        return output, bias


class MCoreMLPMixin(MLP, MCoreAdapterModuleMixin):
    def mcore_register_adapters(self):
        """
        Setup NeMo IA3 adapter to this MCore layer.
        """
        self.set_accepted_adapter_types(
            [
                LoraUnfusedHto4HAdapterConfig._target_,
                LoraHto4HAdapterConfig._target_,
                Lora4HtoHAdapterConfig._target_,
                MLPInfusedAdapterConfig._target_,
            ]
        )  # only self attn (packed qkv) for now
        self.linear_fc1.return_layernorm_output = True  # need layernorm output for lora mlp
        if (
            self.config.sequence_parallel
            and hasattr(self.linear_fc1, "return_layernorm_output_gathered")
            and not self.config.tp_comm_overlap
        ):
            # for LoRA SP, TE v1.5 can return layernorm output gathered so there is no need
            # to perform the redundant gather in the adapter module, unless TP communication
            # overlap is used.
            self.linear_fc1.return_layernorm_output_gathered = True

    def forward(self, hidden_states):
        # [s, b, 4 * h/p]
        if self.linear_fc1.te_return_bias:
            intermediate_parallel, bias_parallel, layernorm_output = self.linear_fc1(hidden_states)
        else:
            # bias_parallel is None
            (intermediate_parallel, layernorm_output), bias_parallel = self.linear_fc1(hidden_states)

        # LoRA logic
        if self.is_adapter_available():
            lora_adapter = None
            lora_fc1_adapter = self.get_adapter_module(AdapterName.LORA_Hto4H_ADAPTER)
            lora_unfused_fc1_adapter = self.get_adapter_module(AdapterName.LORA_UNFUSED_Hto4H_ADAPTER)
            if lora_fc1_adapter and self.adapter_cfg[AdapterName.LORA_Hto4H_ADAPTER]['enabled']:
                lora_adapter = lora_fc1_adapter
            if lora_unfused_fc1_adapter and self.adapter_cfg[AdapterName.LORA_UNFUSED_Hto4H_ADAPTER]['enabled']:
                assert lora_adapter is None, "Expected only one of LORA_Hto4H_ADAPTER or LORA_UNFUSED_Hto4H_ADAPTER"
                lora_adapter = lora_unfused_fc1_adapter

            if lora_adapter:
                lora_output = lora_adapter(layernorm_output)
                intermediate_parallel = intermediate_parallel + lora_output

        if self.config.bias_activation_fusion:
            if self.activation_func == F.gelu:
                if self.config.gated_linear_unit:
                    intermediate_parallel = bias_geglu_impl(intermediate_parallel, bias_parallel)
                else:
                    assert self.config.add_bias_linear is True
                    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
            elif self.activation_func == F.silu and self.config.gated_linear_unit:
                intermediate_parallel = bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    self.config.activation_func_fp8_input_store,
                )

            else:
                raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

        infused_adapter = self.get_adapter_module(AdapterName.MLP_INFUSED)
        if infused_adapter:
            intermediate_parallel = infused_adapter(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)

        # LoRA logic
        if self.is_adapter_available():
            lora_linear_fc2_adapter = self.get_adapter_module(AdapterName.LORA_4HtoH_ADAPTER)
            if lora_linear_fc2_adapter and self.adapter_cfg[AdapterName.LORA_4HtoH_ADAPTER]['enabled']:
                lora_output = lora_linear_fc2_adapter(intermediate_parallel)
                output = output + lora_output

        return output, output_bias


class MCoreGPTEmbeddingMixin(LanguageModelEmbedding, MCoreAdapterModuleMixin):
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
            if (
                ptuning_adapter and self.adapter_cfg[AdapterName.PTUNING_ADAPTER]['enabled'] and _sq >= v
            ):  # The sequence should be longer the v to insert virtual embeddings.
                virtual_embeddings = ptuning_adapter(_bs)
                encoder_input = encoder_input[
                    v:, :, :
                ]  # the first v tokens are pads so that they can be swapped out with virtual embeddings.
                encoder_input = torch.concat([virtual_embeddings, encoder_input], dim=0)
        return encoder_input


class MCoreTransformerLayerMixin(TransformerLayer, MCoreAdapterModuleMixin):
    def mcore_register_adapters(self):
        """
        Setup NeMo (canonical) Adapter to this MCore layer.
        """
        self.set_accepted_adapter_types([ParallelLinearAdapterConfig._target_])

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
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

        # adapter logic
        if self.is_adapter_available():
            adapter_1 = self.get_adapter_module(AdapterName.PRE_ATTN_ADAPTER)
            if adapter_1 and self.adapter_cfg[AdapterName.PRE_ATTN_ADAPTER]['enabled']:
                attention_output, bias = attention_output_with_bias
                attention_output = (
                    adapter_1(attention_output) + attention_output
                )  # simple adapter call with residual connection
                attention_output_with_bias = (attention_output, bias)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

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

        # adapter logic
        if self.is_adapter_available():
            adapter_2 = self.get_adapter_module(AdapterName.POST_ATTN_ADAPTER)
            if adapter_2 and self.adapter_cfg[AdapterName.POST_ATTN_ADAPTER]['enabled']:
                mlp_output, bias = mlp_output_with_bias
                mlp_output = adapter_2(mlp_output) + mlp_output  # simple adapter call with residual connection
                mlp_output_with_bias = (mlp_output, bias)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        return output, context
