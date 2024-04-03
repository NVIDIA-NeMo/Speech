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
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    InfusedAdapterConfig,
    LoraDenseAttentionAdapterConfig,
    LoraKQVAdapterConfig,
    LoraKQVAdapterWeightTyingConfig,
    LoraKVAdapterConfig,
    LoraQAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.fused_softmax import MatchedScaleMaskSoftmax
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.position_embedding import XPOSPositionEmbedding
from nemo.collections.nlp.modules.common.megatron.position_embedding.rotary_position_embedding import (
    apply_rotary_pos_emb,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    _cast_if_autocast_enabled,
    attention_mask_func,
)
from nemo.core import adapter_mixins

try:
    from apex.transformer.enums import AttnMaskType, AttnType
    from apex.transformer.utils import divide as safe_divide

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()


try:
    from megatron.core import ModelParallelConfig, parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False

try:
    # Flash Attention Triton
    import pkg_resources
    from flash_attn.flash_attn_triton import flash_attn_func as flash_attn_func_triton

except (ImportError, ModuleNotFoundError, pkg_resources.DistributionNotFound):

    flash_attn_func_triton = None


try:
    # Flash Attention 1.X
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func

    HAVE_FLASH_ATTENTION = True
    flash_attn_func = None

except (ImportError, ModuleNotFoundError):
    try:
        # Flash Attention 2.X
        from flash_attn import flash_attn_func
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func

        HAVE_FLASH_ATTENTION = True

    except (ImportError, ModuleNotFoundError):

        HAVE_FLASH_ATTENTION = False
        flash_attn_unpadded_func, flash_attn_func = None, None
        unpad_input, pad_input = None, None

    try:
        # Flash Attention 2.2
        from flash_attn import flash_attn_with_kvcache

    except (ImportError, ModuleNotFoundError):

        flash_attn_with_kvcache = None

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


class ParallelAttention(MegatronModule, adapter_mixins.AdapterModuleMixin):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        init_method,
        output_layer_init_method,
        layer_number,
        num_attention_heads,
        hidden_size,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
        precision=16,
        apply_query_key_layer_scaling=False,
        kv_channels=None,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        layer_type=None,
        megatron_legacy=False,
        bias=True,
        headscale=False,
        position_embedding_type='learned_absolute',
        multi_query_attention=False,
        normalize_attention_scores=True,
        use_flash_attention=False,
    ):
        super(ParallelAttention, self).__init__(config=config)
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.normalize_attention_scores = normalize_attention_scores
        self.position_embedding_type = position_embedding_type
        self.multi_query_attention = multi_query_attention
        self.use_flash_attention = use_flash_attention

        self.megatron_legacy = megatron_legacy

        self.set_accepted_adapter_types(
            [
                InfusedAdapterConfig._target_,
                LoraKQVAdapterConfig._target_,
                LoraQAdapterConfig._target_,
                LoraKVAdapterConfig._target_,
                LoraKQVAdapterWeightTyingConfig._target_,
                LoraDenseAttentionAdapterConfig._target_,
            ]
        )

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads
        projection_size = kv_channels * num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = safe_divide(projection_size, num_attention_heads)
        self.num_attention_heads_per_partition = safe_divide(num_attention_heads, world_size)
        self.num_attention_heads_partition_offset = (
            self.num_attention_heads_per_partition * parallel_state.get_tensor_model_parallel_rank()
        )

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                3 * projection_size,
                config=config,
                gather_output=False,
                init_method=init_method,
                bias=bias,
            )
        else:
            assert attention_type == AttnType.cross_attn
            self.query = tensor_parallel.ColumnParallelLinear(
                hidden_size, projection_size, config=config, gather_output=False, init_method=init_method, bias=bias,
            )

            self.key_value = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                2 * projection_size,
                config=config,
                gather_output=False,
                init_method=init_method,
                bias=bias,
            )

        self.core_attention = CoreAttention(
            config=config,
            layer_number=self.layer_number,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_type=self.attention_type,
            attn_mask_type=self.attn_mask_type,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            multi_query_attention=multi_query_attention,
            normalize_attention_scores=normalize_attention_scores,
            position_embedding_type=position_embedding_type,
            use_flash_attention=use_flash_attention,
        )

        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            hidden_size,
            config=config,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            bias=bias,
        )

        self.headscale = headscale
        if headscale:
            self.head_scale_tensor = torch.nn.Parameter(
                torch.ones(1, self.num_attention_heads_per_partition, 1, 1), requires_grad=True
            )

        # Inference key-value memory
        self.inference_key_memory = None
        self.inference_value_memory = None
        self.inference_current_sequence_len = 0

        # relative position embedding
        self.layer_type = layer_type

    def _checkpointed_attention_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        rotary_pos_emb=None,
        relative_position_bias=None,
        headscale_tensor=None,
        inference_mode=None,
    ):
        """Forward method with activation checkpointing."""

        def custom_forward(*inputs):
            if len(inputs) == 7:
                query_layer = inputs[0]
                key_layer = inputs[1]
                value_layer = inputs[2]
                attention_mask = inputs[3]
                rotary_pos_emb = inputs[4]
                relative_position_bias = inputs[5]
                headscale_tensor = inputs[6]
            elif len(inputs) == 8:
                query_layer = inputs[0]
                key_layer = inputs[1]
                value_layer = inputs[2]
                attention_mask = inputs[3]
                rotary_pos_emb = (inputs[4], inputs[5])
                relative_position_bias = inputs[6]
                headscale_tensor = inputs[7]
            else:
                raise ValueError('unexpected number of inputs')
            output_ = self.core_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=headscale_tensor,
                inference_mode=inference_mode,
            )
            return output_

        if rotary_pos_emb is None:
            rot_tuple = (rotary_pos_emb,)
        else:
            rot_tuple = (rotary_pos_emb[0], rotary_pos_emb[1])

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            *rot_tuple,
            relative_position_bias,
            headscale_tensor,
        )

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size, dtype, device):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def _transpose_last_dim(self, mixed_layer, num_splits, num_splits_first):
        input_shape = mixed_layer.size()
        if num_splits_first:
            """[s, b, num_splits * np * hn]
            -->(view) [s, b, num_splits, np, hn]
            -->(tranpose) [s, b, np, num_splits, hn]
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] + (
                num_splits,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-2, -3).contiguous()
        else:
            """[s, b, np * hn * num_splits]
            -->(view) [s, b, np, hn, num_splits]
            -->(tranpose) [s, b, np, num_splits, hn]
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
                num_splits,
            )

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-1, -2).contiguous()
        mixed_layer = mixed_layer.view(*input_shape)

        return mixed_layer

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        encoder_output=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        rotary_pos_emb=None,  # rotary positional embedding
        relative_position_bias=None,
        checkpoint_core_attention=False,
    ):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if set_inference_key_value_memory:
            assert inference_max_sequence_len and inference_max_sequence_len > 0
            self.inference_key_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_value_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype, hidden_states.device
            )
            self.inference_current_sequence_len = 0

        # Some consistency check.
        if inference_max_sequence_len:
            assert self.inference_current_sequence_len < self.inference_key_memory.size(0)
            assert inference_max_sequence_len == self.inference_key_memory.size(0)
        # This is added for safety. In case inference_max_sequence_len
        # is not provided, make sure there is no potential memory left
        # from previous inference.
        if not inference_max_sequence_len:
            self.inference_key_memory = None
            self.inference_value_memory = None

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)
            if self.is_adapter_available():
                lora_kqv_adapter = self.get_adapter_module(AdapterName.LORA_KQV_ADAPTER)
                if lora_kqv_adapter and self.adapter_cfg[AdapterName.LORA_KQV_ADAPTER]['enabled']:
                    lora_mixed_x_layer = lora_kqv_adapter(hidden_states)
                    mixed_x_layer = mixed_x_layer + lora_mixed_x_layer

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            if self.megatron_legacy:
                mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, True)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(
                mixed_x_layer, 3, contiguous_split_chunks=True
            )
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)
            if self.is_adapter_available():
                lora_kv_adapter = self.get_adapter_module(AdapterName.LORA_KV_ADAPTER)
                if lora_kv_adapter and self.adapter_cfg[AdapterName.LORA_KV_ADAPTER]['enabled']:
                    lora_mixed_kv_layer = lora_kv_adapter(encoder_output)
                    mixed_kv_layer = mixed_kv_layer + lora_mixed_kv_layer

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            if self.megatron_legacy:
                mixed_kv_layer = self._transpose_last_dim(mixed_kv_layer, 2, True)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(
                mixed_kv_layer, 2, contiguous_split_chunks=True
            )

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            if self.is_adapter_available():
                lora_q_adapter = self.get_adapter_module(AdapterName.LORA_Q_ADAPTER)
                if lora_q_adapter and self.adapter_cfg[AdapterName.LORA_Q_ADAPTER]['enabled']:
                    lora_q_layer = lora_q_adapter(hidden_states)
                    query_layer = query_layer + lora_q_layer
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        if self.is_adapter_available():
            key_infused_adapter = self.get_adapter_module(AdapterName.KEY_INFUSED)
            value_infused_adapter = self.get_adapter_module(AdapterName.VALUE_INFUSED)
            if key_infused_adapter and self.adapter_cfg[AdapterName.KEY_INFUSED]['enabled']:
                assert value_infused_adapter is not None, "Expected value_infused_adapter not found!"
                kls = key_layer.shape
                key_layer = key_infused_adapter(key_layer.reshape(kls[0], kls[1], -1)).reshape(kls)
            if value_infused_adapter and self.adapter_cfg[AdapterName.VALUE_INFUSED]['enabled']:
                assert key_infused_adapter is not None, "Expected key_infused_adapter not found!"
                vls = value_layer.shape
                value_layer = value_infused_adapter(value_layer.reshape(vls[0], vls[1], -1)).reshape(vls)

        # ===================================================
        # Adjust key, value, and attention mask for inference
        # ===================================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            rotary_pos_emb = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else ((rotary_pos_emb,) * 2)

        if inference_max_sequence_len:
            # Adjust the range variables.
            start = self.inference_current_sequence_len
            self.inference_current_sequence_len += key_layer.size(0)
            end = self.inference_current_sequence_len
            # Copy key and values.
            self.inference_key_memory[start:end, ...] = key_layer
            self.inference_value_memory[start:end, ...] = value_layer
            key_layer = self.inference_key_memory[:end, ...]
            value_layer = self.inference_value_memory[:end, ...]
            # Adjust attention mask
            if attention_mask is not None:
                attention_mask = attention_mask[..., start:end, :end]
            # adjust the key rotary positional embedding
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                if not set_inference_key_value_memory:
                    # In inference, we compute one token at a time.
                    # Select the correct positional embedding.
                    q_pos_emb = q_pos_emb[end - 1 : end]
                else:
                    q_pos_emb = q_pos_emb[:end, :, :, :]
                k_pos_emb = k_pos_emb[:end, :, :, :]
                rotary_pos_emb = (q_pos_emb, k_pos_emb)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)

        if get_key_value:
            present = (key_layer, value_layer)

        if (
            flash_attn_with_kvcache is not None
            and self.use_flash_attention
            and rotary_pos_emb is not None
            and inference_max_sequence_len
            and not set_inference_key_value_memory
        ):
            # Mainly used for decoding with sq=1
            q = _cast_if_autocast_enabled(
                rearrange(apply_rotary_pos_emb(query_layer, rotary_pos_emb[0]), 'sq b np hn -> b sq np hn')
            )
            k = _cast_if_autocast_enabled(
                rearrange(apply_rotary_pos_emb(key_layer, rotary_pos_emb[1]), 'sk b np hn -> b sk np hn')
            )
            v = _cast_if_autocast_enabled(rearrange(value_layer, 'sk b np hn -> b sk np hn'))
            context_layer = flash_attn_with_kvcache(
                q=q, k_cache=k, v_cache=v, causal=self.attn_mask_type == AttnMaskType.causal,
            )
            context_layer = rearrange(context_layer, 'b sq np hn -> sq b (np hn)')

        elif checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=self.head_scale_tensor if self.headscale else None,
                inference_mode=inference_max_sequence_len is not None and query_layer.shape[0] == 1,
            )
        else:
            context_layer = self.core_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                layer_past=layer_past,
                get_key_value=get_key_value,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=self.head_scale_tensor if self.headscale else None,
                inference_mode=inference_max_sequence_len is not None and query_layer.shape[0] == 1,
            )

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)
        if self.is_adapter_available():
            lora_dense_adapter = self.get_adapter_module(AdapterName.LORA_DENSE_ATTENTION_ADAPTER)
            if lora_dense_adapter and self.adapter_cfg[AdapterName.LORA_DENSE_ATTENTION_ADAPTER]['enabled']:
                lora_dense_output = lora_dense_adapter(context_layer)
                output = output + lora_dense_output

        if get_key_value:
            output = [output, present]

        return output, bias


class ParallelChunkedCrossAttention(MegatronModule):
    """Parallel chunked cross-attention layer class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        init_method,
        output_layer_init_method,
        layer_number,
        num_attention_heads,
        hidden_size,
        precision=16,
        apply_query_key_layer_scaling=False,
        kv_channels=None,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        megatron_legacy=False,
        chunk_size=64,  # each chunk, how many tokens
        bias=True,
        headscale=False,
        normalize_attention_scores=True,
    ):
        super(ParallelChunkedCrossAttention, self).__init__(config=config)
        self.cross_attention = ParallelAttention(
            config=config,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_type=AttnType.cross_attn,
            attn_mask_type=AttnMaskType.padding,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            megatron_legacy=megatron_legacy,
            bias=bias,
            headscale=headscale,
            normalize_attention_scores=normalize_attention_scores,
        )
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        rotary_pos_emb=None,
        checkpoint_core_attention=False,
    ):
        if checkpoint_core_attention:
            raise ValueError(
                'checkpoint_core_attention during forward not implemented yet for ParallelChunkedCrossAttention'
            )

        # hidden_states is assumed to have dimension [token length, batch, dimension]
        # derive variables
        # encoder_output here is the retrieved context
        context = encoder_output
        # context is assumed to have dimension [num_chunks, num_neighbors, context_token_len, batch, dimension]
        chunk_size = self.chunk_size
        b, n, dim = (
            hidden_states.shape[1],
            hidden_states.shape[0],
            hidden_states.shape[2],
        )
        default_bias = self.cross_attention.dense.bias
        if set_inference_key_value_memory:
            seq_index = (n // chunk_size) * chunk_size
            self.current_len = n
        elif inference_max_sequence_len is not None:
            # only handles single token increment
            assert n == 1
            self.current_len += n
            chunk_id = self.current_len // chunk_size
            if chunk_id <= 0:
                # if sequence length less than chunk size, do an early return
                return torch.zeros_like(hidden_states), default_bias
            causal_padding = chunk_size - 1
            # pad it as a full chunk, put it at the end of the chunk position
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, causal_padding, 0), value=0.0)
            # only use the relevant context
            context = context[chunk_id - 1 : chunk_id, :, :, :, :]
            attention_mask = rearrange(attention_mask, '(b k) 1 q v -> b k 1 q v', b=b)
            # select the relevant chunk attn mask
            attention_mask = attention_mask[:, chunk_id - 1]
            seq_index = chunk_size
        else:
            # this is normal forward without inference
            seq_index = (n // chunk_size) * chunk_size

        # if sequence length less than chunk size, do an early return
        if n < self.chunk_size and set_inference_key_value_memory and inference_max_sequence_len is not None:
            return torch.zeros_like(hidden_states), default_bias

        num_chunks, num_retrieved = (
            context.shape[-5],
            context.shape[-4],
        )

        # causal padding
        causal_padding = chunk_size - 1

        x = F.pad(hidden_states, (0, 0, 0, 0, -causal_padding, causal_padding), value=0.0)

        # remove sequence which is ahead of the neighbors retrieved (during inference)

        # seq_index = (n // chunk_size) * chunk_size
        x, x_remainder = x[:seq_index], x[seq_index:]

        seq_remain_len = x_remainder.shape[0]

        # take care of rotary positional embedding
        # make sure queries positions are properly shifted to the future

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # currently implementation is broken
            # q need to extend to causal_padding, and just do
            # q_pos_emb = F.pad(q_pos_emb, (0, 0, -causal_padding, 0), value = 0.)
            if inference_max_sequence_len is not None and not set_inference_key_value_memory:
                token_pos = (self.current_len - 1) % chunk_size
                q_pos_emb = F.pad(
                    q_pos_emb, (0, 0, 0, 0, 0, 0, -causal_padding - token_pos, -causal_padding + token_pos), value=0.0
                )
            else:
                q_pos_emb = F.pad(q_pos_emb, (0, 0, 0, 0, 0, 0, -causal_padding, 0), value=0.0)

            k_pos_emb = repeat(k_pos_emb, 'n b h d -> (r n) b h d', r=num_retrieved)
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        # make sure number context chunks is enough
        assert x.shape[0] // chunk_size == num_chunks

        # reshape so we have chunk to chunk attention, without breaking causality
        x = rearrange(x, '(k n) b d -> n (b k) d', k=num_chunks)
        context = rearrange(context, 'k r n b d -> (r n) (b k) d')
        # cross attention
        out, bias = self.cross_attention(x, attention_mask, encoder_output=context, rotary_pos_emb=rotary_pos_emb)

        # reshape back to original sequence

        out = rearrange(out, 'n (b k) d -> (k n) b d', b=b)

        # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)

        out = F.pad(out, (0, 0, 0, 0, causal_padding, -causal_padding + seq_remain_len), value=0.0)
        if not set_inference_key_value_memory and inference_max_sequence_len is not None:
            out = out[-1:]
        return out, bias


class CoreAttention(MegatronModule):
    """ Region where selective activation recomputation is applied.
        See Figure 3. in Reducing Activation Recomputation in Large Transformer Models
        https://arxiv.org/pdf/2205.05198.pdf for more details.

    """

    def __init__(
        self,
        config: ModelParallelConfig,
        layer_number,
        num_attention_heads,
        hidden_size,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
        precision=16,
        apply_query_key_layer_scaling=False,
        kv_channels=None,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        normalize_attention_scores=True,
        multi_query_attention=False,
        position_embedding_type='learned_absolute',
        use_flash_attention=False,
    ):

        super(CoreAttention, self).__init__(config=config)

        self.precision = precision
        self.fp16 = False
        self.bf16 = False
        if precision in ['bf16', 'bf16-mixed']:
            self.bf16 = True
        elif precision in [16, '16', '16-mixed']:
            self.fp16 = True
        self.multi_query_attention = multi_query_attention
        self.position_embedding_type = position_embedding_type

        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = False
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = config.sequence_parallel
        # If True, will scale attention scores by 1 / sqrt(hidden_size_per_attention_head).
        # This arg is been provided mostly to support weight conversion of Huggingface models. (ex: T5v1.1)
        self.normalize_attention_scores = normalize_attention_scores

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        projection_size = kv_channels * num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = safe_divide(projection_size, world_size)
        self.hidden_size_per_attention_head = safe_divide(projection_size, num_attention_heads)
        self.num_attention_heads_per_partition = safe_divide(num_attention_heads, world_size)
        self.num_attention_heads_partition_offset = (
            self.num_attention_heads_per_partition * parallel_state.get_tensor_model_parallel_rank()
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = MatchedScaleMaskSoftmax(
            self.fp16,
            self.bf16,
            self.attn_mask_type,
            masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout_p = attention_dropout
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        if use_flash_attention:
            self.attn_fn = self.flash_attention
        else:
            self.attn_fn = self.torch_attention

        if position_embedding_type.lower() == 'xpos':
            self.xpos = XPOSPositionEmbedding(kv_channels)

    def forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        rotary_pos_emb=None,
        relative_position_bias=None,
        headscale_tensor=None,
        inference_mode=None,
    ):
        b, np, sq, sk, hn = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
            query_layer.size(3),
        )

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================
        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[..., sq - 1, :sk].unsqueeze(2)
                else:
                    attention_mask = attention_mask[..., :sq, :sk]

        # ==================================================
        # Update attention bias. [b, np, sq, sk]
        # ==================================================
        if relative_position_bias is not None:
            relative_position_bias = relative_position_bias[
                :,
                self.num_attention_heads_partition_offset : self.num_attention_heads_partition_offset
                + self.num_attention_heads_per_partition,
                -sq:,
                -sk:,
            ]

        # ==================================================
        # Update query_layer, key_layer, value_layer
        # ==================================================
        # TODO: figure out how to do this
        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        if self.position_embedding_type.lower() == 'xpos':
            query_layer = self.xpos(query_layer, offset=key_layer.shape[-2] - query_layer.shape[-2], downscale=False)
            key_layer = self.xpos(key_layer, offset=0, downscale=True)

        # ==================================================
        # query_layer [sq, b, np, hn]
        # key_layer   [sk, b, np, hn]
        # value_layer [sk, b, np, hn]
        # attention_mask [b, 1, sq, sk] or [b, s]
        # relative_position_bias [b, np, sq, sk]
        # context_layer [b, np, sq, hn]
        # ==================================================
        context_layer = self.attn_fn(
            query_layer, key_layer, value_layer, attention_mask, relative_position_bias, inference_mode
        )

        if headscale_tensor is not None:
            context_layer = context_layer * headscale_tensor

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

    def torch_attention(self, query_layer, key_layer, value_layer, attention_mask, attention_bias, inference_mode):
        sq, b, np, hn = query_layer.shape
        sk = key_layer.shape[0]

        if self.multi_query_attention:
            query_layer = rearrange(query_layer, 'sq b np hn -> b (np sq) hn')
            key_layer = rearrange(key_layer, 'sk b 1 hn -> b hn sk')
            value_layer = rearrange(value_layer, 'sv b np hn -> (b np) sv hn')
        else:
            query_layer = rearrange(query_layer, 'sq b np hn -> (b np) sq hn')
            key_layer = rearrange(key_layer, 'sk b np hn -> (b np) hn sk')
            value_layer = rearrange(value_layer, 'sv b np hn -> (b np) sv hn')

        matmul_input_buffer = torch.empty(
            query_layer.shape[0],
            query_layer.shape[1],
            key_layer.shape[2],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,
            key_layer,
            beta=0.0,
            alpha=(1.0 / self.norm_factor) if self.normalize_attention_scores else 1.0,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(b, np, sq, sk)

        if attention_bias is not None:
            attention_scores += attention_bias

        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel:
            with tensor_parallel.random.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # change view [b * np, sq, sk]
        attention_probs = rearrange(attention_probs, 'b np sq sk -> (b np) sq sk')

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)

        # change view [b, np, sq, hn]
        context_layer = rearrange(context_layer, '(b np) sq hn -> b np sq hn', np=np)

        return context_layer

    def flash_attention(self, query_layer, key_layer, value_layer, attention_mask, attention_bias, inference_mode):
        query_layer = rearrange(query_layer, 'sq b np hn -> b sq np hn')
        key_layer = rearrange(key_layer, 'sk b np hn -> b sk np hn')
        value_layer = rearrange(value_layer, 'sv b np hn -> b sv np hn')

        # Use to ensure dtype cast to fp16 or bf16
        query_layer = _cast_if_autocast_enabled(query_layer)
        key_layer = _cast_if_autocast_enabled(key_layer)
        value_layer = _cast_if_autocast_enabled(value_layer)
        attention_bias = _cast_if_autocast_enabled(attention_bias)

        is_causal = self.attn_mask_type == AttnMaskType.causal and not inference_mode

        if attention_bias is not None:
            return self.flash_attention_triton(
                query_layer, key_layer, value_layer, attention_mask, attention_bias, is_causal,
            )
        else:
            return self.flash_attention_cuda(query_layer, key_layer, value_layer, attention_mask, is_causal,)

    def flash_attention_cuda(self, query_layer, key_layer, value_layer, attention_mask, is_causal):
        batch_size, seqlen, nheads, _ = query_layer.shape

        # True: attend / False: not attend
        if attention_mask is None:
            attention_mask_q = torch.ones(batch_size, query_layer.shape[1], device=query_layer.device).bool()
            attention_mask_kv = torch.ones(batch_size, key_layer.shape[1], device=query_layer.device).bool()
        elif len(attention_mask.shape) == 4:
            # [b, 1, sq, sk] -> [b, sq] / [b, sk]
            attention_mask_q = torch.any(torch.eq(attention_mask, False), dim=3).squeeze(1)
            attention_mask_kv = torch.any(torch.eq(attention_mask, False), dim=2).squeeze(1)
        else:
            assert len(attention_mask.shape) == 2
            attention_mask_q = ~attention_mask
            attention_mask_kv = ~attention_mask

        seqlens_q_in_batch = len(attention_mask_q.sum(dim=-1, dtype=torch.int32).unique())
        seqlens_kv_in_batch = len(attention_mask_kv.sum(dim=-1, dtype=torch.int32).unique())

        if seqlens_q_in_batch == 1 and seqlens_kv_in_batch == 1 and flash_attn_func is not None:
            # [b, sq, np, hn]
            context_layer = flash_attn_func(
                query_layer,
                key_layer,
                value_layer,
                dropout_p=self.attention_dropout_p if self.training else 0.0,
                causal=is_causal,
            )
        else:
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query_layer, attention_mask_q)
            k, _, cu_seqlens_k, max_seqlen_k = unpad_input(key_layer, attention_mask_kv)
            v, _, _, _ = unpad_input(value_layer, attention_mask_kv)

            context_layer = flash_attn_unpadded_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=self.attention_dropout_p if self.training else 0.0,
                causal=is_causal,
            )

            # [b, sq, np, hn]
            context_layer = pad_input(context_layer, indices_q, batch_size, seqlen)

        # [b, sq, np, hn] -> [b, np, sq, hn]
        context_layer = context_layer.permute(0, 2, 1, 3)
        return context_layer

    def flash_attention_triton(self, query_layer, key_layer, value_layer, attention_mask, attention_bias, is_causal):
        if self.attention_dropout_p > 0.0:
            raise NotImplementedError(f'attention_dropout not implemented for flash_attention with attention bias')

        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                # [b, 1, sq, sk] -> [b, 1, sq, 1] / [b, 1, 1, sk]
                attention_mask_q = torch.any(torch.eq(attention_mask, False), dim=3).unsqueeze(3)
                attention_mask_kv = torch.any(torch.eq(attention_mask, False), dim=2).unsqueeze(2)
            else:
                # [b, s] -> [b, 1, s, 1] / [b, 1, 1, s]
                assert len(attention_mask.shape) == 2
                attention_mask_q = (~attention_mask).unsqueeze(1).unsqueeze(3)
                attention_mask_kv = (~attention_mask).unsqueeze(1).unsqueeze(2)

            if attention_bias.shape[2] == attention_mask_q.shape[2]:
                attention_bias = attention_bias.masked_fill(~attention_mask_q, torch.finfo(query_layer.dtype).min)
            if attention_bias.shape[3] == attention_mask_kv.shape[3]:
                attention_bias = attention_bias.masked_fill(~attention_mask_kv, torch.finfo(query_layer.dtype).min)

        context_layer = flash_attn_func_triton(query_layer, key_layer, value_layer, attention_bias, is_causal,)

        # [b, sq, np, hn] -> [b, np, sq, hn]
        context_layer = context_layer.permute(0, 2, 1, 3)

        if attention_mask is not None:
            context_layer = context_layer * attention_mask_q

        return context_layer
