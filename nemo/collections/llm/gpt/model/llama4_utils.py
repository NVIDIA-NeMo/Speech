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
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from megatron.core import parallel_state
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttention as MCoreSelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.utils import deprecate_inference_params, is_fa_min_version
from torch import Tensor

try:
    from flashattn_hopper.flash_attn_interface import _flash_attn_forward

    HAVE_FA3 = True
except ImportError:
    _flash_attn_forward = None

    HAVE_FA3 = False


def chunkify_cu_seqlens(cu_seqlens, cu_seqlens_padded, attention_chunk_size):
    """
    Splits cumulative sequence lengths into chunks based on attention_chunk_size.

    Args:
        cu_seqlens (list[int]): List of cumulative sequence lengths.
        cu_seqlens_padded (list[int]): List of padded cumulative sequence lengths.
        attention_chunk_size (int): The maximum size of each chunk.

    Returns:
        Tuple[list[int], list[int]]: A tuple containing the new chunked cumulative
        sequence lengths and the new chunked padded cumulative sequence lengths.
    """
    new_cu_seqlens = [cu_seqlens[0]]
    new_cu_seqlens_padded = [cu_seqlens_padded[0]]
    for i in range(1, len(cu_seqlens)):
        start = cu_seqlens[i - 1]
        end = cu_seqlens[i]
        start_padded = cu_seqlens_padded[i - 1]
        end_padded = cu_seqlens_padded[i]

        segment_length = end - start
        num_full_chunks = segment_length // attention_chunk_size

        for j in range(1, num_full_chunks + 1):
            new_index = start + j * attention_chunk_size
            new_cu_seqlens.append(new_index)
            new_index_padded = start_padded + j * attention_chunk_size
            new_cu_seqlens_padded.append(new_index_padded)

        if new_cu_seqlens[-1] != end:
            new_cu_seqlens.append(end)
            new_cu_seqlens_padded.append(end_padded)

    return new_cu_seqlens, new_cu_seqlens_padded


def chunkify(x, attention_chunk_size):
    """
    Pads and reshapes a tensor for chunked processing.

    This function takes an input tensor `x` (typically representing query, key, or value
    in attention mechanisms) and pads its sequence dimension (dim 0) to be a multiple
    of `attention_chunk_size`. It then reshapes the tensor so that the sequence dimension
    is split into chunks, and the chunk dimension is combined with the batch dimension.

    Args:
        x (torch.Tensor): Input tensor, expected shape [seq_length, batch_size, ...].
        attention_chunk_size (int): The desired size of chunks along the sequence dimension.

    Returns:
        torch.Tensor: The reshaped tensor with shape
                      [attention_chunk_size, num_chunks * batch_size, ...].
    """
    # Determine original sequence length.
    seq_length = x.shape[0]
    # Compute new sequence length (pad_seq_len) as the smallest multiple of attention_chunk_size
    pad_seq_len = ((seq_length + attention_chunk_size - 1) // attention_chunk_size) * attention_chunk_size

    # If padding is needed, create a pad tensor with the same type and device as x.
    if pad_seq_len != seq_length:
        pad_size = pad_seq_len - seq_length
        pad_tensor = torch.zeros(pad_size, *x.shape[1:], device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad_tensor], dim=0)

    # Compute the number of chunks (each of length attention_chunk_size)
    num_chunks = pad_seq_len // attention_chunk_size

    # Reshape from:
    #   [seq_length, batch_size, num_heads, head_dim]
    # to:
    #   [num_chunks, attention_chunk_size, batch_size, num_heads, head_dim]
    x = x.reshape(num_chunks, attention_chunk_size, *x.shape[1:])

    # Transpose the first two dimensions so that the tensor becomes:
    #   [attention_chunk_size, num_chunks, batch_size, num_heads, head_dim]
    x = x.transpose(0, 1)

    # Combine (collapse) the num_chunks and batch_size dimensions into one.
    x = x.reshape(x.shape[0], -1, *x.shape[3:]).contiguous()

    return x


def get_llama4_layer_spec(config, vp_stage: Optional[int] = None, gpt_decoder_block_spec=None) -> ModuleSpec:
    """Get llama4 layer spec"""

    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    # Use decoder_block_spec: set layer_specs as a list of individual layer specs
    if gpt_decoder_block_spec is None:
        llama4_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)
    else:
        llama4_layer_spec = gpt_decoder_block_spec
    updated_layer_specs = []
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    for idx, layer_spec in enumerate(llama4_layer_spec.layer_specs):
        layer_no = idx + offset
        updated_layer_spec = deepcopy(layer_spec)

        is_nope_layer = config.nope_layer_interval is not None and (layer_no + 1) % config.nope_layer_interval == 0
        updated_layer_spec.submodules.self_attention.module = Llama4SelfAttention
        updated_layer_spec.submodules.self_attention.params = {
            'is_nope_layer': is_nope_layer,
            'attention_chunk_size': config.attention_chunk_size,
            "attn_mask_type": AttnMaskType.causal,
        }
        if config.qk_l2_norm and not is_nope_layer:
            # Use QK Norm
            updated_layer_spec.submodules.self_attention.submodules.q_layernorm = L2Norm
            updated_layer_spec.submodules.self_attention.submodules.k_layernorm = L2Norm
        else:
            updated_layer_spec.submodules.self_attention.submodules.q_layernorm = None
            updated_layer_spec.submodules.self_attention.submodules.k_layernorm = None
        updated_layer_specs.append(updated_layer_spec)

    llama4_layer_spec.layer_specs = updated_layer_specs
    return llama4_layer_spec


class Llama4SelfAttention(MCoreSelfAttention):
    """Updated Transformer Layer to enable skip rope in some layers"""

    def __init__(self, is_nope_layer=False, attention_chunk_size=8192, *args, **kwargs):
        self.is_nope_layer = is_nope_layer
        self.attention_chunk_size = attention_chunk_size
        super(Llama4SelfAttention, self).__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert (HAVE_FA3 and _flash_attn_forward is not None) or is_fa_min_version(
                "2.7.3"
            ), "flash attn verion v2.7.3 and above is required for dynamic batching."

        # hidden_states: [sq, b, h]
        if self.config.flash_decode and not self.training and inference_context is not None:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

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

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        if (
            self.config.flash_decode
            and inference_context is not None
            and inference_context.is_decode_only()
            and not self.training
            and rotary_pos_cos is not None
        ):
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[self.layer_number]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        query, key, value, rotary_pos_emb, attn_mask_type, block_table = self._adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )

        original_shape = None
        original_packed_seq_params = None
        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

            original_seq_len = max(packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv)

            if original_seq_len > self.attention_chunk_size:
                original_packed_seq_params = deepcopy(packed_seq_params)
                packed_seq_params.max_seqlen_q = packed_seq_params.max_seqlen_kv = self.attention_chunk_size
                # limit the each sub seq length to be capped at self.attention_chunk_size
                # assume attention_chunk_size = 10
                # original cu_seqlens_q = [0, 15, 20, 45]
                # new cu_seqlens_q = [0, 10, 15, 20, 30, 40, 45]
                packed_seq_params.cu_seqlens_q, packed_seq_params.cu_seqlens_q_padded = chunkify_cu_seqlens(
                    packed_seq_params.cu_seqlens_q, packed_seq_params.cu_seqlens_q_padded, self.attention_chunk_size
                )
                packed_seq_params.cu_seqlens_kv, packed_seq_params.cu_seqlens_kv_padded = chunkify_cu_seqlens(
                    packed_seq_params.cu_seqlens_kv, packed_seq_params.cu_seqlens_kv_padded, self.attention_chunk_size
                )
        else:
            original_seq_len = query.shape[0]
            if original_seq_len > self.attention_chunk_size:
                # [seq_len, batch_size, hidden_size]
                original_shape = hidden_states.shape
                query = chunkify(query, self.attention_chunk_size)
                key = chunkify(key, self.attention_chunk_size)
                value = chunkify(value, self.attention_chunk_size)
                rotary_pos_emb = rotary_pos_emb[: self.attention_chunk_size] if rotary_pos_emb is not None else None

        if parallel_state.get_context_parallel_world_size() > 1 and original_seq_len > self.attention_chunk_size:
            assert original_seq_len % (parallel_state.get_context_parallel_world_size() * 2) == 0
            cp_chunk_len = original_seq_len // (parallel_state.get_context_parallel_world_size() * 2)
            assert cp_chunk_len % self.attention_chunk_size == 0

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if not self.is_nope_layer and rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                # TODO VIJAY: simplify
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb(
                        query,
                        q_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        cp_group=self.model_comm_pgs.cp,
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(
                        query, q_pos_emb, self.config, cu_seqlens_q, self.model_comm_pgs.cp
                    )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    cp_group=self.model_comm_pgs.cp,
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                # Static batching attention kernel.
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )

            else:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, kv_lengths_decode_only, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    kv_lengths_decode_only,
                    block_table,
                )
                core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

            if original_seq_len > self.attention_chunk_size:
                packed_seq_params.max_seqlen_q = original_packed_seq_params.max_seqlen_q
                packed_seq_params.max_seqlen_kv = original_packed_seq_params.max_seqlen_kv
                packed_seq_params.cu_seqlens_q = original_packed_seq_params.cu_seqlens_q
                packed_seq_params.cu_seqlens_kv = original_packed_seq_params.cu_seqlens_kv
                packed_seq_params.cu_seqlens_q_padded = original_packed_seq_params.cu_seqlens_q_padded
                packed_seq_params.cu_seqlens_kv_padded = original_packed_seq_params.cu_seqlens_kv_padded
        else:
            if original_seq_len > self.attention_chunk_size:
                # Reshape from [attention_chunk_size, num_chunks * batch_size, hidden_size]
                # back to [seq_len, batch_size, hidden_size]
                batch_size = original_shape[1]
                num_chunks = core_attn_out.shape[1] // batch_size
                core_attn_out = core_attn_out.reshape(self.attention_chunk_size, num_chunks, batch_size, -1)
                # [num_chunks, attention_chunk_size, batch_size, hidden_size]
                core_attn_out = core_attn_out.transpose(0, 1)
                core_attn_out = core_attn_out.reshape(num_chunks * self.attention_chunk_size, batch_size, -1)
                core_attn_out = core_attn_out[:original_seq_len]

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)

        return output, bias
