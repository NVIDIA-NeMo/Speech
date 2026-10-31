# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Set

import torch
import torch.nn as nn
from omegaconf import DictConfig

from nemo.collections.asr.modules.transformer.transformer_modules import MultiHeadAttention, PositionWiseFF
from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.common.parts import NEG_INF, form_attention_mask
from nemo.collections.common.parts.utils import ModuleBlock, build_module
from nemo.core.classes.mixins import adapter_mixins

__all__ = ["TransformerDecoder", "TransformerDecoderKeyValueCache", "LayerKeyValueCache"]


@dataclass
class LayerKeyValueCache:
    """The slice of a ``TransformerDecoderKeyValueCache`` one decoder block reads and writes in a step.

    The masks are additive, as ``form_attention_mask`` builds them.

    Attributes:
        self_keys: self-attention keys (B, H, capacity, D), a view the block writes at ``positions``
        self_values: self-attention values (B, H, capacity, D), written like ``self_keys``
        self_attention_mask: mask (B, 1, L_new, capacity) for the queries of the step
        positions: positions (L_new,) of the new tokens in the cache
        cross_keys: cross-attention keys (B, H, L_enc, D) of the encoder states, read only
        cross_values: cross-attention values (B, H, L_enc, D), read only
        cross_attention_mask: mask (B, 1, 1, L_enc) of the encoder states
    """

    self_keys: torch.Tensor
    self_values: torch.Tensor
    self_attention_mask: torch.Tensor
    positions: torch.Tensor
    cross_keys: torch.Tensor
    cross_values: torch.Tensor
    cross_attention_mask: torch.Tensor


class TransformerDecoderKeyValueCache:
    """Bounded key-value state of a ``TransformerDecoder`` for incremental decoding.

    ``TransformerDecoder.create_kv_cache`` builds the cache for a batch of encoder states.
    The constructor allocates buffers of fixed capacities,
    and ``load_encoder_states`` fills them for a batch,
    so buffers and captured CUDA graphs can be reused across batches.
    The cache holds the cross-attention keys and values of every layer, projected once per batch,
    and the self-attention keys and values of the decoded positions,
    which ``TransformerDecoder.forward(..., kv_cache=cache)`` writes for the new positions only.
    Buffer shapes never change while the sequence grows and ``length`` is a device tensor,
    so a decode step over the cache is free of host synchronization and can be captured in a CUDA graph.

    Batch entries are hypotheses:
    ``repeat_interleave`` makes several per source sequence,
    and ``reorder_hypotheses`` applies beam search back references in place.
    Positions whose decoder mask is 0 (padding after an ended hypothesis) are never attended to.

    The attention blocks of the decoder provide the head layout and the projections,
    as ``MultiHeadAttention`` does with ``num_attention_heads``, ``attn_head_size``,
    ``project_queries``, ``project_keys_values`` and ``attend``.

    Args:
        decoder: the ``TransformerDecoder`` the cache belongs to
        batch_size: number of hypotheses
        frame_capacity: number of encoder frames the cross-attention buffers hold
        capacity: number of decoder positions the self-attention buffers hold, prompt plus generated tokens
        device: device of the buffers
        dtype: dtype of the key and value buffers
    """

    def __init__(
        self,
        decoder: "TransformerDecoder",
        batch_size: int,
        frame_capacity: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        for layer in decoder.layers:
            for block in (layer.first_sub_layer, layer.second_sub_layer):
                for name in (
                    "num_attention_heads",
                    "attn_head_size",
                    "project_queries",
                    "project_keys_values",
                    "attend",
                ):
                    if not hasattr(block, name):
                        raise TypeError(f"{type(block).__name__} has no {name}, which the key-value cache needs")
            if getattr(layer.second_sub_layer, 'rotary_pos_enc', None) is not None:
                raise ValueError("The key-value cache supports a rotary position embedding in the self-attention only")
            rotary = getattr(layer.first_sub_layer, 'rotary_pos_enc', None)
            if rotary is not None:
                rotary.extend_pe(capacity, device, dtype)
        num_layers = len(decoder.layers)
        self_attention = decoder.layers[0].first_sub_layer
        cross_attention = decoder.layers[0].second_sub_layer
        self.self_keys = torch.zeros(
            num_layers,
            batch_size,
            self_attention.num_attention_heads,
            capacity,
            self_attention.attn_head_size,
            device=device,
            dtype=dtype,
        )
        self.self_values = torch.zeros_like(self.self_keys)
        self.self_mask = torch.zeros(batch_size, capacity, dtype=torch.bool, device=device)
        self.cross_keys = torch.zeros(
            num_layers,
            batch_size,
            cross_attention.num_attention_heads,
            frame_capacity,
            cross_attention.attn_head_size,
            device=device,
            dtype=dtype,
        )
        self.cross_values = torch.zeros_like(self.cross_keys)
        self.cross_attention_mask = torch.full((batch_size, 1, 1, frame_capacity), NEG_INF, device=device)
        self.length = torch.zeros((), dtype=torch.long, device=device)
        self.position_ids = torch.arange(capacity, device=device)

    @property
    def batch_size(self) -> int:
        return self.self_mask.shape[0]

    @property
    def capacity(self) -> int:
        return self.self_mask.shape[1]

    @property
    def frame_capacity(self) -> int:
        return self.cross_keys.shape[3]

    def load_encoder_states(
        self,
        decoder: "TransformerDecoder",
        encoder_states: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        repeats: int = 1,
    ) -> None:
        """Project a batch of encoder states into the cross-attention buffers and start a new decode.

        Rows ``[0, batch * repeats)`` receive the projections,
        each source sequence repeated ``repeats`` times consecutively (one row per beam hypothesis);
        the frames beyond a sequence, the remaining rows and the remaining frames are masked.
        The self-attention state is reset.

        Args:
            decoder: the ``TransformerDecoder`` the cache was created for
            encoder_states: encoder output (B, L_enc, H), ``B * repeats`` at most ``batch_size``
                and ``L_enc`` at most ``frame_capacity``
            encoder_mask: encoder inputs mask (B, L_enc) with 1 for valid frames, or None for all valid
            repeats: hypotheses per source sequence
        """
        batch, frames, _ = encoder_states.shape
        rows = batch * repeats
        if rows > self.batch_size or frames > self.frame_capacity:
            raise ValueError(
                f"{batch} x {repeats} sequences with {frames} frames exceed the cache of {self.batch_size} rows"
                f" and {self.frame_capacity} frames"
            )
        if encoder_mask is None:
            encoder_mask = torch.ones(batch, frames, device=encoder_states.device)
        mask = torch.zeros(self.batch_size, self.frame_capacity, device=encoder_states.device)
        mask[:rows, :frames] = encoder_mask.to(mask.dtype).repeat_interleave(repeats, dim=0)
        self.cross_attention_mask.copy_(form_attention_mask(mask))
        for index, layer in enumerate(decoder.layers):
            key, value = layer.second_sub_layer.project_keys_values(encoder_states, encoder_states)
            if repeats > 1:
                key, value = key.repeat_interleave(repeats, dim=0), value.repeat_interleave(repeats, dim=0)
            self.cross_keys[index, :rows, :, :frames].copy_(key)
            self.cross_values[index, :rows, :, :frames].copy_(value)
        self.reset()

    def reset(self) -> None:
        """Forget the decoded positions; the cross-attention projections stay."""
        self.self_mask.zero_()
        self.length.zero_()

    def _with_batch(self, select: Callable[[torch.Tensor, int], torch.Tensor]) -> "TransformerDecoderKeyValueCache":
        """A new cache whose batch entries are ``select(tensor, batch_dim)`` of this one."""
        clone = copy.copy(self)
        clone.self_keys = select(self.self_keys, 1)
        clone.self_values = select(self.self_values, 1)
        clone.self_mask = select(self.self_mask, 0)
        clone.cross_keys = select(self.cross_keys, 1)
        clone.cross_values = select(self.cross_values, 1)
        clone.cross_attention_mask = select(self.cross_attention_mask, 0)
        clone.length = self.length.clone()
        return clone

    def repeat_interleave(self, repeats: int) -> "TransformerDecoderKeyValueCache":
        """A new cache for beam search, with every hypothesis repeated consecutively.

        Args:
            repeats: hypotheses per source sequence

        Returns:
            the new cache, with ``repeats`` times the rows
        """
        return self._with_batch(lambda tensor, dim: tensor.repeat_interleave(repeats, dim=dim))

    def index_select(self, indices: torch.Tensor) -> "TransformerDecoderKeyValueCache":
        """A new cache with selected hypotheses.

        Args:
            indices: hypothesis rows (N,) to keep, in the order of the new cache

        Returns:
            the new cache with ``N`` rows
        """
        return self._with_batch(lambda tensor, dim: tensor.index_select(dim, indices))

    def reorder_hypotheses(self, indices: torch.Tensor) -> None:
        """Replace hypothesis ``i`` by hypothesis ``indices[i]`` in place, for beam search back references.

        Only the self-attention state moves:
        every index must select a hypothesis of the same source sequence,
        whose cross-attention projections are therefore already in place.

        Args:
            indices: source row (batch_size,) of every hypothesis
        """
        self.self_keys.copy_(self.self_keys.index_select(1, indices))
        self.self_values.copy_(self.self_values.index_select(1, indices))
        self.self_mask.copy_(self.self_mask.index_select(0, indices))

    def self_attention_mask(self, positions: torch.Tensor) -> torch.Tensor:
        """Additive self-attention mask for the queries of a step.

        A key is attended to when its position was written with a set decoder mask
        and is not after the query.

        Args:
            positions: positions (L_new,) of the queries

        Returns:
            mask (B, 1, L_new, capacity), 0 where attended and ``NEG_INF`` otherwise
        """
        allowed = self.self_mask[:, None, None, :] & (
            self.position_ids[None, None, None, :] <= positions[None, None, :, None]
        )
        return (1 - allowed.to(torch.float)) * NEG_INF

    def layer(self, index: int, positions: torch.Tensor, self_attention_mask: torch.Tensor) -> LayerKeyValueCache:
        """The slice of one layer for a step.

        Args:
            index: layer index
            positions: positions (L_new,) of the new tokens
            self_attention_mask: the mask from ``self_attention_mask(positions)``

        Returns:
            the views and masks the block of that layer reads and writes
        """
        return LayerKeyValueCache(
            self_keys=self.self_keys[index],
            self_values=self.self_values[index],
            self_attention_mask=self_attention_mask,
            positions=positions,
            cross_keys=self.cross_keys[index],
            cross_values=self.cross_values[index],
            cross_attention_mask=self.cross_attention_mask,
        )


class TransformerDecoderBlock(nn.Module, AttentionAdapterModuleMixin):
    """
    Building block of Transformer decoder.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            attention layers, but before layer normalization
        ffn_dropout: probability of dropout applied to FFN output
        hidden_act: activation function used between two linear layers in FFN
        pre_ln: whether to apply the layer norms before the sub-layers instead of after them
        norm: block of the three layer norms, default `torch.nn.LayerNorm`;
            a block is a class, a callable or a Hydra partial (``_target_`` with ``_partial_: true``),
            built from the arguments the parent supplies, so a block section only names what deviates
        self_attn: block of the self-attention, default `MultiHeadAttention`
        cross_attn: block of the attention over the encoder states, default `MultiHeadAttention`
        feed_forward: block of the feed-forward sub-layer, default `PositionWiseFF`

    ``forward`` takes an optional ``kv_cache`` (a ``LayerKeyValueCache``) for incremental decoding:
    the query states are the new positions only,
    their self-attention keys and values are written to the cache,
    and the attention runs over the cached keys and values,
    with ``decoder_keys``, ``encoder_states`` and ``encoder_mask`` unused.
    """

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
        norm: ModuleBlock | None = None,
        self_attn: ModuleBlock | None = None,
        cross_attn: ModuleBlock | None = None,
        feed_forward: ModuleBlock | None = None,
    ):
        super().__init__()
        self.pre_ln = pre_ln
        norm = norm or nn.LayerNorm
        attention_defaults = dict(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
        )
        self.layer_norm_1 = build_module(norm, normalized_shape=hidden_size)
        self.first_sub_layer = build_module(self_attn or MultiHeadAttention, **attention_defaults)
        self.layer_norm_2 = build_module(norm, normalized_shape=hidden_size)
        self.second_sub_layer = build_module(
            cross_attn or MultiHeadAttention, return_xatt_scores=True, **attention_defaults
        )
        self.layer_norm_3 = build_module(norm, normalized_shape=hidden_size)
        self.third_sub_layer = build_module(
            feed_forward or PositionWiseFF,
            hidden_size=hidden_size,
            inner_size=inner_size,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
        )

        # Information for the adapter module mixin
        self.self_attention_model = "transf_abs"

    def _self_attention(
        self,
        decoder_query: torch.Tensor,
        decoder_mask: torch.Tensor | None,
        decoder_keys: torch.Tensor | None,
        kv_cache: LayerKeyValueCache | None,
    ) -> torch.Tensor:
        """Self-attention over the key sequence, or over the layer's key-value cache for the new positions."""
        if kv_cache is None:
            output_states, _ = self.first_sub_layer(decoder_query, decoder_keys, decoder_keys, decoder_mask)
            return output_states
        attention = self.first_sub_layer
        query = attention.project_queries(decoder_query, kv_cache.positions)
        key, value = attention.project_keys_values(decoder_query, decoder_query, kv_cache.positions)
        kv_cache.self_keys.index_copy_(2, kv_cache.positions, key)
        kv_cache.self_values.index_copy_(2, kv_cache.positions, value)
        output_states, _ = attention.attend(
            query, kv_cache.self_keys, kv_cache.self_values, kv_cache.self_attention_mask
        )
        return output_states

    def _cross_attention(
        self,
        decoder_states: torch.Tensor,
        encoder_states: torch.Tensor | None,
        encoder_mask: torch.Tensor | None,
        kv_cache: LayerKeyValueCache | None,
    ) -> tuple[torch.Tensor, dict]:
        """Attention over the encoder states, or over their cached projections."""
        if kv_cache is None:
            return self.second_sub_layer(decoder_states, encoder_states, encoder_states, encoder_mask)
        attention = self.second_sub_layer
        output_states, attention_probs = attention.attend(
            attention.project_queries(decoder_states),
            kv_cache.cross_keys,
            kv_cache.cross_values,
            kv_cache.cross_attention_mask,
        )
        extra_output = {'xatt_scores': attention_probs} if attention.return_xatt_scores else {}
        return output_states, extra_output

    def forward_preln(
        self,
        decoder_query,
        decoder_mask,
        decoder_keys,
        encoder_states,
        encoder_mask,
        kv_cache: LayerKeyValueCache | None = None,
    ):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN
        """
        residual = decoder_query
        decoder_query = self.layer_norm_1(decoder_query)
        if kv_cache is None:
            decoder_keys = self.layer_norm_1(decoder_keys)
        self_attn_output = self._self_attention(decoder_query, decoder_mask, decoder_keys, kv_cache)
        self_attn_output += residual

        if self.is_adapter_available():
            # Call the MHA adapters
            pack_input = {
                'x': self_attn_output,
                'loc': 'mha',
                'att_mask': decoder_mask,
                'pos_emb': None,
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            self_attn_output = pack_input['x']

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        enc_dec_attn_output, extra_output = self._cross_attention(
            self_attn_output, encoder_states, encoder_mask, kv_cache
        )
        enc_dec_attn_output += residual

        residual = enc_dec_attn_output
        enc_dec_attn_output = self.layer_norm_3(enc_dec_attn_output)
        output_states = self.third_sub_layer(enc_dec_attn_output)
        output_states += residual

        if self.is_adapter_available():
            # Call the Linear adapters
            pack_input = {
                'x': output_states,
                'loc': 'post',
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            output_states = pack_input['x']

        return output_states, extra_output

    def forward_postln(
        self,
        decoder_query,
        decoder_mask,
        decoder_keys,
        encoder_states,
        encoder_mask,
        kv_cache: LayerKeyValueCache | None = None,
    ):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        self_attn_output = self._self_attention(decoder_query, decoder_mask, decoder_keys, kv_cache)
        self_attn_output += decoder_query

        if self.is_adapter_available():
            # Call the MHA adapters
            pack_ip = {
                'x': self_attn_output,
                'loc': 'mha',
                'att_mask': decoder_mask,
                'pos_emb': None,
            }
            pack_ip = self.forward_enabled_adapters(pack_ip)
            self_attn_output = pack_ip['x']

        self_attn_output = self.layer_norm_1(self_attn_output)

        enc_dec_attn_output, extra_output = self._cross_attention(
            self_attn_output, encoder_states, encoder_mask, kv_cache
        )
        enc_dec_attn_output += self_attn_output
        enc_dec_attn_output = self.layer_norm_2(enc_dec_attn_output)

        output_states = self.third_sub_layer(enc_dec_attn_output)
        output_states += enc_dec_attn_output

        if self.is_adapter_available():
            # Call the linear adapters
            pack_ip = {
                'x': output_states,
                'loc': 'post',
            }
            pack_ip = self.forward_enabled_adapters(pack_ip)
            output_states = pack_ip['x']

        return self.layer_norm_3(output_states), extra_output

    def forward(
        self,
        decoder_query,
        decoder_mask,
        decoder_keys,
        encoder_states,
        encoder_mask,
        kv_cache: LayerKeyValueCache | None = None,
    ):
        if self.pre_ln:
            return self.forward_preln(
                decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask, kv_cache=kv_cache
            )
        else:
            return self.forward_postln(
                decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask, kv_cache=kv_cache
            )

    def get_accepted_adapter_types(self) -> Set[type]:
        types = super().get_accepted_adapter_types()

        if len(types) == 0:
            self.set_accepted_adapter_types(
                [
                    adapter_utils.LINEAR_ADAPTER_CLASSPATH,
                    adapter_utils.TRANSFORMER_MHA_ADAPTER_CLASSPATH,
                ]
            )
            types = self.get_accepted_adapter_types()
        return types


class TransformerDecoder(nn.Module):
    """
    Stack of Transformer decoder blocks with an optional final layer norm.

    Args:
        num_layers: number of decoder blocks
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of the feed-forward net
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the attention layers,
            but before layer normalization
        ffn_dropout: probability of dropout applied to the feed-forward output
        hidden_act: activation function used between the two linear layers of the feed-forward net
        pre_ln: whether to apply the layer norms before the sub-layers instead of after them
        pre_ln_final_layer_norm: with `pre_ln`, whether a final layer norm follows the last block
        norm: block of every layer norm, including the final one, see `TransformerDecoderBlock`
        self_attn: block of the self-attention of every block, see `TransformerDecoderBlock`
        cross_attn: block of the attention over the encoder states of every block, see `TransformerDecoderBlock`
        feed_forward: block of the feed-forward sub-layer of every block, see `TransformerDecoderBlock`
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        norm: ModuleBlock | None = None,
        self_attn: ModuleBlock | None = None,
        cross_attn: ModuleBlock | None = None,
        feed_forward: ModuleBlock | None = None,
    ):
        super().__init__()

        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = build_module(norm or nn.LayerNorm, normalized_shape=hidden_size)
        else:
            self.final_layer_norm = None

        self.d_model = hidden_size

        layer = TransformerDecoderBlock(
            hidden_size,
            inner_size,
            num_attention_heads,
            attn_score_dropout,
            attn_layer_dropout,
            ffn_dropout,
            hidden_act,
            pre_ln,
            norm=norm,
            self_attn=self_attn,
            cross_attn=cross_attn,
            feed_forward=feed_forward,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.diagonal = 0

    def _get_memory_states(self, decoder_states, decoder_mems_list=None, i=0):
        if decoder_mems_list is not None:
            inp1 = torch.transpose(decoder_mems_list[i], 1, 2)  # Putting seq_len to last dim to handle export cases
            inp2 = torch.transpose(decoder_states, 1, 2)
            memory_states = torch.cat((inp1, inp2), dim=2)
            memory_states = torch.transpose(memory_states, 1, 2)  # Transposing back
        else:
            memory_states = decoder_states
        return memory_states

    def create_kv_cache(
        self, encoder_states: torch.Tensor, encoder_mask: torch.Tensor | None, capacity: int
    ) -> TransformerDecoderKeyValueCache:
        """Key-value cache for incremental decoding over ``encoder_states``.

        Args:
            encoder_states: output of the encoder (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc), or None for all valid
            capacity: number of decoder positions the cache holds, prompt plus generated tokens

        Returns:
            a ``TransformerDecoderKeyValueCache`` with the encoder projections loaded
        """
        batch_size, frames, _ = encoder_states.shape
        cache = TransformerDecoderKeyValueCache(
            self, batch_size, frames, capacity, encoder_states.device, encoder_states.dtype
        )
        cache.load_encoder_states(self, encoder_states, encoder_mask)
        return cache

    def forward(
        self,
        decoder_states,
        decoder_mask,
        encoder_states=None,
        encoder_mask=None,
        decoder_mems_list=None,
        return_mems=False,
        return_mems_as_list=True,
        kv_cache: TransformerDecoderKeyValueCache | None = None,
    ):
        """
        Args:
            decoder_states: output of the embedding layer (B x L_dec x H)
            decoder_mask: decoder inputs mask (B x L_dec)
            encoder_states: output of the encoder (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
            decoder_mems_list: list of the cached decoder hidden states
                for fast autoregressive generation which will be used instead
                of decoder_states as keys and values if not None
            return_mems: bool, whether to return outputs of all decoder layers
                or the last layer only
            return_mems_as_list: bool, when True, mems returned are as a list; otherwise mems are Tensor
            kv_cache: a TransformerDecoderKeyValueCache from ``create_kv_cache`` for fast autoregressive generation;
                decoder_states and decoder_mask then cover the new positions only,
                their keys and values are written at the cache ``length``, which advances,
                encoder_states, encoder_mask and decoder_mems_list must be None,
                and with return_mems the returned states are those of the new positions
        """
        if kv_cache is not None:
            if encoder_states is not None or encoder_mask is not None or decoder_mems_list is not None:
                raise ValueError(
                    "kv_cache carries the encoder projections and the decoder memory; pass encoder_states,"
                    " encoder_mask and decoder_mems_list as None"
                )
            return self._forward_with_kv_cache(
                decoder_states, decoder_mask, kv_cache, return_mems, return_mems_as_list
            )
        decoder_attn_mask = form_attention_mask(decoder_mask, diagonal=self.diagonal)
        encoder_attn_mask = form_attention_mask(encoder_mask)
        memory_states = self._get_memory_states(decoder_states, decoder_mems_list, 0)
        if return_mems:
            if return_mems_as_list:
                cached_mems_list = [memory_states]
            else:
                cached_mems_list = memory_states.unsqueeze(0)

        xatt_scores_list = []

        for i, layer in enumerate(self.layers):
            decoder_states, extra_output = layer(
                decoder_states, decoder_attn_mask, memory_states, encoder_states, encoder_attn_mask
            )
            memory_states = self._get_memory_states(decoder_states, decoder_mems_list, i + 1)
            xatt_scores_list.append(extra_output['xatt_scores'])
            if return_mems:
                if return_mems_as_list:
                    cached_mems_list.append(memory_states)
                else:
                    cached_mems_list = torch.cat((cached_mems_list, memory_states.unsqueeze(0)), dim=0)

        if self.final_layer_norm is not None:
            decoder_states = self.final_layer_norm(decoder_states)
            memory_states = self._get_memory_states(decoder_states, decoder_mems_list, i + 2)
            if return_mems:
                if return_mems_as_list:
                    cached_mems_list.append(memory_states)
                else:
                    cached_mems_list = torch.cat((cached_mems_list, memory_states.unsqueeze(0)), dim=0)

        if return_mems:
            return cached_mems_list, xatt_scores_list
        else:
            return memory_states, xatt_scores_list

    def _forward_with_kv_cache(
        self,
        decoder_states: torch.Tensor,
        decoder_mask: torch.Tensor | None,
        kv_cache: TransformerDecoderKeyValueCache,
        return_mems: bool,
        return_mems_as_list: bool,
    ):
        num_new = decoder_states.shape[1]
        positions = kv_cache.length + torch.arange(num_new, device=decoder_states.device)
        if decoder_mask is None:
            decoder_mask = torch.ones(decoder_states.shape[:2], dtype=torch.bool, device=decoder_states.device)
        kv_cache.self_mask.index_copy_(1, positions, decoder_mask.to(torch.bool))
        self_attention_mask = kv_cache.self_attention_mask(positions)

        cached_mems_list = [decoder_states]
        xatt_scores_list = []
        for index, layer in enumerate(self.layers):
            decoder_states, extra_output = layer(
                decoder_states, None, None, None, None, kv_cache=kv_cache.layer(index, positions, self_attention_mask)
            )
            xatt_scores_list.append(extra_output['xatt_scores'])
            cached_mems_list.append(decoder_states)

        if self.final_layer_norm is not None:
            decoder_states = self.final_layer_norm(decoder_states)
            cached_mems_list.append(decoder_states)
        kv_cache.length.add_(num_new)

        if not return_mems:
            return decoder_states, xatt_scores_list
        if return_mems_as_list:
            return cached_mems_list, xatt_scores_list
        return torch.stack(cached_mems_list), xatt_scores_list

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        input_ids = torch.randint(low=0, high=2048, size=(max_batch, max_dim, 1024), device=sample.device)
        encoder_mask = torch.randint(low=0, high=1, size=(max_batch, max_dim), device=sample.device)
        return tuple([input_ids, encoder_mask, input_ids, encoder_mask])


class TransformerDecoderAdapter(TransformerDecoder, adapter_mixins.AdapterModuleMixin):

    # Higher level forwarding
    def add_adapter(self, name: str, cfg: dict):
        cfg = self._update_adapter_cfg_input_dim(cfg)
        for transformer_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            transformer_layer.add_adapter(name, cfg)

    def is_adapter_available(self) -> bool:
        return any([transformer_layer.is_adapter_available() for transformer_layer in self.layers])

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        for transformer_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            transformer_layer.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        names = set([])
        for transformer_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            names.update(transformer_layer.get_enabled_adapters())

        names = sorted(list(names))
        return names

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self.d_model)
        return cfg


"""
Register any additional information
"""
if adapter_mixins.get_registered_adapter(TransformerDecoder) is None:
    adapter_mixins.register_adapter(base_class=TransformerDecoder, adapter_class=TransformerDecoderAdapter)
