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
from typing import List, Optional, Set

import torch
import torch.nn as nn
from omegaconf import DictConfig

from nemo.collections.asr.modules.transformer.transformer_modules import MultiHeadAttention, PositionWiseFF
from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.submodules.packed_attention import PackedAttentionContext
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.common.parts import form_attention_mask
from nemo.collections.common.parts.utils import ModuleBlock, build_module
from nemo.core.classes.mixins import adapter_mixins

__all__ = ["TransformerDecoder"]


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
        norm, self_attn, cross_attn, feed_forward: configurable blocks, each a class, a callable or a Hydra
            partial (``_target_`` with ``_partial_: true``) built from the arguments the block supplies
            (sizes, dropouts, hidden_act); a block section only names what deviates. Defaults to
            LayerNorm, MultiHeadAttention (both attentions) and PositionWiseFF.
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

    def forward_preln(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN
        """
        residual = decoder_query
        decoder_query = self.layer_norm_1(decoder_query)
        decoder_keys = self.layer_norm_1(decoder_keys)
        self_attn_output, _ = self.first_sub_layer(decoder_query, decoder_keys, decoder_keys, decoder_mask)
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
        enc_dec_attn_output, extra_output = self.second_sub_layer(
            self_attn_output, encoder_states, encoder_states, encoder_mask
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

    def forward_postln(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        self_attn_output, _ = self.first_sub_layer(decoder_query, decoder_keys, decoder_keys, decoder_mask)
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

        enc_dec_attn_output, extra_output = self.second_sub_layer(
            self_attn_output, encoder_states, encoder_states, encoder_mask
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

    def forward(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask):
        if self.pre_ln:
            return self.forward_preln(decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask)
        else:
            return self.forward_postln(decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask)

    def forward_packed(
        self,
        decoder_states: torch.Tensor,
        decoder_context: PackedAttentionContext,
        encoder_states: torch.Tensor,
        encoder_context: PackedAttentionContext,
    ) -> torch.Tensor:
        """The block over token-flat decoder states attending token-flat encoder states.

        See ``TransformerDecoder.forward_packed`` for the contract.

        Args:
            decoder_states: shape ``(U_total, hidden_size)``.
            decoder_context: boundaries and positions of the decoder tokens.
            encoder_states: shape ``(T_total, hidden_size)``, in the same sequence order.
            encoder_context: boundaries of the encoder tokens.

        Returns:
            The output states, shape ``(U_total, hidden_size)``.
        """
        if self.is_adapter_available():
            raise NotImplementedError("Adapters are not supported in the sequence-packed decoder block.")
        if self.pre_ln:
            residual = decoder_states
            normed = self.layer_norm_1(decoder_states)
            attended, _ = self.first_sub_layer.forward_packed(
                normed, normed, normed, decoder_context, decoder_context, causal=True
            )
            decoder_states = attended + residual
            residual = decoder_states
            normed = self.layer_norm_2(decoder_states)
            attended, _ = self.second_sub_layer.forward_packed(
                normed, encoder_states, encoder_states, decoder_context, encoder_context
            )
            decoder_states = attended + residual
            return self.third_sub_layer(self.layer_norm_3(decoder_states)) + decoder_states

        attended, _ = self.first_sub_layer.forward_packed(
            decoder_states, decoder_states, decoder_states, decoder_context, decoder_context, causal=True
        )
        decoder_states = self.layer_norm_1(attended + decoder_states)
        attended, _ = self.second_sub_layer.forward_packed(
            decoder_states, encoder_states, encoder_states, decoder_context, encoder_context
        )
        decoder_states = self.layer_norm_2(attended + decoder_states)
        return self.layer_norm_3(self.third_sub_layer(decoder_states) + decoder_states)

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
        """
        Args:
            norm, self_attn, cross_attn, feed_forward: configurable blocks of every TransformerDecoderBlock
                and of the final norm, see TransformerDecoderBlock
        """
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

    def forward(
        self,
        decoder_states,
        decoder_mask,
        encoder_states,
        encoder_mask,
        decoder_mems_list=None,
        return_mems=False,
        return_mems_as_list=True,
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
        """
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

    def forward_packed(
        self,
        decoder_states: torch.Tensor,
        decoder_context: PackedAttentionContext,
        encoder_states: torch.Tensor,
        encoder_context: PackedAttentionContext,
    ) -> torch.Tensor:
        """Teacher-forced decoding over sequence-packed inputs, the training-time counterpart of ``forward``.

        The decoder and encoder masks of ``forward`` are replaced by the two packings:
        no padded position is computed,
        self-attention is causal within every sequence (the mask of diagonal 0)
        and the cross-attention of a sequence sees only its own encoder states.
        The cached memory lists of incremental decoding, the cross-attention scores and adapters
        are not provided in this form.

        Args:
            decoder_states: embedded decoder inputs of all sequences, shape ``(U_total, hidden_size)``.
            decoder_context: boundaries and positions of the decoder tokens.
            encoder_states: encoder outputs of all sequences, shape ``(T_total, hidden_size)``, in the same order.
            encoder_context: boundaries of the encoder tokens.

        Returns:
            The decoder hidden states, shape ``(U_total, hidden_size)``, after the final norm when configured.
        """
        if self.diagonal != 0:
            raise NotImplementedError("forward_packed implements the causal mask of diagonal 0.")
        for layer in self.layers:
            decoder_states = layer.forward_packed(decoder_states, decoder_context, encoder_states, encoder_context)
        if self.final_layer_norm is not None:
            decoder_states = self.final_layer_norm(decoder_states)
        return decoder_states

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
