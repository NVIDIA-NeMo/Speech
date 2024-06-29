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

import copy
from typing import List, Optional, Set

import torch
import torch.nn as nn
from omegaconf import DictConfig

from nemo.collections.asr.modules.transformer.transformer_modules import MultiHeadAttention, PositionWiseFF
from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.common.parts import form_attention_mask
from nemo.core.classes.mixins import adapter_mixins

__all__ = ["TransformerEncoder"]


class TransformerEncoderBlock(nn.Module, AttentionAdapterModuleMixin):
    """
    Building block of Transformer encoder.

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
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.first_sub_layer = MultiHeadAttention(
            hidden_size, num_attention_heads, attn_score_dropout, attn_layer_dropout
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.second_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)

        # Information for the adapter module mixin
        self.self_attention_model = "transf_abs"

    def forward_preln(self, encoder_query, encoder_mask, encoder_keys):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN
        """
        residual = encoder_query
        encoder_query = self.layer_norm_1(encoder_query)
        encoder_keys = self.layer_norm_1(encoder_keys)
        self_attn_output = self.first_sub_layer(encoder_query, encoder_keys, encoder_keys, encoder_mask)
        self_attn_output += residual

        if self.is_adapter_available():
            # Call the MHA adapters
            pack_input = {
                'x': self_attn_output,
                'loc': 'mha',
                'att_mask': encoder_mask,
                'pos_emb': None,
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            self_attn_output = pack_input['x']

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        output_states = self.second_sub_layer(self_attn_output)
        output_states += residual

        if self.is_adapter_available():
            # Call the Linear adapters
            pack_input = {
                'x': output_states,
                'loc': 'post',
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            output_states = pack_input['x']

        return output_states

    def forward_postln(self, encoder_query, encoder_mask, encoder_keys):
        """
        Post-LayerNorm block
        Order of operations: Self-Attn -> Residual -> LN -> Cross-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        self_attn_output = self.first_sub_layer(encoder_query, encoder_keys, encoder_keys, encoder_mask)
        self_attn_output += encoder_query

        if self.is_adapter_available():
            # Call the MHA adapters
            pack_ip = {
                'x': self_attn_output,
                'loc': 'mha',
                'att_mask': encoder_mask,
                'pos_emb': None,
            }
            pack_ip = self.forward_enabled_adapters(pack_ip)
            self_attn_output = pack_ip['x']

        self_attn_output = self.layer_norm_1(self_attn_output)

        output_states = self.second_sub_layer(self_attn_output)
        output_states += self_attn_output

        if self.is_adapter_available():
            # Call the linear adapters
            pack_ip = {
                'x': output_states,
                'loc': 'post',
            }
            pack_ip = self.forward_enabled_adapters(pack_ip)
            output_states = pack_ip['x']

        output_states = self.layer_norm_2(output_states)

        return output_states

    def forward(self, encoder_query, encoder_mask, encoder_keys):
        if self.pre_ln:
            return self.forward_preln(encoder_query, encoder_mask, encoder_keys)
        else:
            return self.forward_postln(encoder_query, encoder_mask, encoder_keys)

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


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        mask_future: bool = False,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
    ):
        super().__init__()

        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.final_layer_norm = None

        self.d_model = hidden_size

        layer = TransformerEncoderBlock(
            hidden_size,
            inner_size,
            num_attention_heads,
            attn_score_dropout,
            attn_layer_dropout,
            ffn_dropout,
            hidden_act,
            pre_ln,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.diag = 0 if mask_future else None

    def _get_memory_states(self, encoder_states, encoder_mems_list=None, i=0):
        if encoder_mems_list is not None:
            memory_states = torch.cat((encoder_mems_list[i], encoder_states), dim=1)
        else:
            memory_states = encoder_states
        return memory_states

    def forward(self, encoder_states, encoder_mask, encoder_mems_list=None, return_mems=False):
        """
        Args:
            encoder_states: output of the embedding_layer (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
            encoder_mems_list: list of the cached encoder hidden states
                for fast autoregressive generation which will be used instead
                of encoder_states as keys and values if not None
            return_mems: bool, whether to return outputs of all encoder layers
                or the last layer only
        """

        encoder_attn_mask = form_attention_mask(encoder_mask, self.diag)

        memory_states = self._get_memory_states(encoder_states, encoder_mems_list, 0)
        cached_mems_list = [memory_states]

        for i, layer in enumerate(self.layers):
            encoder_states = layer(encoder_states, encoder_attn_mask, memory_states)
            memory_states = self._get_memory_states(encoder_states, encoder_mems_list, i + 1)
            cached_mems_list.append(memory_states)

        if self.final_layer_norm is not None:
            encoder_states = self.final_layer_norm(encoder_states)
            memory_states = self._get_memory_states(encoder_states, encoder_mems_list, i + 1)
            cached_mems_list.append(memory_states)

        if return_mems:
            return cached_mems_list
        else:
            return cached_mems_list[-1]


class TransformerEncoderAdapter(TransformerEncoder, adapter_mixins.AdapterModuleMixin):

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
if adapter_mixins.get_registered_adapter(TransformerEncoder) is None:
    adapter_mixins.register_adapter(base_class=TransformerEncoder, adapter_class=TransformerEncoderAdapter)
