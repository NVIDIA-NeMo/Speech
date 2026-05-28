# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import torch

from nemo.collections.asr.modules.transformer.transformer_encoders import TransformerEncoder

__all__ = ["PoolingEncoder"]


class PoolingEncoder(torch.nn.Module):

    _SUPPORTED_ARCH = ["max", "avg"]

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
        hidden_steps: int = 4,
        hidden_init_method: str = "default",
        hidden_blocks: int = 2,
        pooling_type: str = "max",
    ):
        super().__init__()

        # minimal steps to allow reduction
        self._hidden_steps = hidden_steps
        self._hidden_init_method = hidden_init_method
        self._hidden_blocks = hidden_blocks
        self._pooling_type = pooling_type

        if self._hidden_steps < 2:
            raise ValueError("Expected hidden_steps >= 2 but received hidden_steps = {self._hidden_steps}")

        if self.hidden_init_method not in self.supported_init_methods:
            raise ValueError(
                "Unknown hidden_init_method = {hidden_init_method}, supported methods are {supported_init_methods}".format(
                    hidden_init_method=self.hidden_init_method,
                    supported_init_methods=self.supported_init_methods,
                )
            )

        if self._pooling_type not in self.supported_arch:
            raise ValueError(f"Unknown pooling_type = {pooling_type}. Available values = {self.supported_arch}")

        # self-attention encoder
        layer = TransformerEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            inner_size=inner_size,
            mask_future=mask_future,
            num_attention_heads=num_attention_heads,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )
        self.self_att_layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(hidden_blocks)])

        self.pooling = self._build_pooling_module()

    def _build_pooling_module(self):
        """
        Returns pooling module.
        Allows to override for child classes.
        """
        if self._pooling_type == "max":
            pooling = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        elif self._pooling_type == "avg":
            pooling = torch.nn.AvgPool1d(kernel_size=2, stride=2)

        return pooling

    @property
    def supported_arch(self):
        return self._SUPPORTED_ARCH

    @property
    def supported_init_methods(self):
        return ["default"]

    @property
    def hidden_steps(self):
        return self._hidden_steps

    @property
    def hidden_blocks(self):
        return self._hidden_blocks

    @property
    def hidden_init_method(self):
        return self._hidden_init_method

    def forward(self, encoder_states, encoder_mask):
        """
        Args:
            encoder_states: output of the encoder (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
        """
        # initialize hidden state
        hidden_mask = encoder_mask
        hidden_states = encoder_states

        # apply block (self-attention, max-pool) multiple times
        for self_att in self.self_att_layers:
            residual = hidden_states

            # self-attention over hidden
            hidden_states = self_att(encoder_states=hidden_states, encoder_mask=hidden_mask)

            hidden_states += residual

            # max pool reduction if possible
            if hidden_states.shape[1] >= self.hidden_steps:
                # max pool hidden states
                hidden_states = hidden_states.permute(0, 2, 1)
                hidden_states = self.pooling(hidden_states)
                hidden_states = hidden_states.permute(0, 2, 1)

                # max pool mask
                hidden_mask = (
                    self.pooling(hidden_mask.unsqueeze(0).type_as(hidden_states)).squeeze(0).type_as(hidden_mask)
                )

        return hidden_states, hidden_mask
