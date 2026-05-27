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

import torch

from nemo.collections.asr.modules.transformer.transformer_encoders import TransformerEncoder
from nemo.collections.asr.modules.transformer.transformer_modules import AttentionBridge

__all__ = ["BridgeEncoder"]


class BridgeEncoder(torch.nn.Module):
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
        hidden_steps: int = 32,
        hidden_init_method: str = "default",
        hidden_blocks: int = 0,
    ):
        super().__init__()

        self._hidden_steps = hidden_steps
        self._hidden_init_method = hidden_init_method
        self._hidden_blocks = hidden_blocks

        if self._hidden_init_method == "default":
            self._hidden_init_method = "enc_shared"

        if self.hidden_init_method not in self.supported_init_methods:
            raise ValueError(
                "Unknown hidden_init_method = {hidden_init_method}, supported methods are {supported_init_methods}".format(
                    hidden_init_method=self.hidden_init_method,
                    supported_init_methods=self.supported_init_methods,
                )
            )

        # attention bridge
        self.att_bridge = AttentionBridge(
            hidden_size=hidden_size,
            k=hidden_steps,
            bridge_size=inner_size,
        )

        if self.hidden_init_method == "enc":
            self.init_hidden_enc = TransformerEncoder(
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

        # self attention
        self.hidden_enc = TransformerEncoder(
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

    @property
    def supported_init_methods(self):
        return ["enc_shared", "identity", "enc"]

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
        # self-attention over input
        if self.hidden_init_method == "enc_shared":
            residual = encoder_states
            hidden_states = self.hidden_enc(encoder_states=encoder_states, encoder_mask=encoder_mask)
            # residual connection
            hidden_states += residual
        elif self.hidden_init_method == "identity":
            hidden_states = encoder_states
        elif self.hidden_init_method == "enc":
            residual = encoder_states
            hidden_states = self.init_hidden_enc(encoder_states=encoder_states, encoder_mask=encoder_mask)
            # residual connection
            hidden_states += residual

        # project encoder states to a fixed steps hidden using k attention heads
        hidden_states = self.att_bridge(hidden=hidden_states, hidden_mask=encoder_mask)

        # all hidden values are active
        hidden_mask = torch.ones(
            encoder_states.shape[0], self._hidden_steps, dtype=encoder_mask.dtype, device=encoder_mask.device
        )

        # apply self-attention over fixed-size hidden_states
        for block in range(self._hidden_blocks):
            residual = hidden_states
            hidden_states = self.hidden_enc(encoder_states=hidden_states, encoder_mask=hidden_mask)
            # residual connection
            hidden_states += residual

        return hidden_states, hidden_mask
