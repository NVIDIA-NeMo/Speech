# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import ChannelType, EncodedRepresentation, MaskType, NeuralType

__all__ = ["TokensHeadEmb", "DecEmb", "EncEmb"]


class TokensHeadEmb(torch.nn.Module, Exportable):
    """
    Combines decoder_embedding with the tokens_head layer to simulate the classifier in NemoNMT
    """

    def __init__(self, decoder_embedding, tokens_head, device):
        super(TokensHeadEmb, self).__init__()

        self.decoder_embedding = decoder_embedding
        self.tokens_head_bias = tokens_head.bias
        self.device = device

        # properties needed for export
        self.training = False

    def train(self, dummy_input):
        return None

    def modules(self):
        return []

    def forward(self, dec_output):
        if isinstance(dec_output, list):
            dec_output = dec_output[0]

        if self.tokens_head_bias is not None:
            return F.linear(dec_output, self.decoder_embedding.word_embeddings.weight, self.tokens_head_bias).float()
        return F.linear(dec_output, self.decoder_embedding.word_embeddings.weight).float()

    def input_example(self, max_batch=1, max_dim=768, seq_len=6):
        return [
            torch.randint(low=-3, high=3, size=(max_batch, seq_len, max_dim), device=self.device, dtype=torch.float32)
        ]

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "hidden_states": NeuralType(('B', 'T', 'D'), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"log_probs": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    def input_names(self) -> List[str]:
        return ['hidden_states']

    @property
    def output_names(self) -> List[str]:
        return ['log_probs']


class DecEmb(torch.nn.Module, Exportable):
    """
    Combines decoder_embedding with the decoder component
    """

    def __init__(self, decoder_embedding, decoder, rpe, device):
        super(DecEmb, self).__init__()

        self.decoder_embedding = decoder_embedding
        self.decoder = decoder
        self.device = device
        self.rpe = rpe

        # properties needed for export
        self.training = False

    def train(self, dummy_input):
        return None

    def modules(self):
        return (self.decoder_embedding, self.decoder)

    def forward(self, input_ids, decoder_mask, encoder_mask, encoder_embeddings, decoder_mems):
        position_ids = build_position_ids(input_ids)
        dec_input = self.decoder_embedding(input_ids, position_ids, token_type_ids=None)

        rpe = None
        if self.rpe is not None:
            rpe = self.rpe(query_seq_length=input_ids.size(1), key_seq_length=input_ids.size(1),)

        dec_out = (
            self.decoder(
                dec_input,
                decoder_mask,
                encoder_embeddings.permute(1, 0, 2),
                encoder_mask,
                dec_self_attention_relative_position_bias=rpe,
            )
            .permute(1, 0, 2)
            .float()
        )

        zeros = torch.zeros(
            (decoder_mems.shape[0], self.decoder.num_layers, dec_out.shape[1], decoder_mems.shape[-1])
        ).to(self.device)

        return torch.cat((zeros, dec_out.unsqueeze(1)), dim=1)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def input_example(self, max_batch=1, max_dim=768, seq_len=6):
        enc_output = torch.randint(
            low=-3, high=3, size=(max_batch, seq_len, max_dim), device=self.device, dtype=torch.float32
        )
        enc_attn_mask = torch.tensor([[1 for _ in range(seq_len)]]).to(self.device)

        dec_len = random.randint(10, 128)
        dec_input = torch.randint(low=0, high=1000, size=(max_batch, dec_len), device=self.device)
        dec_attn_mask = torch.tensor([[1 for _ in range(dec_len)]]).to(self.device)

        # constant decoder_mems as placeholder for now
        decoder_mems = torch.zeros([max_batch, self.decoder.num_layers + 1, seq_len, max_dim], dtype=torch.float32).to(
            self.device
        )

        # input_ids, decoder_mask, encoder_mask, encoder_embeddings

        return tuple([dec_input, dec_attn_mask, enc_attn_mask, enc_output, decoder_mems])

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T', 'D'), ChannelType()),
            "decoder_mask": NeuralType(('B', 'T'), MaskType()),
            "encoder_mask": NeuralType(('B', 'T', 'D'), ChannelType()),
            "encoder_embeddings": NeuralType(('B', 'T'), MaskType()),
            "decoder_mems": NeuralType(('B', 'S', 'T', 'D'), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "last_hidden_states": NeuralType(('B', 'S', 'T', 'D'), EncodedRepresentation()),
        }

    @property
    def input_names(self) -> List[str]:
        return ['input_ids', 'decoder_mask', 'encoder_mask', 'encoder_embeddings', 'decoder_mems']

    @property
    def output_names(self) -> List[str]:
        return ['last_hidden_states']


class EncEmb(torch.nn.Module, Exportable):
    """
    Combines encoder_embedding with the encoder component
    """

    def __init__(self, encoder_embedding, encoder, rpe, device):
        super(EncEmb, self).__init__()

        self.encoder_embedding = encoder_embedding
        self.encoder = encoder
        self.device = device
        self.rpe = rpe

        # properties needed for export
        self.training = False

    def train(self, dummy_input):
        return None

    def modules(self):
        return (self.encoder_embedding, self.encoder)

    def forward(self, input_ids, encoder_mask):
        if self.rpe is None:
            position_ids = build_position_ids(input_ids)
        else:
            position_ids = None

        enc_input = self.encoder_embedding(input_ids, position_ids, token_type_ids=None)

        # pass input through the encoder
        enc_seq_length = input_ids.size(1)

        rpe = None
        if self.rpe is not None:
            rpe = self.rpe(query_seq_length=enc_seq_length, key_seq_length=enc_seq_length,)

        return (
            self.encoder(
                enc_input=enc_input, enc_attn_mask=encoder_mask, enc_self_attention_relative_position_bias=rpe
            )
            .permute(1, 0, 2)
            .float()
        )

    def input_example(self, max_batch=1, max_dim=30000, seq_len=6):
        seq_len = random.randint(0, 128)
        return (
            torch.randint(0, max_dim, (max_batch, seq_len)).to(self.device),
            torch.ones((max_batch, seq_len), dtype=int).to(self.device),
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "encoder_mask": NeuralType(('B', 'T'), MaskType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"last_hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    def input_names(self) -> List[str]:
        return ['input_ids', 'encoder_mask']

    @property
    def output_names(self) -> List[str]:
        return ['last_hidden_states']
