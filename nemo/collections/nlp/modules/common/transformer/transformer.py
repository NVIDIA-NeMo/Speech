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

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf.omegaconf import MISSING

from nemo.collections.nlp.modules.common.decoder_module import DecoderModule
from nemo.collections.nlp.modules.common.encoder_module import EncoderModule
from nemo.collections.nlp.modules.common.transformer.transformer_decoders import TransformerDecoder
from nemo.collections.nlp.modules.common.transformer.transformer_encoders import TransformerEncoder
from nemo.collections.nlp.modules.common.transformer.transformer_modules import TransformerEmbedding
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import ChannelType, NeuralType

# @dataclass
# class TransformerConfig:
#     # named model arguments
#     library: str = 'nemo'
#     model_name: Optional[str] = None
#     pretrained: bool = False


@dataclass
class NeMoTransformerConfig:
    # must be configured by the user
    hidden_size: int = MISSING
    num_layers: int = MISSING
    inner_size: int = MISSING
    num_attention_heads: int = MISSING

    # embedding
    max_sequence_length: int = 512
    num_token_types: int = 2
    embedding_dropout: float = 0.0
    learn_positional_encodings: bool = False

    # transformer
    ffn_dropout: float = 0.0
    attn_score_dropout: float = 0.0
    attn_layer_dropout: float = 0.0
    hidden_act: str = 'relu'
    pre_ln: bool = False
    pre_ln_final_layer_norm: bool = True

    # named model arguments
    library: str = 'nemo'
    model_name: Optional[str] = None
    pretrained: bool = False


@dataclass
class NeMoTransformerEncoderConfig(NeMoTransformerConfig):
    mask_future: bool = False


@dataclass
class NeMoTransformerDecoderConfig(NeMoTransformerConfig):
    r2l: bool = False


class TransformerEncoderNM(EncoderModule, Exportable):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        inner_size: int,
        num_attention_heads: int,
        max_sequence_length: int = 512,
        num_token_types: int = 2,
        embedding_dropout: float = 0.0,
        learn_positional_encodings: bool = False,
        ffn_dropout: float = 0.0,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        hidden_act: str = 'relu',
        mask_future: bool = False,
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._max_sequence_length = max_sequence_length

        self._embedding = TransformerEmbedding(
            vocab_size=self._vocab_size,
            hidden_size=self._hidden_size,
            max_sequence_length=max_sequence_length,
            num_token_types=num_token_types,
            embedding_dropout=embedding_dropout,
            learn_positional_encodings=learn_positional_encodings,
            padding_idx=padding_idx,
        )

        self._encoder = TransformerEncoder(
            hidden_size=self._hidden_size,
            num_layers=num_layers,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            ffn_dropout=ffn_dropout,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            hidden_act=hidden_act,
            mask_future=mask_future,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )

    @typecheck()
    def forward(self, input_ids, encoder_mask):
        embeddings = self._embedding(input_ids=input_ids)
        encoder_hidden_states = self._encoder(encoder_states=embeddings, encoder_mask=encoder_mask)
        return encoder_hidden_states

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def embedding(self):
        return self._embedding

    @property
    def encoder(self):
        return self._encoder

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        sz = (max_batch, max_dim)
        input_ids = torch.randint(low=0, high=2048, size=sz, device=sample.device)
        encoder_mask = torch.randint(low=0, high=1, size=sz, device=sample.device)
        return tuple([input_ids, encoder_mask])


class TransformerDecoderNM(DecoderModule, Exportable):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        inner_size: int,
        num_attention_heads: int,
        max_sequence_length: int = 512,
        num_token_types: int = 2,
        embedding_dropout: float = 0.0,
        learn_positional_encodings: bool = False,
        ffn_dropout: float = 0.0,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        hidden_act: str = 'relu',
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._max_sequence_length = max_sequence_length
        self.num_states = num_layers + 1
        self.return_mems = False
        if pre_ln_final_layer_norm:
            self.num_states += 1

        self._embedding = TransformerEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_sequence_length=max_sequence_length,
            num_token_types=num_token_types,
            embedding_dropout=embedding_dropout,
            learn_positional_encodings=learn_positional_encodings,
            padding_idx=padding_idx,
        )

        self._decoder = TransformerDecoder(
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            ffn_dropout=ffn_dropout,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )

    @typecheck()
    def forward(
        self, input_ids, decoder_mask, encoder_embeddings, encoder_mask, decoder_mems=None,
    ):
        start_pos = 0
        if decoder_mems is not None:
            start_pos = input_ids.shape[1] - 1
            input_ids = input_ids[:, -1:]
            decoder_mask = decoder_mask[:, -1:]
            decoder_mems = torch.transpose(decoder_mems, 0, 1)
        decoder_embeddings = self._embedding(input_ids=input_ids, start_pos=start_pos)
        decoder_hidden_states = self._decoder(
            decoder_states=decoder_embeddings,
            decoder_mask=decoder_mask,
            encoder_states=encoder_embeddings,
            encoder_mask=encoder_mask,
            decoder_mems_list=decoder_mems,
            return_mems=self.return_mems,
            return_mems_as_list=False,
        )
        if self.return_mems:
            decoder_hidden_states = torch.transpose(decoder_hidden_states, 0, 1)
        return decoder_hidden_states

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def embedding(self):
        return self._embedding

    @property
    def decoder(self):
        return self._decoder

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        sz = (max_batch, max_dim)
        input_ids = torch.randint(low=0, high=2048, size=sz, device=sample.device)
        encoder_mask = torch.randint(low=0, high=1, size=sz, device=sample.device)
        mem_size = [max_batch, self.num_states, max_dim - 1, self._hidden_size]
        decoder_mems = torch.rand(mem_size, device=sample.device)
        return tuple([input_ids, encoder_mask, self._embedding(input_ids), encoder_mask, decoder_mems])

    def _prepare_for_export(self, **kwargs):
        self._decoder.diagonal = None
        self.return_mems = True
        super()._prepare_for_export(**kwargs)

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        if self.return_mems:
            return {"last_hidden_states": NeuralType(('B', 'D', 'T', 'D'), ChannelType())}
        else:
            return {"last_hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}
