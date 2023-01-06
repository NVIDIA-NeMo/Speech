# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

import numpy as np
import torch
from torch import nn
from torch.nn.functional import gelu

from nemo.collections.common.parts import form_attention_mask
from nemo.utils import logging

__all__ = ["TransformerEmbedding", "AttentionBridge"]


class FixedPositionalEncoding(nn.Module):
    """
    Fixed positional encoding (embedding layer) from sine and cosine functions
    of different frequencies according to https://arxiv.org/abs/1706.03762

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
    """

    def __init__(self, hidden_size, max_sequence_length=512):
        super().__init__()

        self._hidden_size = hidden_size
        self._max_sequence_length = max_sequence_length
        self._build_pos_enc(hidden_size=self._hidden_size, max_sequence_length=self._max_sequence_length)

    def _build_pos_enc(self, hidden_size, max_sequence_length, device=None):
        """
        Builds/replaces pre-computed positional encoding.
        """
        pos_enc = torch.zeros(max_sequence_length, hidden_size, device=device)
        position = torch.arange(0.0, max_sequence_length).unsqueeze(1)
        coef = -math.log(10000.0) / hidden_size
        div_term = torch.exp(coef * torch.arange(0.0, hidden_size, 2))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc.div_(math.sqrt(hidden_size))
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, position_ids):
        max_pos_id = position_ids.max()
        # update positional encoding if needed
        if max_pos_id >= self._max_sequence_length:
            logging.warning(
                f'Max position id {max_pos_id} is greater than max sequence length {self._max_sequence_length}. Expanding position embeddings just for this batch. This is not expected to work very well. Consider chunking your input into smaller sequences.'
            )
            self._build_pos_enc(
                hidden_size=self._hidden_size, max_sequence_length=max_pos_id + 1, device=position_ids.device,
            )

        embeddings = torch.embedding(self.pos_enc, position_ids)

        # Revert expansion of position embeddings since this wall checkpoint size mismatches.
        if max_pos_id >= self._max_sequence_length:
            self._build_pos_enc(
                hidden_size=self._hidden_size,
                max_sequence_length=self._max_sequence_length,
                device=position_ids.device,
            )
        return embeddings


class TransformerEmbedding(nn.Module):
    """
    Embedding from token and position embeddings.
    Optionally add token_type embedding (e.g. type of the sentence in BERT).

    Args:
        vocab_size: size of the vocabulary
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
        num_token_types: number of different token types
            (e.g. tokens of sentence A and tokens of sentence B in BERT)
        embedding_dropout: probability of dropout applied to embeddings
        learn_positional_encodings: whether to learn positional encodings or
            use fixed (sine-cosine) ones
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_sequence_length: int = 512,
        num_token_types: int = 2,
        embedding_dropout: float = 0.0,
        learn_positional_encodings: bool = False,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.learn_positional_encodings = learn_positional_encodings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        if learn_positional_encodings:
            self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)
        else:
            self.position_embedding = FixedPositionalEncoding(hidden_size, max_sequence_length)
        if num_token_types > 0:
            self.token_type_embedding = nn.Embedding(num_token_types, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(embedding_dropout)

    def forward(self, input_ids, token_type_ids=None, start_pos=0):
        seq_length = input_ids.size(1)
        # we fail here only with parametric positional embedding. FixedPositionalEncoding automatically extends.
        if self.learn_positional_encodings and (seq_length > self.max_sequence_length):
            raise ValueError(
                f"Input sequence is longer than maximum allowed sequence length for positional encoding. "
                f"Got {seq_length} and {self.max_sequence_length}"
            )
        position_ids = torch.arange(
            start=start_pos, end=start_pos + seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).repeat(input_ids.size(0), 1)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention layer.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            whole layer, but before layer normalization
    """

    def __init__(self, hidden_size, num_attention_heads, attn_score_dropout=0.0, attn_layer_dropout=0.0):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number "
                "of attention heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_head_size = int(hidden_size / num_attention_heads)
        self.attn_scale = math.sqrt(math.sqrt(self.attn_head_size))

        self.query_net = nn.Linear(hidden_size, hidden_size)
        self.key_net = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_score_dropout)
        self.layer_dropout = nn.Dropout(attn_layer_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, queries, keys, values, attention_mask):

        # attention_mask is needed to hide the tokens which correspond to [PAD]
        # in the case of BERT, or to hide the future tokens in the case of
        # vanilla language modeling and translation
        query = self.query_net(queries)
        key = self.key_net(keys)
        value = self.value_net(values)
        query = self.transpose_for_scores(query) / self.attn_scale
        key = self.transpose_for_scores(key) / self.attn_scale
        value = self.transpose_for_scores(value)

        # for numerical stability we pre-divide query and key by sqrt(sqrt(d))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        # output projection
        output_states = self.out_projection(context)
        output_states = self.layer_dropout(output_states)
        return output_states


class PositionWiseFF(nn.Module):
    """
    Position-wise feed-forward network of Transformer block.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        ffn_dropout: probability of dropout applied to net output
        hidden_act: activation function used between two linear layers
    """

    def __init__(self, hidden_size, inner_size, ffn_dropout=0.0, hidden_act="relu"):
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, inner_size)
        self.dense_out = nn.Linear(inner_size, hidden_size)
        self.layer_dropout = nn.Dropout(ffn_dropout)
        ACT2FN = {"gelu": gelu, "relu": torch.relu}
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        output_states = self.dense_in(hidden_states)
        output_states = self.act_fn(output_states)
        output_states = self.dense_out(output_states)
        output_states = self.layer_dropout(output_states)
        return output_states


class AttentionBridge(torch.nn.Module):
    """
    A multi-head attention bridge to project a variable-size hidden states
    to k hidden states (per attention head).

    Code is based on the paper https://arxiv.org/pdf/1703.03130.pdf
    """

    def __init__(self, hidden_size, k, bridge_size):
        """
        hidden_size - size of input hidden state
        k - number of attention heads
        bridge_size - size of internal feed forward weights (i.e., attention head size)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.k = k
        self.bridge_size = bridge_size

        self.attn_scale = np.sqrt(np.sqrt(self.bridge_size))

        # build model

        self.W1 = torch.nn.Linear(hidden_size, bridge_size, bias=False)
        self.W2 = torch.nn.Linear(bridge_size, k, bias=False)
        self.act = torch.nn.ReLU()

    def forward(self, hidden, hidden_mask=None, return_ortho_loss=False):
        """
        Project hidden [B x N x H] to fixed-size [B x k x H]

        return_ortho_loss - if True returns loss term to encourage
                              orthogonal attention vectors
        """

        attention_scores = self.W2(self.act(self.W1(hidden) / self.attn_scale) / self.attn_scale).transpose(-1, -2)

        attention_mask = form_attention_mask(hidden_mask)
        if attention_mask is not None:
            attention_mask.squeeze_(1)
            attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)

        A = torch.softmax(attention_scores, dim=-1)
        M = A @ hidden

        if return_ortho_loss:
            ortho_loss = ((A @ A.transpose(-1, -2)) - torch.eye(self.k).type_as(A)).pow(2).sum()

            return M, ortho_loss
        else:
            return M
