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

"""
GPT components for the NeMo Models tutorial.
This module contains the neural network components used in the tutorial 01_NeMo_Models.ipynb
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from nemo.core import NeuralModule, typecheck
from nemo.core.neural_types import EmbeddedTextType, EncodedRepresentation, Index, LogitsType, NeuralType
from nemo.core.neural_types.elements import *


# Custom Element Types
class AttentionType(EncodedRepresentation):
    """Basic Attention Element Type"""


class SelfAttentionType(AttentionType):
    """Self Attention Element Type"""


class CausalSelfAttentionType(SelfAttentionType):
    """Causal Self Attention Element Type"""


# Neural Network Modules (not NeMo neural modules)
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, block_size, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, n_embd, block_size, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, block_size, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# NeMo Neural Modules
class GPTEmbedding(NeuralModule):
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, embd_pdrop: float = 0.0):
        super().__init__()

        # input embedding stem: drop(content + position)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

    @typecheck()
    def forward(self, idx):
        b, t = idx.size()

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        return x

    @property
    def input_types(self):
        return {'idx': NeuralType(('B', 'T'), Index())}

    @property
    def output_types(self):
        return {'embeddings': NeuralType(('B', 'T', 'C'), EmbeddedTextType())}


class GPTTransformerEncoder(NeuralModule):
    def __init__(
        self,
        n_embd: int,
        block_size: int,
        n_head: int,
        n_layer: int,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[Block(n_embd, block_size, n_head, attn_pdrop, resid_pdrop) for _ in range(n_layer)]
        )

    @typecheck()
    def forward(self, embed):
        return self.blocks(embed)

    @property
    def input_types(self):
        return {'embed': NeuralType(('B', 'T', 'C'), EmbeddedTextType())}

    @property
    def output_types(self):
        return {'encoding': NeuralType(('B', 'T', 'C'), CausalSelfAttentionType())}


class GPTDecoder(NeuralModule):
    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)  # no need for extra bias due to one in ln_f

    @typecheck()
    def forward(self, encoding):
        x = self.ln_f(encoding)
        logits = self.head(x)
        return logits

    @property
    def input_types(self):
        return {'encoding': NeuralType(('B', 'T', 'C'), EncodedRepresentation())}

    @property
    def output_types(self):
        return {'logits': NeuralType(('B', 'T', 'C'), LogitsType())}
