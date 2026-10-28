# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

import math

import numpy as np
import torch
from torch import nn
from torch.nn.functional import gelu

from nemo.collections.asr.parts.submodules.multi_head_attention import RotaryPositionalEncoding
from nemo.collections.asr.parts.submodules.packed_attention import PackedAttentionContext, packed_cross_attention
from nemo.collections.common.parts import form_attention_mask

__all__ = ["TransformerEmbedding", "AttentionBridge", "RMSNorm"]


class RMSNorm(nn.Module):
    """
    Root mean square normalization (Zhang and Sennrich, 2019) with a learned scale and no bias.

    Written with elementary ops in the input dtype so that ONNX export works: ``torch.nn.RMSNorm`` lowers to
    ``aten::rms_norm``, which the exporter does not support, and dtype casts around the computation leave the
    legacy exporter's graph with dangling references. The state is ``weight``, as in ``torch.nn.RMSNorm``.

    Args:
        normalized_shape: size of the normalized last axis
        eps: added to the mean square before the square root
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class FixedPositionalEncoding(nn.Module):
    """
    Fixed positional encoding (embedding layer) from sine and cosine functions
    of different frequencies according to https://arxiv.org/abs/1706.03762

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
        layout: ``interleaved`` (default) alternates sine and cosine along the feature axis with the
            frequencies ``10000**(-2i/hidden_size)``; ``tensor2tensor`` stores all sines before all
            cosines with the frequencies ``10000**(-i/(hidden_size/2 - 1))`` of the original
            tensor2tensor implementation.
        scale: multiplier of the table; the default ``1/sqrt(hidden_size)`` matches token embeddings
            of unit scale, ``1.0`` matches token embeddings scaled by ``sqrt(hidden_size)``.
    """

    def __init__(self, hidden_size, max_sequence_length=512, layout: str = "interleaved", scale: float | None = None):
        super().__init__()

        if layout not in ("interleaved", "tensor2tensor"):
            raise ValueError(f"layout must be 'interleaved' or 'tensor2tensor', got {layout!r}")
        self._hidden_size = hidden_size
        self._max_sequence_length = max_sequence_length
        self._layout = layout
        self._scale = 1.0 / math.sqrt(hidden_size) if scale is None else scale
        self._build_pos_enc(hidden_size=self._hidden_size, max_sequence_length=self._max_sequence_length)

    def _build_pos_enc(self, hidden_size, max_sequence_length, device=None):
        """
        Builds/replaces pre-computed positional encoding.
        """
        pos_enc = torch.zeros(max_sequence_length, hidden_size, device=device)
        position = torch.arange(0.0, max_sequence_length).unsqueeze(1)
        if self._layout == "tensor2tensor":
            half = hidden_size // 2
            div_term = torch.exp(-math.log(10000.0) / max(half - 1, 1) * torch.arange(0.0, half))
            pos_enc[:, :half] = torch.sin(position * div_term)
            pos_enc[:, half:] = torch.cos(position * div_term)
        else:
            coef = -math.log(10000.0) / hidden_size
            div_term = torch.exp(coef * torch.arange(0.0, hidden_size, 2))
            pos_enc[:, 0::2] = torch.sin(position * div_term)
            pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc.mul_(self._scale)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, position_ids):
        embeddings = torch.embedding(self.pos_enc, position_ids)
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
        scale_embeddings: multiply the token embeddings by ``sqrt(hidden_size)`` as in the original
            Transformer; the fixed positional table is then used unscaled
        positional_encoding_layout: layout of the fixed positional table, see ``FixedPositionalEncoding``
        use_layer_norm: apply the LayerNorm after adding the positions (default True)
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_sequence_length=512,
        num_token_types=2,
        embedding_dropout=0.0,
        learn_positional_encodings=False,
        scale_embeddings: bool = False,
        positional_encoding_layout: str = "interleaved",
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.learn_positional_encodings = learn_positional_encodings
        self.embedding_scale = math.sqrt(hidden_size) if scale_embeddings else None
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if learn_positional_encodings:
            self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)
        else:
            self.position_embedding = FixedPositionalEncoding(
                hidden_size,
                max_sequence_length,
                layout=positional_encoding_layout,
                scale=1.0 if scale_embeddings else None,
            )
        if num_token_types > 0:
            self.token_type_embedding = nn.Embedding(num_token_types, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5) if use_layer_norm else None
        self.dropout = nn.Dropout(embedding_dropout)

    def forward(self, input_ids, token_type_ids=None, start_pos=0):
        seq_length = input_ids.size(1)
        # we fail here only with parametric positional embedding. FixedPositionalEncoding automatically extends.
        if self.learn_positional_encodings and (seq_length > self.max_sequence_length):
            raise ValueError(
                f"Input sequence is longer than maximum allowed sequence length for positional encoding. "
                f"Got {seq_length} and {self.max_sequence_length}"
            )

        # prepare position embedding for asynchronius decoding (canary streaming)
        if torch.is_tensor(start_pos):
            # under tracing the shape-derived start position is a CPU tensor even for CUDA inputs
            shift_pos = start_pos.unsqueeze(-1).to(input_ids.device)
            start_pos = 0
        else:
            shift_pos = None

        position_ids = torch.arange(
            start=start_pos, end=start_pos + seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).repeat(input_ids.size(0), 1)
        if torch.is_tensor(shift_pos):
            # shift_pos is a tensor, so we need to add it to the position_ids
            # and make sure that the resulting position_ids are within the
            # range of the positional embedding
            position_ids = position_ids + shift_pos

        token_embeddings = self.token_embedding(input_ids)
        if self.embedding_scale is not None:
            token_embeddings = token_embeddings * self.embedding_scale
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

    def forward_packed(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Embed token-flat ids with explicit positions.

        Args:
            input_ids: ids of all sequences, shape ``(U_total,)``.
            position_ids: position of every id within its sequence, shape ``(U_total,)``.
            token_type_ids: optional token types, shape ``(U_total,)``.

        Returns:
            The embeddings, shape ``(U_total, hidden_size)``.
        """
        token_embeddings = self.token_embedding(input_ids)
        if self.embedding_scale is not None:
            token_embeddings = token_embeddings * self.embedding_scale
        embeddings = token_embeddings + self.position_embedding(position_ids)
        if token_type_ids is not None:
            embeddings = embeddings + self.token_type_embedding(token_type_ids)
        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention layer.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            whole layer, but before layer normalization
        use_bias: bias in the query, key, value and output projections (default True)
        rotary_base: theta base of a rotary position embedding applied to queries and keys (self-attention
            only); keys longer than the queries are treated as a cache prefix. None (default) for no rotation.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attn_score_dropout=0.0,
        attn_layer_dropout=0.0,
        return_xatt_scores=False,
        use_bias: bool = True,
        rotary_base: float | None = None,
    ):
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

        self.query_net = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.key_net = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.value_net = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.out_projection = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.rotary_pos_enc = (
            RotaryPositionalEncoding(d_k=self.attn_head_size, rope_base=rotary_base)
            if rotary_base is not None
            else None
        )

        self.attn_dropout = nn.Dropout(attn_score_dropout)
        self.layer_dropout = nn.Dropout(attn_layer_dropout)
        self.return_xatt_scores = return_xatt_scores

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
        if self.rotary_pos_enc is not None:
            # the rotation commutes with the scalar scaling above; the table grows with the key length
            self.rotary_pos_enc.extend_pe(key.size(2), key.device, key.dtype)
            query, key = self.rotary_pos_enc(query, key)

        # for numerical stability we pre-divide query and key by sqrt(sqrt(d))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context_hidden_size = context.size()[-1] * self.num_attention_heads
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (context_hidden_size,)
        context = context.view(*new_context_shape)

        # output projection
        output_states = self.out_projection(context)
        output_states = self.layer_dropout(output_states)

        extra_output = {}
        if self.return_xatt_scores:
            extra_output['xatt_scores'] = attention_probs

        return output_states, extra_output

    def forward_packed(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        query_context: PackedAttentionContext,
        key_context: PackedAttentionContext,
        causal: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Attention over token-flat queries and keys with explicit sequence boundaries instead of masks.

        Args:
            queries: shape ``(Q_total, hidden_size)``.
            keys: shape ``(K_total, hidden_size)``, packed in the same sequence order as the queries.
            values: shape ``(K_total, hidden_size)``.
            query_context: boundaries and positions of the queries.
            key_context: boundaries and positions of the keys; the same object as ``query_context`` in self-attention.
            causal: let every query see only the keys at its position or before (self-attention).

        Returns:
            The output states, shape ``(Q_total, hidden_size)``, and an empty ``extra_output``:
            the attention weights that the padded forward returns with ``return_xatt_scores``
            are not formed by the fused kernels.
        """
        heads, head_size = self.num_attention_heads, self.attn_head_size
        query = self.query_net(queries).view(-1, heads, head_size) / self.attn_scale
        key = self.key_net(keys).view(-1, heads, head_size) / self.attn_scale
        value = self.value_net(values).view(-1, heads, head_size)
        if self.rotary_pos_enc is not None:
            # a function-level import: the modules package initializes the encoders, which import this module
            from nemo.collections.asr.modules.transformer_encoder_utils import _apply_packed_rope

            if query_context is not key_context:
                raise NotImplementedError("The rotary position embedding applies to self-attention only.")
            self.rotary_pos_enc.extend_pe(query_context.max_seqlen, key.device, key.dtype)
            query, key = _apply_packed_rope(self.rotary_pos_enc, query, key, query_context.position_ids)
        # the scores are already scaled by the two divisions above
        context = packed_cross_attention(
            query,
            key,
            value,
            query_context,
            key_context,
            causal=causal,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=1.0,
        )
        output_states = self.out_projection(context.reshape(-1, heads * head_size))
        return self.layer_dropout(output_states), {}


class PositionWiseFF(nn.Module):
    """
    Position-wise feed-forward network of Transformer block.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        ffn_dropout: probability of dropout applied to net output
        hidden_act: activation function used between two linear layers: ``relu``, ``gelu``, or a gated
            variant ``swiglu``/``geglu``/``reglu`` whose first projection has twice the inner size and whose
            activated first half gates the second half (Shazeer, 2020)
        use_bias: bias in both projections (default True)
    """

    def __init__(self, hidden_size, inner_size, ffn_dropout=0.0, hidden_act="relu", use_bias: bool = True):
        super().__init__()
        ACT2FN = {"gelu": gelu, "relu": torch.relu, "swiglu": nn.functional.silu, "geglu": gelu, "reglu": torch.relu}
        if hidden_act not in ACT2FN:
            raise ValueError(f"hidden_act must be one of {sorted(ACT2FN)}, got {hidden_act!r}")
        self.gated = hidden_act in ("swiglu", "geglu", "reglu")
        self.dense_in = nn.Linear(hidden_size, 2 * inner_size if self.gated else inner_size, bias=use_bias)
        self.dense_out = nn.Linear(inner_size, hidden_size, bias=use_bias)
        self.layer_dropout = nn.Dropout(ffn_dropout)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        output_states = self.dense_in(hidden_states)
        if self.gated:
            gate, output_states = output_states.chunk(2, dim=-1)
            output_states = self.act_fn(gate) * output_states
        else:
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
