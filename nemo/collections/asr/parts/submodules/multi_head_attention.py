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
#

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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
#

"""
Part of this code is adopted from https://github.com/espnet/espnet
"""

import math
from functools import lru_cache
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.utils import avoid_float16_autocast_context

__all__ = [
    'RelPositionMultiHeadAttention',
    'RelPositionalEncoding',
    'PositionalEncoding',
]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer of Transformer.
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
        use_bias (bool): whether to remove bias in linear and conv layers
    """

    def __init__(self, n_head, n_feat, dropout_rate, max_cache_len=0, use_bias=True):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadAttention, self).__init__()
        self.cache_drop_size = None
        self.use_bias = use_bias
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate)

        self._max_cache_len = max_cache_len

    def forward_qkv(self, query, key, value):
        """Transforms query, key and value.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value (torch.Tensor): (batch, time2, size)
        returns:
            q (torch.Tensor): (batch, head, time1, size)
            k (torch.Tensor): (batch, head, time2, size)
            v (torch.Tensor): (batch, head, time2, size)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): (batch, time2, size)
            scores(torch.Tensor): (batch, time1, time2)
            mask(torch.Tensor): (batch, time1, time2)
        returns:
            value (torch.Tensor): transformed `value` (batch, time2, d_model) weighted by the attention scores
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, -10000.0)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask, pos_emb=None, cache=None):
        """Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            cache (torch.Tensor) : (batch, time_cache, size)

        returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
            out = self.forward_attention(v, scores, mask)
        if cache is None:
            return out
        else:
            return out, cache

    def update_cache(self, key, value, query, cache):
        if cache is not None:
            key = value = torch.cat([cache, key], dim=1)
            q_keep_size = query.shape[1] - self.cache_drop_size
            cache = torch.cat([cache[:, q_keep_size:, :], query[:, :q_keep_size, :]], dim=1)
        return key, value, query, cache


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
        use_bias (bool): whether to apply bias in linear and conv layers of MultiHeadAttention
    """

    def __init__(self, n_head, n_feat, dropout_rate, pos_bias_u, pos_bias_v, max_cache_len=0, use_bias=True):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            dropout_rate=dropout_rate,
            max_cache_len=max_cache_len,
            use_bias=use_bias,
        )
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable biases are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            # nn.init.normal_(self.pos_bias_u, 0.0, 0.02)
            # nn.init.normal_(self.pos_bias_v, 0.0, 0.02)
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
            cache (torch.Tensor) : (batch, time_cache, size)

        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)

            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)

            # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            # (batch, head, time1, time2)
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

            # compute matrix b and matrix d
            # (batch, head, time1, time2)
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            matrix_bd = self.rel_shift(matrix_bd)
            # drops extra elements in the matrix_bd to match the matrix_ac's size
            matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

            scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)

            out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        else:
            return out, cache


class RelPositionMultiHeadAttentionLongformer(RelPositionMultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with sliding window local+global attention from Longformer.
    Partially adapted from allenai (https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py)
    and huggingface (https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py)
    Paper: https://arxiv.org/abs/1901.02860 (Transformer-XL),
           https://arxiv.org/abs/2004.05150 (Longformer)
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
        pos_bias_u (Tensor): the positional bias matrix U
        pos_bias_v (Tensor): the positional bias matrix V
        att_context_size (List[int]): List of 2 ints corresponding to left and right attention context sizes.
        max_cache_len (int): the maximum size of cache
        global_tokens (int): number of tokens to be used for global attention
        global_tokens_spacing (int): how far apart the global tokens are
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate
        use_bias (bool): whether to apply bias in linear and conv layers of MultiHeadAttention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        pos_bias_u,
        pos_bias_v,
        att_context_size,
        max_cache_len=0,
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        use_bias=True,
    ):
        """Construct an RelPositionMultiHeadAttentionLongformer object."""
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            dropout_rate=dropout_rate,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            max_cache_len=max_cache_len,
            use_bias=use_bias,
        )
        self.att_context_size = att_context_size
        self.global_tokens = global_tokens
        self.global_tokens_spacing = global_tokens_spacing
        self.global_attn_separate = global_attn_separate

        if self.global_attn_separate:
            self.global_q = nn.Linear(n_feat, n_feat, bias=use_bias)
            self.global_k = nn.Linear(n_feat, n_feat, bias=use_bias)
            self.global_v = nn.Linear(n_feat, n_feat, bias=use_bias)

    def forward(self, query, key, value, pad_mask, pos_emb, cache=None):
        """Compute Scaled Dot Product Local Attention with rel. positional encoding. using overlapping chunks
        Args:
            query (torch.Tensor): (batch, time, size)
            key (torch.Tensor): (batch, time, size)
            value(torch.Tensor): (batch, time, size)
            pad_mask (torch.Tensor): (batch, time)
            pos_emb (torch.Tensor) : (batch, 2w + 1, size)
            cache (torch.Tensor) : (batch, time_cache, size)
        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """

        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            n_batch, _, T, _ = q.size()

            w = max(self.att_context_size[0], self.att_context_size[1])
            if w <= 0:
                raise ValueError("When using local attention, context size must be set > 0")
            pad_len = (2 * w - T % (2 * w)) % (2 * w)  # pad time to 2w
            q = F.pad(q, (0, 0, 0, pad_len))  # (batch, head, time, size)
            k = F.pad(k, (0, 0, 0, pad_len))  # (batch, head, time, size)
            v = F.pad(v, (0, 0, 0, pad_len))  # (batch, head, time, size)
            mask = F.pad(pad_mask, (0, pad_len), value=1.0)

            q_with_bias_u = q + self.pos_bias_u.unsqueeze(1)  # (batch, head, time, size)
            q_with_bias_v = q + self.pos_bias_v.unsqueeze(1)  # (batch, head, time, size)

            diagonal_matrix_ac = self.sliding_chunks_matmul_qk(
                q_with_bias_u, k, w, padding_value=0.0
            )  # (batch, head, time, 2w + 1)

            # add relative positional embedding

            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k).transpose(1, 2)
            # (batch, head, 2w, size)
            diagonal_matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            # (batch, head, time, 2w + 1)

            start_pos = w - self.att_context_size[0]
            end_pos = w + self.att_context_size[1]

            diagonal_matrix_ac[:, :, :, : self.att_context_size[0]] += diagonal_matrix_bd[
                :, :, :, : self.att_context_size[0]
            ]
            diagonal_matrix_ac[:, :, :, -(self.att_context_size[1] + 1) :] += diagonal_matrix_bd[
                :, :, :, self.att_context_size[0] :
            ]
            scores = diagonal_matrix_ac / self.s_d_k
            # (batch, head, time, 2w + 1)

            # mask invalid positions
            scores[:, :, :, :start_pos] = -10000.0
            scores[:, :, :, end_pos + 1 :] = -10000.0

            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x num_heads x seqlen x hidden_size)
            mask = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = mask.type_as(scores).masked_fill(mask, -10000.0)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            d_mask = self.sliding_chunks_matmul_qk(ones, float_mask, w, padding_value=0.0)
            # (batch, head, time, 2w + 1)

            scores += d_mask

            if self.global_tokens > 0:

                # create q, k, v for global attn
                if self.global_attn_separate:
                    global_q = self.global_q(query).view(n_batch, -1, self.h, self.d_k)
                    global_k = self.global_k(key).view(n_batch, -1, self.h, self.d_k)
                    global_v = self.global_v(value).view(n_batch, -1, self.h, self.d_k)
                    global_q = global_q.transpose(1, 2)
                    global_k = global_k.transpose(1, 2)
                    global_v = global_v.transpose(1, 2)
                    global_q = F.pad(global_q, (0, 0, 0, pad_len))  # (batch, head, time, size)
                    global_k = F.pad(global_k, (0, 0, 0, pad_len))  # (batch, head, time, size)
                    global_v = F.pad(global_v, (0, 0, 0, pad_len))  # (batch, head, time, size)
                else:
                    global_q, global_k, global_v = q, k, v

                global_q /= self.s_d_k

                # assign which tokens are global
                is_index_global_attn = torch.zeros_like(pad_mask)
                is_index_global_attn[
                    :, : self.global_tokens * self.global_tokens_spacing : self.global_tokens_spacing
                ] = 1.0

                # compute global attn indices
                (
                    max_num_global_attn_indices,
                    is_index_global_attn_nonzero,
                    is_local_index_global_attn_nonzero,
                    is_local_index_no_global_attn_nonzero,
                ) = self._get_global_attn_indices(is_index_global_attn=is_index_global_attn)

                # calculate global attn probs with global keys
                # (batch, time, head, max_num_global_attn_indices)
                global_key_attn = self._compute_global_key_attn(
                    query=global_q.transpose(1, 2),
                    key=global_k.transpose(1, 2),
                    max_num_global_attn_indices=max_num_global_attn_indices,
                    is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                    is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                    is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                ).transpose(1, 2)

                # concat to local_attn_probs
                # (batch, time, head, max_num_global_attn_indices + 2*w)
                scores = torch.cat((global_key_attn, scores), dim=-1)

                # free memory
                del global_key_attn

            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
            p_attn = self.dropout(attn)
            # (batch, head, time, 2w + 1)

            if self.global_tokens > 0:
                # compute sum of global and local attn
                out = self._compute_attn_output_with_global_indices(
                    value=v,
                    attn_probs=p_attn,
                    max_num_global_attn_indices=max_num_global_attn_indices,
                    is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                    is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                    w=w,
                )
            else:
                # compute local attn only
                out = self.sliding_chunks_matmul_pv(p_attn, v, w)

            out = out.reshape(n_batch, -1, self.h * self.d_k)[:, :T]

            if self.global_tokens > 0:
                out_global_to_all = self._compute_out_global_to_all(
                    query=global_q,
                    key=global_k,
                    value=global_v,
                    max_num_global_attn_indices=max_num_global_attn_indices,
                    is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                    is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                    is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                    is_index_masked=mask,
                )

                # overwrite values with global attention
                out[is_index_global_attn_nonzero] = out_global_to_all

        ret = self.linear_out(out)

        if cache is None:
            return ret
        else:
            return ret, cache

    def _get_global_attn_indices(self, is_index_global_attn: torch.Tensor) -> Tuple:
        """
        Compute global attention indices.

        Args:
            is_index_global_attn (torch.Tensor): (batch, time) A boolean tensor indicating if an index is a global attention index.

        Returns:
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_global_attn_nonzero (tuple): Indices of non-padding values within global attention indices.
            is_local_index_no_global_attn_nonzero (tuple): Indices of padding values within global attention indices.
        """
        # Calculate the number of global attention indices in the batch
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # Find the maximum number of global attention indices in the batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # Get the indices of global attention (non-zero elements)
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

        # Create a helper tensor to find the local indices of global attention
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # Find the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        # Find the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)

        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _compute_global_key_attn(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        max_num_global_attn_indices: int,
        is_index_global_attn_nonzero: tuple,
        is_local_index_global_attn_nonzero: tuple,
        is_local_index_no_global_attn_nonzero: tuple,
    ) -> torch.Tensor:
        """
        Compute the attention probabilities using only global key vectors.

        Args:
            key (torch.Tensor): (batch, time, head, head_dim) The key vectors.
            query (torch.Tensor): (batch, time, head, head_dim) The query vectors.
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_global_attn_nonzero (tuple): Non-padding values within global attention indices.
            is_local_index_no_global_attn_nonzero (tuple): Padding values within global attention indices.

        Returns:
            attn_probs_from_global_key (torch.Tensor): (batch, time, head, max_num_global_attn_indices) The computed attention probabilities using only global key vectors.
        """
        batch_size = key.shape[0]

        # create only global key vectors
        key_only_global = key.new_zeros(batch_size, max_num_global_attn_indices, self.h, self.d_k)

        key_only_global[is_local_index_global_attn_nonzero] = key[is_index_global_attn_nonzero]

        # (batch_size, seq_len, head, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query, key_only_global))

        # need to transpose since ONNX export only supports consecutive indexing: https://pytorch.org/docs/stable/onnx.html#writes-sets
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(attn_probs_from_global_key.dtype).min
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value: torch.Tensor,
        attn_probs: torch.Tensor,
        max_num_global_attn_indices: int,
        is_index_global_attn_nonzero: tuple,
        is_local_index_global_attn_nonzero: tuple,
        w: int,
    ) -> torch.Tensor:
        """
        Compute the attention output with global indices.

        Args:
            value (torch.Tensor): (batch, head, time, head_dim) The value vectors for global attention.
            attn_probs (torch.Tensor): (batch, time, head, 2w) The attention probabilities.
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_global_attn_nonzero (tuple): Non-padding values within global attention indices.
            w (int): Local context size
        Returns:
            torch.Tensor: (batch, time, head x head_dim) The attention output of all tokens attending to global.
        """
        batch_size, time = attn_probs.shape[0], attn_probs.shape[2]

        value = value.transpose(1, 2)

        # get value vectors for global only
        value_vectors_only_global = value.new_zeros(batch_size, max_num_global_attn_indices, self.h, self.d_k)
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value[is_index_global_attn_nonzero]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        # compute attn output only global
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.clone(), value_vectors_only_global.transpose(1, 2).clone()
        ).transpose(1, 2)

        # reshape attn probs
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()

        # compute attn output with global
        attn_output_without_global = self.sliding_chunks_matmul_pv(attn_probs_without_global, value.transpose(1, 2), w)

        return attn_output_only_global + attn_output_without_global

    def _compute_out_global_to_all(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        max_num_global_attn_indices: int,
        is_local_index_global_attn_nonzero: tuple,
        is_index_global_attn_nonzero: tuple,
        is_local_index_no_global_attn_nonzero: tuple,
        is_index_masked: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the attention output of global tokens attending to all.

        Args:
            query (torch.Tensor): (batch, head, time, head_dim) The queries for global attention.
            key (torch.Tensor): (batch, head, time, head_dim) The keys for global attention.
            value (torch.Tensor): (batch, head, time, head_dim) The values for global attention.
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_local_index_global_attn_nonzero (tuple): Non-padding values within global attention indices.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_no_global_attn_nonzero (tuple): Padding values within global attention indices.
            is_index_masked (torch.Tensor): (batch, time) A boolean tensor indicating if an index is masked.

        Returns:
            global_attn_output (torch.Tensor): (batch, max_num_global_attn_indices, head x head_dim)
            The attention output of global tokens attending to all.
        """

        batch_size = key.shape[0]
        seq_len = key.shape[2]

        global_k = key.reshape(batch_size * self.h, -1, self.d_k)
        global_v = value.reshape(batch_size * self.h, -1, self.d_k)

        global_q = query.transpose(1, 2)
        global_q_from_global = global_q.new_zeros(batch_size, max_num_global_attn_indices, self.h, self.d_k)
        global_q_from_global[is_local_index_global_attn_nonzero] = global_q[is_index_global_attn_nonzero]
        global_q_from_global = global_q_from_global.transpose(0, 1).reshape(batch_size * self.h, -1, self.d_k)

        # compute attn scores
        global_attn_scores = torch.bmm(global_q_from_global, global_k.transpose(1, 2))
        global_attn_scores = global_attn_scores.view(batch_size, self.h, max_num_global_attn_indices, seq_len)

        # need to transpose since ONNX export only supports consecutive indexing: https://pytorch.org/docs/stable/onnx.html#writes-sets
        global_attn_scores = global_attn_scores.transpose(1, 2)
        global_attn_scores[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(global_attn_scores.dtype).min
        global_attn_scores = global_attn_scores.transpose(1, 2)

        global_attn_scores = global_attn_scores.masked_fill(
            is_index_masked.transpose(2, 3),
            torch.finfo(global_attn_scores.dtype).min,
        )

        global_attn_scores = global_attn_scores.view(batch_size * self.h, max_num_global_attn_indices, seq_len)

        # compute global attn probs
        if self.training:
            global_attn_probs_float = nn.functional.softmax(global_attn_scores, dim=-1, dtype=torch.float32)
        else:
            global_attn_probs_float = nn.functional.softmax(global_attn_scores, dim=-1)

        global_attn_probs = self.dropout(global_attn_probs_float)

        # global attn output
        global_attn_output = torch.bmm(global_attn_probs, global_v)
        global_attn_output = global_attn_output.view(batch_size, self.h, max_num_global_attn_indices, self.d_k)

        global_attn_output = global_attn_output[
            is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
        ]

        global_attn_output = global_attn_output.reshape(global_attn_output.shape[0], -1)

        return global_attn_output

    # Longformer implementation for overlap case
    #
    def _skew(self, x: torch.Tensor, direction: List[int], padding_value: float) -> torch.Tensor:
        """Convert diagonals into columns (or columns into diagonals depending on `direction`

        Args:
            x (torch.Tensor): (batch x head, chunk_count, 2w, 2w)
            direction (List[int]): padding directions
            padding_value (float): value to pad with

        Returns:
            output (torch.Tensor): (batch x head, chunk_count, 2w, 2w + 1)

        """
        x_padded = F.pad(x, direction, value=padding_value)
        x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
        return x_padded

    def _skew2(self, x: torch.Tensor, padding_value: float) -> torch.Tensor:
        """Shift every row 1 step to right converting columns into diagonals

        Args:
            x (torch.Tensor): (batch x head, chunks_count + 1, w, 2w + 1)
            padding_value (float): value to pad with

        Returns:
            output (torch.Tensor): (batch x head, chunks_count + 1, w, 3w)
        """
        # X = B x C x M x L
        B, C, M, L = x.size()
        x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
        x = x.view(B, C, -1)  # B x C x ML+MM+M
        x = x[:, :, :-M]  # B x C x ML+MM
        x = x.view(B, C, M, M + L)  # B x C, M x L+M
        x = x[:, :, :, :-1]
        return x

    def _chunk_overlap(self, x: torch.Tensor, w: int) -> torch.Tensor:
        """Convert into overlapping chunks.

        Args:
            x (torch.Tensor): # (batch x head, time, size)
            w (int): Chunk overlap size

        Returns:
            output (torch.Tensor): # (batch x head, chunk_count, 2w, size)
        """

        # non-overlapping chunks of size = 2w
        x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

        # use `as_strided` to make the chunks overlap with an overlap size = w
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @lru_cache()
    def _get_invalid_locations_mask(self, w: int, device: str):

        diagonals_list = []
        for j in range(-w, 1):
            diagonal_mask = torch.zeros(w, device='cpu', dtype=torch.uint8)
            diagonal_mask[:-j] = 1
            diagonals_list.append(diagonal_mask)

        mask = torch.stack(diagonals_list, dim=-1)
        mask = mask[None, None, :, :]

        ending_mask = mask.flip(dims=(2, 3)).bool().to(device)
        return mask.bool().to(device), ending_mask

    def mask_invalid_locations(
        self,
        input_tensor: torch.Tensor,
        w: int,
    ):
        """
        Mask locations invalid for the sliding window attention

        Args:
            input_tensor (torch.Tensor): # (batch x head, time, size)
            w (int): Chunk overlap size
        """
        beginning_mask, ending_mask = self._get_invalid_locations_mask(w, input_tensor.device)
        seq_len = input_tensor.size(2)
        beginning_input = input_tensor[:, :, :w, : w + 1]
        beginning_mask = beginning_mask[:, :, :seq_len].expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask, -float('inf'))

        ending_input = input_tensor[:, :, -w:, -(w + 1) :]
        ending_mask = ending_mask[:, :, -seq_len:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask, -float('inf'))

    def sliding_chunks_matmul_qk(self, q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float) -> torch.Tensor:
        """Matrix multiplication of query x key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w
        with an overlap of size w

        Args:
            q (torch.Tensor): (batch, head, time, size)
            k (torch.Tensor): (batch, head, time, size)
            w (int): Chunk overlap size
            padding_value (float): Value to pad with

        Returns:
            output (torch.Tensor): (batch, head, time, 2w + 1)
        """
        bsz, num_heads, seqlen, head_dim = q.size()
        assert seqlen % (w * 2) == 0
        assert q.size() == k.size()

        chunks_count = seqlen // w - 1

        # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
        q = q.reshape(bsz * num_heads, seqlen, head_dim)
        k = k.reshape(bsz * num_heads, seqlen, head_dim)

        chunk_q = self._chunk_overlap(q, w)  # (batch x head, chunk_count, 2w, size)
        chunk_k = self._chunk_overlap(k, w)  # (batch x head, chunk_count, 2w, size)

        # matrix multipication
        # bcxd: bsz*num_heads x chunks x 2w x head_dim
        # bcyd: bsz*num_heads x chunks x 2w x head_dim
        # bcxy: bsz*num_heads x chunks x 2w x 2w
        chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply
        # (batch x head, chunk_count, 2w, 2w)

        # convert diagonals into columns
        diagonal_chunk_attn = self._skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)
        # (batch x head, chunk_count, 2w, 2w + 1)

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
        # w previous words). The following column is attention score from each word to itself, then
        # followed by w columns for the upper triangle.

        diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))
        # (batch x head, chunk_count + 1, w, 2w + 1)

        # copy parts from diagonal_chunk_attn into the compined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, : w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, : w + 1]
        # - copying the lower triangle
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, -(w + 1) : -1, w + 1 :]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, : w - 1, 1 - w :]

        # separate bsz and num_heads dimensions again
        diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1)
        # (batch, head, time, 2w + 1)

        self.mask_invalid_locations(diagonal_attn, w)

        return diagonal_attn

    def sliding_chunks_matmul_pv(self, prob: torch.Tensor, v: torch.Tensor, w: int):
        """Same as sliding_chunks_matmul_qk but for prob and value tensors.

        Args:
            prob (torch.Tensor): (batch, head, time, size)
            v (torch.Tensor): (batch, head, time, size)
            w (int): Chunk overlap size

        Returns:
            output (torch.Tensor): (batch, time, head, size)
        """
        bsz, num_heads, seqlen, head_dim = v.size()
        chunks_count = seqlen // w - 1
        # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
        chunk_prob = prob.reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)
        # (batch x head, chunks_count + 1, w, 2w + 1)

        # group bsz and num_heads dimensions into one
        v = v.reshape(bsz * num_heads, seqlen, head_dim)
        # (batch x head, time, size)

        # pad seqlen with w at the beginning of the sequence and another w at the end
        padded_v = F.pad(v, (0, 0, w, w), value=-1)
        # (batch x head, time + 2w, size)

        # chunk padded_v into chunks of size 3w and an overlap of size w
        chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)
        # (batch x head, chunks_count + 1, 3w, size)

        skewed_prob = self._skew2(chunk_prob, padding_value=0)
        # (batch x head, chunks_count + 1, w, 3w)

        context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
        # (batch x head, chunks_count + 1, w, size)

        return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)


class PositionalEncoding(torch.nn.Module):
    """Fixed sinusoidal positional encoding.
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=None, dropout_rate_emb=0.0):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        if dropout_rate_emb > 0:
            self.dropout_emb = nn.Dropout(dropout_rate_emb)
        else:
            self.dropout_emb = None

    def create_pe(self, positions, dtype):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        if hasattr(self, 'pe'):
            self.pe = pe
        else:
            self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings if needed."""
        if hasattr(self, 'pe') and self.pe.size(1) >= length:
            return
        positions = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x: torch.Tensor, cache_len=0):
        """Adds positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x+pos_emb (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """
        input_len = x.size(1) + cache_len
        if self.xscale:
            x = x * self.xscale
        pos_emb = self.pe[:, :input_len]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        x = x + pos_emb
        return self.dropout(x), pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for TransformerXL's layers
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings if needed."""
        needed_size = 2 * length - 1
        if hasattr(self, 'pe') and self.pe.size(1) >= needed_size:
            return
        # positions would be from negative numbers to positive
        # positive positions would be used for left positions and negative for right positions
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


class LocalAttRelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for sliding window attention or chunked attention.
    See above for relative positional encoding based on Transformer-XL paper
    Args:
        left_chunk_size (int): number of frames to in past chunks
        chunk size (int): number of frames (max frames if using multimode) in current chunk
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def __init__(self, att_context_size, **kwargs):
        super(LocalAttRelPositionalEncoding, self).__init__(**kwargs)
        self.left_context = att_context_size[0]
        self.right_context = att_context_size[1]

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings only at the beginning"""
        if hasattr(self, 'pe'):
            return

        positions = torch.arange(
            self.left_context, -self.right_context - 1, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        end_pos = self.left_context + self.right_context + 1
        pos_emb = self.pe[:, :end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb
