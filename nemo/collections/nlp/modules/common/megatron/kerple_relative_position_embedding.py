# coding=utf-8
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

import math

import torch

from nemo.collections.nlp.modules.common.megatron.alibi_relative_position_embedding import (
    build_relative_position,
    build_slopes,
)

__all__ = ['KERPLERelativePositionEmbedding']


class KERPLERelativePositionEmbedding(torch.nn.Module):
    """
    kerple (Attention with Linear Biases) relative position embedding for auto-regressive decoder
    and joint encoder (symmetric for forward and backward distance).
    Based on https://arxiv.org/bas/2108.12409
    """

    def __init__(
        self, bidirectional, num_attention_heads, layer_type, num_attention_heads_kerple=None, max_seq_len=512
    ):
        """
        Args:
            bidirectional: Whether to use bidirectional relative position embedding
            num_attention_heads: Number of attention heads
            layer_type: Layer type. Can be one of [LayerType.encoder or LayerType.decoder]. Willdetermine the bias construction
            num_attention_heads_kerple: Number of attention heads for which kerple bias will be used
            max_seq_len: Maximum sequence length for precomputed relative positions. Larger sizes will result in more memory usage by computing kerple mask on-the-fly.
        """
        super().__init__()

        if (num_attention_heads_kerple is None) or (num_attention_heads_kerple <= 0):
            num_attention_heads_kerple = num_attention_heads

        if num_attention_heads_kerple > num_attention_heads:
            raise ValueError(
                f"num_attention_heads_kerple ({num_attention_heads_kerple}) cannot be larger than num_attention_heads ({num_attention_heads})"
            )

        self.bidirectional = bidirectional
        self.num_attention_heads = num_attention_heads
        # LayerType.encoder or LayerType.decoder. Is only needed to determine the group for the all_reduce
        self.layer_type = layer_type
        # define the size of pre-computed relative position slopes.
        # define the number of attention heads for which kerple mask will be pre-computed (the rest are disabled).
        self.num_attention_heads_kerple = num_attention_heads_kerple
        # Larger sizes will result in more memory usage by computing kerple mask on-the-fly.
        self.max_seq_len = max_seq_len

        # initialize the slopes
        self.kerple_b = torch.nn.Parameter(build_slopes(num_attention_heads, num_attention_heads_kerple))
        self.kerple_a = torch.zeros_like(self.kerple_b)
        self.kerple_p = torch.ones_like(self.kerple_b)

        # cache the relative position bias. shape (num_attention_heads, max_seq_len, max_seq_len)
        self.relative_position = build_relative_position(max_seq_len, max_seq_len, num_attention_heads)

    def forward(self, query_seq_length, key_seq_length):
        # used cached relative position if possible
        max_seq_len = max(query_seq_length, key_seq_length)
        if max_seq_len > self.max_seq_len:
            relative_position = build_relative_position(max_seq_len, max_seq_len, self.num_attention_heads)
        else:
            relative_position = self.relative_position
        # shape (num_attention_heads, query_seq_length, key_seq_length)
        relative_position = relative_position[:, :query_seq_length, :key_seq_length]
        # if not bidirectional, mask out the future positions
        if not self.bidirectional:
            relative_position = torch.tril(relative_position)

        # shape (1, num_heads, query_length, key_length)
        return -self.kerple_b * torch.log(1 + self.kerple_a * relative_position.unsqueeze(0).pow(self.kerple_p))
