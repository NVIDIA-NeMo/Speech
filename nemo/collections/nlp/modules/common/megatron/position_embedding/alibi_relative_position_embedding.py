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

__all__ = ['ALiBiRelativePositionEmbedding']


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        slopes = get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        slopes = (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

    return slopes


def build_slopes(num_attention_heads, num_attention_heads_alibi):
    """
    Builds a slopes tensor.
    """
    slopes = (
        torch.Tensor(get_slopes(num_attention_heads_alibi) + [0] * (num_attention_heads - num_attention_heads_alibi))
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

    if torch.cuda.is_available():
        slopes = slopes.to(torch.cuda.current_device())

    return slopes


def build_relative_position(max_seq_len, full=True):
    """
    full=True:  shape (max_seq_len, max_seq_len)
    full=False: shape (max_seq_len)
    """
    relative_position = torch.arange(1 - max_seq_len, 1)[None, :].mul(-1)  # (1, max_seq_len)

    if full:
        memory_position = torch.arange(1 - max_seq_len, 1)[:, None].mul(-1)
        relative_position = torch.abs(memory_position - relative_position)  # (max_seq_len, max_seq_len)

    if torch.cuda.is_available():
        relative_position = relative_position.to(torch.cuda.current_device())

    return relative_position


class ALiBiRelativePositionEmbedding(torch.nn.Module):
    """
    ALiBi (Attention with Linear Biases) relative position embedding for auto-regressive decoder
    and joint encoder (symmetric for forward and backward distance).
    Based on https://arxiv.org/bas/2108.12409
    """

    def __init__(
        self, bidirectional, num_attention_heads, layer_type, num_attention_heads_alibi=None, max_seq_len=512,
    ):
        """
        Args:
            bidirectional: Whether to use bidirectional relative position embedding
            num_attention_heads: Number of attention heads
            layer_type: Layer type. Can be one of [LayerType.encoder or LayerType.decoder]. Willdetermine the bias construction
            num_attention_heads_alibi: Number of attention heads for which alibi bias will be used
            max_seq_len: Maximum sequence length for precomputed relative positions. Larger sizes will result in more memory usage by computing alibi mask on-the-fly.
        """
        super().__init__()

        if (num_attention_heads_alibi is None) or (num_attention_heads_alibi <= 0):
            num_attention_heads_alibi = num_attention_heads

        if num_attention_heads_alibi > num_attention_heads:
            raise ValueError(
                f"num_attention_heads_alibi ({num_attention_heads_alibi}) cannot be larger than num_attention_heads ({num_attention_heads})"
            )

        self.bidirectional = bidirectional
        self.num_attention_heads = num_attention_heads
        # LayerType.encoder or LayerType.decoder. Is only needed to determine the group for the all_reduce
        self.layer_type = layer_type
        # define the size of pre-computed relative position slopes.
        # define the number of attention heads for which alibi mask will be pre-computed (the rest are disabled).
        self.num_attention_heads_alibi = num_attention_heads_alibi
        # Larger sizes will result in more memory usage by computing alibi mask on-the-fly.
        self.max_seq_len = max_seq_len

        # cache the slopes
        self.slopes = build_slopes(num_attention_heads, num_attention_heads_alibi)
        # cache the relative position bias. shape (num_attention_heads, max_seq_len, max_seq_len)
        # if we use causal attention (not bidrectional), we can use singleton relative position
        self.relative_position = (
            build_relative_position(max_seq_len, full=bidirectional).unsqueeze(0).expand(num_attention_heads, -1, -1)
        )

    def forward(self, query_seq_length, key_seq_length):
        # used cached relative position if possible
        max_seq_len = max(query_seq_length, key_seq_length)
        if max_seq_len > self.max_seq_len:
            relative_position = (
                build_relative_position(max_seq_len, full=self.bidirectional)
                .unsqueeze(0)
                .expand(self.num_attention_heads, -1, -1)
            )
        else:
            relative_position = self.relative_position
        # shape (num_attention_heads, query_seq_length, key_seq_length)
        relative_position = relative_position[:, -query_seq_length:, -key_seq_length:]
        # if not bidirectional, mask out the future positions

        # shape (1, num_heads, query_length, key_length)
        return -relative_position.unsqueeze(0) * self.slopes
