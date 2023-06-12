# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.nlp.modules.common.megatron.position_embedding.alibi_relative_position_embedding import (
    build_relative_position,
)
from nemo.utils.decorators import experimental

__all__ = ['SandwichRelativePositionEmbedding']


@experimental
class SandwichRelativePositionEmbedding(torch.nn.Module):
    """
    Dissecting Transformer Length Extrapolation via the Lens of Receptive Field Analysis
    Based on https://arxiv.org/abs/2212.10356
    """

    def __init__(
        self, bidirectional, num_attention_heads, layer_type, hidden_size, max_seq_len=512,
    ):
        """
        Args:
            num_attention_heads: Number of attention heads
            hidden_size: Hidden size per attention head
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.layer_type = layer_type
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.relative_position = build_relative_position(max_seq_len, full=True)

    def forward(self, query_seq_length, key_seq_length):
        # used cached relative position if possible
        max_seq_len = max(query_seq_length, key_seq_length)
        if max_seq_len > self.max_seq_len:
            relative_position = build_relative_position(max_seq_len, full=True)
        else:
            relative_position = self.relative_position

        # shape (query_seq_length, key_seq_length)
        relative_position = relative_position[-query_seq_length:, -key_seq_length:]
        # if not bidirectional, mask out the future positions
        if not self.bidirectional:
            relative_position = torch.tril(relative_position)

        inv_freq = 1.0 / (
            10000
            ** (2 * torch.arange(1, self.hidden_size / 2 + 1, device=relative_position.device) / self.hidden_size)
        )

        _bias = torch.sum((relative_position[:, :, None].repeat(1, 1, len(inv_freq)) * inv_freq).cos(), axis=2)
        bias = _bias.repeat(self.num_attention_heads, 1, 1)

        _bias_scales = torch.arange(1, self.num_attention_heads + 1, 1, device=relative_position.device)
        bias_scales = _bias_scales[:, None, None]

        scaled_bias = (bias - self.hidden_size / 2) / (bias_scales * 8 / self.num_attention_heads).unsqueeze(0)

        return scaled_bias
