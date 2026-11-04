# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small bidirectional Transformer with the learned relative attention of VITS.

The relative key/value attention and index conversions adapt the MIT-licensed
VITS encoder: https://github.com/jaywalnut310/vits/blob/main/attentions.py.
NeMo's masked convolutions implement the reference's convolutional feed-forward
network. Relative offsets outside the learned window contribute zero; content
attention still covers the complete sequence.
"""

# MIT License
# Copyright (c) 2021 Jaehyeon Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.tts.modules.ffn_modules import ConvolutionLayer


class RelativePositionTransformer(nn.Module):
    """VITS-style post-normalized attention and masked convolutional FFN blocks.

    Inputs are ``[B,C,T]`` with a multiplicative ``[B,1,T]`` validity mask.
    ``window_size`` controls learned relative key/value offsets, rather than
    restricting global bidirectional content attention. The relative tables are
    shared across heads. This module adds no absolute positional embeddings.
    """

    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=3,
        p_dropout=0.1,
        window_size=4,
    ):
        super().__init__()
        if type(hidden_channels) is not int or hidden_channels < 1:
            raise ValueError('hidden_channels must be a positive integer')
        if type(filter_channels) is not int or filter_channels < 1:
            raise ValueError('filter_channels must be a positive integer')
        if type(n_heads) is not int or n_heads < 1 or hidden_channels % n_heads:
            raise ValueError('n_heads must be a positive integer dividing hidden_channels')
        if type(n_layers) is not int or n_layers < 0:
            raise ValueError('n_layers must be a nonnegative integer')
        if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be an odd positive integer')
        if type(window_size) is not int or window_size < 0:
            raise ValueError('window_size must be a nonnegative integer')
        if (
            isinstance(p_dropout, bool)
            or not isinstance(p_dropout, (int, float))
            or not math.isfinite(p_dropout)
            or not 0 <= p_dropout < 1
        ):
            raise ValueError('p_dropout must be finite and in [0, 1)')
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList(
            [_RelativePositionSelfAttention(hidden_channels, n_heads, p_dropout, window_size) for _ in range(n_layers)]
        )
        self.norm_layers_1 = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList(
            [_MaskedConvolutionFFN(hidden_channels, filter_channels, kernel_size, p_dropout) for _ in range(n_layers)]
        )
        self.norm_layers_2 = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(n_layers)])

    def forward(self, inputs, mask):
        if inputs.ndim != 3 or inputs.size(1) != self.hidden_channels:
            raise ValueError('inputs must have shape [batch, hidden_channels, time]')
        if mask.shape != (inputs.size(0), 1, inputs.size(2)):
            raise ValueError('mask must have shape [batch, 1, time]')
        mask = mask.to(device=inputs.device, dtype=inputs.dtype)
        if inputs.size(0) == 0 or inputs.size(2) == 0:
            anchor = sum((parameter.sum() * 0.0 for parameter in self.parameters()), inputs.new_zeros(()))
            return inputs * mask + anchor
        hidden = inputs * mask
        for attention, norm_1, ffn, norm_2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        ):
            residual = self.drop(attention(hidden, mask))
            hidden = norm_1((hidden + residual).transpose(1, 2)).transpose(1, 2) * mask
            residual = self.drop(ffn(hidden, mask))
            hidden = norm_2((hidden + residual).transpose(1, 2)).transpose(1, 2) * mask
        return hidden * mask


class _RelativePositionSelfAttention(nn.Module):
    """Content attention plus shared learned relative keys and values."""

    def __init__(self, channels, n_heads, p_dropout, window_size):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.conv_q = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_k = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_o = nn.Conv1d(channels, channels, kernel_size=1)
        self.drop = nn.Dropout(p_dropout)
        relative_shape = (1, 2 * window_size + 1, self.k_channels)
        self.emb_rel_k = nn.Parameter(torch.randn(relative_shape) * self.k_channels**-0.5)
        self.emb_rel_v = nn.Parameter(torch.randn(relative_shape) * self.k_channels**-0.5)
        for projection in (self.conv_q, self.conv_k, self.conv_v):
            nn.init.xavier_uniform_(projection.weight)

    def forward(self, inputs, mask):
        batch, channels, steps = inputs.shape
        inputs = inputs * mask
        query, key, value = (
            projection(inputs).reshape(batch, self.n_heads, self.k_channels, steps).transpose(2, 3)
            for projection in (self.conv_q, self.conv_k, self.conv_v)
        )
        query = query / math.sqrt(self.k_channels)
        scores = torch.matmul(query, key.transpose(2, 3))
        relative_keys = self._get_relative_embeddings(self.emb_rel_k, steps)
        relative_logits = torch.matmul(query, relative_keys.unsqueeze(0).transpose(2, 3))
        scores = scores + self._relative_position_to_absolute_position(relative_logits)
        valid = mask.bool()
        attention_mask = valid.unsqueeze(2) & valid.unsqueeze(3)
        # Finite masking followed by explicit zeros keeps both forward and
        # backward defined for fully masked query rows and invalid-only turns.
        scores = scores.masked_fill(~attention_mask, -1e4)
        probabilities = F.softmax(scores, dim=-1).masked_fill(~attention_mask, 0.0)
        probabilities = self.drop(probabilities)
        output = torch.matmul(probabilities, value)
        relative_probabilities = self._absolute_position_to_relative_position(probabilities)
        relative_values = self._get_relative_embeddings(self.emb_rel_v, steps)
        output = output + torch.matmul(relative_probabilities, relative_values.unsqueeze(0))
        output = output.transpose(2, 3).reshape(batch, channels, steps)
        return self.conv_o(output) * mask

    def _get_relative_embeddings(self, embeddings, length):
        padding = max(length - self.window_size - 1, 0)
        start = max(self.window_size + 1 - length, 0)
        padded = F.pad(embeddings, (0, 0, padding, padding))
        return padded[:, start : start + 2 * length - 1]

    @staticmethod
    def _relative_position_to_absolute_position(inputs):
        batch, heads, length, _ = inputs.shape
        padded = F.pad(inputs, (0, 1)).reshape(batch, heads, length * 2 * length)
        padded = F.pad(padded, (0, length - 1))
        return padded.reshape(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1 :]

    @staticmethod
    def _absolute_position_to_relative_position(inputs):
        batch, heads, length, _ = inputs.shape
        padded = F.pad(inputs, (0, length - 1)).reshape(batch, heads, length * (2 * length - 1))
        padded = F.pad(padded, (length, 0))
        return padded.reshape(batch, heads, length, 2 * length)[:, :, :, 1:]


class _MaskedConvolutionFFN(nn.Module):
    """VITS FFN: masked convolution, ReLU, dropout, masked convolution."""

    def __init__(self, channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.conv_1 = ConvolutionLayer(channels, filter_channels, kernel_size=kernel_size, is_causal=False)
        self.conv_2 = ConvolutionLayer(filter_channels, channels, kernel_size=kernel_size, is_causal=False)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, inputs, mask):
        valid = mask[:, 0]
        hidden = self.drop(torch.relu(self.conv_1(inputs, valid)))
        return self.conv_2(hidden, valid) * mask
