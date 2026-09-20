# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Inference-only encoder for EasyMagpie reference-speaker conditioning."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _ConvolutionLayer(nn.Module):
    """Name-compatible subset of NeMo's non-causal ``ConvolutionLayer``."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, signal: torch.Tensor, signal_mask: torch.Tensor) -> torch.Tensor:
        mask = signal_mask.unsqueeze(1).to(signal.dtype)
        return self.conv(signal * mask) * mask


class _PositionwiseConvFF(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, kernel_size: int) -> None:
        super().__init__()
        self.proj = _ConvolutionLayer(d_model, d_ffn, kernel_size)
        self.o_net = _ConvolutionLayer(d_ffn, d_model, kernel_size)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(inputs.transpose(1, 2), mask)
        hidden = F.gelu(hidden, approximate="tanh")
        return self.o_net(hidden, mask).transpose(1, 2)


class _SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"speaker encoder width {d_model} must be divisible by {n_heads} heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5
        self.o_net = nn.Linear(d_model, d_model, bias=False)
        self.qkv_net = nn.Linear(d_model, 3 * d_model, bias=False)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, time, _ = inputs.shape
        qkv = self.qkv_net(inputs).reshape(batch, time, 3, self.n_heads, self.d_head)
        query, key, value = (part.squeeze(2).transpose(1, 2) for part in qkv.chunk(3, dim=2))
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
        attention_mask = mask[:, None, :, None] & mask[:, None, None, :]
        scores.masked_fill_(~attention_mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).masked_fill(~attention_mask, 0.0)
        output = torch.matmul(probabilities, value).transpose(1, 2).contiguous().view(batch, time, -1)
        return self.o_net(output)


class _SpeakerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, n_heads: int, kernel_size: int) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model, bias=False)
        self.self_attention = _SelfAttention(d_model, n_heads)
        self.norm_pos_ff = nn.LayerNorm(d_model, bias=False)
        self.pos_ff = _PositionwiseConvFF(d_model, d_ffn, kernel_size)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.unsqueeze(-1).to(inputs.dtype)
        hidden = inputs * mask_float
        hidden = hidden + self.self_attention(self.norm_self(hidden), mask)
        hidden = hidden + self.pos_ff(self.norm_pos_ff(hidden), mask)
        return hidden * mask_float


class EasyMagpieReferenceSpeakerEncoder(nn.Module):
    """Transform reference-audio code embeddings into speaker conditioning."""

    def __init__(
        self,
        *,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        kernel_size: int,
        max_length: int,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"speaker encoder kernel_size must be a positive odd integer, got {kernel_size}")
        self.layers = nn.ModuleList(
            [_SpeakerEncoderLayer(d_model, d_ffn, n_heads, kernel_size) for _ in range(n_layers)]
        )
        self.position_embeddings = nn.Embedding(max_length, d_model)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode padded ``[B, T, D]`` context-code embeddings."""
        if inputs.ndim != 3 or lengths.ndim != 1 or inputs.shape[0] != lengths.numel():
            raise ValueError("speaker inputs must be [batch, time, dim] with one length per item")
        if inputs.shape[1] > self.position_embeddings.num_embeddings:
            raise ValueError(
                f"speaker input length {inputs.shape[1]} exceeds configured maximum "
                f"{self.position_embeddings.num_embeddings}"
            )
        positions = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0)
        hidden = inputs + self.position_embeddings(positions)
        mask = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0) < lengths.unsqueeze(1)
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return hidden


__all__ = ["EasyMagpieReferenceSpeakerEncoder"]
