# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small learnable speaker conditioning from frozen codec reference features.

The attentive statistics pooling follows the weighted mean/standard-deviation
construction used by ``audio_codec_modules.ResNetSpeakerEncoder``. Masked NeMo
convolutions replace its waveform/mel frontend and large ResNet. No external
speaker model, batch statistics, or transcript features are required.
"""

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.tts.modules.ffn_modules import ConvolutionLayer
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths


class CodecSpeakerEncoder(nn.Module):
    """Encode ``[B,R,C]`` reference features into one normalized vector per row.

    Inputs are decoded codec latents, not waveform or integer codec IDs. The
    caller freezes/detaches those observations; the encoder itself is trained
    by the alignment likelihood. Missing/zero-length references return exact
    zeros while retaining zero-gradient connections to every parameter.
    """

    def __init__(self, in_channels, hidden_channels=64, embedding_dim=64, num_layers=2):
        super().__init__()
        for name, value in (
            ('in_channels', in_channels),
            ('hidden_channels', hidden_channels),
            ('embedding_dim', embedding_dim),
            ('num_layers', num_layers),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.input_projection = ConvolutionLayer(in_channels, hidden_channels, kernel_size=1)
        self.layers = nn.ModuleList(
            [ConvolutionLayer(hidden_channels, hidden_channels, kernel_size=3) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
        )
        self.output_projection = nn.Linear(2 * hidden_channels, embedding_dim)

    def forward(self, features, lengths):
        if features.ndim != 3 or features.size(2) != self.in_channels:
            raise ValueError('Speaker reference features must have shape [batch, time, in_channels]')
        if lengths.shape != (features.size(0),):
            raise ValueError('Speaker reference lengths must have shape [batch]')
        lengths = lengths.to(device=features.device, dtype=torch.long)
        batch, steps, _ = features.shape
        if batch == 0 or steps == 0:
            anchor = sum((parameter.sum() * 0.0 for parameter in self.parameters()), features.new_zeros(()))
            return features.new_zeros((batch, self.embedding_dim)) + anchor
        valid_rows = (lengths > 0) & (lengths <= steps)
        effective_lengths = torch.where(valid_rows, lengths, 0)
        mask = get_mask_from_lengths(effective_lengths, x=features.transpose(1, 2))
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden = self.input_projection(features.float().transpose(1, 2), mask)
            for layer, norm in zip(self.layers, self.norms):
                residual = torch.relu(layer(hidden, mask))
                hidden = norm((hidden + residual).transpose(1, 2)).transpose(1, 2) * mask.unsqueeze(1)
            logits = self.attention(hidden)
            weights = torch.softmax(logits.masked_fill(~mask.unsqueeze(1), torch.finfo(logits.dtype).min), dim=2)
            weights = weights.masked_fill(~mask.unsqueeze(1), 0.0)
            means = (hidden * weights).sum(dim=2)
            second_moments = (hidden.square() * weights).sum(dim=2)
            deviations = (second_moments - means.square()).clamp_min(1e-5).sqrt()
            pooled = torch.cat([means, deviations], dim=1)
            embedding = F.normalize(self.output_projection(pooled), p=2, dim=1)
            return embedding * valid_rows.unsqueeze(1)
