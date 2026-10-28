# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Small acoustic-likelihood aligner inspired by Glow-TTS (Kim et al., 2020).

This module learns the density of audio-derived features, rather than classifying
the transcript from text-conditioned decoder states. MAS selects a detached path;
the likelihood along that path trains the text prior and the invertible flow.
There is no posterior encoder or variational KL objective.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from nemo.collections.tts.modules.submodules import ConvNorm
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.collections.tts.parts.utils.monotonic_alignment import maximum_path_indices


@dataclass
class GlowTTSAlignmentOutput:
    """Alignment on the input acoustic grid; invalid and padded indices are -1."""

    loss: torch.Tensor
    token_indices: torch.Tensor
    durations: torch.Tensor
    valid_turns: torch.Tensor


class TinyGlowTTSAligner(nn.Module):
    """Learn monotonic alignments using a tiny flow and a unit-Gaussian text prior.

    Args:
        audio_channels: Acoustic feature dimension. It is preserved by the flow.
        text_channels: Dimension of existing text/CAS embeddings. No duplicate
            vocabulary embedding is created.
        hidden_channels: Width of the small convolutional subnetworks.
        num_flows: Number of affine-normalization/coupling/permutation blocks.
        num_text_layers: Number of residual convolution layers in the text prior.
        dequantization_noise: Half-width of uniform feature jitter during training.
            Quantized codec features need a continuous observation model; this
            learns the density of their jittered distribution. Set to zero for
            continuous features. No jitter is added during evaluation.
        mas_backend: ``auto``, ``cpu`` or ``triton``; see maximum_path_indices.
        max_log_scale: Bound on each affine log scale for numerical stability.

    The caller should provide detached audio-only features and detached text
    embeddings when the aligner is an auxiliary teacher for another generator.
    A fixed unit prior variance avoids a learned variance collapsing on repeated
    quantized vectors. All density calculations run in float32 under AMP.
    """

    def __init__(
        self,
        audio_channels: int,
        text_channels: int,
        hidden_channels: int = 64,
        num_flows: int = 4,
        num_text_layers: int = 2,
        dequantization_noise: float = 0.01,
        mas_backend: str = 'auto',
        max_log_scale: float = 2.0,
    ):
        super().__init__()
        if audio_channels < 2 or text_channels < 1 or hidden_channels < 1:
            raise ValueError('Audio dimension must be >= 2; text and hidden dimensions must be positive')
        if num_flows < 1 or num_text_layers < 0:
            raise ValueError('num_flows must be positive and num_text_layers must be nonnegative')
        if dequantization_noise < 0 or max_log_scale <= 0:
            raise ValueError('dequantization_noise must be nonnegative and max_log_scale must be positive')
        if mas_backend not in {'auto', 'cpu', 'triton'}:
            raise ValueError(f'Unknown MAS backend: {mas_backend}')
        self.audio_channels = audio_channels
        self.text_channels = text_channels
        self.dequantization_noise = dequantization_noise
        self.mas_backend = mas_backend
        self.text_encoder = _TextPrior(text_channels, hidden_channels, audio_channels, num_text_layers)
        self.flow = _AudioFlow(audio_channels, hidden_channels, num_flows, max_log_scale)

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_lens: torch.Tensor,
        text_lens: torch.Tensor,
    ) -> GlowTTSAlignmentOutput:
        """Align ``audio_features[B,T,C]`` to ``text_features[B,N,D]``.

        Turns with no text, insufficient audio frames, or out-of-range lengths
        receive no alignment and contribute no loss. The returned loss is the
        negative acoustic log likelihood per valid frame and feature dimension.
        Invalid-only batches retain zero gradients for every trainable parameter,
        allowing mixed tasks and DDP without changing the module graph.
        """
        if audio_features.ndim != 3 or text_features.ndim != 3:
            raise ValueError('Audio and text features must have shape [batch, time, channels]')
        batch_size, audio_steps, channels = audio_features.shape
        if channels != self.audio_channels or text_features.size(2) != self.text_channels:
            raise ValueError('Feature dimensions do not match the aligner configuration')
        if (
            text_features.size(0) != batch_size
            or audio_lens.shape != (batch_size,)
            or text_lens.shape != (batch_size,)
        ):
            raise ValueError('Audio/text batch sizes and length vectors must match')
        if audio_features.device != text_features.device:
            raise ValueError('Audio and text features must be on the same device')
        text_steps = text_features.size(1)
        audio_lens = audio_lens.to(device=audio_features.device, dtype=torch.long)
        text_lens = text_lens.to(device=audio_features.device, dtype=torch.long)
        valid_turns = (
            (text_lens > 0) & (audio_lens >= text_lens) & (audio_lens <= audio_steps) & (text_lens <= text_steps)
        )
        if batch_size == 0 or audio_steps == 0 or text_steps == 0:
            return self._empty_output(audio_features, text_features, valid_turns)

        with torch.autocast(device_type=audio_features.device.type, enabled=False):
            audio = audio_features.float().transpose(1, 2)
            text = text_features.float().transpose(1, 2)
            audio_mask = get_mask_from_lengths(audio_lens.clamp(0, audio_steps), x=audio)
            text_mask = get_mask_from_lengths(text_lens.clamp(0, text_steps), x=text)
            audio_mask = (audio_mask & valid_turns[:, None]).unsqueeze(1).float()
            text_mask = (text_mask & valid_turns[:, None]).unsqueeze(1).float()
            audio = audio * audio_mask
            if self.training and self.dequantization_noise:
                audio = audio + (2.0 * torch.rand_like(audio) - 1.0) * self.dequantization_noise * audio_mask
            z, log_det = self.flow(audio, audio_mask)
            means = self.text_encoder(text * text_mask, text_mask)

            effective_audio_lens = torch.where(valid_turns, audio_lens, 0)
            effective_text_lens = torch.where(valid_turns, text_lens, 0)
            with torch.no_grad():
                # The dense score grid is only needed for discrete search. Keep
                # it out of the autograd graph, including its clamp mask, so
                # microbatched turns do not retain [B,T,N] tensors for backward.
                squared_distance = (
                    z.square().sum(dim=1).unsqueeze(2)
                    + means.square().sum(dim=1).unsqueeze(1)
                    - 2.0 * torch.bmm(z.transpose(1, 2), means)
                ).clamp_min(0.0)
                log_probs = -0.5 * (squared_distance + self.audio_channels * math.log(2.0 * math.pi))
                indices = maximum_path_indices(
                    log_probs, effective_text_lens, effective_audio_lens, backend=self.mas_backend
                )
            frame_mask = (indices >= 0) & audio_mask[:, 0].bool()
            # Recompute the likelihood only along the selected path. This is
            # exactly the same fixed-path objective and keeps gradients into z,
            # the selected text statistics and the flow Jacobian.
            selected_means = means.gather(2, indices.clamp_min(0).unsqueeze(1).expand(-1, self.audio_channels, -1))
            selected_log_probs = -0.5 * (
                (z - selected_means).square().sum(dim=1) + self.audio_channels * math.log(2.0 * math.pi)
            )
            denominator = (frame_mask.sum() * self.audio_channels).clamp_min(1)
            loss = -((selected_log_probs * frame_mask).sum() + log_det.sum()) / denominator
            durations = torch.zeros(batch_size, text_steps, dtype=torch.long, device=indices.device)
            durations.scatter_add_(1, indices.clamp_min(0), frame_mask.long())
        return GlowTTSAlignmentOutput(loss, indices, durations, valid_turns)

    def _empty_output(self, audio_features, text_features, valid_turns):
        # This branch is only needed for tensors whose static time dimension is 0.
        # Anchoring all parameters keeps their DDP reductions well defined.
        loss = sum((parameter.sum() * 0.0 for parameter in self.parameters()), audio_features.new_zeros(()))
        return GlowTTSAlignmentOutput(
            loss=loss,
            token_indices=torch.full(audio_features.shape[:2], -1, dtype=torch.long, device=audio_features.device),
            durations=torch.zeros(text_features.shape[:2], dtype=torch.long, device=audio_features.device),
            valid_turns=valid_turns,
        )


class _TextPrior(nn.Module):
    """Small masked prior over existing token embeddings, with unit variance."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.input_projection = ConvNorm(in_channels, hidden_channels, kernel_size=1)
        self.layers = nn.ModuleList(
            [ConvNorm(hidden_channels, hidden_channels, kernel_size=3, w_init_gain='relu') for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.output_projection = ConvNorm(hidden_channels, out_channels, kernel_size=1)

    def forward(self, features, mask):
        hidden = self.input_projection(features, mask) * mask
        for layer, norm in zip(self.layers, self.norms):
            residual = torch.relu(layer(hidden, mask)) * mask
            hidden = norm((hidden + residual).transpose(1, 2)).transpose(1, 2) * mask
        return self.output_projection(hidden, mask) * mask


class _AffineCoupling(nn.Module):
    """Glow-style affine coupling, with a bounded, explicitly counted log scale."""

    def __init__(self, channels, hidden_channels, max_log_scale):
        super().__init__()
        self.first_channels = channels // 2
        self.second_channels = channels - self.first_channels
        self.max_log_scale = max_log_scale
        self.input_projection = ConvNorm(self.first_channels, hidden_channels, kernel_size=1, w_init_gain='relu')
        self.hidden_projection = ConvNorm(hidden_channels, hidden_channels, kernel_size=3, w_init_gain='relu')
        self.output_projection = nn.Conv1d(hidden_channels, 2 * self.second_channels, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, inputs, mask, reverse=False):
        first, second = inputs.split([self.first_channels, self.second_channels], dim=1)
        hidden = torch.relu(self.input_projection(first, mask)) * mask
        hidden = torch.relu(self.hidden_projection(hidden, mask)) * mask
        shift, raw_scale = (self.output_projection(hidden) * mask).chunk(2, dim=1)
        log_scale = self.max_log_scale * torch.tanh(raw_scale / self.max_log_scale)
        if reverse:
            transformed = (second - shift) * torch.exp(-log_scale)
            log_det = -(log_scale * mask).sum(dim=(1, 2))
        else:
            transformed = second * torch.exp(log_scale) + shift
            log_det = (log_scale * mask).sum(dim=(1, 2))
        return torch.cat([first, transformed], dim=1) * mask, log_det


class _AudioFlow(nn.Module):
    """Invertible acoustic transform with no time downsampling or bottleneck."""

    def __init__(self, channels, hidden_channels, num_flows, max_log_scale):
        super().__init__()
        self.max_log_scale = max_log_scale
        self.shifts = nn.ParameterList([nn.Parameter(torch.zeros(1, channels, 1)) for _ in range(num_flows)])
        self.log_scales = nn.ParameterList([nn.Parameter(torch.zeros(1, channels, 1)) for _ in range(num_flows)])
        self.couplings = nn.ModuleList(
            [_AffineCoupling(channels, hidden_channels, max_log_scale) for _ in range(num_flows)]
        )

    def forward(self, inputs, mask, reverse=False):
        hidden = inputs * mask
        log_det = inputs.new_zeros(inputs.size(0))
        layers = range(len(self.couplings) - 1, -1, -1) if reverse else range(len(self.couplings))
        valid_frames = mask.sum(dim=(1, 2))
        for layer_index in layers:
            log_scale = self.max_log_scale * torch.tanh(self.log_scales[layer_index] / self.max_log_scale)
            shift = self.shifts[layer_index]
            if reverse:
                hidden = hidden.flip(1)
                hidden, coupling_log_det = self.couplings[layer_index](hidden, mask, reverse=True)
                hidden = (hidden - shift) * torch.exp(-log_scale) * mask
                log_det = log_det + coupling_log_det - log_scale.sum() * valid_frames
            else:
                hidden = (hidden * torch.exp(log_scale) + shift) * mask
                log_det = log_det + log_scale.sum() * valid_frames
                hidden, coupling_log_det = self.couplings[layer_index](hidden, mask)
                hidden = hidden.flip(1)
                log_det = log_det + coupling_log_det
        return hidden, log_det
