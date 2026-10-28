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

"""Acoustic-likelihood aligner using the ST-TTS flow configuration.

This module learns the density of audio-derived features, rather than classifying
the transcript from text-conditioned decoder states. MAS selects a detached path;
the likelihood along that path trains the text prior and the invertible flow.
There is no posterior encoder or variational KL objective.

The default acoustic flow follows ST-TTS section 3.1.4: six Glow-style blocks,
128 hidden channels, and four gated residual kernel-five layers per affine
coupling, without additional time squeezing. Existing CAS embeddings and the
small text prior remain Nemotron-specific adaptations. ActNorm uses identity
initialization without data-dependent or distributed initialization. The earlier
compact flow remains available as ``flow_type='legacy'`` for old checkpoints.

References:
    https://www.isca-archive.org/interspeech_2021/kim21n_interspeech.pdf
    https://github.com/jaywalnut310/glow-tts
"""

# The ST-TTS acoustic blocks below adapt the Glow-TTS reference implementation.
# MIT License
# Copyright (c) 2020 Jaehyeon Kim
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
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
    """Learn alignments using an ST-TTS flow and a unit-Gaussian text prior.

    Args:
        audio_channels: Acoustic feature dimension. It is preserved by the flow.
        text_channels: Dimension of existing text/CAS embeddings. No duplicate
            vocabulary embedding is created.
        hidden_channels: Width of the small convolutional subnetworks.
        num_flows: Number of normalization/mixing/coupling blocks.
        num_text_layers: Number of residual convolution layers in the text prior.
        dequantization_noise: Half-width of uniform feature jitter during training.
            Quantized codec features need a continuous observation model; this
            learns the density of their jittered distribution. Set to zero for
            continuous features. No jitter is added during evaluation.
        mas_backend: ``auto``, ``cpu`` or ``triton``; see maximum_path_indices.
        max_log_scale: Optional bound on affine log scales. The ST-TTS path
            defaults to the reference's unrestricted log scales. Legacy defaults
            to its original bound of two when this argument is None.
        flow_type: ``sttts`` for the published flow configuration or ``legacy``
            for the original compact flow and its unchanged checkpoint keys.
        coupling_num_layers: Gated residual/skip layers in each ST-TTS coupling.
        coupling_kernel_size: Temporal kernel size of its residual layers.
        flow_n_split: Size of the shared invertible mixing matrix. A full
            channel matrix is used when the feature dimension is not divisible
            by this even split size, preserving support for odd dimensions.

    The caller should provide detached audio-only features and detached text
    embeddings when the aligner is an auxiliary teacher for another generator.
    A fixed unit prior variance avoids a learned variance collapsing on repeated
    quantized vectors. All density calculations run in float32 under AMP.
    """

    def __init__(
        self,
        audio_channels: int,
        text_channels: int,
        hidden_channels: int = 128,
        num_flows: int = 6,
        num_text_layers: int = 2,
        dequantization_noise: float = 0.01,
        mas_backend: str = 'auto',
        max_log_scale: float | None = None,
        flow_type: str = 'sttts',
        coupling_num_layers: int = 4,
        coupling_kernel_size: int = 5,
        flow_n_split: int = 4,
    ):
        super().__init__()
        if audio_channels < 2 or text_channels < 1 or hidden_channels < 1:
            raise ValueError('Audio dimension must be >= 2; text and hidden dimensions must be positive')
        if num_flows < 1 or num_text_layers < 0:
            raise ValueError('num_flows must be positive and num_text_layers must be nonnegative')
        if not math.isfinite(dequantization_noise) or dequantization_noise < 0:
            raise ValueError('dequantization_noise must be finite and nonnegative')
        if max_log_scale is not None and (not math.isfinite(max_log_scale) or max_log_scale <= 0):
            raise ValueError('max_log_scale must be None or finite and positive')
        if mas_backend not in {'auto', 'cpu', 'triton'}:
            raise ValueError(f'Unknown MAS backend: {mas_backend}')
        if flow_type not in {'sttts', 'legacy'}:
            raise ValueError(f'Unknown flow type: {flow_type}')
        if type(coupling_num_layers) is not int or coupling_num_layers < 1:
            raise ValueError('coupling_num_layers must be a positive integer')
        if type(coupling_kernel_size) is not int or coupling_kernel_size < 1 or coupling_kernel_size % 2 == 0:
            raise ValueError('coupling_kernel_size must be an odd positive integer')
        if type(flow_n_split) is not int or flow_n_split < 2 or flow_n_split % 2:
            raise ValueError('flow_n_split must be an even integer >= 2')
        self.audio_channels = audio_channels
        self.text_channels = text_channels
        self.hidden_channels = hidden_channels
        self.num_flows = num_flows
        self.num_text_layers = num_text_layers
        self.flow_type = flow_type
        self.coupling_num_layers = coupling_num_layers
        self.coupling_kernel_size = coupling_kernel_size
        self.flow_n_split = flow_n_split
        self.max_log_scale = max_log_scale
        self.dequantization_noise = dequantization_noise
        self.mas_backend = mas_backend
        self.text_encoder = _TextPrior(text_channels, hidden_channels, audio_channels, num_text_layers)
        if flow_type == 'legacy':
            self.flow = _AudioFlow(
                audio_channels, hidden_channels, num_flows, 2.0 if max_log_scale is None else max_log_scale
            )
        else:
            self.flow = _STTTSFlow(
                audio_channels,
                hidden_channels,
                num_flows,
                coupling_num_layers,
                coupling_kernel_size,
                flow_n_split,
                max_log_scale,
            )

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


class _STTTSFlow(nn.Module):
    """ST-TTS acoustic flow: ActNorm, learned channel mixing and gated coupling.

    The shared grouped mixing matrix has the same cross-half grouping as
    Glow-TTS's ``InvConvNear``. Normalization starts as the identity, avoiding
    per-rank initialization state or collectives inside independently packed
    turns. Both forward and reverse return the full log absolute determinant.
    """

    def __init__(
        self,
        channels,
        hidden_channels,
        num_flows,
        coupling_num_layers=4,
        coupling_kernel_size=5,
        flow_n_split=4,
        max_log_scale=None,
    ):
        super().__init__()
        if channels < 2 or hidden_channels < 1 or num_flows < 1:
            raise ValueError('Acoustic channels must be >= 2; hidden width and flow count must be positive')
        self.actnorms = nn.ModuleList([_ActNorm(channels, max_log_scale) for _ in range(num_flows)])
        self.mixers = nn.ModuleList([_InvertibleChannelMix(channels, flow_n_split) for _ in range(num_flows)])
        self.couplings = nn.ModuleList(
            [
                _ResidualCoupling(channels, hidden_channels, coupling_num_layers, coupling_kernel_size, max_log_scale)
                for _ in range(num_flows)
            ]
        )

    def forward(self, inputs, mask, reverse=False):
        hidden = inputs * mask
        log_det = inputs.new_zeros(inputs.size(0))
        indices = range(len(self.couplings) - 1, -1, -1) if reverse else range(len(self.couplings))
        for index in indices:
            layers = (
                (self.couplings[index], self.mixers[index], self.actnorms[index])
                if reverse
                else (self.actnorms[index], self.mixers[index], self.couplings[index])
            )
            for layer in layers:
                hidden, layer_log_det = layer(hidden, mask, reverse=reverse)
                log_det = log_det + layer_log_det
        return hidden, log_det


class _ActNorm(nn.Module):
    """Identity-initialized per-channel affine normalization, without data fitting."""

    def __init__(self, channels, max_log_scale=None):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.max_log_scale = max_log_scale

    def forward(self, inputs, mask, reverse=False):
        logs = self.logs
        if self.max_log_scale is not None:
            logs = self.max_log_scale * torch.tanh(logs / self.max_log_scale)
        log_det = logs.sum() * mask.sum(dim=(1, 2))
        if reverse:
            transformed = (inputs - self.bias) * torch.exp(-logs)
            log_det = -log_det
        else:
            transformed = inputs * torch.exp(logs) + self.bias
        return transformed * mask, log_det


class _InvertibleChannelMix(nn.Module):
    """Shared invertible 1x1 transform, grouping both affine-coupling halves.

    When ``channels`` is divisible by the requested even ``n_split``, a small
    matrix is shared across channel groups as in Glow-TTS. Other dimensions use
    a full channel matrix. The latter keeps odd codec feature dimensions valid
    without dropping or padding an observation channel.
    """

    def __init__(self, channels, n_split=4):
        super().__init__()
        if type(n_split) is not int or n_split < 2 or n_split % 2:
            raise ValueError('The channel mixing split must be an even integer >= 2')
        self.channels = channels
        self.grouped = channels % n_split == 0
        self.n_split = n_split if self.grouped else channels
        with torch.no_grad():
            weight, _ = torch.linalg.qr(torch.randn(self.n_split, self.n_split))
            if torch.linalg.det(weight) < 0:
                weight[:, 0].neg_()
        self.weight = nn.Parameter(weight)

    def forward(self, inputs, mask, reverse=False):
        # Determinants and inverses need float32 under low-precision execution;
        # retain float64 for numerical Jacobian/inverse validation.
        weight = self.weight if self.weight.dtype == torch.float64 else self.weight.float()
        log_abs_det = torch.linalg.slogdet(weight).logabsdet
        groups = self.channels // self.n_split
        log_det = log_abs_det * groups * mask.sum(dim=(1, 2))
        if reverse:
            weight = torch.linalg.inv(weight)
            log_det = -log_det
        weight = weight.to(dtype=inputs.dtype)
        batch, channels, steps = inputs.shape
        if self.grouped:
            grouped = inputs.reshape(batch, 2, groups, self.n_split // 2, steps)
            grouped = grouped.permute(0, 1, 3, 2, 4).reshape(batch, self.n_split, groups, steps)
            mixed = F.conv2d(grouped, weight.reshape(self.n_split, self.n_split, 1, 1))
            mixed = mixed.reshape(batch, 2, self.n_split // 2, groups, steps)
            mixed = mixed.permute(0, 1, 3, 2, 4).reshape(batch, channels, steps)
        else:
            mixed = F.conv1d(inputs, weight.unsqueeze(2))
        return mixed * mask, log_det


class _ResidualCoupling(nn.Module):
    """Affine coupling with weight-normalized gated residual/skip convolutions.

    Four width-128 kernel-five layers reproduce the ST-TTS flow capacity. Each
    of the first layers produces a residual and a skip output, while the last
    produces only a skip output. All temporal layers use dilation one. The
    zero-initialized final projection makes the initial coupling the identity.
    """

    def __init__(self, channels, hidden_channels, num_layers=4, kernel_size=5, max_log_scale=None):
        super().__init__()
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError('The affine coupling must have at least one residual layer')
        if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('The affine coupling kernel must be an odd positive integer')
        self.first_channels = channels // 2
        self.second_channels = channels - self.first_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.max_log_scale = max_log_scale
        self.input_projection = ConvNorm(self.first_channels, hidden_channels, kernel_size=1, use_weight_norm=True)
        self.in_layers = nn.ModuleList(
            [
                ConvNorm(hidden_channels, 2 * hidden_channels, kernel_size=kernel_size, use_weight_norm=True)
                for _ in range(num_layers)
            ]
        )
        self.res_skip_layers = nn.ModuleList(
            [
                ConvNorm(
                    hidden_channels,
                    2 * hidden_channels if index < num_layers - 1 else hidden_channels,
                    kernel_size=1,
                    use_weight_norm=True,
                )
                for index in range(num_layers)
            ]
        )
        self.output_projection = nn.Conv1d(hidden_channels, 2 * self.second_channels, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, inputs, mask, reverse=False):
        first, second = inputs.split([self.first_channels, self.second_channels], dim=1)
        hidden = self.input_projection(first, mask) * mask
        skip = torch.zeros_like(hidden)
        for index, (temporal, residual_skip) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            first_gate, second_gate = (temporal(hidden, mask) * mask).chunk(2, dim=1)
            gated = torch.tanh(first_gate) * torch.sigmoid(second_gate) * mask
            updates = residual_skip(gated, mask) * mask
            if index < self.num_layers - 1:
                residual, skip_update = updates.chunk(2, dim=1)
                hidden = (hidden + residual) * mask
                skip = skip + skip_update
            else:
                skip = skip + updates
        shift, log_scale = (self.output_projection(skip * mask) * mask).chunk(2, dim=1)
        if self.max_log_scale is not None:
            log_scale = self.max_log_scale * torch.tanh(log_scale / self.max_log_scale)
        log_det = (log_scale * mask).sum(dim=(1, 2))
        if reverse:
            transformed = (second - shift) * torch.exp(-log_scale)
            log_det = -log_det
        else:
            transformed = second * torch.exp(log_scale) + shift
        return torch.cat([first, transformed], dim=1) * mask, log_det
