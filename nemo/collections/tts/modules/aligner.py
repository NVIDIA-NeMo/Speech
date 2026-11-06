# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ST-TTS acoustic aligner, monotonic search and Nemotron-TTS training integration.

A VITS-style relative-position text encoder and an invertible acoustic flow learn
from detached conditioning embeddings and codec latents. MAS returns positions
on the original audio grid. Optional attentive statistics pooling extracts speaker
conditioning from codec references. Checkpoint helpers restore only the aligner.

The flow follows ST-TTS: https://www.isca-archive.org/interspeech_2021/kim21n_interspeech.pdf.
Flow and relative-attention blocks adapt the MIT-licensed Glow-TTS and VITS code:
https://github.com/jaywalnut310/glow-tts and https://github.com/jaywalnut310/vits.
"""

# MIT License
# Copyright (c) 2020, 2021 Jaehyeon Kim
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
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf, open_dict
from torch import nn

from nemo.collections.tts.modules.attention_alignment import AttentionAlignmentGuidance, AttentionAlignmentTurn
from nemo.collections.tts.modules.ffn_modules import ConvolutionLayer
from nemo.collections.tts.modules.submodules import ConditionalInput, ConvNorm
from nemo.collections.tts.parts.utils.helpers import binarize_attention_parallel, get_mask_from_lengths
from nemo.utils import logging

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class AlignmentEncoder(torch.nn.Module):
    """
    Module for alignment text and mel spectrogram.

    Args:
        n_mel_channels: Dimension of mel spectrogram.
        n_text_channels: Dimension of text embeddings.
        n_att_channels: Dimension of model
        temperature: Temperature to scale distance by.
            Suggested to be 0.0005 when using dist_type "l2" and 15.0 when using "cosine".
        condition_types: List of types for nemo.collections.tts.modules.submodules.ConditionalInput.
        dist_type: Distance type to use for similarity measurement. Supports "l2" and "cosine" distance.
    """

    def __init__(
        self,
        n_mel_channels=80,
        n_text_channels=512,
        n_att_channels=80,
        temperature=0.0005,
        condition_types=[],
        dist_type="l2",
    ):
        super().__init__()
        self.temperature = temperature
        self.cond_input = ConditionalInput(n_text_channels, n_text_channels, condition_types)
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels, n_text_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels * 2, n_att_channels, kernel_size=1, bias=True),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(n_mel_channels, n_mel_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels * 2, n_mel_channels, kernel_size=1, bias=True),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels, n_att_channels, kernel_size=1, bias=True),
        )
        if dist_type == "l2":
            self.dist_fn = self.get_euclidean_dist
        elif dist_type == "cosine":
            self.dist_fn = self.get_cosine_dist
        else:
            raise ValueError(f"Unknown distance type '{dist_type}'")

    @staticmethod
    def _apply_mask(inputs, mask, mask_value):
        if mask is None:
            return

        mask = rearrange(mask, "B T2 1 -> B 1 1 T2")
        inputs.data.masked_fill_(mask, mask_value)

    def get_dist(self, keys, queries, mask=None):
        """Calculation of distance matrix.

        Args:
            queries (torch.tensor): B x C1 x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): B x T2 x 1 tensor, binary mask for variable length entries and also can be used
                for ignoring unnecessary elements from keys in the resulting distance matrix (True = mask element, False = leave unchanged).
        Output:
            dist (torch.tensor): B x T1 x T2 tensor.
        """
        # B x C x T1
        queries_enc = self.query_proj(queries)
        # B x C x T2
        keys_enc = self.key_proj(keys)
        # B x 1 x T1 x T2
        dist = self.dist_fn(queries_enc=queries_enc, keys_enc=keys_enc)

        self._apply_mask(dist, mask, float("inf"))

        return dist.squeeze(1)

    @staticmethod
    def get_euclidean_dist(queries_enc, keys_enc):
        queries_enc = rearrange(queries_enc, "B C T1 -> B C T1 1")
        keys_enc = rearrange(keys_enc, "B C T2 -> B C 1 T2")
        # B x C x T1 x T2
        distance = (queries_enc - keys_enc) ** 2
        # B x 1 x T1 x T2
        l2_dist = distance.sum(axis=1, keepdim=True)
        return l2_dist

    @staticmethod
    def get_cosine_dist(queries_enc, keys_enc):
        queries_enc = rearrange(queries_enc, "B C T1 -> B C T1 1")
        keys_enc = rearrange(keys_enc, "B C T2 -> B C 1 T2")
        cosine_dist = -torch.nn.functional.cosine_similarity(queries_enc, keys_enc, dim=1)
        cosine_dist = rearrange(cosine_dist, "B T1 T2 -> B 1 T1 T2")
        return cosine_dist

    @staticmethod
    def get_durations(attn_soft, text_len, spect_len):
        """Calculation of durations.

        Args:
            attn_soft (torch.tensor): B x 1 x T1 x T2 tensor.
            text_len (torch.tensor): B tensor, lengths of text.
            spect_len (torch.tensor): B tensor, lengths of mel spectrogram.
        """
        attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
        durations = attn_hard.sum(2)[:, 0, :]
        assert torch.all(torch.eq(durations.sum(dim=1), spect_len))
        return durations

    @staticmethod
    def get_mean_dist_by_durations(dist, durations, mask=None):
        """Select elements from the distance matrix for the given durations and mask and return mean distance.

        Args:
            dist (torch.tensor): B x T1 x T2 tensor.
            durations (torch.tensor): B x T2 tensor. Dim T2 should sum to T1.
            mask (torch.tensor): B x T2 x 1 binary mask for variable length entries and also can be used
                for ignoring unnecessary elements in dist by T2 dim (True = mask element, False = leave unchanged).
        Output:
            mean_dist (torch.tensor): B x 1 tensor.
        """
        batch_size, t1_size, t2_size = dist.size()
        assert torch.all(torch.eq(durations.sum(dim=1), t1_size))

        AlignmentEncoder._apply_mask(dist, mask, 0)

        # TODO(oktai15): make it more efficient
        mean_dist_by_durations = []
        for dist_idx in range(batch_size):
            mean_dist_by_durations.append(
                torch.mean(
                    dist[
                        dist_idx,
                        torch.arange(t1_size),
                        torch.repeat_interleave(torch.arange(t2_size), repeats=durations[dist_idx]),
                    ]
                )
            )

        return torch.tensor(mean_dist_by_durations, dtype=dist.dtype, device=dist.device)

    @staticmethod
    def get_mean_distance_for_word(l2_dists, durs, start_token, num_tokens):
        """Calculates the mean distance between text and audio embeddings given a range of text tokens.

        Args:
            l2_dists (torch.tensor): L2 distance matrix from Aligner inference. T1 x T2 tensor.
            durs (torch.tensor): List of durations corresponding to each text token. T2 tensor. Should sum to T1.
            start_token (int): Index of the starting token for the word of interest.
            num_tokens (int): Length (in tokens) of the word of interest.
        Output:
            mean_dist_for_word (float): Mean embedding distance between the word indicated and its predicted audio frames.
        """
        # Need to calculate which audio frame we start on by summing all durations up to the start token's duration
        start_frame = torch.sum(durs[:start_token]).data

        total_frames = 0
        dist_sum = 0

        # Loop through each text token
        for token_ind in range(start_token, start_token + num_tokens):
            # Loop through each frame for the given text token
            for frame_ind in range(start_frame, start_frame + durs[token_ind]):
                # Recall that the L2 distance matrix is shape [spec_len, text_len]
                dist_sum += l2_dists[frame_ind, token_ind]

            # Update total frames so far & the starting frame for the next token
            total_frames += durs[token_ind]
            start_frame += durs[token_ind]

        return dist_sum / total_frames

    def forward(self, queries, keys, mask=None, attn_prior=None, conditioning=None):
        """Forward pass of the aligner encoder.

        Args:
            queries (torch.tensor): B x C1 x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): B x T2 x 1 tensor, binary mask for variable length entries (True = mask element, False = leave unchanged).
            attn_prior (torch.tensor): prior for attention matrix.
            conditioning (torch.tensor): B x 1 x C2 conditioning embedding
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        keys = self.cond_input(keys.transpose(1, 2), conditioning).transpose(1, 2)
        # B x C x T1
        queries_enc = self.query_proj(queries)
        # B x C x T2
        keys_enc = self.key_proj(keys)
        # B x 1 x T1 x T2
        distance = self.dist_fn(queries_enc=queries_enc, keys_enc=keys_enc)
        attn = -self.temperature * distance

        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)

        attn_logprob = attn.clone()

        self._apply_mask(attn, mask, -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob


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
        audio_channels: Internal acoustic width, preserved by the flow. V2 defaults to 80.
        input_audio_channels: Codec feature width before projection. None uses audio_channels.
        text_channels: Dimension of existing text/CAS embeddings. No duplicate
            vocabulary embedding is created.
        hidden_channels: Width of the acoustic coupling networks and text prior.
        num_flows: Number of normalization/mixing/coupling blocks.
        num_text_layers: Number of Transformer blocks.
        dequantization_noise: Half-width of uniform feature jitter during training.
            Quantized codec features need a continuous observation model; this
            learns the density of their jittered distribution. Set to zero for
            continuous features. No jitter is added during evaluation.
        mas_backend: ``auto``, ``cpu`` or ``triton``; see maximum_path_indices.
        max_log_scale: Optional bound on affine log scales. None retains the
            reference's unrestricted log scales.
        flow_type: ``sttts``, retained in serialized architecture configurations.
        coupling_num_layers: Gated residual/skip layers in each ST-TTS coupling.
        coupling_kernel_size: Temporal kernel size of its residual layers.
        flow_n_split: Size of the shared invertible mixing matrix. A full
            channel matrix is used when the feature dimension is not divisible
            by this even split size, preserving support for odd dimensions.
        text_encoder_type: ``relative_transformer``, retained in serialized
            architecture configurations.
        text_num_heads: Number of bidirectional text-attention heads.
        text_filter_channels: Convolutional feed-forward width; None uses four
            times hidden_channels.
        text_kernel_size: Odd kernel size of the two feed-forward convolutions.
        text_dropout: Dropout within attention, FFN and residual branches.
        text_rel_attn_window_size: Maximum learned relative key/value offset.
            Content attention remains global beyond this window.
        use_speaker_conditioning: Enable learnable ASP reference conditioning of
            the ST-TTS flow. Defaults to True.
        speaker_hidden_channels: Width of the masked reference convolution stack.
        speaker_embedding_dim: Size of its global attentive-statistics embedding.
        speaker_num_layers: Number of residual reference convolution layers.

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
        text_encoder_type: str = 'relative_transformer',
        text_num_heads: int = 2,
        text_filter_channels: int | None = None,
        text_kernel_size: int = 3,
        text_dropout: float = 0.1,
        text_rel_attn_window_size: int = 4,
        use_speaker_conditioning: bool = True,
        speaker_hidden_channels: int = 64,
        speaker_embedding_dim: int = 64,
        speaker_num_layers: int = 2,
        input_audio_channels: int | None = None,
    ):
        super().__init__()
        input_audio_channels = audio_channels if input_audio_channels is None else input_audio_channels
        if type(input_audio_channels) is not int or input_audio_channels < 1:
            raise ValueError('input_audio_channels must be a positive integer')
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
        if flow_type != 'sttts':
            raise ValueError('The aligner supports only flow_type=sttts')
        if type(coupling_num_layers) is not int or coupling_num_layers < 1:
            raise ValueError('coupling_num_layers must be a positive integer')
        if type(coupling_kernel_size) is not int or coupling_kernel_size < 1 or coupling_kernel_size % 2 == 0:
            raise ValueError('coupling_kernel_size must be an odd positive integer')
        if type(flow_n_split) is not int or flow_n_split < 2 or flow_n_split % 2:
            raise ValueError('flow_n_split must be an even integer >= 2')
        if text_encoder_type != 'relative_transformer':
            raise ValueError('The aligner supports only text_encoder_type=relative_transformer')
        if type(use_speaker_conditioning) is not bool:
            raise ValueError('use_speaker_conditioning must be a boolean')
        if use_speaker_conditioning:
            for name, value in (
                ('speaker_hidden_channels', speaker_hidden_channels),
                ('speaker_embedding_dim', speaker_embedding_dim),
                ('speaker_num_layers', speaker_num_layers),
            ):
                if type(value) is not int or value < 1:
                    raise ValueError(f'{name} must be a positive integer')
        self.audio_channels = audio_channels
        self.input_audio_channels = input_audio_channels
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
        self.text_encoder_type = text_encoder_type
        self.text_num_heads = text_num_heads
        self.text_filter_channels = 4 * hidden_channels if text_filter_channels is None else text_filter_channels
        self.text_kernel_size = text_kernel_size
        self.text_dropout = text_dropout
        self.text_rel_attn_window_size = text_rel_attn_window_size
        self.use_speaker_conditioning = use_speaker_conditioning
        self.speaker_hidden_channels = speaker_hidden_channels
        self.speaker_embedding_dim = speaker_embedding_dim
        self.speaker_num_layers = speaker_num_layers
        self.audio_projection = (
            nn.Identity()
            if input_audio_channels == audio_channels
            else nn.utils.parametrizations.orthogonal(
                nn.Linear(input_audio_channels, audio_channels, bias=False), orthogonal_map='householder'
            )
        )
        self.text_encoder = _TransformerTextPrior(
            text_channels,
            hidden_channels,
            audio_channels,
            num_text_layers,
            text_num_heads,
            self.text_filter_channels,
            text_kernel_size,
            text_dropout,
            text_rel_attn_window_size,
        )
        self.flow = _STTTSFlow(
            audio_channels,
            hidden_channels,
            num_flows,
            coupling_num_layers,
            coupling_kernel_size,
            flow_n_split,
            max_log_scale,
            speaker_channels=speaker_embedding_dim if use_speaker_conditioning else 0,
        )
        self.speaker_encoder = (
            CodecSpeakerEncoder(audio_channels, speaker_hidden_channels, speaker_embedding_dim, speaker_num_layers)
            if use_speaker_conditioning
            else None
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_lens: torch.Tensor,
        text_lens: torch.Tensor,
        speaker_reference_features: torch.Tensor | None = None,
        speaker_reference_lens: torch.Tensor | None = None,
        speaker_embeddings: torch.Tensor | None = None,
    ) -> GlowTTSAlignmentOutput:
        """Align ``audio_features[B,T,C]`` to ``text_features[B,N,D]``.

        Turns with no text, insufficient audio frames, or out-of-range lengths
        receive no alignment and contribute no loss. The returned loss is the
        negative log likelihood per valid frame and feature dimension, measured
        in the projected acoustic space when the codec width differs.
        Invalid-only batches retain zero gradients for every trainable parameter,
        allowing mixed tasks and DDP without changing the module graph.

        Optional reference features have shape [B,R,C], on the same acoustic
        feature grid. Missing references use zero speaker conditioning; target
        audio is never substituted. Only the observations should be detached,
        so the ASP encoder can learn from the alignment likelihood.
        Alternatively, pass precomputed [B,D] ``speaker_embeddings`` to reuse one
        reference encoding across turn packs. The two interfaces are exclusive.
        """
        if audio_features.ndim != 3 or text_features.ndim != 3:
            raise ValueError('Audio and text features must have shape [batch, time, channels]')
        batch_size, audio_steps, channels = audio_features.shape
        if channels != self.input_audio_channels or text_features.size(2) != self.text_channels:
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
            audio = self._project_audio(audio_features, audio_mask[:, 0].bool()).transpose(1, 2)
            if self.training and self.dequantization_noise:
                audio = audio + (2.0 * torch.rand_like(audio) - 1.0) * self.dequantization_noise * audio_mask
            if self.use_speaker_conditioning:
                if speaker_embeddings is None:
                    speaker_embedding = self._encode_speaker_reference(
                        audio_features, speaker_reference_features, speaker_reference_lens
                    )
                else:
                    if speaker_reference_features is not None or speaker_reference_lens is not None:
                        raise ValueError('Pass speaker embeddings or reference features, not both')
                    if (
                        speaker_embeddings.shape != (batch_size, self.speaker_embedding_dim)
                        or speaker_embeddings.device != audio_features.device
                    ):
                        raise ValueError('Speaker embeddings must match the audio batch, embedding width and device')
                    speaker_embedding = speaker_embeddings.float()
                z, log_det = self.flow(audio, audio_mask, speaker_embedding=speaker_embedding)
            else:
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

    def encode_speaker_reference(self, reference_features, reference_lens):
        """Pool each reference once, retaining ASP gradients across turn packs."""
        if not self.use_speaker_conditioning:
            raise ValueError('Speaker reference encoding requires use_speaker_conditioning=true')
        if reference_features is None or reference_lens is None:
            raise ValueError('Speaker reference features and lengths must be provided together')
        if (
            reference_features.ndim != 3
            or reference_features.size(2) != self.input_audio_channels
            or reference_lens.shape != (reference_features.size(0),)
            or reference_lens.device != reference_features.device
        ):
            raise ValueError('Speaker references must match the configured input width, lengths and device')
        with torch.autocast(device_type=reference_features.device.type, enabled=False):
            mask = (
                torch.arange(reference_features.size(1), device=reference_features.device)[None, :]
                < reference_lens[:, None]
            )
            return self.speaker_encoder(self._project_audio(reference_features, mask), reference_lens)

    def _project_audio(self, features, mask):
        # Orthogonal weights keep the auxiliary likelihood from shrinking its input scale.
        features = features.float().masked_fill(~mask[:, :, None], 0.0)
        return self.audio_projection(features).masked_fill(~mask[:, :, None], 0.0)

    def _encode_speaker_reference(self, audio_features, reference_features, reference_lens):
        if (reference_features is None) != (reference_lens is None):
            raise ValueError('Speaker reference features and lengths must be provided together')
        if reference_features is None:
            reference_features = audio_features.new_zeros((audio_features.size(0), 1, self.input_audio_channels))
            reference_lens = torch.zeros(audio_features.size(0), device=audio_features.device, dtype=torch.long)
        elif (
            reference_features.ndim != 3
            or reference_features.size(0) != audio_features.size(0)
            or reference_features.size(2) != self.input_audio_channels
            or reference_features.device != audio_features.device
            or reference_lens.shape != (audio_features.size(0),)
        ):
            raise ValueError('Speaker references must match the audio batch, feature channels and device')
        return self.encode_speaker_reference(reference_features, reference_lens)

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


@dataclass
class TurnAlignmentOutput:
    """Acoustic likelihood and MAS positions on the original conversation grid."""

    loss: torch.Tensor
    token_indices: torch.Tensor
    valid_turn_count: torch.Tensor
    turn_count: torch.Tensor
    one_frame_tokens: torch.Tensor
    token_count: torch.Tensor


@dataclass
class AttentionAlignmentTrainingOutput:
    teacher: TurnAlignmentOutput
    guidance: AttentionAlignmentGuidance | None
    metrics: dict


def resolve_aligner_config(config: Mapping | None = None) -> dict:
    """Resolve the ST-TTS flow and relative-transformer architecture configuration."""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError('aligned_text_aligner must be a configuration mapping')
    values = dict(config or {})
    defaults = {
        'flow_type': 'sttts',
        'hidden_channels': 128,
        'num_flows': 6,
        'num_text_layers': 2,
        'text_encoder_type': 'relative_transformer',
        'text_num_heads': 2,
        'text_filter_channels': None,
        'text_kernel_size': 3,
        'text_dropout': 0.1,
        'text_rel_attn_window_size': 4,
        'use_speaker_conditioning': True,
        'speaker_hidden_channels': 64,
        'speaker_embedding_dim': 64,
        'speaker_num_layers': 2,
        'coupling_num_layers': 4,
        'coupling_kernel_size': 5,
        'flow_n_split': 4,
        'max_log_scale': None,
        'dequantization_noise': 0.01,
        'mas_backend': 'auto',
    }
    unknown = values.keys() - defaults.keys() - {'audio_channels', 'text_channels', 'input_audio_channels'}
    if unknown:
        raise ValueError(f'Unknown aligned_text_aligner options: {sorted(unknown)}')
    defaults.update(values)
    for name, supported in (('flow_type', 'sttts'), ('text_encoder_type', 'relative_transformer')):
        if defaults[name] != supported:
            raise ValueError(f'aligned_text_aligner.{name} must be {supported!r}, got {defaults[name]!r}')
    return defaults


def resolve_attention_alignment_config(config=None):
    settings = dict(config or {})
    defaults = dict(
        start_step=10000,
        ramp_steps=1000,
        prior_anneal_steps=5000,
        query_chunk_size=128,
        look_back_tokens=0,
        look_ahead_tokens=0,
        loss_weight=0.1,
        prior_strength=1.0,
        band_width=1.0,
        heads=[0],
        layers='middle',
    )
    # V2 always aligns the exact conditioning tokens; retain this legacy option
    # only as an explicit assertion rather than selecting another transcript.
    force_raw = settings.pop('force_raw_text', True)
    if type(force_raw) is not bool:
        raise ValueError('attention_alignment.force_raw_text must be a boolean')
    unknown = settings.keys() - defaults.keys()
    if unknown:
        raise ValueError(f'Unknown attention_alignment options: {sorted(unknown)}')
    defaults.update(settings)
    for name, minimum in (
        ('start_step', 0),
        ('ramp_steps', 0),
        ('prior_anneal_steps', 1),
        ('query_chunk_size', 1),
        ('look_back_tokens', 0),
        ('look_ahead_tokens', 0),
    ):
        if type(defaults[name]) is not int or defaults[name] < minimum:
            raise ValueError(f'attention_alignment.{name} must be an integer >= {minimum}')
    for name in ('loss_weight', 'prior_strength', 'band_width'):
        value = defaults[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'attention_alignment.{name} must be finite and nonnegative')
        defaults[name] = float(value)
    if defaults['band_width'] <= 0 or defaults['prior_strength'] > 1:
        raise ValueError('attention_alignment requires band_width > 0 and prior_strength <= 1')
    heads = defaults['heads']
    layers = defaults['layers']
    if OmegaConf.is_list(layers):
        layers = list(layers)
    checked = AttentionAlignmentGuidance(turns=[], heads=heads, layers=layers)
    defaults['heads'], defaults['layers'] = checked.heads, checked.layers
    return defaults


def load_pretrained_text_speech_aligner(
    aligner: nn.Module, checkpoint_path: str | Path | None, *, aligner_config: Mapping | None = None
) -> tuple[nn.Module, dict]:
    """Partially load aligner weights using the receiving model's configuration.

    Accepts bare state dictionaries, Lightning checkpoints and ``.nemo`` archives.
    Only aligner weights are selected from full TTS checkpoints. Matching tensors
    and compatible leading rows are copied using NeMo's partial initializer;
    missing or incompatible tensors retain their receiving-model initialization.
    Donor configuration metadata is ignored. No source TTS model is instantiated.

    A separate candidate keeps loading atomic and preserves receiver RNG, device,
    dtype, training mode and parameter trainability. Call before optimizer setup.
    ``None`` preserves the receiving object. Non-NeMo checkpoints use pickle and
    must come from a trusted source.
    """
    config = (
        resolve_aligner_config(aligner_config)
        if aligner_config is not None
        else {key: getattr(aligner, key, default) for key, default in resolve_aligner_config().items()}
    )
    dimensions = {
        'audio_channels': aligner.audio_channels,
        'text_channels': aligner.text_channels,
        'input_audio_channels': aligner.input_audio_channels,
    }
    for name, supplied in dimensions.items():
        requested = config.pop(name, supplied)
        if requested is not None and requested != supplied:
            raise ValueError(f'Configured aligner {name}={requested} does not match the receiving model ({supplied})')
    if checkpoint_path is None:
        return aligner, config

    from nemo.collections.speechlm2.parts.pretrained import set_model_dict_for_partial_init

    source_path = str(checkpoint_path)
    state, _ = _read_checkpoint(source_path)
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        candidate = type(aligner)(**dimensions, **config)
    parameter = next(aligner.parameters(), None)
    if parameter is not None:
        candidate = candidate.to(device=parameter.device, dtype=parameter.dtype)
    candidate.train(aligner.training)
    candidate.load_state_dict(aligner.state_dict(), strict=True)
    trainable = {name: parameter.requires_grad for name, parameter in aligner.named_parameters()}
    for name, parameter in candidate.named_parameters():
        parameter.requires_grad_(trainable.get(name, True))

    expected = candidate.state_dict()
    selected = _select_aligner_weights(state, expected, source_path)
    if not all(isinstance(value, torch.Tensor) for value in selected.values()):
        raise ValueError(f'Checkpoint {source_path!r} contains non-tensor aligner weights.')
    if any(value.is_meta for value in selected.values()):
        raise ValueError(f'Checkpoint {source_path!r} contains meta aligner tensors with no values.')
    if not expected.keys() & selected.keys():
        raise ValueError(f'Checkpoint {source_path!r} contains no matching text_aligner weights.')
    weights = set_model_dict_for_partial_init(selected, expected, allow_partial_copy=True)
    candidate.load_state_dict(weights, strict=True)
    return candidate, config


def init_training_alignment(model, cfg):
    """Create the training teacher only when explicitly enabled."""
    model.use_attention_alignment = cfg.get('use_attention_alignment', False)
    if type(model.use_attention_alignment) is not bool:
        raise ValueError('use_attention_alignment must be a boolean')
    model.attention_alignment_config = resolve_attention_alignment_config(cfg.get('attention_alignment'))
    model.aligner_loss_weight = float(cfg.get('aligner_loss_weight', 1.0))
    if not math.isfinite(model.aligner_loss_weight) or model.aligner_loss_weight < 0:
        raise ValueError('aligner_loss_weight must be finite and nonnegative')
    for name, default in (('aligned_text_max_turns_per_batch', 32), ('aligned_text_max_frames_per_batch', 8192)):
        value = cfg.get(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not model.use_attention_alignment:
        return

    restoring = bool(model._is_model_being_restored())
    settings = resolve_aligner_config(cfg.get('aligned_text_aligner'))
    channels = int(model._codec_model.vector_quantizer.codebook_dim) * model.frame_stacking_factor
    dimensions = dict(
        audio_channels=settings.pop('audio_channels', 80),
        input_audio_channels=channels,
        text_channels=int(cfg.embedding_dim),
    )
    for name in ('input_audio_channels', 'text_channels'):
        actual = dimensions[name]
        configured = settings.pop(name, actual)
        if configured is not None and configured != actual:
            raise ValueError(f'aligned_text_aligner.{name}={configured} does not match model features ({actual})')
    model.text_aligner = TinyGlowTTSAligner(**dimensions, **settings)
    path = cfg.get('pretrained_model_text_speech_aligner')
    if path and not restoring:
        model.text_aligner, settings = load_pretrained_text_speech_aligner(
            model.text_aligner, path, aligner_config=settings
        )
    model.aligned_text_aligner_config = settings
    with open_dict(model.cfg):
        model.cfg.aligned_text_aligner = dict(settings, **dimensions)
    logging.info(
        f'V2 aligner acoustic features: {channels} codec channels -> '
        f'{model.text_aligner.audio_channels} flow channels; '
        f'speaker conditioning={model.text_aligner.use_speaker_conditioning}.'
    )
    attention_blocks = [
        index for index, kind in enumerate(model.decoder.config.layers_block_type) if kind == 'attention'
    ]
    selection = AttentionAlignmentGuidance(turns=[], layers=model.attention_alignment_config['layers'])
    chosen = selection.resolve_layers(len(attention_blocks))
    heads = model.attention_alignment_config['heads']
    if heads is not None and any(head >= model.decoder.config.num_attention_heads for head in heads):
        raise ValueError('attention_alignment.heads contains an index outside the backbone head count')
    logging.info(
        f'V2 alignment supervises main-backbone attention ordinals {chosen}, '
        f'blocks {tuple(attention_blocks[index] for index in chosen)}, heads {heads}.'
    )


def reload_pretrained_aligner(model, checkpoint_path=None):
    """Apply an explicit teacher archive after loading ordinary TTS pretraining."""
    path = checkpoint_path or model.cfg.get('pretrained_model_text_speech_aligner')
    if checkpoint_path and not model.use_attention_alignment:
        raise ValueError('pretrained_model_text_speech_aligner requires use_attention_alignment=True')
    if not model.use_attention_alignment or not path:
        return

    model.text_aligner, settings = load_pretrained_text_speech_aligner(
        model.text_aligner, path, aligner_config=model.aligned_text_aligner_config
    )
    model.aligned_text_aligner_config = settings
    with open_dict(model.cfg):
        model.cfg.aligned_text_aligner = dict(
            settings,
            audio_channels=model.text_aligner.audio_channels,
            input_audio_channels=model.text_aligner.input_audio_channels,
            text_channels=model.text_aligner.text_channels,
        )
        model.cfg.pretrained_model_text_speech_aligner = path


def alignment_schedule(settings, step):
    elapsed = int(step) - settings['start_step']
    ramp = float(elapsed >= 0) if settings['ramp_steps'] == 0 else min(1.0, max(0.0, elapsed / settings['ramp_steps']))
    decay = min(1.0, max(0.0, (elapsed - settings['ramp_steps']) / settings['prior_anneal_steps']))
    return ramp, settings['prior_strength'] * ramp * (1.0 - decay)


def prepare_attention_alignment_training(
    model,
    audio_codes,
    audio_lens,
    audio_targets,
    text,
    text_lens,
    context_lens,
    audio_delay,
    reference_codes=None,
    reference_lens=None,
    dropout_conditioning=False,
):
    """Train the acoustic likelihood and prepare a detached, causal attention teacher."""
    tokens, token_lens, turn_indices, key_positions = compact_conditioning_turns(
        model, text, text_lens, audio_lens, bool(model.cfg.get('use_multiturn_dataset', False))
    )
    raw_valid = valid_original_codec_frames(model, audio_codes, audio_lens)
    within = torch.arange(audio_codes.size(2), device=audio_codes.device)[None, :] < audio_lens[:, None]
    invalid = F.pad(within & ~raw_valid, (0, (-audio_codes.size(2)) % model.frame_stacking_factor))
    stacked_steps = (audio_codes.size(2) + model.frame_stacking_factor - 1) // model.frame_stacking_factor
    invalid = invalid.view(audio_codes.size(0), stacked_steps, model.frame_stacking_factor).any(-1)
    for row, (sample, start, end) in enumerate(turn_indices.detach().cpu().tolist()):
        if start < 0 or end <= start or bool(invalid[sample, start:end].any()):
            turn_indices[row, 2] = start
    features = decode_alignment_features(model, audio_codes, audio_lens)
    target_steps = audio_targets.size(2)
    features = F.pad(features, (0, 0, 0, max(0, target_steps - features.size(1))))[:, :target_steps]
    reference_features, reference_lengths = None, None
    if model.text_aligner.use_speaker_conditioning and reference_codes is not None:
        reference_codes, reference_lens = compact_speaker_reference(model, reference_codes, reference_lens)
        reference_features = decode_alignment_features(model, reference_codes, reference_lens)
        reference_lengths = torch.div(
            reference_lens + model.frame_stacking_factor - 1, model.frame_stacking_factor, rounding_mode='floor'
        )
    teacher = align_text_turns(
        model.text_aligner,
        features,
        tokens,
        token_lens,
        turn_indices,
        embed_text=lambda ids, lengths: model.embed_text_tokens(ids, text_lens=lengths),
        max_turns_per_batch=model.cfg.get('aligned_text_max_turns_per_batch', 32),
        max_frames_per_batch=model.cfg.get('aligned_text_max_frames_per_batch', 8192),
        speaker_reference_features=reference_features,
        speaker_reference_lens=reference_lengths,
    )
    settings = model.attention_alignment_config
    loss_scale, prior_strength = alignment_schedule(settings, model.global_step)
    turns, causal_frames = [], tokens.new_zeros(())
    metadata = torch.cat([turn_indices, token_lens[:, None]], dim=1).detach().cpu().tolist()
    for row, (sample, start, end, count) in enumerate(metadata):
        if start < 0 or end <= start or end > teacher.token_indices.size(1) or not count:
            continue
        path = teacher.token_indices[sample, start:end].detach()
        keys = key_positions[row, :count].long() + context_lens[sample]
        queries = torch.arange(start, end, device=text.device) + audio_delay[sample]
        selected_keys = keys.gather(0, path.clamp(0, count - 1))
        causal = (path >= 0) & (selected_keys <= queries)
        causal_frames += causal.sum()
        if not dropout_conditioning and loss_scale > 0 and causal.any():
            turns.append(AttentionAlignmentTurn(sample, queries[causal].long(), keys, path[causal].long()))
    guidance = (
        AttentionAlignmentGuidance(
            turns=turns,
            prior_strength=prior_strength,
            band_width=settings['band_width'],
            heads=settings['heads'],
            query_chunk_size=settings['query_chunk_size'],
            layers=settings['layers'],
            look_back_tokens=settings['look_back_tokens'],
            look_ahead_tokens=settings['look_ahead_tokens'],
        )
        if turns
        else None
    )
    spans = turn_indices[:, 2] - turn_indices[:, 1]
    metrics = dict(
        turns=teacher.turn_count,
        valid_turns=teacher.valid_turn_count,
        teacher_frames=teacher.token_indices.ge(0).sum(),
        causal_frames=causal_frames,
        supervised_turns=tokens.new_tensor(len(turns)),
        invalid_spans=spans.le(0).sum(),
        short_turns=((spans > 0) & (spans < token_lens)).sum(),
        empty_turns=token_lens.eq(0).sum(),
        one_frame_tokens=teacher.one_frame_tokens,
        token_count=teacher.token_count,
        loss_scale=loss_scale,
        prior_strength=prior_strength,
    )
    return AttentionAlignmentTrainingOutput(teacher, guidance, metrics)


def compact_conditioning_turns(model, text, text_lens, audio_lens, is_multiturn):
    """Recover raw conditioning IDs, exact key positions, and acoustic spans.

    Multiturn text is already on the stacked acoustic grid. BOS marks the
    supervision start; text EOS ends its transcript, while the interruption
    marker supplies its exclusive speech end. The last marker may fall exactly
    at the crop boundary and therefore be absent. No lexical token is dropped.
    """
    ends = torch.div(audio_lens + model.frame_stacking_factor - 1, model.frame_stacking_factor, rounding_mode='floor')
    rows, indices, positions = [], [], []
    special = {model.pad_id, model.bos_id, model.eos_id, model.interruption_token_id}
    source = text.detach().cpu().tolist()
    lengths = text_lens.detach().cpu().tolist()
    acoustic_ends = ends.detach().cpu().tolist()
    for sample, (values, length, acoustic_end) in enumerate(zip(source, lengths, acoustic_ends)):
        values = values[:length]
        if not is_multiturn:
            keys = [i for i, token in enumerate(values) if token not in special and 0 <= token < model.text_vocab_size]
            if keys:
                rows.append([values[i] for i in keys])
                positions.append(keys)
                indices.append([sample, 0, acoustic_end])
            continue
        start, ids, keys, ended_text = None, [], [], False

        def append_turn(end):
            if ids:
                rows.append(list(ids))
                positions.append(list(keys))
                indices.append([sample, start, min(end, acoustic_end)])

        for position, token in enumerate(values):
            if token == model.bos_id:
                append_turn(position)
                start, ids, keys, ended_text = position, [], [], False
            elif token == model.interruption_token_id:
                append_turn(position)
                start, ids, keys, ended_text = None, [], [], False
            elif token == model.eos_id:
                ended_text = True
            elif token not in special and 0 <= token < model.text_vocab_size:
                if ended_text:
                    append_turn(position)
                    start, ids, keys, ended_text = position, [], [], False
                if start is None:
                    start = position
                ids.append(token)
                keys.append(position)
        append_turn(acoustic_end)
    width = max((len(row) for row in rows), default=0)
    compact = text.new_full((len(rows), width), model.pad_id)
    key_positions = text.new_full((len(rows), width), -1)
    for row, (ids, keys) in enumerate(zip(rows, positions)):
        compact[row, : len(ids)] = text.new_tensor(ids)
        key_positions[row, : len(keys)] = text.new_tensor(keys)
    return compact, text.new_tensor([len(row) for row in rows]), text.new_tensor(indices).reshape(-1, 3), key_positions


def align_text_turns(
    aligner: nn.Module,
    audio_features: torch.Tensor,
    text_tokens: torch.Tensor,
    text_lens: torch.Tensor,
    turn_indices: torch.Tensor,
    embed_text: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_turns_per_batch: int = 32,
    max_frames_per_batch: int = 8192,
    speaker_reference_features: torch.Tensor | None = None,
    speaker_reference_lens: torch.Tensor | None = None,
) -> TurnAlignmentOutput:
    """Align compact text turns and restore MAS positions to the conversation grid.

    ``turn_indices`` has rows ``(sample, start, end_exclusive)`` on the acoustic
    grid. Text contains content tokens only: the caller owns BOS/EOS handling,
    channel delays and autoregressive shifts. Acoustic inputs and the existing
    text embeddings are detached from the TTS generator; the likelihood still
    trains every aligner parameter.

    Turns are sorted by audio length and packed under a padded-frame budget.
    No acoustic-score tensor is transferred to CPU. Only the small index/length
    table is copied once to organize batches. Invalid turns are excluded from
    likelihood and attention supervision. A single turn longer than the frame
    budget raises rather than truncating speech.

    Detached MAS positions restart from zero in each compact transcript;
    invalid frames contain -1. These are positions, not vocabulary token IDs.

    Optional speaker references use the conversation batch axis, rather than
    the compact turn axis. The learnable encoder pools each reference once;
    turn packs gather its vector using the original sample index. Reference
    features are detached, while the encoder and flow conditioning learn from
    the acoustic likelihood. Missing references use zero conditioning.
    """
    if audio_features.ndim != 3:
        raise ValueError('Expected audio_features [B,T,C]')
    if text_tokens.ndim != 2 or text_lens.shape != (text_tokens.size(0),):
        raise ValueError('Expected compact text_tokens [turns,N] and text_lens [turns]')
    if turn_indices.shape != (text_tokens.size(0), 3):
        raise ValueError('turn_indices must have shape [turns,3]')
    if max_turns_per_batch < 1 or max_frames_per_batch < 1:
        raise ValueError('Turn and frame batch limits must be positive')
    if text_lens.dtype not in (torch.int32, torch.int64) or turn_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError('Text lengths and turn indices must contain integers')
    if text_tokens.device != audio_features.device:
        raise ValueError('Audio and compact text must be on the same device')

    metadata = torch.cat([turn_indices, text_lens.to(turn_indices.device).unsqueeze(1)], dim=1).detach().cpu().tolist()
    batch_size, audio_steps, channels = audio_features.shape
    valid_rows = []
    for row, (sample, start, end, num_text) in enumerate(metadata):
        if (
            0 <= sample < batch_size
            and 0 <= start < end <= audio_steps
            and 0 < num_text <= text_tokens.size(1)
            and end - start >= num_text
        ):
            if end - start > max_frames_per_batch:
                raise ValueError(
                    f'Turn {row} has {end - start} frames, exceeding max_frames_per_batch={max_frames_per_batch}'
                )
            valid_rows.append(row)
    # Overlapping turns would make restoration depend on the sorting order.
    previous_ends = {}
    for row in sorted(valid_rows, key=lambda index: metadata[index][:2]):
        sample, start, end, _ = metadata[row]
        if start < previous_ends.get(sample, 0):
            raise ValueError('Alignment turn spans must not overlap within a sample')
        previous_ends[sample] = end
    valid_rows.sort(key=lambda index: metadata[index][2] - metadata[index][1])
    batches = _turn_batches(valid_rows, metadata, max_turns_per_batch, max_frames_per_batch)

    token_indices = text_tokens.new_full((batch_size, audio_steps), -1)
    weighted_loss = audio_features.new_zeros((), dtype=torch.float32)
    counts = torch.zeros(4, dtype=torch.long, device=audio_features.device)
    speaker_embeddings = None
    if getattr(aligner, 'use_speaker_conditioning', False):
        if (speaker_reference_features is None) != (speaker_reference_lens is None):
            raise ValueError('Speaker reference features and lengths must be provided together')
        if speaker_reference_features is not None:
            if speaker_reference_features.size(0) != batch_size:
                raise ValueError('Speaker references must use the conversation batch dimension')
            speaker_embeddings = aligner.encode_speaker_reference(
                speaker_reference_features.detach(), speaker_reference_lens
            )
    # Counts: valid turns, one-frame text tokens, text tokens, acoustic frames.
    for batch_rows in batches:
        max_audio = max(metadata[row][2] - metadata[row][1] for row in batch_rows)
        max_text = max(metadata[row][3] for row in batch_rows)
        packed_audio = audio_features.new_zeros(len(batch_rows), max_audio, channels)
        audio_lengths = []
        for packed_row, row in enumerate(batch_rows):
            sample, start, end, _ = metadata[row]
            packed_audio[packed_row, : end - start] = audio_features[sample, start:end].detach()
            audio_lengths.append(end - start)
        row_indices = torch.tensor(batch_rows, device=text_tokens.device, dtype=torch.long)
        packed_text = text_tokens.index_select(0, row_indices)[:, :max_text]
        packed_text_lens = text_lens.to(audio_features.device).index_select(0, row_indices)
        packed_audio_lens = torch.tensor(audio_lengths, device=audio_features.device, dtype=torch.long)
        with torch.no_grad():
            text_features = embed_text(packed_text, packed_text_lens).detach()
        speaker_kwargs = {}
        if speaker_embeddings is not None:
            sample_indices = torch.tensor(
                [metadata[row][0] for row in batch_rows], device=audio_features.device, dtype=torch.long
            )
            speaker_kwargs['speaker_embeddings'] = speaker_embeddings.index_select(0, sample_indices)
        result = aligner(packed_audio, text_features, packed_audio_lens, packed_text_lens, **speaker_kwargs)
        valid = result.valid_turns.to(dtype=torch.bool, device=audio_features.device)
        frame_count = (packed_audio_lens * valid).sum()
        weighted_loss = weighted_loss + result.loss * frame_count
        counts[0] += valid.sum()
        counts[1] += ((result.durations == 1) & valid[:, None]).sum()
        counts[2] += (packed_text_lens * valid).sum()
        counts[3] += frame_count
        for packed_row, row in enumerate(batch_rows):
            sample, start, end, _ = metadata[row]
            path = result.token_indices[packed_row, : end - start]
            valid_frames = valid[packed_row] & (path >= 0)
            token_indices[sample, start:end] = torch.where(valid_frames, path, -1)

    loss = weighted_loss / counts[3].clamp_min(1)
    if not batches:
        loss = loss + _zero_parameter_loss(aligner, audio_features)
    return TurnAlignmentOutput(
        loss=loss,
        valid_turn_count=counts[0],
        turn_count=torch.tensor(text_tokens.size(0), device=audio_features.device, dtype=torch.long),
        one_frame_tokens=counts[1],
        token_count=counts[2],
        token_indices=token_indices,
    )


def decode_alignment_features(model, codes, lengths):
    """Frozen original-codec latents, concatenated across each acoustic stack."""
    if codes.ndim != 3 or lengths.shape != (codes.size(0),):
        raise ValueError('Expected original codec codes [B,C,T] and lengths [B]')
    with torch.no_grad(), torch.autocast(device_type=codes.device.type, enabled=False):
        if codes.size(2) == 0:
            return torch.zeros(codes.size(0), 0, model.text_aligner.input_audio_channels, device=codes.device)
        mask = valid_original_codec_frames(model, codes, lengths)
        safe = codes.long().masked_fill(~mask[:, None, :], 0)
        features = model._codec_model.vector_quantizer.decode(indices=safe.transpose(0, 1), input_len=lengths).float()
        features = features.masked_fill(~mask[:, None, :], 0.0)
        stack = model.frame_stacking_factor
        features = F.pad(features, (0, (-features.size(2)) % stack))
        batch, channels, frames = features.shape
        return (
            features.view(batch, channels, frames // stack, stack)
            .permute(0, 2, 1, 3)
            .reshape(batch, frames // stack, channels * stack)
        )


def valid_original_codec_frames(model, codes, lengths):
    """Exclude controls using the original codec vocabulary, before conversion."""
    size = int(model._codec_model.codebook_size)
    within = torch.arange(codes.size(2), device=codes.device)[None, :] < lengths[:, None]
    return within & ((codes >= 0) & (codes < size)).all(dim=1)


def compact_speaker_reference(model, codes, lengths):
    """Pool real reference frames only; missing-reference control pairs have none."""
    valid = valid_original_codec_frames(model, codes, lengths)
    # With codec index conversion enabled the dummy IDs use the *new* vocabulary
    # and may numerically overlap legal original codec IDs. Match the complete
    # two-frame sentinel instead of treating every such integer as a control.
    if codes.size(2) >= 2:
        dummy = (
            lengths.eq(2)
            & codes[:, :, 0].eq(model.context_audio_bos_id).all(1)
            & codes[:, :, 1].eq(model.context_audio_eos_id).all(1)
        )
        valid = valid & ~dummy[:, None]
    counts = valid.sum(1)
    width = int(counts.max().item()) if counts.numel() else 0
    packed = codes.new_zeros(codes.size(0), codes.size(1), width)
    for row in range(codes.size(0)):
        selected = codes[row, :, valid[row]]
        packed[row, :, : selected.size(1)] = selected
    return packed, counts.to(dtype=torch.long, device=lengths.device)


@torch.no_grad()
def maximum_path_indices(log_scores, text_lens, audio_lens, backend='auto'):
    """Find the maximum-score monotonic, non-skipping path for every turn.

    Args:
        log_scores: Floating-point acoustic scores, shaped ``(B, T, N)``.
            Scores are accumulated in float32, including for bf16/fp16 inputs.
        text_lens: Integer text lengths, shaped ``(B,)``.
        audio_lens: Integer audio lengths, shaped ``(B,)``.
        backend: ``auto``, ``cpu`` or ``triton``. Auto uses Triton for CUDA
            tensors when available and padded text length is at most 4096;
            otherwise it uses the existing CPU implementation. Explicit
            ``triton`` requests raise rather than silently falling back.

    Returns:
        Long tensor shaped ``(B, T)`` on the score device, containing one text
        position per frame. Padding and invalid turns are ``-1``. Zero-length,
        out-of-bounds and ``T < N`` turns are invalid. A path with a nonfinite
        total score is also invalid. Every valid path starts at position zero,
        ends at ``N - 1`` and advances by zero or one at each frame. Ties choose
        advancement during traceback, matching NeMo's ``mas_width1``.

    This discrete search is intentionally nondifferentiable. Gather the selected
    scores from the original tensor to train the acoustic likelihood.
    """
    if log_scores.ndim != 3 or not torch.is_floating_point(log_scores):
        raise ValueError('log_scores must be a floating-point tensor shaped (B, T, N)')
    batch_size, max_audio, max_text = log_scores.shape
    if text_lens.shape != (batch_size,) or audio_lens.shape != (batch_size,):
        raise ValueError('text_lens and audio_lens must have shape (B,)')
    if text_lens.dtype not in (torch.int32, torch.int64) or audio_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError('text_lens and audio_lens must contain integers')
    if backend not in ('auto', 'cpu', 'triton'):
        raise ValueError(f'Unsupported MAS backend: {backend!r}')
    use_triton = log_scores.is_cuda and TRITON_AVAILABLE and max_text <= 4096
    if backend == 'triton' and not use_triton:
        raise ValueError('Triton MAS requires CUDA, Triton, and padded text length <= 4096')
    path = torch.full((batch_size, max_audio), -1, device=log_scores.device, dtype=torch.long)
    if batch_size == 0 or max_audio == 0 or max_text == 0:
        return path
    if backend == 'cpu' or not use_triton:
        return _maximum_path_cpu(log_scores, text_lens, audio_lens, path)

    text_lens = text_lens.to(device=log_scores.device).contiguous()
    audio_lens = audio_lens.to(device=log_scores.device).contiguous()
    decisions = torch.empty(log_scores.shape, device=log_scores.device, dtype=torch.uint8)
    valid = torch.empty(batch_size, device=log_scores.device, dtype=torch.uint8)
    block_text = triton.next_power_of_2(max_text)
    _mas_forward[(batch_size,)](
        log_scores,
        text_lens,
        audio_lens,
        decisions,
        valid,
        *log_scores.stride(),
        MAX_AUDIO=max_audio,
        MAX_TEXT=max_text,
        BLOCK_TEXT=block_text,
        num_warps=4 if block_text >= 128 else 1,
    )
    _mas_traceback[(batch_size,)](
        decisions, text_lens, audio_lens, valid, path, MAX_AUDIO=max_audio, MAX_TEXT=max_text, num_warps=1
    )
    return path


class _TransformerTextPrior(nn.Module):
    """Project existing text features through a small relative-position encoder."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        num_heads,
        filter_channels,
        kernel_size,
        dropout,
        window_size,
    ):
        super().__init__()
        self.input_projection = ConvNorm(in_channels, hidden_channels, kernel_size=1)
        self.encoder = RelativePositionTransformer(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=num_heads,
            n_layers=num_layers,
            kernel_size=kernel_size,
            p_dropout=dropout,
            window_size=window_size,
        )
        self.output_projection = ConvNorm(hidden_channels, out_channels, kernel_size=1)

    def forward(self, features, mask):
        hidden = self.input_projection(features, mask) * mask
        hidden = self.encoder(hidden, mask)
        return self.output_projection(hidden, mask) * mask


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
        speaker_channels=0,
    ):
        super().__init__()
        if channels < 2 or hidden_channels < 1 or num_flows < 1:
            raise ValueError('Acoustic channels must be >= 2; hidden width and flow count must be positive')
        self.actnorms = nn.ModuleList([_ActNorm(channels, max_log_scale) for _ in range(num_flows)])
        self.mixers = nn.ModuleList([_InvertibleChannelMix(channels, flow_n_split) for _ in range(num_flows)])
        self.couplings = nn.ModuleList(
            [
                _ResidualCoupling(
                    channels,
                    hidden_channels,
                    coupling_num_layers,
                    coupling_kernel_size,
                    max_log_scale,
                    speaker_channels,
                )
                for _ in range(num_flows)
            ]
        )

    def forward(self, inputs, mask, reverse=False, speaker_embedding=None):
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
                if layer is self.couplings[index]:
                    hidden, layer_log_det = layer(hidden, mask, reverse=reverse, speaker_embedding=speaker_embedding)
                else:
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

    def __init__(self, channels, hidden_channels, num_layers=4, kernel_size=5, max_log_scale=None, speaker_channels=0):
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
        self.condition_projection = (
            nn.Linear(speaker_channels, 2 * hidden_channels * num_layers, bias=False) if speaker_channels else None
        )

    def forward(self, inputs, mask, reverse=False, speaker_embedding=None):
        first, second = inputs.split([self.first_channels, self.second_channels], dim=1)
        hidden = self.input_projection(first, mask) * mask
        skip = torch.zeros_like(hidden)
        condition = None
        if self.condition_projection is not None:
            if speaker_embedding is None:
                speaker_embedding = inputs.new_zeros((inputs.size(0), self.condition_projection.in_features))
            if speaker_embedding.shape != (inputs.size(0), self.condition_projection.in_features):
                raise ValueError('Speaker embedding must have shape [batch, speaker_channels]')
            condition = self.condition_projection(speaker_embedding).reshape(
                inputs.size(0), self.num_layers, 2 * self.hidden_channels, 1
            )
        for index, (temporal, residual_skip) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            preactivation = temporal(hidden, mask)
            if condition is not None:
                preactivation = preactivation + condition[:, index]
            first_gate, second_gate = (preactivation * mask).chunk(2, dim=1)
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


def _read_checkpoint(source_path):
    if Path(source_path).suffix.lower() == '.nemo':
        with tarfile.open(source_path, 'r:*') as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and Path(member.name).name == 'model_weights.ckpt'
            ]
            if len(members) != 1:
                raise ValueError(f'Checkpoint {source_path!r} must contain one unsharded model_weights.ckpt.')
            with archive.extractfile(members[0]) as stream:
                state = torch.load(stream, map_location='cpu', weights_only=True)
    else:
        state = torch.load(source_path, map_location='cpu', weights_only=False)
    if isinstance(state, Mapping) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, Mapping):
        raise ValueError(f'Checkpoint {source_path!r} does not contain a model state dictionary.')
    return state, None


def _select_aligner_weights(state, expected, source_path):
    normalized = {}
    wrappers = ('model.', 'module.', '_orig_mod.')
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError(f'Checkpoint {source_path!r} contains a non-string state dictionary key.')
        while key.startswith(wrappers):
            key = key.split('.', 1)[1]
        if key in normalized:
            raise ValueError(f'Checkpoint {source_path!r} contains duplicate unwrapped state key {key!r}.')
        normalized[key] = value
    namespaces = [
        {key[len(prefix) :]: value for key, value in normalized.items() if key.startswith(prefix)}
        for prefix in ('text_aligner.', 'alignment.text_aligner.')
    ]
    namespaces = [values for values in namespaces if values]
    if len(namespaces) > 1:
        raise ValueError(f'Checkpoint {source_path!r} contains multiple text_aligner namespaces.')
    if namespaces:
        return namespaces[0]
    if expected.keys() & normalized.keys():
        return normalized
    raise ValueError(f'Checkpoint {source_path!r} contains no text_aligner weights.')


def _turn_batches(rows, metadata, max_turns, max_frames):
    batches, current = [], []
    max_audio = 0
    for row in rows:
        audio_length = metadata[row][2] - metadata[row][1]
        next_max = max(max_audio, audio_length)
        if current and (len(current) >= max_turns or (len(current) + 1) * next_max > max_frames):
            batches.append(current)
            current, max_audio = [], 0
        current.append(row)
        max_audio = max(max_audio, audio_length)
    if current:
        batches.append(current)
    return batches


def _zero_parameter_loss(module, reference):
    return sum(
        (parameter.sum() * 0.0 for parameter in module.parameters()), reference.new_zeros((), dtype=torch.float32)
    )


def _maximum_path_cpu(log_scores, text_lens, audio_lens, path):
    from nemo.collections.tts.parts.utils.helpers import b_mas

    scores = log_scores.detach().float().cpu().numpy()
    text_lengths = text_lens.detach().cpu().numpy()
    audio_lengths = audio_lens.detach().cpu().numpy()
    batch_size, max_audio, max_text = scores.shape
    valid = (
        (text_lengths > 0)
        & (text_lengths <= max_text)
        & (audio_lengths >= text_lengths)
        & (audio_lengths <= max_audio)
    )
    # The legacy width-one traceback assumes at least two text positions. A
    # one-token turn has exactly one path and needs no dynamic programming.
    single = valid & (text_lengths == 1)
    path_cpu = np.full((batch_size, max_audio), -1, dtype=np.int64)
    for index in np.flatnonzero(single):
        length = audio_lengths[index]
        if np.isfinite(scores[index, :length, 0].sum(dtype=np.float32)):
            path_cpu[index, :length] = 0
    indices = np.flatnonzero(valid & ~single)
    if indices.size:
        selected_scores = np.ascontiguousarray(scores[indices, None])
        selected_text_lens = np.ascontiguousarray(text_lengths[indices])
        selected_audio_lens = np.ascontiguousarray(audio_lengths[indices])
        hard_path = b_mas(selected_scores, selected_text_lens, selected_audio_lens, width=1)
        for local_index, batch_index in enumerate(indices):
            length = audio_lengths[batch_index]
            positions = hard_path[local_index, 0, :length].argmax(axis=-1)
            total = scores[batch_index, np.arange(length), positions].sum(dtype=np.float32)
            if np.isfinite(total):
                path_cpu[batch_index, :length] = positions
    path.copy_(torch.from_numpy(path_cpu).to(device=path.device))
    return path


if TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=['STRIDE_BATCH', 'STRIDE_AUDIO', 'STRIDE_TEXT', 'MAX_AUDIO', 'MAX_TEXT'])
    def _mas_forward(
        scores,
        text_lens,
        audio_lens,
        decisions,
        valid,
        STRIDE_BATCH,
        STRIDE_AUDIO,
        STRIDE_TEXT,
        MAX_AUDIO,
        MAX_TEXT,
        BLOCK_TEXT: tl.constexpr,
    ):
        batch = tl.program_id(0)
        num_text = tl.load(text_lens + batch).to(tl.int32)
        num_audio = tl.load(audio_lens + batch).to(tl.int32)
        turn_valid = (num_text > 0) & (num_text <= MAX_TEXT) & (num_audio >= num_text) & (num_audio <= MAX_AUDIO)
        tl.store(valid + batch, 0)
        if turn_valid:
            token = tl.arange(0, BLOCK_TEXT)
            previous = tl.where(token == 0, 0.0, -float('inf'))
            for frame in range(num_audio):
                advanced = tl.gather(previous, tl.maximum(token - 1, 0), axis=0)
                advanced = tl.where(token > 0, advanced, -float('inf'))
                moved = (advanced >= previous) & (token > 0)
                score = tl.load(
                    scores + batch * STRIDE_BATCH + frame * STRIDE_AUDIO + token * STRIDE_TEXT,
                    mask=token < num_text,
                    other=-float('inf'),
                ).to(tl.float32)
                current = tl.maximum(previous, advanced) + score
                reachable = (token < num_text) & (token <= frame) & (num_text - 1 - token <= num_audio - 1 - frame)
                previous = tl.where(reachable, current, -float('inf'))
                tl.store(
                    decisions + (batch * MAX_AUDIO + frame) * MAX_TEXT + token,
                    moved.to(tl.uint8),
                    mask=token < num_text,
                )
            final_score = tl.sum(tl.where(token == num_text - 1, previous, 0.0), axis=0)
            tl.store(valid + batch, (final_score > -float('inf')) & (final_score < float('inf')))

    @triton.jit(do_not_specialize=['MAX_AUDIO', 'MAX_TEXT'])
    def _mas_traceback(
        decisions,
        text_lens,
        audio_lens,
        valid,
        path,
        MAX_AUDIO,
        MAX_TEXT,
    ):
        batch = tl.program_id(0)
        if tl.load(valid + batch):
            token = tl.load(text_lens + batch).to(tl.int32) - 1
            num_audio = tl.load(audio_lens + batch).to(tl.int32)
            for step in range(num_audio):
                frame = num_audio - 1 - step
                tl.store(path + batch * MAX_AUDIO + frame, token)
                moved = tl.load(decisions + (batch * MAX_AUDIO + frame) * MAX_TEXT + token).to(tl.int32)
                token -= moved
