# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture-specific acoustic-code predictors for EasyMagpie serving."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from easymagpie_vllm_omni.config import NUM_SPECIAL_AUDIO_TOKENS, EasyMagpieOmniArch
from easymagpie_vllm_omni.local_transformer import EasyMagpieLocalTransformerPredictor
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.nemotron_h import NemotronHAttentionDecoderLayer, NemotronHMLPDecoderLayer

_DEFAULT_TOP_K = 80
_MIN_SAMPLING_TEMPERATURE = 1e-4


class CodecLatentAudioEmbedder(nn.Module):
    """Embed stacked FSQ indices exactly like NemotronTTS v2."""

    def __init__(self, arch: EasyMagpieOmniArch) -> None:
        super().__init__()
        self.num_codebooks = arch.num_stacked_codebooks
        self.codebook_size = arch.codebook_size
        self.embedding_dim = arch.embedding_dim
        self.latent_dim = len(arch.codec_num_levels_per_group)

        levels = torch.tensor(arch.codec_num_levels_per_group, dtype=torch.long)
        bases = torch.cumprod(torch.tensor([1, *arch.codec_num_levels_per_group[:-1]]), dim=0)
        self.register_buffer("levels", levels, persistent=False)
        self.register_buffer("bases", bases, persistent=False)
        self.register_buffer("channels", torch.arange(self.num_codebooks).view(1, -1), persistent=False)

        self.audio_code_projection = nn.Linear(self.num_codebooks * self.latent_dim, self.embedding_dim)
        self.audio_special_embeddings = nn.Embedding(
            self.num_codebooks * NUM_SPECIAL_AUDIO_TOKENS,
            self.embedding_dim,
        )

    def _decode(self, codes: torch.Tensor) -> torch.Tensor:
        nonnegative = torch.div(codes.unsqueeze(-1), self.bases, rounding_mode="floor") % self.levels
        scale = torch.div(self.levels, 2, rounding_mode="floor")
        return ((nonnegative - scale) / scale).to(self.audio_code_projection.weight.dtype)

    def _special_rows(self, channels: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        special_ids = (codes - self.codebook_size).clamp(min=0, max=NUM_SPECIAL_AUDIO_TOKENS - 1)
        return channels * NUM_SPECIAL_AUDIO_TOKENS + special_ids

    def embed_audio_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed every stacked codebook in one serving frame."""
        if codes.shape[-1] != self.num_codebooks:
            raise ValueError(f"expected {self.num_codebooks} code channels, got {codes.shape[-1]}")
        codes = codes.long()
        is_special = codes >= self.codebook_size
        latent = self._decode(codes.clamp(max=self.codebook_size - 1)).masked_fill(is_special.unsqueeze(-1), 0.0)
        embedded = self.audio_code_projection(latent.flatten(start_dim=1))

        special = self.audio_special_embeddings(self._special_rows(self.channels, codes))
        return embedded + (special * is_special.unsqueeze(-1)).sum(dim=1)

    def embed_codebooks(self, codes: torch.Tensor, codebook_indices: Sequence[int]) -> torch.Tensor:
        """Embed selected predictor channels using their slices of the shared projection."""
        if codes.shape[-1] != len(codebook_indices):
            raise ValueError(f"got {codes.shape[-1]} codes for {len(codebook_indices)} codebook indices")
        codes = codes.long()
        embedded = None
        for input_index, codebook_index in enumerate(codebook_indices):
            channel_codes = codes[:, input_index]
            is_special = (channel_codes >= self.codebook_size).unsqueeze(-1)
            latent = self._decode(channel_codes.clamp(max=self.codebook_size - 1)).masked_fill(is_special, 0.0)
            start = codebook_index * self.latent_dim
            stop = start + self.latent_dim
            weight = self.audio_code_projection.weight[:, start:stop]
            channel_embedded = torch.nn.functional.linear(latent, weight, self.audio_code_projection.bias)
            rows = self._special_rows(codebook_index, channel_codes)
            channel_embedded = channel_embedded + self.audio_special_embeddings(rows) * is_special
            embedded = channel_embedded if embedded is None else embedded + channel_embedded
        assert embedded is not None
        return embedded / len(codebook_indices)
