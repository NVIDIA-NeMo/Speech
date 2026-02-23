# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Mimi audio codec encoder wrapper with multi-codebook delay pattern."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MimiEncoder(nn.Module):
    """
    Wraps HuggingFace MimiModel for audio encoding with multi-codebook support
    and delay pattern.
    """

    FRAME_RATE = 12.5  # Hz
    FRAME_SHIFT = 0.08  # seconds (80ms)
    SAMPLE_RATE = 24000  # Mimi native sample rate
    CODEBOOK_SIZE = 2048
    NUM_CODEBOOKS = 8  # Mimi default (can use fewer)

    def __init__(self, pretrained_model: str = "kyutai/mimi", num_codebooks: int = 8):
        super().__init__()
        from transformers import AutoFeatureExtractor, MimiModel

        self.model = MimiModel.from_pretrained(pretrained_model)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model)
        self.num_codebooks = num_codebooks
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    @property
    def token_equivalent_duration(self) -> float:
        return self.FRAME_SHIFT

    @property
    def codebook_size(self) -> int:
        return self.CODEBOOK_SIZE

    @torch.no_grad()
    def encode(self, audio: Tensor, audio_lens: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode raw audio to discrete codes.

        Args:
            audio: (B, T_samples) at 24kHz (resample externally if needed)
            audio_lens: (B,) sample counts

        Returns:
            codes: (B, num_codebooks, T_frames) discrete codebook indices
            code_lens: (B,) frame counts
        """
        B, T = audio.shape
        # HF MimiModel.encode expects (B, channels, T) for both input and mask
        audio_3d = audio.unsqueeze(1)  # (B, 1, T)
        # Build padding mask from audio_lens (1 = valid, 0 = masked)
        padding_mask = (
            torch.arange(T, device=audio.device).unsqueeze(0) < audio_lens.unsqueeze(1)
        ).unsqueeze(1).float()  # (B, 1, T)
        encoder_outputs = self.model.encode(audio_3d, padding_mask)
        codes = encoder_outputs.audio_codes[:, :self.num_codebooks, :]
        actual_T = codes.shape[2]
        code_lens = torch.clamp((audio_lens / (self.SAMPLE_RATE * self.FRAME_SHIFT)).long(), max=actual_T)
        return codes, code_lens

    @staticmethod
    def apply_delay_pattern(
        codes: Tensor, code_lens: Tensor, pad_value: int = 2048
    ) -> Tensor:
        """
        Apply delay pattern to multi-codebook codes.

        Codebook k is delayed by k frames. Padding positions use pad_value
        (= codebook_size, used as padding_idx in embedding).

        Args:
            codes: (B, K, T) original codes
            code_lens: (B,) lengths

        Returns:
            delayed: (B, K, T) with delay pattern applied
        """
        B, K, T = codes.shape
        delayed = codes.new_full((B, K, T), fill_value=pad_value)
        for k in range(K):
            if k < T:
                delayed[:, k, k:] = codes[:, k, : T - k]
        # Mask out padding positions per example
        for b in range(B):
            delayed[b, :, code_lens[b] :] = pad_value
        return delayed
