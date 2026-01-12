# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn
from torchmetrics.image.fid import FrechetInceptionDistance

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.tts.models import AudioCodecModel
from nemo.utils import logging


class CodecEmbedder(nn.Module):
    """
    Converts codec codes to dequantized codec embeddings.
    The class implements the right API to be used as a custom feature extractor
    provided to `torchmetrics.image.fid`.
    """

    def __init__(self, codec: AudioCodecModel):
        super().__init__()
        self.codec = codec

    def forward(self, x: Tensor) -> Tensor:
        """
        Embeds a batch of audio codes into the codec's (dequantized) embedding space.
        Each frame is treated independently.

        Args:
            x: Audio codes tensor of shape (B*T, C)

        Returns:
            Embeddings tensor of shape (B*T, D)
        """
        # We treat all frames as one large batch element, since the codec requires (B, C, T) input and
        # we don't have the per-batch-element lengths at this point due to FID API limitations

        # Consturct a length tensor: one batch element, all frames.
        x_len = torch.tensor(x.shape[0], device=x.device, dtype=torch.long).unsqueeze(0)  # (1, 1)
        tokens = x.permute(1, 0).unsqueeze(0)  # 1, C, B*T
        embeddings = self.codec.dequantize(tokens=tokens, tokens_len=x_len)  # (B, D, T)
        # we treat each time step as a separate example
        embeddings = rearrange(embeddings, 'B D T -> (B T) D')
        return embeddings

    @property
    def num_features(self) -> int:
        return self.codec.vector_quantizer.codebook_dim


class FrechetCodecDistance(FrechetInceptionDistance):
    """
    A metric that measures the Frechet Distance between a collection of real and
    generated codec frames. The distance is measured in the codec's embedding space,
    i.e. the continuous vectors obtained by dequantizing the codec frames. Each
    multi-codebook frame is treated as a separate example.

    We subclass `torchmetrics.image.fid.FrechetInceptionDistance` and use the codec
    embedder as a custom feature extractor.
    """

    def __init__(self, codec_name: str):
        """
        Initializes the FrechetCodecDistance metric.

        Args:
            codec_name: The name of the codec model to use.
                Can be a local .nemo file or a HuggingFace or NGC model.
                If the name ends with ".nemo", it is assumed to be a local .nemo file.
                Otherwise, it should start with "nvidia/", and is assumed to be a HuggingFace or NGC model.
        """
        if codec_name.endswith(".nemo"):
            # Local .nemo file
            codec = AudioCodecModel.restore_from(codec_name, strict=False)
        elif codec_name.startswith("nvidia/"):
            # Model on HuggingFace or NGC
            codec = AudioCodecModel.from_pretrained(codec_name)
        else:
            raise ValueError(
                f"Invalid codec name: {codec_name}. Must be a local .nemo file or a HuggingFace or NGC model name starting with 'nvidia/'"
            )
        codec.eval()
        feature = CodecEmbedder(codec)
        super().__init__(feature=feature)
        self.codec = codec
        self.updated_since_last_reset = False

    def _encode_audio_file(self, audio_path: str) -> Tuple[Tensor, Tensor]:
        """
        Encodes an audio file using the audio codec.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Tuple of tensors containing the codec codes and the lengths of the codec codes.
        """
        audio_segment = AudioSegment.from_file(audio_path, target_sr=self.codec.sample_rate)
        assert np.issubdtype(audio_segment.samples.dtype, np.floating)
        audio_min = audio_segment.samples.min()
        audio_max = audio_segment.samples.max()
        eps = 0.01  # certain ways of normalizing audio can result in samples that are slightly outside of [-1, 1]
        if audio_min < (-1.0 - eps) or audio_max > (1.0 + eps):
            logging.warning(f"Audio samples are not normalized: min={audio_min}, max={audio_max}")
        samples = torch.tensor(audio_segment.samples, device=self.codec.device).unsqueeze(0)
        audio_len = torch.tensor(samples.shape[1], device=self.codec.device).unsqueeze(0)
        codes, codes_len = self.codec.encode(audio=samples, audio_len=audio_len)
        return codes, codes_len

    def update(self, codes: Tensor, codes_len: Tensor, is_real: bool):
        """
        Updates the metric with a batch of codec frames.

        Args:
            codes: Tensor of shape (B, C, T) containing the codec codes.
            codes_len: Tensor of shape (B,) containing the lengths of the codec codes.
            is_real: Boolean indicating whether the codes are real or generated.
        """
        if codes.numel() == 0:
            logging.warning("FCD: No valid codes to update, skipping update")
            return
        if codes.shape[1] != self.codec.num_codebooks:
            logging.warning(
                f"FCD: Number of codebooks mismatch: {codes.shape[1]} != {self.codec.num_codebooks}, skipping update"
            )
            return

        # Keep only valid frames
        codes_batch_all = []
        for batch_idx in range(codes.shape[0]):
            codes_batch = codes[batch_idx, :, : codes_len[batch_idx]]  # (C, T)
            codes_batch_all.append(codes_batch)

        # Combine into a single tensor. We treat each frame independently so we can concatenate them all.
        codes_batch_all = torch.cat(codes_batch_all, dim=-1).permute(1, 0)  # (B*T, C)
        if len(codes_batch_all) == 0:
            logging.warning("FCD: No valid codes to update, skipping update")
            return

        # Update the metric
        super().update(codes_batch_all, real=is_real)
        self.updated_since_last_reset = True

    def reset(self):
        """
        Resets the metric. Should be called after each compute.
        """
        super().reset()
        self.updated_since_last_reset = False

    def update_from_audio_file(self, audio_path: str, is_real: bool):
        """
        Updates the metric with codes representing a single audio file.
        Uses the codec to encode the audio file into codec codes and updates the metric.

        Args:
            audio_path: Path to the audio file.
            is_real: Boolean indicating whether the audio file is real or generated.
        """
        codes, codes_len = self._encode_audio_file(audio_path=audio_path)
        self.update(codes=codes, codes_len=codes_len, is_real=is_real)

    def compute(self) -> Tensor:
        """
        Computes the Frechet Distance between the real and generated codec frame distributions.
        """
        if not self.updated_since_last_reset:
            logging.warning("FCD: No updates since last reset, returning 0")
            return torch.tensor(0.0, device=self.device)
        fcd = super().compute()
        min_allowed_fcd = -0.01  # a bit of tolerance for numerical issues
        fcd_value = fcd.cpu().item()
        if fcd_value < min_allowed_fcd:
            logging.warning(f"FCD value is negative: {fcd_value}")
            raise ValueError(f"FCD value is negative: {fcd_value}")
        # FCD should be non-negative
        fcd = fcd.clamp(min=0)
        return fcd
