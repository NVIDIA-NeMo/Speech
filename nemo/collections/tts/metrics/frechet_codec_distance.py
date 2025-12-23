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

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn
from torchmetrics.image.fid import FrechetInceptionDistance

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.tts.models import AudioCodecModel
from nemo.utils import logging


class CodecEmbedder(nn.Module):
    def __init__(self, codec: AudioCodecModel):
        super().__init__()
        self.codec = codec

    def forward(self, x: Tensor) -> Tensor:
        """
        Embeds a batch of audio codec codes into the codec's (dequantized) embedding space.
        """
        # x: (B*T, C)
        x_len = torch.tensor(x.shape[0], device=x.device, dtype=torch.long).unsqueeze(0)  # (1, 1)
        # pretend it's one huge batch element, since codec requires (B, C, T) input and
        # we don't have the per-batch-element lengths at this point due to FID API limitations
        tokens = x.permute(1, 0).unsqueeze(0)  # 1, C, B*T
        embeddings = self.codec.dequantize(tokens=tokens, tokens_len=x_len)  # (B, D, T)
        # we treat each time step as a separate example
        embeddings = rearrange(embeddings, 'B D T -> (B T) D')
        return embeddings

    @property
    def num_features(self) -> int:
        return self.codec.vector_quantizer.codebook_dim


class FrechetCodecDistance(FrechetInceptionDistance):
    def __init__(self, codec_name: str):
        if codec_name.endswith(".nemo"):
            codec = AudioCodecModel.restore_from(codec_name, strict=False)
        elif codec_name.startswith("nvidia/"):
            # HuggingFace or NGC model name
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

    def encode_from_file(self, audio_path: str) -> Tensor:
        """
        Encodes an audio file into audio codec codes.
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
        if codes.numel() == 0:
            logging.warning(f"FCD: No valid codes to update, skipping update")
            return
        if codes.shape[1] != self.codec.num_codebooks:
            logging.warning(
                f"FCD: Number of codebooks mismatch: {codes.shape[1]} != {self.codec.num_codebooks}, skipping update"
            )
            return
        # keep only valid codes
        codes_batch_all = []
        for batch_idx in range(codes.shape[0]):
            codes_batch = codes[batch_idx, :, : codes_len[batch_idx]]  # (C, T)
            codes_batch_all.append(codes_batch)
        # combine into a single tensor. We treat each timestep independently so we can concatenate them all.
        codes_batch_all = torch.cat(codes_batch_all, dim=-1).permute(1, 0)  # (B*T, C)
        if len(codes_batch_all) == 0:
            logging.warning(f"FCD: No valid codes to update, skipping update")
            return
        # update
        super().update(codes_batch_all, real=is_real)
        self.updated_since_last_reset = True

    def reset(self):
        super().reset()
        self.updated_since_last_reset = False

    def update_from_audio_file(self, audio_path: str, is_real: bool):
        codes, codes_len = self.encode_from_file(audio_path=audio_path)
        self.update(codes=codes, codes_len=codes_len, is_real=is_real)

    def compute(self) -> Tensor:
        if not self.updated_since_last_reset:
            logging.warning(f"FCD: No updates since last reset, returning 0")
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
