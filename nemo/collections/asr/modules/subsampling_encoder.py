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

from typing import Optional, Tuple
import torch
import torch.nn as nn
from nemo.collections.asr.parts.mixins.streaming import DummyCacheAwareStreamingMixin
from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking, StackingSubsampling


class SubsamplingEncoder(nn.Module, DummyCacheAwareStreamingMixin):
    def __init__(
        self,
        feat_in: int = 128,
        d_model: int = 512,
        feat_out: int = -1,
        subsampling: str = 'feature_stacking',
        subsampling_factor: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self._feat_in = feat_in
        self.subsampling = subsampling
        self.subsampling_factor = subsampling_factor
        if subsampling == 'feature_stacking':
            self.pre_encode = FeatureStacking(subsampling_factor, feat_in, d_model)
        elif subsampling and subsampling_factor > 1:
            if subsampling in ['stacking', 'stacking_norm']:
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    norm=True if subsampling == 'stacking_norm' else False,
                )
            else:
                raise ValueError(
                    f"subsampling='{subsampling}' is not supported. "
                    "Currently only 'feature_stacking', 'stacking', and 'stacking_norm' are available."
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        self._feat_out = d_model

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None

    def forward(
        self, audio_signal: torch.Tensor, length: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_signal: (B, D, T) input features (mel spectrogram).
            length: (B,) valid lengths per sample.
        Returns:
            x: (B, D', T') encoded representation (D' = d_model or feat_out).
            length: (B,) output lengths after subsampling.
        """
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),),
                audio_signal.size(-1),
                dtype=torch.int64,
                device=audio_signal.device,
            )

        if isinstance(self.pre_encode, FeatureStacking):
            x, length = self.pre_encode(audio_signal, length)  # (B, D, T) -> (B, T, D)
        else:
            x = torch.transpose(audio_signal, 1, 2)  # (B, D, T) -> (B, T, D)

        if isinstance(self.pre_encode, nn.Linear):
            x = self.pre_encode(x)
        elif not isinstance(self.pre_encode, FeatureStacking):
            x, length = self.pre_encode(x=x, lengths=length)

        length = length.to(torch.int64)

        if self.out_proj is not None:
            x = self.out_proj(x)
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        length = length.to(dtype=torch.int64)
        return x, length
