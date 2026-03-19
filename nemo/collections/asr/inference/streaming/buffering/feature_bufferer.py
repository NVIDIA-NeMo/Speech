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

import torch
from omegaconf import DictConfig

from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer
from nemo.collections.asr.inference.utils.constants import LOG_MEL_ZERO


class FeatureBufferer:
    """
    Feature bufferer class
    It buffers the feature chunks and maintains the buffer.
    """

    def __init__(
        self,
        sample_rate: int,
        buffer_size_in_secs: float,
        preprocessor_cfg: DictConfig,
        device: torch.device,
        fill_value: float = LOG_MEL_ZERO,
    ):
        """
        Args:
            sample_rate (int): sample rate
            buffer_size_in_secs (float): buffer size in seconds
            preprocessor_cfg (DictConfig): preprocessor config
            device (torch.device): device
            fill_value (float): value to fill the feature buffer with
        """
        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.device = device

        if hasattr(preprocessor_cfg, 'log') and preprocessor_cfg.log:
            self.ZERO_LEVEL_SPEC_DB_VAL = LOG_MEL_ZERO
        else:
            self.ZERO_LEVEL_SPEC_DB_VAL = fill_value

        self.n_feat = preprocessor_cfg.features
        self.feature_buffer_len = int(buffer_size_in_secs / preprocessor_cfg.window_stride)
        self.feature_buffer = torch.full(
            [self.n_feat, self.feature_buffer_len],
            self.ZERO_LEVEL_SPEC_DB_VAL,
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self) -> None:
        """
        Reset the buffer to zero
        """
        self.feature_buffer.fill_(self.ZERO_LEVEL_SPEC_DB_VAL)

    def update(self, fbuffer: FeatureBuffer) -> None:
        """
        Replace feature buffer with new data
        Args:
            fbuffer (FeatureBuffer): feature buffer to update
        """
        # Resize if needed (optional)
        if fbuffer.size != self.feature_buffer.shape[1]:
            self.feature_buffer = torch.full(
                [self.n_feat, fbuffer.size],
                self.ZERO_LEVEL_SPEC_DB_VAL,
                dtype=torch.float32,
                device=self.device,
            )

        self.feature_buffer.copy_(fbuffer.features)

    def get_feature_buffer(self) -> torch.Tensor:
        """
        Get the current feature buffer
        Returns:
            torch.Tensor: current state of the feature buffer
        """
        return self.feature_buffer.clone()


class BatchedFeatureBufferer:
    """
    Batched feature bufferer class
    It buffers the feature chunks from multiple streams and maintains the buffers.
    """

    def __init__(
        self,
        sample_rate: int,
        buffer_size_in_secs: float,
        preprocessor_cfg: DictConfig,
        device: torch.device,
    ):
        """
        Args:
            sample_rate (int): sample rate
            buffer_size_in_secs (float): buffer size in seconds
            preprocessor_cfg (DictConfig): preprocessor config
            device (torch.device): device
        """
        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.preprocessor_cfg = preprocessor_cfg
        self.device = device
        self.bufferers = {}

    def reset(self) -> None:
        """Reset bufferers"""
        self.bufferers = {}

    def rm_bufferer(self, stream_id: int) -> None:
        """
        Remove bufferer for the given stream id
        Args:
            stream_id (int): stream id
        """
        self.bufferers.pop(stream_id, None)

    def update(self, fbuffers: list[FeatureBuffer]) -> list[torch.Tensor]:
        """
        Update the feature bufferers with the new feature buffers.
        Feature buffers can come from different streams (audios), so we need to maintain a bufferer for each stream.
        Args:
            fbuffers (list[FeatureBuffer]): list of feature buffers
        Returns:
            list[torch.Tensor]: list of feature buffers, one per input frame
        """
        result_buffers = []
        for fbuffer in fbuffers:
            bufferer = self.bufferers.get(fbuffer.stream_id, None)

            if bufferer is None:
                bufferer = FeatureBufferer(
                    self.sample_rate,
                    self.buffer_size_in_secs,
                    self.preprocessor_cfg,
                    self.device,
                )
                self.bufferers[fbuffer.stream_id] = bufferer

            bufferer.update(fbuffer)
            result_buffers.append(bufferer.get_feature_buffer())

            if fbuffer.is_last:
                self.rm_bufferer(fbuffer.stream_id)

        return result_buffers
