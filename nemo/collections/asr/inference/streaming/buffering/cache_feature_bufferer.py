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


import math

import torch
from omegaconf import DictConfig

from nemo.collections.asr.inference.streaming.buffering.audio_bufferer import AudioBufferer
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.utils.constants import LOG_MEL_ZERO
from nemo.collections.asr.models import ASRModel


class CacheFeatureBufferer:
    """
    Cache feature bufferer class
    It buffers the feature chunks and maintains the buffer.
    """

    def __init__(
        self,
        sample_rate: int,
        buffer_size_in_secs: float,
        chunk_size_in_secs: float,
        preprocessor_cfg: DictConfig,
        device: torch.device,
        fill_value: float = LOG_MEL_ZERO,
        right_padding_ratio: float = 0.8,
    ):
        """
        Args:
            sample_rate (int): sample rate
            buffer_size_in_secs (float): buffer size in seconds
            chunk_size_in_secs (float): chunk size in seconds
            preprocessor_cfg (DictConfig): preprocessor config
            device (torch.device): device
            fill_value (float): value to fill the feature buffer with
            right_padding_ratio (float): what fraction of actual right padding of the last frame to use for padding mask,
                                         some models perform better with extra padding at the end of the audio
        """
        if buffer_size_in_secs < chunk_size_in_secs:
            raise ValueError(
                f"Buffer size ({buffer_size_in_secs}s) should be no less than chunk size ({chunk_size_in_secs}s)"
            )

        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.chunk_size_in_secs = chunk_size_in_secs
        self.device = device

        if hasattr(preprocessor_cfg, 'log') and preprocessor_cfg.log:
            self.ZERO_LEVEL_SPEC_DB_VAL = LOG_MEL_ZERO  # Log-Mel spectrogram value for zero signals
        else:
            self.ZERO_LEVEL_SPEC_DB_VAL = fill_value  # Custom fill value for the feature buffer

        self.n_feat = preprocessor_cfg.features
        self.timestep_duration = preprocessor_cfg.window_stride
        self.n_chunk_look_back = int(self.timestep_duration * self.sample_rate)
        self.chunk_size = int(self.chunk_size_in_secs * self.sample_rate)
        self.sample_buffer = AudioBufferer(sample_rate, buffer_size_in_secs)

        self.feature_buffer_len = int(buffer_size_in_secs / self.timestep_duration)
        self.feature_chunk_len = int(chunk_size_in_secs / self.timestep_duration)
        self.feature_buffer = torch.full(
            [self.n_feat, self.feature_buffer_len],
            self.ZERO_LEVEL_SPEC_DB_VAL,
            dtype=torch.float32,
            device=self.device,
        )

        self.preprocessor = ASRModel.from_config_dict(preprocessor_cfg)
        self.preprocessor.to(self.device)

        self.right_padding_ratio = right_padding_ratio
        self.right_padding = 0

    def reset(self) -> None:
        """
        Reset the buffer to zero
        """
        self.sample_buffer.reset()
        self.feature_buffer.fill_(self.ZERO_LEVEL_SPEC_DB_VAL)
        self.right_padding = 0

    def _update_feature_buffer(self, feat_chunk: torch.Tensor) -> None:
        """
        Add an extracted feature to `feature_buffer`
        Args:
            feat_chunk (torch.Tensor): feature chunk
        """
        self.feature_buffer[:, : -self.feature_chunk_len] = self.feature_buffer[:, self.feature_chunk_len :].clone()
        self.feature_buffer[:, -self.feature_chunk_len :] = feat_chunk.clone()

    def preprocess(
        self, audio_signal: torch.Tensor, right_padding: int = 0, expected_feat_len: int = None
    ) -> tuple[torch.Tensor, int]:
        """
        Preprocess the audio signal using the preprocessor
        Args:
            audio_signal (torch.Tensor): audio signal
            right_padding (int): right padding
            expected_feat_len (int): expected feature length
        Returns:
            torch.Tensor: preprocessed features
            int: right padding
        """
        sig_len = len(audio_signal)
        if right_padding > 0:
            right_padding = int(right_padding * self.right_padding_ratio)

        sig_len -= right_padding
        features, _ = self.preprocessor(
            input_signal=audio_signal.unsqueeze_(0).to(self.device),
            length=torch.tensor([sig_len], device=self.device),
        )

        if features.shape[2] > expected_feat_len:
            features = features[:, :, :expected_feat_len]

        features = features.squeeze()
        right_padding = math.floor(right_padding / self.sample_rate / self.timestep_duration)
        return features, right_padding

    def update(self, frame: Frame) -> None:
        """
        Update the sample and feature buffers with the new frame
        Args:
            frame (Frame): frame to update the buffer with
        """

        # Update the sample buffer with the new frame
        self.sample_buffer.update(frame)
        right_padding = frame.size - frame.valid_size

        plus_one = 0
        if math.isclose(self.buffer_size_in_secs, self.chunk_size_in_secs):
            # If the buffer size is equal to the chunk size, just take the whole buffer
            samples = self.sample_buffer.sample_buffer.clone()
        else:
            # Add look_back to have context for the first feature
            samples = self.sample_buffer.sample_buffer[-(self.n_chunk_look_back + self.chunk_size) :]
            plus_one = 1

        # Get the mel spectrogram
        features, right_padding = self.preprocess(
            samples, right_padding, expected_feat_len=self.feature_chunk_len + plus_one
        )

        # Update the feature buffer with the new features
        self._update_feature_buffer(features[:, -self.feature_chunk_len :])
        self.right_padding = right_padding

    def get_feature_buffer(self) -> torch.Tensor:
        """
        Get the current feature buffer
        Returns:
            torch.Tensor: current state of the feature buffer
        """
        return self.feature_buffer.clone()

    def get_right_padding(self) -> int:
        """
        Get the right padding
        Returns:
            int: right padding
        """
        return self.right_padding


class BatchedCacheFeatureBufferer:
    """
    Batched cache feature bufferer class
    It buffers the feature chunks from multiple streams and maintains the buffers.
    """

    def __init__(
        self,
        sample_rate: int,
        buffer_size_in_secs: float,
        chunk_size_in_secs: float,
        preprocessor_cfg: DictConfig,
        device: torch.device,
        right_padding_ratio: float = 0.8,
    ):
        """
        Args:
            sample_rate (int): sample rate
            buffer_size_in_secs (float): buffer size in seconds
            chunk_size_in_secs (float): chunk size in seconds
            preprocessor_cfg (DictConfig): preprocessor config
            device (torch.device): device
            right_padding_ratio (float): what fraction of actual right padding to use to create padding mask,
                                         some models perform better with extra padding at the end of the audio
        """

        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.bufferers = {}
        self.chunk_size_in_secs = chunk_size_in_secs
        self.preprocessor_cfg = preprocessor_cfg
        self.device = device
        self.right_padding_ratio = right_padding_ratio

    def reset(self) -> None:
        """
        Reset bufferers
        """
        self.bufferers = {}

    def rm_bufferer(self, stream_id: int) -> None:
        """
        Remove bufferer for the given stream id
        Args:
            stream_id (int): stream id
        """
        self.bufferers.pop(stream_id, None)

    def update(self, frames: list[Frame]) -> tuple[list[torch.Tensor], list[int]]:
        """
        Update the feature bufferers with the new frames.
        Frames can come from different streams (audios), so we need to maintain a bufferer for each stream.
        Args:
            frames (list[Frame]): list of frames
        Returns:
            tuple[list[torch.Tensor], list[int]]:
                feature_buffers: list of feature buffers, one per input frame
                right_paddings: list of right paddings, one per input frame
        """
        fbuffers = []
        right_paddings = []
        for frame in frames:
            bufferer = self.bufferers.get(frame.stream_id, None)

            if bufferer is None:
                bufferer = CacheFeatureBufferer(
                    sample_rate=self.sample_rate,
                    buffer_size_in_secs=self.buffer_size_in_secs,
                    chunk_size_in_secs=self.chunk_size_in_secs,
                    preprocessor_cfg=self.preprocessor_cfg,
                    device=self.device,
                    right_padding_ratio=self.right_padding_ratio,
                )
                self.bufferers[frame.stream_id] = bufferer

            bufferer.update(frame)
            fbuffers.append(bufferer.get_feature_buffer())
            right_paddings.append(bufferer.get_right_padding())

            if frame.is_last:
                self.rm_bufferer(frame.stream_id)

        return fbuffers, right_paddings
