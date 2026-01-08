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
# NOTE: This file will be deprecated in the future, as the new inference pipeline will replace it.

import math

import numpy as np
import torch
from omegaconf import DictConfig

import nemo.collections.asr as nemo_asr

LOG_MEL_ZERO = -16.635


class AudioBufferer:
    def __init__(self, sample_rate: int, buffer_size_in_secs: float):
        self.buffer_size = int(buffer_size_in_secs * sample_rate)
        self.sample_buffer = torch.zeros(self.buffer_size, dtype=torch.float32)

    def reset(self) -> None:
        """
        Reset the buffer to zero
        """
        self.sample_buffer.zero_()

    def update(self, audio: np.ndarray) -> None:
        """
        Update the buffer with the new frame
        Args:
            frame (Frame): frame to update the buffer with
        """
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)

        audio_size = audio.shape[0]
        if audio_size > self.buffer_size:
            raise ValueError(f"Frame size ({audio_size}) exceeds buffer size ({self.buffer_size})")

        shift = audio_size
        self.sample_buffer[:-shift] = self.sample_buffer[shift:].clone()
        self.sample_buffer[-shift:] = audio.clone()

    def get_buffer(self) -> torch.Tensor:
        """
        Get the current buffer
        Returns:
            torch.Tensor: current state of the buffer
        """
        return self.sample_buffer.clone()

    def is_buffer_empty(self) -> bool:
        """
        Check if the buffer is empty
        Returns:
            bool: True if the buffer is empty, False otherwise
        """
        return self.sample_buffer.sum() == 0


class CacheFeatureBufferer:
    def __init__(
        self,
        sample_rate: int,
        buffer_size_in_secs: float,
        chunk_size_in_secs: float,
        preprocessor_cfg: DictConfig,
        device: torch.device,
        fill_value: float = LOG_MEL_ZERO,
    ):

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
            self.ZERO_LEVEL_SPEC_DB_VAL = fill_value

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

        self.preprocessor = nemo_asr.models.ASRModel.from_config_dict(preprocessor_cfg)
        self.preprocessor.to(self.device)

    def is_buffer_empty(self) -> bool:
        """
        Check if the buffer is empty
        Returns:
            bool: True if the buffer is empty, False otherwise
        """
        return self.sample_buffer.is_buffer_empty()

    def reset(self) -> None:
        """
        Reset the buffer to zero
        """
        self.sample_buffer.reset()
        self.feature_buffer.fill_(self.ZERO_LEVEL_SPEC_DB_VAL)

    def _update_feature_buffer(self, feat_chunk: torch.Tensor) -> None:
        """
        Add an extracted feature to `feature_buffer`
        """
        self.feature_buffer[:, : -self.feature_chunk_len] = self.feature_buffer[:, self.feature_chunk_len :].clone()
        self.feature_buffer[:, -self.feature_chunk_len :] = feat_chunk.clone()

    def preprocess(self, audio_signal: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the audio signal using the preprocessor
        Args:
            audio_signal (torch.Tensor): audio signal
        Returns:
            torch.Tensor: preprocessed features
        """
        audio_signal = audio_signal.unsqueeze_(0).to(self.device)
        audio_signal_len = torch.tensor([audio_signal.shape[1]], device=self.device)
        features, _ = self.preprocessor(
            input_signal=audio_signal,
            length=audio_signal_len,
        )
        features = features.squeeze()
        return features

    def update(self, audio: np.ndarray) -> None:
        """
        Update the sample anf feature buffers with the new frame
        Args:
            frame (Frame): frame to update the buffer with
        """

        # Update the sample buffer with the new frame
        self.sample_buffer.update(audio)

        if math.isclose(self.buffer_size_in_secs, self.chunk_size_in_secs):
            # If the buffer size is equal to the chunk size, just take the whole buffer
            samples = self.sample_buffer.sample_buffer.clone()
        else:
            # Add look_back to have context for the first feature
            samples = self.sample_buffer.sample_buffer[-(self.n_chunk_look_back + self.chunk_size) :]

        # Get the mel spectrogram
        features = self.preprocess(samples)

        # If the features are longer than supposed to be, drop the last frames
        # Drop the last diff frames because they might be incomplete
        if (diff := features.shape[1] - self.feature_chunk_len - 1) > 0:
            features = features[:, :-diff]

        # Update the feature buffer with the new features
        self._update_feature_buffer(features[:, -self.feature_chunk_len :])

    def get_buffer(self) -> torch.Tensor:
        """
        Get the current sample buffer
        Returns:
            torch.Tensor: current state of the buffer
        """
        return self.sample_buffer.get_buffer()

    def get_feature_buffer(self) -> torch.Tensor:
        """
        Get the current feature buffer
        Returns:
            torch.Tensor: current state of the feature buffer
        """
        return self.feature_buffer.clone()
