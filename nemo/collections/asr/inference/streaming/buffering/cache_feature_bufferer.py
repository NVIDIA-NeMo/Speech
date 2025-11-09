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
from queue import Queue

import torch
from omegaconf import DictConfig
from torch import Tensor

from nemo.collections.asr.inference.streaming.buffering.audio_bufferer import AudioBufferer
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.utils.constants import LOG_MEL_ZERO
from nemo.collections.asr.models import ASRModel


class BatchedCacheFeatureBufferer:
    """
    Batched cache feature bufferer class
    Buffers feature chunks from multiple audio streams and manages their storage.
    Maintains a tensor of shape (num_slots, n_feat, feature_buffer_len), where each slot
    corresponds to a single audio stream. The number of slots equals the number of
    active (or open) audio streams.
    """

    def __init__(
        self,
        num_slots: int,
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
            num_slots (int): number of slots, where each slot contains feature buffer for a single audio stream
            sample_rate (int): sample rate
            buffer_size_in_secs (float): buffer size in seconds
            chunk_size_in_secs (float): chunk size in seconds
            preprocessor_cfg (DictConfig): preprocessor configuration
            device (torch.device): device
            fill_value (float): fill value for the feature buffer
            right_padding_ratio (float): right padding ratio
        """
        if buffer_size_in_secs < chunk_size_in_secs:
            raise ValueError(
                f"Buffer size ({buffer_size_in_secs}s) should be no less than chunk size ({chunk_size_in_secs}s)"
            )

        self.num_slots = num_slots
        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.chunk_size_in_secs = chunk_size_in_secs
        self.preprocessor_cfg = preprocessor_cfg
        self.device = device
        self.right_padding_ratio = right_padding_ratio

        self.is_buffer_size_equal_to_chunk_size = math.isclose(self.buffer_size_in_secs, self.chunk_size_in_secs)
        self.plus_one = 0 if self.is_buffer_size_equal_to_chunk_size else 1

        if hasattr(preprocessor_cfg, 'log') and preprocessor_cfg.log:
            self.ZERO_LEVEL_SPEC_DB_VAL = LOG_MEL_ZERO  # Log-Mel spectrogram value for zero signals
        else:
            self.ZERO_LEVEL_SPEC_DB_VAL = fill_value  # Custom fill value for the feature buffer

        self.n_feat = preprocessor_cfg.features
        self.timestep_duration = preprocessor_cfg.window_stride
        self.n_chunk_look_back = int(self.timestep_duration * self.sample_rate)
        self.chunk_size = int(self.chunk_size_in_secs * self.sample_rate)
        self.extended_chunk_size = self.n_chunk_look_back + self.chunk_size
        self.audio_bufferers = [
            AudioBufferer(self.sample_rate, self.buffer_size_in_secs) for _ in range(self.num_slots)
        ]

        self.feature_buffer_len = int(buffer_size_in_secs / self.timestep_duration)
        self.feature_chunk_len = int(chunk_size_in_secs / self.timestep_duration)
        self.feature_buffer = torch.full(
            [self.num_slots, self.n_feat, self.feature_buffer_len],
            self.ZERO_LEVEL_SPEC_DB_VAL,
            dtype=torch.float32,
            device=self.device,
        )

        self.preprocessor = ASRModel.from_config_dict(preprocessor_cfg)
        self.preprocessor.to(self.device)

        self.streamidx2slotidx, self.slotidx2streamidx = {}, {}
        self.available_slots = Queue(self.num_slots)
        for i in range(self.num_slots):
            self.available_slots.put(i)

    def free_slots(self, slot_ids: list[int]) -> None:
        """
        Free the slots for the given slot_ids
        Args:
            slot_ids (list[int]): list of slot ids
        """
        for slot_id in slot_ids:
            self.available_slots.put(slot_id)
            stream_id = self.slotidx2streamidx[slot_id]
            del self.slotidx2streamidx[slot_id], self.streamidx2slotidx[stream_id]

    def reset_slots(self, slot_ids: list[int]) -> None:
        """
        Reset the slots for the given slot_ids
        Args:
            slot_ids (list[int]): list of slot ids
        """
        slot_ids_tensor = torch.tensor(slot_ids, device=self.device, dtype=torch.long)
        self.feature_buffer.index_fill_(0, slot_ids_tensor, self.ZERO_LEVEL_SPEC_DB_VAL)
        for slot_id in slot_ids:
            self.audio_bufferers[slot_id].reset()

    def preprocess(
        self, audio_buffers: list[Tensor], right_paddings: Tensor, expected_feat_len: int
    ) -> tuple[Tensor, Tensor]:
        """
        Preprocess the audio buffers with the given right paddings and expected feature length
        Args:
            audio_buffers (list[Tensor]): list of audio buffers
            right_paddings (Tensor): right paddings: right paddings are not zero for last frames
            expected_feat_len (int): expected feature length
        Returns:
            tuple[Tensor, Tensor]: features and right paddings
        """
        signals = torch.vstack(audio_buffers).to(self.device)  # B x T
        signals_len = torch.tensor([signals.shape[1]] * signals.shape[0], device=self.device, dtype=torch.long)  # B
        right_paddings = right_paddings * self.right_padding_ratio
        signals_len = signals_len - right_paddings.long()
        features, _ = self.preprocessor(input_signal=signals, length=signals_len)
        if features.shape[2] > expected_feat_len:
            features = features[:, :, :expected_feat_len]  # B x F x T
        right_padding = torch.floor(right_paddings / self.sample_rate / self.timestep_duration)  # B
        return features, right_padding

    def _update_feature_buffer(self, slot_ids: int, feat_chunk: Tensor) -> None:
        """
        Add an extracted feature to `feature_buffer`
        Args:
            slot_ids (list[int]): list of slot ids
            feat_chunk (Tensor): feature chunk of shape (B, F, T)
        """
        for i, slot_id in enumerate(slot_ids):
            chunk_len = feat_chunk[i].shape[-1]
            if chunk_len > self.feature_buffer_len:
                raise ValueError(f"feat_chunk ({chunk_len}) longer than buffer ({self.feature_buffer_len})")

            self.feature_buffer[slot_id, :, :-chunk_len].copy_(self.feature_buffer[slot_id, :, chunk_len:])
            self.feature_buffer[slot_id, :, -chunk_len:].copy_(feat_chunk[i])

    def update(self, frames: list[Frame]) -> tuple[list[Tensor], list[int]]:
        """
        Update the feature bufferers with the new frames.
        Args:
            frames (list[Frame]): list of frames with length equal to batch size
        Returns:
            tuple[list[Tensor], list[int]]: feature buffers and right paddings
        """
        # if there are no frames, return empty lists
        if len(frames) == 0:
            return [], []

        # if the stream_id is new, we need to assign a slot to it
        slot_ids, slots_to_reset, slots_to_free = [], [], []
        for frame in frames:
            stream_id = frame.stream_id
            slot_idx = self.streamidx2slotidx.get(stream_id, None)
            if stream_id not in self.streamidx2slotidx:
                if self.available_slots.empty():
                    raise RuntimeError("No free slots available")
                slot_idx = self.available_slots.get()
                self.streamidx2slotidx[stream_id] = slot_idx
                self.slotidx2streamidx[slot_idx] = stream_id
                slots_to_reset.append(slot_idx)

            slot_ids.append(slot_idx)
            if frame.is_last:
                slots_to_free.append(slot_idx)

        # reset the slots for the new stream_ids
        if len(slots_to_reset) > 0:
            self.reset_slots(slots_to_reset)

        right_paddings = torch.zeros(len(frames), dtype=torch.long, device=self.device)
        audio_buffers = []
        for i, frame in enumerate(frames):
            slot_id = slot_ids[i]
            right_paddings[i] = frame.size - frame.valid_size
            self.audio_bufferers[slot_id].update(frame)

            buffer = self.audio_bufferers[slot_id].sample_buffer
            if not self.is_buffer_size_equal_to_chunk_size:
                # Add look_back to have context for the first feature
                audio_buffers.append(buffer[-(self.n_chunk_look_back + self.chunk_size) :])
            else:
                # If the buffer size is equal to the chunk size, just take the whole buffer
                audio_buffers.append(buffer)

        features, right_paddings = self.preprocess(
            audio_buffers=audio_buffers,
            right_paddings=right_paddings,
            expected_feat_len=self.feature_chunk_len + self.plus_one,
        )
        self._update_feature_buffer(slot_ids=slot_ids, feat_chunk=features[:, :, -self.feature_chunk_len :])
        fbuffers = list(self.feature_buffer[slot_ids].unbind(0))

        if len(slots_to_free) > 0:
            self.free_slots(slots_to_free)

        return fbuffers, right_paddings.tolist()
