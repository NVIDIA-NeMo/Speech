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


from dataclasses import dataclass
from typing import TypeAlias

import torch

from nemo.collections.asr.inference.streaming.framing.request_options import RequestOptions


@dataclass(frozen=True, slots=True)
class Frame:
    """
    Immutable dataclass representing

    Args:
        samples (torch.Tensor): The actual frame data. For audio, shape is (T,).
        stream_id (int): Unique identifier for the stream this frame belongs to
        is_first (bool): Flag indicating if this is the first frame in the stream
        is_last (bool): Flag indicating if this is the last frame in the stream
        length (int): Length of the frame without padding.
                      If -1, returns the size of the frame including padding.
        vad_segments (torch.Tensor | None): Optional VAD segments to use for end-of-utterance detection.
                                     Shape is [num_vad_segments, 2] where each segment contains
                                     [start_time, end_time]. Variable for each stream.
        options (RequestOptions | None): Optional options for the request
    """

    samples: torch.Tensor
    stream_id: int
    is_first: bool = False
    is_last: bool = False
    length: int = -1
    vad_segments: torch.Tensor | None = None
    options: RequestOptions | None = None

    @property
    def size(self) -> int:
        """Returns the size of the frame including padding"""
        return self.samples.shape[0]

    @property
    def valid_size(self) -> int:
        """Returns the size of the frame without padding"""
        return self.size if self.length == -1 else self.length


@dataclass(frozen=True, slots=True)
class FeatureBuffer:
    """
    Immutable dataclass representing a buffer of features.
    Args:
        features (torch.Tensor): The actual frame data. For features, shape is (feature_dim, T).
        stream_id (int): Unique identifier for the stream this frame belongs to
        is_first (bool): Flag indicating if this is the first frame in the stream
        is_last (bool): Flag indicating if this is the last frame in the stream
        right_pad_features (bool): Flag indicating if the features are right padded
        length (int): Length of the valid features in the buffer
                      If -1, returns the size of the buffer including padding
        left_padding_length (int): Length of the left padding in the buffer
                                   It is used to roll features to the right
        vad_segments (torch.Tensor | None): Optional VAD segments to use for end-of-utterance detection.
                                     Shape is [num_vad_segments, 2] where each segment contains
                                     [start_time, end_time]. Variable for each stream.
        options (RequestOptions | None): Optional options for the request
    """

    features: torch.Tensor
    stream_id: int
    is_first: bool = False
    is_last: bool = False
    right_pad_features: bool = False
    length: int = -1
    left_padding_length: int = 0
    vad_segments: torch.Tensor | None = None
    options: RequestOptions | None = None

    @property
    def size(self) -> int:
        """Returns the number of features in the buffer including padding"""
        return self.features.shape[1]

    @property
    def valid_size(self) -> int:
        """Returns the size of the buffer without padding. It is a actual length of the signal"""
        return self.size if self.length == -1 else self.length

    @property
    def roll_size(self) -> int:
        """Returns the size of the buffer to roll to the right. It only makes sense for right padded feature buffers"""
        return self.left_padding_length if self.right_pad_features else 0


Request: TypeAlias = Frame | FeatureBuffer
