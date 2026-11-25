# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


Request: TypeAlias = Frame
