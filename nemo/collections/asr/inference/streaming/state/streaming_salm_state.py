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
"""Pipeline state for the StreamingSALM streaming inference pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nemo.collections.asr.inference.streaming.state.state import StreamingState

if TYPE_CHECKING:
    from nemo.collections.speechlm2.models.streaming_salm import StreamingState as ModelStreamingState


class StreamingSALMStreamingState(StreamingState):
    """
    Streaming pipeline state that bridges the model's :class:`StreamingState`
    (KV cache) with the pipeline's :class:`StreamingState` (output management).

    Attributes:
        model_state: The model-level KV cache / streaming state (or ``None``
            before the first audio chunk).
        audio_residual: Sub-frame audio samples left over from the previous
            step (Mimi operates at 12.5 Hz = 1920 samples per frame at 24 kHz).
        audio_residual_len: Derived from ``audio_residual``; number of valid
            samples in the residual tensor.
    """

    def __init__(self):
        super().__init__()

    def _reset_streaming_state(self) -> None:
        super()._reset_streaming_state()
        self.model_state: ModelStreamingState | None = None
        self.audio_residual: torch.Tensor | None = None

    @property
    def audio_residual_len(self) -> int:
        return self.audio_residual.shape[0] if self.audio_residual is not None else 0
