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

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo.collections.asr.inference.streaming.state.cache_aware_state import CacheAwareStreamingState
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

if TYPE_CHECKING:
    from nemo.collections.asr.parts.submodules.rnnt_malsd_batched_computer import MALSDStateItem


class CacheAwareRNNTStreamingState(CacheAwareStreamingState):
    """
    State of the cache aware RNNT streaming pipelines
    """

    def __init__(self):
        """
        Initialize the CacheAwareRNNTStreamingState
        """
        super().__init__()
        self._additional_params_reset()

    def reset(self) -> None:
        """
        Reset the state
        """
        super().reset()
        self._additional_params_reset()

    def _additional_params_reset(self) -> None:
        """
        Reset non-inherited parameters
        """
        super()._additional_params_reset()
        self.previous_hypothesis = None

    def set_previous_hypothesis(self, previous_hypothesis: Hypothesis) -> None:
        """
        Set the previous hypothesis
        Args:
            previous_hypothesis: (Hypothesis) The previous hypothesis to store for the next transcribe step
        """
        self.previous_hypothesis = previous_hypothesis

    def get_previous_hypothesis(self) -> Hypothesis | None:
        """
        Get the previous hypothesis
        Returns:
            (Hypothesis) The previous hypothesis
        """
        return self.previous_hypothesis

    def reset_previous_hypothesis(self) -> None:
        """
        Reset the previous hypothesis to None
        """
        self.previous_hypothesis = None


class CacheAwareRNNTBeamStreamingState(CacheAwareRNNTStreamingState):
    """Cache-aware RNNT state for MALSD beam-search streaming."""

    def _additional_params_reset(self) -> None:
        super()._additional_params_reset()
        self.hyp_decoding_state: MALSDStateItem | None = None
        # Finalized transcript prefix at the last EOU; identical for every beam slot.
        self.window_committed_tokens: list[int] = []
        # Frame timestamps aligned with ``window_committed_tokens``.
        self.window_committed_timestamps: list[int] = []

        # Per-beam suffix since last EOU; slot k may differ while beams compete.
        self.window_beam_tokens: list[list[int]] | None = None
        # Per-beam frame timestamps aligned with ``window_beam_tokens``.
        self.window_beam_timestamps: list[list[int]] | None = None

        # Index into cumulative ``hyp.y_sequence`` where the current utterance starts
        # (skips tokens from prior utterances still present in the cumulative hyp).
        self._malsd_utterance_start: int = 0

    def reset_previous_hypothesis(self) -> None:
        """Reset carry and windowed-beam state at end-of-stream."""
        super().reset_previous_hypothesis()
        self.hyp_decoding_state = None
        self.window_committed_tokens = []
        self.window_committed_timestamps = []
        self.window_beam_tokens = None
        self.window_beam_timestamps = None
        # _malsd_utterance_start is bumped by the pipeline on EOU, not here.
