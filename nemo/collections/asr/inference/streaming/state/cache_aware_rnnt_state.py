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
    """MALSD beam-search streaming state; decoder carry + cumulative/partial tokens.

    ``hyp_decoding_state``: K-beam MALSD carry across chunks (collapsed to top1 on EOU in the pipeline).
    ``cumulative_*``: tokens/timestamps sealed at each EOU (prior utterances in a stream).
    ``partial_*[k]``: per-beam in-flight suffix since last EOU (chunk-local exports merged via lineage).
    ``partial_top1_slot``: beam index for publish (chunk argmax).

    Chunk overlap is handled inside MALSD carry (same as greedy label-looping); Python only
    concatenates chunk-local exports onto ``partial_*`` following ``root_ptrs`` beam lineage.
    ``_cumulative_tokens_len`` slices ``state.tokens`` to the current utterance for publish.
    On EOU: ``update_(eou_detected=True)`` folds tokens into ``cumulative_*``, clears ``partial_*``.
    """

    def _additional_params_reset(self) -> None:
        super()._additional_params_reset()
        self.hyp_decoding_state: MALSDStateItem | None = None
        self.cumulative_tokens: list[int] = []
        self.cumulative_timestamps: list[int] = []
        self.partial_tokens: list[list[int]] | None = None
        self.partial_timestamps: list[list[int]] | None = None
        self._cumulative_tokens_len: int = 0
        self.partial_top1_slot: int | None = None

    def reset_beam_decoding_state_(self) -> None:
        """Clear MALSD carry and cumulative/partial tokens when a stream ends."""
        self.hyp_decoding_state = None
        self.cumulative_tokens = []
        self.cumulative_timestamps = []
        self.partial_tokens = None
        self.partial_timestamps = None
        self._cumulative_tokens_len = 0
        self.partial_top1_slot = None

    def append_chunk_beam_(
        self,
        chunk_tokens: list[list[int]],
        chunk_timestamps: list[list[int]],
        root_ptrs: list[int],
        beam_size: int,
        top1_slot: int,
    ) -> None:
        """Append chunk-local beam exports into state."""
        prev_t = self.partial_tokens or [[] for _ in range(beam_size)]
        prev_ts = self.partial_timestamps or [[] for _ in range(beam_size)]
        next_tokens: list[list[int]] = []
        next_timestamps: list[list[int]] = []
        for k in range(beam_size):
            lineage = int(root_ptrs[k])
            next_tokens.append(prev_t[lineage] + list(chunk_tokens[k]))
            next_timestamps.append(prev_ts[lineage] + list(chunk_timestamps[k]))
        self.partial_tokens = next_tokens
        self.partial_timestamps = next_timestamps
        self.partial_top1_slot = top1_slot

    def get_top1_beam_index(self) -> int:
        """Beam slot used for publish (chunk argmax, or score argmax from carry)."""
        if self.partial_top1_slot is not None:
            return int(self.partial_top1_slot)
        if self.hyp_decoding_state is None:
            raise RuntimeError("Cannot resolve top-1 beam index without decoding carry.")
        return int(self.hyp_decoding_state.score.argmax().item())

    def _get_tokens(self) -> tuple[list[int], list[int]]:
        """``cumulative_*`` plus the current top-1 ``partial_*`` suffix."""
        if self.partial_tokens is None or self.hyp_decoding_state is None:
            return [], []
        top1 = self.get_top1_beam_index()
        return (
            self.cumulative_tokens + list(self.partial_tokens[top1]),
            self.cumulative_timestamps + list(self.partial_timestamps[top1]),
        )

    def get_hypothesis(self, score: float) -> Hypothesis:
        """Build the publishable cumulative hypothesis for the current top-1 beam."""
        cum_tokens, cum_ts = self._get_tokens()
        return Hypothesis(
            score=score,
            y_sequence=cum_tokens,
            timestamp=cum_ts,
            length=len(cum_tokens),
        )

    def update_(self, eou_detected: bool) -> None:
        """Refresh publish tokens; on EOU fold utterance into ``cumulative_*`` and clear ``partial_*``."""
        cum_tokens, cum_ts = self._get_tokens()
        if cum_tokens:
            start = max(0, min(int(self._cumulative_tokens_len), len(cum_tokens)))
            tokens = list(cum_tokens[start:])
            timesteps = list(cum_ts[start:])
            self.tokens = tokens
            self.timesteps = timesteps
            self.confidences = [0.0] * len(tokens)
            if tokens:
                self.last_token = tokens[-1]
                self.last_token_idx = timesteps[-1] if timesteps else None

        if not eou_detected:
            return

        if cum_tokens:
            self._cumulative_tokens_len = len(cum_tokens)
            self.cumulative_tokens = list(cum_tokens)
            self.cumulative_timestamps = list(cum_ts)
        self.partial_tokens = None
        self.partial_timestamps = None
        self.partial_top1_slot = None
