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

from nemo.collections.asr.inference.streaming.state.state import StreamingState
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


class RNNTStreamingState(StreamingState):
    """
    State of the streaming RNNT pipeline
    """

    def __init__(self):
        """
        Initialize the RNNTStreamingState
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
        self.timestamp_offset = 0
        self.hyp_decoding_state = None


class RNNTBeamStreamingState(RNNTStreamingState):
    """Beam search streaming state; decoder carry + cumulative/partial tokens.

    ``hyp_decoding_state``: K-beam carry across chunks (collapsed to top1 on EOU in the pipeline).
    ``cumulative_*``: tokens/timestamps sealed at each EOU (prior utterances in a stream).
    ``partial_*[k]``: per-beam in-flight suffix since last EOU (chunk-local exports merged via lineage).
    ``best_hyp_idx``: index into ``partial_*`` for the chunk argmax beam used to publish.
    """

    @staticmethod
    def _append_emissions_by_frame(
        prev_tokens: list[int],
        prev_timestamps: list[int],
        new_tokens: list[int],
        new_timestamps: list[int],
    ) -> tuple[list[int], list[int]]:
        """Keep only emissions on frames not yet present in the beam window."""
        if not prev_timestamps:
            return list(new_tokens), list(new_timestamps)
        max_frame = max(prev_timestamps)
        start = 0
        while start < len(new_timestamps) and new_timestamps[start] <= max_frame:
            start += 1
        return prev_tokens + new_tokens[start:], prev_timestamps + new_timestamps[start:]

    def _additional_params_reset(self) -> None:
        super()._additional_params_reset()
        self.cumulative_tokens: list[int] = []
        self.cumulative_timestamps: list[int] = []
        self.partial_tokens: list[list[int]] | None = None
        self.partial_timestamps: list[list[int]] | None = None
        self._cumulative_tokens_len: int = 0
        self.best_hyp_idx: int | None = None

    def reset_beam_decoding_state_(self) -> None:
        """Clear beam search carry and cumulative/partial tokens when a stream ends."""
        self.hyp_decoding_state = None
        self.cumulative_tokens = []
        self.cumulative_timestamps = []
        self.partial_tokens = None
        self.partial_timestamps = None
        self._cumulative_tokens_len = 0
        self.best_hyp_idx = None

    def append_chunk_beam_(
        self,
        chunk_tokens: list[list[int]],
        chunk_timestamps: list[list[int]],
        root_ptrs: list[int],
        beam_size: int,
        best_hyp_idx: int,
        ts_offset: int = 0,
    ) -> None:
        """Append deduplicated chunk-local beam exports into state."""
        prev_t = self.partial_tokens or [[] for _ in range(beam_size)]
        prev_ts = self.partial_timestamps or [[] for _ in range(beam_size)]
        next_tokens: list[list[int]] = []
        next_timestamps: list[list[int]] = []
        for k in range(beam_size):
            lineage = int(root_ptrs[k])
            cts_global = [t + ts_offset for t in chunk_timestamps[k]]
            tokens, timestamps = self._append_emissions_by_frame(
                prev_t[lineage], prev_ts[lineage], chunk_tokens[k], cts_global
            )
            next_tokens.append(tokens)
            next_timestamps.append(timestamps)
        self.partial_tokens = next_tokens
        self.partial_timestamps = next_timestamps
        self.best_hyp_idx = best_hyp_idx

    def get_best_hyp_idx(self) -> int:
        """Index into ``partial_*`` for publish (chunk argmax, or score argmax from carry)."""
        if self.best_hyp_idx is not None:
            return int(self.best_hyp_idx)
        if self.hyp_decoding_state is None:
            raise RuntimeError("Cannot resolve top-1 beam index without decoding carry.")
        return int(self.hyp_decoding_state.score.argmax().item())

    def _get_tokens(self) -> tuple[list[int], list[int]]:
        """``cumulative_*`` plus the current top-1 ``partial_*`` suffix."""
        if self.partial_tokens is None or self.hyp_decoding_state is None:
            return [], []
        best_hyp_idx = self.get_best_hyp_idx()
        return (
            self.cumulative_tokens + list(self.partial_tokens[best_hyp_idx]),
            self.cumulative_timestamps + list(self.partial_timestamps[best_hyp_idx]),
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
        self.best_hyp_idx = None
