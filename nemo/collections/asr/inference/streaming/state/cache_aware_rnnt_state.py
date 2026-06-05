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


from nemo.collections.asr.inference.streaming.state.cache_aware_state import CacheAwareStreamingState
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


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
        # Per-stream MALSD batched-state item carried across chunks (and across
        # utterances within a stream, same as the buffered RNNT pipeline). Stays
        # ``None`` when the pipeline is running greedy decoding via the
        # high-level ``rnnt_decoder_predictions_tensor`` path.
        self.hyp_decoding_state = None
        # --- Windowed-beam (cross-chunk MALSD) tracking ---
        # Only populated when the cache-aware MALSD pipeline runs with
        # ``streaming.chunks_per_beam_reset > 1``. Reset on every collapse
        # boundary, on stream end, and on full state reset.
        #
        # ``window_committed_tokens`` is the immutable cumulative non-blank
        # token list at the most recent collapse boundary (i.e. the prefix
        # that all surviving beams share). The published per-chunk hypothesis
        # is built as ``committed_tokens + window_beam_tokens[top1_slot]``.
        self.window_committed_tokens: list[int] = []
        self.window_committed_timestamps: list[int] = []
        # ``window_beam_tokens`` holds the chunk-local cumulative tokens that
        # each MALSD slot has accumulated since the last collapse, indexed by
        # the current (post-chunk) slot id. ``None`` means "no window in
        # flight" (first chunk after a collapse / stream init); initialised on
        # first windowed chunk.
        self.window_beam_tokens: list[list[int]] | None = None
        self.window_beam_timestamps: list[list[int]] | None = None
        # Per-stream MALSD chunk counter. Drives the absolute frame range
        # covered by the rolling EOU label buffer in
        # :meth:`CacheAwareRNNTPipeline.run_malsd_decoder` (where the buffer
        # is rebuilt from scratch from the current top-1's cumulative each
        # chunk, rather than rolled in incrementally via
        # ``update_label_buffer``). Reset on EOU + stream end to keep frame
        # numbering aligned with each new utterance's MALSD state.
        self._malsd_chunk_count: int = 0
        # Index into ``hyp.y_sequence`` at which the current utterance starts.
        # ``hyp.y_sequence`` is cumulative since stream start (the windowed
        # state only resets on stream end, not on EOU), so when EOU clears
        # ``state.tokens`` we snapshot the current cumulative length here and
        # publish only the suffix on subsequent chunks. Mirrors the implicit
        # behaviour of the greedy path's ``state.offset`` across EOU
        # boundaries.
        self._malsd_utterance_start: int = 0

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
        self.hyp_decoding_state = None
        self.window_committed_tokens = []
        self.window_committed_timestamps = []
        self.window_beam_tokens = None
        self.window_beam_timestamps = None
        self._malsd_chunk_count = 0
        self._malsd_utterance_start = 0
