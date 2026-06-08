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


from typing import Any

from nemo.collections.asr.inference.streaming.state.cache_aware_state import CacheAwareStreamingState
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


class CacheAwareRNNTStreamingState(CacheAwareStreamingState):
    """
    State of the cache aware RNNT streaming pipelines (greedy decoder).

    Extends :class:`CacheAwareStreamingState` with greedy-decoding bookkeeping
    (``previous_hypothesis``). The MALSD beam-search variant adds its own
    per-stream carry in :class:`CacheAwareRNNTMALSDStreamingState`.
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
        Reset the previous hypothesis. Called at utterance end (EOU).
        """
        self.previous_hypothesis = None


class CacheAwareRNNTMALSDStreamingState(CacheAwareRNNTStreamingState):
    """
    Cache-aware RNNT state with MALSD beam-search per-stream bookkeeping.

    Adds the following fields on top of the greedy state:

    - ``hyp_decoding_state``: per-stream beam carry (``MALSDStateItem``-like)
      shuttled between :meth:`merge_to_batched_state` and :meth:`split_batched_state`.
    - ``window_committed_tokens`` / ``window_committed_timestamps``: cumulative
      prefix shared by all surviving beams at the most recent collapse boundary.
    - ``window_beam_tokens`` / ``window_beam_timestamps``: per-slot chunk-local
      cumulative emissions since the last collapse (one list per beam slot).
    - ``_malsd_chunk_count``: number of MALSD chunks processed since the last
      collapse - used by ``chunks_per_beam_reset`` to decide when to collapse.
    - ``_malsd_utterance_start``: position in the cumulative ``hyp.y_sequence``
      where the current utterance begins, so EOU + ``cleanup_after_eou`` can
      correctly slice past previously emitted (and cleared) utterances.
    """

    def _additional_params_reset(self) -> None:
        """
        Reset MALSD per-stream carry on top of the greedy state.
        """
        super()._additional_params_reset()
        self.hyp_decoding_state: Any = None
        self.window_committed_tokens: list[int] = []
        self.window_committed_timestamps: list[int] = []
        self.window_beam_tokens: list[list[int]] | None = None
        self.window_beam_timestamps: list[list[int]] | None = None
        self._malsd_chunk_count: int = 0
        self._malsd_utterance_start: int = 0

    def reset_previous_hypothesis(self) -> None:
        """
        Reset the previous hypothesis and all MALSD beam-search bookkeeping.

        Called at utterance end (EOU). Zeroes out the MALSD per-stream carry so
        the next utterance starts from SOS with an empty windowed-beam state.
        """
        super().reset_previous_hypothesis()
        self.hyp_decoding_state = None
        self.window_committed_tokens = []
        self.window_committed_timestamps = []
        self.window_beam_tokens = None
        self.window_beam_timestamps = None
        self._malsd_chunk_count = 0
        # NB: ``_malsd_utterance_start`` is intentionally NOT reset here because
        # the cumulative ``hyp.y_sequence`` it indexes is owned by the pipeline
        # and bumped after the call when the previous utterance is being
        # finalised. The pipeline bumps it explicitly after publishing.
