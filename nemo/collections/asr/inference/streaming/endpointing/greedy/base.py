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


from nemo.collections.asr.inference.streaming.endpointing.greedy.token_classifier import TokenClassifier
from nemo.collections.asr.inference.utils.endpointing_utils import millisecond_to_frames


class GreedyEndpointerBase:
    """Shared state for the greedy endpointers.

    Holds the silence thresholds and a composed `TokenClassifier` (the per-token semantics). Each concrete
    endpointer adds only the detection method(s) its pipeline uses:
      - `BufferedCTCEndpointer`   -> `detect_eou` / `detect_eou_given_emissions`
      - `BufferedRNNTEndpointer`  -> `detect_eou_given_timestamps` / `detect_eou_vad`
      - `CacheAwareEndpointer`    -> `detect_eou_in_buffer` (shared by the CTC and RNNT cache-aware leaves)

    `stop_history_eou` is given in milliseconds; positive values are pre-converted to frames here (negative
    or zero values are passed through unchanged, as the detection methods expect). The millisecond value is
    kept in `stop_history_eou_ms` for the VAD path.
    """

    def __init__(
        self,
        token_classifier: TokenClassifier,
        ms_per_timestep: int,
        stop_history_eou: int = -1,
        residue_tokens_at_end: int = 0,
        stop_history_eou_end: int | None = None,
    ) -> None:
        self.token_classifier = token_classifier
        self.vocabulary = token_classifier.vocabulary
        self.ms_per_timestep = ms_per_timestep
        self.sec_per_timestep = ms_per_timestep / 1000
        self.residue_tokens_at_end = residue_tokens_at_end

        self.stop_history_eou_ms = stop_history_eou
        self.stop_history_eou = stop_history_eou
        if self.stop_history_eou > 0:
            self.stop_history_eou = millisecond_to_frames(stop_history_eou, ms_per_timestep)

        self.stop_history_eou_end = stop_history_eou_end
        if self.stop_history_eou_end is not None and self.stop_history_eou_end > 0:
            self.stop_history_eou_end = millisecond_to_frames(self.stop_history_eou_end, ms_per_timestep)

    def is_token_start_of_word(self, token_id: int) -> bool:
        """Check if the token is the start of a word."""
        return self.token_classifier.is_token_start_of_word(token_id)

    def is_token_silent(self, token_id: int) -> bool:
        """Check if the token is silent (blank)."""
        return self.token_classifier.is_token_silent(token_id)

    def is_token_to_absorb(self, token_id: int) -> bool:
        """Check if the token should be absorbed into the text preceding the EoU."""
        return self.token_classifier.is_token_to_absorb(token_id)
