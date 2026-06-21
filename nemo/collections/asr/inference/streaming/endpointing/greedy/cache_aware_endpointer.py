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


from nemo.collections.asr.inference.streaming.endpointing.greedy.base import GreedyEndpointerBase
from nemo.collections.asr.inference.utils.endpointing_utils import get_custom_stop_history_eou


class CacheAwareEndpointer(GreedyEndpointerBase):
    """Endpointing for the cache-aware pipelines (shared by the CTC and RNNT leaves).

    Detects an EoU by scanning a dense label buffer right-to-left with dual mid-buffer / end-of-buffer
    silence thresholds and punctuation absorption. The only difference between the CTC and RNNT leaves is
    the composed `TokenClassifier`; the detection algorithm is identical.
    """

    def detect_eou_in_buffer(
        self,
        emissions: list[int],
        search_start_point: int = 0,
        stop_history_eou: int | None = None,
        stop_history_eou_end: int | None = None,
    ) -> tuple[bool, int, int]:
        """
        Detect end of utterance (EoU) by scanning the emissions buffer right-to-left and returning
        at the most recent (rightmost) qualifying silence run, so as much text as possible is
        committed to the user as soon as the speaker pauses.

        A silence run qualifies when it is strictly longer than a threshold whose value depends on
        what follows the run:
          - Mid-buffer EoU: a real word-start token is observed after the run (possibly past absorbed
            punctuation/language tokens). The run only needs to exceed `stop_history_eou`; seeing the
            next word confirms the boundary, so this stays responsive.
          - End-of-buffer EoU: the run runs to the end of the buffer (trailing silence, or only
            absorbed punctuation after it), so no following word has been observed yet. The run must
            exceed the larger `stop_history_eou_end`. This avoids cutting a word when the speaker
            merely pauses mid-word at the buffer edge -- a short pause no longer triggers a premature
            end-of-buffer EoU; the cut only happens once the silence is long enough to be a real end.

        Punctuation/language tokens (see `absorb_token_ids`) right after the run are absorbed into the
        text preceding the EoU (with any silence trailing them), so late-emitted sentence punctuation
        stays with the finalized utterance. A non-absorbable, non-word-start token after the run means
        the silence fell mid-word: that run is rejected and the scan keeps moving left.

        Args:
            emissions (list[int]): dense per-timestep labels (blank == silent) of the label buffer
            search_start_point (int): buffer-local index to start searching from (inclusive)
            stop_history_eou (int | None): regular (mid-buffer) silence threshold in milliseconds; if
                None use the class default (already in frames)
            stop_history_eou_end (int | None): end-of-buffer silence threshold in milliseconds; if None
                use the class default, falling back to the regular threshold. Clamped to be >= regular.
                A negative value (e.g. -1) disables only *unconfirmed* end-of-buffer EoUs: a run of pure
                trailing silence (nothing after it) no longer triggers an EoU. CONFIRMED EoUs still fire --
                a word-start after the run (mid-buffer) or trailing punctuation (the model committed a
                sentence-final token), both at the regular threshold. So under -1 boundaries are only
                committed when confirmed by the next word or by punctuation.
        Returns:
            tuple[bool, int, int]:
                eou_detected: True if a valid EoU is detected, False otherwise
                eou_center: index of the EoU center within `emissions`, -1 if not detected
                resume_index: index of the first token of the next utterance (past any absorbed
                    punctuation/language tokens); equals len(emissions) if the buffer ends in silence
                    or only trailing punctuation. -1 if not detected.
        """
        sequence_length = len(emissions)
        stop_history_eou = get_custom_stop_history_eou(stop_history_eou, self.stop_history_eou, self.ms_per_timestep)

        if sequence_length == 0 or stop_history_eou < 0:
            return False, -1, -1

        # stop_history_eou == 0 -> offline/Riva mode: finalize the whole buffer immediately.
        if stop_history_eou == 0:
            return True, sequence_length - 1, sequence_length

        # End-of-buffer threshold: explicit override, else class default, else the regular threshold.
        # A negative value (e.g. -1) DISABLES end-of-buffer EoUs: trailing silence at the buffer edge is
        # never treated as an EoU; only word-confirmed mid-buffer EoUs can fire.
        end_default = self.stop_history_eou_end if self.stop_history_eou_end is not None else self.stop_history_eou
        stop_history_eou_end = get_custom_stop_history_eou(stop_history_eou_end, end_default, self.ms_per_timestep)
        end_enabled = stop_history_eou_end >= 0
        if end_enabled:
            # Never weaker than the regular threshold.
            stop_history_eou_end = max(stop_history_eou, stop_history_eou_end)

        lower_bound = max(0, search_start_point)
        # Scan right-to-left so the most recent (rightmost) qualifying silence run wins.
        i = sequence_length - 1
        while i >= lower_bound:
            if not self.is_token_silent(emissions[i]):
                i -= 1
                continue

            # End of a silence run at index `i`; walk left to its start (not past the search bound).
            run_end = i
            run_start = i
            while run_start - 1 >= lower_bound and self.is_token_silent(emissions[run_start - 1]):
                run_start -= 1
            run_len = run_end - run_start + 1
            j = run_end + 1

            # Trailing silence to the buffer end -> end-of-buffer EoU (no following word observed).
            if j >= sequence_length:
                if end_enabled and run_len > stop_history_eou_end:
                    return True, run_start + stop_history_eou_end // 2, sequence_length
                i = run_start - 1
                continue

            # Absorb trailing punctuation/language tokens into the pre-EoU side. Once at least one
            # such token is absorbed, also skip any silence that follows it: the punctuation closed
            # the previous utterance, so the next utterance only begins at the next real token. This
            # keeps late-emitted sentence punctuation with the finalized text.
            k = j
            absorbed_any = False
            while k < sequence_length:
                if self.is_token_to_absorb(emissions[k]):
                    absorbed_any = True
                    k += 1
                elif absorbed_any and self.is_token_silent(emissions[k]):
                    k += 1
                else:
                    break

            if k >= sequence_length:
                # Run followed only by punctuation/language tokens (+ trailing silence) to the buffer
                # end. The punctuation CONFIRMS the utterance end (just like a following word-start
                # confirms a mid-buffer EoU), so there is no mid-word risk: fire at the regular threshold
                # and do so even when end-of-buffer EoUs are disabled (`stop_history_eou_end < 0`).
                if run_len > stop_history_eou:
                    return True, run_start + stop_history_eou // 2, sequence_length
            elif self.is_token_start_of_word(emissions[k]):
                # Mid-buffer EoU: the next word-start is observed, so the regular threshold applies.
                if run_len > stop_history_eou:
                    return True, run_start + stop_history_eou // 2, k
            # Otherwise the run is mid-word or below threshold: not a valid EoU here.

            # Keep scanning to the left of this run.
            i = run_start - 1

        return False, -1, -1
