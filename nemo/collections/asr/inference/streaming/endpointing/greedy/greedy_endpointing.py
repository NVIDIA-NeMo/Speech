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


import torch

from nemo.collections.asr.inference.utils.endpointing_utils import get_custom_stop_history_eou, millisecond_to_frames


class GreedyEndpointing:
    """Greedy endpointing for the streaming ASR pipelines"""

    def __init__(
        self,
        vocabulary: list[str],
        ms_per_timestep: int,
        effective_buffer_size_in_secs: float = None,
        stop_history_eou: int = -1,
        residue_tokens_at_end: int = 0,
        absorb_token_ids: set[int] | None = None,
        stop_history_eou_end: int | None = None,
    ) -> None:
        """
        Initialize the GreedyEndpointing class
        Args:
            vocabulary: (list[str]) List of vocabulary
            ms_per_timestep: (int) Number of milliseconds per timestep
            effective_buffer_size_in_secs: (float, optional) Effective buffer size for VAD-based EOU detection.
            stop_history_eou: (int) Number of silent tokens to trigger a EOU, if -1 then it is disabled
            residue_tokens_at_end: (int) Number of residue tokens at the end, if 0 then it is disabled
            absorb_token_ids: (set[int] | None) Token ids (e.g. punctuation/language tokens) that, when found
                right after the silence gap, are absorbed into the text preceding the EoU instead of starting
                a new utterance. Mirrors the buffered-RNNT `update_punctuation_and_language_tokens_timestamps`.
            stop_history_eou_end: (int | None) Silence threshold (ms) for an EoU detected at the buffer end
                (trailing silence, no following word observed yet). Should be >= `stop_history_eou`; a brief
                mid-word pause at the buffer edge then does not trigger a premature cut. If None, the regular
                `stop_history_eou` is used for end-of-buffer EoUs too (`detect_eou_in_buffer` only).
                A negative value (e.g. -1) disables only *unconfirmed* end-of-buffer EoUs: pure trailing
                silence no longer cuts. Confirmed EoUs still fire -- a following word-start (mid-buffer) or
                trailing punctuation -- so boundaries are only committed when confirmed by a word or punct.
        """

        self.vocabulary = vocabulary
        self.ms_per_timestep = ms_per_timestep
        self.sec_per_timestep = ms_per_timestep / 1000
        self.stop_history_eou = stop_history_eou
        self.stop_history_eou_ms = stop_history_eou
        self.effective_buffer_size_in_secs = effective_buffer_size_in_secs
        if self.stop_history_eou > 0:
            self.stop_history_eou = millisecond_to_frames(self.stop_history_eou, ms_per_timestep)
        self.residue_tokens_at_end = residue_tokens_at_end
        self.absorb_token_ids = set(absorb_token_ids) if absorb_token_ids else set()
        # End-of-buffer silence threshold (frames). None -> fall back to the regular threshold.
        self.stop_history_eou_end = stop_history_eou_end
        if self.stop_history_eou_end is not None and self.stop_history_eou_end > 0:
            self.stop_history_eou_end = millisecond_to_frames(self.stop_history_eou_end, ms_per_timestep)

    def detect_eou_given_emissions(
        self,
        emissions: list[int],
        pivot_point: int,
        search_start_point: int = 0,
        stop_history_eou: int | None = None,
    ) -> tuple[bool, int]:
        """
        Detect end of utterance (EOU) given the emissions and pivot point
        Args:
            emissions (list[int]): list of emissions at each timestep
            pivot_point (int): pivot point around which to detect EOU
            search_start_point (int): start point for searching EOU
            stop_history_eou (int | None): stop history of EOU, if None then use the stop history of EOU from the class
        Returns:
            Tuple[bool, int]: True if EOU is detected, False otherwise, and the point at which EOU is detected
        """
        sequence_length = len(emissions)
        if pivot_point < 0 or pivot_point >= sequence_length:
            raise ValueError("Pivot point is out of range")

        if search_start_point > pivot_point:
            raise ValueError("Search start point is greater than pivot_point")

        if self.residue_tokens_at_end > 0:
            sequence_length = max(0, sequence_length - self.residue_tokens_at_end)

        stop_history_eou = get_custom_stop_history_eou(stop_history_eou, self.stop_history_eou, self.ms_per_timestep)
        eou_detected, eou_detected_at = False, -1

        if stop_history_eou > 0:
            n_silent_tokens = 0
            silence_start_position = -1
            fst_non_silent_token = None
            end_point = max(0, search_start_point, pivot_point - stop_history_eou)
            current_position = max(0, sequence_length - 1)
            while current_position >= end_point:
                if self.is_token_silent(emissions[current_position]):
                    n_silent_tokens += 1
                    eou_detected = n_silent_tokens > stop_history_eou
                    is_token_start_of_word = (fst_non_silent_token is None) or self.is_token_start_of_word(
                        fst_non_silent_token
                    )
                    eou_detected = eou_detected and is_token_start_of_word
                    if eou_detected:
                        silence_start_position = current_position
                else:
                    if eou_detected:
                        break
                    n_silent_tokens = 0
                    eou_detected = False
                    silence_start_position = -1
                    fst_non_silent_token = emissions[current_position]
                current_position -= 1

            eou_detected = n_silent_tokens > stop_history_eou
            if eou_detected:
                eou_detected_at = int(silence_start_position + stop_history_eou // 2)

        return eou_detected, eou_detected_at

    def detect_eou_given_timestamps(
        self,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        alignment_length: int,
        stop_history_eou: int | None = None,
    ) -> tuple[bool, int]:
        """
        Detect end of utterance (EOU) given timestamps and tokens using tensor operations.
        Args:
            timesteps (torch.Tensor): timestamps of the tokens
            tokens (torch.Tensor): tokens
            alignment_length (int): length of the alignment
            stop_history_eou (int | None): stop history of EOU, if None then use the stop history of EOU from the class
        Returns:
            tuple[bool, int]: True if EOU is detected, False otherwise, and the point at which EOU is detected
        """
        eou_detected, eou_detected_at = False, -1

        if len(timesteps) != len(tokens):
            raise ValueError("timesteps and tokens must have the same length")

        stop_history_eou = get_custom_stop_history_eou(stop_history_eou, self.stop_history_eou, self.ms_per_timestep)

        # If stop_history_eou is negative, don't detect EOU.
        if len(timesteps) == 0 or stop_history_eou < 0:
            return eou_detected, eou_detected_at

        # This is the condition for Riva streaming offline mode. The output of entire buffer needs to be sent as is to the client.
        if stop_history_eou == 0:
            return True, alignment_length

        if self.residue_tokens_at_end > 0:
            alignment_length = max(0, alignment_length - self.residue_tokens_at_end)

        # Check trailing silence at the end
        last_timestamp = timesteps[-1].item()
        trailing_silence = max(0, alignment_length - last_timestamp - 1)
        if trailing_silence > stop_history_eou:
            eou_detected = True
            eou_detected_at = last_timestamp + 1 + stop_history_eou // 2
            return eou_detected, eou_detected_at

        # Check gaps between consecutive non-silent tokens
        if len(timesteps) > 1:
            gaps = timesteps[1:] - timesteps[:-1] - 1
            large_gap_mask = gaps > stop_history_eou
            if large_gap_mask.any():
                # Get the last (rightmost) large gap index for backwards compatibility
                large_gap_indices = torch.where(large_gap_mask)[0]
                gap_idx = large_gap_indices[-1].item()

                eou_detected = True
                eou_detected_at = timesteps[gap_idx].item() + 1 + stop_history_eou // 2
                return eou_detected, eou_detected_at
        return eou_detected, eou_detected_at

    def detect_eou_vad(
        self, vad_segments: torch.Tensor, search_start_point: float = 0, stop_history_eou: int | None = None
    ) -> tuple[bool, float]:
        """
        Detect end of utterance (EOU) using VAD segments.

        Args:
            vad_segments (torch.Tensor): VAD segments in format [N, 2] where each row is [start_time, end_time]
            search_start_point (float): Start time for searching EOU in seconds
            stop_history_eou (int | None): Stop history of EOU in milliseconds, if None then use the stop history of EOU from the class
        Returns:
            tuple[bool, float]: (is_eou, eou_detected_at_time)
        """
        if self.effective_buffer_size_in_secs is None:
            raise ValueError("Effective buffer size in seconds is required for VAD-based EOU detection")

        # Use default stop history of EOU from the class if stop_history_eou is not provided
        stop_history_eou = self.stop_history_eou_ms if stop_history_eou is None else stop_history_eou
        if stop_history_eou < 0:
            return False, -1

        search_start_point = search_start_point * self.sec_per_timestep
        stop_history_eou_in_secs = stop_history_eou / 1000
        # Round to 4 decimal places first (vectorized)
        rounded_segments = torch.round(vad_segments, decimals=4)

        # Filter segments where end_time > search_start_point
        valid_mask = rounded_segments[:, 1] > search_start_point
        if not valid_mask.any():
            return False, -1

        filtered_segments = rounded_segments[valid_mask]

        # Clip start times to search_start_point
        filtered_segments[:, 0] = torch.clamp(filtered_segments[:, 0], min=search_start_point)
        # Initialize EOU detection variables
        is_eou = False
        eou_detected_at = -1

        # Check gap to buffer end
        last_segment = filtered_segments[-1]
        gap_to_buffer_end = self.effective_buffer_size_in_secs - last_segment[1]
        if gap_to_buffer_end > stop_history_eou_in_secs:
            # EOU detected at buffer end
            is_eou = True
            eou_detected_at = last_segment[1] + stop_history_eou_in_secs / 2

        elif len(filtered_segments) >= 2:
            # Check gaps between segments (reverse order to find last gap)
            for i in range(len(filtered_segments) - 2, -1, -1):
                segment = filtered_segments[i]
                next_segment = filtered_segments[i + 1]
                gap = next_segment[0] - segment[1]
                if gap > stop_history_eou_in_secs:
                    is_eou = True
                    eou_detected_at = segment[1] + stop_history_eou_in_secs / 2
                    break

        # Convert to timesteps (only if EOU was detected)
        if is_eou:
            eou_detected_at = int(eou_detected_at // self.sec_per_timestep)
        else:
            eou_detected_at = -1

        return is_eou, eou_detected_at

    def is_token_start_of_word(self, token_id: int) -> bool:
        """Check if the token is the start of a word"""
        raise NotImplementedError("Subclass of GreedyEndpointing should implement `is_token_start_of_word` method!")

    def is_token_silent(self, token_id: int) -> bool:
        """Check if the token is silent"""
        raise NotImplementedError("Subclass of GreedyEndpointing should implement `is_token_silent` method!")

    def is_token_to_absorb(self, token_id: int) -> bool:
        """
        Check if the token should be absorbed into the text preceding the EoU (e.g. punctuation or
        language tokens that the model tends to emit late, after the silence gap).
        Args:
            token_id (int): token id
        Returns:
            bool: True if the token must be absorbed into the previous utterance, False otherwise
        """
        return token_id in self.absorb_token_ids

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
