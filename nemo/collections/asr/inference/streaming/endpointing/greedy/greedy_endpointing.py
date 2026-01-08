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
    ) -> None:
        """
        Initialize the GreedyEndpointing class
        Args:
            vocabulary: (list[str]) List of vocabulary
            ms_per_timestep: (int) Number of milliseconds per timestep
            effective_buffer_size_in_secs: (float, optional) Effective buffer size for VAD-based EOU detection.
            stop_history_eou: (int) Number of silent tokens to trigger a EOU, if -1 then it is disabled
            residue_tokens_at_end: (int) Number of residue tokens at the end, if 0 then it is disabled
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

    def detect_eou_near_pivot(
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
            tuple[bool, int]: True if EOU is detected, False otherwise, and the point at which EOU is detected
        """

        sequence_length = len(emissions)

        if pivot_point < 0 or pivot_point >= sequence_length:
            raise ValueError("Pivot point is out of range")

        if search_start_point > pivot_point:
            raise ValueError("Search start point is greater then pivot_point")

        if self.residue_tokens_at_end > 0:
            sequence_length = max(0, sequence_length - self.residue_tokens_at_end)

        stop_history_eou = get_custom_stop_history_eou(stop_history_eou, self.stop_history_eou, self.ms_per_timestep)
        eou_detected, eou_detected_at = False, -1

        if stop_history_eou > 0:

            # number of silent tokens in the range [search_start_point, pivot_point)
            n_silent_tokens_before = 0
            i = pivot_point - 1
            while i >= search_start_point:
                if self.is_token_silent(emissions[i]):
                    n_silent_tokens_before += 1
                else:
                    break
                i -= 1

            # number of silent tokens in the range [pivot_point, sequence_length)
            n_silent_tokens_after = 0
            i = pivot_point
            fst_non_silent_token_after = None
            while i < sequence_length:
                if self.is_token_silent(emissions[i]):
                    n_silent_tokens_after += 1
                else:
                    fst_non_silent_token_after = emissions[i]
                    break
                i += 1

            # additional check for the first non-silent token after the pivot point
            if fst_non_silent_token_after is not None:
                if not self.is_token_start_of_word(fst_non_silent_token_after):
                    eou_detected, eou_detected_at = False, -1
            else:
                # check if the number of silent tokens before and after the pivot point is greater than the threshold
                val_cnt = n_silent_tokens_before + n_silent_tokens_after
                eou_detected = val_cnt > stop_history_eou
                eou_detected_at = (
                    int(pivot_point - n_silent_tokens_before + stop_history_eou // 2) if eou_detected else -1
                )

        return eou_detected, eou_detected_at
