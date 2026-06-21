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

from nemo.collections.asr.inference.streaming.endpointing.greedy.base import GreedyEndpointerBase
from nemo.collections.asr.inference.streaming.endpointing.greedy.token_classifier import TokenClassifier
from nemo.collections.asr.inference.utils.endpointing_utils import get_custom_stop_history_eou


class BufferedRNNTEndpointer(GreedyEndpointerBase):
    """Endpointing for the buffered RNNT pipeline.

    Detects an EoU from the decoded token timestamps (trailing silence / gap between tokens) or, when VAD
    segments are available, from the silence gaps between VAD segments.
    """

    def __init__(
        self,
        vocabulary: list[str],
        ms_per_timestep: int,
        effective_buffer_size_in_secs: float = None,
        stop_history_eou: int = -1,
        residue_tokens_at_end: int = 0,
    ) -> None:
        """
        Args:
            vocabulary: (list[str]) List of vocabulary tokens.
            ms_per_timestep: (int) Number of milliseconds per timestep.
            effective_buffer_size_in_secs: (float, optional) Effective buffer size for VAD-based EoU
                detection (stateless / stateful RNNT). If None, VAD functionality is disabled.
            stop_history_eou: (int) Silence (ms) to trigger an EoU; -1 disables it.
            residue_tokens_at_end: (int) Number of residue tokens at the end; 0 disables it.
        """
        super().__init__(
            TokenClassifier.for_rnnt(vocabulary),
            ms_per_timestep,
            stop_history_eou=stop_history_eou,
            residue_tokens_at_end=residue_tokens_at_end,
        )
        self.effective_buffer_size_in_secs = effective_buffer_size_in_secs

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
