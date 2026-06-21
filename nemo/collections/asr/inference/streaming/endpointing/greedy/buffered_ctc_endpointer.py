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

from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_ctc_decoder import CTCGreedyDecoder
from nemo.collections.asr.inference.streaming.endpointing.greedy.base import GreedyEndpointerBase
from nemo.collections.asr.inference.streaming.endpointing.greedy.token_classifier import TokenClassifier
from nemo.collections.asr.inference.utils.endpointing_utils import get_custom_stop_history_eou


class BufferedCTCEndpointer(GreedyEndpointerBase):
    """Endpointing for the buffered CTC pipeline.

    Detects an EoU by scanning the dense per-timestep emissions of a decoded buffer around a pivot point
    (trailing silence run confirmed by a following word start).
    """

    def __init__(
        self,
        vocabulary: list[str],
        ms_per_timestep: int,
        stop_history_eou: int = -1,
        residue_tokens_at_end: int = 0,
    ) -> None:
        """
        Args:
            vocabulary: (list[str]) List of vocabulary tokens.
            ms_per_timestep: (int) Number of milliseconds per timestep.
            stop_history_eou: (int) Silence (ms) to trigger an EoU; -1 disables it.
            residue_tokens_at_end: (int) Number of residue tokens at the end; 0 disables it.
        """
        super().__init__(
            TokenClassifier.for_ctc(vocabulary),
            ms_per_timestep,
            stop_history_eou=stop_history_eou,
            residue_tokens_at_end=residue_tokens_at_end,
        )

    def detect_eou(
        self,
        probs_seq: torch.Tensor,
        pivot_point: int,
        search_start_point: int = 0,
        stop_history_eou: int | None = None,
    ) -> tuple[bool, int]:
        """
        Detect end of utterance (EOU) given the probabilities sequence and pivot point.
        Args:
            probs_seq (torch.Tensor): probabilities sequence
            pivot_point (int): pivot point
            search_start_point (int): start point for searching EOU
            stop_history_eou (int | None): stop history of EOU, if None then use the stop history of EOU from the class
        Returns:
            bool: True if EOU is detected, False otherwise
            int: index of the EOU detected at
        """
        emissions = CTCGreedyDecoder.get_labels(probs_seq)
        return self.detect_eou_given_emissions(emissions, pivot_point, search_start_point, stop_history_eou)

    def detect_eou_given_emissions(
        self,
        emissions: list[int],
        pivot_point: int,
        search_start_point: int = 0,
        stop_history_eou: int | None = None,
    ) -> tuple[bool, int]:
        """
        Detect end of utterance (EOU) given the emissions and pivot point.
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
