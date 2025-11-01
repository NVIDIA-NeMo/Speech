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


from typing import Callable
import torch

from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_decoder import GreedyDecoder


class CTCGreedyDecoder(GreedyDecoder):
    """CTC Greedy decoder class"""

    def __init__(self, vocabulary: list[str], conf_func: Callable = None):
        """
        Initialize the CTCGreedyDecoder
        Args:
            vocabulary (list[str]): list of vocabulary tokens
            conf_func (Callable): function to compute confidence
        """

        super().__init__(vocabulary, conf_func)

    @staticmethod
    def get_labels(log_probs: torch.Tensor) -> list[int]:
        """
        Perform greedy decoding on the log probabilities
        Args:
            log_probs (torch.Tensor): log probabilities
        Returns:
            list[int]: list of tokens
        """
        if log_probs.dim() != 2:
            raise ValueError("log_probs must be 2D tensor")

        labels = log_probs.argmax(dim=-1).cpu()  # T
        return labels.tolist()

    def __call__(self, log_probs: torch.Tensor, compute_confidence: bool = True, previous: int = None) -> dict:
        """
        Greedy decode the log probabilities
        Args:
            log_probs (torch.Tensor): log probabilities
            compute_confidence (bool): compute confidence or not
        Returns:
            dict: output dictionary containing tokens, timesteps, and confidences
        """

        compute_confidence = compute_confidence and self.conf_func is not None

        if log_probs.dim() != 2:
            raise ValueError("log_probs must be 2D tensor")

        if compute_confidence:
            # Add batch dimension
            log_probs = log_probs.unsqueeze(0)  # 1 x T x N
            # Compute confidences
            confidences = torch.zeros(log_probs.shape[0], log_probs.shape[1])  # 1 x T
            confidences[0] = self.conf_func(log_probs[0], v=log_probs.shape[2])  # 1 x T
            # Remove batch dimension and convert to list
            confidences = confidences.squeeze(0).tolist()  # T
            # Remove batch dimension
            log_probs = log_probs.squeeze(0)  # T x N

        labels = self.get_labels(log_probs)  # T
        output = {"tokens": [], "timesteps": [], "confidences": []}
        previous = self.blank_id if previous is None else previous
        for i, p in enumerate(labels):
            if p != previous and p != self.blank_id:
                output["tokens"].append(p)
                output["timesteps"].append(i)
                if compute_confidence:
                    output["confidences"].append(confidences[i])
            previous = p

        output["labels"] = labels
        return output


class ClippedCTCGreedyDecoder:
    """
    Clipped CTC Greedy decoder class
    Decodes the tokens within a given clip range and returns the clipped tokens and timestamps.
    """

    def __init__(self, vocabulary: list[str], tokens_per_frame: int, conf_func: Callable = None, endpointer=None):
        """
        Initialize the ClippedCTCGreedyDecoder
        Args:
            vocabulary (list[str]): list of vocabulary tokens
            tokens_per_frame (int): number of tokens per frame
            conf_func (Callable): function to compute confidence
            endpointer (Any): endpointer to detect EOU
        """
        self.greedy_decoder = CTCGreedyDecoder(vocabulary, conf_func)
        self.endpointer = endpointer
        self.tokens_per_frame = tokens_per_frame

    def __call__(
        self,
        log_probs: torch.Tensor,
        clip_start: int,
        clip_end: int,
        is_last: bool = False,
        is_start: bool = True,
        return_partial_result: bool = True,
        state_start_idx: int = 0,
        state_end_idx: int = 0,
        stop_history_eou: int = None,
        compute_confidence: bool = True,
    ) -> tuple[dict, dict, bool, int, int]:
        """
        Decode the log probabilities within the clip range (clip_start, clip_end)
        Args:
            log_probs (torch.Tensor): log probabilities
            clip_start (int): start index of the clip
            clip_end (int): end index of the clip
            is_last (bool): is the last frame or not
            is_start (bool): is the first frame for this stream or not
            return_partial_result (bool): return partial result left after clip_end in the buffer
            state_start_idx (int): start index from stream state
            state_end_idx (int): end index from stream state
            stop_history_eou (int): stop history of EOU, if None then use the default stop history
            compute_confidence (bool): compute confidence or not
        Returns:
            tuple[dict, dict, bool, int, int]:
                clipped output, tail output, is_eou, updated start_idx, updated end_idx
        """

        is_eou = is_last
        eou_detected_at = len(log_probs)
        # Initialize state tracking variables from input parameters
        start_idx, end_idx = state_start_idx, state_end_idx
        # Update indices for next processing step
        if end_idx > clip_start:
            end_idx -= self.tokens_per_frame
            start_idx = end_idx

        if is_start or end_idx <= clip_start:
            start_idx, end_idx = clip_start, clip_end

        all_output = self.greedy_decoder(log_probs, compute_confidence=compute_confidence)

        clipped_output = {"tokens": [], "timesteps": [], "confidences": [], "last_token": None, "last_token_idx": None}
        tail_output = {"tokens": []}

        # check if EOU is detected or is the last frame
        if not is_eou and self.endpointer is not None:
            is_eou, eou_detected_at = self.endpointer.detect_eou(
                log_probs, pivot_point=start_idx, search_start_point=clip_start, stop_history_eou=stop_history_eou
            )

        # if EOU is detected, and it is after the clip end, update the end index to the EOU
        if is_eou and eou_detected_at > end_idx:
            end_idx = eou_detected_at

        # if the end index is within the clip range, update the end index to the clip end
        if clip_start <= end_idx < clip_end:
            end_idx = clip_end
            is_eou = False

        # clip the output within the clip range [clip_start, clip_end)
        timesteps = all_output["timesteps"]
        i = 0
        while i < len(timesteps):
            if start_idx <= timesteps[i] < end_idx:
                clipped_output["tokens"].append(all_output["tokens"][i])
                clipped_output["timesteps"].append(timesteps[i])
                if compute_confidence:
                    clipped_output["confidences"].append(all_output["confidences"][i])
            elif timesteps[i] >= end_idx:
                break
            i += 1

        if end_idx - 1 < len(all_output["labels"]):
            clipped_output["last_token"] = all_output["labels"][end_idx - 1]
            clipped_output["last_token_idx"] = end_idx - 1

        # return the partial result left after clip_end in the buffer
        if return_partial_result:
            while i < len(timesteps):
                if timesteps[i] >= end_idx:
                    tail_output["tokens"] = all_output["tokens"][i:]
                    break
                else:
                    i += 1

        return clipped_output, tail_output, is_eou, start_idx, end_idx
