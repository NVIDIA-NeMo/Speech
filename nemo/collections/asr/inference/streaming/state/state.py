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

from nemo.collections.asr.inference.streaming.framing.request import RequestOptions
from nemo.collections.asr.inference.utils.constants import POST_WORD_PUNCTUATION
from nemo.collections.asr.inference.utils.state_management_utils import (
    detect_overlap,
    merge_segment_tail,
    merge_timesteps,
    merge_word_tail,
)
from nemo.collections.asr.inference.utils.text_segment import TextSegment, Word

CLOSE_IN_TIME_TH = 2.0
OVERLAP_SEARCH_TH = 3


class StreamingState:
    """
    Generic state for the streaming ASR pipeline
    """

    def __init__(self):
        """
        Initialize the StreamingState
        """
        self._reset_streaming_state()

    def reset(self) -> None:
        """
        Reset the state to its initial values
        """
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """
        Initialize the state with default values
        """

        # Global offset is used to keep track of the timestamps
        self.global_offset = 0

        # All tokens, timestamps and conf scores that have been processed since the last EOU
        self.tokens = []
        self.timesteps = []
        self.confidences = []

        # Predicted tokens for the current step
        self.current_step_tokens = []

        # Last token and its index are used to detect overlap between the current and the previous output
        self.last_token = None
        self.last_token_idx = None

        # Tokens left in the right padding segment of the buffer
        self.incomplete_segment_tokens = []

        # final_transcript, partial_transcript, current_step_transcript and final_segments will be sent to the client
        self.final_transcript = ""
        self.partial_transcript = ""
        self.current_step_transcript = ""
        self.concat_with_space = True
        self.final_segments = []

        # Word-level ASR output attributes (cleared after cleanup_after_response):
        # - words: Raw word-level ASR output
        # - pnc_words: Words with punctuation and capitalization applied
        #   * When automatic punctuation is ENABLED: Contains punctuation marks and capitalization
        #     (from either external PnC model or built-in ASR model PnC)
        #   * When automatic punctuation is DISABLED: No punctuation or capitalization
        #     (any punctuation in raw ASR output will be removed)
        # - itn_words: Words after applying both PnC and ITN (Inverse Text Normalization)
        # - word_alignment: ITN word alignment
        # Segment-level ASR output attributes (cleared after cleanup_after_response):
        # - segments: Raw segment-level ASR output
        # - processed_segment_mask: Mask indicating which segments have been processed
        # - final_segments: Final segment-level ASR output
        self.words = []
        self.pnc_words = []
        self.itn_words = []
        self.word_alignment = []
        self.segments = []
        self.processed_segment_mask = []

        # Flag to indicate if EOU was detected before, used in merging logic
        self.eou_detected_before = False

        # Used in EoU detection logic
        self.decoder_start_idx = 0
        self.decoder_end_idx = 0

        # Request options
        self.options = None

    def set_options(self, options: RequestOptions) -> None:
        """
        Set the options
        Args:
            options: (RequestOptions) The request options to store in the state
        """
        self.options = options

    def set_incomplete_segment_tokens(self, incomplete_segment_tokens: list) -> None:
        """
        Set the partial tokens
        Args:
            incomplete_segment_tokens: (list) The partial tokens to store in the state
        """
        self.incomplete_segment_tokens = incomplete_segment_tokens

    def set_global_offset(self, start_offset: float) -> None:
        """
        Set the global offset
        Args:
            start_offset: (float) The global offset to store in the state
        """
        self.global_offset = start_offset

    def set_last_token(self, token: int | None, idx: int | None) -> None:
        """
        Set the last token
        Args:
            token: (int | None) The last token to store in the state
            idx: (int | None) The index of the last token to store in the state
        """
        if None not in [token, idx]:
            self.last_token_idx = idx + self.global_offset
            self.last_token = token
        else:
            self.last_token_idx = None
            self.last_token = None

    def increment_global_offset(self, shift: float) -> None:
        """
        Increment the global offset by the given shift
        Args:
            shift: (float) The shift to increment the global offset by
        """
        self.global_offset += shift

    def _update_state(self, output: dict, skip: int) -> None:
        """
        Extend the tokens, timesteps and confidences, optionally skipping the first few tokens
        Args:
            output: (dict) The output to update the state with
            skip: (int) The number of tokens to skip
        """
        current_tokens = output["tokens"]
        current_timesteps = output["timesteps"]
        current_confidences = output["confidences"]
        if skip > 0:
            current_tokens = current_tokens[skip:]
            current_timesteps = current_timesteps[skip:]
            current_confidences = current_confidences[skip:]

        self.current_step_tokens.extend(current_tokens)
        self.tokens.extend(current_tokens)
        self.confidences.extend(current_confidences)
        self.timesteps = merge_timesteps(self.timesteps, current_timesteps)

    def update_state(self, completed_output: dict, eou_detected: bool) -> None:
        """
        Update the state with the completed output
        Args:
            completed_output: (dict) The completed output to update the state with
            eou_detected: (bool) Whether EOU was detected
        """

        if len(completed_output) == 0 or len(completed_output["tokens"]) == 0:
            self.last_token = None
            self.last_token_idx = None
            return

        timesteps = completed_output["timesteps"]
        for i, t in enumerate(timesteps):
            timesteps[i] = t + self.global_offset

        overlap = 0
        if not self.eou_detected_before:
            overlap = detect_overlap(
                state_tokens=self.tokens,
                state_timesteps=self.timesteps,
                new_tokens=completed_output["tokens"],
                new_timesteps=timesteps,
                overlap_search_th=OVERLAP_SEARCH_TH,
                close_in_time_th=CLOSE_IN_TIME_TH,
            )

        # In case when the tokens are empty after EoU,
        # we need to check if the last token is the same as the first token of the completed output
        if (
            self.eou_detected_before
            and self.last_token == completed_output["tokens"][0]
            and self.last_token_idx is not None
            and abs(self.last_token_idx - timesteps[0]) <= CLOSE_IN_TIME_TH
        ):
            overlap = max(overlap, 1)

        self._update_state(completed_output, overlap)
        self.eou_detected_before = eou_detected

    def update_from_decoder_results(self, start_idx: int, end_idx: int) -> None:
        """
        Update state based on decoder results
        This is used to dynamically understand current token start and end indices
        Args:
            start_idx: (int) The start index of the decoder results
            end_idx: (int) The end index of the decoder results
        """
        self.decoder_start_idx = start_idx
        self.decoder_end_idx = end_idx

    def cleanup_after_eou(self) -> None:
        """
        Cleanup the state after an EOU is detected
        """
        self.tokens.clear()
        self.timesteps.clear()
        self.confidences.clear()

    def cleanup_after_response(self) -> None:
        """
        Cleanup the state after a response is sent
        Specifically used to clean the state after final transcript is sent
        """

        if self.options.is_word_level_output():
            self.words.clear()
            self.pnc_words.clear()
            self.itn_words.clear()
            self.word_alignment.clear()
        else:
            self.segments.clear()
            self.processed_segment_mask.clear()

        self.final_transcript = ""
        self.final_segments.clear()
        self.current_step_transcript = ""
        self.current_step_tokens.clear()
        self.concat_with_space = True

    def push_back_segment(
        self,
        segment: TextSegment,
        need_merge: bool,
        conf_aggregator: Callable = None,
    ) -> None:
        """
        Push back the decoded segment to the state
        Args:
            segment: (TextSegment) The decoded segment to push back to the state
            need_merge: (bool) Whether to merge the segment with the last segment in the state
            conf_aggregator: (Callable) The function to aggregate the confidence
        """

        # concat_with_space is used to determine if the final transcript should be concatenated with a space
        if len(self.final_segments) == 0 and need_merge:
            self.concat_with_space = False
        else:
            self.concat_with_space = True

        if need_merge and len(self.segments) > 0:
            head = merge_segment_tail(
                segment_head=self.segments[-1],
                segment_tail=segment,
                conf_aggregator=conf_aggregator,
            )
            self.segments[-1] = head
            self.processed_segment_mask[-1] = False
        else:
            self.segments.append(segment)
            self.processed_segment_mask.append(False)

    def push_back_words(
        self,
        decoded_words: list[Word],
        merge_first_word: bool = False,
        merge_first_word_punctuation: bool = True,
        conf_aggregator: Callable = None,
    ) -> None:
        """
        Push back the decoded words to the state
        Args:
            decoded_words: (list[Word]) The decoded words to push back to the state
            merge_first_word: (bool) Whether to merge the first word with the last word in the state
            merge_first_word_punctuation: (bool) Whether to merge the first word punctuation with the last word in the state
            conf_aggregator: (Callable) The function to aggregate the confidence
        """
        if not decoded_words:
            return

        # concat_with_space is used to determine if the final transcript should be concatenated with a space
        if len(self.final_segments) == 0 and merge_first_word:
            self.concat_with_space = False
        else:
            self.concat_with_space = True

        if (
            (fst_word_txt := decoded_words[0].text)
            and fst_word_txt in POST_WORD_PUNCTUATION
            and merge_first_word_punctuation
        ):
            # if the first word is a punctuation mark, merge it with the last word stored in the state
            if len(self.words) > 0:
                self.words[-1].text += fst_word_txt
            decoded_words = decoded_words[1:]

        elif merge_first_word and len(self.words) > 0:
            head, pnc_head = merge_word_tail(
                word_head=self.words[-1],
                word_tail=decoded_words[0],
                pnc_word_head=self.pnc_words[-1] if len(self.pnc_words) > 0 else None,
                conf_aggregator=conf_aggregator,
            )
            self.words[-1] = head
            if pnc_head is not None:
                self.pnc_words[-1] = pnc_head
            decoded_words = decoded_words[1:]

        self.words.extend(decoded_words)
