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

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

from omegaconf import DictConfig

from nemo.collections.asr.inference.streaming.state.state import StreamingState
from nemo.collections.asr.inference.utils.constants import POST_WORD_PUNCTUATION
from nemo.collections.asr.inference.utils.pipeline_utils import (
    get_leading_punctuation_regex_pattern,
    get_repeated_punctuation_regex_pattern,
)
from nemo.collections.asr.inference.utils.text_segment import Word

if TYPE_CHECKING:
    from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer


class StreamingTextProcessor:
    """
    A streaming text post-processing module to standardize the ASR output.
    """

    def __init__(
        self,
        itn_cfg: DictConfig,
        itn_model: AlignmentPreservingInverseNormalizer | None,
        asr_supported_puncts: set,
        asr_supports_punctuation: bool,
        confidence_aggregator: Callable,
        sep: str,
        enable_itn: bool = False,
    ):
        """
        Initialize the streaming text processor.

        Args:
            itn_cfg (DictConfig): ITN parameters.
            itn_model (AlignmentPreservingInverseNormalizer | None): Model for inverse text normalization (ITN).
            asr_supported_puncts (set): Set of punctuation marks recognized by the ASR model.
            asr_supports_punctuation (bool): Boolean indicating if the ASR model outputs punctuation.
            confidence_aggregator (Callable): Function for aggregating confidence scores.
            sep (str): String separator used in ASR output processing.
            enable_itn (bool): Boolean to enable ITN. Default is False.
        """

        self.supports_punctuation = asr_supports_punctuation

        self.itn_model = itn_model
        self.itn_enabled = False
        if enable_itn:
            self.itn_enabled = itn_model is not None

        self.itn_runtime_params = {
            "batch_size": itn_cfg.batch_size,
            "n_jobs": itn_cfg.n_jobs,
        }
        self.itn_left_padding_size = itn_cfg.left_padding_size

        self.asr_supported_puncts = asr_supported_puncts
        self.asr_supported_puncts_str = ''.join(self.asr_supported_puncts)
        self.sep = sep

        puncts_to_process = self.asr_supported_puncts
        self.leading_punctuation_regex_pattern = get_leading_punctuation_regex_pattern(puncts_to_process)
        self.repeated_punctuation_regex_pattern = get_repeated_punctuation_regex_pattern(puncts_to_process)

        self.alignment_aware_itn_model = None
        if self.itn_enabled:
            from nemo.collections.asr.inference.itn.batch_inverse_normalizer import (
                BatchAlignmentPreservingInverseNormalizer,
            )

            self.alignment_aware_itn_model = BatchAlignmentPreservingInverseNormalizer(
                itn_model=self.itn_model,
                sep=self.sep,
                asr_supported_puncts=self.asr_supported_puncts,
                post_word_punctuation=POST_WORD_PUNCTUATION,
                conf_aggregate_fn=confidence_aggregator,
            )

    def process(self, states: list[StreamingState]) -> None:
        """
        Post-process the states.
        Args:
            states: (list[StreamingState]) List of StreamingState objects
        """
        word_boundary_states, segment_boundary_states = [], []
        for state in states:
            if state.options.is_word_level_output():
                word_boundary_states.append(state)
            else:
                segment_boundary_states.append(state)

        # Process states with word boundaries
        if word_boundary_states:
            self.process_states_with_word_boundaries(word_boundary_states)

        # Process states with segment boundaries
        if segment_boundary_states:
            self.process_states_with_segment_boundaries(segment_boundary_states)

        # Generate final transcript
        self.generate_final_transcript(word_boundary_states, segment_boundary_states)

    def process_states_with_segment_boundaries(self, states: list[StreamingState]) -> None:
        """
        Post-process the states with segment boundaries.
        Args:
            states (list[StreamingState]): List of StreamingState objects that have segments
        """
        states_with_text = [state for state in states if len(state.segments) > 0]
        if len(states_with_text) == 0:
            return

        # Apply ITN
        if self.itn_enabled:
            # collect texts
            texts = []
            for i, state in enumerate(states_with_text):
                # if ITN is disabled for this state
                if not state.options.enable_itn:
                    continue

                for j, seg in enumerate(state.segments):
                    if state.processed_segment_mask[j]:  # if the segment is already processed, skip it
                        continue
                    texts.append((i, j, seg.text))

            if len(texts) > 0:
                # apply ITN
                processed_texts = self.itn_model.inverse_normalize_list(
                    texts=[text for _, _, text in texts], params=self.itn_runtime_params
                )
                # update states with ITN-processed texts
                for (i, j, _), processed_text in zip(texts, processed_texts):
                    states_with_text[i].segments[j].text = processed_text

        # mark all segments as processed
        for state in states_with_text:
            if self.supports_punctuation:
                for seg in state.segments:
                    if self.leading_punctuation_regex_pattern:
                        seg.text = re.sub(self.leading_punctuation_regex_pattern, r'\1', seg.text)
                    if self.repeated_punctuation_regex_pattern:
                        seg.text = re.sub(self.repeated_punctuation_regex_pattern, r'\1', seg.text)
            state.processed_segment_mask = [True] * len(state.segments)

    def process_states_with_word_boundaries(self, states: list[StreamingState]) -> None:
        """
        Post-process the states with word boundaries.
        Args:
            states: (list[StreamingState]) List of StreamingState objects
        """
        # Get the indices of the states that have new words to process
        indices, asr_words_list = self.prepare_asr_words(states)

        # Keep the words as is
        for idx, jdx, z in indices:
            states[idx].pnc_words[-z:] = asr_words_list[jdx][-z:]

        # If ITN is disabled globally, do nothing
        if not self.itn_enabled:
            return

        # Apply Inverse Text Normalization (ITN)
        self.apply_itn(states, indices)

    def prepare_asr_words(self, states: list[StreamingState]) -> tuple[list[tuple], list[list[Word]]]:
        """
        Find the indices of the states that have words to process.
        Args:
            states: (list[StreamingState]) List of StreamingState objects
        Returns:
            tuple[list[tuple], list[list[Word]]]:
                indices: list of indices of the states that have words to process
                asr_words_list: list of words to process
        """
        indices, asr_words_list = [], []

        jdx = 0
        for idx, state in enumerate(states):
            if (n_not_punctuated_words := len(state.words) - len(state.pnc_words)) == 0:
                continue

            words_list = [word.copy() for word in state.words[-n_not_punctuated_words:]]
            asr_words_list.append(words_list)
            state.pnc_words.extend([None] * n_not_punctuated_words)
            indices.append((idx, jdx, len(words_list)))
            jdx += 1

        return indices, asr_words_list

    def apply_itn(self, states: list[StreamingState], indices: list[tuple]) -> None:
        """
        Apply Inverse Text Normalization (ITN) on the states.
        Calculates the lookback for ITN and updates the states with the ITN results.
        Args:
            states: (list[StreamingState]) List of StreamingState objects
            indices: (list[tuple]) List of indices of the states that have words to process
        """
        itn_indices, asr_words_list, pnc_words_list = [], [], []
        jdx = 0
        for state_idx, _, _ in indices:
            state = states[state_idx]
            if not state.options.enable_itn:
                continue
            s, t, cut_point = self.calculate_itn_lookback(state)
            asr_words_list.append([word.copy() for word in state.words[s:]])
            pnc_words_list.append([word.copy() for word in state.pnc_words[s:]])
            itn_indices.append((state_idx, jdx, s, t, cut_point))
            jdx += 1
        output = self.alignment_aware_itn_model(
            asr_words_list, pnc_words_list, self.itn_runtime_params, return_alignment=True
        )
        self.update_itn_words(states, output, itn_indices)

    def calculate_itn_lookback(self, state: StreamingState) -> tuple[int, int, int]:
        """
        Calculate the lookback for ITN.
        Args:
            state: (StreamingState) StreamingState object
        Returns:
            Start index (int): Start index of the source (non itn-ed) words
            Target index (int): Start index of the target (itn-ed) words
            Cut point (int): Index to cut the source words
        """
        s, t, cut_point = 0, 0, len(state.itn_words)
        word_alignment = list(reversed(state.word_alignment))
        for idx, (sidx, tidx, _) in enumerate(word_alignment, start=1):
            s, t = sidx[0], tidx[0]
            state.word_alignment.pop()
            cut_point -= 1
            if idx == self.itn_left_padding_size:
                break
        return s, t, cut_point

    @staticmethod
    def update_itn_words(states: list[StreamingState], output: list[tuple], indices: list[tuple]) -> None:
        """
        Update the states with the ITN results.
        Updates the word_alignment and itn_words in the states.
        Args:
            states: (list[StreamingState]) List of StreamingState objects
            output: (list[tuple]) List of output tuples containing the spans and alignment
            indices: (list[tuple]) List of indices of the states that have words to process
        """
        for state_idx, jdx, s, t, cut_point in indices:
            state = states[state_idx]
            spans, alignment = output[jdx]
            for sidx, tidx, sclass in alignment:
                sidx = [k + s for k in sidx]
                tidx = [k + t for k in tidx]
                state.word_alignment.append((sidx, tidx, sclass))

            state.itn_words = state.itn_words[:cut_point] + spans
            assert len(state.word_alignment) == len(state.itn_words)

    def generate_final_transcript(
        self, word_boundary_states: list[StreamingState], segment_boundary_states: list[StreamingState]
    ) -> None:
        """
        Generate final transcript based on enabled features and word count.
        Args:
            word_boundary_states (list[StreamingState]): The streaming state containing words
            segment_boundary_states (list[StreamingState]): The streaming state containing segments
        """
        # Generate final transcript for word boundary states
        for state in word_boundary_states:
            attr_name = "itn_words" if state.options.enable_itn else "pnc_words"
            words = getattr(state, attr_name)
            for word in words:
                state.final_segments.append(word.copy())
                state.final_transcript += word.text + self.sep
            state.final_transcript = state.final_transcript.rstrip(self.sep)

        # Generate final transcript for segment boundary states
        for state in segment_boundary_states:
            for segment in state.segments:
                state.final_segments.append(segment.copy())
                state.final_transcript += segment.text + self.sep
            state.final_transcript = state.final_transcript.rstrip(self.sep)
