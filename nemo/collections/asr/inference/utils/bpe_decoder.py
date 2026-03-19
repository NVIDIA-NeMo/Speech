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


from functools import lru_cache
from typing import Callable

import numpy as np

from nemo.collections.asr.inference.streaming.state.state import StreamingState
from nemo.collections.asr.inference.utils.constants import (
    POST_WORD_PUNCTUATION,
    ROUND_PRECISION,
    SENTENCEPIECE_UNDERSCORE,
)
from nemo.collections.asr.inference.utils.text_segment import TextSegment, Word
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class BPEDecoder:
    """
    BPEDecoder class for decoding BPE (Byte Pair Encoding) tokens into words and segments by preserving timestamps and confidence scores
    """

    def __init__(
        self,
        vocabulary: list[str],
        tokenizer: TokenizerSpec,
        confidence_aggregator: Callable,
        asr_supported_puncts: set,
        word_boundary_tolerance: float,
        token_duration_in_secs: float,
    ):
        """
        Initialize the BPEDecoder.
        Args:
            vocabulary (list[str]): List of vocabulary tokens.
            tokenizer (TokenizerSpec): Tokenizer object.
            confidence_aggregator (Callable): Confidence aggregator function.
            asr_supported_puncts (set): Set of supported punctuation symbols.
            word_boundary_tolerance (float): Word boundary tolerance for timestamp refinement.
            token_duration_in_secs (float): Token duration in seconds.
        """

        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.confidence_aggregator = confidence_aggregator
        self.asr_supported_puncts = asr_supported_puncts
        self.punct_marks_with_underscore = asr_supported_puncts.union({SENTENCEPIECE_UNDERSCORE})
        self.word_boundary_tolerance = word_boundary_tolerance
        self.token_duration_in_secs = token_duration_in_secs
        self.start_of_word_cache = {
            token_id: token.startswith(SENTENCEPIECE_UNDERSCORE) for token_id, token in enumerate(self.vocabulary)
        }
        self.punct_cache = {
            token_id: (token in self.asr_supported_puncts, token in self.punct_marks_with_underscore)
            for token_id, token in enumerate(self.vocabulary)
        }

    @lru_cache(maxsize=10000)
    def cached_ids_to_text(self, tokens_slice: tuple[int]) -> str:
        """
        Cached tokenizer output to avoid repeated calls to the tokenizer.
        Args:
            tokens_slice (tuple): Tuple of token indices to be detokenized.
        Returns:
            str: Detokenized text.
        """
        word_text = self.tokenizer.ids_to_text(list(tokens_slice)).strip()
        return word_text

    def decode_bpe_tokens(self, state: StreamingState) -> None:
        """
        Decodes BPE tokens into words or segments with timestamps and confidence scores.
        Args:
            state (StreamingState): The state object containing the BPE tokens, timestamps, and confidence scores.
        """
        if state.options.is_word_level_output():
            # Form words and push them to the state
            decoded_words, need_merge = self.group_tokens_into_words(state.tokens, state.timesteps, state.confidences)
            state.push_back_words(decoded_words, need_merge, self.confidence_aggregator)
        elif state.options.is_segment_level_output():
            # Form text segment and push it to the state
            if state.tokens:
                decoded_segment, need_merge = self.group_tokens_into_segment(
                    state.tokens, state.timesteps, state.confidences
                )
                state.push_back_segment(decoded_segment, need_merge, self.confidence_aggregator)
        else:
            raise ValueError(f"Invalid output granularity: {state.options.asr_output_granularity}")

    def group_tokens_into_segment(
        self, tokens: list, timesteps: list, confidences: list
    ) -> tuple[TextSegment | None, bool]:
        """
        Group tokens into a text segment with timestamps and confidence scores.
        Args:
            tokens (list): List of token indices.
            timesteps (list): List of token timestamps.
            confidences (list): List of token confidence scores.
        Returns:
            (tuple[TextSegment | None, bool]) Text segment with text, start time, end time, and confidence score.
            Also returns a boolean to indicate if the text segment should be merged with the last segment stored in the state
        """
        n_tokens = len(tokens)

        if n_tokens != len(timesteps) or n_tokens != len(confidences):
            raise ValueError("tokens, timesteps and confidences must have the same length")

        if n_tokens == 0:
            return None, False

        need_merge = not bool(self.start_of_word_cache[tokens[0]])

        # Get the segment text
        segment_text = self.tokenizer.ids_to_text(tokens).strip()

        # Refine the start and end timestamps of the text segment
        start, end = self.refine_text_segment_timestamp(tokens, timesteps)

        # Convert token timestamps to seconds
        start = round(start * self.token_duration_in_secs, ROUND_PRECISION)
        end = round(end * self.token_duration_in_secs, ROUND_PRECISION)

        # Aggregate the confidence score of the text segment
        conf = self.confidence_aggregator(confidences)

        # Create a text segment
        return TextSegment(text=segment_text, start=start, end=end, conf=conf), need_merge

    def group_tokens_into_words(self, tokens: list, timesteps: list, confidences: list) -> tuple[list[Word], bool]:
        """
        Decodes BPE tokens into words with timestamps and confidence scores.
        Args:
            tokens (list): List of token indices.
            timesteps (list): List of token timesteps.
            confidences (list): List of token confidence scores.
        Returns:
            (tuple[list[Word], bool]) List of decoded words with text, start time, end time, and confidence score.
            Also returns a boolean to indicate if the first word should be merged with the last word stored in the state
        """
        n_tokens = len(tokens)

        if n_tokens != len(timesteps) or n_tokens != len(confidences):
            raise ValueError("tokens, timesteps and confidences must have the same length")

        if n_tokens == 0:
            return [], False

        # Group tokens into words
        is_start_mask = np.fromiter((self.start_of_word_cache[tok] for tok in tokens), dtype=np.int32)
        word_ids = np.cumsum(is_start_mask)

        start_indices = np.nonzero(np.diff(word_ids, prepend=word_ids[0] - 1))[0]
        end_indices = np.append(start_indices[1:], n_tokens)

        decoded_words, prev_word_end = [], None

        # If the first word is the start of a word, we need to merge it with the last word stored in the state
        need_merge = not bool(is_start_mask[0])

        for start_idx, end_idx in zip(start_indices, end_indices):

            tokens_slice = tokens[start_idx:end_idx]
            time_slice = timesteps[start_idx:end_idx]
            conf_slice = confidences[start_idx:end_idx]

            word_text = self.cached_ids_to_text(tuple(tokens_slice))

            # Ignore empty text
            if not word_text:
                continue

            # Append the post word punctuation to the previous word
            if word_text in POST_WORD_PUNCTUATION and len(decoded_words) > 0:
                prev_word = decoded_words[-1]
                prev_word.text += word_text
                continue

            # Refine timestamps
            word_start_tms, word_end_tms = self.refine_text_segment_timestamp(
                current_tokens=tokens_slice,
                current_timesteps=time_slice,
                next_segment_start_timestep=timesteps[end_idx] if end_idx < n_tokens else None,
                need_merge_with_next_segment=(
                    self.start_of_word_cache[tokens[end_idx]] if end_idx < n_tokens else None
                ),
                prev_segment_end=prev_word_end,
            )
            prev_word_end = word_end_tms

            # Aggregate confidence
            word_conf = self.confidence_aggregator(conf_slice)

            # Convert token timestamps to seconds
            start_sec = round(word_start_tms * self.token_duration_in_secs, ROUND_PRECISION)
            end_sec = round(word_end_tms * self.token_duration_in_secs, ROUND_PRECISION)

            decoded_words.append(Word(text=word_text, start=start_sec, end=end_sec, conf=word_conf))

        return decoded_words, need_merge

    def refine_text_segment_timestamp(
        self,
        current_tokens: list[int],
        current_timesteps: list[float],
        next_segment_start_timestep: float | None = None,
        need_merge_with_next_segment: bool | None = None,
        prev_segment_end: float | None = None,
    ) -> tuple[float, float]:
        """
        Refines the text segment timestamp based on the current tokens, timestamps, and the next segment start timestamp.
        Args:
            current_tokens (list[int]): List of token indices.
            current_timesteps (list[float]): List of token timestamps.
            next_segment_start_timestep (float | None): The start timestamp of the next segment.
            need_merge_with_next_segment (bool | None): True if the current segment should be merged with the next segment.
            prev_segment_end (float | None): The end timestamp of the previous segment.
        Returns:
            tuple(float, float): The refined start and end timestamps.
        """

        start, end = current_timesteps[0], current_timesteps[-1]

        # --- Correct the start timestamp if the first token is underscore or punctuation ---
        first_token = current_tokens[0]
        if self.punct_cache[first_token][1]:
            start = next(
                (tms for tms, token in zip(current_timesteps, current_tokens) if not self.punct_cache[token][1]),
                start,
            )

        # --- Correct the end timestamp if the last token is punctuation ---
        last_token = current_tokens[-1]
        if self.punct_cache[last_token][0]:
            end = next(
                (
                    current_timesteps[i]
                    for i in reversed(range(len(current_tokens)))
                    if not self.punct_cache[current_tokens[i]][0]
                ),
                end,
            )

        # --- If the next segment is close to the end of the current segment, merge timestamps ---
        if next_segment_start_timestep is not None and need_merge_with_next_segment:
            if next_segment_start_timestep - end <= self.word_boundary_tolerance:
                end = next_segment_start_timestep

        # --- Adjust the start and end timestamps based on the previous segment end ---
        delta = 0
        if prev_segment_end is not None:
            if prev_segment_end > start:
                delta = prev_segment_end - start

        start = start + delta
        end = end + delta
        return start, end + (1 if start == end else 0)
