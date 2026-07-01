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

from difflib import SequenceMatcher
from typing import List, Optional, Tuple, Union

import torch

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.timestamp_utils import get_segment_offsets, get_words_offsets
from nemo.utils import logging


class VocabularyAdapter:
    """
    Adapter that exposes tokenizer-like ids_to_tokens / ids_to_text / text_to_ids
    for models that use a plain vocabulary (list of strings) instead of a tokenizer.
    Used by character-based models (e.g. hybrid RNNT-CTC) in chunking utilities.
    """

    def __init__(self, vocabulary: List[str]):
        if not vocabulary:
            raise ValueError("vocabulary must be a non-empty list of strings")
        self.vocabulary = list(vocabulary)
        self._vocab_set = set(self.vocabulary)

    def ids_to_tokens(self, ids, lang_id=None):
        """Convert token id(s) to token string(s). ids can be int or list of int."""
        if isinstance(ids, (list, tuple)):
            id_list = ids
        else:
            id_list = [ids]
        out = []
        for i in id_list:
            if 0 <= i < len(self.vocabulary):
                out.append(self.vocabulary[i])
            # skip blank or out-of-range (e.g. blank_id == len(vocabulary))
        return out

    def ids_to_text(self, ids, lang_id=None):
        """Convert list of token ids to a single string (character-level join)."""
        tokens = self.ids_to_tokens(ids, lang_id=lang_id)
        return "".join(tokens)

    def text_to_ids(self, text, lang_id=None):
        """Convert text to list of token ids (character-level for plain vocabulary)."""
        if not text:
            return []
        return [self.vocabulary.index(c) if c in self._vocab_set else 0 for c in text]


def word_similarity(word1: str, word2: str) -> float:
    """
    Calculate similarity ratio between two words using SequenceMatcher.
    Returns a value between 0.0 (completely different) and 1.0 (identical).
    """
    if not word1 or not word2:
        return 0.0
    return SequenceMatcher(None, word1.lower(), word2.lower()).ratio()


def find_optimal_chunk_size(
    total_len: int,
    min_sec: int = 30,
    max_sec: int = 40,
    sample_rate: int = 16000,
    overlap_sec: float = 1.0,
) -> int:
    """
    Determine the chunk size (in samples) that minimizes padding for the final chunk.
    """
    if total_len < max_sec * sample_rate:
        return total_len

    best_chunk_size = min_sec * sample_rate
    best_last_chunk_len = 0
    overlap_size = int(overlap_sec * sample_rate)

    for sec in range(min_sec, max_sec + 1):
        candidate = sec * sample_rate
        step_size = candidate - overlap_size

        if step_size <= 0 or candidate > total_len:
            continue

        n_chunks = (total_len + step_size - 1) // step_size
        last_chunk_len = total_len - step_size * (n_chunks - 1)

        if last_chunk_len > best_last_chunk_len:
            best_last_chunk_len = last_chunk_len
            best_chunk_size = candidate

    return best_chunk_size


def chunk_waveform(
    waveform: torch.Tensor,
    chunk_range: Optional[List],
    overlap_sec: float = 1.0,
    sample_rate: int = 16000,
) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    """
    Split a single waveform into overlapping chunks and record each chunk length and start offset.

    Returns:
        chunks: List of chunk tensors (padded to uniform size).
        chunk_lens: True (unpadded) length of each chunk in samples.
        chunk_starts: Start offset of each chunk in the original waveform, in samples.
    """
    total_len = waveform.shape[0]
    if chunk_range is None:
        chunk_size = find_optimal_chunk_size(
            total_len=total_len,
            sample_rate=sample_rate,
            overlap_sec=overlap_sec,
        )
    else:
        if not isinstance(chunk_range, List) or len(chunk_range) != 2:
            raise ValueError("Chunk size should be list with the minimum and maximum length of the chunk.")
        chunk_size = find_optimal_chunk_size(
            total_len=total_len,
            min_sec=chunk_range[0],
            max_sec=chunk_range[1],
            sample_rate=sample_rate,
            overlap_sec=overlap_sec,
        )

    if chunk_size >= total_len:
        return [waveform], [total_len], [0]

    overlap_size = int(overlap_sec * sample_rate)
    step_size = chunk_size - overlap_size

    if step_size <= 0:
        raise ValueError("chunk_size must be greater than the overlap size.")

    chunks: List[torch.Tensor] = []
    chunk_lens: List[int] = []
    chunk_starts: List[int] = []
    start = 0

    while start + overlap_size < total_len:
        end = min(start + chunk_size, total_len)
        chunk = waveform[start:end]
        length = chunk.shape[0]

        if length < chunk_size:
            pad = torch.zeros(chunk_size - length, dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat([chunk, pad], dim=0)

        chunks.append(chunk)
        chunk_lens.append(length)
        chunk_starts.append(start)
        start += step_size

    return chunks, chunk_lens, chunk_starts


def _get_token_ids(hypothesis, return_hypotheses, timestamps, lang_id, tokenizer):
    """
    Extract token IDs from a hypothesis for merging.

    Prefer token_sequence (final token IDs after CTC collapse) when present;
    otherwise use y_sequence  (in RNNT and AED models)

    Args:
        hypothesis: A Hypothesis object.
        return_hypotheses: Whether return_hypotheses mode is active.
        timestamps: Whether timestamps are enabled.
        lang_id: Optional language id for multilingual tokenizers.
        tokenizer: Tokenizer for text_to_ids fallback.

    Returns:
        List[int]: Token IDs suitable for merging.
    """
    if timestamps and lang_id is None:
        # Timestamps path extracts token IDs from char timestamps; handled by caller.
        return None
    if lang_id is not None:
        return tokenizer.text_to_ids(hypothesis.text, lang_id=lang_id)
    if hasattr(hypothesis, 'token_sequence') and hypothesis.token_sequence is not None:
        seq = hypothesis.token_sequence
        return seq.tolist() if isinstance(seq, torch.Tensor) else list(seq)
    return hypothesis.y_sequence.tolist()


def merge_chunked_hypotheses(
    hypotheses,
    encoded_len,
    timestamps,
    tokenizer=None,
    subsampling_factor=None,
    window_stride=None,
    timestamps_type=None,
    lang_id=None,
    vocabulary=None,
    return_hypotheses=False,
):
    """
    Merges hypotheses from parallel chunks into a single hypothesis with proper text,
    token sequences, and timestamps.

    Args:
        hypotheses: List of Hypothesis objects from each chunk
        encoded_len: Tensor of encoded lengths for each chunk to use for finding offsets
        timestamps: Timestamps generation is enabled
        tokenizer: Optional tokenizer for id/text conversion (e.g. BPE models).
        subsampling_factor: The encoder's subsampling factor
        window_stride: The preprocessor's window stride
        vocabulary: Optional list of token strings (e.g. character-based models without tokenizer).
            Used when tokenizer is None to build an internal adapter.
        timestamps_type: Types of timestamps to include.
        lang_id: Optional language id for multilingual tokenizers.
        return_hypotheses: Whether return_hypotheses mode is active (y_sequence may contain logits).

    Returns:
        Hypothesis: A single merged hypothesis with combined text, tokens, and timestamps
    """
    if tokenizer is None and vocabulary is not None:
        tokenizer = VocabularyAdapter(vocabulary)
    if tokenizer is None and vocabulary is None:
        raise ValueError("Either tokenizer or vocabulary must be provided.")
    # we take the overlap to be 1 second, and count number of tokens in it
    delay = int(1 / (subsampling_factor / 100))
    if timestamps and lang_id is None:
        # Function to normalize tokens from TDT/RNNT
        def ensure_char_token(entry):
            char_value = entry.get('char', '')
            if isinstance(char_value, List):
                char_value = char_value[0] if char_value else ''
                entry['char'] = char_value

            token_id = entry.get('token_id')
            if isinstance(token_id, (list, tuple)):
                entry['token_id'] = token_id[0]

            if 'token' not in entry or entry['token'] is None:
                token = tokenizer.ids_to_tokens(token_id)
                entry['token'] = token[1] if len(token) > 1 else token[0]
            return entry

        if hypotheses[0].timestamp['char']:
            merged_tokens = []
            for char in hypotheses[0].timestamp['char']:
                char = ensure_char_token(char)
                merged_tokens.append(char['token_id'])
        else:
            if hypotheses[0].text != '':
                logging.warning("Cannot provide reliable timestamps for the current audio file.")
            merged_tokens = _get_token_ids(hypotheses[0], return_hypotheses, timestamps, lang_id, tokenizer) or []
    else:
        merged_tokens = _get_token_ids(hypotheses[0], return_hypotheses, timestamps, lang_id, tokenizer) or []

    # avoid circular import
    from nemo.collections.asr.parts.utils.streaming_utils import lcs_alignment_merge_buffer

    for i in range(1, len(hypotheses)):
        if timestamps and lang_id is None:
            if hypotheses[i].timestamp['char']:
                data = []
                for char in hypotheses[i].timestamp['char']:
                    char = ensure_char_token(char)
                    data.append(char['token_id'])
            else:
                if hypotheses[0].text != '':
                    logging.warning("Cannot provide reliable timestamps for the current audio file.")
                data = _get_token_ids(hypotheses[i], return_hypotheses, timestamps, lang_id, tokenizer) or []
        else:
            data = _get_token_ids(hypotheses[i], return_hypotheses, timestamps, lang_id, tokenizer) or []
        merged_tokens = lcs_alignment_merge_buffer(
            buffer=merged_tokens,
            data=data[: int(delay * 0.6)],  # only approximately 60% of the frames have corresponding tokens
            delay=delay,
            max_steps_per_timestep=2,
            min_lcs_length=1,
            parallel_chunking=True,
        )
        merged_tokens += data[int(delay * 0.6) :]

    # Convert merged tokens to text
    # Use ids_to_text which handles token ID offsets internally and works with AggregateTokenizer
    # CTC models include blank token id (e.g. vocab_size) in y_sequence; filter it out so tokenizer decode does not fail
    vocab_size = None
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    elif hasattr(tokenizer, 'vocabulary'):
        vocab_size = len(tokenizer.vocabulary)
    if vocab_size is not None:
        merged_tokens = [int(t) for t in merged_tokens if isinstance(t, (int, float)) and 0 <= int(t) < vocab_size]
    final_text = tokenizer.ids_to_text(merged_tokens)
    merged_hypotheses = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([]),
        timestamp=([] if not timestamps else {'word': [], 'segment': []}),
    )

    # When return_hypotheses is True, y_sequence contains logits (2D: [T, V]).
    if return_hypotheses and hasattr(hypotheses[0], 'token_sequence') and hypotheses[0].token_sequence is not None:
        merged_hypotheses.y_sequence = torch.cat([hyp.y_sequence for hyp in hypotheses], dim=0)
        merged_hypotheses.token_sequence = torch.tensor(merged_tokens)
    else:
        merged_hypotheses.y_sequence = torch.tensor(merged_tokens)

    merged_alignments = join_alignments(hypotheses)
    if merged_alignments is not None:
        merged_hypotheses.alignments = merged_alignments

    merged_hypotheses = join_confidence_values(merged_hypotheses, hypotheses)
    merged_hypotheses.text = final_text
    # Set score to minimum of all chunk scores, length to sum of all chunk lengths
    merged_hypotheses.score = min(h.score for h in hypotheses)
    merged_hypotheses.length = sum(h.length if isinstance(h.length, int) else h.length.item() for h in hypotheses)
    # Merge timestamps and add word and segment level timestamps
    chunk_offsets = [0] + [
        (x * subsampling_factor - 100) if i >= 1 else (x * subsampling_factor)
        for i, x in enumerate(encoded_len.tolist(), start=1)
    ]
    if timestamps and lang_id is None:

        merged_hypotheses = join_char_timestamp_add_word_and_segment_level_timestamps(
            merged_hypotheses,
            hypotheses,
            chunk_offsets,
            subsampling_factor,
            window_stride,
            tokenizer,
            merged_tokens,
            timestamps_type,
        )
    elif timestamps:
        # lang_id is provided - timestamps are word-level, not char-level
        merged_hypotheses = join_word_level_timestamps_add_segment_level_timestamps(
            merged_hypotheses,
            hypotheses,
            chunk_offsets,
            subsampling_factor,
            window_stride,
            final_text,
            timestamps_type,
        )
    return merged_hypotheses


def update_timestamps(hypotheses, tokenizer=None, timestamps_type=None, lang_id=None, vocabulary=None):
    """
    Generate word and segment timestamps from character timestamps.

    Args:
        hypotheses: Hypothesis to update with timestamps
        tokenizer: Optional tokenizer for id/text conversion.
        timestamps_type: Types of timestamps to include.
        lang_id: Optional language id for multilingual tokenizers.
        vocabulary: Optional list of token strings when tokenizer is None (e.g. character-based models).

    Returns:
        Hypothesis: Updated merged_hypotheses with word and segment timestamps
    """
    if tokenizer is None and vocabulary is not None:
        tokenizer = VocabularyAdapter(vocabulary)
    if tokenizer is None:
        raise ValueError("Either tokenizer or vocabulary must be provided.")
    # Create encoded_char_offsets for word/segment generation
    char_timestamps = hypotheses.timestamp['char']
    encoded_char_offsets = []
    for char_offset in char_timestamps:
        enc_char_offset = char_offset.copy()
        if lang_id:
            enc_char_offset['char'] = tokenizer.ids_to_tokens(
                (
                    [enc_char_offset['token_id']]
                    if isinstance(enc_char_offset['token_id'], int)
                    else enc_char_offset['token_id']
                ),
                lang_id=lang_id,
            )
        else:
            enc_char_offset['char'] = tokenizer.ids_to_tokens(
                [enc_char_offset['token_id']]
                if isinstance(enc_char_offset['token_id'], int)
                else enc_char_offset['token_id']
            )

        char_offset.pop('token_id', None)
        char_offset.pop('token', None)
        encoded_char_offsets.append(enc_char_offset)

    # Generate word-level timestamps from combined char timestamps
    # Wrap tokens_to_text to include lang_id if specified
    if lang_id:
        decode_fn = lambda tokens: tokenizer.ids_to_text(tokens, lang_id=lang_id)
    else:
        decode_fn = tokenizer.ids_to_text

    word_offsets = get_words_offsets(
        char_offsets=char_timestamps,
        decode_tokens_to_str=decode_fn,
        encoded_char_offsets=encoded_char_offsets,
        supported_punctuation={',', '.', '!', '?'},
    )
    # Generate segment-level timestamps from word timestamps
    segment_offsets = get_segment_offsets(word_offsets=word_offsets, segment_delimiter_tokens={'.', '!', '?', "..."})
    # Update the merged hypothesis with word and segment timestamps
    if timestamps_type is not None:
        if 'word' in timestamps_type or 'all' in timestamps_type:
            hypotheses.timestamp['word'] = word_offsets
        else:
            hypotheses.timestamp.pop('word', None)
        if 'segment' in timestamps_type or 'all' in timestamps_type:
            hypotheses.timestamp['segment'] = segment_offsets
        else:
            hypotheses.timestamp.pop('segment', None)
        if 'char' in timestamps_type or 'all' in timestamps_type:
            hypotheses.timestamp['char'] = char_timestamps
        else:
            hypotheses.timestamp.pop('char', None)
    else:
        hypotheses.timestamp['word'] = word_offsets
        hypotheses.timestamp['segment'] = segment_offsets
        hypotheses.timestamp.pop('char', None)
    return hypotheses


def join_alignments(
    hypotheses: List[Hypothesis],
) -> Optional[Union[torch.Tensor, List]]:
    """
    Concatenate alignments from multiple chunk hypotheses into a single sequence.

    Supports both CTC alignments (1D: list of ints or tensor) and RNNT alignments
    (2D: list of lists, one inner list per time step). If any hypothesis has no
    alignments, returns None and the caller should leave merged alignments unset.

    Args:
        hypotheses: List of Hypothesis objects, each possibly having an alignments attribute.

    Returns:
        Concatenated alignments (tensor or list), or None if any hypothesis has no alignments.
    """
    if not hypotheses:
        return None
    if not all(getattr(h, 'alignments', None) is not None for h in hypotheses):
        return None

    alignments_list = [h.alignments for h in hypotheses]

    # CTC: alignments are a 1D tensor
    if isinstance(alignments_list[0], torch.Tensor):
        return torch.cat(alignments_list, dim=0)

    # RNNT: alignments are list of lists (one list per time step T)
    first_nonempty = next((a for a in alignments_list if len(a) > 0), None)
    if first_nonempty is not None and isinstance(first_nonempty[0], (list, tuple)):
        result = []
        for a in alignments_list:
            result.extend(a)
        return result

    # CTC: alignments are a flat list of ints
    result = []
    for a in alignments_list:
        result.extend(a.tolist() if isinstance(a, torch.Tensor) else a)
    return result


def join_confidence_values(merged_hypothesis, hypotheses):
    """
    Concatenate confidence values from multiple hypotheses into a single sequence.

    Args:
        merged_hypothesis: Target hypothesis to update with concatenated confidence
        hypotheses: List of hypotheses containing confidence values

    Returns:
        Hypothesis: Updated merged_hypothesis with concatenated confidence values
    """
    # Merge frame_confidence
    frame_confidences = [h.frame_confidence for h in hypotheses if h.frame_confidence is not None]
    if frame_confidences:
        if isinstance(frame_confidences[0], torch.Tensor):
            merged_hypothesis.frame_confidence = torch.cat(frame_confidences)
        elif isinstance(frame_confidences[0], list):
            merged_hypothesis.frame_confidence = [c for conf_list in frame_confidences for c in conf_list]

    # Merge token_confidence
    token_confidences = [h.token_confidence for h in hypotheses if h.token_confidence is not None]
    if token_confidences:
        if isinstance(token_confidences[0], torch.Tensor):
            merged_hypothesis.token_confidence = torch.cat(token_confidences)
        elif isinstance(token_confidences[0], list):
            merged_hypothesis.token_confidence = [c for conf_list in token_confidences for c in conf_list]

    # Merge word_confidence
    word_confidences = [h.word_confidence for h in hypotheses if h.word_confidence is not None]
    if word_confidences:
        if isinstance(word_confidences[0], torch.Tensor):
            merged_hypothesis.word_confidence = torch.cat(word_confidences)
        elif isinstance(word_confidences[0], list):
            merged_hypothesis.word_confidence = [c for conf_list in word_confidences for c in conf_list]

    return merged_hypothesis


def join_char_timestamp_add_word_and_segment_level_timestamps(
    merged_hypotheses,
    hypotheses,
    chunk_offsets,
    subsampling_factor,
    window_stride,
    tokenizer,
    merged_tokens=None,
    timestamps_type=None,
):
    """
    Combine character-level timestamps from chunks and generate word/segment timestamps.

    Args:
        merged_hypotheses: Target hypothesis to update with timestamps
        hypotheses: List of hypotheses from different chunks
        chunk_offsets: Frame offsets for each chunk
        subsampling_factor: Subsampling factor of the encoder
        window_stride: Time stride per frame in seconds
        decoding: Decoding that is used for decoding tokens into text in `get_words_offsets`
        merged_tokens: Optional token sequence for filtering (default: None)

    Returns:
        Hypothesis: Updated merged_hypotheses with word and segment timestamps
    """

    # First, combine char-level timestamps from all chunks
    char_timestamps = join_char_level_timestamps(
        hypotheses, chunk_offsets, subsampling_factor, window_stride, merged_tokens
    )
    merged_hypotheses.timestamp['char'] = char_timestamps

    return update_timestamps(merged_hypotheses, tokenizer, timestamps_type)


def join_word_level_timestamps_add_segment_level_timestamps(
    merged_hypotheses,
    hypotheses,
    chunk_offsets,
    subsampling_factor,
    window_stride,
    merged_text,
    timestamps_type=None,
    similarity_threshold=0.7,
):
    """
    Merge word-level timestamps from chunked hypotheses when lang_id is provided.

    When lang_id is given, timestamps are word-level (not char-level), so we need
    a different approach to merge them. This function:
    1. Gets the expected words from the merged token text as ground truth
    2. Adjusts word timestamps with chunk offsets
    3. Handles overlapping words at chunk boundaries using similarity matching against expected words
    4. Removes duplicate words that appear in the overlap region
    5. Concatenates words that were split across chunks (verified against expected words)

    Args:
        merged_hypotheses: Target hypothesis to update with merged timestamps
        hypotheses: List of hypotheses from different chunks
        chunk_offsets: Frame offsets for each chunk
        subsampling_factor: Subsampling factor of the encoder
        window_stride: Time stride per frame in seconds
        merged_text: Merged text string whose whitespace-split words serve as ground truth
        timestamps_type: Types of timestamps to include ('word', 'segment', 'all')
        similarity_threshold: Threshold for word similarity matching (0.0-1.0)

    Returns:
        Hypothesis: Updated merged_hypotheses with word and segment timestamps
    """
    subsamp = subsampling_factor
    stride = window_stride

    expected_words = merged_text.split()

    # Collect all word timestamps from all chunks with offset adjustments
    all_word_timestamps = []
    cumulative_time_offset = 0.0
    cumulative_frame_offset = 0

    delay = int(1 / (subsampling_factor / 100))  # overlap in frames
    overlap_time = delay * stride * subsamp  # overlap time in seconds

    for chunk_idx, hyp in enumerate(hypotheses):
        chunk_words = hyp.timestamp.get('word', [])
        if not chunk_words:
            continue

        # Calculate offset for this chunk
        if chunk_idx > 0:
            chunk_frame_offset = chunk_offsets[chunk_idx] // subsamp
            cumulative_frame_offset += chunk_frame_offset
            cumulative_time_offset = cumulative_frame_offset * stride * subsamp

        # Adjust timestamps for this chunk
        adjusted_words = []
        for word_info in chunk_words:
            adjusted_word = word_info.copy()

            # Adjust frame offsets
            if 'start_offset' in adjusted_word and adjusted_word['start_offset'] != -1:
                adjusted_word['start_offset'] += cumulative_frame_offset
            if 'end_offset' in adjusted_word and adjusted_word['end_offset'] != -1:
                adjusted_word['end_offset'] += cumulative_frame_offset

            # Adjust time values
            if 'start' in adjusted_word and adjusted_word['start'] != -1:
                adjusted_word['start'] += cumulative_time_offset
            if 'end' in adjusted_word and adjusted_word['end'] != -1:
                adjusted_word['end'] += cumulative_time_offset

            # Track which chunk this word came from
            adjusted_word['_chunk_idx'] = chunk_idx
            adjusted_words.append(adjusted_word)

        all_word_timestamps.append(
            {
                'chunk_idx': chunk_idx,
                'words': adjusted_words,
                'overlap_start_time': cumulative_time_offset if chunk_idx > 0 else None,
            }
        )

    # Merge words handling overlaps, using expected_words as ground truth
    merged_words = _merge_word_timestamps_with_overlap(
        all_word_timestamps,
        expected_words,
        overlap_time,
        similarity_threshold,
    )

    # Clean up internal tracking fields
    for word in merged_words:
        word.pop('_chunk_idx', None)
        word.pop('_overlap_start', None)

    # Generate segment timestamps from word timestamps
    segment_offsets = get_segment_offsets(
        word_offsets=merged_words,
        segment_delimiter_tokens={'.', '!', '?', "..."},
    )

    # Update the merged hypothesis with timestamps
    if timestamps_type is not None:
        if 'word' in timestamps_type or 'all' in timestamps_type:
            merged_hypotheses.timestamp['word'] = merged_words
        else:
            merged_hypotheses.timestamp.pop('word', None)
        if 'segment' in timestamps_type or 'all' in timestamps_type:
            merged_hypotheses.timestamp['segment'] = segment_offsets
        else:
            merged_hypotheses.timestamp.pop('segment', None)
    else:
        merged_hypotheses.timestamp['word'] = merged_words
        merged_hypotheses.timestamp['segment'] = segment_offsets

    return merged_hypotheses


def _merge_word_timestamps_with_overlap(
    all_word_timestamps: List[dict],
    expected_words: List[str],
    overlap_time: float,
    similarity_threshold: float = 0.7,
) -> List[dict]:
    """
    Merge word timestamps from multiple chunks by matching against expected words from merged text.

    This function uses the expected words (from the merged token text) as ground truth to:
    1. Determine which chunk words to keep vs. remove (duplicates from overlap)
    2. Determine when to concatenate words (by checking if concatenation matches expected word)
    3. Ensure the final word list matches the expected merged text

    Args:
        all_word_timestamps: List of dicts with 'chunk_idx', 'words', 'overlap_start_time'
        expected_words: List of expected words from the merged token text (ground truth)
        overlap_time: Duration of overlap between chunks in seconds
        similarity_threshold: Threshold for considering words as matches

    Returns:
        List of merged word timestamp dictionaries
    """
    if not all_word_timestamps:
        return []

    if not expected_words:
        # No expected words - just concatenate all chunk words
        all_words = []
        for chunk_data in all_word_timestamps:
            all_words.extend(chunk_data['words'])
        return all_words

    # Flatten all chunk words into a single list with chunk info preserved
    all_chunk_words = []
    for chunk_data in all_word_timestamps:
        chunk_idx = chunk_data['chunk_idx']
        overlap_start = chunk_data.get('overlap_start_time')
        for word in chunk_data['words']:
            word_copy = word.copy()
            word_copy['_chunk_idx'] = chunk_idx
            word_copy['_overlap_start'] = overlap_start
            all_chunk_words.append(word_copy)

    # Match expected words against chunk words
    merged_words = []
    chunk_word_idx = 0

    for _, expected_word in enumerate(expected_words):
        best_match = None
        best_match_idx = -1
        best_similarity = 0.0
        concat_match = None
        concat_end_idx = -1

        # Only search for matches if we still have chunk words
        if chunk_word_idx < len(all_chunk_words):
            # Search for the best match within a window
            search_window = min(10, len(all_chunk_words) - chunk_word_idx)

            for i in range(chunk_word_idx, chunk_word_idx + search_window):
                if i >= len(all_chunk_words):
                    break

                candidate = all_chunk_words[i]
                candidate_text = candidate.get('word', '')

                # Check direct match
                similarity = word_similarity(candidate_text, expected_word)
                if similarity > best_similarity and similarity >= similarity_threshold:
                    best_similarity = similarity
                    best_match = candidate
                    best_match_idx = i

                # Check if concatenating with next word(s) matches expected word
                # This handles split words like "exclu" + "des" -> "excludes"
                if i + 1 < len(all_chunk_words):
                    next_candidate = all_chunk_words[i + 1]
                    combined_text = candidate_text + next_candidate.get('word', '')
                    concat_similarity = word_similarity(combined_text, expected_word)

                    if concat_similarity > best_similarity and concat_similarity >= similarity_threshold:
                        # Concatenation is a better match
                        best_similarity = concat_similarity
                        concat_match = (candidate, next_candidate)
                        concat_end_idx = i + 1
                        best_match = None  # Use concat instead

        # Apply the best match found
        if concat_match is not None:
            # Concatenate the words - use expected_word text to guarantee exact match
            word1, word2 = concat_match
            merged_word = word1.copy()
            merged_word['word'] = expected_word  # Use expected word, not concatenated chunk words
            merged_word['end'] = word2.get('end', word1.get('end', 0))
            merged_word['end_offset'] = word2.get('end_offset', word1.get('end_offset', -1))

            # Fix timing if needed
            if merged_words:
                last_end = merged_words[-1].get('end', 0)
                if merged_word.get('start', 0) < last_end:
                    merged_word['start'] = last_end
                    merged_word['start_offset'] = merged_words[-1].get('end_offset', -1)

            merged_words.append(merged_word)
            chunk_word_idx = concat_end_idx + 1

        elif best_match is not None:
            merged_word = best_match.copy()
            # Use expected word text to guarantee exact match with merged text
            merged_word['word'] = expected_word

            # Fix timing if needed
            if merged_words:
                last_end = merged_words[-1].get('end', 0)
                if merged_word.get('start', 0) < last_end:
                    merged_word['start'] = last_end
                    merged_word['start_offset'] = merged_words[-1].get('end_offset', -1)

            merged_words.append(merged_word)
            chunk_word_idx = best_match_idx + 1

            # Skip duplicate words from overlap (same word appearing in next chunk)
            while chunk_word_idx < len(all_chunk_words):
                next_word = all_chunk_words[chunk_word_idx]
                next_text = next_word.get('word', '')

                # Check if this is a duplicate of what we just added
                dup_similarity = word_similarity(next_text, expected_word)
                if dup_similarity >= similarity_threshold:
                    # Check time proximity to confirm it's in overlap region
                    time_diff = abs(next_word.get('start', 0) - merged_word.get('end', 0))
                    if time_diff <= overlap_time:
                        # This is a duplicate from overlap - skip it
                        chunk_word_idx += 1
                        continue
                break
        else:
            # No match found - add expected word with zero-duration timestamp
            # Use the previous word's end time as both start and end
            if merged_words:
                last_word = merged_words[-1]
                merged_word = {
                    'word': expected_word,
                    'start': last_word.get('end', 0),
                    'end': last_word.get('end', 0),
                    'start_offset': last_word.get('end_offset', -1),
                    'end_offset': last_word.get('end_offset', -1),
                }
            else:
                # First word with no match - use zeros
                merged_word = {
                    'word': expected_word,
                    'start': 0,
                    'end': 0,
                    'start_offset': 0,
                    'end_offset': 0,
                }

            merged_words.append(merged_word)

    # Final timing fix pass to ensure no overlaps
    merged_words = _fix_word_timing(merged_words)

    # At this point, merged_words has exactly len(expected_words) items
    # Each word text matches the corresponding expected word

    return merged_words


def _fix_word_timing(words: List[dict]) -> List[dict]:
    """
    Fix timing issues in merged words to ensure non-overlapping timestamps.

    This function only adjusts timing - it does NOT remove any words,
    preserving the exact correspondence with expected words from merged text.

    Args:
        words: List of word timestamp dictionaries

    Returns:
        List of word timestamps with fixed timing
    """
    if not words:
        return words

    fixed = []

    for word in words:
        word_copy = word.copy()

        if fixed:
            last_word = fixed[-1]
            last_end = last_word.get('end', 0)

            # Fix timing overlap - ensure current word starts after last word ends
            if word_copy.get('start', 0) < last_end:
                word_copy['start'] = last_end
                if 'start_offset' in word_copy and 'end_offset' in last_word:
                    word_copy['start_offset'] = last_word['end_offset']

        fixed.append(word_copy)

    return fixed


def join_char_level_timestamps(
    hypotheses,
    chunk_offsets,
    subsampling_factor,
    window_stride,
    merged_tokens=None,
):
    """
    Merge per-chunk character-level timestamps into a single global timeline.

    This function stitches together character timestamp dictionaries coming from
    consecutive chunks of the same audio. It shifts each chunk's offsets into a
    global frame-of-reference and converts subsampled frame offsets to seconds.

    Args:
        hypotheses: List of hypotheses.
        chunk_offsets: List of raw-frame offsets (one per chunk) used for shifting.
        subsampling_factor: Encoder subsampling factor (int). Number of raw
            frames per one subsampled step.
        window_stride: Time (in seconds) per raw input frame (float).
        merged_tokens: Optional list of global token ids. If provided, only
            characters whose `token_id` matches the next id in this list are
            retained; leading overlapped characters within a chunk are trimmed.

    Returns:
        List[dict]: Character timestamp dicts placed on a global timeline
    """
    char_timestamps = []
    cumulative_offset = 0  # raw (pre-subsampling) frames already emitted
    j_token = 0  # cursor in merged_tokens

    subsamp = subsampling_factor
    stride = window_stride  # sec per raw frame
    for i, h in enumerate(hypotheses):
        chunk_frame_offset = chunk_offsets[i] // subsamp
        cumulative_offset += chunk_frame_offset

        # 1) figure out how much of the *front* of this chunk we will drop
        for char in h.timestamp['char']:
            if not char:
                continue
            current_token_id = char['token_id']
            keep = merged_tokens is None or (
                j_token < len(merged_tokens) and current_token_id == merged_tokens[j_token]
            )
            if not keep:
                continue
            # adjust offsets: chunk start + global chunk shift − total removed
            upd = dict(char)
            if char['start_offset'] != -1:
                upd['start_offset'] = char['start_offset'] + cumulative_offset  # place chunk globally
            if char['end_offset'] != -1:
                upd['end_offset'] = char['end_offset'] + cumulative_offset

            if char_timestamps:
                if upd['start_offset'] != -1 and upd['start_offset'] < char_timestamps[-1]['end_offset']:
                    upd['start_offset'] = char_timestamps[-1]['end_offset']
                    upd['end_offset'] = char_timestamps[-1]['end_offset']
            # convert to seconds
            upd['start'] = -1 if upd['start_offset'] == -1 else upd['start_offset'] * stride * subsamp
            upd['end'] = -1 if upd['end_offset'] == -1 else upd['end_offset'] * stride * subsamp

            char_timestamps.append(upd)
            j_token += 1

    return char_timestamps


def merge_flat_chunk_hypotheses(
    hypotheses_list,
    timestamps,
    tokenizer=None,
    subsampling_factor=None,
    window_stride=None,
    vocabulary=None,
    return_hypotheses=False,
):
    """
    Merge a flat list of individual per-chunk hypotheses into one hypothesis per source utterance.

    Each hypothesis must carry attributes stamped by ``_transcribe_output_processing``:

    * ``id``            – source cut ID (groups chunks of the same utterance)
    * ``chunk_start``   – start offset of the chunk within its parent (used for ordering)
    * ``_encoded_len``  – encoder output length for this chunk (scalar int/float)
    * ``_timestamps_type`` – the original timestamp type before it was overridden to 'char'

    Hypotheses without an ``id`` attribute are returned as-is.

    Args:
        hypotheses_list: Flat list of Hypothesis objects from all batches.
        timestamps: True if timestamps are enabled.
        tokenizer: Optional BPE tokenizer.
        subsampling_factor: Encoder subsampling factor.
        window_stride: Preprocessor window stride.
        vocabulary: Optional character vocabulary (for char-based models without tokenizer).
        return_hypotheses: Whether return_hypotheses mode is active.

    Returns:
        List[Hypothesis]: One merged hypothesis per unique ``id``, preserving original ordering
        of first-seen IDs.
    """
    # Group in order of first appearance, preserving the manifest order that the
    # sampler produces.  Do NOT sort globally by str(id): transcribe_speech presorts
    # the manifest by duration and restore_transcription_order expects results back
    # in that same presorted order.  A global alphabetical sort on string IDs would
    # scramble that ordering.
    # Within each group we sort by chunk_start so that LCS merge sees chunks in
    # temporal order.
    groups: dict = {}
    for h in hypotheses_list:
        gid = getattr(h, 'id', None)
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(h)

    merged_list = []
    for gid, group in groups.items():
        if gid is None:
            merged_list.extend(group)
            continue

        # Sort within the group by chunk_start for correct temporal ordering.
        group = sorted(group, key=lambda h: getattr(h, 'chunk_start', 0))

        if len(group) == 1:
            h = group[0]
            if timestamps:
                timestamps_type = getattr(h, '_timestamps_type', None)
                h = update_timestamps(
                    h,
                    tokenizer=tokenizer,
                    timestamps_type=timestamps_type,
                    vocabulary=vocabulary,
                    lang_id=getattr(h, '_lang_id', None),
                )
            merged_list.append(h)
        else:
            timestamps_type = getattr(group[0], '_timestamps_type', None)
            encoded_len = torch.tensor([int(getattr(h, '_encoded_len', 0)) for h in group])
            merged = merge_chunked_hypotheses(
                hypotheses=group,
                encoded_len=encoded_len,
                timestamps=timestamps,
                tokenizer=tokenizer,
                subsampling_factor=subsampling_factor,
                window_stride=window_stride,
                timestamps_type=timestamps_type,
                lang_id=getattr(group[0], '_lang_id', None),
                vocabulary=vocabulary,
                return_hypotheses=return_hypotheses,
            )
            setattr(merged, 'id', gid)
            merged_list.append(merged)

    return merged_list
