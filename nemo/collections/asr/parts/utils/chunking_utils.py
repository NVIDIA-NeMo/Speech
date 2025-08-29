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

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.timestamp_utils import get_segment_offsets, get_words_offsets


def merge_parallel_chunks(hypotheses, encoded_len, model, timestamps, subsampling_factor, window_stride, decoding):
    """
    Merges hypotheses from parallel chunks into a single hypothesis with proper text,
    token sequences, and timestamps.

    Args:
        hypotheses: List of Hypothesis objects from each chunk
        encoded_len: Tensor of encoded lengths for each chunk to use for finding offsets
        model: The ASR model instance (needed for LCS alignment)
        timestamps: Timestamps generation is enabled
        subsampling_factor: The encoder's subsampling factor
        window_stride: The preprocessor's window stride
        decoding: The decoding instance for converting tokens to text

    Returns:
        Hypothesis: A single merged hypothesis with combined text, tokens, and timestamps
    """
    # we take the overlap to be 1 second, and count number of tokens in it
    delay = int(1 / (subsampling_factor / 100))
    # Merge tokens from character level timestamps if timestamps are enabled
    if timestamps:
        merged_tokens = [char['token_id'] for char in hypotheses[0].timestamp['char']]
    else:
        merged_tokens = hypotheses[0].y_sequence.tolist()
    # avoid circular import
    from nemo.collections.asr.parts.utils.streaming_utils import lcs_alignment_merge_buffer

    for i in range(1, len(hypotheses)):
        if timestamps:
            data = [char['token_id'] for char in hypotheses[i].timestamp['char']]
        else:
            data = hypotheses[i].y_sequence.tolist()
        merged_tokens = lcs_alignment_merge_buffer(
            buffer=merged_tokens,
            data=data[: int(delay * 0.6)],  # only approximately 60% of the tokens are non blank
            delay=delay,
            model=model,
            max_steps_per_timestep=2,
            min_lcs_length=1,
            parallel_chunking=True,
        )
        merged_tokens += data[int(delay * 0.6) :]

    # Convert merged tokens to text
    final_text = decoding.decode_tokens_to_str(merged_tokens)

    merged_hypotheses = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([]),
        timestamp=([] if not timestamps else {'word': [], 'segment': []}),
    )
    merged_hypotheses = join_y_sequence(merged_hypotheses, hypotheses)
    merged_hypotheses.text = final_text
    # Merge timestamps and add word and segment level timestamps
    if timestamps:
        chunk_offsets = [0] + [
            (x * subsampling_factor - 100) if i >= 1 else (x * subsampling_factor)
            for i, x in enumerate(encoded_len.tolist(), start=1)
        ]
        merged_hypotheses = join_timestamp_and_add_word_and_segment_level_timestamps(
            merged_hypotheses, hypotheses, chunk_offsets, subsampling_factor, window_stride, decoding, merged_tokens
        )

    return merged_hypotheses


def join_y_sequence(merged_hypothesis, hypotheses):
    """
    Concatenate y_sequence tensors from multiple hypotheses into a single sequence.

    Args:
        merged_hypothesis: Target hypothesis to update with concatenated sequence
        hypotheses: List of hypotheses containing y_sequence tensors

    Returns:
        Hypothesis: Updated merged_hypothesis with concatenated y_sequence
    """
    merged_hypothesis.y_sequence = torch.cat([h.y_sequence for h in hypotheses])
    return merged_hypothesis


def join_timestamp_and_add_word_and_segment_level_timestamps(
    merged_hypotheses, hypotheses, chunk_offsets, subsampling_factor, window_stride, decoding, merged_tokens=None
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
    # Create encoded_char_offsets for word/segment generation
    encoded_char_offsets = []
    for char_offset in char_timestamps:
        enc_char_offset = char_offset.copy()
        enc_char_offset['char'] = enc_char_offset['token']
        encoded_char_offsets.append(enc_char_offset)

    # Generate word-level timestamps from combined char timestamps
    word_offsets = get_words_offsets(
        char_offsets=char_timestamps,
        decode_tokens_to_str=decoding.decode_tokens_to_str,
        encoded_char_offsets=encoded_char_offsets,
        supported_punctuation={',', '.', '!', '?'},
    )

    # Generate segment-level timestamps from word timestamps
    segment_offsets = get_segment_offsets(word_offsets=word_offsets, segment_delimiter_tokens={'.', '!', '?', "..."})
    # Update the merged hypothesis with word and segment timestamps
    merged_hypotheses.timestamp['word'] = word_offsets
    merged_hypotheses.timestamp['segment'] = segment_offsets

    return merged_hypotheses


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
            keep = merged_tokens is None or (
                j_token < len(merged_tokens) and char['token_id'] == merged_tokens[j_token]
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


def merge_all_hypotheses(hypotheses_list, timestamps, subsampling_factor, chunk_duration_seconds=3600):
    """
    Group hypotheses by ID and merge each group into a single hypothesis.

    Args:
        hypotheses_list: List of hypothesis objects with 'id' attributes
        timestamps: True if timestamps generation is enabled
        subsampling_factor: Subsampling factor of the encoder
        chunk_duration_seconds: Duration of each chunk in seconds (default: 3600)

    Returns:
        List[Hypothesis]: List of merged hypotheses, one per unique ID
    """
    same_audio_hypotheses = []
    all_merged_hypotheses = []
    prev_id = None
    for h in hypotheses_list:
        current_id = h.id

        # If this is a new ID (different from previous), process the accumulated hypotheses
        if prev_id is not None and current_id != prev_id:
            if same_audio_hypotheses:  # Only merge if we have hypotheses to merge

                all_merged_hypotheses.append(
                    merge_hypotheses_of_same_audio(
                        same_audio_hypotheses, timestamps, subsampling_factor, chunk_duration_seconds
                    )
                )
            same_audio_hypotheses = []

        # Add current hypothesis to the group
        same_audio_hypotheses.append(h)
        prev_id = current_id

    # Process the final group of hypotheses
    if same_audio_hypotheses:
        all_merged_hypotheses.append(
            merge_hypotheses_of_same_audio(
                same_audio_hypotheses, timestamps, subsampling_factor, chunk_duration_seconds
            )
        )
    return all_merged_hypotheses


def merge_hypotheses_of_same_audio(hypotheses_list, timestamps, subsampling_factor, chunk_duration_seconds=3600):
    """
    Merge hypotheses from the same audio source into a single hypothesis.
    Used for combining results when long audio is split into hour-long segments
    processed as separate batches.

    Args:
        hypotheses_list: List of hypothesis objects from time chunks
        timestamps: True if timestamps generation is enabled
        subsampling_factor: Subsampling factor of the encoder
        chunk_duration_seconds: Duration of each chunk in seconds (default: 3600)

    Returns:
        Hypothesis: Single merged hypothesis
    """

    # Create merged hypothesis with empty initial values
    merged_hypothesis = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([]),
        timestamp=([] if not timestamps else {'word': [], 'segment': []}),
    )

    merged_hypothesis.y_sequence = torch.cat([h.y_sequence for h in hypotheses_list])

    # Create final text by joining text from all hypotheses
    text_parts = []
    for hyp in hypotheses_list:
        if hyp.text:
            text_parts.append(hyp.text.strip())
    merged_hypothesis.text = ' '.join(text_parts)

    # Handle timestamps with proper time offsets (word and segment only)
    if timestamps and len(hypotheses_list) > 0 and getattr(hypotheses_list[0], "timestamp", {}):
        # Calculate time offsets for each chunk (in seconds)
        merged_word_timestamps = []
        merged_segment_timestamps = []

        for chunk_idx, hyp in enumerate(hypotheses_list):
            if not hasattr(hyp, 'timestamp') or not hyp.timestamp:
                continue

            # Time offset for this chunk
            time_offset = chunk_idx * chunk_duration_seconds
            # Frame offset for this chunk (convert time to frames)
            frame_offset = int(time_offset * 1000 / subsampling_factor)

            # Merge word timestamps with offset
            if 'word' in hyp.timestamp and hyp.timestamp['word']:
                for word_info in hyp.timestamp['word']:
                    if isinstance(word_info, dict):
                        adjusted_word = word_info.copy()
                        # Adjust start and end times
                        if (
                            'start' in adjusted_word
                            and adjusted_word['start'] is not None
                            and adjusted_word['start'] != -1
                        ):
                            adjusted_word['start'] += time_offset
                        if 'end' in adjusted_word and adjusted_word['end'] is not None and adjusted_word['end'] != -1:
                            adjusted_word['end'] += time_offset
                        # Adjust start and end offsets (frame counts)
                        if (
                            'start_offset' in adjusted_word
                            and adjusted_word['start_offset'] is not None
                            and adjusted_word['start_offset'] != -1
                        ):
                            adjusted_word['start_offset'] += frame_offset
                        if (
                            'end_offset' in adjusted_word
                            and adjusted_word['end_offset'] is not None
                            and adjusted_word['end_offset'] != -1
                        ):
                            adjusted_word['end_offset'] += frame_offset
                        merged_word_timestamps.append(adjusted_word)
                    else:
                        merged_word_timestamps.append(word_info)

            # Merge segment timestamps with offset
            if 'segment' in hyp.timestamp and hyp.timestamp['segment']:
                for segment_info in hyp.timestamp['segment']:
                    if isinstance(segment_info, dict):
                        adjusted_segment = segment_info.copy()
                        # Adjust start and end times
                        if (
                            'start' in adjusted_segment
                            and adjusted_segment['start'] is not None
                            and adjusted_segment['start'] != -1
                        ):
                            adjusted_segment['start'] += time_offset
                        if (
                            'end' in adjusted_segment
                            and adjusted_segment['end'] is not None
                            and adjusted_segment['end'] != -1
                        ):
                            adjusted_segment['end'] += time_offset
                        # Adjust start and end offsets (frame counts)
                        if (
                            'start_offset' in adjusted_segment
                            and adjusted_segment['start_offset'] is not None
                            and adjusted_segment['start_offset'] != -1
                        ):
                            adjusted_segment['start_offset'] += frame_offset
                        if (
                            'end_offset' in adjusted_segment
                            and adjusted_segment['end_offset'] is not None
                            and adjusted_segment['end_offset'] != -1
                        ):
                            adjusted_segment['end_offset'] += frame_offset
                        merged_segment_timestamps.append(adjusted_segment)
                    else:
                        merged_segment_timestamps.append(segment_info)

        # Set the merged timestamps
        merged_hypothesis.timestamp = {
            'word': merged_word_timestamps,
            'segment': merged_segment_timestamps,
        }
    elif len(hypotheses_list) == 1 and timestamps:
        merged_hypothesis.timestamp = {
            'word': hypotheses_list[0].timestamp['word'],
            'segment': hypotheses_list[0].timestamp['segment'],
        }

    return merged_hypothesis
