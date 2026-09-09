# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import math
import random
import re
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from lhotse import CutSet
from lhotse.dataset.collation import collate_audio
from omegaconf import DictConfig, ListConfig
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.parts.alignments import WordAlignment, get_word_alignments_for_batch
from nemo.collections.speechlm2.parts.utils import to_dataclass

AUDIO_TOKEN_IDX = -200
IGNORE_INDEX = -100


def right_collate_vectors(
    tensors: Iterable[Union[torch.Tensor, np.ndarray]],
    padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
) -> torch.Tensor:
    tensors = [torch.as_tensor(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="right")


@dataclass
class StreamingSTTBatch:
    """
    A batch of data for StreamingSTTModel.

    Attributes:
        audios: (B, T) audio signals.
        audio_lens: (B,) lengths of the audio signals in samples.
        input_tokens: (B, L) input token IDs for the LLM. Audio positions are marked with AUDIO_TOKEN_IDX.
        input_token_lens: (B,) lengths of the input token sequences.
        target_tokens: (B, L) target token IDs for the LLM. Non-trainable positions are IGNORE_INDEX.
        target_token_lens: (B,) lengths of the target token sequences.
        text: list of ground-truth transcription strings.
        cuts: Optional[CutSet] containing the cuts for the batch.
    """

    audios: Optional[torch.Tensor] = None
    audio_lens: Optional[torch.Tensor] = None
    input_tokens: Optional[torch.Tensor] = None
    input_token_lens: Optional[torch.Tensor] = None
    target_tokens: Optional[torch.Tensor] = None
    target_token_lens: Optional[torch.Tensor] = None
    text: Optional[List[str]] = None
    cuts: Optional[CutSet] = None
    # Fixed-chunk frame count used to build this batch's turns. When the data
    # config's ``chunk_size`` is a list, one value is drawn per batch and stored
    # here so the model can align the encoder's attention context (look-ahead)
    # to the chunk size actually used. ``None`` / non-positive for dynamic
    # (0) or offline (-1) chunking.
    chunk_size: Optional[int] = None
    # K-frame grouping for dynamic chunking (effective only when
    # ``chunk_size == 0``). With a list ``data.chunk_step`` config, one K is
    # drawn per batch and stored here. ``None`` / 1 → no K-grouping.
    chunk_step: Optional[int] = None


@dataclass
class StreamingSTTDataConfig:
    sample_rate: int
    frame_length_in_secs: float
    # Frames per chunk. ``> 0`` fixed chunking, ``0`` dynamic, ``< 0`` offline.
    # May also be a list of positive ints (e.g. ``[2, 6, 13]``) — then one value
    # is drawn at random per batch (multi chunk-size training). All list entries
    # must be positive (fixed chunking only).
    chunk_size: Union[int, List[int]]
    num_delay_frames: int = 0
    words_per_group: int = 1
    audio_tag: str = "<audio>"
    blank_token: str = "<blank>"
    system_role: str = "system"
    system_prompt: str = "Transcribe the audio into text."
    prompt_field: str = "system_prompt"
    compact_template: bool = False
    # ``write_token`` is the start-of-text emit gate with ONE meaning in both
    # compact and non-compact modes: prepended to non-blank assistant content
    # only, gated by ``prepend_write_token``. It is LM-supervised — the model
    # predicts it as the binary emit decision.
    write_token: str = "<|write|>"
    # ``end_of_audio_token`` is the compact per-chunk audio->text scaffold anchor
    # (emitted once per chunk). It is force-fed at inference and NOT LM-supervised
    # in fixed chunking. Default ``<|im_start|>`` is already in Qwen3's vocab, so
    # the tokenizer is unchanged for the scaffold role. Compact-only; the
    # non-compact path uses the chat template's role headers for this boundary.
    end_of_audio_token: str = "<|im_start|>"
    # When True, prepend ``write_token`` to non-empty assistant content during
    # training data construction. Should match the model's ``prepend_write_token``
    # config. See StreamingSTTModelConfig.prepend_write_token for details.
    prepend_write_token: bool = False
    # K — only effective in dynamic chunking (chunk_size == 0). Each audio
    # segment is rounded UP to a multiple of K frames (and total audio is
    # padded to K-multiple). The model implicitly learns to emit only at
    # K-aligned positions; deploy-time K' (any multiple of K_train) is set via
    # dynamic_min_chunk_size / dynamic_max_chunk_size. Default 1 = no-op.
    # May also be a list of positive ints (e.g. ``[1, 3, 7]``) for multi
    # chunk-step training; one K is drawn per batch and recorded on the batch.
    chunk_step: Union[int, List[int]] = 1
    # Rung 1: drop the blank token and its turn scaffolding from the LLM context on
    # silent chunks. The blank is still SUPERVISED — it stays the target at the gate
    # position — it just never becomes an input token, so later chunks do not attend
    # to it. Requires a non-empty ``blank_token``; forced off with a warning otherwise
    # (there is no blank to drop, and the gate target would be undefined).
    drop_blank_from_context: bool = False
    # Rung 2: additionally drop the per-chunk gate anchor, so a silent chunk
    # contributes ONLY its audio frames and consecutive silent chunks become one
    # contiguous acoustic run. The gate moves to the last audio frame of each chunk
    # and becomes a binary blank / write choice, so ``prepend_write_token`` is
    # required. Implies ``drop_blank_from_context``.
    collapse_silent_audio: bool = False
    # Flush token: an explicit end-of-audio signal, fed by the harness after the last
    # audio frame, whose position supervises everything still pending.
    #
    # Without it the last chunk is special in training (``is_last_chunk`` force-emits a
    # sub-``words_per_group`` buffer, and residual words that the delay pushed past the
    # final boundary are folded back into the last assistant turn) but the model cannot
    # observe WHICH chunk is last at inference -- audio simply stops. Turning this on
    # removes both special cases: every chunk boundary obeys the same alignment/delay
    # rule, and everything left over is emitted after the flush token, which IS an
    # observable input. Must match the model flag of the same name.
    use_flush_token: bool = False
    # The flush marker itself. Registered as a new special token and learned from
    # scratch (no pretrained meaning to warm-start from), like ``write_token``.
    flush_token: str = "<|flush|>"


def decode_with_blank(
    ids: list[int],
    blank_token: str,
    tokenizer: AutoTokenizer,
    replace_blank: Optional[str] = None,
    strip_whitespace: bool = False,
    collapse_whitespace: bool = True,
    join_with: Optional[str] = " ",
    write_token: Optional[str] = None,
    replace_write: Optional[str] = None,
) -> str:
    """Decode token IDs, treating blank tokens as segment boundaries.

    Splits the token sequence at ``blank_token`` boundaries, decodes each
    segment separately (preserving BPE within each turn), then joins with
    spaces.

    Args:
        ids: Token IDs to decode.
        blank_token: The blank token string (e.g., ``"<blank>"``).
        tokenizer: NeMo AutoTokenizer.
        replace_blank: If provided, blank tokens are replaced with this string
            in the output instead of being skipped.  For example,
            ``replace_blank=""`` keeps the spacing, ``replace_blank="..."``
            inserts an ellipsis.
        strip_whitespace: If True, strip whitespace from the output.
        collapse_whitespace: If True, collapse multiple consecutive whitespace characters into a single space.
        join_with: If provided, join the segments divided by blank tokens with this string, else join with empty string.
        write_token: Optional write token string. When set, the write token is
            recognized (by default stripped from output; see ``replace_write``).
        replace_write: If provided, write tokens are replaced with this string
            in the output instead of being skipped. Symmetric to ``replace_blank``.
            Useful for diagnostic annotation (e.g., ``"[WRITE] "``).
    """
    if blank_token == "":
        # No blank token: use EOS (e.g. <|im_end|>) as chunk separator so
        # per-chunk outputs get joined with spaces instead of BPE-merged into one run.
        blank_id = tokenizer.tokenizer.eos_token_id
    else:
        blank_id = tokenizer.tokenizer.convert_tokens_to_ids(blank_token)
    write_id = None
    if write_token is not None:
        write_id = tokenizer.tokenizer.convert_tokens_to_ids(write_token)

    segments = []
    current = []
    for tid in ids:
        if tid == blank_id:
            if current:
                segments.append(tokenizer.ids_to_tokens(current))
                current = []
            if replace_blank is not None:
                segments.append(replace_blank)
        elif tid == write_id:
            if current:
                segments.append(tokenizer.ids_to_tokens(current))
                current = []
            if replace_write is not None:
                segments.append(replace_write)
        else:
            current.append(tid)
    if current:
        segments.append(tokenizer.ids_to_tokens(current))

    text_segments = []
    for seg in segments:
        if isinstance(seg, str):
            text_segments.append(seg)
        else:
            text_segments.append(tokenizer.tokens_to_text(seg, remove_special_tokens=True))
    text = join_with.join(text_segments) if join_with else "".join(text_segments)

    if strip_whitespace:
        text = text.strip()
    if collapse_whitespace:
        text = re.sub(r'\s+', ' ', text)
    return text


def compute_word_spans(
    alignments: List[WordAlignment],
    transcript: str,
    preserve_trailing_whitespace: bool = False,
    preserve_leading_whitespace: bool = False,
) -> List[tuple[int, int]]:
    """Find (start, end) character positions for each alignment word in the transcript.

    Trailing punctuation (non-alphanumeric, non-whitespace characters) that
    immediately follows a word is always included in the span so that commas,
    periods, quotes, etc. are preserved.

    Args:
        alignments: Word-level alignment results.
        transcript: Original transcription string.
        preserve_trailing_whitespace: When True, each span extends through
            trailing whitespace up to (but not including) the next alphanumeric
            character.  This is useful when extracting multi-word spans so
            that ``transcript[first_span[0]:last_span[1]]`` includes the
            inter-word spaces.
        preserve_leading_whitespace: When True, each span extends backward
            through preceding whitespace (not crossing the previous word's
            span end).  This matches GPT-style BPE tokenization where a
            leading space is part of the word token (e.g. ``" world"`` vs
            ``"world"``).  For ``"hello world"`` this yields
            ``[(0,5), (5,11)]`` = ``"hello"``, ``" world"``.

    Returns a list parallel to *alignments*.  If a word cannot be located, its
    span is ``None``.
    """

    if preserve_trailing_whitespace and preserve_leading_whitespace:
        raise ValueError(
            "preserve_trailing_whitespace and preserve_leading_whitespace cannot be True at the same time"
        )
    spans: List[tuple[int, int] | None] = []
    search_pos = 0
    for word in alignments:
        idx = transcript.lower().find(word.text.lower(), search_pos)
        if idx == -1:
            spans.append(None)
            continue
        start = idx
        # Optionally extend start backward through leading whitespace,
        # clamped at the previous word's span end.
        if preserve_leading_whitespace:
            while start > search_pos and transcript[start - 1].isspace():
                start -= 1
        end = idx + len(word.text)
        # Include trailing punctuation (e.g., comma, period, quotes)
        while end < len(transcript) and not transcript[end].isalnum() and not transcript[end].isspace():
            end += 1
        # Optionally include trailing whitespace up to the next word
        if preserve_trailing_whitespace:
            while end < len(transcript) and transcript[end].isspace():
                end += 1
        spans.append((start, end))
        search_pos = end
    return spans


def get_llm_messages_for_sample(
    system_role: str,
    system_prompt: str,
    audio_tag: str,
    blank_token: str,
    chunk_size: int,
    num_delay_frames: int,
    audio_duration_secs: float,
    frame_length_in_secs: float,
    alignments: Optional[List[WordAlignment]] = None,
    transcript: Optional[str] = None,
    words_per_group: int = 1,
    chunk_step: int = 1,
    prepend_write_token: bool = False,
    write_token: str = "",
    use_flush_token: bool = False,
    flush_token: str = "",
) -> List[dict]:
    """
    Get the LLM messages for a sample, using the alignments to determine the turns for the audio and text.

    The conversation is structured as alternating user (audio chunks) and assistant (transcription or blank) turns.
    A word becomes "ready" at the chunk whose end frame >= word_end_frame + num_delay_frames.

    For example, if the alignments are:
    [
        WordAlignment(text="Hello", start_time=0.16, end_time=0.48),
        WordAlignment(text="World", start_time=0.60, end_time=0.80),
    ]
    And the audio duration is 1s, audio_tag is "<audio>", chunk_size is 2, frame_length_in_secs is 0.08s,
    num_delay_frames is 0, then the messages will be:
    [
        {"role": "system", "content": "Transcribe the audio into text."},
        {"role": "user", "content": "<audio><audio>"},  # frames 0-1, 0~0.16s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 2-3, 0.16~0.32s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 4-5, 0.32~0.48s
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "<audio><audio>"},  # frames 6-7, 0.48~0.64s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 8-9, 0.64~0.80s
        {"role": "assistant", "content": "World"},
        {"role": "user", "content": "<audio><audio>"},  # frames 10-11, 0.80~0.96s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 12-13, 0.96~1.12s
        {"role": "assistant", "content": "<blank>"},
    ]

    Note: the last chunk may extend beyond audio_duration_secs since num_frames is
    ceiled to a multiple of chunk_size. The model must pad the audio accordingly.

    Args:
        system_role: The role of the system.
        system_prompt: The prompt for the system.
        audio_tag: The tag for the audio placeholder.
        blank_token: The token for blank/no-emission.
        chunk_size: The number of frames per chunk. If -1, the whole audio is used as a single chunk.
        num_delay_frames: Number of frames to delay word emission after word end.
        audio_duration_secs: The duration of the audio in seconds.
        frame_length_in_secs: The length of a single frame in seconds.
        alignments: List of WordAlignment objects for the sample.
    """

    messages = [{"role": system_role, "content": system_prompt}]

    num_frames = math.ceil(audio_duration_secs / frame_length_in_secs)

    if chunk_size < 0 or chunk_size is None:
        # Offline mode: use the whole audio as a single chunk
        num_chunks = 1 if num_frames > 0 else 0
        chunk_size = num_frames
        offline_mode = True
        num_delay_frames = 0  # delay is not used in offline mode
    else:
        offline_mode = False

    if alignments is None:
        alignments = []

    if offline_mode and not alignments:
        messages.append({"role": "user", "content": audio_tag * num_frames})
        messages.append({"role": "assistant", "content": transcript if transcript is not None else blank_token})
        return messages

    # Pre-compute word character spans if transcript is provided.
    word_spans = compute_word_spans(alignments, transcript, preserve_leading_whitespace=True) if transcript else None

    if chunk_size == 0:
        # Dynamic chunking: one user turn per word group, sized to word boundary.
        # The model learns to predict when to stop listening via audio-position targets.
        # When chunk_step > 1, each segment's frame count is rounded UP to a
        # multiple of K so the model only ever emits at K-aligned positions.
        K = max(int(chunk_step), 1)
        prev_end_frame = 0
        word_buffer: list[int] = []  # indices of buffered words

        for word_idx, word in enumerate(alignments):
            word_buffer.append(word_idx)

            # Emit when buffer reaches words_per_group or this is the last word
            if len(word_buffer) < words_per_group and word_idx < len(alignments) - 1:
                continue

            # Chunk boundary = end frame of the last word in this group, snapped
            # UP to the next multiple of K. num_frames here is already K-padded
            # (caller guarantees this), so the clamp keeps things K-aligned.
            last_word = alignments[word_buffer[-1]]
            group_end_frame = math.ceil(last_word.end_time / frame_length_in_secs) + num_delay_frames
            if K > 1:
                group_end_frame = ((group_end_frame + K - 1) // K) * K
            group_end_frame = min(group_end_frame, num_frames)
            n_frames_chunk = group_end_frame - prev_end_frame

            if n_frames_chunk > 0:
                messages.append({"role": "user", "content": audio_tag * n_frames_chunk})

            # Build assistant content from all buffered words
            if word_spans and transcript:
                first_span = word_spans[word_buffer[0]]
                last_span = word_spans[word_buffer[-1]]
                if first_span is not None and last_span is not None:
                    content = transcript[first_span[0] : last_span[1]]
                else:
                    content = " ".join(alignments[i].text for i in word_buffer)
            else:
                content = " ".join(alignments[i].text for i in word_buffer)

            if n_frames_chunk <= 0 and messages[-1]["role"] == "assistant":
                # Words at same boundary as previous group — append
                messages[-1]["content"] += " " + content
            else:
                if prepend_write_token and write_token:
                    content = write_token + content
                messages.append({"role": "assistant", "content": content})

            prev_end_frame = group_end_frame
            word_buffer = []

        # Trailing silence frames (after last word) — user turn only, no assistant.
        if prev_end_frame < num_frames:
            messages.append({"role": "user", "content": audio_tag * (num_frames - prev_end_frame)})
    else:
        # Fixed chunking: split the audio into equal-sized chunks.
        num_chunks = math.ceil(num_frames / chunk_size) if num_frames > 0 else 0

        word_idx = 0
        word_buffer: list[int] = []  # indices of words buffered for words_per_group grouping
        for chunk_i in range(num_chunks):
            chunk_end_frame = (chunk_i + 1) * chunk_size

            # User turn: one audio tag per frame in the chunk
            messages.append({"role": "user", "content": audio_tag * chunk_size})

            # Collect indices of words whose end_time (in frames) + delay <= chunk_end_frame
            while word_idx < len(alignments):
                word = alignments[word_idx]
                word_end_frame = math.ceil(word.end_time / frame_length_in_secs)
                ready_frame = word_end_frame + num_delay_frames
                if ready_frame <= chunk_end_frame:
                    word_buffer.append(word_idx)
                    word_idx += 1
                else:
                    break

            # Emit words when buffer reaches words_per_group, or at the last chunk.
            # The last-chunk exception is conditioned on `is_last_chunk`, which the
            # model CANNOT observe at inference (audio just stops). With
            # use_flush_token the exception is dropped so every boundary obeys one
            # rule, and whatever stays pending is emitted after the flush token --
            # an observable input. See StreamingSTTDataConfig.use_flush_token.
            is_last_chunk = (chunk_i == num_chunks - 1) and not use_flush_token
            if word_buffer and (len(word_buffer) >= words_per_group or is_last_chunk):
                if word_spans and transcript:
                    first_span = word_spans[word_buffer[0]]
                    last_span = word_spans[word_buffer[-1]]
                    if first_span is not None and last_span is not None:
                        content = transcript[first_span[0] : last_span[1]]
                    else:
                        content = " ".join(alignments[i].text for i in word_buffer)
                else:
                    content = " ".join(alignments[i].text for i in word_buffer)
                if prepend_write_token and write_token:
                    content = write_token + content
                messages.append({"role": "assistant", "content": content})
                word_buffer = []
            else:
                # Empty chunk: blank_token alone, NOT prefixed with write_token.
                messages.append({"role": "assistant", "content": blank_token})

        if use_flush_token:
            # Everything the per-chunk rule left pending: words still buffered below
            # words_per_group, plus words the delay pushed past the final boundary.
            # Both index ranges are contiguous and adjacent, so one slice covers them.
            start = word_buffer[0] if word_buffer else word_idx
            residual_indices = list(range(start, len(alignments)))
            messages.append({"role": "user", "content": flush_token})
            if residual_indices:
                content = _content_for_words(residual_indices, alignments, word_spans, transcript)
                if prepend_write_token and write_token:
                    content = write_token + content
                messages.append({"role": "assistant", "content": content})
            else:
                # Nothing pending. Still supervised -- the model must learn that
                # flush can legitimately mean "I have nothing left".
                messages.append({"role": "assistant", "content": blank_token})
            return messages

        # Append any residual words that weren't emitted (e.g., due to delay pushing
        # them past the last chunk boundary, or alignment end_time > audio_duration).
        if word_idx < len(alignments):
            residual_indices = list(range(word_idx, len(alignments)))
            if word_spans and transcript:
                first_span = word_spans[residual_indices[0]]
                last_span = word_spans[residual_indices[-1]]
                if first_span is not None and last_span is not None:
                    content = transcript[first_span[0] : last_span[1]]
                else:
                    content = " ".join(alignments[i].text for i in residual_indices)
            else:
                content = " ".join(alignments[i].text for i in residual_indices)
            if messages[-1]["role"] == "assistant" and messages[-1]["content"] == blank_token:
                # Replacing a blank-only chunk with real content — needs write_token prefix
                if prepend_write_token and write_token:
                    content = write_token + content
                messages[-1]["content"] = content
            elif messages[-1]["role"] == "assistant":
                # Appending to an existing non-empty assistant turn — already has the prefix
                messages[-1]["content"] += " " + content
            else:
                if prepend_write_token and write_token:
                    content = write_token + content
                messages.append({"role": "assistant", "content": content})

    return messages


def get_llm_messages_for_batch(
    system_role: str,
    system_prompt: List[str],
    audio_tag: str,
    blank_token: str,
    chunk_size: int,
    num_delay_frames: int,
    audio_durations_secs: List[float],
    frame_length_in_secs: float,
    alignments: Optional[List[List[WordAlignment]]] = None,
    transcripts: Optional[List[str]] = None,
    words_per_group: int = 1,
    chunk_step: int = 1,
    prepend_write_token: bool = False,
    write_token: str = "",
    use_flush_token: bool = False,
    flush_token: str = "",
) -> List[List[dict]]:
    """
    Get the LLM messages for a batch of samples.

    Args:
        system_role: The role of the system.
        system_prompt: The list of prompts for each sample in the batch.
        audio_tag: The tag for the audio placeholder.
        blank_token: The token for blank/no-emission.
        chunk_size: The number of frames per chunk.
        num_delay_frames: Number of frames to delay word emission after word end.
        audio_durations_secs: List of audio durations in seconds, one per sample.
        frame_length_in_secs: The length of a single frame in seconds.
        alignments: List of lists of WordAlignment objects for the batch.
        transcripts: Original transcription strings, one per sample.  When provided,
            assistant turn content preserves punctuation and spacing from the transcript.
        words_per_group: Minimum number of words to buffer before emitting an
            assistant turn (default 1 = emit each word immediately).
    """
    if transcripts is None:
        transcripts = [None] * len(audio_durations_secs)
    batch_messages = []
    for sample_alignments, duration_secs, prompt, transcript in zip(
        alignments,
        audio_durations_secs,
        system_prompt,
        transcripts,
    ):
        batch_messages.append(
            get_llm_messages_for_sample(
                system_role=system_role,
                system_prompt=prompt,
                audio_tag=audio_tag,
                blank_token=blank_token,
                chunk_size=chunk_size,
                num_delay_frames=num_delay_frames,
                audio_duration_secs=duration_secs,
                frame_length_in_secs=frame_length_in_secs,
                alignments=sample_alignments,
                transcript=transcript,
                words_per_group=words_per_group,
                chunk_step=chunk_step,
                prepend_write_token=prepend_write_token,
                write_token=write_token,
                use_flush_token=use_flush_token,
                flush_token=flush_token,
            )
        )
    return batch_messages


def resolve_pad_id(tokenizer: AutoTokenizer) -> int:
    """Padding token ID for a tokenizer that may not define one.

    Falls back to ``<unk>`` and finally to id 0. Some LLM tokenizers ship without
    a pad token (e.g. NVIDIA-Nemotron-3-Nano, where ``pad_id`` is ``None`` and
    ``unk_id`` is 0) — passing that ``None`` into ``pad_sequence`` raises
    ``TypeError: argument 'padding_value' must be float, not NoneType``.

    The dataset and :attr:`StreamingSTTModel.text_pad_id` MUST agree on this
    value: the model derives its attention mask as ``input_tokens != pad_id``, so
    a mismatch would either unmask padding or mask real tokens. Both call here.
    """
    pad_id = tokenizer.pad_id
    if pad_id is None:
        pad_id = getattr(tokenizer, "unk_id", None)
    if pad_id is None:
        warnings.warn(
            "The text tokenizer has no <pad> or <unk> token; using id 0 for padding "
            "(this may lead to silent bugs).",
            stacklevel=2,
        )
        pad_id = 0
    return int(pad_id)


def apply_chat_template_ids(hf_tok, messages: List[dict], **kwargs) -> list[int]:
    """``apply_chat_template(..., tokenize=True)`` normalized to a flat list of token IDs.

    transformers 4.x returns a plain ``list[int]`` from a tokenizing call, while
    transformers 5.x always returns a ``BatchEncoding`` (``return_dict`` became
    the default). Callers here only ever want the IDs, so unwrap the mapping —
    and a batched ``[[ids]]`` layout too, in case a future version wraps single
    conversations.

    Args:
        hf_tok: A HuggingFace tokenizer (``tokenizer.tokenizer``).
        messages: ``[{"role": ..., "content": ...}, ...]``.
        kwargs: Forwarded to ``apply_chat_template`` (e.g. ``add_generation_prompt``,
            ``enable_thinking``). Do not pass ``tokenize`` or ``return_dict``.

    Returns:
        The conversation's token IDs as a flat ``list[int]``.
    """
    out = hf_tok.apply_chat_template(messages, tokenize=True, **kwargs)
    if hasattr(out, "keys"):  # BatchEncoding / dict — transformers >= 5
        out = out["input_ids"]
    if hasattr(out, "tolist"):  # torch tensor (only when return_tensors was requested)
        out = out.tolist()
    if len(out) > 0 and isinstance(out[0], (list, tuple)):  # batched [[ids]]
        out = out[0]
    return list(out)


def parse_chat_template_ids(
    hf_tok,
    last_turn: bool = False,
    probe_content: str = "<audio>",
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Discover turn-structure token IDs from a HuggingFace chat template.

    Returns the four structural spans that surround user and assistant content in
    a *mid-conversation* turn, so that streaming inference can reproduce, token
    for token, what ``apply_chat_template`` emits for that turn during training.

    The spans are located by **index in the token-id sequence**, by diffing
    renders that differ only in message content — never by splitting the rendered
    template string and re-encoding fragments. Re-encoding a fragment can tokenize
    differently than the same text does in context (BPE merges at the artificial
    boundary, SentencePiece word-start markers), and the failure is silent.

    Both probe renders are anchored to a **system message**. This matters:
    templates that emit a system block unconditionally (several Nemotron
    variants) would otherwise fold an empty system block into ``user_header``,
    which streaming inference then re-feeds once per chunk while training emits
    it only once, at the front of the conversation.

    ``last_turn`` selects which assistant header is returned. Qwen3 injects
    ``<think>``/``</think>`` suppression tags only on the *final* assistant turn,
    so a mid-stream chunk needs the non-final variant; a single-turn offline
    prompt needs the final one. (Nemotron emits thinking tags on every assistant
    turn, so the distinction is a no-op there.)

    Args:
        hf_tok: A HuggingFace tokenizer (``tokenizer.tokenizer``).
        last_turn: When True, return the assistant header as it renders on the
            last turn of a conversation (may include thinking-suppression tags).
        probe_content: User-turn content used to locate the audio span. Any
            non-empty string works; the audio tag is the natural choice because
            it is what really occupies that position.

    Returns:
        ``(user_header_ids, user_footer_ids, asst_header_ids, asst_footer_ids)``

        - *user_header_ids*: tokens between the system block and user content
          (e.g. ``[<|im_start|>, user, \n]``).
        - *user_footer_ids*: tokens after user content, up to the assistant
          header (e.g. ``[<|im_end|>, \n]``).
        - *asst_header_ids*: tokens before assistant content
          (e.g. ``[<|im_start|>, assistant, \n]``).
        - *asst_footer_ids*: tokens after assistant content
          (e.g. ``[<|im_end|>, \n]``). Empty when the template's footer is
          whitespace-only — see the guard at the end of this function.
    """
    # The derived spans must be invariant to this string's content: it is excluded
    # by the ``A[:len(S)] == S`` prefix. Deliberately distinctive so that the
    # post-condition below can detect a template folding it into the user turn.
    _SYS = "ZZPROBESYSZZ"
    _SENTINEL = "XSENTINELX"

    def render(messages: List[dict]) -> list[int]:
        return apply_chat_template_ids(
            hf_tok,
            messages,
            add_generation_prompt=False,
            enable_thinking=False,
        )

    sys_msg = {"role": "system", "content": _SYS}
    user_msg = {"role": "user", "content": probe_content}

    # --- user spans: diff a rendered user turn against the same turn emptied ---
    s_ids = render([sys_msg])
    a_ids = render([sys_msg, user_msg])
    a_empty = render([sys_msg, {"role": "user", "content": ""}])

    _assert_prefix(a_ids, s_ids, hf_tok, "system-only render is not a prefix of the user render")

    head = len(s_ids) + _common_prefix_len(a_ids[len(s_ids) :], a_empty[len(s_ids) :])
    tail = _common_suffix_len(a_ids, a_empty)
    user_header_ids = a_ids[len(s_ids) : head]
    user_footer_ids = a_ids[len(a_ids) - tail :]
    _assert_partitions(a_ids, len(s_ids), head, tail, hf_tok, _SYS, user_header_ids)

    # --- assistant spans: append an assistant turn and diff again ---
    b2 = render([sys_msg, user_msg, {"role": "assistant", "content": _SENTINEL}])
    b2_empty = render([sys_msg, user_msg, {"role": "assistant", "content": ""}])
    _assert_prefix(b2, a_ids, hf_tok, "user render is not a prefix of the user+assistant render")

    asst_footer_ids = b2[len(b2) - _common_suffix_len(b2, b2_empty) :]

    if last_turn:
        asst_header_ids = b2[len(a_ids) : len(a_ids) + _common_prefix_len(b2[len(a_ids) :], b2_empty[len(a_ids) :])]
    else:
        # Place the probed assistant turn in the middle of the conversation so
        # last-turn-only template behaviour (Qwen3 thinking tags) is excluded.
        trailing = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "x"}]
        b4 = render([sys_msg, user_msg, {"role": "assistant", "content": _SENTINEL}] + trailing)
        b4_empty = render([sys_msg, user_msg, {"role": "assistant", "content": ""}] + trailing)
        _assert_prefix(b4, a_ids, hf_tok, "user render is not a prefix of the 4-message render")
        asst_header_ids = b4[len(a_ids) : len(a_ids) + _common_prefix_len(b4[len(a_ids) :], b4_empty[len(a_ids) :])]

    # A whitespace-only assistant footer is treated as absent, matching the
    # historical behaviour. This is not cosmetic: ``_autoregressive_decode``
    # stops a stream whenever its tail matches ``asst_footer_ids``, so a footer
    # of a bare newline byte (Nemotron-Mini) would truncate any hypothesis at
    # its first newline.
    if asst_footer_ids and not hf_tok.decode(asst_footer_ids).strip():
        asst_footer_ids = []

    return user_header_ids, user_footer_ids, asst_header_ids, asst_footer_ids


def build_compact_turn_markers(hf_tok, end_of_audio_token: str) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return the compact-format analogue of ``parse_chat_template_ids``.

    Compact format drops the user/assistant role delimiters: turns look like
    ``<audio>*N <end_of_audio_token> TEXT <eos>`` with no header before audio and
    only the ``end_of_audio_token`` marking the audio→text transition.  The
    turn-end is the tokenizer's native EOS.

    The end-of-audio anchor is returned as the **user footer** (with an empty
    assistant header), not the other way round: it plays the role of the
    audio→text boundary, which is what ``_user_footer_first_id`` must point at.

    ``end_of_audio_token`` should be an existing vocab token the LLM saw
    pretraining as a turn-boundary marker (e.g. ``"<|im_start|>"`` for Qwen3,
    ``"<start_of_turn>"`` for Gemma).
    """
    eoa_ids = hf_tok.encode(end_of_audio_token, add_special_tokens=False)
    if len(eoa_ids) != 1:
        raise ValueError(
            f"end_of_audio_token {end_of_audio_token!r} must encode to exactly 1 token, got {eoa_ids}. "
            f"Pick a tokenizer-native turn-boundary token or override via config."
        )
    eos_id = getattr(hf_tok, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is required for compact_template=True")
    return [], [eoa_ids[0]], [], [eos_id]


def _tokenize_compact_with_assistant_mask(
    messages: List[dict],
    tokenizer: AutoTokenizer,
    end_of_audio_id: int,
    eos_id: int,
    drop_blank_from_context: bool = False,
    blank_token: str = "",
    collapse_silent_audio: bool = False,
    flush_id: Optional[int] = None,
) -> tuple[list[int], list[int]]:
    """Tokenize chat messages in compact format and return (input_ids, assistant_mask).

    Compact per-turn layout (no role wrapping between audio and text):
        [system_wrapped] [user_content, <eoa>, asst_content, <eos>]*K

    where ``<eoa>`` is the end-of-audio scaffold anchor and ``asst_content`` may
    begin with the ``write_token`` emit gate (prepended upstream in
    ``get_llm_messages_for_sample`` when ``prepend_write_token`` is set).

    The system prompt IS still wrapped via ``apply_chat_template`` (Qwen3 system
    block), only the per-turn scaffolding is compacted.  Loss is applied on
    assistant content (incl. the ``write_token`` gate) and ``<eos>``.  The
    ``<eoa>`` anchor is force-fed at inference and is NOT LM-supervised in fixed
    chunking (``assistant_mask=0``) — mirroring the non-compact path where the
    ``<|im_start|>assistant\\n`` boundary header is also untrained.  (For dynamic
    chunking the boundary IS supervised, but via the audio-position target
    override in ``get_batch_data``, not this mask.)
    """
    hf_tok = tokenizer.tokenizer

    input_ids: list[int] = []
    assistant_mask: list[int] = []

    # --- System section: keep Qwen3-style wrapping ---
    system_msgs = [m for m in messages if m["role"] == "system"]
    if system_msgs:
        system_ids = apply_chat_template_ids(
            hf_tok,
            system_msgs,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        input_ids.extend(system_ids)
        assistant_mask.extend([0] * len(system_ids))

    # --- Per-turn compact encoding ---
    turn_msgs = [m for m in messages if m["role"] != "system"]
    # Pairs: (user, assistant). The final turn may be user-only (trailing silence).
    i = 0
    while i < len(turn_msgs):
        msg = turn_msgs[i]
        if msg["role"] == "user":
            user_ids = hf_tok.encode(msg["content"], add_special_tokens=False) if msg["content"] else []
            is_flush = flush_id is not None and user_ids == [flush_id]
            input_ids.extend(user_ids)
            assistant_mask.extend([0] * len(user_ids))
            i += 1
            # Pair with following assistant turn if present.
            if i < len(turn_msgs) and turn_msgs[i]["role"] == "assistant":
                asst = turn_msgs[i]
                is_silent = bool(blank_token) and asst["content"] == blank_token
                if is_flush and collapse_silent_audio:
                    # Rung 2 keeps no anchor, so the flush marker IS this turn's gate:
                    # a speaking turn runs straight from it into the write token, and a
                    # silent one contributes nothing at all -- exactly what rung 2 does
                    # at every other boundary. The "nothing left" case is supervised as
                    # a blank TARGET on the marker (see get_batch_data), never as a
                    # blank input, which is the whole point of the rung.
                    if is_silent:
                        i += 1
                        continue
                    asst_ids = hf_tok.encode(asst["content"], add_special_tokens=False)
                    input_ids.extend(asst_ids)
                    assistant_mask.extend([1] * len(asst_ids))
                    input_ids.append(eos_id)
                    assistant_mask.append(1)
                    i += 1
                    continue
                # Rungs 0 and 1 fall through to the normal turn handling below, so the
                # flush turn keeps the <eoa> anchor. Emission then triggers off the same
                # position it does at every other chunk boundary; flush only modifies
                # THAT boundary rather than introducing a second, rarely-seen trigger.
                if collapse_silent_audio:
                    # Rung 2: no anchor at all. A silent chunk contributes nothing
                    # beyond its audio; a speaking chunk goes straight from the last
                    # audio frame into the write token and its text.
                    if is_silent:
                        i += 1
                        continue
                    asst_ids = hf_tok.encode(asst["content"], add_special_tokens=False)
                    input_ids.extend(asst_ids)
                    assistant_mask.extend([1] * len(asst_ids))
                    input_ids.append(eos_id)
                    assistant_mask.append(1)
                    i += 1
                    continue
                if drop_blank_from_context and blank_token and is_silent:
                    # Silent chunk under rung 1: emit the gate anchor and nothing else.
                    # The blank stays supervised via a target override in
                    # ``get_batch_data`` -- it is simply never an input token, so
                    # subsequent chunks do not attend to it.
                    input_ids.append(end_of_audio_id)
                    assistant_mask.append(0)
                    i += 1
                    continue
                asst_ids = hf_tok.encode(asst["content"], add_special_tokens=False) if asst["content"] else []
                # <eoa> anchor: force-fed scaffold, not LM-supervised (mask=0).
                input_ids.append(end_of_audio_id)
                assistant_mask.append(0)
                # assistant content (may start with the write_token emit gate)
                input_ids.extend(asst_ids)
                assistant_mask.extend([1] * len(asst_ids))
                # eos
                input_ids.append(eos_id)
                assistant_mask.append(1)
                i += 1
        else:
            # Orphan assistant (shouldn't normally occur) — treat as standalone asst segment.
            asst_ids = hf_tok.encode(msg["content"], add_special_tokens=False) if msg["content"] else []
            input_ids.append(end_of_audio_id)
            assistant_mask.append(0)
            input_ids.extend(asst_ids)
            assistant_mask.extend([1] * len(asst_ids))
            input_ids.append(eos_id)
            assistant_mask.append(1)
            i += 1

    return input_ids, assistant_mask


def _tokenize_with_assistant_mask(
    messages: List[dict],
    tokenizer: AutoTokenizer,
) -> tuple[list[int], list[int]]:
    """
    Tokenize chat messages and return (input_ids, assistant_mask).

    First tries HF's ``return_assistant_tokens_mask`` (requires ``{% generation %}``
    in the chat template).  If that returns an all-zero mask, falls back to a
    sequential-search strategy: tokenize each assistant turn's content separately
    and locate it in the full token sequence.

    Args:
        messages: list of ``{"role": ..., "content": ...}`` dicts.
        tokenizer: NeMo AutoTokenizer (``tokenizer.tokenizer`` is the HF tokenizer).

    Returns:
        (input_ids, assistant_mask) — both plain Python lists of ints.
    """
    hf_tok = tokenizer.tokenizer

    # --- primary path: use HF's built-in mask ---
    result = hf_tok.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=False,
    )
    input_ids = list(result["input_ids"])
    assistant_mask = list(result["assistant_masks"])

    if any(assistant_mask):
        return input_ids, assistant_mask

    # --- fallback: diff-based content detection ---
    # Tokenize the same messages but with all assistant contents replaced by
    # a single-character sentinel.  The two-pointer walk then identifies
    # content tokens (present in full but replaced by the sentinel in the
    # reference).
    #
    # We use a sentinel instead of "" (empty string) to preserve BPE context
    # boundaries.  With "", template tokens adjacent to the content can merge
    # (e.g. "assistant\n" + content "\n" → token "\n\n" vs "assistant\n" + "" →
    # token "\n"), causing the two-pointer to desync.  A sentinel like "X"
    # tokenizes to exactly 1 token and prevents BPE merging with neighbors.
    _SENTINEL_CHAR = "X"
    assistant_mask = [0] * len(input_ids)

    msgs_sentinel = [{**m, "content": _SENTINEL_CHAR} if m["role"] == "assistant" else m for m in messages]
    ids_sentinel = apply_chat_template_ids(hf_tok, msgs_sentinel, enable_thinking=False)

    eos_id = getattr(hf_tok, 'eos_token_id', None)
    i, j = 0, 0  # pointers into input_ids and ids_sentinel
    while i < len(input_ids) and j < len(ids_sentinel):
        if input_ids[i] == ids_sentinel[j]:
            i += 1
            j += 1
        else:
            # Divergence: ids_sentinel has the sentinel (1 token) where
            # input_ids has the actual content (1+ tokens).
            j += 1  # skip the sentinel token
            while i < len(input_ids) and (j >= len(ids_sentinel) or input_ids[i] != ids_sentinel[j]):
                assistant_mask[i] = 1
                i += 1
            # Include EOS token in the footer so the model learns to emit it.
            if eos_id is not None and i < len(input_ids) and input_ids[i] == eos_id:
                assistant_mask[i] = 1

    # Any remaining tokens in input_ids are also content.
    while i < len(input_ids):
        assistant_mask[i] = 1
        i += 1

    return input_ids, assistant_mask


def _replace_audio_chunks(
    token_ids: list[int],
    chunk_ids: list[int],
    chunk_size: int,
    mask: list | None = None,
) -> list[int] | tuple[list[int], list]:
    """Replace each occurrence of *chunk_ids* with *chunk_size* copies of ``AUDIO_TOKEN_IDX``.

    This handles multi-token audio tags where BPE merges tokens across adjacent
    tags (e.g., ``<audio><audio>`` tokenizes differently from ``encode("<audio>") * 2``).
    By matching the full chunk at once, we avoid the BPE boundary problem.

    When *mask* is provided it is adjusted in sync: each matched span is replaced
    with *chunk_size* copies of the first element of that span (typically 0 for
    user-turn content).

    Returns:
        new_token_ids            when mask is None
        (new_token_ids, new_mask) when mask is provided
    """
    chunk_len = len(chunk_ids)
    new_ids: list[int] = []
    new_mask: list | None = [] if mask is not None else None
    i = 0
    n = len(token_ids)
    while i < n:
        if token_ids[i : i + chunk_len] == chunk_ids:
            new_ids.extend([AUDIO_TOKEN_IDX] * chunk_size)
            if new_mask is not None:
                new_mask.extend([mask[i]] * chunk_size)
            i += chunk_len
        else:
            new_ids.append(token_ids[i])
            if new_mask is not None:
                new_mask.append(mask[i])
            i += 1
    return (new_ids, new_mask) if mask is not None else new_ids


class StreamingSTTDataset(torch.utils.data.Dataset):
    """
    Dataset for StreamingSTTModel.
    Operates directly on Lhotse Cuts (no NeMoMultimodalConversation wrapper).
    """

    def __init__(self, cfg: DictConfig | dict, tokenizer: AutoTokenizer, defer_get_batch: bool = False):
        """
        Args:
            cfg: Configuration for the dataset.
            tokenizer: Tokenizer for the dataset.
            defer_get_batch: If True, defer the get_batch_data call to the __getitem__ method and let the model do it.
                This is used in online forced alignment mode.
        """
        self.defer_get_batch = defer_get_batch
        self.tokenizer = tokenizer
        self.cfg: StreamingSTTDataConfig = to_dataclass(StreamingSTTDataConfig, cfg)
        # Unescape Python escape sequences (e.g. "\\n" → "\n") because Hydra/OmegaConf
        # loads YAML strings literally without interpreting backslash escapes.
        self.cfg.blank_token = self.cfg.blank_token.encode().decode('unicode_escape')
        self.cfg.write_token = self.cfg.write_token.encode().decode('unicode_escape')
        self.cfg.end_of_audio_token = self.cfg.end_of_audio_token.encode().decode('unicode_escape')

        # Normalize chunk_size into a list of candidate fixed-chunk sizes for
        # per-batch random selection. A scalar config yields ``None`` (no random
        # selection — the single value is used directly). A list config enables
        # multi chunk-size training and must contain only positive sizes.
        # If chunk_size_override is provided, use it instead of chunk_size,
        # which is used for validation loop and inference.
        cs = self.cfg.chunk_size
        if isinstance(cs, (list, tuple, ListConfig)):
            self._chunk_size_candidates = [int(x) for x in cs]
            if not self._chunk_size_candidates:
                raise ValueError("chunk_size list must be non-empty")
            if any(x <= 0 for x in self._chunk_size_candidates):
                raise ValueError(
                    f"All chunk sizes in a list must be positive (fixed chunking), "
                    f"got {self._chunk_size_candidates}"
                )
            logging.info(f"Multi chunk-size training enabled: candidates={self._chunk_size_candidates}")
        else:
            self._chunk_size_candidates = None

        # Normalize chunk_step into a list of candidate K values for per-batch
        # random selection (multi chunk-step training). Scalar → ``None`` (the
        # single value is used directly). Only effective for dynamic chunking
        # (chunk_size == 0); ignored otherwise.
        cs_step = getattr(self.cfg, "chunk_step", 1)
        if isinstance(cs_step, (list, tuple, ListConfig)):
            self._chunk_step_candidates = [max(int(x), 1) for x in cs_step]
            if not self._chunk_step_candidates:
                raise ValueError("chunk_step list must be non-empty")
            logging.info(f"Multi chunk-step training enabled: candidates={self._chunk_step_candidates}")
        else:
            self._chunk_step_candidates = None

        # Tokenize the full audio chunk string (audio_tag * chunk_size) to get
        # its token ID sequence, one entry per positive fixed-chunk size.  We must
        # encode the full chunk as a single string because BPE may merge tokens
        # across adjacent audio tags (e.g., "<audio><audio>" tokenizes differently
        # from encode("<audio>") * 2).  When chunk_size=-1 (offline) or 0 (dynamic),
        # audio tag counts vary per turn and are encoded on demand in get_batch_data.
        if self._chunk_size_candidates is not None:
            positive_sizes = self._chunk_size_candidates
        elif isinstance(cs, int) and cs > 0:
            positive_sizes = [cs]
        else:
            positive_sizes = []
        # When the audio tag is a single vocab id (``model.register_audio_token``),
        # every frame is exactly one token and the whole-chunk matcher is unnecessary:
        # the replacement becomes a per-token map that cannot mis-fire. Otherwise fall
        # back to matching ``audio_tag * chunk_size`` as a unit, which is what keeps a
        # multi-token tag safe against BPE merging across adjacent tags.
        audio_tag_ids = self.tokenizer.tokenizer.encode(self.cfg.audio_tag, add_special_tokens=False)
        self._audio_token_id: Optional[int] = audio_tag_ids[0] if len(audio_tag_ids) == 1 else None
        if self._audio_token_id is not None:
            logging.info(
                f"audio_tag {self.cfg.audio_tag!r} is a single token (id={self._audio_token_id}); "
                "using the per-token audio mapping."
            )
        self._audio_chunk_ids_by_size: dict[int, list[int]] = (
            {}
            if self._audio_token_id is not None
            else {
                size: self.tokenizer.tokenizer.encode(self.cfg.audio_tag * size, add_special_tokens=False)
                for size in positive_sizes
            }
        )

        # blank_token is part of the LLM output vocabulary — it must be a single
        # special token, otherwise loss is dominated by multi-token blanks and
        # generation becomes unreliable.  The model's __init__ should have called
        # tokenizer.add_special_tokens() before passing the tokenizer here.
        # An empty blank_token ("") disables the explicit blank: chunks without
        # words get empty assistant turns, stop signal is <|im_end|> alone.
        if self.cfg.blank_token == "":
            if self.cfg.chunk_size == 0:
                raise ValueError(
                    "blank_token='' is not supported with dynamic chunking (chunk_size=0) — "
                    "dynamic chunking requires a token to predict at non-final audio positions."
                )
            self.blank_id = -1
            logging.info("blank_token is empty: blank token mechanism disabled (fixed chunking only)")
        else:
            blank_ids = self.tokenizer.tokenizer.encode(self.cfg.blank_token, add_special_tokens=False)
            logging.info(f"blank_token: {str(self.cfg.blank_token)}, blank_id: {blank_ids}")
            if len(blank_ids) != 1:
                raise ValueError(
                    f"blank_token '{self.cfg.blank_token}' tokenizes into {len(blank_ids)} tokens {blank_ids}. "
                    f"It must be a single special token. Make sure the model adds it via "
                    f"tokenizer.add_special_tokens() before constructing the dataset."
                )
            self.blank_id = blank_ids[0]

        # Rung 1 flag, resolved once. Requires a real blank token: with
        # ``blank_token: ""`` a silent chunk is already just ``[audio][eoa][eos]``,
        # there is no blank to drop, and the gate target would be undefined.
        self._drop_blank = bool(getattr(self.cfg, "drop_blank_from_context", False))
        if self._drop_blank and self.cfg.blank_token == "":
            warnings.warn(
                "drop_blank_from_context=True requires a non-empty blank_token; disabling it. "
                "There is no blank token to drop and the gate target would be undefined.",
                stacklevel=2,
            )
            self._drop_blank = False
        self._collapse_audio = bool(getattr(self.cfg, "collapse_silent_audio", False))
        if self._collapse_audio:
            self._drop_blank = True  # rung 2 implies rung 1
            if self.cfg.blank_token == "":
                raise ValueError(
                    "collapse_silent_audio=True requires a non-empty blank_token: the gate at the "
                    "last audio frame distinguishes 'silent' from 'text starts here' by predicting "
                    "blank, so there must be a blank to predict."
                )
            logging.info("collapse_silent_audio enabled: silent chunks contribute audio frames only")

        # Flush token, resolved once. Fixed chunking only: in dynamic chunking (K=0)
        # segments are already sized to word boundaries, so there is no
        # ``is_last_chunk`` special case to remove and nothing pending at audio end.
        self._use_flush = bool(getattr(self.cfg, "use_flush_token", False))
        self._flush_id = None
        if self._use_flush:
            cs = self.cfg.chunk_size
            sizes = cs if isinstance(cs, (list, tuple)) else [cs]
            if any(int(s) <= 0 for s in sizes):
                raise ValueError(
                    "use_flush_token=True requires fixed chunking (all chunk_size > 0); got "
                    f"chunk_size={cs!r}. Dynamic/offline modes have no last-chunk special case "
                    "to remove."
                )
            flush_ids = self.tokenizer.tokenizer.encode(self.cfg.flush_token, add_special_tokens=False)
            if len(flush_ids) != 1:
                raise ValueError(
                    f"flush_token {self.cfg.flush_token!r} must encode to exactly 1 token, got "
                    f"{flush_ids}. Register it on the tokenizer first (the model does this when "
                    "StreamingSTTModelConfig.use_flush_token is set)."
                )
            self._flush_id = flush_ids[0]
            logging.info(
                f"use_flush_token enabled: flush_token={self.cfg.flush_token!r} (id={self._flush_id}); "
                "the last-chunk emit exception is disabled and residual words move to the flush turn"
            )

        if self._drop_blank and not self.cfg.compact_template:
            raise NotImplementedError(
                "drop_blank_from_context is only implemented for compact_template=True so far "
                "(the non-compact two-stage turn feed is a later milestone)."
            )
        if self._drop_blank:
            logging.info("drop_blank_from_context enabled: silent chunks contribute [audio, anchor] only")

        # Compact template: cache the end-of-audio anchor id and eos_id. Skip the
        # parse_chat_template_ids call since we derive the markers directly from config.
        if self.cfg.compact_template:
            hf_tok = self.tokenizer.tokenizer
            _, uf_ids, _, af_ids = build_compact_turn_markers(hf_tok, self.cfg.end_of_audio_token)
            self._eoa_id = uf_ids[0]
            self._compact_eos_id = af_ids[0]
            logging.info(
                f"compact_template enabled: end_of_audio_token={self.cfg.end_of_audio_token!r} "
                f"(id={self._eoa_id}), eos_id={self._compact_eos_id}"
            )
        else:
            self._eoa_id = None
            self._compact_eos_id = None

        # For dynamic chunking (chunk_size=0): cache the first token of the
        # user footer sequence (e.g. <|im_end|>).  This is the target the model
        # predicts at the last audio frame of each chunk to signal "ready to transcribe".
        if self.cfg.chunk_size == 0:
            if self.cfg.compact_template:
                # Compact: the boundary target is the end-of-audio anchor
                # (<|im_start|> by default). write_token is reserved for the
                # text-emit gate and is never a boundary signal.
                self._user_footer_first_id = self._eoa_id
            else:
                hf_tok = self.tokenizer.tokenizer
                _, user_footer_ids, asst_header_ids, _ = parse_chat_template_ids(
                    hf_tok, probe_content=self.cfg.audio_tag
                )
                # Must stay character-for-character identical to the model-side
                # derivation in ``_ensure_inference_cache``: this id is the
                # supervised target at the last audio frame of every dynamic
                # chunk, and the model tests generated tokens against its own
                # copy. A divergence raises nothing — the emit gate simply
                # never fires.
                self._user_footer_first_id = (
                    user_footer_ids[0] if user_footer_ids else (asst_header_ids[0] if asst_header_ids else None)
                )
        else:
            self._user_footer_first_id = None

    def __getitem__(self, cuts: CutSet) -> StreamingSTTBatch | None:
        try:
            audios, audio_lens, cuts = collate_audio(cuts, fault_tolerant=True)
        except Exception as e:
            logging.warning(f"Error collating audio from cuts: {e}")
            return None
        if len(cuts) == 0:
            logging.warning("No cuts found in the batch")
            return None

        text = [cut.supervisions[0].text for cut in cuts]

        if self.defer_get_batch:
            return StreamingSTTBatch(
                cuts=cuts,
                audios=audios,
                audio_lens=audio_lens,
                text=text,
            )

        alignments = get_word_alignments_for_batch(cuts)

        return self.get_batch_data(cuts, audios, audio_lens, alignments, text)

    def get_batch_data(
        self,
        cuts: CutSet,
        audios: torch.Tensor,
        audio_lens: torch.Tensor,
        alignments: List[List[WordAlignment]],
        text: List[str],
    ) -> StreamingSTTBatch:
        audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()

        # Pick the fixed-chunk size for this batch. With a list config, draw one
        # value at random (multi chunk-size training); otherwise use the scalar.
        if self._chunk_size_candidates is not None:
            chunk_size = random.choice(self._chunk_size_candidates)
        else:
            chunk_size = self.cfg.chunk_size

        # K-step alignment (dynamic chunking only): pad each waveform up to a
        # multiple of K frames so the encoder produces exactly that many
        # embeddings, matching the K-snapped segment lengths the dataset will
        # construct below. K=1 → no-op.
        # With a list ``chunk_step`` config, draw one K per batch (multi
        # chunk-step training); otherwise use the scalar.
        if self._chunk_step_candidates is not None:
            K = random.choice(self._chunk_step_candidates)
        else:
            K = max(int(getattr(self.cfg, "chunk_step", 1)), 1)
        if K > 1 and chunk_size == 0:
            new_lens = []
            for dur in audio_durations_secs:
                num_frames = math.ceil(dur / self.cfg.frame_length_in_secs)
                num_frames_padded = math.ceil(num_frames / K) * K
                samples_padded = math.ceil(num_frames_padded * self.cfg.frame_length_in_secs * self.cfg.sample_rate)
                new_lens.append(samples_padded)
            max_samples = max(new_lens) if new_lens else int(audio_lens.max().item())
            if audios.shape[1] < max_samples:
                audios = F.pad(audios, (0, max_samples - audios.shape[1]))
            audio_lens = torch.tensor(new_lens, dtype=audio_lens.dtype, device=audio_lens.device)
            audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()

        system_prompts = [cut.custom.get(self.cfg.prompt_field, self.cfg.system_prompt) for cut in cuts]

        batch_messages = get_llm_messages_for_batch(
            system_role=self.cfg.system_role,
            system_prompt=system_prompts,
            audio_tag=self.cfg.audio_tag,
            blank_token=self.cfg.blank_token,
            chunk_size=chunk_size,
            num_delay_frames=self.cfg.num_delay_frames,
            audio_durations_secs=audio_durations_secs,
            frame_length_in_secs=self.cfg.frame_length_in_secs,
            alignments=alignments,
            transcripts=text,
            words_per_group=self.cfg.words_per_group,
            chunk_step=K,
            prepend_write_token=self.cfg.prepend_write_token,
            write_token=self.cfg.write_token,
            use_flush_token=self._use_flush,
            flush_token=self.cfg.flush_token,
        )

        # Pre-computed audio chunk token IDs for this batch's fixed-chunk size
        # (``None`` for dynamic / offline modes, where audio tags vary per turn).
        audio_chunk_ids = self._audio_chunk_ids_by_size.get(chunk_size)

        all_input_ids = []
        all_target_ids = []

        for sample_idx, messages in enumerate(batch_messages):
            # Tokenize and compute assistant content mask.
            if self.cfg.compact_template:
                input_ids, assistant_mask = _tokenize_compact_with_assistant_mask(
                    messages,
                    self.tokenizer,
                    self._eoa_id,
                    self._compact_eos_id,
                    drop_blank_from_context=self._drop_blank,
                    blank_token=self.cfg.blank_token,
                    collapse_silent_audio=self._collapse_audio,
                    flush_id=self._flush_id,
                )
            else:
                input_ids, assistant_mask = _tokenize_with_assistant_mask(messages, self.tokenizer)

            # Replace each audio chunk token sequence with chunk_size AUDIO_TOKEN_IDX markers.
            # We match the full chunk (audio_tag * chunk_size) as a unit because BPE
            # may merge tokens across adjacent audio tags.
            if self._audio_token_id is not None:
                # Single-token audio tag: one id per frame, so a plain map suffices and
                # works identically for fixed, dynamic and offline chunking.
                input_ids = [AUDIO_TOKEN_IDX if t == self._audio_token_id else t for t in input_ids]
            elif audio_chunk_ids is not None:
                # Fixed chunking: single pre-computed pattern
                input_ids, assistant_mask = _replace_audio_chunks(
                    input_ids, audio_chunk_ids, chunk_size, mask=assistant_mask
                )
            else:
                # Offline (chunk_size=-1) or dynamic (chunk_size=0): variable audio tag
                # counts per user turn.  Replace each user turn's audio tags separately.
                hf_tok = self.tokenizer.tokenizer
                for msg in messages:
                    if msg["role"] != "user":
                        continue
                    n_tags = msg["content"].count(self.cfg.audio_tag)
                    if n_tags == 0:
                        continue
                    chunk_ids = hf_tok.encode(self.cfg.audio_tag * n_tags, add_special_tokens=False)
                    input_ids, assistant_mask = _replace_audio_chunks(
                        input_ids, chunk_ids, n_tags, mask=assistant_mask
                    )

            # Build targets: next-token prediction with loss only on assistant content.
            # target[i] corresponds to input[i] and holds the token at position i+1.
            # Loss is applied only where assistant_mask[i+1] is True.
            target_ids = input_ids[1:] + [IGNORE_INDEX]
            target_mask = assistant_mask[1:] + [0]
            target_ids = [tid if m else IGNORE_INDEX for tid, m in zip(target_ids, target_mask)]

            # Rung 2: with the anchors gone there is no boundary marker left in the token
            # stream, so the gate is located by counting audio frames — every chunk
            # contributes exactly ``chunk_size`` of them, so every C-th audio token is a
            # gate. Its target is ``write`` when the chunk speaks (the write token
            # follows it directly) and ``blank`` when it does not (the next token is more
            # audio, or the sequence ends).
            if self._collapse_audio and chunk_size > 0:
                n_ids = len(input_ids)
                audio_seen = 0
                for i, tid in enumerate(input_ids):
                    if tid != AUDIO_TOKEN_IDX:
                        continue
                    audio_seen += 1
                    if audio_seen % chunk_size:
                        continue
                    nxt = input_ids[i + 1] if i + 1 < n_ids else None
                    # Silent when nothing follows but more audio (or the sequence ends);
                    # otherwise whatever token actually starts the emission is the target.
                    # With ``prepend_write_token`` that is the write token, giving a binary
                    # gate; without it, it is the first text token. Both are "not blank",
                    # which is all the inference-side gate tests.
                    #
                    # The flush token is force-fed scaffold, never predicted: a boundary
                    # followed by flush emitted nothing, so its target is blank. Making
                    # flush the target would train the model to announce end-of-audio,
                    # which is precisely the unobservable this knob exists to remove.
                    if nxt is None or nxt == AUDIO_TOKEN_IDX or (self._flush_id is not None and nxt == self._flush_id):
                        target_ids[i] = self.blank_id
                    else:
                        target_ids[i] = nxt

            # Rung 2 + flush: the marker has no audio frame of its own, so it is its own
            # gate -- blank when the turn emitted nothing, otherwise whatever token
            # starts the emission. Identical rule to the audio-frame gate above.
            if self._collapse_audio and self._flush_id is not None and chunk_size > 0:
                n_ids = len(input_ids)
                for i, tid in enumerate(input_ids):
                    if tid != self._flush_id:
                        continue
                    nxt = input_ids[i + 1] if i + 1 < n_ids else None
                    target_ids[i] = self.blank_id if (nxt is None or nxt == AUDIO_TOKEN_IDX) else nxt

            # Rung 1: the blank is gone from the input, so re-attach it as the target at
            # the gate. A gate anchor that ends a SILENT chunk is followed by audio (the
            # next chunk) or by nothing (end of sequence); a speaking chunk's anchor is
            # followed by the write token or text, so the two cannot be confused.
            if self._drop_blank and not self._collapse_audio and chunk_size > 0:
                n_ids = len(input_ids)
                for i, tid in enumerate(input_ids):
                    if tid != self._eoa_id:
                        continue
                    nxt = input_ids[i + 1] if i + 1 < n_ids else None
                    # Flush is force-fed scaffold (see the rung-2 branch above): an
                    # anchor followed by flush emitted nothing, so its target is blank.
                    if nxt is None or nxt == AUDIO_TOKEN_IDX or (self._flush_id is not None and nxt == self._flush_id):
                        target_ids[i] = self.blank_id

            # Dynamic chunking: train the model to predict at audio positions.
            # Non-final audio frames → target = blank_id ("need more audio")
            # Final audio frame (before user footer) → target = user_footer first token ("ready")
            if chunk_size == 0:
                user_footer_id = self._user_footer_first_id
                for i in range(len(input_ids)):
                    if input_ids[i] != AUDIO_TOKEN_IDX:
                        continue
                    next_is_audio = i + 1 < len(input_ids) and input_ids[i + 1] == AUDIO_TOKEN_IDX
                    target_ids[i] = self.blank_id if next_is_audio else user_footer_id

            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_target_ids.append(torch.tensor(target_ids, dtype=torch.long))

        pad_id = resolve_pad_id(self.tokenizer)
        if chunk_size >= 0:  # fixed chunking or dynamic chunking: right-pad
            input_tokens = right_collate_vectors(all_input_ids, padding_value=pad_id)
            target_tokens = right_collate_vectors(all_target_ids, padding_value=IGNORE_INDEX)
            input_token_lens = torch.tensor([len(ids) for ids in all_input_ids], dtype=torch.long)
            target_token_lens = torch.tensor([len(ids) for ids in all_target_ids], dtype=torch.long)
        else:  # offline mode: left-pad
            input_tokens = left_collate_vectors(all_input_ids, padding_value=pad_id)
            target_tokens = left_collate_vectors(all_target_ids, padding_value=IGNORE_INDEX)
            # length is the same size as input_tokens.shape[1] since they're left-padded
            input_token_lens = torch.tensor(
                [input_tokens.shape[1] for _ in range(len(all_input_ids))], dtype=torch.long
            )
            target_token_lens = torch.tensor(
                [target_tokens.shape[1] for _ in range(len(all_target_ids))], dtype=torch.long
            )

        return StreamingSTTBatch(
            audios=audios,
            audio_lens=audio_lens,
            input_tokens=input_tokens,
            input_token_lens=input_token_lens,
            target_tokens=target_tokens,
            target_token_lens=target_token_lens,
            text=text,
            chunk_size=chunk_size,
            chunk_step=K,
        )


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token-ID sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _common_suffix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common suffix of two token-ID sequences."""
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        n += 1
    return n


def _assert_partitions(
    a_ids: list[int],
    sys_len: int,
    head: int,
    tail: int,
    hf_tok,
    sys_probe: str,
    user_header_ids: list[int],
) -> None:
    """Fail loudly when the prefix/suffix diff did not isolate the user content.

    The longest common prefix/suffix against the emptied-content render is only
    the right answer if the two renders differ *exactly* in the content. Three
    ways that can break, all of which would otherwise yield silently truncated or
    overlapping spans rather than an error:

    - the two renders are identical (empty or stripped-away content), so the
      prefix runs to the end and the suffix to the start;
    - the header and footer spans overlap;
    - the template folds the system message into the first user turn, so the
      system-only render carries no tokens to exclude and the probe system prompt
      itself lands inside ``user_header``. (No backbone in use does this, but the
      ``A[:len(S)] == S`` guard passes vacuously when it happens, so it needs its
      own check.)
    """
    name = getattr(hf_tok, "name_or_path", type(hf_tok).__name__)
    content_start, content_end = head, len(a_ids) - tail
    if content_start >= content_end:
        raise ValueError(
            f"Chat template for {name!r}: probe content did not survive rendering "
            f"(user span is empty or the header and footer overlap). Pick a probe_content that the "
            f"template preserves verbatim."
        )
    if content_start < sys_len:
        raise ValueError(
            f"Chat template for {name!r}: the user turn overlaps the system block; cannot derive " f"per-turn spans."
        )
    if sys_probe in hf_tok.decode(user_header_ids):
        raise ValueError(
            f"Chat template for {name!r} folds the system message into the first user turn, so the "
            f"derived user_header would carry the probe system prompt. parse_chat_template_ids "
            f"cannot derive per-turn spans for this template."
        )


def _assert_prefix(longer: list[int], shorter: list[int], hf_tok, what: str) -> None:
    """Fail loudly when a chat template breaks the append-only prefix relation.

    ``parse_chat_template_ids`` derives turn spans by appending messages and
    diffing renders, which is only valid if appending a message never rewrites
    the tokens already emitted for earlier ones.  Every template in use satisfies
    this (their cross-turn state keys off the last *user* index, which appending
    an assistant turn does not move), but a template that emitted a trailing
    element only when the conversation ends on a user turn would not — and would
    otherwise yield silently wrong spans rather than an error.
    """
    if longer[: len(shorter)] != shorter:
        name = getattr(hf_tok, "name_or_path", type(hf_tok).__name__)
        raise ValueError(
            f"Chat template for {name!r} is not append-only: {what}. "
            f"parse_chat_template_ids cannot derive turn spans for this template."
        )


def _content_for_words(
    indices: list[int],
    alignments: List[WordAlignment],
    word_spans: Optional[list],
    transcript: Optional[str],
) -> str:
    """Render a contiguous run of aligned words as assistant content.

    Prefers slicing the original transcript by character span, which preserves the
    source punctuation and spacing exactly; falls back to joining the alignment
    texts when spans are unavailable.
    """
    if word_spans and transcript:
        first_span = word_spans[indices[0]]
        last_span = word_spans[indices[-1]]
        if first_span is not None and last_span is not None:
            return transcript[first_span[0] : last_span[1]]
    return " ".join(alignments[i].text for i in indices)
