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
"""Interleaving algorithm for audio+text sequence construction in StreamingSALM."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class WordAlignment:
    """Word-level alignment result from forced aligner."""

    text: str
    start_time: float
    end_time: float


def build_interleaved_sequence(
    audio_embs: Tensor,
    alignment: list[WordAlignment],
    latency: int,
    blank_id: int,
    tokenizer,
    frame_shift: float,
    pad_embed: Tensor | None = None,
) -> tuple[list[Tensor | None], list[int], list[int]]:
    """
    Build the interleaved audio+text sequence structure and labels.

    Text positions are marked with ``None`` placeholders in *input_parts* and
    the corresponding token IDs are collected in *text_token_ids*.  The caller
    is responsible for embedding all text token IDs in a single batched call
    and filling the ``None`` slots afterward.

    Algorithm:
    - Process audio frames sequentially
    - After each audio frame, check if a text token is "ready" based on alignment + latency
    - When a text token is ready, its label goes at the audio frame position,
      and a None marker is inserted as input (to be filled with the embedding later)
    - When no text is ready, label = blank
    - After all audio frames, remaining ready tokens are emitted during a
      "flush" phase using ``pad_embed`` as the audio input (matching inference).

    Args:
        audio_embs: (T, H) per-frame audio embeddings
        alignment: word-level forced alignment results
        latency: K value (number of audio frames from word start before emission)
        blank_id: token ID for blank/no-emission
        tokenizer: tokenizer with text_to_ids(text) -> list[int]
        frame_shift: duration of one audio frame in seconds
        pad_embed: (H,) padding audio embedding for flush frames.
            When provided, tokens with ``ready_frame >= T`` are emitted
            using this embedding instead of being silently dropped.

    Returns:
        input_parts: list of (H,) tensors at audio positions, None at text positions
        label_parts: list of int token IDs — blank or text token at each position
        text_token_ids: ordered list of text token IDs (one per None in input_parts)
    """
    T = len(audio_embs)

    # Prepare ordered list of (ready_frame, token_id) from alignment
    ready_tokens: list[tuple[int, int]] = []
    for word_idx, word in enumerate(alignment):
        start_frame = round(word.start_time / frame_shift)
        # Prepend space for non-first words so BPE tokenizer produces
        # space-prefixed subword tokens (e.g. " the" not "the").
        word_text = word.text if word_idx == 0 else " " + word.text
        subword_ids = tokenizer.text_to_ids(word_text)
        for j, tok_id in enumerate(subword_ids):
            ready_frame = start_frame + latency - 1 + j
            ready_tokens.append((ready_frame, tok_id))
    ready_tokens.sort(key=lambda x: x[0])

    ready_iter = iter(ready_tokens)
    next_ready = next(ready_iter, None)

    input_parts: list[Tensor | None] = []
    label_parts: list[int] = []
    text_token_ids: list[int] = []

    for f in range(T):
        # Add audio frame
        input_parts.append(audio_embs[f])

        # Check if text token is ready
        if next_ready is not None and next_ready[0] <= f:
            tok_id = next_ready[1]
            # Label at audio position = text token
            label_parts.append(tok_id)
            # Placeholder for text embedding — filled by caller
            input_parts.append(None)
            text_token_ids.append(tok_id)
            # Label at text position = blank (already emitted)
            label_parts.append(blank_id)
            next_ready = next(ready_iter, None)
        else:
            label_parts.append(blank_id)

    # Flush: emit remaining tokens that weren't ready before the audio ended.
    # This matches the flush phase in generate() which sends exactly ``latency``
    # padding audio frames, so we cap at ``latency`` tokens to keep consistency.
    if pad_embed is not None:
        flush_emitted = 0
        while next_ready is not None and flush_emitted < latency:
            tok_id = next_ready[1]
            input_parts.append(pad_embed)
            label_parts.append(tok_id)
            input_parts.append(None)
            text_token_ids.append(tok_id)
            label_parts.append(blank_id)
            next_ready = next(ready_iter, None)
            flush_emitted += 1

    return input_parts, label_parts, text_token_ids
