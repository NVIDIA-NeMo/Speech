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
"""Context biasing strategies for StreamingSALM training."""

from __future__ import annotations

import random

from torch import Tensor

from nemo.collections.speechlm2.parts.interleaving import WordAlignment


def maybe_apply_context_biasing(
    audio_embs: Tensor,
    alignment: list[WordAlignment],
    transcript: str,
    T: int,
    context_biasing_prob: float,
    frame_shift: float,
) -> tuple[str | None, Tensor, list[WordAlignment], int]:
    """
    Optionally apply context biasing.

    Two strategies (50/50 split):
    1. Random word context: pick 1-3 random words from transcript
    2. Prefix context: split at a word boundary, use first half as context,
       truncate audio to second half only

    Args:
        audio_embs: (T, H) audio frame embeddings
        alignment: word-level forced alignment results
        transcript: full transcript text
        T: number of valid audio frames
        context_biasing_prob: probability of applying biasing (0-1)
        frame_shift: duration of one audio frame in seconds

    Returns:
        context_text: the biasing context string, or None
        audio_embs: possibly truncated audio embeddings
        alignment: possibly adjusted alignment
        T: possibly adjusted frame count
    """
    if random.random() >= context_biasing_prob or not alignment:
        return None, audio_embs, alignment, T

    if random.random() < 0.5:
        # Strategy 1: Random word context
        words = [w.text for w in alignment]
        n_words = min(random.randint(1, 3), len(words))
        selected = random.sample(words, n_words)
        context_text = " ".join(selected)
        return context_text, audio_embs, alignment, T
    else:
        # Strategy 2: Prefix context with audio truncation
        if len(alignment) < 2:
            return None, audio_embs, alignment, T
        split_idx = random.randint(1, len(alignment) - 1)
        prefix_words = alignment[:split_idx]
        suffix_words = alignment[split_idx:]
        context_text = " ".join(w.text for w in prefix_words)

        # Truncate audio to start at the split point
        split_time = suffix_words[0].start_time
        split_frame = round(split_time / frame_shift)
        if split_frame >= T - 1:
            return None, audio_embs, alignment, T
        audio_embs = audio_embs[split_frame:]
        T = len(audio_embs)

        # Adjust alignment times relative to new start
        adjusted_alignment = [
            WordAlignment(
                text=w.text,
                start_time=w.start_time - split_time,
                end_time=w.end_time - split_time,
            )
            for w in suffix_words
        ]
        return context_text, audio_embs, adjusted_alignment, T
