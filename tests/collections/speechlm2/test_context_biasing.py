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
"""Test context biasing strategies for StreamingSALM."""

import random

import pytest
import torch

from nemo.collections.speechlm2.parts.context_biasing import maybe_apply_context_biasing
from nemo.collections.speechlm2.parts.interleaving import WordAlignment


@pytest.fixture
def sample_alignment():
    """5-word alignment spanning 0.0s to 2.0s."""
    return [
        WordAlignment(text="the", start_time=0.0, end_time=0.24),
        WordAlignment(text="quick", start_time=0.32, end_time=0.72),
        WordAlignment(text="brown", start_time=0.80, end_time=1.20),
        WordAlignment(text="fox", start_time=1.28, end_time=1.60),
        WordAlignment(text="jumps", start_time=1.68, end_time=2.00),
    ]


@pytest.fixture
def sample_audio_embs():
    """25 audio frames (2.0s at 12.5 Hz = 80ms frame shift)."""
    return torch.randn(25, 16)


FRAME_SHIFT = 0.08


class TestContextBiasing:
    def test_no_biasing_when_prob_zero(self, sample_audio_embs, sample_alignment):
        """context_biasing_prob=0 should never apply biasing."""
        for _ in range(50):
            context_text, audio_embs, alignment, T = maybe_apply_context_biasing(
                audio_embs=sample_audio_embs,
                alignment=sample_alignment,
                transcript="the quick brown fox jumps",
                T=len(sample_audio_embs),
                context_biasing_prob=0.0,
                frame_shift=FRAME_SHIFT,
            )
            assert context_text is None
            assert torch.equal(audio_embs, sample_audio_embs)
            assert alignment is sample_alignment
            assert T == len(sample_audio_embs)

    def test_always_biasing_when_prob_one(self, sample_audio_embs, sample_alignment):
        """context_biasing_prob=1 should always apply biasing."""
        num_biased = 0
        for _ in range(50):
            context_text, audio_embs, alignment, T = maybe_apply_context_biasing(
                audio_embs=sample_audio_embs,
                alignment=sample_alignment,
                transcript="the quick brown fox jumps",
                T=len(sample_audio_embs),
                context_biasing_prob=1.0,
                frame_shift=FRAME_SHIFT,
            )
            if context_text is not None:
                num_biased += 1
        assert num_biased == 50

    def test_random_word_strategy_picks_from_transcript(self, sample_audio_embs, sample_alignment):
        """Strategy 1: selected words should be in the transcript."""
        random.seed(42)
        # Force strategy 1 by mocking: seed so random.random() < 0.5 after first check
        # Run many times and check all results contain valid words
        valid_words = {"the", "quick", "brown", "fox", "jumps"}
        for _ in range(100):
            context_text, audio_embs, alignment, T = maybe_apply_context_biasing(
                audio_embs=sample_audio_embs,
                alignment=sample_alignment,
                transcript="the quick brown fox jumps",
                T=len(sample_audio_embs),
                context_biasing_prob=1.0,
                frame_shift=FRAME_SHIFT,
            )
            if context_text is not None and torch.equal(audio_embs, sample_audio_embs):
                # This is strategy 1 (audio unchanged)
                context_words = set(context_text.split())
                assert context_words.issubset(valid_words), (
                    f"Context words {context_words} not in transcript words {valid_words}"
                )

    def test_random_word_strategy_picks_1_to_3_words(self, sample_audio_embs, sample_alignment):
        """Strategy 1: should pick 1-3 words."""
        word_counts = set()
        for seed in range(200):
            random.seed(seed)
            context_text, audio_embs, alignment, T = maybe_apply_context_biasing(
                audio_embs=sample_audio_embs,
                alignment=sample_alignment,
                transcript="the quick brown fox jumps",
                T=len(sample_audio_embs),
                context_biasing_prob=1.0,
                frame_shift=FRAME_SHIFT,
            )
            if context_text is not None and torch.equal(audio_embs, sample_audio_embs):
                # Strategy 1: audio unchanged
                n_words = len(context_text.split())
                word_counts.add(n_words)
                assert 1 <= n_words <= 3

    def test_prefix_strategy_truncates_audio(self, sample_audio_embs, sample_alignment):
        """Strategy 2: audio_embs should be shortened after split."""
        found_truncated = False
        for seed in range(200):
            random.seed(seed)
            context_text, audio_embs, alignment, T = maybe_apply_context_biasing(
                audio_embs=sample_audio_embs,
                alignment=sample_alignment,
                transcript="the quick brown fox jumps",
                T=len(sample_audio_embs),
                context_biasing_prob=1.0,
                frame_shift=FRAME_SHIFT,
            )
            if context_text is not None and not torch.equal(audio_embs, sample_audio_embs):
                # Strategy 2: audio was truncated
                found_truncated = True
                assert len(audio_embs) < len(sample_audio_embs)
                assert T == len(audio_embs)
        assert found_truncated, "Never observed prefix truncation strategy"

    def test_prefix_strategy_adjusts_alignment_times(self, sample_audio_embs, sample_alignment):
        """Strategy 2: alignment times should be relative to new audio start."""
        for seed in range(200):
            random.seed(seed)
            context_text, audio_embs, alignment, T = maybe_apply_context_biasing(
                audio_embs=sample_audio_embs,
                alignment=sample_alignment,
                transcript="the quick brown fox jumps",
                T=len(sample_audio_embs),
                context_biasing_prob=1.0,
                frame_shift=FRAME_SHIFT,
            )
            if context_text is not None and not torch.equal(audio_embs, sample_audio_embs):
                # Strategy 2: times should start near 0
                assert alignment[0].start_time >= 0.0
                assert alignment[0].start_time < 0.1  # should be close to 0

    def test_prefix_strategy_context_matches_prefix_words(self, sample_audio_embs, sample_alignment):
        """Strategy 2: context text should contain the prefix words."""
        all_words = [w.text for w in sample_alignment]
        for seed in range(200):
            random.seed(seed)
            context_text, audio_embs, alignment, T = maybe_apply_context_biasing(
                audio_embs=sample_audio_embs,
                alignment=sample_alignment,
                transcript="the quick brown fox jumps",
                T=len(sample_audio_embs),
                context_biasing_prob=1.0,
                frame_shift=FRAME_SHIFT,
            )
            if context_text is not None and not torch.equal(audio_embs, sample_audio_embs):
                # Strategy 2: context should be a prefix of the word list
                context_words = context_text.split()
                assert context_words == all_words[: len(context_words)]
                # Remaining alignment should cover the suffix words
                alignment_words = [w.text for w in alignment]
                assert alignment_words == all_words[len(context_words) :]

    def test_prefix_strategy_skipped_for_single_word(self):
        """Strategy 2: cannot split single-word alignment, falls back to None."""
        single_alignment = [WordAlignment(text="hello", start_time=0.0, end_time=0.4)]
        audio_embs = torch.randn(5, 16)
        # With seed that would pick strategy 2 and single word, should return None
        none_count = 0
        for seed in range(200):
            random.seed(seed)
            context_text, out_audio, out_align, T = maybe_apply_context_biasing(
                audio_embs=audio_embs,
                alignment=single_alignment,
                transcript="hello",
                T=5,
                context_biasing_prob=1.0,
                frame_shift=FRAME_SHIFT,
            )
            # When strategy 2 is selected but only 1 word, should return None
            # Strategy 1 can still return context
            if context_text is None:
                none_count += 1
        # Some seeds should hit strategy 2 and fail gracefully
        assert none_count > 0

    def test_empty_alignment_returns_none(self):
        """No alignment -> no biasing possible."""
        audio_embs = torch.randn(5, 16)
        context_text, out_audio, out_align, T = maybe_apply_context_biasing(
            audio_embs=audio_embs,
            alignment=[],
            transcript="hello world",
            T=5,
            context_biasing_prob=1.0,
            frame_shift=FRAME_SHIFT,
        )
        assert context_text is None
        assert torch.equal(out_audio, audio_embs)
        assert T == 5
