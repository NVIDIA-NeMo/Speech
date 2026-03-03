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
"""Test the interleaving algorithm for audio+text sequence construction."""

import pytest
import torch

from nemo.collections.speechlm2.parts.interleaving import WordAlignment, build_interleaved_sequence


class MockTokenizer:
    """Simple mock tokenizer that maps each character to its ord() value.
    Strips leading spaces to keep existing tests unchanged (they predate the
    space-prefix fix in build_interleaved_sequence)."""

    def text_to_ids(self, text):
        return [ord(c) for c in text.lstrip(" ")]


class SpacePrefixTokenizer:
    """
    Mock tokenizer that mimics BPE space-prefix behavior.

    A leading space is encoded as a separate token (ID=32 for ' '),
    so " hello" -> [32, ...] while "hello" -> [...] without the space token.
    This simulates the real behavior of Qwen/GPT-style tokenizers where
    word boundaries are encoded via leading spaces in subword tokens.
    """

    def __init__(self):
        # Vocab: each character maps to its ord(), space = 32
        pass

    def text_to_ids(self, text):
        return [ord(c) for c in text]


def mock_embed(token_id, H=4):
    """Simple mock embed function: returns a constant vector."""
    return torch.full((H,), float(token_id))


def _fill_none_slots(input_parts, text_token_ids, H=4):
    """Fill None placeholders in input_parts with mock embeddings."""
    text_iter = iter(text_token_ids)
    for j, part in enumerate(input_parts):
        if part is None:
            input_parts[j] = mock_embed(next(text_iter), H)


@pytest.fixture
def blank_id():
    return 0


@pytest.fixture
def frame_shift():
    return 0.08  # 80ms


class TestBuildInterleavedSequence:
    def test_example_a_from_plan(self, blank_id, frame_shift):
        """
        Example A from the plan:
        5 frames, 2 text tokens, K=2
        Alignment: "Hel" starts frame 0, "lo" starts frame 2
        Expected input:  [A0, A1, T0, A2, A3, T1, A4]
        Expected labels: [B,  T0, B,  B,  T1, B,  B ]
        """
        H = 4
        T = 5
        audio_embs = torch.arange(T * H).reshape(T, H).float()
        alignment = [
            WordAlignment(text="x", start_time=0.0, end_time=0.16),  # frames 0-1
            WordAlignment(text="y", start_time=0.16, end_time=0.32),  # frames 2-3
        ]
        K = 2
        T0_id = ord("x")  # mock tokenizer
        T1_id = ord("y")

        input_parts, label_parts, text_token_ids = build_interleaved_sequence(
            audio_embs=audio_embs,
            alignment=alignment,
            latency=K,
            blank_id=blank_id,
            tokenizer=MockTokenizer(),
            frame_shift=frame_shift,
        )

        # Input should be: [A0, A1, None, A2, A3, None, A4] = 7 items
        assert len(input_parts) == 7
        assert text_token_ids == [T0_id, T1_id]
        # Labels should be: [B, T0, B, B, T1, B, B] = 7 items
        assert label_parts == [blank_id, T0_id, blank_id, blank_id, T1_id, blank_id, blank_id]

    def test_example_b_from_plan(self, blank_id, frame_shift):
        """
        Example B from the plan:
        5 frames, 2 text tokens, K=4
        Alignment: word starts frame 0
        Expected input:  [A0, A1, A2, A3, T0, A4, T1]
        Expected labels: [B,  B,  B,  T0, B,  T1, B ]
        """
        H = 4
        T = 5
        audio_embs = torch.arange(T * H).reshape(T, H).float()
        alignment = [
            WordAlignment(text="xy", start_time=0.0, end_time=0.4),
        ]
        K = 4
        T0_id = ord("x")
        T1_id = ord("y")

        input_parts, label_parts, text_token_ids = build_interleaved_sequence(
            audio_embs=audio_embs,
            alignment=alignment,
            latency=K,
            blank_id=blank_id,
            tokenizer=MockTokenizer(),
            frame_shift=frame_shift,
        )

        assert len(input_parts) == 7
        assert text_token_ids == [T0_id, T1_id]
        assert label_parts == [blank_id, blank_id, blank_id, T0_id, blank_id, T1_id, blank_id]

    def test_all_blanks_when_latency_exceeds_frames(self, blank_id, frame_shift):
        """If latency K > T, all labels should be blank (text never becomes ready)."""
        T = 3
        H = 4
        audio_embs = torch.randn(T, H)
        alignment = [WordAlignment(text="a", start_time=0.0, end_time=0.08)]
        K = 10

        input_parts, label_parts, text_token_ids = build_interleaved_sequence(
            audio_embs=audio_embs,
            alignment=alignment,
            latency=K,
            blank_id=blank_id,
            tokenizer=MockTokenizer(),
            frame_shift=frame_shift,
        )

        # No text tokens inserted, all labels are blank
        assert len(input_parts) == T  # only audio frames
        assert label_parts == [blank_id] * T
        assert text_token_ids == []

    def test_no_alignment_all_blanks(self, blank_id, frame_shift):
        """Empty alignment -> all blanks, input = audio only."""
        T = 4
        H = 4
        audio_embs = torch.randn(T, H)
        input_parts, label_parts, text_token_ids = build_interleaved_sequence(
            audio_embs=audio_embs,
            alignment=[],
            latency=1,
            blank_id=blank_id,
            tokenizer=MockTokenizer(),
            frame_shift=frame_shift,
        )
        assert len(input_parts) == T
        assert label_parts == [blank_id] * T
        assert text_token_ids == []

    def test_k1_minimum_latency(self, blank_id, frame_shift):
        """K=1: text emitted as early as possible (at the word's start frame)."""
        T = 4
        H = 4
        audio_embs = torch.randn(T, H)
        # Word starts at frame 0, single token
        alignment = [WordAlignment(text="a", start_time=0.0, end_time=0.16)]
        K = 1
        tok_id = ord("a")

        input_parts, label_parts, text_token_ids = build_interleaved_sequence(
            audio_embs=audio_embs,
            alignment=alignment,
            latency=K,
            blank_id=blank_id,
            tokenizer=MockTokenizer(),
            frame_shift=frame_shift,
        )

        # K=1: ready at frame 0+1-1=0, so label[0] = tok_id
        assert label_parts[0] == tok_id
        assert label_parts[1] == blank_id  # after fed-back text
        # Input should have text token inserted after first audio frame
        assert len(input_parts) == T + 1  # 4 audio + 1 text
        assert text_token_ids == [tok_id]

    def test_multiple_words_correct_ordering(self, blank_id, frame_shift):
        """Multiple words should be emitted in order based on alignment times."""
        T = 10
        H = 4
        audio_embs = torch.randn(T, H)
        alignment = [
            WordAlignment(text="a", start_time=0.0, end_time=0.16),
            WordAlignment(text="b", start_time=0.24, end_time=0.40),
            WordAlignment(text="c", start_time=0.56, end_time=0.72),
        ]
        K = 2

        input_parts, label_parts, text_token_ids = build_interleaved_sequence(
            audio_embs=audio_embs,
            alignment=alignment,
            latency=K,
            blank_id=blank_id,
            tokenizer=MockTokenizer(),
            frame_shift=frame_shift,
        )

        # Extract non-blank labels in order
        text_labels = [l for l in label_parts if l != blank_id]
        assert text_labels == [ord("a"), ord("b"), ord("c")]
        assert text_token_ids == [ord("a"), ord("b"), ord("c")]

    def test_fed_back_text_token_in_input(self, blank_id, frame_shift):
        """When text is predicted, verify the None slot and text_token_ids are correct."""
        T = 3
        H = 4
        audio_embs = torch.randn(T, H)
        alignment = [WordAlignment(text="a", start_time=0.0, end_time=0.16)]
        K = 1
        tok_id = ord("a")

        input_parts, label_parts, text_token_ids = build_interleaved_sequence(
            audio_embs=audio_embs,
            alignment=alignment,
            latency=K,
            blank_id=blank_id,
            tokenizer=MockTokenizer(),
            frame_shift=frame_shift,
        )

        # input_parts[0] = audio frame 0
        # input_parts[1] = None (placeholder for text token "a")
        # input_parts[2] = audio frame 1
        assert input_parts[1] is None
        assert text_token_ids == [tok_id]

        # Fill None slots and verify the embedding matches
        _fill_none_slots(input_parts, text_token_ids, H)
        expected_text_emb = mock_embed(tok_id, H)
        assert torch.allclose(input_parts[1], expected_text_emb)

    def test_multiword_subword_overflow(self, blank_id, frame_shift):
        """Word with more subword tokens than remaining frames: tokens are clipped (no pad_embed)."""
        T = 3
        H = 4
        audio_embs = torch.randn(T, H)
        # "abcde" -> 5 tokens, but only 3 frames available with K=1
        alignment = [WordAlignment(text="abcde", start_time=0.0, end_time=0.24)]
        K = 1

        input_parts, label_parts, text_token_ids = build_interleaved_sequence(
            audio_embs=audio_embs,
            alignment=alignment,
            latency=K,
            blank_id=blank_id,
            tokenizer=MockTokenizer(),
            frame_shift=frame_shift,
        )

        text_labels = [l for l in label_parts if l != blank_id]
        # Should emit as many as fit (up to 3 tokens with T=3)
        assert len(text_labels) <= T
        # First tokens should be in order
        assert text_labels == [ord("a"), ord("b"), ord("c")]
        assert text_token_ids == [ord("a"), ord("b"), ord("c")]

    def test_flush_basic(self, blank_id, frame_shift):
        """With pad_embed, overflow tokens are flushed after audio ends."""
        T = 3
        H = 4
        audio_embs = torch.randn(T, H)
        pad_embed = torch.zeros(H)
        # K=4 with word at frame 0: ready_frame = 0+4-1+j = 3+j
        # For "ab" -> ready at frames 3, 4 — both beyond T=3
        alignment = [WordAlignment(text="ab", start_time=0.0, end_time=0.16)]
        K = 4

        input_parts, label_parts, text_token_ids = build_interleaved_sequence(
            audio_embs=audio_embs,
            alignment=alignment,
            latency=K,
            blank_id=blank_id,
            tokenizer=MockTokenizer(),
            frame_shift=frame_shift,
            pad_embed=pad_embed,
        )

        # Audio phase: 3 frames, all blank (no token ready yet)
        # Flush phase: 2 tokens flushed (both fit within latency=4)
        # Each flush token takes 2 positions: [pad_embed, text_placeholder]
        assert text_token_ids == [ord("a"), ord("b")]
        text_labels = [l for l in label_parts if l != blank_id]
        assert text_labels == [ord("a"), ord("b")]
        # Total: 3 audio + 2*(pad + text) = 7
        assert len(input_parts) == 7
        assert len(label_parts) == 7

    def test_flush_limited_by_latency(self, blank_id, frame_shift):
        """Flush emits at most `latency` tokens to match inference behavior."""
        T = 2
        H = 4
        audio_embs = torch.randn(T, H)
        pad_embed = torch.zeros(H)
        # "abcde" -> 5 tokens with K=1: ready at frames 0,1,2,3,4
        # Audio T=2 can emit tokens at frames 0,1 (2 tokens)
        # Remaining: 3 tokens overflow, but flush limited to K=1
        alignment = [WordAlignment(text="abcde", start_time=0.0, end_time=0.16)]
        K = 1

        input_parts, label_parts, text_token_ids = build_interleaved_sequence(
            audio_embs=audio_embs,
            alignment=alignment,
            latency=K,
            blank_id=blank_id,
            tokenizer=MockTokenizer(),
            frame_shift=frame_shift,
            pad_embed=pad_embed,
        )

        # Audio: frame 0 emits 'a', frame 1 emits 'b'
        # Flush: K=1 so only 1 more token ('c') flushed
        # 'd' and 'e' are dropped
        assert text_token_ids == [ord("a"), ord("b"), ord("c")]
        text_labels = [l for l in label_parts if l != blank_id]
        assert text_labels == [ord("a"), ord("b"), ord("c")]

    def test_flush_preserves_input_label_consistency(self, blank_id, frame_shift):
        """len(input_parts) == len(label_parts) with pad_embed across configurations."""
        H = 4
        pad_embed = torch.zeros(H)
        for T in range(1, 8):
            for K in range(1, 6):
                audio_embs = torch.randn(T, H)
                alignment = [WordAlignment(text="ab", start_time=0.0, end_time=0.16)]
                input_parts, label_parts, text_token_ids = build_interleaved_sequence(
                    audio_embs=audio_embs,
                    alignment=alignment,
                    latency=K,
                    blank_id=blank_id,
                    tokenizer=MockTokenizer(),
                    frame_shift=frame_shift,
                    pad_embed=pad_embed,
                )
                assert len(input_parts) == len(label_parts), (
                    f"Mismatch at T={T}, K={K}: {len(input_parts)} vs {len(label_parts)}"
                )
                none_count = sum(1 for p in input_parts if p is None)
                assert none_count == len(text_token_ids), (
                    f"None count mismatch at T={T}, K={K}: {none_count} vs {len(text_token_ids)}"
                )

    def test_input_and_label_length_consistency(self, blank_id, frame_shift):
        """len(input_parts) == len(label_parts) always."""
        for T in range(1, 8):
            for K in range(1, 6):
                audio_embs = torch.randn(T, 4)
                alignment = [WordAlignment(text="ab", start_time=0.0, end_time=0.16)]
                input_parts, label_parts, text_token_ids = build_interleaved_sequence(
                    audio_embs=audio_embs,
                    alignment=alignment,
                    latency=K,
                    blank_id=blank_id,
                    tokenizer=MockTokenizer(),
                    frame_shift=frame_shift,
                )
                assert len(input_parts) == len(label_parts), (
                    f"Mismatch at T={T}, K={K}: {len(input_parts)} vs {len(label_parts)}"
                )
                # Number of None slots should match text_token_ids
                none_count = sum(1 for p in input_parts if p is None)
                assert none_count == len(text_token_ids), (
                    f"None count mismatch at T={T}, K={K}: {none_count} vs {len(text_token_ids)}"
                )

    def test_non_first_words_get_space_prefix(self, blank_id, frame_shift):
        """
        Regression test: non-first words must be tokenized with a leading space
        so BPE tokenizers produce space-prefixed subword IDs.

        Without the fix, each word was tokenized in isolation (no leading space),
        causing the model to learn tokens without spaces and produce concatenated
        output like "hisambitionwas" instead of "his ambition was".
        """
        T = 20
        H = 4
        audio_embs = torch.randn(T, H)
        alignment = [
            WordAlignment(text="his", start_time=0.0, end_time=0.16),
            WordAlignment(text="cat", start_time=0.24, end_time=0.40),
            WordAlignment(text="sat", start_time=0.56, end_time=0.72),
        ]
        K = 2

        # SpacePrefixTokenizer uses ord() per char, so " cat" -> [32, 99, 97, 116]
        # while "cat" -> [99, 97, 116]. The space char (ord 32) is the marker.
        input_parts, label_parts, text_token_ids = build_interleaved_sequence(
            audio_embs=audio_embs,
            alignment=alignment,
            latency=K,
            blank_id=blank_id,
            tokenizer=SpacePrefixTokenizer(),
            frame_shift=frame_shift,
        )

        text_labels = [l for l in label_parts if l != blank_id]

        # First word "his" should NOT have a leading space
        assert text_labels[0] == ord("h")
        assert text_labels[1] == ord("i")
        assert text_labels[2] == ord("s")

        # Second word " cat" SHOULD have a leading space (ord 32)
        assert text_labels[3] == ord(" "), (
            f"Second word should start with space token (32), got {text_labels[3]}"
        )
        assert text_labels[4] == ord("c")

        # Third word " sat" SHOULD also have a leading space
        sat_start = text_labels.index(ord(" "), 4)  # find second space
        assert text_labels[sat_start] == ord(" "), (
            f"Third word should start with space token (32), got {text_labels[sat_start]}"
        )
