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

"""
Tests for StreamingSTTDataset message generation, token replacement, and
input/target construction.

The primary reference is the docstring example in get_llm_messages_for_sample:

    alignments = [
        WordAlignment(text="Hello", start_time=0.16, end_time=0.48),
        WordAlignment(text="World", start_time=0.60, end_time=0.80),
    ]
    audio_duration = 1s, chunk_size = 2, frame_length = 0.08s, delay = 0

    → 13 frames, 7 chunks, assistant responses:
      [<blank>, <blank>, Hello, <blank>, World, <blank>, <blank>]
"""

import math
from types import SimpleNamespace

import pytest
import torch

from omegaconf import OmegaConf

from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    AUDIO_TOKEN_IDX,
    IGNORE_INDEX,
    StreamingSTTDataConfig,
    StreamingSTTDataset,
    _replace_audio_chunks,
    _tokenize_compact_with_assistant_mask,
    _tokenize_with_assistant_mask,
    apply_chat_template_ids,
    build_compact_turn_markers,
    compute_word_spans,
    decode_with_blank,
    get_llm_messages_for_batch,
    get_llm_messages_for_sample,
    parse_chat_template_ids,
    resolve_pad_id,
    right_collate_vectors,
)
from nemo.collections.speechlm2.parts.alignments import WordAlignment

# ---------------------------------------------------------------------------
# Shared constants & helpers matching the docstring example
# ---------------------------------------------------------------------------
AUDIO_TAG = "<audio>"
BLANK_TOKEN = "<blank>"
SYSTEM_ROLE = "system"
SYSTEM_PROMPT = "Transcribe the audio into text."
CHUNK_SIZE = 2
FRAME_LEN = 0.08  # seconds
DOCSTRING_ALIGNMENTS = [
    WordAlignment(text="Hello", start_time=0.16, end_time=0.48),
    WordAlignment(text="World", start_time=0.60, end_time=0.80),
]


def _make_messages(**overrides):
    """Convenience wrapper around get_llm_messages_for_sample with docstring defaults."""
    kw = dict(
        system_role=SYSTEM_ROLE,
        system_prompt=SYSTEM_PROMPT,
        audio_tag=AUDIO_TAG,
        blank_token=BLANK_TOKEN,
        chunk_size=CHUNK_SIZE,
        num_delay_frames=0,
        audio_duration_secs=1.0,
        frame_length_in_secs=FRAME_LEN,
        alignments=DOCSTRING_ALIGNMENTS,
    )
    kw.update(overrides)
    return get_llm_messages_for_sample(**kw)


# ---------------------------------------------------------------------------
# Mock tokenizer used by TestTokenPositions
# ---------------------------------------------------------------------------
class _MockHFTokenizer:
    """
    Deterministic HF tokenizer mock.

    Token layout per message:
        [HEADER_START, ROLE_ID, HEADER_END, ...content..., FOOTER, NEWLINE]

    Content encoding:
        system  → [50]
        user    → [AUDIO_TAG_ID] per <audio> tag in content
        assistant → [BLANK_ID] for "<blank>", else [200, 201, ...] per word
    """

    HEADER_START = 1
    ROLE_IDS = {"system": 10, "user": 11, "assistant": 12}
    HEADER_END = 2
    FOOTER = 3
    NEWLINE = 4
    AUDIO_TAG_ID = 100
    BLANK_ID = 101
    SYSTEM_CONTENT_ID = 50

    HEADER = [HEADER_START, None, HEADER_END]  # None → filled per role
    N_HEADER = 3
    N_FOOTER = 2

    def __init__(self, audio_tag=AUDIO_TAG, blank_token=BLANK_TOKEN):
        self.audio_tag = audio_tag
        self.blank_token = blank_token
        self.unk_token_id = 0
        self.eos_token_id = _MockHFTokenizer.FOOTER  # EOS = first footer token (like Qwen3)
        self._next_word_id = 200
        # Cache for content → token IDs mapping so encode() and apply_chat_template() agree.
        self._content_cache: dict[str, list[int]] = {}

    def _content_to_ids(self, content: str, role: str) -> list[int]:
        """Deterministic content → token IDs, consistent between encode() and apply_chat_template()."""
        if role == "user":
            return [self.AUDIO_TAG_ID] * content.count(self.audio_tag)
        if role == "assistant":
            if content == self.blank_token:
                return [self.BLANK_ID]
            # Assign stable IDs per unique content string
            if content not in self._content_cache:
                ids = []
                for _ in content.split():
                    ids.append(self._next_word_id)
                    self._next_word_id += 1
                self._content_cache[content] = ids
            return list(self._content_cache[content])
        # system
        return [self.SYSTEM_CONTENT_ID]

    def decode(self, ids, **kwargs):
        """Minimal inverse of encode(), enough for the whitespace-footer guard."""
        pieces = {
            self.HEADER_START: "<|im_start|>",
            self.HEADER_END: "\n",
            self.FOOTER: "<|im_end|>",
            self.NEWLINE: "\n",
            self.AUDIO_TAG_ID: self.audio_tag,
            self.BLANK_ID: self.blank_token,
        }
        return "".join(pieces.get(int(i), "x") for i in ids)

    def encode(self, text, add_special_tokens=False):
        if text == self.audio_tag:
            return [self.AUDIO_TAG_ID]
        if text == self.blank_token:
            return [self.BLANK_ID]
        # Footer text from the ChatML-like text template
        if text == "<|im_end|>\n":
            return [self.FOOTER, self.NEWLINE]
        # Handle repeated audio tags (chunk encoding)
        if self.audio_tag in text and text == self.audio_tag * text.count(self.audio_tag):
            return [self.AUDIO_TAG_ID] * text.count(self.audio_tag)
        # For assistant word content, use the cache
        if text in self._content_cache:
            return list(self._content_cache[text])
        # Unknown text — assign stable IDs
        ids = []
        for _ in text.split():
            ids.append(self._next_word_id)
            self._next_word_id += 1
        self._content_cache[text] = ids
        return list(ids)

    def apply_chat_template(self, messages, **kwargs):
        tokenize = kwargs.get("tokenize", True)

        if not tokenize:
            # Return ChatML-like text form for sentinel-based footer discovery.
            text = ""
            for msg in messages:
                text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            return text

        input_ids = []
        assistant_masks = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            header = [self.HEADER_START, self.ROLE_IDS[role], self.HEADER_END]
            input_ids.extend(header)
            assistant_masks.extend([0] * len(header))

            ids = self._content_to_ids(content, role)

            input_ids.extend(ids)
            assistant_masks.extend([1 if role == "assistant" else 0] * len(ids))

            footer = [self.FOOTER, self.NEWLINE]
            input_ids.extend(footer)
            assistant_masks.extend([0] * len(footer))

        return {"input_ids": input_ids, "assistant_masks": assistant_masks}


class _MockHFTokenizerMultiToken(_MockHFTokenizer):
    """Mock where <audio> tokenizes into 3 tokens: [60, 61, 62].

    Simulates BPE merging across adjacent tags: ``<audio><audio>`` tokenizes as
    [60, 61, 70, 61, 62] (5 tokens) instead of [60, 61, 62, 60, 61, 62] (6 tokens),
    because ``62`` (``>``) and ``60`` (``<``) merge into ``70`` (``><``).
    """

    MULTI_AUDIO_TAG_IDS = [60, 61, 62]
    MERGED_BOUNDARY = 70  # simulates BPE merge of > + <

    def encode(self, text, add_special_tokens=False):
        if text == self.audio_tag:
            return list(self.MULTI_AUDIO_TAG_IDS)
        # Simulate BPE merging across adjacent audio tags
        n = text.count(self.audio_tag)
        if n > 0 and text == self.audio_tag * n:
            # First tag: [60, 61, 62], subsequent: [70, 61, 62] (merged boundary)
            ids = list(self.MULTI_AUDIO_TAG_IDS)
            for _ in range(n - 1):
                ids.append(self.MERGED_BOUNDARY)
                ids.extend(self.MULTI_AUDIO_TAG_IDS[1:])  # skip first token, use merged
            return ids
        return super().encode(text, add_special_tokens=add_special_tokens)

    def _content_to_ids(self, content: str, role: str) -> list[int]:
        if role == "user":
            return self.encode(content, add_special_tokens=False)
        return super()._content_to_ids(content, role)


class _MockHFTokenizerNoGeneration(_MockHFTokenizer):
    """Mock that simulates a tokenizer without {% generation %} — returns all-zero masks."""

    def apply_chat_template(self, messages, **kwargs):
        result = super().apply_chat_template(messages, **kwargs)
        # Zero out the masks to simulate missing {% generation %} support.
        # When tokenize=False, result is a string — pass through unchanged.
        if isinstance(result, dict):
            result["assistant_masks"] = [0] * len(result["assistant_masks"])
        return result


class _MockHFTokenizerNoEOS(_MockHFTokenizerNoGeneration):
    """Mock without eos_token_id — footer trimming should fall back to full footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eos_token_id = None


class _MockNemoTokenizer:
    """Wraps a mock HF tokenizer to mimic NeMo AutoTokenizer interface."""

    def __init__(self, hf_tok):
        self.tokenizer = hf_tok


def _run_pipeline(messages, mock_hf_tok, chunk_size=CHUNK_SIZE):
    """Simulate the __getitem__ tokenization pipeline: tokenize → replace → build targets."""
    audio_chunk_ids = mock_hf_tok.encode(AUDIO_TAG * chunk_size, add_special_tokens=False)
    nemo_tok = _MockNemoTokenizer(mock_hf_tok)

    input_ids, assistant_mask = _tokenize_with_assistant_mask(messages, nemo_tok)

    input_ids, assistant_mask = _replace_audio_chunks(
        input_ids,
        audio_chunk_ids,
        chunk_size,
        mask=assistant_mask,
    )

    target_ids = input_ids[1:] + [IGNORE_INDEX]
    target_mask = assistant_mask[1:] + [0]
    target_ids = [tid if m else IGNORE_INDEX for tid, m in zip(target_ids, target_mask)]

    return input_ids, target_ids, assistant_mask


# ===========================================================================
# Tests: get_llm_messages_for_sample
# ===========================================================================
class TestGetLlmMessagesForSample:

    def test_docstring_example_structure(self):
        """Total messages: 1 system + 7*(user + assistant) = 15."""
        msgs = _make_messages()
        assert len(msgs) == 15
        assert msgs[0] == {"role": SYSTEM_ROLE, "content": SYSTEM_PROMPT}

    def test_docstring_example_roles_alternate(self):
        msgs = _make_messages()
        roles = [m["role"] for m in msgs]
        assert roles[0] == "system"
        for i in range(1, len(roles), 2):
            assert roles[i] == "user"
            assert roles[i + 1] == "assistant"

    def test_docstring_example_user_turns(self):
        msgs = _make_messages()
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 7
        assert all(m["content"] == AUDIO_TAG * CHUNK_SIZE for m in user_msgs)

    def test_docstring_example_assistant_responses(self):
        msgs = _make_messages()
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst == [BLANK_TOKEN, BLANK_TOKEN, "Hello", BLANK_TOKEN, "World", BLANK_TOKEN, BLANK_TOKEN]

    def test_num_chunks(self):
        """ceil(13 frames / 2) = 7 chunks."""
        msgs = _make_messages()
        assert sum(1 for m in msgs if m["role"] == "user") == 7

    def test_total_audio_tags_equals_chunks_times_chunk_size(self):
        msgs = _make_messages()
        user_msgs = [m for m in msgs if m["role"] == "user"]
        total = sum(m["content"].count(AUDIO_TAG) for m in user_msgs)
        assert total == len(user_msgs) * CHUNK_SIZE

    def test_delay_shifts_emission(self):
        """With delay=2, Hello (end_frame=6) → ready_frame=8 → chunk 3 (end=8)."""
        msgs = _make_messages(
            num_delay_frames=2,
            alignments=[WordAlignment(text="Hello", start_time=0.16, end_time=0.48)],
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[:3] == [BLANK_TOKEN, BLANK_TOKEN, BLANK_TOKEN]
        assert asst[3] == "Hello"

    def test_empty_alignments_all_blank(self):
        msgs = _make_messages(alignments=[])
        for m in msgs:
            if m["role"] == "assistant":
                assert m["content"] == BLANK_TOKEN

    def test_none_alignments_all_blank(self):
        msgs = _make_messages(alignments=None)
        for m in msgs:
            if m["role"] == "assistant":
                assert m["content"] == BLANK_TOKEN

    def test_multiple_words_in_same_chunk(self):
        alignments = [
            WordAlignment(text="A", start_time=0.0, end_time=0.04),
            WordAlignment(text="B", start_time=0.05, end_time=0.08),
        ]
        msgs = _make_messages(alignments=alignments, audio_duration_secs=0.16)
        # A: end_frame=round(0.04/0.08)=0, B: end_frame=1. Both ≤ chunk 0 end=2.
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[0] == "A B"

    def test_zero_duration_only_system(self):
        msgs = _make_messages(audio_duration_secs=0.0, alignments=[])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_chunk_size_1(self):
        msgs = _make_messages(chunk_size=1, alignments=[])
        # 13 frames → 13 chunks
        assert sum(1 for m in msgs if m["role"] == "user") == 13
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert all(m["content"] == AUDIO_TAG for m in user_msgs)

    def test_residual_words_appended_to_last_turn(self):
        """Words whose ready_frame exceeds the last chunk should still appear."""
        # Audio is 0.16s → 2 frames → 1 chunk (end_frame=2).
        # Word ends at 0.20s → end_frame=ceil(0.20/0.08)=3. With delay=0, ready_frame=3 > 2.
        # The word would be dropped without the residual fix.
        alignments = [WordAlignment(text="Late", start_time=0.10, end_time=0.20)]
        msgs = _make_messages(
            audio_duration_secs=0.16,
            alignments=alignments,
            num_delay_frames=0,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert "Late" in asst[-1], f"Residual word 'Late' not in last turn: {asst}"

    def test_residual_words_with_delay(self):
        """Delay pushes a word past the last chunk — it should still be emitted."""
        # Audio is 1.0s. Word ends at 0.96s → end_frame=12. With delay=2, ready_frame=14.
        # Last chunk end_frame = ceil(13/2)*2 = 14. So ready_frame=14 <= 14, it fits.
        # But if word ends at 1.0s → end_frame=13, ready_frame=15 > 14. Residual.
        alignments = [
            WordAlignment(text="Hello", start_time=0.0, end_time=0.48),
            WordAlignment(text="World", start_time=0.80, end_time=1.0),
        ]
        msgs = _make_messages(
            audio_duration_secs=1.0,
            alignments=alignments,
            num_delay_frames=2,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        all_text = " ".join(a for a in asst if a != BLANK_TOKEN)
        assert "Hello" in all_text, f"'Hello' missing: {asst}"
        assert "World" in all_text, f"'World' missing: {asst}"

    def test_residual_replaces_blank_last_turn(self):
        """If last turn was blank and there are residual words, blank is replaced."""
        # Short audio, word ends after it
        alignments = [WordAlignment(text="Overflow", start_time=0.0, end_time=0.20)]
        msgs = _make_messages(
            audio_duration_secs=0.08,  # 1 frame → 1 chunk (end_frame=2 with chunk_size=2? No, ceil(1/2)=1 chunk, end_frame=2)
            alignments=alignments,
            num_delay_frames=0,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        # The word should appear, not blank
        assert asst[-1] == "Overflow", f"Expected 'Overflow' but got: {asst}"


# ===========================================================================
# Tests: compute_word_spans
# ===========================================================================
class TestComputeWordSpans:

    def test_simple(self):
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.3),
            WordAlignment(text="world", start_time=0.4, end_time=0.6),
        ]
        spans = compute_word_spans(alignments, "hello world")
        assert spans == [(0, 5), (6, 11)]

    def test_trailing_punctuation_included(self):
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.3),
            WordAlignment(text="world", start_time=0.4, end_time=0.6),
        ]
        spans = compute_word_spans(alignments, "hello, world!")
        assert spans == [(0, 6), (7, 13)]  # "hello," and "world!"

    def test_quotes_included(self):
        alignments = [
            WordAlignment(text="good", start_time=0.0, end_time=0.2),
            WordAlignment(text="night", start_time=0.3, end_time=0.5),
        ]
        spans = compute_word_spans(alignments, "'good night'")
        # "good" found at idx 1, trailing: nothing (space follows)
        # "night" found at idx 6, trailing: "'"
        assert spans == [(1, 5), (6, 12)]

    def test_case_insensitive_match(self):
        alignments = [WordAlignment(text="Hello", start_time=0.0, end_time=0.3)]
        spans = compute_word_spans(alignments, "HELLO world")
        assert spans == [(0, 5)]

    def test_word_not_found(self):
        alignments = [WordAlignment(text="missing", start_time=0.0, end_time=0.3)]
        spans = compute_word_spans(alignments, "hello world")
        assert spans == [None]

    def test_sequential_search(self):
        """Repeated words match sequentially, not all to the first occurrence."""
        alignments = [
            WordAlignment(text="the", start_time=0.0, end_time=0.1),
            WordAlignment(text="the", start_time=0.5, end_time=0.6),
        ]
        spans = compute_word_spans(alignments, "the cat and the dog")
        assert spans == [(0, 3), (12, 15)]

    def test_empty_alignments(self):
        assert compute_word_spans([], "hello world") == []

    # --- preserve_trailing_whitespace ---

    def test_preserve_trailing_whitespace_extends_to_next_word(self):
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.3),
            WordAlignment(text="world", start_time=0.4, end_time=0.6),
        ]
        spans = compute_word_spans(alignments, "hello world", preserve_trailing_whitespace=True)
        # "hello " (includes trailing space), "world" (no trailing space at end)
        assert spans == [(0, 6), (6, 11)]

    def test_preserve_trailing_whitespace_with_punctuation(self):
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.3),
            WordAlignment(text="world", start_time=0.4, end_time=0.6),
        ]
        spans = compute_word_spans(alignments, "hello, world!", preserve_trailing_whitespace=True)
        # "hello, " (punct + space), "world!" (punct, no trailing space)
        assert spans == [(0, 7), (7, 13)]

    def test_preserve_trailing_whitespace_multi_space(self):
        alignments = [
            WordAlignment(text="a", start_time=0.0, end_time=0.1),
            WordAlignment(text="b", start_time=0.2, end_time=0.3),
        ]
        spans = compute_word_spans(alignments, "a   b", preserve_trailing_whitespace=True)
        # "a   " (3 spaces consumed), "b"
        assert spans == [(0, 4), (4, 5)]

    def test_preserve_trailing_whitespace_last_word_no_trailing(self):
        """Last word's span should not extend beyond the transcript."""
        alignments = [WordAlignment(text="end", start_time=0.0, end_time=0.3)]
        spans = compute_word_spans(alignments, "the end", preserve_trailing_whitespace=True)
        assert spans == [(4, 7)]  # no trailing space to consume

    # --- preserve_leading_whitespace ---

    def test_preserve_leading_whitespace_basic(self):
        """First word gets no leading space; subsequent words own their leading space."""
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.3),
            WordAlignment(text="world", start_time=0.4, end_time=0.6),
        ]
        spans = compute_word_spans(alignments, "hello world", preserve_leading_whitespace=True)
        # "hello" (no leading space at idx=0), " world" (leading space included)
        assert spans == [(0, 5), (5, 11)]

    def test_preserve_leading_whitespace_with_punctuation(self):
        """Leading whitespace extends back through spaces but stops at punctuation."""
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.3),
            WordAlignment(text="world", start_time=0.4, end_time=0.6),
        ]
        spans = compute_word_spans(alignments, "hello, world!", preserve_leading_whitespace=True)
        # "hello," (trailing punct), " world!" (leading space + trailing punct).
        # Previous word's end=6 (after comma), so " " at idx=6 is consumed by "world".
        assert spans == [(0, 6), (6, 13)]

    def test_preserve_leading_whitespace_multi_space(self):
        """Multiple spaces between words: all go to the following word."""
        alignments = [
            WordAlignment(text="a", start_time=0.0, end_time=0.1),
            WordAlignment(text="b", start_time=0.2, end_time=0.3),
        ]
        spans = compute_word_spans(alignments, "a   b", preserve_leading_whitespace=True)
        # "a" (no leading), "   b" (3 spaces)
        assert spans == [(0, 1), (1, 5)]

    def test_preserve_leading_whitespace_first_word_not_at_zero(self):
        """If the first word isn't at position 0, leading spaces still go to it."""
        alignments = [WordAlignment(text="end", start_time=0.0, end_time=0.3)]
        spans = compute_word_spans(alignments, "  end", preserve_leading_whitespace=True)
        # First word — search_pos=0, extends back to start → "  end"
        assert spans == [(0, 5)]

    def test_preserve_leading_whitespace_no_overlap_with_prev_span(self):
        """Leading whitespace must not overlap with the previous word's span end."""
        alignments = [
            WordAlignment(text="a", start_time=0.0, end_time=0.1),
            WordAlignment(text="b", start_time=0.2, end_time=0.3),
            WordAlignment(text="c", start_time=0.4, end_time=0.5),
        ]
        spans = compute_word_spans(alignments, "a b c", preserve_leading_whitespace=True)
        # "a" (0,1), " b" (1,3), " c" (3,5) — no overlap.
        assert spans == [(0, 1), (1, 3), (3, 5)]

    def test_preserve_leading_whitespace_concat_matches_full(self):
        """Concatenating the span texts yields the full transcript (modulo leading chars)."""
        alignments = [
            WordAlignment(text="Hello", start_time=0.0, end_time=0.3),
            WordAlignment(text="world", start_time=0.4, end_time=0.6),
            WordAlignment(text="Nice", start_time=0.7, end_time=1.0),
            WordAlignment(text="day", start_time=1.1, end_time=1.4),
        ]
        transcript = "Hello, world! Nice day."
        spans = compute_word_spans(alignments, transcript, preserve_leading_whitespace=True)
        # Each span "owns" its leading whitespace (if any).
        assert spans == [(0, 6), (6, 13), (13, 18), (18, 23)]
        pieces = [transcript[s:e] for s, e in spans]
        assert pieces == ["Hello,", " world!", " Nice", " day."]
        assert "".join(pieces) == transcript

    def test_preserve_leading_and_trailing_whitespace_rejected(self):
        """Using both leading and trailing whitespace modes simultaneously is rejected."""
        alignments = [WordAlignment(text="hello", start_time=0.0, end_time=0.3)]
        with pytest.raises(ValueError, match="cannot be True at the same time"):
            compute_word_spans(
                alignments,
                "hello",
                preserve_leading_whitespace=True,
                preserve_trailing_whitespace=True,
            )


# ===========================================================================
# Tests: decode_with_blank
# ===========================================================================
class TestDecodeWithBlank:
    """Tests for decode_with_blank, covering both the standard blank path and
    the empty-blank fallback (which splits on EOS instead)."""

    @pytest.fixture
    def qwen3_tok(self):
        try:
            from transformers import AutoTokenizer as HFAutoTokenizer

            hf_tok = HFAutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        except Exception:
            pytest.skip("Qwen3 tokenizer not available")
        if "<blank>" not in hf_tok.get_vocab():
            hf_tok.add_special_tokens({"additional_special_tokens": ["<blank>"]})

        class _Wrapper:
            def __init__(self, hf):
                self.tokenizer = hf

            def ids_to_tokens(self, ids):
                return self.tokenizer.convert_ids_to_tokens(ids)

            def tokens_to_text(self, tokens, remove_special_tokens=True):
                if remove_special_tokens:
                    tokens = [t for t in tokens if t not in self.tokenizer.all_special_tokens]
                return self.tokenizer.convert_tokens_to_string(tokens)

        return _Wrapper(hf_tok)

    # --- empty-blank path: split on EOS (<|im_end|>) ---

    def test_empty_blank_splits_on_eos(self, qwen3_tok):
        """Two per-chunk segments separated by EOS decode and join with a space."""
        hf = qwen3_tok.tokenizer
        eos_id = hf.eos_token_id
        hello_ids = hf.encode("hello", add_special_tokens=False)
        world_ids = hf.encode("world", add_special_tokens=False)
        ids = hello_ids + [eos_id] + world_ids + [eos_id]

        text = decode_with_blank(ids, blank_token="", tokenizer=qwen3_tok)
        assert text == "hello world"

    def test_empty_blank_no_eos_decodes_as_one_segment(self, qwen3_tok):
        """Without EOS separators, BPE-merges ruin the output (documents current behavior)."""
        hf = qwen3_tok.tokenizer
        ids = hf.encode("hello", add_special_tokens=False) + hf.encode("world", add_special_tokens=False)

        text = decode_with_blank(ids, blank_token="", tokenizer=qwen3_tok)
        # Without a separator between chunks, tokens decode as one run.
        assert text == "helloworld"

    def test_empty_blank_preserves_inline_spaces(self, qwen3_tok):
        """Leading-space tokens inside a chunk decode naturally (BPE keeps spacing)."""
        hf = qwen3_tok.tokenizer
        eos_id = hf.eos_token_id
        # Multi-word chunk (leading-space BPE tokens) then a second single-word chunk.
        multi_ids = hf.encode("hello world", add_special_tokens=False)
        nice_ids = hf.encode(" nice", add_special_tokens=False)
        ids = multi_ids + [eos_id] + nice_ids + [eos_id]

        text = decode_with_blank(ids, blank_token="", tokenizer=qwen3_tok)
        # collapse_whitespace=True (default) squashes the leading-space join to single spaces.
        assert text == "hello world nice"

    def test_empty_blank_single_chunk(self, qwen3_tok):
        """A single trailing EOS still produces the correct text."""
        hf = qwen3_tok.tokenizer
        eos_id = hf.eos_token_id
        ids = hf.encode("hello", add_special_tokens=False) + [eos_id]

        text = decode_with_blank(ids, blank_token="", tokenizer=qwen3_tok)
        assert text == "hello"

    def test_empty_blank_no_content(self, qwen3_tok):
        """Only EOS separators (silent chunks) → empty string."""
        eos_id = qwen3_tok.tokenizer.eos_token_id
        text = decode_with_blank([eos_id, eos_id], blank_token="", tokenizer=qwen3_tok)
        assert text == ""

    def test_empty_blank_strip_whitespace(self, qwen3_tok):
        """strip_whitespace removes leading/trailing whitespace from the final output."""
        hf = qwen3_tok.tokenizer
        eos_id = hf.eos_token_id
        # Leading-space BPE token at the very start of the sequence.
        ids = hf.encode(" hello", add_special_tokens=False) + [eos_id]

        text = decode_with_blank(ids, blank_token="", tokenizer=qwen3_tok, strip_whitespace=True)
        assert text == "hello"

    # --- standard blank path (sanity) ---

    def test_explicit_blank_splits_on_blank_id(self, qwen3_tok):
        """With an explicit <blank> token, decoding splits on its id (not EOS)."""
        hf = qwen3_tok.tokenizer
        blank_id = hf.convert_tokens_to_ids("<blank>")
        hello_ids = hf.encode("hello", add_special_tokens=False)
        world_ids = hf.encode("world", add_special_tokens=False)
        ids = hello_ids + [blank_id] + world_ids + [blank_id]

        text = decode_with_blank(ids, blank_token="<blank>", tokenizer=qwen3_tok)
        assert text == "hello world"


# ===========================================================================
# Tests: get_llm_messages_for_sample with transcript
# ===========================================================================
class TestTranscriptPreservation:

    def test_punctuation_preserved(self):
        """Trailing punctuation from transcript should be in assistant content."""
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.08),
            WordAlignment(text="world", start_time=0.10, end_time=0.16),
        ]
        msgs = get_llm_messages_for_sample(
            system_role=SYSTEM_ROLE,
            system_prompt=SYSTEM_PROMPT,
            audio_tag=AUDIO_TAG,
            blank_token=BLANK_TOKEN,
            chunk_size=2,
            num_delay_frames=0,
            audio_duration_secs=0.16,
            frame_length_in_secs=FRAME_LEN,
            alignments=alignments,
            transcript="Hello, World!",
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[0] == "Hello, World!"

    def test_multi_word_chunk_preserves_spacing(self):
        """Multiple words in one chunk use the transcript's inter-word text."""
        alignments = [
            WordAlignment(text="said", start_time=0.0, end_time=0.06),
            WordAlignment(text="good", start_time=0.07, end_time=0.10),
        ]
        msgs = get_llm_messages_for_sample(
            system_role=SYSTEM_ROLE,
            system_prompt=SYSTEM_PROMPT,
            audio_tag=AUDIO_TAG,
            blank_token=BLANK_TOKEN,
            chunk_size=2,
            num_delay_frames=0,
            audio_duration_secs=0.16,
            frame_length_in_secs=FRAME_LEN,
            alignments=alignments,
            transcript="she said good night",
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        # Trailing space is excluded because preserve_leading_whitespace=True
        #  ensures correct concatenation when turns are joined.
        assert asst[0] == " said good"

    def test_without_transcript_falls_back(self):
        """Without transcript, words are joined with plain space."""
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.08),
            WordAlignment(text="world", start_time=0.10, end_time=0.16),
        ]
        msgs = get_llm_messages_for_sample(
            system_role=SYSTEM_ROLE,
            system_prompt=SYSTEM_PROMPT,
            audio_tag=AUDIO_TAG,
            blank_token=BLANK_TOKEN,
            chunk_size=2,
            num_delay_frames=0,
            audio_duration_secs=0.16,
            frame_length_in_secs=FRAME_LEN,
            alignments=alignments,
            transcript=None,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[0] == "hello world"

    def test_single_word_with_comma(self):
        """A single word followed by comma should include the comma."""
        alignments = [
            WordAlignment(text="yes", start_time=0.0, end_time=0.08),
            WordAlignment(text="indeed", start_time=0.20, end_time=0.30),
        ]
        msgs = get_llm_messages_for_sample(
            system_role=SYSTEM_ROLE,
            system_prompt=SYSTEM_PROMPT,
            audio_tag=AUDIO_TAG,
            blank_token=BLANK_TOKEN,
            chunk_size=2,
            num_delay_frames=0,
            audio_duration_secs=0.32,
            frame_length_in_secs=FRAME_LEN,
            alignments=alignments,
            transcript="yes, indeed",
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        # "yes" is ready at chunk 0 (end_frame=1 <= 2), alone in its chunk.
        # Trailing space excluded via preserve_leading_whitespace (space before "indeed").
        assert asst[0] == "yes,"

    def test_blanks_unchanged_with_transcript(self):
        """Blank chunks still produce <blank> even when transcript is provided."""
        msgs = _make_messages(transcript="Hello World")
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        # First two chunks are blank
        assert asst[0] == BLANK_TOKEN
        assert asst[1] == BLANK_TOKEN


# ===========================================================================
# Tests: get_llm_messages_for_batch
# ===========================================================================
class TestGetLlmMessagesForBatch:

    def test_per_sample_duration(self):
        """Each sample gets messages based on its own duration, not a shared max."""
        alignments = [[], []]
        durations = [0.16, 0.32]  # 2 frames → 1 chunk, 4 frames → 2 chunks
        batch = get_llm_messages_for_batch(
            system_role=SYSTEM_ROLE,
            system_prompt=[SYSTEM_PROMPT, SYSTEM_PROMPT],
            audio_tag=AUDIO_TAG,
            blank_token=BLANK_TOKEN,
            chunk_size=CHUNK_SIZE,
            num_delay_frames=0,
            audio_durations_secs=durations,
            frame_length_in_secs=FRAME_LEN,
            alignments=alignments,
        )
        assert len(batch) == 2
        chunks_0 = sum(1 for m in batch[0] if m["role"] == "user")
        chunks_1 = sum(1 for m in batch[1] if m["role"] == "user")
        assert chunks_0 == 1
        assert chunks_1 == 2

    def test_per_sample_system_prompt(self):
        """Each sample gets its own system prompt from the list."""
        prompts = ["Transcribe in English.", "Transcribe in French."]
        alignments = [[], []]
        durations = [0.16, 0.16]
        batch = get_llm_messages_for_batch(
            system_role=SYSTEM_ROLE,
            system_prompt=prompts,
            audio_tag=AUDIO_TAG,
            blank_token=BLANK_TOKEN,
            chunk_size=CHUNK_SIZE,
            num_delay_frames=0,
            audio_durations_secs=durations,
            frame_length_in_secs=FRAME_LEN,
            alignments=alignments,
        )
        assert batch[0][0] == {"role": SYSTEM_ROLE, "content": "Transcribe in English."}
        assert batch[1][0] == {"role": SYSTEM_ROLE, "content": "Transcribe in French."}


# ===========================================================================
# Tests: _replace_audio_chunks
# ===========================================================================
class TestReplaceAudioChunks:

    AUD = AUDIO_TOKEN_IDX

    def test_single_token_chunk(self):
        """Single-token audio tag, chunk_size=2 → 2 AUDIO_TOKEN_IDX per chunk."""
        ids = [1, 100, 100, 2]
        result = _replace_audio_chunks(ids, [100, 100], chunk_size=2)
        assert result == [1, self.AUD, self.AUD, 2]

    def test_multi_token_chunk_with_bpe_merge(self):
        """Simulates BPE merge: <audio><audio> → [60, 61, 70, 61, 62] (5 tokens, not 6)."""
        chunk_ids = [60, 61, 70, 61, 62]
        ids = [1, 2, 3] + chunk_ids + [4, 5]
        result = _replace_audio_chunks(ids, chunk_ids, chunk_size=2)
        assert result == [1, 2, 3, self.AUD, self.AUD, 4, 5]

    def test_multiple_chunks(self):
        chunk_ids = [60, 61, 70, 61, 62]
        ids = chunk_ids + [99] + chunk_ids + [88]
        result = _replace_audio_chunks(ids, chunk_ids, chunk_size=2)
        assert result == [self.AUD, self.AUD, 99, self.AUD, self.AUD, 88]
        assert result.count(self.AUD) == 4  # 2 chunks × 2

    def test_chunk_size_1(self):
        """chunk_size=1: each chunk token sequence replaced with 1 AUDIO_TOKEN_IDX."""
        chunk_ids = [60, 61, 62]  # single <audio> as 3 BPE tokens
        ids = [1] + chunk_ids + [2] + chunk_ids + [3]
        result = _replace_audio_chunks(ids, chunk_ids, chunk_size=1)
        assert result == [1, self.AUD, 2, self.AUD, 3]

    def test_chunk_size_4(self):
        """chunk_size=4: each chunk replaced with 4 AUDIO_TOKEN_IDX."""
        chunk_ids = [10, 11, 12, 13]
        ids = [1] + chunk_ids + [2]
        result = _replace_audio_chunks(ids, chunk_ids, chunk_size=4)
        assert result == [1, self.AUD, self.AUD, self.AUD, self.AUD, 2]

    def test_mask_sync(self):
        chunk_ids = [60, 61, 70, 61, 62]
        ids = [1] + chunk_ids + [2]
        mask = [0] + [0, 0, 0, 0, 0] + [1]
        new_ids, new_mask = _replace_audio_chunks(ids, chunk_ids, chunk_size=2, mask=mask)
        assert new_ids == [1, self.AUD, self.AUD, 2]
        assert new_mask == [0, 0, 0, 1]
        assert len(new_ids) == len(new_mask)

    def test_mask_length_with_different_chunk_size(self):
        """Mask length must match ids length after chunk replacement."""
        chunk_ids = [10, 11, 12, 13, 14]  # 5 BPE tokens
        ids = [1] + chunk_ids + [2] + chunk_ids + [3]
        mask = [0] + [0] * 5 + [1] + [0] * 5 + [1]
        new_ids, new_mask = _replace_audio_chunks(ids, chunk_ids, chunk_size=3, mask=mask)
        # 5 tokens → 3 AUDIO_TOKEN_IDX per chunk, 2 chunks
        assert new_ids.count(self.AUD) == 6
        assert len(new_ids) == len(new_mask)

    def test_no_match(self):
        result = _replace_audio_chunks([1, 2, 3], [100, 100], chunk_size=2)
        assert result == [1, 2, 3]


# ===========================================================================
# Tests: token positions (full pipeline through mock tokenizer)
# ===========================================================================
class TestTokenPositions:
    """
    Verify audio/text token counts and positions in input_ids and target_ids
    using the docstring example.
    """

    def test_audio_token_count_single_token_tag(self):
        """AUDIO_TOKEN_IDX count == num_chunks * chunk_size (single-token tag)."""
        msgs = _make_messages()
        (
            input_ids,
            _,
            _,
        ) = _run_pipeline(msgs, _MockHFTokenizer())
        num_chunks = 7
        assert input_ids.count(AUDIO_TOKEN_IDX) == num_chunks * CHUNK_SIZE

    def test_audio_token_count_multi_token_tag(self):
        """Same count even when the audio tag tokenizes into 3 tokens."""
        msgs = _make_messages()
        input_ids, _, _ = _run_pipeline(msgs, _MockHFTokenizerMultiToken())
        num_chunks = 7
        assert input_ids.count(AUDIO_TOKEN_IDX) == num_chunks * CHUNK_SIZE

    def test_no_audio_token_at_assistant_position(self):
        msgs = _make_messages()
        input_ids, _, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        for i, (tid, m) in enumerate(zip(input_ids, assistant_mask)):
            if m:
                assert tid != AUDIO_TOKEN_IDX, f"Audio token at assistant position {i}"

    def test_no_audio_token_at_assistant_position_multi(self):
        msgs = _make_messages()
        input_ids, _, assistant_mask = _run_pipeline(msgs, _MockHFTokenizerMultiToken())
        for i, (tid, m) in enumerate(zip(input_ids, assistant_mask)):
            if m:
                assert tid != AUDIO_TOKEN_IDX, f"Audio token at assistant position {i}"

    def test_target_ignore_at_non_assistant(self):
        """Every non-assistant position in target must be IGNORE_INDEX."""
        msgs = _make_messages()
        input_ids, target_ids, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        shifted_mask = assistant_mask[1:] + [0]
        for i, (tid, m) in enumerate(zip(target_ids, shifted_mask)):
            if not m:
                assert tid == IGNORE_INDEX, f"target[{i}]={tid} should be IGNORE_INDEX"

    def test_target_real_at_assistant(self):
        """Every assistant position in target must hold a real token ID."""
        msgs = _make_messages()
        input_ids, target_ids, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        shifted_mask = assistant_mask[1:] + [0]
        for i, (tid, m) in enumerate(zip(target_ids, shifted_mask)):
            if m:
                assert tid != IGNORE_INDEX, f"target[{i}] should be a real token"

    def test_target_equals_next_input_at_assistant(self):
        """target[i] must equal input[i+1] at trainable positions (next-token prediction)."""
        msgs = _make_messages()
        input_ids, target_ids, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        shifted = input_ids[1:] + [IGNORE_INDEX]
        shifted_mask = assistant_mask[1:] + [0]
        for i, m in enumerate(shifted_mask):
            if m:
                assert target_ids[i] == shifted[i], f"target[{i}]={target_ids[i]} != next input {shifted[i]}"

    def test_input_target_same_length(self):
        msgs = _make_messages()
        input_ids, target_ids, _ = _run_pipeline(msgs, _MockHFTokenizer())
        assert len(input_ids) == len(target_ids)

    def test_input_target_same_length_multi_token(self):
        msgs = _make_messages()
        input_ids, target_ids, _ = _run_pipeline(msgs, _MockHFTokenizerMultiToken())
        assert len(input_ids) == len(target_ids)

    def test_mask_length_matches_input_after_replace(self):
        """After multi-token collapse, mask and input_ids must have the same length."""
        msgs = _make_messages()
        input_ids, _, assistant_mask = _run_pipeline(msgs, _MockHFTokenizerMultiToken())
        assert len(input_ids) == len(assistant_mask)

    def test_all_blank_targets_with_no_alignments(self):
        """With no alignments, every assistant content token in input should be BLANK_ID."""
        msgs = _make_messages(alignments=[])
        input_ids, _, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        blank_id = _MockHFTokenizer.BLANK_ID
        for i, (tid, m) in enumerate(zip(input_ids, assistant_mask)):
            if m:
                assert tid == blank_id, f"Expected blank at position {i}, got {tid}"

    def test_hello_appears_at_chunk_2(self):
        """'Hello' (end_time=0.48s, end_frame=6) is emitted at chunk 2 (end_frame=6)."""
        msgs = _make_messages(
            alignments=[WordAlignment(text="Hello", start_time=0.16, end_time=0.48)],
        )
        input_ids, _, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        blank_id = _MockHFTokenizer.BLANK_ID

        # Collect assistant content token values in order
        asst_tokens = [tid for tid, m in zip(input_ids, assistant_mask) if m]
        # Chunks: 0=blank, 1=blank, 2=Hello (non-blank), 3..6=blank
        assert asst_tokens[0] == blank_id
        assert asst_tokens[1] == blank_id
        assert asst_tokens[2] != blank_id  # Hello word token
        assert all(t == blank_id for t in asst_tokens[3:])

    def test_trainable_token_count(self):
        """Number of trainable positions in target == number of assistant content tokens."""
        msgs = _make_messages()
        _, target_ids, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        n_trainable = sum(1 for t in target_ids if t != IGNORE_INDEX)
        n_assistant = sum(assistant_mask)
        # The shifted mask loses the first assistant token's prediction target
        # when it's preceded by a non-assistant token, but gains/loses nothing
        # else.  The exact count: sum(shifted_mask).
        shifted_mask = assistant_mask[1:] + [0]
        assert n_trainable == sum(shifted_mask)


# ===========================================================================
# Tests: _tokenize_with_assistant_mask fallback
# ===========================================================================
class TestTokenizeWithAssistantMaskFallback:
    """
    Verify the sequential-search fallback produces the same mask as the
    primary path when the tokenizer doesn't support {% generation %}.
    """

    def test_fallback_includes_primary_content_plus_footer(self):
        """Fallback mask should include all primary-masked (content) positions plus footer tokens."""
        msgs = _make_messages()
        primary_tok = _MockHFTokenizer()
        fallback_tok = _MockHFTokenizerNoGeneration()

        nemo_primary = _MockNemoTokenizer(primary_tok)
        nemo_fallback = _MockNemoTokenizer(fallback_tok)

        ids_p, mask_p = _tokenize_with_assistant_mask(msgs, nemo_primary)
        ids_f, mask_f = _tokenize_with_assistant_mask(msgs, nemo_fallback)

        assert ids_p == ids_f, "Token IDs should be identical"
        # Fallback mask includes all primary-masked (content) positions
        for i, (mp, mf) in enumerate(zip(mask_p, mask_f)):
            if mp:
                assert mf, f"Position {i}: primary has mask=1 but fallback has mask=0"
        # Fallback has additional masked positions (footer tokens)
        assert sum(mask_f) > sum(mask_p), "Fallback should have additional footer positions"

    def test_fallback_has_nonzero_mask(self):
        """Fallback should produce assistant-masked tokens, not all zeros."""
        msgs = _make_messages()
        tok = _MockHFTokenizerNoGeneration()
        nemo_tok = _MockNemoTokenizer(tok)

        _, mask = _tokenize_with_assistant_mask(msgs, nemo_tok)
        assert any(mask), "Fallback mask should have at least one assistant token"

    def test_fallback_mask_count_includes_eos(self):
        """Number of masked tokens should equal assistant content + 1 EOS token per turn."""
        msgs = _make_messages()
        tok = _MockHFTokenizerNoGeneration()
        nemo_tok = _MockNemoTokenizer(tok)

        _, mask = _tokenize_with_assistant_mask(msgs, nemo_tok)
        # 7 assistant turns: 7 content tokens + 7 * 1 EOS token (FOOTER only, not NEWLINE) = 14
        assert sum(mask) == 7 + 7 * 1

    def test_fallback_eos_in_mask_but_not_rest_of_footer(self):
        """Fallback should mask the EOS token but not post-EOS footer tokens."""
        msgs = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": "<blank>"},
        ]
        tok = _MockHFTokenizerNoGeneration()
        nemo_tok = _MockNemoTokenizer(tok)

        input_ids, mask = _tokenize_with_assistant_mask(msgs, nemo_tok)

        # Find the blank token position
        blank_id = _MockHFTokenizer.BLANK_ID
        blank_pos = input_ids.index(blank_id)

        # FOOTER (= eos_token_id) should be masked, NEWLINE should NOT
        assert input_ids[blank_pos + 1] == _MockHFTokenizer.FOOTER
        assert input_ids[blank_pos + 2] == _MockHFTokenizer.NEWLINE
        assert mask[blank_pos] == 1, "Content token should be masked"
        assert mask[blank_pos + 1] == 1, "EOS (FOOTER) should be masked"
        assert mask[blank_pos + 2] == 0, "Post-EOS (NEWLINE) should NOT be masked"

    def test_fallback_no_eos_only_content_masked(self):
        """When eos_token_id is None, only content is masked (no footer)."""
        msgs = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": "<blank>"},
        ]
        tok = _MockHFTokenizerNoEOS()
        nemo_tok = _MockNemoTokenizer(tok)

        input_ids, mask = _tokenize_with_assistant_mask(msgs, nemo_tok)

        blank_id = _MockHFTokenizer.BLANK_ID
        blank_pos = input_ids.index(blank_id)

        # Without EOS, only content is masked — footer tokens are not.
        assert mask[blank_pos] == 1, "Content should be masked"
        assert mask[blank_pos + 1] == 0, "FOOTER should NOT be masked (no EOS)"
        assert mask[blank_pos + 2] == 0, "NEWLINE should NOT be masked"

    def test_fallback_pipeline_produces_trainable_targets(self):
        """Full pipeline with fallback tokenizer should have non-zero trainable targets."""
        msgs = _make_messages()
        tok = _MockHFTokenizerNoGeneration()
        input_ids, target_ids, assistant_mask = _run_pipeline(msgs, tok)

        n_trainable = sum(1 for t in target_ids if t != IGNORE_INDEX)
        assert n_trainable > 0, "Should have trainable targets with fallback mask"

    def test_fallback_eot_in_target(self):
        """After shift, the model should be trained to predict the end-of-turn token."""
        msgs = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": "<blank>"},
        ]
        tok = _MockHFTokenizerNoGeneration()
        input_ids, target_ids, _ = _run_pipeline(msgs, tok)

        # The FOOTER token (end-of-turn) should appear as a trainable target
        footer_id = _MockHFTokenizer.FOOTER
        assert footer_id in target_ids, "FOOTER should appear as a trainable target"
        # Verify it's not masked out
        footer_target_pos = target_ids.index(footer_id)
        assert target_ids[footer_target_pos] != IGNORE_INDEX


# ===========================================================================
# Tests: chunk_size=-1 (offline / single-chunk mode)
# ===========================================================================
class TestOfflineSingleChunk:
    """Verify chunk_size=-1 treats the whole audio as one chunk."""

    def test_single_chunk_structure(self):
        """chunk_size=-1 should produce exactly 1 user turn + 1 assistant turn."""
        msgs = _make_messages(chunk_size=-1)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(user_msgs) == 1
        assert len(asst_msgs) == 1

    def test_audio_tags_equal_num_frames(self):
        """The single user turn should have num_frames audio tags."""
        msgs = _make_messages(chunk_size=-1, audio_duration_secs=1.0)
        user_msg = [m for m in msgs if m["role"] == "user"][0]
        num_frames = math.ceil(1.0 / FRAME_LEN)  # 13
        assert user_msg["content"] == AUDIO_TAG * num_frames

    def test_all_words_in_single_turn(self):
        """All words should appear in the single assistant turn."""
        msgs = _make_messages(chunk_size=-1)
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert len(asst) == 1
        assert "Hello" in asst[0]
        assert "World" in asst[0]

    def test_no_blanks(self):
        """With all audio in one chunk, all words are ready — no blanks."""
        msgs = _make_messages(chunk_size=-1)
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert BLANK_TOKEN not in asst

    def test_transcript_preserved(self):
        """Punctuation from transcript should be preserved in single-chunk mode."""
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.08),
            WordAlignment(text="world", start_time=0.10, end_time=0.16),
        ]
        msgs = _make_messages(
            chunk_size=-1,
            audio_duration_secs=0.16,
            alignments=alignments,
            transcript="Hello, World!",
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[0] == "Hello, World!"

    def test_empty_alignments_no_transcript_produces_blank(self):
        """No alignments and no transcript → single blank turn."""
        msgs = _make_messages(chunk_size=-1, alignments=[], transcript=None)
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst == [BLANK_TOKEN]

    def test_empty_alignments_with_transcript_uses_transcript(self):
        """No alignments but transcript provided → uses raw transcript."""
        msgs = _make_messages(
            chunk_size=-1,
            alignments=[],
            audio_duration_secs=1.0,
            transcript="Hello, World!",
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst == ["Hello, World!"]

    def test_none_alignments_with_transcript_uses_transcript(self):
        """None alignments with transcript → uses raw transcript."""
        msgs = _make_messages(
            chunk_size=-1,
            alignments=None,
            audio_duration_secs=1.0,
            transcript="some text here",
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst == ["some text here"]

    def test_zero_duration(self):
        """Zero-duration audio with no alignments → early return with empty user turn and blank."""
        msgs = _make_messages(chunk_size=-1, audio_duration_secs=0.0, alignments=[])
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1] == {"role": "user", "content": ""}
        assert msgs[2] == {"role": "assistant", "content": BLANK_TOKEN}

    def test_delay_ignored_single_chunk(self):
        """With one chunk spanning all frames, delay shouldn't matter
        (all words fit within the single chunk's end frame)."""
        msgs = _make_messages(chunk_size=-1, num_delay_frames=2)
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        # num_frames=13, chunk_end_frame=13.
        # Hello: end_frame=6, ready=8 <= 13 ✓
        # World: end_frame=10, ready=12 <= 13 ✓
        assert len(asst) == 1
        assert "Hello" in asst[0]
        assert "World" in asst[0]

    def test_delay_causes_residual_in_single_chunk(self):
        """Large delay can push a word past the single chunk's end frame → residual."""
        # Audio 1.0s → 13 frames → chunk_end_frame=13
        # World: end_frame=ceil(0.80/0.08)=10, with delay=5 → ready_frame=15 > 13
        msgs = _make_messages(chunk_size=-1, num_delay_frames=5)
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert len(asst) == 1
        # World should still appear via the residual logic
        assert "Hello" in asst[0]
        assert "World" in asst[0]

    def test_matches_large_chunk_size(self):
        """chunk_size=-1 should produce the same result as chunk_size=num_frames."""
        num_frames = math.ceil(1.0 / FRAME_LEN)
        msgs_offline = _make_messages(chunk_size=-1)
        msgs_large = _make_messages(chunk_size=num_frames)
        # Both should have 1 system + 1 user + 1 assistant = 3 messages
        assert len(msgs_offline) == len(msgs_large)
        # Same assistant content
        asst_offline = [m["content"] for m in msgs_offline if m["role"] == "assistant"]
        asst_large = [m["content"] for m in msgs_large if m["role"] == "assistant"]
        assert asst_offline == asst_large
        # Same user content (same number of audio tags)
        user_offline = [m["content"] for m in msgs_offline if m["role"] == "user"]
        user_large = [m["content"] for m in msgs_large if m["role"] == "user"]
        assert user_offline == user_large

    def test_chunk_size_zero_is_dynamic(self):
        """chunk_size=0 is dynamic chunking — should not raise."""
        msgs = _make_messages(chunk_size=0)
        # Dynamic chunking with the docstring alignments should produce
        # user turns with variable frame counts (not fixed chunk_size).
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) >= 1


# ===========================================================================
# Tests: chunk_size=0 (dynamic chunking)
# ===========================================================================
class TestDynamicChunking:
    """Verify chunk_size=0 creates variable-size chunks aligned to word boundaries."""

    def test_docstring_example(self):
        """The plan example: Hello at 0.48s, World at 0.80s, 1s audio."""
        alignments = [
            WordAlignment(text="Hello", start_time=0.16, end_time=0.48),
            WordAlignment(text="World", start_time=0.60, end_time=0.80),
        ]
        msgs = _make_messages(chunk_size=0, alignments=alignments, audio_duration_secs=1.0)

        user_msgs = [m for m in msgs if m["role"] == "user"]
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]

        # 3 user turns: 6 frames (0-0.48s), 4 frames (0.48-0.80s), 3 frames (0.80-1.04s trailing)
        assert len(user_msgs) == 3
        assert user_msgs[0]["content"] == AUDIO_TAG * 6
        assert user_msgs[1]["content"] == AUDIO_TAG * 4
        assert user_msgs[2]["content"] == AUDIO_TAG * 3  # trailing silence

        # 2 assistant turns (no assistant for trailing silence)
        assert len(asst_msgs) == 2
        assert asst_msgs[0]["content"] == "Hello"
        assert asst_msgs[1]["content"] == "World"

    def test_single_word(self):
        """One word → 1 user+assistant turn + trailing silence user turn."""
        alignments = [WordAlignment(text="Hi", start_time=0.0, end_time=0.16)]
        msgs = _make_messages(chunk_size=0, alignments=alignments, audio_duration_secs=0.32)

        user_msgs = [m for m in msgs if m["role"] == "user"]
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]

        assert len(user_msgs) == 2  # word chunk + trailing silence
        assert user_msgs[0]["content"] == AUDIO_TAG * 2  # 0.16s / 0.08s = 2 frames
        assert user_msgs[1]["content"] == AUDIO_TAG * 2  # trailing: 0.16-0.32s = 2 frames
        assert len(asst_msgs) == 1
        assert asst_msgs[0]["content"] == "Hi"

    def test_no_trailing_silence(self):
        """Word ends exactly at audio duration → no trailing user turn."""
        # 0.16s audio, word ends at 0.16s → 2 frames, no trailing
        alignments = [WordAlignment(text="Hi", start_time=0.0, end_time=0.16)]
        msgs = _make_messages(chunk_size=0, alignments=alignments, audio_duration_secs=0.16)

        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 1  # no trailing silence
        assert user_msgs[0]["content"] == AUDIO_TAG * 2

    def test_empty_alignments(self):
        """No words → single user turn with all frames, no assistant."""
        msgs = _make_messages(chunk_size=0, alignments=[], audio_duration_secs=0.32)

        user_msgs = [m for m in msgs if m["role"] == "user"]
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]

        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == AUDIO_TAG * 4
        assert len(asst_msgs) == 0  # no words → no assistant

    def test_adjacent_words_same_boundary(self):
        """Two words ending at same frame → both in same assistant turn."""
        alignments = [
            WordAlignment(text="A", start_time=0.0, end_time=0.08),
            WordAlignment(text="B", start_time=0.08, end_time=0.08),  # ends at same frame
        ]
        msgs = _make_messages(chunk_size=0, alignments=alignments, audio_duration_secs=0.16)

        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        # B ends at same frame as A → both should be in the same turn
        assert any("A" in m["content"] and "B" in m["content"] for m in asst_msgs)

    def test_with_transcript_preserves_punctuation(self):
        """Transcript punctuation is preserved in dynamic chunks."""
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.16),
            WordAlignment(text="world", start_time=0.20, end_time=0.32),
        ]
        msgs = _make_messages(
            chunk_size=0,
            alignments=alignments,
            audio_duration_secs=0.32,
            transcript="Hello, World!",
        )
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert asst_msgs[0]["content"] == "Hello,"
        assert asst_msgs[1]["content"] == " World!"

    def test_with_delay(self):
        """Delay frames shift chunk boundaries."""
        alignments = [WordAlignment(text="Hello", start_time=0.0, end_time=0.16)]
        # Word ends at frame 2, delay=2 → ready at frame 4
        msgs = _make_messages(
            chunk_size=0,
            alignments=alignments,
            audio_duration_secs=0.48,
            num_delay_frames=2,
        )
        user_msgs = [m for m in msgs if m["role"] == "user"]
        # Word chunk: frames 0-3 (4 frames = end_frame 2 + delay 2)
        assert user_msgs[0]["content"] == AUDIO_TAG * 4

    def test_trailing_turn_has_no_assistant(self):
        """Trailing silence user turn should NOT have a paired assistant turn."""
        alignments = [WordAlignment(text="Hi", start_time=0.0, end_time=0.08)]
        msgs = _make_messages(chunk_size=0, alignments=alignments, audio_duration_secs=0.32)

        # Last message should be a user turn (trailing silence), not assistant
        assert msgs[-1]["role"] == "user"


# ===========================================================================
# Tests: words_per_group > 1 (word grouping)
# ===========================================================================
class TestWordsPerChunk:
    """Verify words_per_group groups words into larger assistant turns."""

    FIVE_WORD_ALIGNMENTS = [
        WordAlignment(text="Hello", start_time=0.16, end_time=0.48),
        WordAlignment(text="World", start_time=0.60, end_time=0.80),
        WordAlignment(text="How", start_time=0.90, end_time=1.00),
        WordAlignment(text="Are", start_time=1.10, end_time=1.20),
        WordAlignment(text="You", start_time=1.30, end_time=1.50),
    ]

    def test_dynamic_wpc2_groups_words(self):
        """Dynamic chunking with words_per_group=2 groups pairs."""
        msgs = _make_messages(
            chunk_size=0,
            alignments=self.FIVE_WORD_ALIGNMENTS,
            audio_duration_secs=2.0,
            words_per_group=2,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[0] == "Hello World"
        assert asst[1] == "How Are"
        assert asst[2] == "You"  # remainder

    def test_dynamic_wpc3_groups_words(self):
        """Dynamic chunking with words_per_group=3."""
        msgs = _make_messages(
            chunk_size=0,
            alignments=self.FIVE_WORD_ALIGNMENTS,
            audio_duration_secs=2.0,
            words_per_group=3,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[0] == "Hello World How"
        assert asst[1] == "Are You"  # remainder

    def test_dynamic_wpc1_is_default(self):
        """words_per_group=1 produces one word per turn (same as default)."""
        msgs = _make_messages(
            chunk_size=0,
            alignments=self.FIVE_WORD_ALIGNMENTS,
            audio_duration_secs=2.0,
            words_per_group=1,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert len(asst) == 5

    def test_dynamic_wpc_larger_than_words(self):
        """words_per_group larger than total words → all in one turn."""
        msgs = _make_messages(
            chunk_size=0,
            alignments=self.FIVE_WORD_ALIGNMENTS,
            audio_duration_secs=2.0,
            words_per_group=10,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert len(asst) == 1
        assert "Hello" in asst[0] and "You" in asst[0]

    def test_dynamic_wpc2_audio_frames(self):
        """Audio frame counts match word group boundaries."""
        msgs = _make_messages(
            chunk_size=0,
            alignments=self.FIVE_WORD_ALIGNMENTS,
            audio_duration_secs=2.0,
            words_per_group=2,
        )
        user_msgs = [m for m in msgs if m["role"] == "user"]
        # Group 1: Hello+World → end at frame ceil(0.80/0.08) = 10
        assert user_msgs[0]["content"] == AUDIO_TAG * 10
        # Group 2: How+Are → frames 10 to ceil(1.20/0.08) = 15, so 5 frames
        assert user_msgs[1]["content"] == AUDIO_TAG * 5

    def test_fixed_chunk_wpc3_buffers_words(self):
        """Fixed chunking with words_per_group=3 buffers words across chunks."""
        msgs = _make_messages(
            chunk_size=2,
            alignments=self.FIVE_WORD_ALIGNMENTS,
            audio_duration_secs=2.0,
            words_per_group=3,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        text_turns = [a for a in asst if a != BLANK_TOKEN]
        # First text turn should have 3 words
        assert "Hello" in text_turns[0] and "How" in text_turns[0]
        # Second text turn should have remaining 2 words
        assert "Are" in text_turns[1] and "You" in text_turns[1]

    def test_fixed_chunk_wpc1_is_default(self):
        """Fixed chunking with words_per_group=1 emits words immediately."""
        msgs = _make_messages(
            chunk_size=2,
            alignments=self.FIVE_WORD_ALIGNMENTS,
            audio_duration_secs=2.0,
            words_per_group=1,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        text_turns = [a for a in asst if a != BLANK_TOKEN]
        # Each word in its own turn (possibly grouped if in same chunk)
        assert len(text_turns) >= 3  # some words may share a chunk naturally

    def test_dynamic_wpc_with_transcript(self):
        """Transcript punctuation preserved with word grouping."""
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.16),
            WordAlignment(text="world", start_time=0.20, end_time=0.32),
            WordAlignment(text="how", start_time=0.40, end_time=0.48),
        ]
        msgs = _make_messages(
            chunk_size=0,
            alignments=alignments,
            audio_duration_secs=0.56,
            transcript="Hello, World! How?",
            words_per_group=2,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[0] == "Hello, World!"
        assert asst[1] == " How?"


class TestDynamicChunkTargets:
    """Verify target construction for dynamic chunking (chunk_size=0)."""

    def test_audio_targets_blank_and_footer(self):
        """Audio positions get blank (non-final) or user_footer (final) targets."""
        from nemo.collections.speechlm2.data.streaming_stt_dataset import (
            _replace_audio_chunks,
            _tokenize_with_assistant_mask,
        )

        # Use the mock tokenizer
        tok = _MockHFTokenizerNoGeneration()
        nemo_tok = _MockNemoTokenizer(tok)

        alignments = [
            WordAlignment(text="Hello", start_time=0.16, end_time=0.48),
            WordAlignment(text="World", start_time=0.60, end_time=0.80),
        ]
        msgs = _make_messages(chunk_size=0, alignments=alignments, audio_duration_secs=1.0)

        # Tokenize
        input_ids, mask = _tokenize_with_assistant_mask(msgs, nemo_tok)

        # Replace audio chunks (variable sizes per turn)
        for msg in msgs:
            if msg["role"] != "user":
                continue
            n_tags = msg["content"].count(AUDIO_TAG)
            if n_tags == 0:
                continue
            chunk_ids = tok.encode(AUDIO_TAG * n_tags, add_special_tokens=False)
            input_ids, mask = _replace_audio_chunks(input_ids, chunk_ids, n_tags, mask=mask)

        # Build targets
        target_ids = input_ids[1:] + [IGNORE_INDEX]
        target_mask = mask[1:] + [0]
        target_ids = [tid if m else IGNORE_INDEX for tid, m in zip(target_ids, target_mask)]

        # Simulate dynamic chunk target override
        blank_id = _MockHFTokenizer.BLANK_ID
        footer_id = _MockHFTokenizer.FOOTER  # first token of user footer
        for i in range(len(input_ids)):
            if input_ids[i] != AUDIO_TOKEN_IDX:
                continue
            next_is_audio = i + 1 < len(input_ids) and input_ids[i + 1] == AUDIO_TOKEN_IDX
            target_ids[i] = blank_id if next_is_audio else footer_id

        # Verify: audio positions should have blank or footer targets, never IGNORE_INDEX
        audio_positions = [i for i in range(len(input_ids)) if input_ids[i] == AUDIO_TOKEN_IDX]
        assert len(audio_positions) > 0, "Should have audio positions"

        for i in audio_positions:
            assert target_ids[i] in (blank_id, footer_id), f"Audio position {i} has unexpected target {target_ids[i]}"

        # The last audio frame before each assistant turn should have footer target
        footer_positions = [i for i in audio_positions if target_ids[i] == footer_id]
        assert (
            len(footer_positions) >= 2
        ), f"Expected at least 2 footer targets (Hello + World boundaries), got {len(footer_positions)}"

    def test_no_audio_token_idx_in_targets(self):
        """AUDIO_TOKEN_IDX (-200) must never appear in target_ids."""
        tok = _MockHFTokenizerNoGeneration()
        nemo_tok = _MockNemoTokenizer(tok)

        alignments = [WordAlignment(text="Hello", start_time=0.16, end_time=0.48)]
        msgs = _make_messages(chunk_size=0, alignments=alignments, audio_duration_secs=1.0)

        input_ids, mask = _tokenize_with_assistant_mask(msgs, nemo_tok)

        for msg in msgs:
            if msg["role"] != "user":
                continue
            n_tags = msg["content"].count(AUDIO_TAG)
            if n_tags == 0:
                continue
            chunk_ids = tok.encode(AUDIO_TAG * n_tags, add_special_tokens=False)
            input_ids, mask = _replace_audio_chunks(input_ids, chunk_ids, n_tags, mask=mask)

        target_ids = input_ids[1:] + [IGNORE_INDEX]
        target_mask = mask[1:] + [0]
        target_ids = [tid if m else IGNORE_INDEX for tid, m in zip(target_ids, target_mask)]

        # Apply dynamic chunk override
        blank_id = _MockHFTokenizer.BLANK_ID
        footer_id = _MockHFTokenizer.FOOTER
        for i in range(len(input_ids)):
            if input_ids[i] != AUDIO_TOKEN_IDX:
                continue
            next_is_audio = i + 1 < len(input_ids) and input_ids[i + 1] == AUDIO_TOKEN_IDX
            target_ids[i] = blank_id if next_is_audio else footer_id

        assert AUDIO_TOKEN_IDX not in target_ids, "AUDIO_TOKEN_IDX leaked into targets"

    def test_trailing_silence_all_blank(self):
        """Trailing silence audio frames should all have blank targets."""
        tok = _MockHFTokenizerNoGeneration()
        nemo_tok = _MockNemoTokenizer(tok)

        alignments = [WordAlignment(text="Hi", start_time=0.0, end_time=0.08)]
        msgs = _make_messages(chunk_size=0, alignments=alignments, audio_duration_secs=0.32)

        input_ids, mask = _tokenize_with_assistant_mask(msgs, nemo_tok)

        for msg in msgs:
            if msg["role"] != "user":
                continue
            n_tags = msg["content"].count(AUDIO_TAG)
            if n_tags == 0:
                continue
            chunk_ids = tok.encode(AUDIO_TAG * n_tags, add_special_tokens=False)
            input_ids, mask = _replace_audio_chunks(input_ids, chunk_ids, n_tags, mask=mask)

        target_ids = input_ids[1:] + [IGNORE_INDEX]
        target_mask = mask[1:] + [0]
        target_ids = [tid if m else IGNORE_INDEX for tid, m in zip(target_ids, target_mask)]

        blank_id = _MockHFTokenizer.BLANK_ID
        footer_id = _MockHFTokenizer.FOOTER
        for i in range(len(input_ids)):
            if input_ids[i] != AUDIO_TOKEN_IDX:
                continue
            next_is_audio = i + 1 < len(input_ids) and input_ids[i + 1] == AUDIO_TOKEN_IDX
            target_ids[i] = blank_id if next_is_audio else footer_id

        # Find trailing silence: audio positions in the last user turn
        # (after the last assistant turn)
        last_asst_idx = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)
        # Trailing silence audio positions should all be blank (not footer)
        # since there's no word boundary after them (except the very last one
        # which transitions to the next non-audio token)
        trailing_audio = []
        in_trailing = False
        for i in range(len(input_ids)):
            if input_ids[i] == AUDIO_TOKEN_IDX:
                if in_trailing:
                    trailing_audio.append(i)
            else:
                in_trailing = False
        # The last contiguous run of audio tokens is the trailing silence
        last_run_start = None
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == AUDIO_TOKEN_IDX:
                last_run_start = i
            elif last_run_start is not None:
                break
        if last_run_start is not None:
            trailing_positions = [i for i in range(last_run_start, len(input_ids)) if input_ids[i] == AUDIO_TOKEN_IDX]
            # All trailing except the very last should be blank
            for i in trailing_positions[:-1]:
                assert target_ids[i] == blank_id, f"Trailing position {i} should be blank"


# ===========================================================================
# Tests: _tokenize_with_assistant_mask with real HF tokenizers
# ===========================================================================


def _try_load_tokenizer(model_id):
    """Try to load a HF tokenizer, return None if unavailable."""
    try:
        from transformers import AutoTokenizer as HFAutoTokenizer

        tok = HFAutoTokenizer.from_pretrained(model_id)
        if getattr(tok, "chat_template", None):
            return tok
    except Exception:
        pass
    # Fallback: try AutoProcessor (e.g. Gemma-4 multimodal)
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)
        tok = processor.tokenizer
        if getattr(tok, "chat_template", None):
            return tok
    except Exception:
        pass
    return None


def _make_nemo_tok(hf_tok):
    """Wrap an HF tokenizer to mimic NeMo AutoTokenizer interface."""

    class _Wrapper:
        def __init__(self, hf):
            self.tokenizer = hf

    return _Wrapper(hf_tok)


# Model IDs to test
_REAL_TOKENIZER_MODELS = {
    "qwen3": "Qwen/Qwen3-1.7B",
    "nemotron_mini": "nvidia/Nemotron-Mini-4B-Instruct",
    "nemotron_nano_v3": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "lightning35": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "gemma4": "google/gemma-4-E4B-it",
}

# Backbones whose chat template emits a system block unconditionally. Their
# per-turn user header therefore does NOT contain one, while the legacy
# string-splitting parser folded an empty system block into it — see
# TestParseChatTemplateSpans.test_matches_legacy_parser_where_legacy_was_correct.
_SYSTEM_BLOCK_BACKBONES = {"nemotron_mini", "nemotron_nano_v3", "lightning35"}


def _run_mask_test(hf_tok, messages):
    """Run _tokenize_with_assistant_mask and return (input_ids, mask, decoded_trained)."""
    nemo_tok = _make_nemo_tok(hf_tok)
    # Add <blank> if not in vocab
    if "<blank>" not in hf_tok.get_vocab():
        hf_tok.add_special_tokens({"additional_special_tokens": ["<blank>"]})
    input_ids, mask = _tokenize_with_assistant_mask(messages, nemo_tok)
    trained = [hf_tok.decode([input_ids[i]]) for i in range(len(input_ids)) if mask[i]]
    return input_ids, mask, trained


@pytest.fixture(params=list(_REAL_TOKENIZER_MODELS.keys()))
def real_tokenizer(request):
    """Parametrized fixture that yields (label, hf_tok) for each available model."""
    label = request.param
    model_id = _REAL_TOKENIZER_MODELS[label]
    hf_tok = _try_load_tokenizer(model_id)
    if hf_tok is None:
        pytest.skip(f"Tokenizer {model_id} not available")
    return label, hf_tok


class TestResolvePadId:
    """Some LLM tokenizers ship without a pad token (e.g. NVIDIA-Nemotron-3-Nano:
    ``pad_id`` None, ``unk_id`` 0). Feeding that None to ``pad_sequence`` raises,
    and the dataset's padding value must match the model's ``text_pad_id``,
    which is how the attention mask is derived."""

    @staticmethod
    def _tok(pad_id, unk_id):
        return SimpleNamespace(pad_id=pad_id, unk_id=unk_id)

    def test_uses_pad_id_when_present(self):
        assert resolve_pad_id(self._tok(pad_id=7, unk_id=0)) == 7

    def test_falls_back_to_unk(self):
        assert resolve_pad_id(self._tok(pad_id=None, unk_id=0)) == 0

    def test_falls_back_to_zero_with_warning(self):
        with pytest.warns(UserWarning, match="no <pad> or <unk> token"):
            assert resolve_pad_id(self._tok(pad_id=None, unk_id=None)) == 0

    def test_tolerates_tokenizer_without_unk_attribute(self):
        with pytest.warns(UserWarning):
            assert resolve_pad_id(SimpleNamespace(pad_id=None)) == 0

    def test_collation_pads_with_the_resolved_id(self):
        """A padless tokenizer must not crash collation, and the pad value used
        must be the resolved one (not None)."""
        pad = resolve_pad_id(self._tok(pad_id=None, unk_id=0))
        collated = right_collate_vectors(
            [torch.tensor([5, 6, 7]), torch.tensor([8])],
            padding_value=pad,
        )
        assert collated[1].tolist() == [8, pad, pad]


class TestApplyChatTemplateIds:
    """``apply_chat_template(tokenize=True)`` returns ``list[int]`` on transformers 4.x
    but a ``BatchEncoding`` on 5.x. ``apply_chat_template_ids`` must flatten both to
    the same ``list[int]`` — otherwise callers iterate the mapping's *keys* and blow up
    later with "too many dimensions 'str'"."""

    MESSAGES = [{"role": "system", "content": "hi"}]

    class _FakeTok:
        """Returns whichever shape a given transformers version would."""

        IDS = [1, 2, 3]

        def __init__(self, shape):
            self.shape = shape
            self.seen_kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.seen_kwargs = kwargs
            if self.shape == "list":  # transformers 4.x
                return list(self.IDS)
            if self.shape == "dict":  # transformers 5.x
                return {"input_ids": list(self.IDS), "attention_mask": [1, 1, 1]}
            if self.shape == "batched_dict":  # hypothetical future batching
                return {"input_ids": [list(self.IDS)], "attention_mask": [[1, 1, 1]]}
            if self.shape == "tensor_dict":
                import torch

                return {"input_ids": torch.tensor(self.IDS)}
            raise AssertionError(self.shape)

    @pytest.mark.parametrize("shape", ["list", "dict", "batched_dict", "tensor_dict"])
    def test_all_return_shapes_flatten_to_ids(self, shape):
        tok = self._FakeTok(shape)
        assert apply_chat_template_ids(tok, self.MESSAGES) == [1, 2, 3]

    def test_forwards_kwargs_and_forces_tokenize(self):
        tok = self._FakeTok("dict")
        apply_chat_template_ids(tok, self.MESSAGES, add_generation_prompt=False, enable_thinking=False)
        assert tok.seen_kwargs == {"tokenize": True, "add_generation_prompt": False, "enable_thinking": False}

    def test_matches_the_raw_call_on_this_transformers_version(self, real_tokenizer):
        """Whatever the installed version returns, the helper yields the same IDs."""
        _, hf_tok = real_tokenizer
        raw = hf_tok.apply_chat_template(self.MESSAGES, tokenize=True, add_generation_prompt=False)
        expected = list(raw["input_ids"]) if hasattr(raw, "keys") else list(raw)
        assert apply_chat_template_ids(hf_tok, self.MESSAGES, add_generation_prompt=False) == expected


class TestTokenizeWithAssistantMaskRealTokenizers:
    """Verify the diff-based fallback works with real HF tokenizers."""

    def test_single_turn_content_masked(self, real_tokenizer):
        """Content tokens should be masked in a single-turn (offline) message."""
        label, hf_tok = real_tokenizer
        messages = [
            {"role": "system", "content": "Transcribe the audio into text."},
            {"role": "user", "content": "<audio><audio><audio>"},
            {"role": "assistant", "content": "hello world"},
        ]
        input_ids, mask, trained = _run_mask_test(hf_tok, messages)

        assert any("hello" in t.lower() for t in trained), f"[{label}] 'hello' not in trained: {trained}"
        assert any("world" in t.lower() for t in trained), f"[{label}] 'world' not in trained: {trained}"

    def test_single_turn_eos_masked(self, real_tokenizer):
        """EOS token should be masked if it exists in the footer."""
        label, hf_tok = real_tokenizer
        messages = [
            {"role": "system", "content": "Transcribe the audio into text."},
            {"role": "user", "content": "<audio><audio><audio>"},
            {"role": "assistant", "content": "hello world"},
        ]
        input_ids, mask, trained = _run_mask_test(hf_tok, messages)
        eos_id = hf_tok.eos_token_id

        if eos_id is not None and eos_id in input_ids:
            # Find EOS positions that follow content
            content_positions = [i for i, m in enumerate(mask) if m]
            if content_positions:
                last_content = max(content_positions)
                # Check if EOS right after content is masked
                for i in range(last_content, min(last_content + 3, len(input_ids))):
                    if input_ids[i] == eos_id:
                        assert mask[i] == 1, f"[{label}] EOS at position {i} should be masked"
                        break

    def test_single_turn_system_not_masked(self, real_tokenizer):
        """System and user tokens should NOT be masked."""
        label, hf_tok = real_tokenizer
        messages = [
            {"role": "system", "content": "Transcribe the audio into text."},
            {"role": "user", "content": "<audio><audio><audio>"},
            {"role": "assistant", "content": "hello world"},
        ]
        input_ids, mask, trained = _run_mask_test(hf_tok, messages)

        assert not any(
            "transcribe" in t.lower() for t in trained
        ), f"[{label}] system content should not be trained: {trained}"
        assert not any(
            "audio" in t.lower() for t in trained
        ), f"[{label}] user content should not be trained: {trained}"

    def test_single_turn_think_tags_not_masked(self, real_tokenizer):
        """Qwen3-style <think> tags should NOT be masked."""
        label, hf_tok = real_tokenizer
        messages = [
            {"role": "system", "content": "Transcribe the audio into text."},
            {"role": "user", "content": "<audio><audio><audio>"},
            {"role": "assistant", "content": "hello world"},
        ]
        _, _, trained = _run_mask_test(hf_tok, messages)

        assert not any("<think>" in t for t in trained), f"[{label}] <think> should not be trained: {trained}"
        assert not any("</think>" in t for t in trained), f"[{label}] </think> should not be trained: {trained}"

    def test_multi_turn_all_contents_masked(self, real_tokenizer):
        """All assistant contents should be masked in multi-turn (streaming) messages."""
        label, hf_tok = real_tokenizer
        messages = [
            {"role": "system", "content": "Transcribe the audio into text."},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": "<blank>"},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": "world"},
        ]
        _, mask, trained = _run_mask_test(hf_tok, messages)

        assert sum(mask) > 0, f"[{label}] Should have some trained tokens"
        assert any("blank" in t.lower() for t in trained), f"[{label}] '<blank>' not in trained: {trained}"
        assert any("hello" in t.lower() for t in trained), f"[{label}] 'hello' not in trained: {trained}"
        assert any("world" in t.lower() for t in trained), f"[{label}] 'world' not in trained: {trained}"


# ===========================================================================
# Tests: compact chat template
# ===========================================================================
class TestSingleTokenAudioTag:
    """M2: a single-token audio tag replaces the whole-chunk matcher.

    The matcher exists only because a multi-token ``<audio>`` can BPE-merge with
    adjacent tags. With one id per frame the replacement is a plain map, which
    cannot mis-fire — but it must produce byte-identical token streams, and the
    legacy path must stay the default so existing checkpoints keep loading (R8).
    """

    ALIGNED = [
        WordAlignment(text=w, start_time=0.35 * i, end_time=0.35 * i + 0.30)
        for i, w in enumerate(["the", "quick", "brown", "fox", "jumps", "over"])
    ]
    TRANSCRIPT = "the quick brown fox jumps over"

    def _dataset(self, hf_tok_id, audio_tag, register):
        from nemo.collections.common.tokenizers import AutoTokenizer as NeMoTok

        tok = NeMoTok(hf_tok_id, use_fast=True)
        tok.add_special_tokens({"additional_special_tokens": ["<blank>"]})
        if register:
            tok.add_special_tokens({"additional_special_tokens": [audio_tag]})
        cfg = OmegaConf.create(
            {
                "sample_rate": 16000,
                "frame_length_in_secs": 0.08,
                "chunk_size": [2, 4, 7, 14],
                "num_delay_frames": 3,
                "audio_tag": audio_tag,
                "blank_token": "<blank>",
                "system_role": "system",
                "system_prompt": "Transcribe the audio into text.",
                "compact_template": True,
            }
        )
        return StreamingSTTDataset(cfg=cfg, tokenizer=tok), tok

    def _stream(self, ds, tok, chunk_size):
        msgs = get_llm_messages_for_sample(
            system_role="system",
            system_prompt="Transcribe the audio into text.",
            audio_tag=ds.cfg.audio_tag,
            blank_token="<blank>",
            chunk_size=chunk_size,
            num_delay_frames=3,
            audio_duration_secs=2.4,
            frame_length_in_secs=0.08,
            alignments=self.ALIGNED,
            transcript=self.TRANSCRIPT,
        )
        ids, mask = _tokenize_compact_with_assistant_mask(msgs, tok, ds._eoa_id, ds._compact_eos_id)
        if ds._audio_token_id is not None:
            return [AUDIO_TOKEN_IDX if t == ds._audio_token_id else t for t in ids]
        ids, _ = _replace_audio_chunks(ids, ds._audio_chunk_ids_by_size[chunk_size], chunk_size, mask=mask)
        return ids

    @pytest.mark.parametrize("chunk_size", [2, 4, 7, 14])
    def test_streams_are_identical_to_the_whole_chunk_matcher(self, chunk_size):
        """The whole point: this is a refactor, not a change to what the model sees."""
        legacy, tok_l = self._dataset("Qwen/Qwen3-1.7B", "<audio>", register=False)
        fast, tok_f = self._dataset("Qwen/Qwen3-1.7B", "<|_audio_placeholder_|>", register=True)
        assert legacy._audio_token_id is None, "multi-token tag must use the matcher"
        assert fast._audio_token_id is not None, "single-token tag must use the map"
        assert self._stream(legacy, tok_l, chunk_size) == self._stream(fast, tok_f, chunk_size)

    def test_per_size_cache_is_retired_on_the_fast_path(self):
        fast, _ = self._dataset("Qwen/Qwen3-1.7B", "<|_audio_placeholder_|>", register=True)
        legacy, _ = self._dataset("Qwen/Qwen3-1.7B", "<audio>", register=False)
        assert fast._audio_chunk_ids_by_size == {}
        assert sorted(legacy._audio_chunk_ids_by_size) == [2, 4, 7, 14]

    def test_legacy_tag_is_untouched_by_default(self):
        """R8: no recipe sets model.audio_tag, so the default path must not move."""
        legacy, _ = self._dataset("Qwen/Qwen3-1.7B", "<audio>", register=False)
        assert legacy._audio_token_id is None
        assert legacy.cfg.audio_tag == "<audio>"


class TestParseChatTemplateSpans:
    """M1 acceptance: the index-derived turn spans are the ones training emits.

    ``parse_chat_template_ids`` locates the four structural spans of a
    mid-conversation turn by diffing renders, replacing a parser that split the
    rendered template string and re-encoded the fragments.  The property that
    matters is not "matches the old parser" but "matches what
    ``apply_chat_template`` actually emits for a turn during training" — the two
    differ on every backbone whose template emits a system block unconditionally.
    """

    @pytest.fixture(scope="class")
    def qwen3_hf(self):
        tok = _try_load_tokenizer("Qwen/Qwen3-1.7B")
        if tok is None:
            pytest.skip("Qwen3 tokenizer not available")
        return tok

    @staticmethod
    def _count(haystack, needle):
        if not needle:
            return -1
        return sum(1 for i in range(len(haystack) - len(needle) + 1) if haystack[i : i + len(needle)] == needle)

    @staticmethod
    def _training_render(hf_tok, n_turns=3, content="<audio><audio>"):
        """A realistic training conversation: one system turn, then N user/assistant pairs."""
        messages = [{"role": "system", "content": "Transcribe the audio into text."}]
        for i in range(n_turns):
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": f"word{i}"})
        return apply_chat_template_ids(hf_tok, messages, add_generation_prompt=False, enable_thinking=False)

    def test_tokenizer_matrix_is_actually_populated(self):
        """Guard against the whole class reporting green because every load failed.

        ``real_tokenizer`` skips silently when a tokenizer cannot be loaded, so a
        cold HF cache would otherwise turn these acceptance tests into no-ops.
        """
        loaded = [m for m in _REAL_TOKENIZER_MODELS.values() if _try_load_tokenizer(m) is not None]
        assert len(loaded) >= 2, f"Need >=2 real tokenizers to make this suite meaningful, got {loaded}"

    @pytest.mark.parametrize("last_turn", [False, True])
    def test_spans_occur_in_a_real_training_render(self, real_tokenizer, last_turn):
        """The invariant that matters: inference feeds exactly what training emitted.

        This is what catches a header carrying a spurious empty system block —
        such a header appears zero times in a real conversation, so inference
        would inject it once per chunk while training never did.
        """
        label, hf_tok = real_tokenizer
        uh, uf, ah, _ = parse_chat_template_ids(hf_tok, last_turn=last_turn, probe_content="<audio>")
        render = self._training_render(hf_tok, n_turns=3)

        assert self._count(render, uh) == 3, f"{label}: user_header occurs {self._count(render, uh)}x, want 3"
        if not last_turn:
            # The assistant header is only guaranteed to match mid-conversation
            # turns; the last-turn variant may carry thinking-suppression tags.
            assert self._count(render, uf + ah) == 3, f"{label}: user_footer+asst_header should occur 3x"

    def test_content_slice_decodes_back_to_the_probe(self, real_tokenizer):
        """Boundaries are exact: what sits between the spans is the content, nothing more."""
        label, hf_tok = real_tokenizer
        uh, uf, _, _ = parse_chat_template_ids(hf_tok, probe_content="<audio>")
        sys_msg = {"role": "system", "content": "probe"}
        s_ids = apply_chat_template_ids(hf_tok, [sys_msg], add_generation_prompt=False, enable_thinking=False)
        a_ids = apply_chat_template_ids(
            hf_tok,
            [sys_msg, {"role": "user", "content": "<audio>"}],
            add_generation_prompt=False,
            enable_thinking=False,
        )
        content = a_ids[len(s_ids) + len(uh) : len(a_ids) - len(uf)]
        assert hf_tok.decode(content).strip() == "<audio>", f"{label}: content slice is {hf_tok.decode(content)!r}"

    @pytest.mark.parametrize("last_turn", [False, True])
    def test_matches_legacy_parser_where_legacy_was_correct(self, real_tokenizer, last_turn):
        """Byte-equality with the pre-M1 parser, and the exact expected divergence where not.

        Qwen3 and Gemma-4 only render a system block when a system message is
        present, so the legacy system-less probe was already correct for them and
        M1 must be a pure refactor.  The Nemotron templates always render one, so
        the legacy ``user_header`` carried an empty system block that training
        never emits — there M1 is a deliberate fix and the spans MUST differ.
        """
        label, hf_tok = real_tokenizer
        uh, uf, ah, af = parse_chat_template_ids(hf_tok, last_turn=last_turn, probe_content="<audio>")
        legacy_uh, legacy_mid, legacy_af = _legacy_parse_chat_template_ids(hf_tok, last_turn=last_turn)

        assert af == legacy_af, f"{label}: assistant footer changed"

        if label == "nemotron_mini":
            # Third behaviour change, pinned rather than waved through. The legacy
            # parser re-encoded a split string fragment, which prepended a spurious
            # SentencePiece empty-string token to the boundary span. Dropping it
            # moves `_user_footer_first_id` — the dynamic-chunking supervised target
            # and the inference emit-gate id — from 252303 to 1014 ('\n'), so a
            # model trained before this change would not fire its gate after it.
            assert legacy_mid[0] == 252303, "expected the legacy spurious empty-string token"
            assert hf_tok.decode([252303]) == "", "252303 should decode to the empty string"
            assert uf + ah == legacy_mid[1:], f"{label}: boundary span should be the legacy one minus 252303"
            assert (uf or ah)[0] == 1014, f"{label}: emit-gate id should now be the newline byte"
        else:
            assert uf + ah == legacy_mid, f"{label}: boundary span changed unexpectedly"

        if label in _SYSTEM_BLOCK_BACKBONES:
            assert uh != legacy_uh, f"{label}: expected the empty-system-block header to be fixed"
            assert self._count(legacy_uh, uh) == 1, f"{label}: fixed header should be a suffix of the legacy one"
            assert len(legacy_uh) > len(uh), f"{label}: legacy header should be the longer, buggy one"
        else:
            assert uh == legacy_uh, f"{label}: M1 must be a pure refactor for {label}"

    def test_probe_content_does_not_affect_the_spans(self, real_tokenizer):
        """The derivation must not depend on how much content the probe carries."""
        label, hf_tok = real_tokenizer
        one = parse_chat_template_ids(hf_tok, probe_content="<audio>")
        many = parse_chat_template_ids(hf_tok, probe_content="<audio>" * 13)
        assert one == many, f"{label}: spans changed with probe length"

    def test_system_folding_template_raises(self, qwen3_hf):
        """A template that folds the system message into the first user turn must fail loudly.

        This is the case the append-only guard cannot see: the system-only render
        carries nothing to exclude, so ``A[:len(S)] == S`` passes trivially and the
        probe system prompt would land inside ``user_header`` — which streaming
        inference then re-feeds once per chunk. Only the post-condition catches it.
        """

        class _SystemFoldingTokenizer:
            """Renders system content inline in the first user turn, ChatML-style."""

            name_or_path = "fake/system-folding-template"

            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def apply_chat_template(self, messages, **kwargs):
                sys_txt = "".join(m["content"] for m in messages if m["role"] == "system")
                text = ""
                for m in messages:
                    if m["role"] == "system":
                        continue
                    body = m["content"]
                    if m["role"] == "user" and sys_txt:
                        body = f"{sys_txt}\n{body}"
                        sys_txt = ""
                    text += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
                ids = self._inner.encode(text, add_special_tokens=False)
                return {"input_ids": ids} if kwargs.get("return_dict") else ids

        with pytest.raises(ValueError, match="folds the system message"):
            parse_chat_template_ids(_SystemFoldingTokenizer(qwen3_hf), probe_content="<audio>")

    def test_degenerate_probe_content_raises(self, qwen3_hf):
        """Content the template strips away leaves nothing to diff against."""
        with pytest.raises(ValueError, match="did not survive rendering"):
            parse_chat_template_ids(qwen3_hf, probe_content="")

    def test_qwen3_golden_ids(self, qwen3_hf):
        """Drift canary: the exact Qwen3 spans this refactor must preserve."""
        im_start, im_end, newline = 151644, 151645, 198
        uh, uf, ah, af = parse_chat_template_ids(qwen3_hf, last_turn=False, probe_content="<audio>")
        assert uh == [im_start, 872, newline]  # <|im_start|>user\n
        assert uf == [im_end, newline]  # <|im_end|>\n
        assert ah == [im_start, 77091, newline]  # <|im_start|>assistant\n
        assert af == [im_end, newline]  # <|im_end|>\n

    def test_non_append_only_template_raises(self, qwen3_hf):
        """A template that rewrites earlier turns must fail loudly, not silently mis-slice."""

        class _RewritingTokenizer:
            name_or_path = "fake/rewriting-template"

            def __init__(self, inner):
                self._inner = inner
                self._calls = 0

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def apply_chat_template(self, messages, **kwargs):
                out = self._inner.apply_chat_template(messages, **kwargs)
                self._calls += 1
                # Corrupt the head of a later render to break the prefix relation.
                if self._calls > 2 and kwargs.get("tokenize", True):
                    ids = list(out["input_ids"] if hasattr(out, "keys") else out)
                    return [ids[0] + 1] + ids[1:]
                return out

        with pytest.raises(ValueError, match="not append-only"):
            parse_chat_template_ids(_RewritingTokenizer(qwen3_hf), probe_content="<audio>")


class TestCompactTemplate:
    """Tests for the compact chat template feature (Qwen3 tokenizer).

    Compact mode drops per-turn role wrapping, yielding
    ``[system_wrapped] <audio>*N <eoa> [<write>] text <eos> ...``.

    Post-refactor semantics:
      - ``<eoa>`` (end_of_audio_token, default ``<|im_start|>``) is a per-chunk
        audio->text scaffold anchor. It is force-fed at inference and is NOT
        LM-supervised in fixed chunking (``assistant_mask=0``).
      - ``<write>`` (write_token) is the start-of-text emit gate. It is prepended
        to non-blank content only when ``prepend_write_token=True`` and is
        LM-supervised (``assistant_mask=1``) — the model predicts it as the
        binary emit decision at the ``<eoa>`` position's logits.
      - trailing ``<eos>`` is LM-supervised.
    """

    @pytest.fixture
    def qwen3_hf(self):
        try:
            from transformers import AutoTokenizer as HFAutoTokenizer

            return HFAutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        except Exception:
            pytest.skip("Qwen3 tokenizer not available")

    @pytest.fixture
    def qwen3_tok(self, qwen3_hf):
        class _Wrapper:
            def __init__(self, hf):
                self.tokenizer = hf

        return _Wrapper(qwen3_hf)

    def test_build_compact_turn_markers_qwen3(self, qwen3_hf):
        """Default <|im_start|> → 1-token header; <|im_end|> → 1-token footer."""
        uh, uf, ah, af = build_compact_turn_markers(qwen3_hf, "<|im_start|>")
        assert uh == []
        im_start_id = qwen3_hf.convert_tokens_to_ids("<|im_start|>")
        im_end_id = qwen3_hf.eos_token_id
        # The end-of-audio anchor is the USER FOOTER (the audio->text boundary),
        # with an empty assistant header — not the other way round. This is what
        # `_user_footer_first_id` must point at on both the training and the
        # inference side.
        assert uf == [im_start_id]
        assert ah == []
        assert af == [im_end_id]

    def test_build_compact_turn_markers_multi_token_raises(self, qwen3_hf):
        """A write_token that tokenizes to >1 piece should fail loudly."""
        with pytest.raises(ValueError, match="must encode to exactly 1 token"):
            build_compact_turn_markers(qwen3_hf, "this is definitely not one token")

    def test_tokenize_compact_structure(self, qwen3_tok):
        """Sequence shape: [system_wrapped] [<audio>*N <|im_start|> text <|im_end|>] * K."""
        hf = qwen3_tok.tokenizer
        write_id = hf.convert_tokens_to_ids("<|im_start|>")
        eos_id = hf.eos_token_id
        messages = [
            {"role": "system", "content": "Transcribe."},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": "world"},
        ]
        input_ids, mask = _tokenize_compact_with_assistant_mask(messages, qwen3_tok, write_id, eos_id)

        # Two turns → write_id and eos_id should each appear at least twice.
        assert input_ids.count(write_id) >= 2
        assert input_ids.count(eos_id) >= 2

        # Every write_id and (each turn's trailing) eos_id should be mask=1.
        # Also "hello"/"world" tokens should be mask=1.
        hello_ids = hf.encode("hello", add_special_tokens=False)
        world_ids = hf.encode("world", add_special_tokens=False)
        # Find first hello token position and check mask
        for hid in hello_ids:
            if hid in input_ids:
                assert mask[input_ids.index(hid)] == 1
        for wid in world_ids:
            if wid in input_ids:
                assert mask[input_ids.index(wid)] == 1

    def test_tokenize_compact_assistant_mask(self, qwen3_tok):
        """Mask=1 on text/eos; mask=0 on the <eoa> scaffold, system wrapping, and audio."""
        hf = qwen3_tok.tokenizer
        eoa_id = hf.convert_tokens_to_ids("<|im_start|>")
        eos_id = hf.eos_token_id
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content": "X"},
        ]
        input_ids, mask = _tokenize_compact_with_assistant_mask(messages, qwen3_tok, eoa_id, eos_id)

        # Final token should be the trailing EOS with mask=1.
        assert input_ids[-1] == eos_id
        assert mask[-1] == 1

        # User content (the audio tag tokens) must have mask=0. Verify each audio
        # BPE piece appears at least once with mask=0 (at the user-content occurrence).
        audio_ids = hf.encode("<audio>", add_special_tokens=False)
        for aid in audio_ids:
            positions = [i for i, t in enumerate(input_ids) if t == aid]
            assert any(mask[p] == 0 for p in positions), f"audio token {aid} should appear with mask=0"

        # The <eoa> scaffold marker sits immediately after the audio run and is
        # NOT LM-supervised (mask=0). With eoa defaulting to <|im_start|>, it now
        # shares its id with the system-wrapping token — so EVERY <|im_start|>
        # occurrence must be mask=0 (no inserted marker is trainable anymore).
        eoa_positions = [i for i, t in enumerate(input_ids) if t == eoa_id]
        assert eoa_positions, "expected at least one <eoa>/<|im_start|> occurrence"
        assert all(mask[p] == 0 for p in eoa_positions), "<eoa> scaffold must have mask=0"

        # The assistant content token "X" must be trainable (mask=1).
        x_ids = hf.encode("X", add_special_tokens=False)
        for xid in x_ids:
            positions = [i for i, t in enumerate(input_ids) if t == xid]
            assert any(mask[p] == 1 for p in positions), "assistant content must have mask=1"

    def test_tokenize_compact_no_trailing_asst(self, qwen3_tok):
        """A trailing user-only turn (no asst) should not append a write/eos pair."""
        hf = qwen3_tok.tokenizer
        write_id = hf.convert_tokens_to_ids("<|im_start|>")
        eos_id = hf.eos_token_id
        messages = [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "<audio>"},  # trailing user-only
        ]
        input_ids, mask = _tokenize_compact_with_assistant_mask(messages, qwen3_tok, write_id, eos_id)
        # Sequence should end with the audio token of the last user-only turn
        # (not with write/eos). Count: one paired (user+asst) + one orphan user.
        assert input_ids.count(write_id) == 1 + input_ids[
            : input_ids.index(hf.encode("<audio>", add_special_tokens=False)[0])
        ].count(write_id)

    def test_tokenize_compact_mask_length_matches_ids(self, qwen3_tok):
        """assistant_mask must be parallel to input_ids."""
        hf = qwen3_tok.tokenizer
        write_id = hf.convert_tokens_to_ids("<|im_start|>")
        eos_id = hf.eos_token_id
        messages = [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "<audio><audio><audio>"},
            {"role": "assistant", "content": "hello world"},
        ]
        input_ids, mask = _tokenize_compact_with_assistant_mask(messages, qwen3_tok, write_id, eos_id)
        assert len(input_ids) == len(mask)
        assert all(m in (0, 1) for m in mask)

    def test_tokenize_compact_empty_blank_combo(self, qwen3_tok):
        """Verify compact_template=True + blank_token="" combination.

        When blank_token="", silent chunks produce empty assistant content.
        Expected shape per silent chunk: ``<audio>*N <eoa> <eos>`` (the <eoa>
        scaffold then an immediate close-of-turn). Post-refactor the <eoa>
        marker is NOT trainable (mask=0); only the trailing <eos> is trainable.
        """
        hf = qwen3_tok.tokenizer
        eoa_id = hf.convert_tokens_to_ids("<|im_start|>")
        eos_id = hf.eos_token_id
        # Mix of silent (empty content) and non-silent chunks — mirrors what
        # get_llm_messages_for_sample produces when blank_token="" is used
        # with fixed chunking.
        messages = [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": ""},  # silent chunk
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": "hello"},  # chunk with word
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": ""},  # silent chunk
        ]
        input_ids, mask = _tokenize_compact_with_assistant_mask(messages, qwen3_tok, eoa_id, eos_id)

        assert len(input_ids) == len(mask)

        # Locate the *inserted* <eoa> markers: an <|im_start|> immediately preceded
        # by an audio-tag token (distinguishes them from the system-wrapping one).
        audio_piece_ids = set(hf.encode("<audio>", add_special_tokens=False))
        eoa_positions = [
            i for i, t in enumerate(input_ids) if t == eoa_id and i > 0 and input_ids[i - 1] in audio_piece_ids
        ]
        assert len(eoa_positions) == 3, "one <eoa> per turn (3 turns)"

        # The <eoa> scaffold is force-fed, not predicted → mask=0 everywhere.
        for pos in eoa_positions:
            assert mask[pos] == 0, "<eoa> scaffold marker must have mask=0"

        # For each silent chunk, <eos> immediately follows <eoa> and IS trainable.
        adjacent_pairs = 0
        for pos in eoa_positions:
            if pos + 1 < len(input_ids) and input_ids[pos + 1] == eos_id:
                adjacent_pairs += 1
                assert mask[pos + 1] == 1, "adjacent <eos> in silent chunk must be trainable"
        assert adjacent_pairs == 2, "two silent chunks should each produce adjacent <eoa><eos>"

        # The middle ("hello") chunk: content tokens between <eoa> and <eos> are trainable.
        hello_ids = set(hf.encode("hello", add_special_tokens=False))
        trainable_content = [input_ids[i] for i in range(len(input_ids)) if mask[i] == 1 and input_ids[i] in hello_ids]
        assert len(trainable_content) > 0, "hello content tokens must be trainable"

        # Sanity: trailing token is the per-turn eos (chunk separator / decode splitter).
        assert input_ids[-1] == eos_id


# ===========================================================================
# Tests: write_token refactor — unified emit gate + dedicated end_of_audio_token
# ===========================================================================
class TestCompactWriteTokenRefactor:
    """Post-refactor contract:

      - ``write_token`` is the start-of-text emit gate with ONE meaning in both
        compact and non-compact modes: prepended to non-blank assistant content
        only, gated by ``prepend_write_token``. It is LM-supervised.
      - ``end_of_audio_token`` (default ``<|im_start|>``) is the compact per-chunk
        scaffold anchor: force-fed, NOT LM-supervised in fixed chunking (mask=0).

    Compact wire format (gate on):
        text chunk:  [audio*N] <eoa> <write> Hello <eos>
        blank chunk: [audio*N] <eoa> <eos>
    """

    WRITE_TOKEN = "<|write|>"

    @pytest.fixture
    def qwen3_hf(self):
        try:
            from transformers import AutoTokenizer as HFAutoTokenizer

            hf = HFAutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        except Exception:
            pytest.skip("Qwen3 tokenizer not available")
        # The model registers write_token as a new special token at init; mirror
        # that here so it encodes to a single id (like <blank>, no warm start).
        if hf.convert_tokens_to_ids(self.WRITE_TOKEN) == hf.unk_token_id:
            hf.add_special_tokens({"additional_special_tokens": [self.WRITE_TOKEN]})
        return hf

    @pytest.fixture
    def qwen3_tok(self, qwen3_hf):
        class _Wrapper:
            def __init__(self, hf):
                self.tokenizer = hf

            def ids_to_tokens(self, ids):
                return self.tokenizer.convert_ids_to_tokens(ids)

            def tokens_to_text(self, tokens, remove_special_tokens=True):
                if remove_special_tokens:
                    tokens = [t for t in tokens if t not in self.tokenizer.all_special_tokens]
                return self.tokenizer.convert_tokens_to_string(tokens)

        return _Wrapper(qwen3_hf)

    # --- message builder: emit gate is prepend-to-nonblank-only, template-agnostic ---

    def test_prepend_write_token_only_on_nonblank_content(self):
        """write_token prefixes non-blank assistant turns; blank turns stay bare."""
        messages = _make_messages(prepend_write_token=True, write_token=self.WRITE_TOKEN)
        asst = [m["content"] for m in messages if m["role"] == "assistant"]
        assert any(c == BLANK_TOKEN for c in asst), "expected some blank turns"
        for c in asst:
            if c == BLANK_TOKEN:
                assert not c.startswith(self.WRITE_TOKEN), "blank turn must NOT get a write prefix"
            else:
                assert c.startswith(self.WRITE_TOKEN), "non-blank turn must start with write_token"

    def test_prepend_write_token_off_leaves_content_bare(self):
        messages = _make_messages(prepend_write_token=False, write_token=self.WRITE_TOKEN)
        asst = [m["content"] for m in messages if m["role"] == "assistant"]
        assert all(not c.startswith(self.WRITE_TOKEN) for c in asst)

    # --- compact tokenizer: gate-on layout <eoa>(0) <write>(1) text(1) <eos>(1) ---

    def test_compact_gate_on_layout(self, qwen3_tok):
        hf = qwen3_tok.tokenizer
        eoa_id = hf.convert_tokens_to_ids("<|im_start|>")
        write_id = hf.convert_tokens_to_ids(self.WRITE_TOKEN)
        eos_id = hf.eos_token_id
        # Message content as produced by get_llm_messages_for_sample with the gate on.
        messages = [
            {"role": "system", "content": "S."},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": self.WRITE_TOKEN + "hello"},  # text chunk
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": ""},  # blank chunk (blank_token="")
        ]
        input_ids, mask = _tokenize_compact_with_assistant_mask(messages, qwen3_tok, eoa_id, eos_id)
        assert len(input_ids) == len(mask)

        audio_piece_ids = set(hf.encode("<audio>", add_special_tokens=False))
        eoa_positions = [
            i for i, t in enumerate(input_ids) if t == eoa_id and i > 0 and input_ids[i - 1] in audio_piece_ids
        ]
        assert len(eoa_positions) == 2, "one <eoa> per chunk (2 chunks)"

        # Text chunk: <eoa>(mask 0) -> <write>(mask 1) -> content(mask 1).
        p_text = eoa_positions[0]
        assert mask[p_text] == 0, "<eoa> must be mask=0"
        assert input_ids[p_text + 1] == write_id, "text chunk: <write> immediately follows <eoa>"
        assert mask[p_text + 1] == 1, "<write> emit gate must be LM-supervised"

        # Blank chunk: <eoa>(mask 0) -> <eos>(mask 1), no <write> between.
        p_blank = eoa_positions[1]
        assert mask[p_blank] == 0
        assert input_ids[p_blank + 1] == eos_id, "blank chunk: <eos> immediately follows <eoa>"
        assert input_ids[p_blank + 1] != write_id, "blank chunk must NOT contain <write>"

    # --- decode_with_blank strips <write>, never renders <eoa> (absent) ---

    def test_decode_with_blank_strips_write_token(self, qwen3_tok):
        hf = qwen3_tok.tokenizer
        write_id = hf.convert_tokens_to_ids(self.WRITE_TOKEN)
        eos_id = hf.eos_token_id
        hello_ids = hf.encode("hello", add_special_tokens=False)
        # Stream as emitted at inference: <write> hello <eos> (no <eoa> — it is
        # force-fed scaffold, never appended to the generated token stream).
        ids = [write_id] + hello_ids + [eos_id]
        text = decode_with_blank(ids, blank_token="", tokenizer=qwen3_tok, write_token=self.WRITE_TOKEN)
        assert "hello" in text
        assert self.WRITE_TOKEN not in text

    # --- end-to-end get_batch_data supervision contract ---

    def _make_compact_dataset(self, qwen3_hf, *, prepend_write_token, blank_token, chunk_size=2):
        from nemo.collections.speechlm2.data.streaming_stt_dataset import StreamingSTTDataset

        if blank_token and qwen3_hf.convert_tokens_to_ids(blank_token) == qwen3_hf.unk_token_id:
            qwen3_hf.add_special_tokens({"additional_special_tokens": [blank_token]})

        class _Wrapper:
            def __init__(self, hf):
                self.tokenizer = hf
                self.pad_id = hf.pad_token_id if hf.pad_token_id is not None else 0

        cfg = {
            "sample_rate": 16000,
            "frame_length_in_secs": FRAME_LEN,
            "chunk_size": chunk_size,
            "audio_tag": AUDIO_TAG,
            "blank_token": blank_token,
            "compact_template": True,
            "prepend_write_token": prepend_write_token,
            "write_token": self.WRITE_TOKEN,
            "end_of_audio_token": "<|im_start|>",
        }
        return StreamingSTTDataset(cfg, _Wrapper(qwen3_hf))

    def test_get_batch_data_eoa_unsupervised_gate_predicts_write(self, qwen3_hf):
        """<eoa> is force-fed (target IGNORE); the emit decision (write/eos) is
        predicted AT the <eoa> position via next-token shift."""
        import torch

        from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX, IGNORE_INDEX

        ds = self._make_compact_dataset(qwen3_hf, prepend_write_token=True, blank_token="<blank>")
        eoa_id = qwen3_hf.convert_tokens_to_ids("<|im_start|>")
        write_id = qwen3_hf.convert_tokens_to_ids(self.WRITE_TOKEN)

        audios = torch.zeros(1, 16000)  # 1s -> 13 frames @ 80ms
        audio_lens = torch.tensor([16000])
        alignments = [[WordAlignment(text="hello", start_time=0.0, end_time=0.16)]]
        from types import SimpleNamespace

        batch = ds.get_batch_data([SimpleNamespace(custom={})], audios, audio_lens, alignments, ["hello"])
        inp = batch.input_tokens[0].tolist()
        tgt = batch.target_tokens[0].tolist()

        eoa_positions = [i for i, t in enumerate(inp) if t == eoa_id and i > 0 and inp[i - 1] == AUDIO_TOKEN_IDX]
        assert eoa_positions, "expected inserted <eoa> markers after audio runs"

        for p in eoa_positions:
            # The <eoa> token itself is never a prediction target (mask=0).
            assert tgt[p - 1] == IGNORE_INDEX, "position predicting <eoa> must be IGNORE"
            # The emit decision is predicted at the <eoa> position (next-token shift).
            assert tgt[p] != IGNORE_INDEX, "emit decision at <eoa> position must be supervised"

        # At least one chunk (the 'hello' chunk) predicts <write> at its <eoa> position.
        assert any(tgt[p] == write_id for p in eoa_positions), "text chunk must predict <write> as emit gate"

    def test_get_batch_data_gate_off_no_write_token(self, qwen3_hf):
        """Gate off: no <write> in the stream; <eoa> still mask=0 (unsupervised)."""
        import torch

        from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX, IGNORE_INDEX

        ds = self._make_compact_dataset(qwen3_hf, prepend_write_token=False, blank_token="")
        eoa_id = qwen3_hf.convert_tokens_to_ids("<|im_start|>")
        write_id = qwen3_hf.convert_tokens_to_ids(self.WRITE_TOKEN)

        audios = torch.zeros(1, 16000)
        audio_lens = torch.tensor([16000])
        alignments = [[WordAlignment(text="hello", start_time=0.0, end_time=0.16)]]
        from types import SimpleNamespace

        batch = ds.get_batch_data([SimpleNamespace(custom={})], audios, audio_lens, alignments, ["hello"])
        inp = batch.input_tokens[0].tolist()
        tgt = batch.target_tokens[0].tolist()

        assert write_id not in inp, "gate off: <write> must not appear in the stream"
        eoa_positions = [i for i, t in enumerate(inp) if t == eoa_id and i > 0 and inp[i - 1] == AUDIO_TOKEN_IDX]
        for p in eoa_positions:
            assert tgt[p - 1] == IGNORE_INDEX, "<eoa> is unsupervised even with the gate off"


# ---------------------------------------------------------------------------
# Tests: multi chunk-size training (chunk_size as a list) + backward
# compatibility when chunk_size is a plain integer as before.
# ---------------------------------------------------------------------------
class TestMultiChunkSizeDataset:
    """``StreamingSTTDataConfig.chunk_size`` may be a list of positive ints;
    a scalar int keeps the original single-mode behavior."""

    def _make_dataset(self, chunk_size):
        from nemo.collections.speechlm2.data.streaming_stt_dataset import StreamingSTTDataset

        cfg = {
            "sample_rate": 16000,
            "frame_length_in_secs": FRAME_LEN,
            "chunk_size": chunk_size,
            "audio_tag": AUDIO_TAG,
            "blank_token": BLANK_TOKEN,
        }
        nemo_tok = _MockNemoTokenizer(_MockHFTokenizer())
        nemo_tok.pad_id = 0  # needed by right/left_collate_vectors in get_batch_data
        return StreamingSTTDataset(cfg, nemo_tok)

    # --- __init__ normalization & precompute ---

    @staticmethod
    def _assert_audio_mapping_ready(ds, sizes):
        """Every candidate chunk size must be mappable, by whichever path applies.

        A single-token audio tag maps one id per frame and needs no per-size
        patterns; a multi-token tag needs ``audio_tag * chunk_size`` precomputed
        for each size so BPE cannot merge across adjacent tags.
        """
        if ds._audio_token_id is not None:
            assert ds._audio_chunk_ids_by_size == {}, "single-token path should not build per-size patterns"
        else:
            assert set(ds._audio_chunk_ids_by_size) == set(sizes)
            for size in sizes:
                assert len(ds._audio_chunk_ids_by_size[size]) == size

    def test_list_candidates_and_audio_ids(self):
        ds = self._make_dataset([2, 4])
        assert ds._chunk_size_candidates == [2, 4]
        self._assert_audio_mapping_ready(ds, [2, 4])

    def test_scalar_backward_compatible(self):
        ds = self._make_dataset(2)
        assert ds._chunk_size_candidates is None
        self._assert_audio_mapping_ready(ds, [2])

    def test_scalar_dynamic_and_offline_have_no_audio_ids(self):
        for cs in (0, -1):
            ds = self._make_dataset(cs)
            assert ds._chunk_size_candidates is None
            assert ds._audio_chunk_ids_by_size == {}

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            self._make_dataset([])

    def test_non_positive_in_list_raises(self):
        with pytest.raises(ValueError):
            self._make_dataset([2, 0, 4])
        with pytest.raises(ValueError):
            self._make_dataset([2, -1])

    # --- per-batch selection through get_batch_data ---

    def _run_batch(self, ds, forced_chunk_size, monkeypatch):
        """Run get_batch_data on a 1-second mono sample, forcing the random pick."""
        import torch

        import nemo.collections.speechlm2.data.streaming_stt_dataset as mod

        if forced_chunk_size is not None:
            monkeypatch.setattr(mod.random, "choice", lambda seq: forced_chunk_size)

        audios = torch.zeros(1, 16000)  # 1.0 s @ 16 kHz → 13 frames @ 80 ms
        audio_lens = torch.tensor([16000])
        alignments = [[WordAlignment(text="hello", start_time=0.0, end_time=0.16)]]
        text = ["hello"]
        from types import SimpleNamespace

        cuts = [SimpleNamespace(custom={})]
        return ds.get_batch_data(cuts, audios, audio_lens, alignments, text)

    def test_batch_records_selected_chunk_size(self, monkeypatch):
        ds = self._make_dataset([2, 4, 8])
        batch = self._run_batch(ds, forced_chunk_size=4, monkeypatch=monkeypatch)
        assert batch.chunk_size == 4
        # 13 frames, chunk_size 4 → ceil(13/4)=4 chunks → 16 audio slots.
        n_audio = int((batch.input_tokens == AUDIO_TOKEN_IDX).sum().item())
        assert n_audio == 16

    def test_batch_audio_slots_track_chunk_size(self, monkeypatch):
        ds = self._make_dataset([2, 8])
        b2 = self._run_batch(ds, forced_chunk_size=2, monkeypatch=monkeypatch)
        b8 = self._run_batch(ds, forced_chunk_size=8, monkeypatch=monkeypatch)
        n2 = int((b2.input_tokens == AUDIO_TOKEN_IDX).sum().item())
        n8 = int((b8.input_tokens == AUDIO_TOKEN_IDX).sum().item())
        assert b2.chunk_size == 2 and b8.chunk_size == 8
        assert n2 == 14  # ceil(13/2)=7 chunks * 2
        assert n8 == 16  # ceil(13/8)=2 chunks * 8

    def test_scalar_batch_backward_compatible(self, monkeypatch):
        # No random selection for a scalar config; chunk_size flows straight through.
        ds = self._make_dataset(2)
        batch = self._run_batch(ds, forced_chunk_size=None, monkeypatch=monkeypatch)
        assert batch.chunk_size == 2
        n_audio = int((batch.input_tokens == AUDIO_TOKEN_IDX).sum().item())
        assert n_audio == 14  # ceil(13/2)=7 chunks * 2


def _legacy_parse_chat_template_ids(hf_tok, last_turn: bool = False):
    """Byte-for-byte copy of ``parse_chat_template_ids`` as of commit f1e14f5e2a.

    Kept ONLY as the reference for
    ``TestParseChatTemplateSpans.test_matches_legacy_parser_where_legacy_was_correct``,
    which pins M1 as a pure refactor on the backbones where this implementation
    was already correct, and pins the exact divergence where it was not.

    Do not "fix" this function — its bugs are the point. It renders a
    system-less probe conversation, so on templates that emit a system block
    unconditionally the empty block ends up inside ``user_header_ids``.

    Returns the old 3-tuple ``(user_header, user_footer_and_asst_header, asst_footer)``.
    """
    _SENTINEL = "XSENTINELX"
    convo_2msg = hf_tok.apply_chat_template(
        [{"role": "user", "content": _SENTINEL}, {"role": "assistant", "content": _SENTINEL}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    parts = convo_2msg.split(_SENTINEL)
    user_header_ids = hf_tok.encode(parts[0], add_special_tokens=False)
    asst_footer_ids = hf_tok.encode(parts[2], add_special_tokens=False) if parts[2].strip() else []

    bos_id = getattr(hf_tok, "bos_token_id", None)
    if user_header_ids and bos_id is not None and user_header_ids[0] == bos_id:
        user_header_ids = user_header_ids[1:]

    if last_turn:
        mid_ids = hf_tok.encode(parts[1], add_special_tokens=False)
    else:
        convo_4msg = hf_tok.apply_chat_template(
            [
                {"role": "user", "content": _SENTINEL},
                {"role": "assistant", "content": _SENTINEL},
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "x"},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        mid_ids = hf_tok.encode(convo_4msg.split(_SENTINEL)[1], add_special_tokens=False)

    return user_header_ids, mid_ids, asst_footer_ids


class TestDropBlankFromContext:
    """M3 / rung 1: the blank leaves the context but stays supervised.

    The point of the knob is that the model still learns to predict blank at the
    gate — it simply never attends to a blank it emitted earlier. So the two
    properties to pin are complementary: zero blanks among the INPUT ids, and an
    unchanged count of blanks among the TARGETS.
    """

    WORDS = ["hello", "world", "again"]
    ALIGNED = [WordAlignment(text=w, start_time=0.6 * i, end_time=0.6 * i + 0.25) for i, w in enumerate(WORDS)]

    def _dataset(self, drop, blank_token="<blank>", compact=True):
        from nemo.collections.common.tokenizers import AutoTokenizer as NeMoTok

        tok = NeMoTok("Qwen/Qwen3-1.7B", use_fast=True)
        tok.add_special_tokens({"additional_special_tokens": ["<blank>", "<|write|>"]})
        cfg = OmegaConf.create(
            {
                "sample_rate": 16000,
                "frame_length_in_secs": 0.08,
                "chunk_size": 2,
                "num_delay_frames": 3,
                "audio_tag": "<audio>",
                "blank_token": blank_token,
                "system_role": "system",
                "system_prompt": "Transcribe the audio into text.",
                "compact_template": compact,
                "prepend_write_token": True,
                "write_token": "<|write|>",
                "drop_blank_from_context": drop,
            }
        )
        return StreamingSTTDataset(cfg=cfg, tokenizer=tok), tok

    def _encode(self, ds, tok):
        msgs = get_llm_messages_for_sample(
            system_role="system",
            system_prompt="Transcribe the audio into text.",
            audio_tag="<audio>",
            blank_token=ds.cfg.blank_token,
            chunk_size=2,
            num_delay_frames=3,
            audio_duration_secs=2.0,
            frame_length_in_secs=0.08,
            alignments=self.ALIGNED,
            transcript=" ".join(self.WORDS),
            prepend_write_token=True,
            write_token="<|write|>",
        )
        ids, mask = _tokenize_compact_with_assistant_mask(
            msgs,
            tok,
            ds._eoa_id,
            ds._compact_eos_id,
            drop_blank_from_context=ds._drop_blank,
            blank_token=ds.cfg.blank_token,
        )
        ids, mask = _replace_audio_chunks(ids, ds._audio_chunk_ids_by_size[2], 2, mask=mask)
        targets = [t if m else IGNORE_INDEX for t, m in zip(ids[1:] + [IGNORE_INDEX], mask[1:] + [0])]
        if ds._drop_blank:
            n = len(ids)
            for i, tid in enumerate(ids):
                if tid != ds._eoa_id:
                    continue
                nxt = ids[i + 1] if i + 1 < n else None
                if nxt is None or nxt == AUDIO_TOKEN_IDX:
                    targets[i] = ds.blank_id
        return ids, targets

    def test_blank_leaves_the_input_but_stays_supervised(self):
        off_ds, off_tok = self._dataset(drop=False)
        on_ds, on_tok = self._dataset(drop=True)
        off_ids, off_tgt = self._encode(off_ds, off_tok)
        on_ids, on_tgt = self._encode(on_ds, on_tok)

        blank = off_ds.blank_id
        assert sum(1 for t in off_ids if t == blank) > 0, "baseline should contain blanks as input"
        assert sum(1 for t in on_ids if t == blank) == 0, "blank must not be an input token"
        assert sum(1 for t in on_tgt if t == blank) == sum(
            1 for t in off_tgt if t == blank
        ), "every blank that was supervised before must still be supervised"

    def test_audio_is_untouched_and_the_sequence_shrinks(self):
        off_ds, off_tok = self._dataset(drop=False)
        on_ds, on_tok = self._dataset(drop=True)
        off_ids, _ = self._encode(off_ds, off_tok)
        on_ids, _ = self._encode(on_ds, on_tok)

        n_audio = lambda ids: sum(1 for t in ids if t == AUDIO_TOKEN_IDX)
        assert n_audio(on_ids) == n_audio(off_ids), "dropping blanks must not change the audio span"
        assert len(on_ids) < len(off_ids)

    def test_emitted_text_is_unchanged(self):
        """Only silent chunks lose scaffolding; spoken content must survive verbatim."""
        off_ds, off_tok = self._dataset(drop=False)
        on_ds, on_tok = self._dataset(drop=True)
        off_ids, _ = self._encode(off_ds, off_tok)
        on_ids, _ = self._encode(on_ds, on_tok)

        keep = lambda ids, ds: [
            t for t in ids if t not in (AUDIO_TOKEN_IDX, ds._eoa_id, ds._compact_eos_id, ds.blank_id)
        ]
        assert keep(on_ids, on_ds) == keep(off_ids, off_ds)

    def test_disabled_without_a_blank_token(self):
        """blank_token='' leaves nothing to drop and no defined gate target."""
        with pytest.warns(UserWarning, match="requires a non-empty blank_token"):
            ds, _ = self._dataset(drop=True, blank_token="")
        assert ds._drop_blank is False

    def test_non_compact_is_rejected_not_half_applied(self):
        with pytest.raises(NotImplementedError, match="compact_template=True"):
            self._dataset(drop=True, compact=False)

    def test_off_by_default(self):
        ds, _ = self._dataset(drop=False)
        assert ds._drop_blank is False
        assert StreamingSTTDataConfig.drop_blank_from_context is False


class TestCollapseSilentAudio:
    """M5 / rung 2: the per-chunk anchor goes too, so silent audio becomes contiguous.

    The gate moves to the last audio frame of each chunk. Since no boundary marker
    survives in the token stream, it is located by counting audio frames — every
    chunk contributes exactly ``chunk_size``. Its target is ``blank`` when nothing
    but more audio follows, and otherwise whatever token actually starts the
    emission, so the inference-side test is ``!= blank`` and needs no fixed token.
    """

    WORDS = ["hello", "world", "again"]
    ALIGNED = [WordAlignment(text=w, start_time=0.6 * i, end_time=0.6 * i + 0.25) for i, w in enumerate(WORDS)]
    C = 2

    def _dataset(self, collapse, prepend=True, blank_token="<blank>"):
        from nemo.collections.common.tokenizers import AutoTokenizer as NeMoTok

        tok = NeMoTok("Qwen/Qwen3-1.7B", use_fast=True)
        tok.add_special_tokens({"additional_special_tokens": ["<blank>", "<|write|>"]})
        cfg = OmegaConf.create(
            {
                "sample_rate": 16000,
                "frame_length_in_secs": 0.08,
                "chunk_size": self.C,
                "num_delay_frames": 3,
                "audio_tag": "<audio>",
                "blank_token": blank_token,
                "system_role": "system",
                "system_prompt": "Transcribe the audio into text.",
                "compact_template": True,
                "prepend_write_token": prepend,
                "write_token": "<|write|>",
                "collapse_silent_audio": collapse,
            }
        )
        return StreamingSTTDataset(cfg=cfg, tokenizer=tok), tok

    def _encode(self, ds, tok, prepend=True):
        msgs = get_llm_messages_for_sample(
            system_role="system",
            system_prompt="Transcribe the audio into text.",
            audio_tag="<audio>",
            blank_token=ds.cfg.blank_token,
            chunk_size=self.C,
            num_delay_frames=3,
            audio_duration_secs=2.0,
            frame_length_in_secs=0.08,
            alignments=self.ALIGNED,
            transcript=" ".join(self.WORDS),
            prepend_write_token=prepend,
            write_token="<|write|>",
        )
        ids, mask = _tokenize_compact_with_assistant_mask(
            msgs,
            tok,
            ds._eoa_id,
            ds._compact_eos_id,
            drop_blank_from_context=ds._drop_blank,
            blank_token=ds.cfg.blank_token,
            collapse_silent_audio=ds._collapse_audio,
        )
        ids, mask = _replace_audio_chunks(ids, ds._audio_chunk_ids_by_size[self.C], self.C, mask=mask)
        targets = [t if m else IGNORE_INDEX for t, m in zip(ids[1:] + [IGNORE_INDEX], mask[1:] + [0])]
        gates = []
        if ds._collapse_audio:
            n, seen = len(ids), 0
            for i, tid in enumerate(ids):
                if tid != AUDIO_TOKEN_IDX:
                    continue
                seen += 1
                if seen % self.C:
                    continue
                nxt = ids[i + 1] if i + 1 < n else None
                targets[i] = ds.blank_id if (nxt is None or nxt == AUDIO_TOKEN_IDX) else nxt
                gates.append(targets[i])
        return ids, targets, gates

    @staticmethod
    def _longest_audio_run(ids):
        best = cur = 0
        for t in ids:
            cur = cur + 1 if t == AUDIO_TOKEN_IDX else 0
            best = max(best, cur)
        return best

    def test_silent_chunks_merge_into_one_contiguous_audio_run(self):
        """The M5 acceptance criterion."""
        off_ds, off_tok = self._dataset(collapse=False)
        on_ds, on_tok = self._dataset(collapse=True)
        off_ids, _, _ = self._encode(off_ds, off_tok)
        on_ids, _, _ = self._encode(on_ds, on_tok)

        assert self._longest_audio_run(off_ids) == self.C, "baseline audio is chunked every C frames"
        assert self._longest_audio_run(on_ids) > self.C, "consecutive silent chunks should merge"
        assert len(on_ids) < len(off_ids)

    def test_no_anchor_survives_and_audio_is_untouched(self):
        on_ds, on_tok = self._dataset(collapse=True)
        off_ds, off_tok = self._dataset(collapse=False)
        on_ids, _, _ = self._encode(on_ds, on_tok)
        off_ids, _, _ = self._encode(off_ds, off_tok)

        n_audio = lambda ids: sum(1 for t in ids if t == AUDIO_TOKEN_IDX)
        assert n_audio(on_ids) == n_audio(off_ids), "dropping anchors must not change the audio span"
        # The only <eoa> left is the system block's, which is not a per-chunk anchor.
        assert sum(1 for t in on_ids if t == on_ds._eoa_id) < sum(1 for t in off_ids if t == off_ds._eoa_id)

    @pytest.mark.parametrize("prepend", [True, False])
    def test_gate_target_is_never_blank_on_a_speaking_chunk(self, prepend):
        """The gate needs no fixed token: 'not blank' identifies emission either way.

        With prepend_write_token the emit target is the write token; without it, the
        first text token. Both are distinguishable from blank, which is all the
        inference-side gate tests.
        """
        ds, tok = self._dataset(collapse=True, prepend=prepend)
        _, _, gates = self._encode(ds, tok, prepend=prepend)

        assert gates, "expected at least one gate"
        emit = [g for g in gates if g != ds.blank_id]
        blank = [g for g in gates if g == ds.blank_id]
        assert emit and blank, "both silent and speaking chunks should be represented"
        if prepend:
            write_id = tok.tokenizer.convert_tokens_to_ids("<|write|>")
            assert all(g == write_id for g in emit)
        else:
            assert all(g != ds.blank_id for g in emit)

    def test_supervision_count_matches_the_unmodified_format(self):
        """Rung 2 changes the context, never how many decisions are supervised."""
        off_ds, off_tok = self._dataset(collapse=False)
        on_ds, on_tok = self._dataset(collapse=True)
        _, off_tgt, _ = self._encode(off_ds, off_tok)
        _, on_tgt, on_gates = self._encode(on_ds, on_tok)

        assert sum(1 for t in on_tgt if t == on_ds.blank_id) == sum(1 for t in off_tgt if t == off_ds.blank_id)

    def test_requires_a_blank_token(self):
        with pytest.raises(ValueError, match="requires a non-empty blank_token"):
            self._dataset(collapse=True, blank_token="")

    def test_implies_rung_1(self):
        ds, _ = self._dataset(collapse=True)
        assert ds._drop_blank is True, "collapse_silent_audio implies drop_blank_from_context"

    def test_off_by_default(self):
        assert StreamingSTTDataConfig.collapse_silent_audio is False


class TestFlushToken:
    """``use_flush_token``: replace the unobservable last chunk with an observable marker.

    Training makes the final chunk special twice over -- ``is_last_chunk`` force-emits a
    sub-``words_per_group`` buffer, and delay-pushed residual words are folded back into
    the last assistant turn -- but at inference the model cannot tell which chunk is last,
    because audio simply stops. With the knob on, every boundary obeys one rule and
    everything left over moves to a flush turn the harness actually feeds.
    """

    WORDS = ["hello", "world", "again"]
    # The last word ends at 1.95s: frame 25, +3 delay = 28, past the final boundary at
    # frame 26 -- so it is genuinely residual and must be carried by the flush turn.
    ALIGNED = [
        WordAlignment(text="hello", start_time=0.00, end_time=0.25),
        WordAlignment(text="world", start_time=0.60, end_time=0.85),
        WordAlignment(text="again", start_time=1.70, end_time=1.95),
    ]
    C = 2
    FLUSH = "<|flush|>"

    def _dataset(self, use_flush, collapse=False, chunk_size=None):
        from nemo.collections.common.tokenizers import AutoTokenizer as NeMoTok

        tok = NeMoTok("Qwen/Qwen3-1.7B", use_fast=True)
        tok.add_special_tokens({"additional_special_tokens": ["<blank>", "<|write|>", self.FLUSH]})
        cfg = OmegaConf.create(
            {
                "sample_rate": 16000,
                "frame_length_in_secs": 0.08,
                "chunk_size": self.C if chunk_size is None else chunk_size,
                "num_delay_frames": 3,
                "audio_tag": "<audio>",
                "blank_token": "<blank>",
                "system_role": "system",
                "system_prompt": "Transcribe the audio into text.",
                "compact_template": True,
                "prepend_write_token": True,
                "write_token": "<|write|>",
                "collapse_silent_audio": collapse,
                "use_flush_token": use_flush,
                "flush_token": self.FLUSH,
            }
        )
        return StreamingSTTDataset(cfg=cfg, tokenizer=tok), tok

    def _messages(self, use_flush, words_per_group=1):
        return get_llm_messages_for_sample(
            system_role="system",
            system_prompt="Transcribe the audio into text.",
            audio_tag="<audio>",
            blank_token="<blank>",
            chunk_size=self.C,
            num_delay_frames=3,
            audio_duration_secs=2.0,
            frame_length_in_secs=0.08,
            alignments=self.ALIGNED,
            transcript=" ".join(self.WORDS),
            words_per_group=words_per_group,
            prepend_write_token=True,
            write_token="<|write|>",
            use_flush_token=use_flush,
            flush_token=self.FLUSH,
        )

    def _encode(self, ds, tok, messages):
        ids, mask = _tokenize_compact_with_assistant_mask(
            messages,
            tok,
            ds._eoa_id,
            ds._compact_eos_id,
            drop_blank_from_context=ds._drop_blank,
            blank_token=ds.cfg.blank_token,
            collapse_silent_audio=ds._collapse_audio,
            flush_id=ds._flush_id,
        )
        ids, mask = _replace_audio_chunks(ids, ds._audio_chunk_ids_by_size[self.C], self.C, mask=mask)
        targets = [t if m else IGNORE_INDEX for t, m in zip(ids[1:] + [IGNORE_INDEX], mask[1:] + [0])]
        # Mirrors the rung-2 gate loop in get_batch_data, including its flush case.
        if ds._collapse_audio:
            n, seen = len(ids), 0
            for i, tid in enumerate(ids):
                if tid != AUDIO_TOKEN_IDX:
                    continue
                seen += 1
                if seen % self.C:
                    continue
                nxt = ids[i + 1] if i + 1 < n else None
                if nxt is None or nxt == AUDIO_TOKEN_IDX or (ds._flush_id is not None and nxt == ds._flush_id):
                    targets[i] = ds.blank_id
                else:
                    targets[i] = nxt
            if ds._flush_id is not None:
                for i, tid in enumerate(ids):
                    if tid != ds._flush_id:
                        continue
                    nxt = ids[i + 1] if i + 1 < len(ids) else None
                    targets[i] = ds.blank_id if (nxt is None or nxt == AUDIO_TOKEN_IDX) else nxt
        return ids, mask, targets

    def test_flush_turn_is_appended_and_carries_the_tail(self):
        msgs = self._messages(use_flush=True)
        assert msgs[-2]["role"] == "user" and msgs[-2]["content"] == self.FLUSH
        assert msgs[-1]["role"] == "assistant"
        # Every word must still be supervised exactly once across the whole sample.
        emitted = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
        for w in self.WORDS:
            assert emitted.count(w) == 1, f"{w!r} should appear exactly once"

    def test_last_chunk_exception_is_dropped(self):
        """The whole point: with the knob on, no chunk boundary behaves specially."""
        off = self._messages(use_flush=False, words_per_group=2)
        on = self._messages(use_flush=True, words_per_group=2)

        # Baseline: a leftover buffer below words_per_group is force-emitted at the
        # last chunk, conditioned on `is_last_chunk` -- unobservable at inference.
        assert off[-1]["role"] == "assistant" and off[-1]["content"] != "<blank>"
        # With flush: the same leftover is carried by the flush turn instead.
        assert on[-2]["content"] == self.FLUSH
        assert on[-1]["content"] != "<blank>", "the leftover words should land after flush"

    def test_no_pending_words_still_supervises_blank(self):
        """Flush must also teach 'nothing left', or it only ever means 'dump'."""
        msgs = get_llm_messages_for_sample(
            system_role="system",
            system_prompt="p",
            audio_tag="<audio>",
            blank_token="<blank>",
            chunk_size=self.C,
            num_delay_frames=0,
            audio_duration_secs=2.0,
            frame_length_in_secs=0.08,
            alignments=[WordAlignment(text="hi", start_time=0.0, end_time=0.1)],
            transcript="hi",
            prepend_write_token=True,
            write_token="<|write|>",
            use_flush_token=True,
            flush_token=self.FLUSH,
        )
        assert msgs[-2]["content"] == self.FLUSH
        assert msgs[-1]["content"] == "<blank>"

    def test_flush_turn_follows_the_rung_convention(self):
        """Flush is a prefix; the turn after it is formatted like any other turn.

        Rungs 0/1 keep the anchor, so emission still triggers off the same position it
        does at every other chunk boundary instead of a second, once-per-utterance
        trigger. Rung 2 has no anchor anywhere, so none appears here either.
        """
        ds, tok = self._dataset(use_flush=True, collapse=False)
        ids, _, _ = self._encode(ds, tok, self._messages(use_flush=True))
        i = ids.index(ds._flush_id)
        assert ids[i + 1] == ds._eoa_id, "rungs 0/1 keep the anchor on the flush turn"

        ds2, tok2 = self._dataset(use_flush=True, collapse=True)
        ids2, _, _ = self._encode(ds2, tok2, self._messages(use_flush=True))
        j = ids2.index(ds2._flush_id)
        assert ids2[j + 1] != ds2._eoa_id, "rung 2 keeps no anchor, including here"

    @pytest.mark.parametrize("collapse", [False, True])
    def test_flush_turn_emission_is_supervised(self, collapse):
        """Whatever the layout, the flush turn's emission must be trained.

        Rungs 0/1 put the anchor between marker and content, rung 2 does not, so the
        content index differs -- but in both the marker is scaffold, the content is
        supervised, and the position right before the content carries it as its target.
        """
        ds, tok = self._dataset(use_flush=True, collapse=collapse)
        ids, mask, targets = self._encode(ds, tok, self._messages(use_flush=True))
        i = ids.index(ds._flush_id)
        assert mask[i] == 0, "the marker itself is force-fed scaffold"

        content = next(k for k in range(i + 1, len(ids)) if mask[k] == 1)
        assert content - i <= 2, "content should follow the marker (plus at most an anchor)"
        assert targets[content - 1] == ids[content], "the gate position must predict the emission"

    def test_flush_is_never_a_target(self):
        """The model must never be trained to predict end-of-audio -- it cannot know it."""
        for collapse in (False, True):
            ds, tok = self._dataset(use_flush=True, collapse=collapse)
            _, _, targets = self._encode(ds, tok, self._messages(use_flush=True))
            assert ds._flush_id not in targets, f"flush leaked into targets (collapse={collapse})"

    def test_gate_before_flush_targets_blank(self):
        ds, tok = self._dataset(use_flush=True, collapse=True)
        ids, _, targets = self._encode(ds, tok, self._messages(use_flush=True))
        i = ids.index(ds._flush_id)
        assert ids[i - 1] == AUDIO_TOKEN_IDX, "flush should follow the final audio frame"
        assert targets[i - 1] == ds.blank_id, "a boundary that emitted nothing targets blank"

    def _messages_empty_tail(self, use_flush=True):
        """One early word, so the alignment rule leaves nothing pending at flush."""
        return get_llm_messages_for_sample(
            system_role="system",
            system_prompt="Transcribe the audio into text.",
            audio_tag="<audio>",
            blank_token="<blank>",
            chunk_size=self.C,
            num_delay_frames=0,
            audio_duration_secs=2.0,
            frame_length_in_secs=0.08,
            alignments=[WordAlignment(text="hi", start_time=0.0, end_time=0.1)],
            transcript="hi",
            prepend_write_token=True,
            write_token="<|write|>",
            use_flush_token=use_flush,
            flush_token=self.FLUSH,
        )

    def test_rung2_empty_flush_is_a_target_not_an_input(self):
        """'Nothing left' must not put a blank back into the context.

        Rung 2 represents a no-emit decision as a blank TARGET at the gate with nothing
        added to the context; the flush turn follows the same convention, with the
        marker acting as its own gate. Writing `<blank> <eos>` here instead would
        reintroduce exactly the tokens the rung exists to remove.
        """
        ds, tok = self._dataset(use_flush=True, collapse=True)
        ids, _, targets = self._encode(ds, tok, self._messages_empty_tail())
        i = ids.index(ds._flush_id)
        assert i == len(ids) - 1, "nothing may follow the marker when nothing is pending"
        assert targets[i] == ds.blank_id, "the marker is its own gate and targets blank"
        assert ds.blank_id not in ids, "no blank ever enters rung 2's context"

    def test_rung1_empty_flush_keeps_anchor_and_targets_blank(self):
        """Rung 1's convention: the anchor stays, the blank is only a target."""
        ds, tok = self._dataset(use_flush=True, collapse=False)
        ids, _, targets = self._encode(ds, tok, self._messages_empty_tail())
        i = ids.index(ds._flush_id)
        assert ids[i + 1] == ds._eoa_id, "rung 1 keeps the anchor on the flush turn"
        assert targets[i + 1] == ds.blank_id, "and the anchor carries the blank target"

    def test_dynamic_chunking_is_rejected(self):
        with pytest.raises(ValueError, match="requires fixed chunking"):
            self._dataset(use_flush=True, chunk_size=0)

    def test_multi_token_flush_marker_is_rejected(self):
        with pytest.raises(ValueError, match="exactly 1 token"):
            from nemo.collections.common.tokenizers import AutoTokenizer as NeMoTok

            tok = NeMoTok("Qwen/Qwen3-1.7B", use_fast=True)
            tok.add_special_tokens({"additional_special_tokens": ["<blank>", "<|write|>"]})
            cfg = OmegaConf.create(
                {
                    "sample_rate": 16000,
                    "frame_length_in_secs": 0.08,
                    "chunk_size": self.C,
                    "num_delay_frames": 3,
                    "audio_tag": "<audio>",
                    "blank_token": "<blank>",
                    "system_role": "system",
                    "system_prompt": "p",
                    "compact_template": True,
                    "prepend_write_token": True,
                    "write_token": "<|write|>",
                    "use_flush_token": True,
                    "flush_token": "not a registered single token",
                }
            )
            StreamingSTTDataset(cfg=cfg, tokenizer=tok)
