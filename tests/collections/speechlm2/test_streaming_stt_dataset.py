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

import pytest

from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    AUDIO_TOKEN_IDX,
    IGNORE_INDEX,
    _replace_audio_chunks,
    _tokenize_compact_with_assistant_mask,
    _tokenize_with_assistant_mask,
    build_compact_turn_markers,
    compute_word_spans,
    decode_with_blank,
    get_llm_messages_for_batch,
    get_llm_messages_for_sample,
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
    "gemma4": "google/gemma-4-E4B-it",
}


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
class TestCompactTemplate:
    """Tests for the compact chat template feature (Qwen3 tokenizer).

    Compact mode drops per-turn role wrapping, yielding
    ``[system_wrapped] <audio>*N <write> text <eos> <audio>*N <write> text <eos>``.
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
        uh, ufah, af = build_compact_turn_markers(qwen3_hf, "<|im_start|>")
        assert uh == []
        im_start_id = qwen3_hf.convert_tokens_to_ids("<|im_start|>")
        im_end_id = qwen3_hf.eos_token_id
        assert ufah == [im_start_id]
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
        """Mask=1 on write/text/eos; mask=0 on system wrapping and audio user content."""
        hf = qwen3_tok.tokenizer
        write_id = hf.convert_tokens_to_ids("<|im_start|>")
        eos_id = hf.eos_token_id
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content": "X"},
        ]
        input_ids, mask = _tokenize_compact_with_assistant_mask(messages, qwen3_tok, write_id, eos_id)

        # Final token should be the trailing EOS with mask=1.
        assert input_ids[-1] == eos_id
        assert mask[-1] == 1

        # User content (the audio tag tokens) must have mask=0. Verify each audio
        # BPE piece appears at least once with mask=0 (at the user-content occurrence).
        audio_ids = hf.encode("<audio>", add_special_tokens=False)
        for aid in audio_ids:
            positions = [i for i, t in enumerate(input_ids) if t == aid]
            assert any(mask[p] == 0 for p in positions), f"audio token {aid} should appear with mask=0"

        # At least one write_id occurrence must have mask=1 (the one we inserted).
        write_positions = [i for i, t in enumerate(input_ids) if t == write_id]
        assert any(mask[p] == 1 for p in write_positions), "inserted write_id should have mask=1"

        # System wrapping contains write_id with mask=0 (e.g. <|im_start|>system\n...).
        assert any(mask[p] == 0 for p in write_positions), "system-wrapping write_id should have mask=0"

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
        Expected shape per silent chunk: ``<audio>*N <|im_start|> <|im_end|>``
        (adjacent write+eos with no text between). Model learns to emit an
        immediate close-of-turn for silent chunks. Both write and eos must
        be trainable (mask=1).
        """
        hf = qwen3_tok.tokenizer
        write_id = hf.convert_tokens_to_ids("<|im_start|>")
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
        input_ids, mask = _tokenize_compact_with_assistant_mask(messages, qwen3_tok, write_id, eos_id)

        assert len(input_ids) == len(mask)

        # Locate the *inserted* write_id positions (mask=1). System-wrapping
        # write_id occurrences have mask=0.
        inserted_write_positions = [i for i, t in enumerate(input_ids) if t == write_id and mask[i] == 1]
        assert len(inserted_write_positions) == 3, "one <|im_start|> per turn (3 turns)"

        # For each inserted write, the next token is either:
        #   - eos_id with mask=1 (silent chunk: <|im_start|><|im_end|> adjacent)
        #   - a content token with mask=1 (word chunk: <|im_start|> hello ...)
        # Verify the two silent chunks have immediate <|im_end|> right after <|im_start|>.
        adjacent_pairs = 0
        for pos in inserted_write_positions:
            if pos + 1 < len(input_ids) and input_ids[pos + 1] == eos_id:
                adjacent_pairs += 1
                # Adjacent eos must also be trainable (mask=1).
                assert mask[pos + 1] == 1, "adjacent <|im_end|> in silent chunk must be trainable"
        assert adjacent_pairs == 2, "two silent chunks should each produce adjacent <|im_start|><|im_end|>"

        # The middle ("hello") chunk: <|im_start|> should NOT be directly followed by <|im_end|>.
        # Verify at least one hello-content token exists with mask=1 between a write and an eos.
        hello_ids = set(hf.encode("hello", add_special_tokens=False))
        trainable_content = [input_ids[i] for i in range(len(input_ids)) if mask[i] == 1 and input_ids[i] in hello_ids]
        assert len(trainable_content) > 0, "hello content tokens must be trainable"

        # Sanity: state-machine inference separator / decode_with_blank splitter.
        # When has_blank=False, code appends `self._eos_id` as the chunk separator
        # and decode_with_blank splits on `eos_token_id`. The training sequence's
        # per-turn eos_id plays exactly that role. Verify the trailing token is eos.
        assert input_ids[-1] == eos_id


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

    def test_list_candidates_and_audio_ids(self):
        ds = self._make_dataset([2, 4])
        assert ds._chunk_size_candidates == [2, 4]
        assert set(ds._audio_chunk_ids_by_size) == {2, 4}
        assert len(ds._audio_chunk_ids_by_size[2]) == 2
        assert len(ds._audio_chunk_ids_by_size[4]) == 4

    def test_scalar_backward_compatible(self):
        ds = self._make_dataset(2)
        assert ds._chunk_size_candidates is None
        assert set(ds._audio_chunk_ids_by_size) == {2}
        assert len(ds._audio_chunk_ids_by_size[2]) == 2

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
