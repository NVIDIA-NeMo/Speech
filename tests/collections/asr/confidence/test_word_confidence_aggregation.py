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

"""Unit tests for subword word-confidence aggregation.

The aggregation must produce exactly one confidence per whitespace-delimited word of the decoded
text (``len(word_confidence) == len(words)`` where ``words = text.split()``), including for
vocabularies that emit word-initial punctuation pieces, ``<unk>``, and byte-fallback whitespace
characters that carry no SentencePiece word-boundary marker.
"""

import re

import pytest

from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMixin

UNDERLINE = '▁'  # SentencePiece word-boundary marker


class _FakeAgg:
    """Minimal stand-in exposing the surface ``_aggregate_token_confidence_subwords_sentencepiece``
    uses: a vocabulary of ``token_id -> (piece, token_text)`` where ``piece`` is the SentencePiece
    piece (as ``decode_ids_to_tokens`` returns it) and ``token_text`` is that token's contribution
    to the decoded text (empty for partial byte-fallback bytes)."""

    _aggregate_token_confidence_subwords_sentencepiece = (
        ConfidenceMixin._aggregate_token_confidence_subwords_sentencepiece
    )

    def __init__(self, vocab):
        self._vocab = vocab

    def decode_ids_to_tokens(self, ids):
        return [self._vocab[int(i)][0] for i in ids]

    def decode_ids_to_str(self, ids):
        text = ''.join(self._vocab[int(i)][1] for i in ids)
        return text.replace(UNDERLINE, ' ').strip()

    def _aggregate_confidence(self, confidences):
        return sum(confidences) / max(len(confidences), 1)


def _run(vocab, ids, confidences=None):
    agg = _FakeAgg(vocab)
    text = agg.decode_ids_to_str(ids)
    words = text.split()
    wc = agg._aggregate_token_confidence_subwords_sentencepiece(
        words, confidences or [1.0] * len(ids), ids
    )
    return words, wc


@pytest.mark.unit
def test_plain_words_one_confidence_each():
    vocab = {
        1: (f"{UNDERLINE}Hallo", f"{UNDERLINE}Hallo"),
        2: (f"{UNDERLINE}Welt", f"{UNDERLINE}Welt"),
    }
    words, wc = _run(vocab, [1, 2])
    assert words == ["Hallo", "Welt"]
    assert len(wc) == len(words)


@pytest.mark.unit
def test_word_initial_punctuation_piece():
    # ▁CDU ▁- ▁CSU: a word-initial punctuation piece must not desynchronize the count
    # from the decoded text's whitespace-delimited words.
    vocab = {
        1: (f"{UNDERLINE}CDU", f"{UNDERLINE}CDU"),
        2: (f"{UNDERLINE}-", f"{UNDERLINE}-"),
        3: (f"{UNDERLINE}CSU", f"{UNDERLINE}CSU"),
    }
    words, wc = _run(vocab, [1, 2, 3])
    assert len(wc) == len(words)


@pytest.mark.unit
def test_standalone_separator_piece_is_not_a_word():
    # A pure ▁ separator piece contributes whitespace only and must not open an extra word.
    vocab = {
        1: (f"{UNDERLINE}Hallo", f"{UNDERLINE}Hallo"),
        2: (UNDERLINE, UNDERLINE),
        3: (",", ","),
        4: (f"{UNDERLINE}Welt", f"{UNDERLINE}Welt"),
    }
    words, wc = _run(vocab, [1, 2, 3, 4])
    assert len(wc) == len(words)


@pytest.mark.unit
def test_unk_adjacency():
    # <unk> renders as its own whitespace-delimited token in the decoded text.
    vocab = {
        1: (f"{UNDERLINE}Hallo", f"{UNDERLINE}Hallo"),
        2: ("<unk>", f"{UNDERLINE}<unk>"),
        3: (f"{UNDERLINE}Welt", f"{UNDERLINE}Welt"),
    }
    words, wc = _run(vocab, [1, 2, 3])
    assert len(wc) == len(words)


@pytest.mark.unit
def test_punctuation_attaches_to_word():
    # ▁Hallo , ▁Welt -> "Hallo, Welt" (comma has no ▁, must stay with previous word)
    vocab = {
        1: (f"{UNDERLINE}Hallo", f"{UNDERLINE}Hallo"),
        2: (",", ","),
        3: (f"{UNDERLINE}Welt", f"{UNDERLINE}Welt"),
    }
    words, wc = _run(vocab, [1, 2, 3])
    assert words == ["Hallo,", "Welt"]
    assert len(wc) == len(words)


class _StripPunctAgg(_FakeAgg):
    """Stand-in for the RNNT path, where ``hypothesis.text`` removes the space before punctuation,
    so ``words`` and the raw decode disagree on where the words are."""

    def decode_with_strip_punctuation(self, ids):
        return re.sub(r' +([\-,.!?;:])', r'\1', self.decode_ids_to_str(ids))


@pytest.mark.unit
def test_stripped_punctuation_attributes_confidence_to_the_right_word():
    # ▁CDU ▁- ▁CSU decodes raw as "CDU - CSU" (3 words) but the hypothesis text is "CDU- CSU"
    # (2 words). The dash belongs to the first word, so its confidence must land there.
    vocab = {
        1: (f"{UNDERLINE}CDU", f"{UNDERLINE}CDU"),
        2: (f"{UNDERLINE}-", f"{UNDERLINE}-"),
        3: (f"{UNDERLINE}CSU", f"{UNDERLINE}CSU"),
    }
    agg = _StripPunctAgg(vocab)
    ids = [1, 2, 3]
    words = agg.decode_with_strip_punctuation(ids).split()
    assert words == ["CDU-", "CSU"]

    wc = agg._aggregate_token_confidence_subwords_sentencepiece(
        words, [0.9, 0.1, 0.8], ids, agg.decode_with_strip_punctuation
    )
    assert wc == pytest.approx([0.5, 0.8])


@pytest.mark.unit
def test_default_oracle_still_follows_the_raw_decode():
    # Callers whose text is the plain decode must keep the raw boundaries: three words here.
    vocab = {
        1: (f"{UNDERLINE}CDU", f"{UNDERLINE}CDU"),
        2: (f"{UNDERLINE}-", f"{UNDERLINE}-"),
        3: (f"{UNDERLINE}CSU", f"{UNDERLINE}CSU"),
    }
    words, wc = _run(vocab, [1, 2, 3], [0.9, 0.1, 0.8])
    assert words == ["CDU", "-", "CSU"]
    assert wc == pytest.approx([0.9, 0.1, 0.8])


class _CountingAgg(_FakeAgg):
    """Fake aggregator that counts how often the expensive decode+split oracle is invoked."""

    def __init__(self, vocab):
        super().__init__(vocab)
        self.oracle_calls = 0

    def decode_ids_to_str(self, ids):
        # The exact oracle re-decodes growing prefixes; every such call is what we want to avoid on
        # the common (fast) path. Count only the multi-token prefix decodes that the oracle makes.
        if len(ids) > 1:
            self.oracle_calls += 1
        return super().decode_ids_to_str(ids)


@pytest.mark.unit
def test_large_plain_sequence_uses_fast_path_only():
    # Thousands of plain ▁-words (a realistic ~10-min chunk) must be handled by the O(n) fast path
    # WITHOUT ever falling back to the O(n^2) decode+split oracle.
    n = 800
    vocab = {i: (f"{UNDERLINE}w{i}", f"{UNDERLINE}w{i}") for i in range(1, n + 1)}
    ids = list(range(1, n + 1))

    agg = _CountingAgg(vocab)
    text = agg.decode_ids_to_str(ids)
    words = text.split()
    assert len(words) == n
    agg.oracle_calls = 0  # reset: ignore the setup decode above

    wc = agg._aggregate_token_confidence_subwords_sentencepiece(words, [1.0] * n, ids)
    assert len(wc) == len(words)
    # The fast path must have carried the whole sequence: no per-prefix oracle re-decodes.
    assert agg.oracle_calls == 0


@pytest.mark.unit
def test_large_sequence_with_byte_fallback_whitespace_oracle_scales():
    # A large sequence that DOES contain a byte-fallback-whitespace word forces the oracle path;
    # the count invariant must still hold at scale.
    vocab = {}
    ids = []
    tid = 1
    for i in range(200):
        vocab[tid] = (f"{UNDERLINE}w{i}", f"{UNDERLINE}w{i}")
        ids.append(tid)
        tid += 1
    # Insert a NBSP-joined "z. B." in the middle (byte-fallback whitespace, no ▁ marker).
    vocab[tid] = (f"{UNDERLINE}z.", f"{UNDERLINE}z.")
    ids.append(tid)
    tid += 1
    vocab[tid] = ("<0xC2>", "")
    ids.append(tid)
    tid += 1
    vocab[tid] = ("<0xA0>", " ")
    ids.append(tid)
    tid += 1
    vocab[tid] = ("B.", "B.")
    ids.append(tid)
    tid += 1
    for i in range(200, 400):
        vocab[tid] = (f"{UNDERLINE}w{i}", f"{UNDERLINE}w{i}")
        ids.append(tid)
        tid += 1

    words, wc = _run(vocab, ids)
    assert len(wc) == len(words)
