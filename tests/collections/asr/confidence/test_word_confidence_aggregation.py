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

import os
import re
from types import SimpleNamespace

import pytest
import sentencepiece as spm

from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
from nemo.collections.asr.parts.utils import asr_confidence_utils
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
    wc = agg._aggregate_token_confidence_subwords_sentencepiece(words, confidences or [1.0] * len(ids), ids)
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
    words, wc = _run(vocab, [1, 2, 3], [0.9, 0.1, 0.8])
    assert words == ["CDU", "-", "CSU"]
    assert wc == pytest.approx([0.9, 0.1, 0.8])


@pytest.mark.unit
def test_standalone_separator_piece_is_not_a_word():
    # A pure ▁ separator piece contributes whitespace only and must not open an extra word.
    vocab = {
        1: (f"{UNDERLINE}Hallo", f"{UNDERLINE}Hallo"),
        2: (UNDERLINE, UNDERLINE),
        3: (",", ","),
        4: (f"{UNDERLINE}Welt", f"{UNDERLINE}Welt"),
    }
    words, wc = _run(vocab, [1, 2, 3, 4], [0.9, 0.2, 0.4, 0.8])
    # "Hallo , Welt": the separator opens the "," word, as it would any word it precedes.
    assert words == ["Hallo", ",", "Welt"]
    assert wc == pytest.approx([0.9, 0.3, 0.8])


@pytest.mark.unit
def test_unk_adjacency():
    # <unk> renders as its own whitespace-delimited token in the decoded text.
    vocab = {
        1: (f"{UNDERLINE}Hallo", f"{UNDERLINE}Hallo"),
        2: ("<unk>", f"{UNDERLINE}<unk>"),
        3: (f"{UNDERLINE}Welt", f"{UNDERLINE}Welt"),
    }
    words, wc = _run(vocab, [1, 2, 3], [0.9, 0.2, 0.8])
    assert words == ["Hallo", "<unk>", "Welt"]
    assert wc == pytest.approx([0.9, 0.2, 0.8])


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

    supported_punctuation = {'-', ',', '.', '!', '?', ';', ':', "'", '%', '¿'}
    space_before_punct_pattern = re.compile(r'(\s)(' + '|'.join(re.escape(p) for p in supported_punctuation) + ')')

    def decode_with_strip_punctuation(self, ids):
        return self.space_before_punct_pattern.sub(r'\2', self.decode_ids_to_str(ids))


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


class _FakeCTCBPE(_StripPunctAgg):
    """Runs the real ``CTCBPEDecoding._aggregate_token_confidence``, whose hypothesis text is
    decoded with punctuation stripping."""

    _aggregate_token_confidence = CTCBPEDecoding._aggregate_token_confidence

    def decode_tokens_to_str(self, ids):
        return self.decode_ids_to_str(ids)

    def decode_tokens_to_str_with_strip_punctuation(self, ids):
        return self.decode_with_strip_punctuation(ids)


@pytest.mark.unit
def test_ctc_bpe_word_confidence_follows_the_stripped_text():
    # The CTC text is "CDU- CSU" (2 words), so there must be two confidences, the dash on the first.
    vocab = {
        1: (f"{UNDERLINE}CDU", f"{UNDERLINE}CDU"),
        2: (f"{UNDERLINE}-", f"{UNDERLINE}-"),
        3: (f"{UNDERLINE}CSU", f"{UNDERLINE}CSU"),
    }
    agg = _FakeCTCBPE(vocab)
    ids = [1, 2, 3]
    hypothesis = SimpleNamespace(text=(ids, [1, 1, 1]), token_confidence=[0.9, 0.1, 0.8])

    wc = agg._aggregate_token_confidence(hypothesis)
    assert len(wc) == len(agg.decode_tokens_to_str_with_strip_punctuation(ids).split())
    assert wc == pytest.approx([0.5, 0.8])


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

    confidences = [0.9] * len(ids)
    confidences[200:204] = [0.1, 0.2, 0.3, 0.4]  # ▁z. <0xC2> <0xA0> B.
    words, wc = _run(vocab, ids, confidences)
    assert words[200:202] == ["z.", "B."]
    expected = [0.9] * 200 + [(0.1 + 0.2 + 0.3) / 3, 0.4] + [0.9] * 200
    assert wc == pytest.approx(expected)


@pytest.mark.unit
def test_lone_underline_before_mark_stripped():
    vocab = {
        1: (f'{UNDERLINE}son', f'{UNDERLINE}son'),
        2: (UNDERLINE, UNDERLINE),
        3: (',', ','),
        4: (f'{UNDERLINE}and', f'{UNDERLINE}and'),
    }
    agg = _StripPunctAgg(vocab)
    ids = [1, 2, 3, 4]
    words = agg.decode_with_strip_punctuation(ids).split()
    assert words == ['son,', 'and']

    wc = agg._aggregate_token_confidence_subwords_sentencepiece(
        words,
        [0.9, 0.2, 0.8, 0.7],
        ids,
        agg.decode_with_strip_punctuation,
    )
    assert wc == pytest.approx([(0.9 + 0.2 + 0.8) / 3, 0.7])


# The parakeet-tdt-0.6b-v2 split of "from 20 growth": a lone ▁ before the digits.
_FROM_20_GROWTH = {
    1: (f"{UNDERLINE}from", f"{UNDERLINE}from"),
    2: (UNDERLINE, UNDERLINE),
    3: ("2", "2"),
    4: ("0", "0"),
    5: (f"{UNDERLINE}gr", f"{UNDERLINE}gr"),
    6: ("ow", "ow"),
    7: ("th", "th"),
    8: (",", ","),
    9: ("¿", "¿"),
    10: ("qu", "qu"),
    11: ("é", "é"),
}


@pytest.mark.unit
def test_lone_underline_goes_to_the_word_it_starts_on_both_paths():
    # A trailing "▁ ," is stripped onto "growth", so its word count only matches through the decode.
    # The words before it must get the same values as without it: the lone ▁ belongs to "20".
    agg = _StripPunctAgg(_FROM_20_GROWTH)
    fast_ids, exact_ids = [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 2, 8]
    conf = {1: 0.9, 2: 0.1, 3: 0.9, 4: 0.9, 5: 0.9, 6: 0.9, 7: 0.9, 8: 0.9}
    results = []
    for ids in (fast_ids, exact_ids):
        words = agg.decode_with_strip_punctuation(ids).split()
        results.append(
            agg._aggregate_token_confidence_subwords_sentencepiece(
                words, [conf[i] for i in ids], ids, agg.decode_with_strip_punctuation
            )
        )
    assert results[0] == pytest.approx([0.9, (0.1 + 0.9 + 0.9) / 3, 0.9])
    assert results[1][:2] == pytest.approx(results[0][:2])


@pytest.mark.unit
def test_lone_underline_at_the_start_is_kept():
    # "▁ ¿ qu é ▁ ," decodes to "¿qué,": the leading ▁ is part of that one word, not dropped.
    agg = _StripPunctAgg(_FROM_20_GROWTH)
    ids = [2, 9, 10, 11, 2, 8]
    words = agg.decode_with_strip_punctuation(ids).split()
    assert words == ["¿qué,"]
    wc = agg._aggregate_token_confidence_subwords_sentencepiece(
        words, [0.1, 0.9, 0.9, 0.9, 0.9, 0.9], ids, agg.decode_with_strip_punctuation
    )
    assert wc == pytest.approx([(0.1 + 0.9 * 5) / 6])


@pytest.mark.unit
def test_fast_groups_must_decode_to_their_words():
    # "▁a ▁, <0x0A> b" with the space before "," stripped reads "a,\nb": two words and two ▁ groups,
    # but the groups are [▁a] and [▁, <0x0A> b], which puts the comma on "b".
    vocab = {
        1: (f"{UNDERLINE}a", f"{UNDERLINE}a"),
        2: (f"{UNDERLINE},", f"{UNDERLINE},"),
        3: ("<0x0A>", "\n"),
        4: ("b", "b"),
    }
    agg = _StripPunctAgg(vocab)
    ids = [1, 2, 3, 4]
    words = agg.decode_with_strip_punctuation(ids).split()
    assert words == ["a,", "b"]
    wc = agg._aggregate_token_confidence_subwords_sentencepiece(
        words, [0.9, 0.1, 0.5, 0.8], ids, agg.decode_with_strip_punctuation
    )
    assert wc == pytest.approx([(0.9 + 0.1 + 0.5) / 3, 0.8])


class _WorkCountingAgg(_StripPunctAgg):
    """Counts the token ids every boundary decode is given, which is the aggregation's cost."""

    def __init__(self, vocab):
        super().__init__(vocab)
        self.decoded = 0

    def decode_with_strip_punctuation(self, ids):
        self.decoded += len(ids)
        return super().decode_with_strip_punctuation(ids)


@pytest.mark.unit
@pytest.mark.parametrize("n_words", [500, 2000])
def test_exact_path_work_is_linear(n_words):
    # Words joined by byte-fallback newlines carry no ▁, so only the decode can split them and the
    # whole hypothesis takes the exact path. Its decodes must cover a bounded number of ids per token,
    # not every prefix.
    vocab = {1: ("<0x0A>", "\n")}
    ids = []
    for i in range(n_words):
        vocab[i + 2] = (f"w{i}", f"w{i}")
        ids += [1, i + 2] if ids else [i + 2]
    agg = _WorkCountingAgg(vocab)
    words = agg.decode_with_strip_punctuation(ids).split()
    assert len(words) == n_words
    agg.decoded = 0
    wc = agg._aggregate_token_confidence_subwords_sentencepiece(
        words, [0.5] * len(ids), ids, agg.decode_with_strip_punctuation
    )
    assert wc == pytest.approx([0.5] * n_words)
    assert agg.decoded <= 12 * len(ids)


@pytest.mark.unit
def test_padding_or_truncating_warns_once(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        asr_confidence_utils.logging, "warning", lambda msg, *a, **k: warnings.append((msg, k.get("mode")))
    )
    vocab = {1: (f"{UNDERLINE}Hallo", f"{UNDERLINE}Hallo")}
    agg = _FakeAgg(vocab)
    # One token, but the caller's text was changed after decoding to two words.
    wc = agg._aggregate_token_confidence_subwords_sentencepiece(["Hallo", "Welt"], [0.7], [1])
    assert wc == pytest.approx([0.7, 0.7])
    assert len(warnings) == 1 and warnings[0][1] == asr_confidence_utils.LogMode.ONCE


class _FakeRNNTBPE(_StripPunctAgg):
    """Runs the real ``RNNTBPEDecoding._aggregate_token_confidence``, whose ``hypothesis.words``
    come from the stripped text."""

    _aggregate_token_confidence = RNNTBPEDecoding._aggregate_token_confidence

    def decode_tokens_to_str_with_strip_punctuation(self, ids):
        return self.decode_with_strip_punctuation(ids)


@pytest.mark.unit
def test_rnnt_bpe_word_confidence_follows_the_stripped_text():
    vocab = {
        1: (f"{UNDERLINE}CDU", f"{UNDERLINE}CDU"),
        2: (f"{UNDERLINE}-", f"{UNDERLINE}-"),
        3: (f"{UNDERLINE}CSU", f"{UNDERLINE}CSU"),
    }
    agg = _FakeRNNTBPE(vocab)
    ids = [1, 2, 3]
    hypothesis = SimpleNamespace(
        words=agg.decode_with_strip_punctuation(ids).split(), y_sequence=ids, token_confidence=[0.9, 0.1, 0.8]
    )
    assert hypothesis.words == ["CDU-", "CSU"]
    assert agg._aggregate_token_confidence(hypothesis) == pytest.approx([0.5, 0.8])


class _RealSentencePieceAgg(_StripPunctAgg):
    """The aggregation over a real SentencePiece model, decoded as the decoders' tokenizers do."""

    def __init__(self, model_file):
        # No fake vocabulary: the real model replaces both decode methods.
        super().__init__({})
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

    def decode_ids_to_tokens(self, ids):
        return [self.sp.id_to_piece(int(i)) for i in ids]

    def decode_ids_to_str(self, ids):
        return self.sp.decode([int(i) for i in ids])


def _per_word_expected(sp, text, confidences):
    """Each word's mean confidence, from encoding the text's words one at a time."""
    expected, at = [], 0
    for word in text.split():
        n = len(sp.encode(word))
        expected.append(sum(confidences[at : at + n]) / n)
        at += n
    assert at == len(confidences)
    return expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "name, text",
    [
        # an4 has no digits: "2" is "▁ <unk>", a lone ▁ before an unknown piece.
        ("an4_spe_128", "two 2 three oh enter"),
        # This canary model is all special tokens, so every word is "▁ <unk>".
        ("canary", "a b c"),
    ],
)
def test_real_sentencepiece_word_confidence(test_data_dir, name, text):
    agg = _RealSentencePieceAgg(os.path.join(test_data_dir, "asr", "tokenizers", name, "tokenizer.model"))
    ids = agg.sp.encode(text)
    assert agg.sp.piece_to_id(UNDERLINE) in ids
    confidences = [round(0.1 + 0.8 * (k % 7) / 6, 3) for k in range(len(ids))]
    words = agg.decode_ids_to_str(ids).split()
    assert len(words) == len(text.split())
    wc = agg._aggregate_token_confidence_subwords_sentencepiece(words, confidences, ids, agg.decode_ids_to_str)
    assert wc == pytest.approx(_per_word_expected(agg.sp, text, confidences))


@pytest.mark.unit
def test_lone_underlines_stay_on_the_fast_path():
    # "▁ ▁from ▁ 2 0 ▁gr ow th ▁": lone ▁ at the start, before digits, and at the end. The marker
    # groups them as the decode would, so each id is decoded once, to check its group.
    agg = _WorkCountingAgg(_FROM_20_GROWTH)
    ids = [2, 1, 2, 3, 4, 5, 6, 7, 2]
    conf = [0.1, 0.9, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3]
    words = agg.decode_with_strip_punctuation(ids).split()
    assert words == ["from", "20", "growth"]
    agg.decoded = 0
    wc = agg._aggregate_token_confidence_subwords_sentencepiece(words, conf, ids, agg.decode_with_strip_punctuation)
    assert wc == pytest.approx([(0.1 + 0.9) / 2, (0.2 + 0.9 + 0.9) / 3, (0.9 * 3 + 0.3) / 4])
    assert agg.decoded == len(ids)
