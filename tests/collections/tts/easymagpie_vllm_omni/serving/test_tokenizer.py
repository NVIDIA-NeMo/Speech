# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from types import SimpleNamespace

import pytest
from easymagpie_vllm_omni.tokenizer import EasyMagpieTextTokenizer


class _FakeBaseTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return [100 + ord(char) for char in text]


class _FakePhonemeTokenizer:
    def get_vocab(self):
        return {f"p{i}": i for i in range(5)}

    def encode(self, text):
        return SimpleNamespace(ids=[ord(char) - ord("a") for char in text])


def _tokenizer(**kwargs):
    return EasyMagpieTextTokenizer(
        _FakeBaseTokenizer(),
        text_vocab_size=208,
        phoneme_tokenizer=_FakePhonemeTokenizer(),
        text_phoneme_token_offset=200,
        text_phoneme_vocab_size=8,
        **kwargs,
    )


def test_mixed_text_uses_base_and_offset_ipa_ids():
    tokenizer = _tokenizer()

    tokens = tokenizer.encode("Hi <bop>ab<eop>!")

    assert tokens == [100 + ord(char) for char in "Hi "] + [200, 201] + [100 + ord("!")]
    assert tokenizer.base_tokenizer.calls == [("Hi ", False), ("!", False)]


def test_mixed_text_supports_multiple_unicode_spans():
    tokenizer = _tokenizer()
    tokenizer.phoneme_tokenizer.encode = lambda text: SimpleNamespace(ids=[1] * len(text))

    tokens = tokenizer.encode("<bop> ə <eop>x<bop>ɐ<eop>")

    assert tokens == [201, 100 + ord("x"), 201]


def test_custom_markers_and_empty_span():
    tokenizer = _tokenizer(bop_marker="[P]", eop_marker="[/P]")

    assert tokenizer.encode("a[P][/P]b") == [100 + ord("a"), 100 + ord("b")]


@pytest.mark.parametrize("text", ["x<eop>", "<bop>x"])
def test_malformed_markers_raise(text):
    with pytest.raises(ValueError):
        _tokenizer().encode(text)


def test_disabled_tokenizer_preserves_literal_markers():
    base = _FakeBaseTokenizer()
    tokenizer = EasyMagpieTextTokenizer(base, text_vocab_size=200)
    text = "x<bop>ab<eop>"

    assert tokenizer.encode(text) == [100 + ord(char) for char in text]
    assert base.calls == [(text, False)]


def test_context_always_uses_base_tokenizer_defaults():
    tokenizer = _tokenizer()

    tokenizer.encode_context("[EN]")

    assert tokenizer.base_tokenizer.calls == [("[EN]", True)]


def test_incremental_encoder_accepts_markers_and_ipa_across_chunks():
    tokenizer = _tokenizer()
    encoder = tokenizer.incremental_encoder()
    tokens = []
    for chunk in ("Hi <bo", "p>a", "b<eo", "p>!"):
        tokens.extend(encoder.push(chunk))
    tokens.extend(encoder.finish())

    assert tokens == [100 + ord(char) for char in "Hi "] + [200, 201] + [100 + ord("!")]
    assert encoder.clean


def test_incremental_encoder_accepts_every_boundary_split():
    tokenizer = _tokenizer()
    text = "x<bop>ab<eop>y"
    encoder = tokenizer.incremental_encoder()
    tokens = []
    for char in text:
        tokens.extend(encoder.push(char))
    tokens.extend(encoder.finish())

    assert tokens == tokenizer.encode(text)


def test_incremental_finish_flushes_partial_marker_as_text():
    tokenizer = _tokenizer()
    encoder = tokenizer.incremental_encoder()

    tokens = encoder.push("x<bo") + encoder.finish()

    assert tokens == [100 + ord(char) for char in "x<bo"]


def test_incremental_finish_rejects_open_phoneme_span():
    encoder = _tokenizer().incremental_encoder()
    encoder.push("x<bop>ab")

    with pytest.raises(ValueError, match="without a matching"):
        encoder.finish()


@pytest.fixture(scope="module")
def byte_bpe_base():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        [
            "Thanks for calling, we'll get back to you shortly.",
            "I don't think we're calling twice. I'd call again.",
            "Hello.\nNext line!\r\nOne  two\tthree.",
            "Café, déjà vu — good morning!",
            "The amount is 1,234.56 dollars. Hi <bop>ab<eop> there.",
        ],
        trainers.BpeTrainer(
            vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["[UNK]"]
        ),
    )

    class BaseTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return tokenizer.encode(text, add_special_tokens=False).ids

        def get_vocab(self):
            return tokenizer.get_vocab()

    return BaseTokenizer()


@pytest.mark.parametrize("pronunciation_control", [False, True])
@pytest.mark.parametrize("chunking", ["characters", "trailing_words", "subwords"])
def test_incremental_bpe_is_independent_of_text_chunk_boundaries(byte_bpe_base, pronunciation_control, chunking):
    import re

    tokenizer = _byte_bpe_text_tokenizer(byte_bpe_base, pronunciation_control)
    text = "Thanks for calling, we'll get back to you shortly."
    chunks = {
        "characters": list(text),
        "trailing_words": re.findall(r"\S+\s*", text),
        "subwords": ["Thanks for call", "ing, we'", "ll get back to you short", "ly."],
    }[chunking]
    encoder = tokenizer.incremental_encoder()
    actual = [token for chunk in chunks for token in encoder.push(chunk)] + encoder.finish()

    assert actual == tokenizer.encode(text)


@pytest.mark.parametrize("pronunciation_control", [False, True])
@pytest.mark.parametrize(
    "text",
    [
        "Thanks for calling, we'll get back to you shortly.",
        "I don't think we're calling twice. I'd call again.",
        "Hello.\nNext line!\r\nOne  two\tthree.",
        "Café, déjà vu — good morning!",
        "The amount is 1,234.56 dollars.",
        "Hi <bop>ab<eop> there.",
    ],
)
def test_incremental_bpe_matches_every_two_way_split(byte_bpe_base, pronunciation_control, text):
    tokenizer = _byte_bpe_text_tokenizer(byte_bpe_base, pronunciation_control)
    expected = tokenizer.encode(text)
    for offset in range(1, len(text)):
        encoder = tokenizer.incremental_encoder()
        actual = encoder.push(text[:offset]) + encoder.push(text[offset:]) + encoder.finish()
        assert actual == expected, (text, offset)


@pytest.mark.parametrize("pronunciation_control", [False, True])
def test_incremental_bpe_emits_complete_words_before_finish(byte_bpe_base, pronunciation_control):
    tokenizer = _byte_bpe_text_tokenizer(byte_bpe_base, pronunciation_control)
    encoder = tokenizer.incremental_encoder()

    assert encoder.push("Thank") == []
    assert encoder.push("s ") == byte_bpe_base.encode("Thanks")
    assert encoder.push("for ") == byte_bpe_base.encode(" for")
    assert encoder.clean
    assert encoder.finish() == byte_bpe_base.encode(" ")


def _byte_bpe_text_tokenizer(base, pronunciation_control):
    offset = len(base.get_vocab())
    return EasyMagpieTextTokenizer(
        base,
        text_vocab_size=offset + 8,
        phoneme_tokenizer=_FakePhonemeTokenizer() if pronunciation_control else None,
        text_phoneme_token_offset=offset if pronunciation_control else None,
        text_phoneme_vocab_size=8 if pronunciation_control else 0,
    )
