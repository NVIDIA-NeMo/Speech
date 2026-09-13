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

import json
from pathlib import Path

import pytest

from nemo.collections.common.tokenizers.word_tokenizer import WordTokenizer


@pytest.fixture
def word_vocab_file(tmp_path: Path) -> str:
    vocab_path = tmp_path / "vocab.txt"
    special_tokens = {
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    }
    with vocab_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(special_tokens) + "\n")
        for char in ["a", "b", "c", "d", "e"]:
            f.write(f"{char!r}\n")
    return str(vocab_path)


@pytest.fixture
def word_vocab_no_special_file(tmp_path: Path) -> str:
    vocab_path = tmp_path / "vocab_no_special.txt"
    with vocab_path.open("w", encoding="utf-8") as f:
        for char in ["x", "y", "z"]:
            f.write(f"{char!r}\n")
    return str(vocab_path)


class TestWordTokenizer:
    @pytest.mark.unit
    def test_ids_to_text(self, word_vocab_file: str):
        tokenizer = WordTokenizer(vocab_file=word_vocab_file)
        ids = tokenizer.tokens_to_ids(["a", "b", "c"])
        # Should join tokens with space
        assert tokenizer.ids_to_text(ids) == "a b c"

    @pytest.mark.unit
    def test_ids_to_text_strips_special_tokens(self, word_vocab_file: str):
        tokenizer = WordTokenizer(vocab_file=word_vocab_file)
        ids = tokenizer.tokens_to_ids(["a", "b"])
        # Embed bos, pad, and eos token ids around tokens
        token_ids_with_special = (
            [tokenizer.bos_id, tokenizer.pad_id]
            + ids
            + [tokenizer.pad_id, tokenizer.eos_id]
        )
        assert tokenizer.ids_to_text(token_ids_with_special) == "a b"

    @pytest.mark.unit
    def test_ids_to_text_empty_and_special_only(self, word_vocab_file: str):
        tokenizer = WordTokenizer(vocab_file=word_vocab_file)
        assert tokenizer.ids_to_text([]) == ""
        assert tokenizer.ids_to_text([tokenizer.bos_id, tokenizer.pad_id, tokenizer.eos_id]) == ""

    @pytest.mark.unit
    def test_text_to_tokens(self, word_vocab_file: str):
        tokenizer = WordTokenizer(vocab_file=word_vocab_file)
        tokens = tokenizer.text_to_tokens("  a b unknown_token c  ")
        assert tokens == ["a", "b", "<unk>", "c"]

    @pytest.mark.unit
    def test_tokens_to_ids_and_ids_to_tokens(self, word_vocab_file: str):
        tokenizer = WordTokenizer(vocab_file=word_vocab_file)
        tokens = ["a", "b", "c"]
        ids = tokenizer.tokens_to_ids(tokens)
        assert len(ids) == 3
        assert tokenizer.ids_to_tokens(ids) == tokens

    @pytest.mark.unit
    def test_roundtrip_text_to_ids_to_text(self, word_vocab_file: str):
        tokenizer = WordTokenizer(vocab_file=word_vocab_file)
        text = "a b c"
        ids = tokenizer.text_to_ids(text)
        assert tokenizer.ids_to_text(ids) == text

    @pytest.mark.unit
    def test_tokenizer_without_special_tokens(self, word_vocab_no_special_file: str):
        tokenizer = WordTokenizer(vocab_file=word_vocab_no_special_file)
        assert tokenizer.ids_to_text(tokenizer.text_to_ids("x y z")) == "x y z"
