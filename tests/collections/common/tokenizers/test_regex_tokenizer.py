# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytest

from nemo.collections.common.tokenizers.regex_tokenizer import RegExTokenizer


class TestRegExTokenizer:
    @pytest.mark.unit
    def test_save_load_preserves_multi_character_tokens(self, tmp_path):
        # SMILES-style regex with multi-character tokens (e.g. Cl, Br, @@, [nH]).
        regex = r"\[[^\]]+]|Br|Cl|@@|@|[0-9]"
        tokenizer = RegExTokenizer(regex=regex)

        # Multi-character tokens as build_vocab_from_csv / build_vocab_from_text would add them,
        # alongside the default multi-character special tokens (<PAD>, <MASK>, <SEP>).
        for token in ["Cl", "Br", "@@", "[nH]"]:
            tokenizer.vocab[token] = len(tokenizer.vocab)

        expected_vocab = dict(tokenizer.vocab)

        regex_file = tmp_path / "tokenizer.model"
        vocab_file = tmp_path / "tokenizer.vocab"
        tokenizer.save_tokenizer(regex_file=str(regex_file), vocab_file=str(vocab_file))

        reloaded = RegExTokenizer().load_tokenizer(regex_file=str(regex_file), vocab_file=str(vocab_file))

        # save_tokenizer must persist each full token, not just its first character,
        # otherwise multi-character tokens are truncated and collide on reload.
        assert reloaded.vocab == expected_vocab
