# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.common.tokenizers.word_tokenizer import WordTokenizer


class TestWordTokenizer:
    @pytest.mark.unit
    def test_ids_to_text_round_trip(self, tmp_path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("'a'\n'b'\n'c'\n")

        tokenizer = WordTokenizer(
            vocab_file=str(vocab_file),
            bos_token="<BOS>",
            eos_token="<EOS>",
            pad_token="<PAD>",
            unk_token="<UNK>",
        )

        ids = tokenizer.text_to_ids("a b c")
        # ids_to_text must round-trip the encoded ids back to the original
        # space-separated text, and must not raise while doing so.
        assert tokenizer.ids_to_text(ids) == "a b c"

    @pytest.mark.unit
    def test_ids_to_text_strips_special_tokens(self, tmp_path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("'a'\n'b'\n'c'\n")

        tokenizer = WordTokenizer(
            vocab_file=str(vocab_file),
            bos_token="<BOS>",
            eos_token="<EOS>",
            pad_token="<PAD>",
            unk_token="<UNK>",
        )

        ids = [tokenizer.bos_id] + tokenizer.text_to_ids("a b c") + [tokenizer.eos_id, tokenizer.pad_id]
        assert tokenizer.ids_to_text(ids) == "a b c"
