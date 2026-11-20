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

from pathlib import Path

import youtokentome as yttm

from nemo.collections.common.tokenizers import TokenizerSpec

__all__ = ['YouTokenToMeTokenizer']


class YouTokenToMeTokenizer(TokenizerSpec):
    """Tokenizer backed by a YouTokenToMe model."""

    def __init__(self, model_path, bpe_dropout=0.0, legacy=False, r2l=False):
        model_path = Path(model_path).expanduser()
        self.tokenizer = yttm.BPE(model=str(model_path))
        self.vocab_size = len(self.tokenizer.vocab())
        self.special_tokens = self.tokens_to_ids(["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
        self.bpe_dropout = bpe_dropout
        self.legacy = legacy
        self.r2l = r2l

    def text_to_tokens(self, text):
        return self.tokenizer.encode(
            text, output_type=yttm.OutputType.SUBWORD, dropout_prob=self.bpe_dropout, reverse=self.r2l
        )

    def tokens_to_text(self, tokens):
        return self.ids_to_text(self.tokens_to_ids(tokens))

    def text_to_ids(self, text):
        return self.tokenizer.encode(
            text, output_type=yttm.OutputType.ID, dropout_prob=self.bpe_dropout, reverse=self.r2l
        )

    def ids_to_text(self, ids):
        ids_ = [id_ for id_ in ids if id_ not in self.special_tokens]
        if self.r2l:
            ids_ = ids_[::-1]
        return self.tokenizer.decode([ids_])[0]

    def tokens_to_ids(self, tokens):
        return [self.tokenizer.subword_to_id(token) for token in tokens]

    def ids_to_tokens(self, ids):
        if self.legacy:
            ids_ = [id_ for id_ in ids if id_ not in self.special_tokens]
        else:
            ids_ = ids
        return [self.tokenizer.id_to_subword(id_) for id_ in ids_]

    @property
    def pad_id(self):
        return self.tokenizer.subword_to_id("<PAD>")

    @property
    def bos_id(self):
        return self.tokenizer.subword_to_id("<BOS>")

    @property
    def eos_id(self):
        return self.tokenizer.subword_to_id("<EOS>")

    @property
    def unk_id(self):
        return self.tokenizer.subword_to_id("<UNK>")
