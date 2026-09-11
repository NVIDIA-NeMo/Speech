# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from typing import Dict, List, Optional, Union

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = ['ByteLevelProcessor', 'ByteLevelTokenizer']


class ByteLevelProcessor:
    """
    A very basic tokenization and detokenization class for use with byte-level
    tokenization.
    """

    def detokenize(self, tokens: List[str]) -> str:
        """
        Detokenize a list of tokens into a string.
        """
        return ' '.join(tokens)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string into a list of tokens.
        """
        return list(text)

    def normalize(self, text: str) -> str:
        """
        Normalize a string.
        """
        return text


class ByteLevelTokenizer(TokenizerSpec):
    """
    A byte-level tokenizer that encodes text as UTF-8 bytes with user control over the EOS, BOS, and PAD
        tokens as well as the vocabulary size and a mapping of other special tokens to their IDs.
    """

    def __init__(
        self,
        special_tokens: Optional[Union[Dict[str, str], List[str]]] = None,
        vocab_size: int = 512,
        _eos_id: int = 0,
        _pad_id: int = 1,
        _bos_id: int = None,
    ):
        """A byte-level tokenizer that encodes text as UTF-8 bytes.

        This tokenizer treats each byte as a token, with a default vocabulary size of 512 to accommodate
        UTF-8 byte values (0-255) plus special tokens. It can handle arbitrary text input by encoding
        it into bytes.

        Args:
            special_tokens: Dictionary or list of special tokens to add to the vocabulary.
                These tokens will be assigned IDs at the end of the vocabulary.
                Defaults to None.
            vocab_size: Size of the vocabulary, should be at least 256 to handle all byte values.
                Special tokens will be added after this size.
                Defaults to 512.
            _eos_id: ID to use for the end-of-sequence token.
                Defaults to 0.
            _pad_id: ID to use for the padding token.
                Defaults to 1.
            _bos_id: ID to use for the beginning-of-sequence token.
                Defaults to None.
        """
        self._eos_id = _eos_id
        self._pad_id = _pad_id
        self._bos_id = _bos_id
        self.special_token_to_id = {
            self.pad_id: self.pad_id,
            self.bos_id: self.bos_id,
            self.eos_id: self.eos_id,
        }
        # Track special byte-tokens at end of vocabulary.
        self.vocab_size = vocab_size if special_tokens is None else vocab_size + len(special_tokens)
        self.special_start = self.vocab_size
        special_tokens = {} if special_tokens is None else special_tokens
        for tok in special_tokens:
            self.special_start -= 1
            self.special_token_to_id[tok] = self.special_start
        self.id_to_special_token = {v: k for k, v in self.special_token_to_id.items()}

    # no distinction between tokens and ids.
    def text_to_tokens(self, text):
        """
        Convert a text to a list of tokens.
        """
        return self.text_to_ids(text)

    def tokens_to_text(self, tokens):
        """
        Convert a list of tokens to a text.
        """
        return self.ids_to_text(tokens)

    def text_to_ids(self, text):
        """
        Convert a text to a list of IDs.
        """
        return list(text.encode('utf-8'))

    def ids_to_text(self, ids):
        """
        Convert a list of IDs to a text.
        """
        # remove special tokens.
        ids = [x for x in ids if x < self.special_start]
        return bytes(ids).decode('utf-8', errors='ignore').rstrip()

    def tokens_to_ids(self, tokens):
        """
        Convert a list of tokens to a list of IDs.
        """
        if isinstance(tokens, str):
            tokens = [tokens]
        ids = []
        for token in tokens:
            ids.append(self.token_to_id(token))
        return ids

    def ids_to_tokens(self, ids):
        """
        Convert a list of IDs to a list of tokens.
        """
        if isinstance(ids, int):
            ids = [ids]
        tokens = []
        for id in ids:
            tokens.append(self.id_to_token(id))
        return tokens

    def token_to_id(self, token):
        """
        Convert a token to its corresponding ID.
        """
        if token in self.special_token_to_id:
            return self.special_token_to_id[token]
        else:
            return token

    def id_to_token(self, id):
        """
        Convert an ID to its corresponding token.
        """
        if id < self.special_start:
            return id
        else:
            return self.id_to_special_token[id]

    @property
    def pad_id(self):
        """
        Get the padding ID.
        """
        return self._pad_id

    @property
    def bos_id(self):
        """
        Get the beginning-of-sequence ID.
        """
        return self._bos_id

    @property
    def eos_id(self):
        """
        Get the end-of-sequence ID.
        """
        return self._eos_id

    @property
    def unk_id(self):
        """
        Get the unknown ID.
        """
        return 259  # unused
