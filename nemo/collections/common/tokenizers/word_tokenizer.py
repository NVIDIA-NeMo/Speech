# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

from nemo.collections.common.tokenizers.char_tokenizer import CharTokenizer

__all__ = ['WordTokenizer']


class WordTokenizer(CharTokenizer):
    "Tokenizes at word boundary"

    def __init__(
        self,
        vocab_file: str,
        mask_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        unk_token: Optional[str] = None,
    ):
        """
        Args:
            vocab_file: path to file with vocabulary which consists
                of characters separated by \n
            mask_token: mask token
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token. Usually equal to sep_token
            pad_token: token to use for padding
            sep_token: token used for separating sequences
            cls_token: class token. Usually equal to bos_token
            unk_token: token to use for unknown tokens
        """

        super().__init__(
            vocab_file=vocab_file,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
        )

    def text_to_tokens(self, text):
        token_candidates = text.strip().split()
        tokens = []
        for token in token_candidates:
            if token in self.vocab:
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        return tokens

    def ids_to_text(self, ids):
        ids_ = [id_ for id_ in ids if id_ not in self.special_tokens]
        return " ".join(self.ids_to_tokens(ids_))
