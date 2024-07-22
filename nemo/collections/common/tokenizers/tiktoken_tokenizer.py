# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tiktoken
except ImportError:
    pass

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = ['TiktokenTokenizer']


def reload_mergeable_ranks(
    path: str,
    max_vocab: Optional[int] = None,
) -> Dict[bytes, int]:
    """
    Reload the tokenizer JSON file and convert it to Tiktoken format.
    """
    assert path.endswith(".json")

    # reload vocab
    with open(path, "r") as f:
        vocab = json.load(f)
    assert isinstance(vocab, list)
    print(f"Vocab size: {len(vocab)}")
    if max_vocab is not None:
        vocab = vocab[:max_vocab]
        print(f"Cutting vocab to first {len(vocab)} tokens.")

    # build ranks
    ranks: Dict[bytes, int] = {}
    for i, x in enumerate(vocab):
        assert x.keys() == {"rank", "token_bytes", "token_str"}
        assert x["rank"] == i
        merge = base64.b64decode(x["token_bytes"])
        assert i >= 256 or merge == bytes([i])
        ranks[merge] = x["rank"]

    # sanity check
    assert len(ranks) == len(vocab)
    assert set(ranks.values()) == set(range(len(ranks)))

    return ranks


PATTERN_TIKTOKEN = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
DEFAULT_TIKTOKEN_MAX_VOCAB = 2**17  # 131072
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>"]
SPECIAL_TOKEN_TEMPLATE = "<SPECIAL_{id}>"


class TiktokenTokenizer(TokenizerSpec):
    """
    TiktokenTokenizer https://github.com/openai/tiktoken.

    Args:
        model_path: path to tokenizer vocabulary
        num_special_tokens: number of special tokens to generate
        special_tokens: template for user-defined special tokens
        pattern: Regex pattern to split the text
    """

    def __init__(
        self,
        vocab_file: str,
        pattern: str = PATTERN_TIKTOKEN,
        vocab_size: int = DEFAULT_TIKTOKEN_MAX_VOCAB,  # 131072
        num_special_tokens: int = 1000,
        special_tokens: Optional[List[str]] = None,
    ):
        if not vocab_file or not os.path.exists(vocab_file):
            raise ValueError(f"vocab_file: {vocab_file} is invalid")

        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS.copy()

        assert len(special_tokens) == len(set(special_tokens)), f"Special tokens should be unique: {special_tokens}"
        assert len(special_tokens) <= num_special_tokens < vocab_size
        assert set(SPECIAL_TOKENS) <= set(special_tokens), f"Custom special tokens should include {SPECIAL_TOKENS}"

        self._unk_id = special_tokens.index("<unk>")
        self._bos_id = special_tokens.index("<s>")
        self._eos_id = special_tokens.index("</s>")

        self._vocab_size = vocab_size
        print(f'{self._vocab_size = }')
        self.num_special_tokens = num_special_tokens
        special_filler = [SPECIAL_TOKEN_TEMPLATE.format(id=i) for i in range(len(special_tokens), num_special_tokens)]
        if special_filler:
            print(f"Adding special tokens {special_filler[0]}, ..., {special_filler[-1]}")
        self.special_tokens = special_tokens + special_filler
        assert len(set(self.special_tokens)) == len(self.special_tokens) == num_special_tokens, self.special_tokens
        self.inner_vocab_size = vocab_size - num_special_tokens

        # reload vocab
        self.token2id = reload_mergeable_ranks(vocab_file, max_vocab=self.inner_vocab_size)
        self.id2token = {v: k for k, v in self.token2id.items()}
        assert set(range(self.inner_vocab_size)) == set(self.id2token.keys())

        self.shifted_id2token = {i: tok for i, tok in enumerate(self.special_tokens)}
        for key, value in self.id2token.items():
            self.shifted_id2token[key + self.num_special_tokens] = value

        self.tokenizer = tiktoken.Encoding(
            name=Path(vocab_file).parent.name,
            pat_str=pattern,
            mergeable_ranks=self.token2id,
            special_tokens={},  # special tokens are handled manually
        )

    def text_to_tokens(self, text: str):
        token_ids = self.tokenizer.encode(text)
        return [self.tokenizer.decode_single_token_bytes(token) for token in token_ids]

    def tokens_to_text(self, tokens: List[int]):
        token_ids = [self.tokenizer.encode_single_token(tokens) for tokens in tokens]
        return self.tokenizer.decode(token_ids)

    def token_to_id(self, token):
        return self.tokenizer.encode_single_token(token)

    def tokens_to_ids(self, tokens):
        return [self.tokenizer.encode_single_token(token) for token in tokens]

    def ids_to_tokens(self, token_ids):
        tokens = []
        for token_id in token_ids:
            if token_id < self.num_special_tokens:
                tokens.append(self.special_tokens[token_id])
            else:
                token_id -= self.num_special_tokens
                token_bytes = self.tokenizer.decode_single_token_bytes(token_id)
                tokens.append(token_bytes.decode('utf-8', errors='replace'))
        return tokens

    def text_to_ids(self, text: str):
        tokens = self.tokenizer.encode(text)
        tokens = [t + self.num_special_tokens for t in tokens]
        return tokens

    def ids_to_text(self, tokens: List[int]):
        # Filter out special tokens and adjust the remaining tokens
        adjusted_tokens = [
            t - self.num_special_tokens
            for t in tokens
            if t not in {self.bos, self.eos} and t >= self.num_special_tokens
        ]

        # Decode only if there are tokens left after filtering
        if adjusted_tokens:
            return self.tokenizer.decode(adjusted_tokens)
        else:
            return ""  # Return an empty string if all tokens were filtered out

    @property
    def bos_id(self):
        return self._bos_id

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def vocab(self):
        return self.token2id

    @property
    def decoder(self):
        return self.shifted_id2token

    @property
    def encoder(self):
        return self.vocab

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
