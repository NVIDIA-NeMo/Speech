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
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tiktoken
import torch

PATTERN_TIKTOKEN = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
DEFAULT_TIKTOKEN_MAX_VOCAB = 2**17  # 131072
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>"]
SPECIAL_TOKEN_TEMPLATE = "<SPECIAL_{id}>"


def reload_mergeable_ranks(
    path: str,
    max_vocab: Optional[int] = None,
) -> Dict[bytes, int]:
    """
    Reload the tokenizer JSON file and convert it to Tiktoken format.
    """
    assert path.endswith(".json")

    # reload vocab
    with open(path, "r", encoding='utf-8') as f:
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


class TiktokenTokenizer:
    def __init__(self, vocab_file: str):

        self.num_special_tokens = 1000
        vocab_size = DEFAULT_TIKTOKEN_MAX_VOCAB
        pattern = PATTERN_TIKTOKEN
        special_tokens = SPECIAL_TOKENS.copy()
        inner_vocab_size = vocab_size - self.num_special_tokens

        token2id = reload_mergeable_ranks(vocab_file, max_vocab=inner_vocab_size)
        self.tokenizer = tiktoken.Encoding(
            name=Path(vocab_file).parent.name,
            pat_str=pattern,
            mergeable_ranks=token2id,
            special_tokens={},  # special tokens are handled manually
        )

        # BOS / EOS / Pad token IDs
        self._bos_id = special_tokens.index("<s>")
        self._eos_id = special_tokens.index("</s>")

    def encode(self, text):
        tokens = self.tokenizer.encode(text)
        tokens = [t + self.num_special_tokens for t in tokens]
        return tokens

    def decode(self, tokens):
        # Filter out special tokens and adjust the remaining tokens
        adjusted_tokens = [
            t - self.num_special_tokens
            for t in tokens
            if t not in {self._bos_id, self._eos_id} and t >= self.num_special_tokens
        ]

        # Decode only if there are tokens left after filtering
        if adjusted_tokens:
            return self.tokenizer.decode(adjusted_tokens)
        else:
            return ""  # Return an empty string if all tokens were filtered out

    def batch_decode(self, ids):
        if isinstance(ids, np.ndarray) or torch.is_tensor(ids):
            ids = ids.tolist()

        if isinstance(ids[0], list):
            ids = ids[0]

        return self.decode(ids)

    @property
    def pad_id(self):
        return self._eos_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def eos_token_id(self):
        return self._eos_id
