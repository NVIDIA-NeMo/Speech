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
import os
from functools import cached_property
from pathlib import Path
from typing import Dict, List

from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model

from nemo.utils import logging

__all__ = ['CanaryTokenizer']

# Default tokens for compatibility with Canary.
DEFAULT_TOKENS = ["<|nospeech|>", "<pad>", "<|endoftext|>", "<|startoftranscript|>", "<|pnc|>", "<|nopnc|>"]


class CanaryTokenizer(AggregateTokenizer):
    """
    Thin wrapper around AggregateTokenizer to provide quick access to special tokens
    """

    def __init__(self, tokenizers: Dict):
        super().__init__(tokenizers)

        # for easy access of special tokens
        self.special_tokens = {}
        for special in tokenizers['spl_tokens'].vocab:
            # Search for special prompting tokens
            if (special.startswith("<|") and special.endswith("|>")) or special == "<pad>":
                self.special_tokens[special] = self.token_to_id(special, lang_id='spl_tokens')

    @cached_property
    def eos_id(self) -> int:
        return self.special_tokens["<|endoftext|>"]

    @cached_property
    def bos_id(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def nospeech_id(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def pad_id(self) -> int:
        return self.special_tokens["<pad>"]

    def spl_token_to_id(self, token):
        if token_id := self.special_tokens.get(f"<|{token}|>", None):
            return token_id
        raise KeyError(f"Token {token} not found in tokenizer.")

    @staticmethod
    def build_special_tokenizer(
        tokens: List[str], model_dir: str | Path, force_rebuild: bool = False
    ) -> SentencePieceTokenizer:
        if force_rebuild:
            logging.info("Building special tokenizer")
            # Checks for artifacts of previous build.
            for file in ["tokenizer.model", "tokenizer.vocab", "vocab.txt", "train_text.txt"]:
                if os.path.exists(file):
                    os.remove(file)
        tokens = DEFAULT_TOKENS + [f"<|{t}|>" for t in tokens]
        output_dir = Path(model_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        text_path = output_dir / "train_text.txt"
        train_text = "\n".join(tokens)
        text_path.write_text(train_text)
        model_path = output_dir / "tokenizer.model"
        create_spt_model(
            str(text_path),
            vocab_size=len(tokens) + 2,
            sample_size=-1,
            do_lower_case=False,
            output_dir=str(output_dir),
            user_defined_symbols=tokens,
        )
        spl_tokenizer = SentencePieceTokenizer(str(model_path))
        return spl_tokenizer
