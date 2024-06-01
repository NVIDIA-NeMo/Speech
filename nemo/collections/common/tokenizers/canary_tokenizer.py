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
CANARY_BOS = "<|startoftranscript|>"
CANARY_EOS = "<|endoftext|>"
CANARY_PAD = "<pad>"
CANARY_NOSPEECH = "<|nospeech|>"
CANARY_PNC = "<|pnc|>"
CANARY_NOPNC = "<|nopnc|>"
DEFAULT_TOKENS = [CANARY_NOSPEECH, CANARY_PAD, CANARY_EOS, CANARY_BOS, CANARY_PNC, CANARY_NOPNC]

CANARY_SPECIAL_TOKENIZER = "spl_tokens"


class CanaryTokenizer(AggregateTokenizer):
    """
    Thin wrapper around AggregateTokenizer to provide quick access to special tokens
    """

    def __init__(self, tokenizers: Dict):
        super().__init__(tokenizers)

        # for easy access of special tokens
        self.special_tokens = {}
        for special in tokenizers[CANARY_SPECIAL_TOKENIZER].vocab:
            # Search for special prompting tokens
            if (special.startswith("<|") and special.endswith("|>")) or special == CANARY_PAD:
                self.special_tokens[special] = self.token_to_id(special, lang_id=CANARY_SPECIAL_TOKENIZER)

    @cached_property
    def eos_id(self) -> int:
        return self.special_tokens[CANARY_EOS]

    @cached_property
    def bos_id(self) -> int:
        return self.special_tokens[CANARY_BOS]

    @cached_property
    def nospeech_id(self) -> int:
        return self.special_tokens[CANARY_NOSPEECH]

    @cached_property
    def pad_id(self) -> int:
        return self.special_tokens[CANARY_PAD]

    def text_to_ids(self, text, lang_id) -> list[int]:
        if lang_id == CANARY_SPECIAL_TOKENIZER:
            return self._tokenize_special_prompt(text)
        if text.endswith(CANARY_EOS):
            return super().text_to_ids(text[: -len(CANARY_EOS)], lang_id) + [self.eos_id]
        return super().text_to_ids(text[-len(CANARY_EOS) :], lang_id)

    def _tokenize_special_prompt(self, text: str) -> list[int]:
        """
        Tokenize the input special prompt of the following schema:

        <|startoftranscript|><|source_lang|><|taskname|><|target_lang|><|pnc|>

        Required because otherwise self.text_to_ids() returns a different result than what Canary had been trained with.
        """
        ans = []
        assert text.count('>') == 5, f"Expected exactly 5 special tokens in Canary's prompt, got: {text}."
        assert text.startswith(CANARY_BOS), text
        for _ in range(5):
            token = text[: text.find(">") + 1]
            ans.append(self.special_tokens[token])
            text = text[len(token) :]
        assert len(text) == 0, text
        return ans

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
