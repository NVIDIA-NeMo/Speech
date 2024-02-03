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

from functools import cached_property
from pathlib import Path
from typing import Dict

from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model

__all__ = ['CanaryTokenizer']


LANGUAGES = {
    "en": "english",
    "de": "german",
    "es": "spanish",
    "fr": "french",
}

TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
}

SPECIAL_TOKENS = [
    "<pad>",
    "<|endoftext|>",
    "<|startoftranscript|>",
    *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())],
    "<|transcribe|>",
    "<|translate|>",
    "<|nopnc|>",
    "<|pnc|>",
    "<|nospeech|>",
]

UNUSED_SPECIAL_TOKENS = [f"<|spltoken{i}|>" for i in range(18)]


class CanaryTokenizer(AggregateTokenizer):
    """
    Thin wrapper around AggregateTokenizer to provide quick access to special tokens
    """

    def __init__(self, tokenizers: Dict):
        super().__init__(tokenizers)

        # for easy access of special tokens
        special_tokens: Dict[str, int] = {}
        for special in SPECIAL_TOKENS:
            special_tokens[special] = self.token_to_id(special, lang_id='spl_tokens')

        self.special_tokens = special_tokens

    @cached_property
    def eos_id(self) -> int:
        return self.special_tokens["<|endoftext|>"]

    @cached_property
    def bos_id(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def transcribe_id(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate_id(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def nopnc_id(self) -> int:
        return self.special_tokens["<|nopnc|>"]

    @cached_property
    def pnc_id(self) -> int:
        return self.special_tokens["<|pnc|>"]

    @cached_property
    def nospeech_id(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def pad_id(self) -> int:
        return self.special_tokens["<pad>"]

    def to_language_id(self, language):
        if token_id := self.special_tokens.get(f"<|{language}|>", None):
            return token_id

        raise KeyError(f"Language {language} not found in tokenizer.")

    @staticmethod
    def build_special_tokenizer(output_dir: str | Path) -> SentencePieceTokenizer:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        text_path = output_dir / "train_text.txt"
        all_tokens = SPECIAL_TOKENS + UNUSED_SPECIAL_TOKENS
        train_text = "\n".join(all_tokens)
        text_path.write_text(train_text)
        model_path = output_dir / "tokenizer.model"
        create_spt_model(
            str(text_path),
            vocab_size=32,
            sample_size=-1,
            do_lower_case=False,
            output_dir=str(output_dir),
            user_defined_symbols=all_tokens,
        )
        return SentencePieceTokenizer(str(model_path))
