# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List

from nemo.utils.dependency import assert_optional_dependency_available


class MosesProcessor:
    """
    Tokenizer, Detokenizer and Normalizer utilities in Moses
    """

    def __init__(self, lang_id: str):
        assert_optional_dependency_available("sacremoses")
        from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

        self.moses_tokenizer = MosesTokenizer(lang=lang_id)
        self.moses_detokenizer = MosesDetokenizer(lang=lang_id)
        self.normalizer = MosesPunctNormalizer(lang=lang_id)

    def detokenize(self, tokens: List[str]) -> str:
        """
        Detokenizes a list of tokens
        Args:
            tokens: list of strings as tokens
        Returns:
            detokenized string
        """
        return self.moses_detokenizer.detokenize(tokens)

    def tokenize(self, text: str):
        """
        Tokenizes text using Moses -> Sentencepiece.
        """
        return self.moses_tokenizer.tokenize(text, escape=False, return_str=True)

    def normalize(self, text: str):
        """Normalize text with Moses punctuation normalization."""
        return self.normalizer.normalize(text)
