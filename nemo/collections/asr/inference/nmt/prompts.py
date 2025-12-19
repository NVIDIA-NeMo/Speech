# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import re
from abc import ABC, abstractmethod


class PromptTemplate(ABC):
    """
    Base class for prompt templates.
    Derived classes should implement the format and extract methods.
        - format: format the prompt template with the given arguments
        - extract: extract the answer from the response
    """

    @classmethod
    @abstractmethod
    def format(cls, **kwargs) -> str:
        """
        Format the prompt template with the given arguments.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def extract(cls, response: str) -> str:
        """
        Extract the answer from the response.
        """
        raise NotImplementedError()


class EuroLLMTranslatorPromptTemplate(PromptTemplate):
    """
    Provides a prompt template for the EuroLLM model to perform translation.
    """

    PROMPT_TEMPLATE = (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n"
        "Translate the following {src_lang} source text to {tgt_lang}. Always output text in the {tgt_lang} language:\n"
        "{src_lang}: {src_text}\n"
        "{tgt_lang}: <|im_end|>\n"
        "<|im_start|>assistant\n"
        "{tgt_text}"
    )

    @classmethod
    def format(
        cls,
        src_lang: str,
        tgt_lang: str,
        src_prefix: str,
        tgt_prefix: str,
        src_context: str = "",
        tgt_context: str = "",
    ) -> str:
        """
        Generate a translation prompt for the EuroLLM model.
        Args:
            src_lang (str): Source language name.
            tgt_lang (str): Target language name.
            src_prefix (str): Source text to translate.
            tgt_prefix (str): Optional target prefix or placeholder for completion.
            src_context (str): Optional source context to start translation from.
            tgt_context (str): Optional target context to start translation from.
        Returns:
            str: Formatted translation prompt.
        """
        src_text = f"{src_context} {src_prefix}"
        tgt_text = f"{tgt_context} {tgt_prefix}"
        src_text = re.sub(r'\s+', ' ', src_text).strip()
        tgt_text = re.sub(r'\s+', ' ', tgt_text).strip()
        return cls.PROMPT_TEMPLATE.format(src_lang=src_lang, tgt_lang=tgt_lang, src_text=src_text, tgt_text=tgt_text)

    @classmethod
    def extract(cls, response: str) -> str:
        """
        Extract the first line of text from a model response.
        Args:
            response (str): The full response from the model.
        Returns:
            str: The text before the first newline.
        """
        return response.split('\n')[0]
