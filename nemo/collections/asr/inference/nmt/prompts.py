# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import re
import textwrap
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


class RivaTranslatorPromptTemplate(PromptTemplate):
    """
    Prompt template for the NVIDIA Riva-Translate-4B-Instruct model
    (https://huggingface.co/nvidia/Riva-Translate-4B-Instruct).
    """

    SYSTEM_MESSAGE_TEMPLATE = "You are an expert at translating text from {src_lang} to {tgt_lang}."
    USER_CONTENT_TEMPLATE = "What is the {tgt_lang} translation of the sentence: {src_text}?"

    PROMPT_TEMPLATE = "<s>System\n{system_message}</s>\n<s>User\n{user_content}</s>\n<s>Assistant\n{tgt_text}"

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
        Generate a translation prompt for the Riva-Translate model.
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
        src_text = re.sub(r"\s+", " ", src_text).strip()
        tgt_text = re.sub(r"\s+", " ", tgt_text).strip()

        system_message = cls.SYSTEM_MESSAGE_TEMPLATE.format(src_lang=src_lang, tgt_lang=tgt_lang)
        user_content = cls.USER_CONTENT_TEMPLATE.format(tgt_lang=tgt_lang, src_text=src_text)
        return cls.PROMPT_TEMPLATE.format(system_message=system_message, user_content=user_content, tgt_text=tgt_text)

    @classmethod
    def extract(cls, response: str) -> str:
        """
        Extract the first line of text from a model response (Riva-Translate is not a reasoning
        model, so no <think> block stripping is needed -- mirrors EuroLLMTranslatorPromptTemplate).
        Args:
            response (str): The full response from the model.
        Returns:
            str: The text before the first newline.
        """
        return response.split('\n')[0]


class QwenReasoningTranslatorPromptTemplate(PromptTemplate):
    """
    Chat-style prompt template for Qwen Reasoning model to perform translation.
    Thinking is disabled by appending the empty think block (<think>\\n\\n</think>\\n\\n)
    after <|im_start|>assistant\\n, matching the tokenizer.apply_chat_template(..., enable_thinking=False) behavior.

    The system and user prompts reproduce those used in the NVIDIA NeMo team's IWSLT 2026 submission:
    Grigoryan, Bataev, Andrusenko, et al. (2026), "NeMo@IWSLT 2026: Cascaded System for Simultaneous
    Speech Translation" (https://aclanthology.org/2026.iwslt-1.23.pdf).
    """

    SYSTEM_MESSAGE = textwrap.dedent(
        """\
        You are a professional machine translation assistant.
        Translate the input text into the target language.
        - Output only the translation.
        - Do not complete or extend the text.
        - The input may be incomplete; preserve incompleteness.
        - Do not infer missing content.
        - Stop immediately after translating.
        - Preserve named entities, numbers, punctuation, and formatting.\
        """
    )

    # Empty think block appended so model goes straight to answer (enable_thinking=False behavior)
    _THINK_DISABLED_SUFFIX = "<think>\n\n</think>\n\n"

    USER_CONTENT_TEMPLATE = (
        "Translate the following {src_lang} source text to {tgt_lang}:\n" "{src_lang}: {src_text}\n" "{tgt_lang}: "
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
        Generate a translation prompt in Qwen3 chat format (thinking disabled).
        Args:
            src_lang, tgt_lang, src_prefix, tgt_prefix, src_context, tgt_context: same as other templates.
        Returns:
            str: Formatted prompt string.
        """
        src_text = f"{src_context} {src_prefix}"
        tgt_text = f"{tgt_context} {tgt_prefix}"
        src_text = re.sub(r"\s+", " ", src_text).strip()
        tgt_text = re.sub(r"\s+", " ", tgt_text).strip()
        user_content = cls.USER_CONTENT_TEMPLATE.format(
            src_lang=src_lang, tgt_lang=tgt_lang, src_text=src_text, tgt_text=tgt_text
        )
        assistant_text = f"{cls._THINK_DISABLED_SUFFIX}{tgt_text}"

        system_block = f"<|im_start|>system\n{cls.SYSTEM_MESSAGE}<|im_end|>\n"
        return (
            system_block + f"<|im_start|>user\n{user_content}<|im_end|>\n" + f"<|im_start|>assistant\n{assistant_text}"
        )

    @classmethod
    def messages(
        cls,
        src_lang: str,
        tgt_lang: str,
        src_prefix: str,
        tgt_prefix: str,
        src_context: str = "",
        tgt_context: str = "",
    ):
        """
        Return chat messages for tokenizer.apply_chat_template().
        System message instructs the model not to use <think> (thinking disabled).
        """
        src_text = re.sub(r"\s+", " ", f"{src_context} {src_prefix}".strip()).strip()
        tgt_text = re.sub(r"\s+", " ", f"{tgt_context} {tgt_prefix}".strip()).strip()
        user_content = cls.USER_CONTENT_TEMPLATE.format(
            src_lang=src_lang, tgt_lang=tgt_lang, src_text=src_text, tgt_text=tgt_text
        )
        return [
            {"role": "system", "content": cls.SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": tgt_text},
        ]

    @classmethod
    def extract(cls, response: str) -> str:
        """
        Extract the translation from the model response. Strips any think block
        (<think>...</think>) so only the actual translation is returned (thinking disabled
        at decode time). Falls back to first line if no think block is present.
        """
        response = response.strip()
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        if "<think>" in response:
            response = response.split("<think>")[-1].strip()
        if not response:
            return ""

        # Remove any trailing punctuation to reduce the risk of hallucination
        response = response.removesuffix("...")
        if not response:
            return ""

        parts = response.rsplit(maxsplit=1)
        parts[-1] = parts[-1].replace("...", "")
        response = " ".join(parts)
        return response
