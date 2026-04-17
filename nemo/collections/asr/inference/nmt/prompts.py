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
import textwrap


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

    SYSTEM_MESSAGE = ""
    USER_CONTENT_TEMPLATE = (
        "Translate the following {src_lang} source text to {tgt_lang}:\n"
        "{src_lang}: {src_text}\n"
        "{tgt_lang}: "
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
        system_content: str | None = None,
        user_content_template: str | None = None,
        use_system: bool = True,
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
        user_template = user_content_template or cls.USER_CONTENT_TEMPLATE
        user_content = user_template.format(src_lang=src_lang, tgt_lang=tgt_lang, src_text=src_text, tgt_text=tgt_text)

        prompt = ""
        if use_system:
            system = cls.SYSTEM_MESSAGE if system_content is None else system_content
            prompt += f"<|im_start|>system\n{system}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{tgt_text}"
        return prompt

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


class EuroLLMTranslatorPromptTemplateV2(EuroLLMTranslatorPromptTemplate):
    PROMPT_TEMPLATE = (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n"
        "Translate the following {src_lang} source text to {tgt_lang}. "
        "Preserve all named entities, such as person names, model names, dataset names, and metric names, exactly as they appear in the input text:\n"
        "{src_lang}: {src_text}\n"
        "{tgt_lang}: <|im_end|>\n"
        "<|im_start|>assistant\n"
        "{tgt_text}"
    )

# class EuroLLMChatTranslatorPromptTemplate(PromptTemplate):
#     """
#     Chat-style prompt template for EuroLLM translation, matching the format used
#     with tokenizer.apply_chat_template() and the official EuroLLM system message.
#     """

#     SYSTEM_MESSAGE = (
#         "You are EuroLLM --- an AI assistant specialized in European languages "
#         "that provides safe, educational and helpful answers."
#     )

#     PROMPT_TEMPLATE = (
#         "<|im_start|>system\n{system_content}<|im_end|>\n"
#         "<|im_start|>user\n"
#         "Translate the following {src_lang} source text to {tgt_lang}:\n"
#         "{src_lang}: {src_text}\n"
#         "{tgt_lang}: <|im_end|>\n"
#         "<|im_start|>assistant\n"
#         "{tgt_text}"
#     )

#     @classmethod
#     def format(
#         cls,
#         src_lang: str,
#         tgt_lang: str,
#         src_prefix: str,
#         tgt_prefix: str,
#         src_context: str = "",
#         tgt_context: str = "",
#         system_content: str = None,
#     ) -> str:
#         """
#         Generate a translation prompt in EuroLLM chat format.
#         Args:
#             src_lang (str): Source language name.
#             tgt_lang (str): Target language name.
#             src_prefix (str): Source text to translate.
#             tgt_prefix (str): Optional target prefix or placeholder for completion.
#             src_context (str): Optional source context to start translation from.
#             tgt_context (str): Optional target context to start translation from.
#             system_content (str, optional): Override system message; defaults to EuroLLM system message.
#         Returns:
#             str: Formatted translation prompt (same token sequence as apply_chat_template).
#         """
#         src_text = f"{src_context} {src_prefix}"
#         tgt_text = f"{tgt_context} {tgt_prefix}"
#         src_text = re.sub(r'\s+', ' ', src_text).strip()
#         tgt_text = re.sub(r'\s+', ' ', tgt_text).strip()
#         system = system_content if system_content is not None else cls.SYSTEM_MESSAGE
#         return cls.PROMPT_TEMPLATE.format(
#             system_content=system,
#             src_lang=src_lang,
#             tgt_lang=tgt_lang,
#             src_text=src_text,
#             tgt_text=tgt_text,
#         )

#     @classmethod
#     def messages(
#         cls,
#         src_lang: str,
#         tgt_lang: str,
#         src_prefix: str,
#         tgt_prefix: str,
#         src_context: str = "",
#         tgt_context: str = "",
#         system_content: str = None,
#     ):
#         """
#         Return chat messages (list of dicts) for use with tokenizer.apply_chat_template().
#         """
#         src_text = re.sub(r'\s+', ' ', f"{src_context} {src_prefix}".strip()).strip()
#         tgt_text = re.sub(r'\s+', ' ', f"{tgt_context} {tgt_prefix}".strip()).strip()
#         system = system_content if system_content is not None else cls.SYSTEM_MESSAGE
#         user_content = (
#             f"Translate the following {src_lang} source text to {tgt_lang}:\n"
#             f"{src_lang}: {src_text}\n"
#             f"{tgt_lang}: "
#         )
#         return [
#             {"role": "system", "content": system},
#             {"role": "user", "content": user_content},
#             {"role": "assistant", "content": tgt_text},
#         ]

#     @classmethod
#     def extract(cls, response: str) -> str:
#         """
#         Extract the first line of text from a model response.
#         Args:
#             response (str): The full response from the model.
#         Returns:
#             str: The text before the first newline.
#         """
#         return response.split('\n')[0]


class Qwen3TranslatorPromptTemplate(PromptTemplate):
    """
    Chat-style prompt template for Qwen3 translation.
    Uses <|im_start|>user / <|im_start|>assistant format.
    Thinking is disabled by appending the empty think block (<think>\\n\\n</think>\\n\\n)
    after <|im_start|>assistant\\n, matching tokenizer.apply_chat_template(..., enable_thinking=False).
    """

    # SYSTEM_MESSAGE = (
    #     textwrap.dedent("""You are a professional machine translation assistant. Translate the input text into the target language. Output text only in target language.""")
    # )

    SYSTEM_MESSAGE = """
    You are a professional machine translation assistant.
    Translate the input text into the target language.
    """
    
    SYSTEM_MESSAGE = (
        """
            You are a professional machine translation assistant.
            Translate the input text into the target language.

            - Output only the translation.
            - Do not complete or extend the text.
            - The input may be incomplete. Preserve incompleteness.
            - Do not infer missing content.
            - Stop immediately after translating.
            - Preserve named entities, numbers, punctuation, and formatting.
        """
    )

    _CHAT_TOKENS = ("<|im_start|>", "<|im_end|>")

    # Empty think block appended so model goes straight to answer (enable_thinking=False behavior)
    _THINK_DISABLED_SUFFIX = "<think>\n\n</think>\n\n"

    USER_CONTENT_TEMPLATE = (
        "Translate the following {src_lang} source text to {tgt_lang}:\n"
        "{src_lang}: {src_text}\n"
        "{tgt_lang}: "
    )
    
    # USER_CONTENT_TEMPLATE = (
    #     "Translate the following segment into {tgt_lang}, without additional explanation.\n\n"
    #     "{src_text}<|im_end|>\n"
    # )

    @classmethod
    def format(
        cls,
        src_lang: str,
        tgt_lang: str,
        src_prefix: str,
        tgt_prefix: str,
        src_context: str = "",
        tgt_context: str = "",
        system_content: str = None,
        user_content_template: str | None = None,
        use_system: bool = True,
    ) -> str:
        """
        Generate a translation prompt in Qwen3 chat format (thinking disabled).
        Args:
            src_lang, tgt_lang, src_prefix, tgt_prefix, src_context, tgt_context: same as other templates.
            system_content: Override system message; used only if use_system is True.
            use_system: If True, prepend system message (default). Set False for user-only prompt.
        Returns:
            str: Formatted prompt string.
        """
        src_text = f"{src_context} {src_prefix}"
        tgt_text = f"{tgt_context} {tgt_prefix}"
        src_text = re.sub(r"\s+", " ", src_text).strip()
        tgt_text = re.sub(r"\s+", " ", tgt_text).strip()
        user_template = user_content_template or cls.USER_CONTENT_TEMPLATE
        user_content = user_template.format(src_lang=src_lang, tgt_lang=tgt_lang, src_text=src_text, tgt_text=tgt_text)
        assistant_text = f"{cls._THINK_DISABLED_SUFFIX}{tgt_text}"

        if use_system:
            system = system_content if system_content is not None else cls.SYSTEM_MESSAGE
            start, end = cls._CHAT_TOKENS
            system_block = f"{start}system\n{system}{end}\n"
            return (
                system_block
                + f"<|im_start|>user\n{user_content}<|im_end|>\n"
                + f"<|im_start|>assistant\n{assistant_text}"
            )
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_text}"

    @classmethod
    def messages(
        cls,
        src_lang: str,
        tgt_lang: str,
        src_prefix: str,
        tgt_prefix: str,
        src_context: str = "",
        tgt_context: str = "",
        system_content: str = None,
        user_content_template: str | None = None,
        use_system: bool = True,
    ):
        """
        Return chat messages for tokenizer.apply_chat_template().
        System message instructs the model not to use <think> (thinking disabled).
        """
        src_text = re.sub(r"\s+", " ", f"{src_context} {src_prefix}".strip()).strip()
        tgt_text = re.sub(r"\s+", " ", f"{tgt_context} {tgt_prefix}".strip()).strip()
        user_template = user_content_template or cls.USER_CONTENT_TEMPLATE
        user_content = user_template.format(src_lang=src_lang, tgt_lang=tgt_lang, src_text=src_text, tgt_text=tgt_text)
        msgs = [{"role": "user", "content": user_content}, {"role": "assistant", "content": tgt_text}]
        if use_system:
            system = system_content if system_content is not None else cls.SYSTEM_MESSAGE
            msgs.insert(0, {"role": "system", "content": system})
        return msgs

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
        # return response.split('\n')[0]
        if response.endswith("..."):
            response = response[:-3]
        if not response:
            return ""
        
        if "..." in response.split()[-1]:
            words = response.split()
            words[-1] = words[-1].replace("...", "")
            response = " ".join(words)
        if response.endswith("."):
            response = response[:-1]
        if response.endswith("?"):
            response = response[:-1]
        if response.endswith("!"):
            response = response[:-1]
        return response


# class Qwen3ChatTranslatorPromptTemplate(PromptTemplate):
#     """
#     Chat-style Qwen3 prompt template aligned with chat_prompt.py behavior:
#     stricter system instructions and robust assistant-output cleanup.
#     """

#     SYSTEM_MESSAGE = textwrap.dedent(
#         """
#         You are a professional machine translation assistant.
#         Translate the input text into the target language. Output text only in target language.

#         - Output only the translation.
#         - Do not complete or extend the text.
#         - The input may be incomplete. Preserve incompleteness.
#         - Do not infer missing content.
#         - Stop immediately after translating.
#         - Preserve named entities, numbers, punctuation, and formatting.
#         """
#     ).strip()

#     _CHAT_TOKENS = ("<|im_start|>", "<|im_end|>")
#     _THINK_DISABLED_SUFFIX = "<think>\n\n</think>\n\n"

#     PROMPT_TEMPLATE = (
#         "<|im_start|>user\n"
#         "Translate from {src_lang} to {tgt_lang}.\n"
#         "Text:\n{src_text}<|im_end|>\n"
#         "<|im_start|>assistant\n"
#         "<think>\n\n</think>\n\n"
#         "{tgt_text}"
#     )

#     @classmethod
#     def format(
#         cls,
#         src_lang: str,
#         tgt_lang: str,
#         src_prefix: str,
#         tgt_prefix: str,
#         src_context: str = "",
#         tgt_context: str = "",
#         system_content: str = None,
#         use_system: bool = True,
#     ) -> str:
#         src_text = f"{src_context} {src_prefix}"
#         tgt_text = f"{tgt_context} {tgt_prefix}"
#         src_text = re.sub(r"\s+", " ", src_text).strip()
#         tgt_text = re.sub(r"\s+", " ", tgt_text).strip()
#         use_system = False
#         if use_system:
#             system = system_content if system_content is not None else cls.SYSTEM_MESSAGE
#             system = textwrap.dedent(system).strip()
#             start, end = cls._CHAT_TOKENS
#             system_block = f"{start}system\n{system}{end}\n"
#             return system_block + cls.PROMPT_TEMPLATE.format(
#                 src_lang=src_lang,
#                 tgt_lang=tgt_lang,
#                 src_text=src_text,
#                 tgt_text=tgt_text,
#             )
#         return cls.PROMPT_TEMPLATE.format(
#             src_lang=src_lang,
#             tgt_lang=tgt_lang,
#             src_text=src_text,
#             tgt_text=tgt_text,
#         )

#     @classmethod
#     def messages(
#         cls,
#         src_lang: str,
#         tgt_lang: str,
#         src_prefix: str,
#         tgt_prefix: str,
#         src_context: str = "",
#         tgt_context: str = "",
#         system_content: str = None,
#         use_system: bool = True,
#     ):
#         src_text = re.sub(r"\s+", " ", f"{src_context} {src_prefix}".strip()).strip()
#         tgt_text = re.sub(r"\s+", " ", f"{tgt_context} {tgt_prefix}".strip()).strip()
#         user_content = (
#             f"Translate from {src_lang} to {tgt_lang}.\n\n"
#             f"Text:\n{src_text}"
#         )
#         msgs = [{"role": "user", "content": user_content}, {"role": "assistant", "content": tgt_text}]
#         if use_system:
#             system = system_content if system_content is not None else cls.SYSTEM_MESSAGE
#             msgs.insert(0, {"role": "system", "content": textwrap.dedent(system).strip()})
#         return msgs

#     @classmethod
#     def extract(cls, response: str) -> str:
#         text = response.strip()
#         if not text:
#             return ""

#         if "</think>" in text:
#             text = text.split("</think>")[-1].strip()
#         if "<think>" in text:
#             text = text.split("<think>")[-1].strip()

#         if "<|im_start|>assistant" in text:
#             text = text.split("<|im_start|>assistant")[-1].strip()

#         if "\nassistant\n" in text:
#             text = text.split("\nassistant\n")[-1].strip()
#         elif text.startswith("assistant\n"):
#             text = text[len("assistant\n") :].strip()

#         text = text.replace("<|im_end|>", "").strip()

#         prefixes = (
#             "assistant:",
#             "Assistant:",
#             "translation:",
#             "Translation:",
#             "assistant",
#             "Assistant",
#         )
#         for prefix in prefixes:
#             if text.startswith(prefix):
#                 text = text[len(prefix) :].strip()
#                 break

#         lines = [line.strip() for line in text.splitlines() if line.strip()]
#         if not lines:
#             return ""

#         return lines[0].strip().strip('"').strip("'")


# class Qwen3ChatTranslatorPromptTemplate(PromptTemplate):
#     """
#     Chat-style Qwen3 prompt template aligned with chat_prompt.py behavior:
#     stricter system instructions and robust assistant-output cleanup.
#     """

#     SYSTEM_MESSAGE = textwrap.dedent(
#         """
#         You are a professional machine translation assistant.
#         Translate the input text into the target language. Output text only in target language.

#         - Output only the translation.
#         - Do not complete or extend the text.
#         - The input may be incomplete. Preserve incompleteness.
#         - Do not infer missing content.
#         - Stop immediately after translating.
#         - Preserve named entities, numbers, punctuation, and formatting.
#         """
#     ).strip()

#     _CHAT_TOKENS = ("<|im_start|>", "<|im_end|>")
#     _THINK_DISABLED_SUFFIX = "<think>\n\n</think>\n\n"

#     PROMPT_TEMPLATE = (
#         "<|im_start|>user\n"
#         "Translate the following segment into {tgt_lang}, without additional explanation.\n\n"
#         "{src_text}<|im_end|>\n"
#         "<|im_start|>assistant\n"
#         "<think>\n\n</think>\n\n"
#         "{tgt_text}"
#     )

#     @classmethod
#     def format(
#         cls,
#         src_lang: str,
#         tgt_lang: str,
#         src_prefix: str,
#         tgt_prefix: str,
#         src_context: str = "",
#         tgt_context: str = "",
#         system_content: str = None,
#         use_system: bool = True,
#     ) -> str:
#         src_text = f"{src_context} {src_prefix}"
#         tgt_text = f"{tgt_context} {tgt_prefix}"
#         src_text = re.sub(r"\s+", " ", src_text).strip()
#         tgt_text = re.sub(r"\s+", " ", tgt_text).strip()
#         use_system = False
#         if use_system:
#             system = system_content if system_content is not None else cls.SYSTEM_MESSAGE
#             system = textwrap.dedent(system).strip()
#             start, end = cls._CHAT_TOKENS
#             system_block = f"{start}system\n{system}{end}\n"
#             return system_block + cls.PROMPT_TEMPLATE.format(
#                 src_lang=src_lang,
#                 tgt_lang=tgt_lang,
#                 src_text=src_text,
#                 tgt_text=tgt_text,
#             )
#         return cls.PROMPT_TEMPLATE.format(
#             src_lang=src_lang,
#             tgt_lang=tgt_lang,
#             src_text=src_text,
#             tgt_text=tgt_text,
#         )

#     @classmethod
#     def messages(
#         cls,
#         src_lang: str,
#         tgt_lang: str,
#         src_prefix: str,
#         tgt_prefix: str,
#         src_context: str = "",
#         tgt_context: str = "",
#         system_content: str = None,
#         use_system: bool = True,
#     ):
#         src_text = re.sub(r"\s+", " ", f"{src_context} {src_prefix}".strip()).strip()
#         tgt_text = re.sub(r"\s+", " ", f"{tgt_context} {tgt_prefix}".strip()).strip()
#         user_content = (
#             f"Translate the following segment into {tgt_lang}, without additional explanation.\n\n"
#             f"{src_text}"
#         )
#         msgs = [{"role": "user", "content": user_content}, {"role": "assistant", "content": tgt_text}]
#         if use_system:
#             system = system_content if system_content is not None else cls.SYSTEM_MESSAGE
#             msgs.insert(0, {"role": "system", "content": textwrap.dedent(system).strip()})
#         return msgs

#     @classmethod
#     def extract(cls, response: str) -> str:
#         text = response.strip()
#         if not text:
#             return ""

#         if "</think>" in text:
#             text = text.split("</think>")[-1].strip()
#         if "<think>" in text:
#             text = text.split("<think>")[-1].strip()

#         if "<|im_start|>assistant" in text:
#             text = text.split("<|im_start|>assistant")[-1].strip()

#         if "\nassistant\n" in text:
#             text = text.split("\nassistant\n")[-1].strip()
#         elif text.startswith("assistant\n"):
#             text = text[len("assistant\n") :].strip()

#         text = text.replace("<|im_end|>", "").strip()

#         prefixes = (
#             "assistant:",
#             "Assistant:",
#             "translation:",
#             "Translation:",
#             "assistant",
#             "Assistant",
#         )
#         for prefix in prefixes:
#             if text.startswith(prefix):
#                 text = text[len(prefix) :].strip()
#                 break

#         lines = [line.strip() for line in text.splitlines() if line.strip()]
#         if not lines:
#             return ""

#         return lines[0].strip().strip('"').strip("'")


