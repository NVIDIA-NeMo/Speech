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


import os
import string

import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.inference.nmt.prompts import EuroLLMTranslatorPromptTemplate, PromptTemplate

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise ImportError("Failed to import vLLM.") from e

from nemo.utils import logging

EURO_LLM_INSTRUCT_SMALL = "utter-project/EuroLLM-1.7B-Instruct"
EURO_LLM_INSTRUCT_LARGE = "utter-project/EuroLLM-9B-Instruct"
SUPPORTED_TRANSLATION_MODELS = [EURO_LLM_INSTRUCT_SMALL, EURO_LLM_INSTRUCT_LARGE]


class LLMTranslator:
    """
    A vLLM-based LLM translator for ASR transcripts.
    It takes ASR transcripts and prefixes to start translation from, and returns corresponding continuations of translations.
    """

    def __init__(
        self,
        model_name: str,
        source_language: str,
        target_language: str,
        waitk: int = -1,
        device: str = "cuda",
        device_id: int = 0,
        batch_size: int = -1,
        llm_params: dict | DictConfig | None = None,
        sampling_params: dict | DictConfig | None = None,
    ):
        """
        A model for translating ASR transcripts with LLM.
        Args:
            model_name: (str) path to the model name on HuggingFace.
            source_language: (str) source language
            target_language: (str) target language
            waitk: (int) sets the maximum number of words the translation is allowed to lag behind the ASR transcript.
                         If the translation falls more than waitk words behind, it automatically extends the prefix
                         using the current translation. -1 disables this rule and relies on the longest common prefix (LCP)
                         between current and previous translations. Larger values of waitk lead to more coherent translations,
                         but the cost of generating the translation increases, because the model needs to generate more tokens.
            device: (str) device to run the model on
            device_id: (int) device ID to run the model on
            batch_size: (int) batch size for the LLM model, in case of -1, the batch size is set to the number of ASR transcripts
            llm_params: (dict | DictConfig | None) parameters for the LLM model
            sampling_params: (dict | DictConfig | None) parameters for the sampling
        """
        self.model_name = model_name
        if model_name not in SUPPORTED_TRANSLATION_MODELS:
            raise ValueError(
                f"Model {model_name} is not supported for translation. Supported models are: {SUPPORTED_TRANSLATION_MODELS}"
            )

        llm_params = self.convert_to_dict(llm_params)
        sampling_params = self.convert_to_dict(sampling_params)

        self.device_str, self.device_id = self.setup_device(device, device_id)

        self.batch_size = batch_size
        self.split_batch = self.batch_size > 0

        self.nmt_model = self.load_model(llm_params)
        self.sampling_params = SamplingParams(**sampling_params)

        self.source_language = source_language
        self.target_language = target_language
        self.prompt_template = self.get_prompt_template(model_name)
        self.waitk = waitk

    @staticmethod
    def convert_to_dict(params: dict | DictConfig | None) -> dict:
        """
        Convert DictConfig to dict.
        Args:
            params: (dict | DictConfig | None) parameters to convert
        Returns:
            dict: converted parameters
        """
        if params is None:
            return dict()
        if isinstance(params, DictConfig):
            return OmegaConf.to_container(params)
        return params

    @staticmethod
    def setup_device(device: str, device_id: int) -> tuple[str, int]:
        """
        Setup device for the LLM model.
        Args:
            device: (str) device to run the model on
            device_id: (int) device ID to run the model on
        Returns:
            device_str: (str) device string, e.g. "cuda:1"
            device_id: (int) device ID, e.g. 1
        Raises:
            ValueError: if device is not supported, or CUDA is not available
        """
        if device == "cpu":
            raise ValueError("Currently, CPU is not supported for vLLM.")

        if device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available.")

            if device_id >= torch.cuda.device_count():
                logging.warning(f"Device ID {device_id} is not available. Using GPU 0 instead.")
                device_id = 0

            device_str = f"cuda:{device_id}"
            return device_str, device_id

        raise ValueError(f"Unsupported device: {device}")

    @staticmethod
    def get_prompt_template(model_name: str) -> PromptTemplate:
        """
        Returns prompt template for the LLM model.
        Args:
            model_name: (str) name of the model to get prompt template for
        Returns:
            PromptTemplate: prompt template for the LLM model
        Raises:
            ValueError: if model is not supported for translation
        """
        if model_name in [EURO_LLM_INSTRUCT_SMALL, EURO_LLM_INSTRUCT_LARGE]:
            return EuroLLMTranslatorPromptTemplate

        raise ValueError(
            f"Model {model_name} is not supported for translation. Supported models are: {SUPPORTED_TRANSLATION_MODELS}"
        )

    def load_model(self, llm_params: dict) -> LLM:
        """
        Load NMT model in vLLM format.
        Args:
            llm_params: (dict) parameters for the LLM model
        Returns:
            Loaded LLM instance.
        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
            model = LLM(model=self.model_name, **llm_params)
            return model
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def translate_batch(
        self,
        asr_transcripts: list[str],
        prefixes: list[str],
        src_langs: list[str],
        tgt_langs: list[str],
        src_contexts: list[str],
        tgt_contexts: list[str],
    ) -> list[str]:
        """
        Translate ASR transcripts starting from pre-defined prefixes in target language.
        Args:
            asr_transcripts: (list[str]) batch of ASR transcripts to be translated
            prefixes: (list[str]) batch of prefixes to start translation from
            src_langs: (list[str]) batch of source languages
            tgt_langs: (list[str]) batch of target languages
            src_contexts: (list[str]) batch of source contexts
            tgt_contexts: (list[str]) batch of target contexts
        Returns:
            list[str] translations of ASR transcripts
        """
        input_texts = []
        for src_lang, tgt_lang, src_prefix, tgt_prefix, src_context, tgt_context in zip(
            src_langs, tgt_langs, asr_transcripts, prefixes, src_contexts, tgt_contexts
        ):
            text = self.prompt_template.format(src_lang, tgt_lang, src_prefix, tgt_prefix, src_context, tgt_context)
            input_texts.append(text)

        outputs = self.nmt_model.generate(input_texts, self.sampling_params, use_tqdm=False)
        translations = []
        for tgt_prefix, output in zip(prefixes, outputs):
            output_text = output.outputs[0].text
            output_text = self.prompt_template.extract(output_text)
            translations.append(f"{tgt_prefix}{output_text}")
        return translations

    def translate(
        self,
        asr_transcripts: list[str],
        prefixes: list[str],
        src_langs: list[str],
        tgt_langs: list[str],
        src_contexts: list[str],
        tgt_contexts: list[str],
    ) -> list[str]:
        """
        Translate ASR transcript starting from pre-defined prefix in target language.
        Args:
            asr_transcripts: (list[str]) ASR transcripts to be translated
            prefixes: (list[str]) prefixes to start translation from
            src_langs: (list[str]) source languages
            tgt_langs: (list[str]) target languages
            src_contexts: (list[str]) source contexts
            tgt_contexts: (list[str]) target contexts
        Returns:
            list[str] translations of ASR transcripts
        """
        all_translations = []
        n_requests = len(asr_transcripts)
        bs = self.batch_size if self.split_batch else n_requests
        for i in range(0, n_requests, bs):
            all_translations.extend(
                self.translate_batch(
                    asr_transcripts=asr_transcripts[i : i + bs],
                    prefixes=prefixes[i : i + bs],
                    src_langs=src_langs[i : i + bs],
                    tgt_langs=tgt_langs[i : i + bs],
                    src_contexts=src_contexts[i : i + bs],
                    tgt_contexts=tgt_contexts[i : i + bs],
                )
            )
        return all_translations

    def get_prefixes(
        self,
        asr_transcripts: list[str],
        translations: list[str],
        prev_translations: list[str],
    ) -> list[str]:
        """
        Generates new prefixes in target language for the next translation step.
        Args:
            asr_transcripts: (list[str]) current ASR transcripts to be translated
            translations: (list[str]) translations obtained with LLM on current step
            prev_translations: (list[str]) translations obtained with LLM on previous step
        Returns:
            list[str] new prefixes for LLM translation
        """

        new_prefixes = []
        for asr, trans, prev_trans in zip(asr_transcripts, translations, prev_translations):

            # Longest common prefix of translations on current and previous steps
            lcp = os.path.commonprefix([prev_trans, trans])
            had_leading_space = lcp.startswith(" ")

            # If lcp happens mid-word, remove generated ending up to the first full word
            if (len(lcp) > 0) and (lcp[-1] not in f"{string.punctuation} "):
                lcp = " ".join(lcp.split()[:-1])

            # Remove trailing whitespaces
            lcp = lcp.strip()

            # Remove hallucinations if ASR transcript is empty string
            if len(asr) == 0:
                lcp = ""

            # If the LLM-generated translations disagree too much between steps,
            # and the translation falls more than waitk words behind the ASR transcript,
            # the algorithm forcibly advances the prefix based on the current translation.
            n_asr_words = len(asr.split())
            n_lcp_words = len(lcp.split())
            if (self.waitk > 0) and (n_asr_words - n_lcp_words > self.waitk):
                num_words_to_pick = n_asr_words - self.waitk
                new_prefix = " ".join(trans.split()[:num_words_to_pick])
            else:
                new_prefix = lcp

            # Preserve leading space if it was present in the previous translation
            if len(new_prefix) > 0 and had_leading_space and not new_prefix.startswith(" "):
                new_prefix = " " + new_prefix

            new_prefixes.append(new_prefix)

        return new_prefixes
