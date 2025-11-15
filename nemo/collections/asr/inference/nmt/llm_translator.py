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

from nemo.collections.asr.inference.nmt.prompts import EuroLLMTranslatorPromptTemplate, PromptTemplate

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise ImportError("Failed to import vLLM.") from e


EURO_LLM_INSTRUCT_SMALL = "utter-project/EuroLLM-1.7B-Instruct"
EURO_LLM_INSTRUCT_LARGE = "utter-project/EuroLLM-9B-Instruct"
SUPPORTED_TRANSLATION_MODELS = [EURO_LLM_INSTRUCT_SMALL, EURO_LLM_INSTRUCT_LARGE]


class LLMTranslator:
    """
    A vLLM-based LLM translator for ASR transcripts.
    """

    def __init__(
        self,
        model_name: str,
        source_language: str,
        target_language: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
        waitk: int = -1,
    ):
        """
        A model for translating ASR transcripts with LLM.
        Args:
            model_name: (str) path to the model name on HuggingFace.
            source_language: (str) source language
            target_language: (str) target language
            max_tokens: (int) maximum number of tokens to generate with LLM
            temperature: (float) LLM sampling temperature, default for translation is 0 (greedy)
            waitk: (int) parameter that controls latency by forcing the generation of new
                   prefix of up to |asr|-waitk words if both translations do not agree
                   on at least of |asr|-waitk words
        """
        self.model_name = model_name
        if model_name not in SUPPORTED_TRANSLATION_MODELS:
            raise ValueError(f"Model {model_name} is not supported for translation.")
        self.nmt_model = self.load_model()
        self.prompt_template = self.get_prompt_template(model_name)
        self.waitk = waitk
        self.sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        self.source_language = source_language
        self.target_language = target_language

    def get_prompt_template(self, model_name: str) -> PromptTemplate:
        """
        Returns prompt template for the LLM model.
        Args:
            model_name: (str) name of the model to get prompt template for
        Returns:
            PromptTemplate: prompt template for the LLM model
        Raises:
            ValueError: if model is not supported for translation
        """
        if model_name == EURO_LLM_INSTRUCT_SMALL or model_name == EURO_LLM_INSTRUCT_LARGE:
            return EuroLLMTranslatorPromptTemplate

        raise ValueError(f"Model {model_name} is not supported for translation.")

    def load_model(self) -> LLM:
        """
        Load NMT model in vLLM format.
        Returns:
            Loaded LLM instance.
        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            model = LLM(model=self.model_name)
            return model
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def translate(
        self, asr_transcripts: list[str], nmt_prefixes: list[str], src_langs: list[str], tgt_langs: list[str]
    ) -> list[str]:
        """
        Translate ASR transcripts starting from pre-defined prefixes in target language.
        Args:
            asr_transcripts: (list[str]) texts in source language to be translated
            nmt_prefixes: (list[str]) texts in target language to start translation from
            src_langs: (list[str]) source languages
            tgt_langs: (list[str]) target languages
        Returns:
            list[str] translations of ASR transcripts
        """
        input_texts = []
        for src_lang, tgt_lang, src_prefix, tgt_prefix in zip(src_langs, tgt_langs, asr_transcripts, nmt_prefixes):
            text = self.prompt_template.format(src_lang, tgt_lang, src_prefix, tgt_prefix)
            input_texts.append(text)

        outputs = self.nmt_model.generate(input_texts, self.sampling_params)
        translations = []
        for tgt_prefix, output in zip(nmt_prefixes, outputs):
            output_text = output.outputs[0].text
            output_text = self.prompt_template.extract(output_text)
            translations.append(f"{tgt_prefix}{output_text}")
        return translations

    def get_nmt_prefixes(
        self,
        asr_transcripts: list[str],
        translations: list[str],
        prev_translations: list[str],
    ) -> list[str]:
        """
        Pick translation prefixes given the translations generated on the previous and current
        time steps. New prefixes are selected as Longest Common Prefixes of previous and current
        translations.
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

            # If lcp happends mid-word, remove generated ending up to the first full word
            if (len(lcp) > 0) and (lcp[-1] not in f"{string.punctuation} "):
                lcp = " ".join(lcp.split()[:-1])

            # Remove tralining whitespaces
            lcp = lcp.strip()

            # Remove hallucinations if ASR transcript is empty string
            if len(asr) == 0:
                lcp = ""

            # Force translation of up to |asr|-k words if there is no agreement
            # between current and previous translations
            n_asr_words = len(asr.split())
            n_lcp_words = len(lcp.split())
            if (self.waitk > 0) and (n_asr_words - n_lcp_words > self.waitk):
                num_words_to_pick = n_asr_words - self.waitk
                new_prefix = " ".join(trans.split()[:num_words_to_pick])
            else:
                new_prefix = lcp

            new_prefixes.append(new_prefix)

        return new_prefixes
