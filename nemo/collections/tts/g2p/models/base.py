# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Optional
from nemo.utils import logging


class BaseG2p(ABC):
    def __init__(
        self,
        phoneme_dict=None,
        word_tokenize_func=lambda x: x,
        apply_to_oov_word=None,
        mapping_file: Optional[str] = None,
    ):
        """Abstract class for creating an arbitrary module to convert grapheme words
        to phoneme sequences, leave unchanged, or use apply_to_oov_word.
        Args:
            phoneme_dict: Arbitrary representation of dictionary (phoneme -> grapheme) for known words.
            word_tokenize_func: Function for tokenizing text to words.
            apply_to_oov_word: Function that will be applied to out of phoneme_dict word.
        """
        self.phoneme_dict = phoneme_dict
        self.word_tokenize_func = word_tokenize_func
        self.apply_to_oov_word = apply_to_oov_word
        self.mapping_file = mapping_file
        self.heteronym_model = None  # heteronym classification model

    @abstractmethod
    def __call__(self, text: str) -> str:
        pass

    # TODO @xueyang: replace `wordid_to_phonemes_file` default variable with a global variable defined in util file.
    def setup_heteronym_model(
        self,
        heteronym_model,
        wordid_to_phonemes_file: str = "../../../scripts/tts_dataset_files/wordid_to_ipa-0.7b_nv22.10.tsv",
    ):
        """
        Add heteronym classification model to TTS preprocessing pipeline to disambiguate heteronyms.
            Heteronym model has a list of supported heteronyms but only heteronyms specified in
            wordid_to_phonemes_file will be converted to phoneme form during heteronym model inference;
            the rest will be left in grapheme form.

        Args:
            heteronym_model: Initialized HeteronymClassificationModel
            wordid_to_phonemes_file: Path to a file with mapping from wordid predicted by heteronym model to phonemes
        """

        try:
            from nemo.collections.tts.g2p.models.heteronym_classification import HeteronymClassificationModel

            self.heteronym_model = heteronym_model
            self.heteronym_model.set_wordid_to_phonemes(wordid_to_phonemes_file)
        except ImportError as e:
            logging.warning("Heteronym model setup will be skipped")
            logging.error(e)
