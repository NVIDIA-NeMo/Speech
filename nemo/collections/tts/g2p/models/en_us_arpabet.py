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

import pathlib
import random
import re
import time
from typing import Optional

import nltk
import torch

from nemo.collections.common.tokenizers.text_to_speech.tokenizer_utils import english_word_tokenize
from nemo.collections.tts.g2p.models.base import BaseG2p
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


class EnglishG2p(BaseG2p):
    def __init__(
        self,
        phoneme_dict=None,
        word_tokenize_func=english_word_tokenize,
        apply_to_oov_word=None,
        ignore_ambiguous_words=True,
        heteronyms=None,
        encoding='latin-1',
        phoneme_probability: Optional[float] = None,
        mapping_file: Optional[str] = None,
    ):
        """English G2P module. This module converts words from grapheme to phoneme representation using phoneme_dict in CMU dict format.
        Optionally, it can ignore words which are heteronyms, ambiguous or marked as unchangeable by word_tokenize_func (see code for details).
        Ignored words are left unchanged or passed through apply_to_oov_word for handling.
        Args:
            phoneme_dict (str, Path, Dict): Path to file in CMUdict format or dictionary of CMUdict-like entries.
            word_tokenize_func: Function for tokenizing text to words.
                It has to return List[Tuple[Union[str, List[str]], bool]] where every tuple denotes word representation and flag whether to leave unchanged or not.
                It is expected that unchangeable word representation will be represented as List[str], other cases are represented as str.
                It is useful to mark word as unchangeable which is already in phoneme representation.
            apply_to_oov_word: Function that will be applied to out of phoneme_dict word.
            ignore_ambiguous_words: Whether to not handle word via phoneme_dict with ambiguous phoneme sequences. Defaults to True.
            heteronyms (str, Path, List): Path to file with heteronyms (every line is new word) or list of words.
            encoding: Encoding type.
            phoneme_probability (Optional[float]): The probability (0.<var<1.) that each word is phonemized. Defaults to None which is the same as 1.
                Note that this code path is only run if the word can be phonemized. For example: If the word does not have an entry in the g2p dict, it will be returned
                as characters. If the word has multiple entries and ignore_ambiguous_words is True, it will be returned as characters.
        """
        phoneme_dict = (
            self._parse_as_cmu_dict(phoneme_dict, encoding)
            if isinstance(phoneme_dict, str) or isinstance(phoneme_dict, pathlib.Path) or phoneme_dict is None
            else phoneme_dict
        )

        if apply_to_oov_word is None:
            logging.warning(
                "apply_to_oov_word=None, This means that some of words will remain unchanged "
                "if they are not handled by any of the rules in self.parse_one_word(). "
                "This may be intended if phonemes and chars are both valid inputs, otherwise, "
                "you may see unexpected deletions in your input."
            )

        super().__init__(
            phoneme_dict=phoneme_dict,
            word_tokenize_func=word_tokenize_func,
            apply_to_oov_word=apply_to_oov_word,
            mapping_file=mapping_file,
        )

        self.ignore_ambiguous_words = ignore_ambiguous_words
        self.heteronyms = (
            set(self._parse_file_by_lines(heteronyms, encoding))
            if isinstance(heteronyms, str) or isinstance(heteronyms, pathlib.Path)
            else heteronyms
        )
        self.phoneme_probability = phoneme_probability
        self._rng = random.Random()

    @staticmethod
    def _parse_as_cmu_dict(phoneme_dict_path=None, encoding='latin-1'):
        if phoneme_dict_path is None:
            # this part of code downloads file, but it is not rank zero guarded
            # Try to check if torch distributed is available, if not get global rank zero to download corpora and make
            # all other ranks sleep for a minute
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = torch.distributed.group.WORLD
                if is_global_rank_zero():
                    try:
                        nltk.data.find('corpora/cmudict.zip')
                    except LookupError:
                        nltk.download('cmudict', quiet=True)
                torch.distributed.barrier(group=group)
            elif is_global_rank_zero():
                logging.error(
                    f"Torch distributed needs to be initialized before you initialized EnglishG2p. This class is prone to "
                    "data access race conditions. Now downloading corpora from global rank 0. If other ranks pass this "
                    "before rank 0, errors might result."
                )
                try:
                    nltk.data.find('corpora/cmudict.zip')
                except LookupError:
                    nltk.download('cmudict', quiet=True)
            else:
                logging.error(
                    f"Torch distributed needs to be initialized before you initialized EnglishG2p. This class is prone to "
                    "data access race conditions. This process is not rank 0, and now going to sleep for 1 min. If this "
                    "rank wakes from sleep prior to rank 0 finishing downloading, errors might result."
                )
                time.sleep(60)

            logging.warning(
                f"English g2p_dict will be used from nltk.corpus.cmudict.dict(), because phoneme_dict_path=None. "
                "Note that nltk.corpus.cmudict.dict() has old version (0.6) of CMUDict. "
                "You can use the latest official version of CMUDict (0.7b) with additional changes from NVIDIA directly from NeMo "
                "using the path scripts/tts_dataset_files/cmudict-0.7b_nv22.10."
            )

            return nltk.corpus.cmudict.dict()

        _alt_re = re.compile(r'\([0-9]+\)')
        g2p_dict = {}
        with open(phoneme_dict_path, encoding=encoding) as file:
            for line in file:
                if len(line) and ('A' <= line[0] <= 'Z' or line[0] == "'"):
                    parts = line.split('  ')
                    word = re.sub(_alt_re, '', parts[0])
                    word = word.lower()

                    pronunciation = parts[1].strip().split(" ")
                    if word in g2p_dict:
                        g2p_dict[word].append(pronunciation)
                    else:
                        g2p_dict[word] = [pronunciation]
        return g2p_dict

    @staticmethod
    def _parse_file_by_lines(p, encoding):
        with open(p, encoding=encoding) as f:
            return [l.rstrip() for l in f.readlines()]

    def is_unique_in_phoneme_dict(self, word):
        return len(self.phoneme_dict[word]) == 1

    def parse_one_word(self, word: str):
        """
        Returns parsed `word` and `status` as bool.
        `status` will be `False` if word wasn't handled, `True` otherwise.
        """

        if self.phoneme_probability is not None and self._rng.random() > self.phoneme_probability:
            return word, True

        # punctuation or whitespace.
        if re.search(r"[a-zA-ZÀ-ÿ\d]", word) is None:
            return list(word), True

        # heteronyms
        if self.heteronyms is not None and word in self.heteronyms:
            return word, True

        # `'s` suffix
        if (
            len(word) > 2
            and word.endswith("'s")
            and (word not in self.phoneme_dict)
            and (word[:-2] in self.phoneme_dict)
            and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word[:-2]))
        ):
            return self.phoneme_dict[word[:-2]][0] + ["Z"], True

        # `s` suffix
        if (
            len(word) > 1
            and word.endswith("s")
            and (word not in self.phoneme_dict)
            and (word[:-1] in self.phoneme_dict)
            and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word[:-1]))
        ):
            return self.phoneme_dict[word[:-1]][0] + ["Z"], True

        # phoneme dict
        if word in self.phoneme_dict and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word)):
            return self.phoneme_dict[word][0], True

        if self.apply_to_oov_word is not None:
            return self.apply_to_oov_word(word), True
        else:
            return word, False

    def __call__(self, text):
        words = self.word_tokenize_func(text)

        prons = []
        for word, without_changes in words:
            if without_changes:
                prons.extend(word)
                continue

            word_str = word[0]
            word_by_hyphen = word_str.split("-")
            pron, is_handled = self.parse_one_word(word_str)

            if not is_handled and len(word_by_hyphen) > 1:
                pron = []
                for sub_word in word_by_hyphen:
                    p, _ = self.parse_one_word(sub_word)
                    pron.extend(p)
                    pron.extend(["-"])
                pron.pop()

            prons.extend(pron)

        return prons
