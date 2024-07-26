# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from collections import defaultdict
from typing import Dict, List, Optional, Union

from nemo.collections.common.tokenizers.text_to_speech.ipa_lexicon import (
    get_grapheme_character_set,
    get_ipa_punctuation_list,
)
from nemo.collections.tts.g2p.models.base import BaseG2p
from nemo.collections.tts.g2p.utils import set_grapheme_case
from nemo.utils import logging


class JapaneseG2p(BaseG2p):
    def __init__(
        self,
        phoneme_dict: Union[str, pathlib.Path, Dict[str, List[str]]],
        phoneme_prefix: str = "",
        ascii_letter_prefix: str = "#",
        ascii_letter_case: str = "upper",
        word_tokenize_func=None,
        apply_to_oov_word=None,
        mapping_file: Optional[str] = None,
        word_segmenter: Optional[str] = None,
    ):
        """
        Japanese G2P module. This module first segments Japanese characters into words using Janome, then
            these separated words are converted into phoneme sequences by looking them up in the 'phoneme_dict'.
        Args:
            phoneme_dict (str, Path, Dict): Path to ja_JP_wordtoipa.txt dict file or a dict object.
            phoneme_prefix (str): Prepend a special symbol to any phonemes in order to distinguish phonemes from
                graphemes because there may be overlaps between the two sets. It is suggested to choose a prefix that
                is not used or preserved somewhere else. Default to "#".
            ascii_letter_prefix (str): Prepend a special symbol to any ASCII letters. Default to "".
            ascii_letter_case (str): Specify the case chosen from `"lower"`, `"upper"`, or `"mixed"`, and process the
                cases of non-Chinese words. Default to `"upper"`.
            word_tokenize_func: Function for tokenizing text to words.
                It has to return List[Tuple[Union[str, List[str]], bool]] where every tuple denotes word representation
                    and flag whether to leave unchanged or not.
                It is expected that unchangeable word representation will be represented as List[str], other cases are
                    represented as str.
                It is useful to mark word as unchangeable which is already in phoneme representation.
            apply_to_oov_word: Function that will be applied to out of phoneme_dict word.
            word_segmenter: method that will be applied to segment utterances into words for better polyphone disambiguation.
        """
        assert phoneme_dict is not None, "Please set the phoneme_dict path."
        assert word_segmenter in [
            None,
            "janome",
        ], f"{word_segmenter} is not supported now. Please choose correct word_segmenter."

        if phoneme_prefix is None:
            phoneme_prefix = ""
        if ascii_letter_prefix is None:
            ascii_letter_prefix = ""

        # phonemes
        phoneme_dict = (
            self._parse_ja_phoneme_dict(phoneme_dict, phoneme_prefix)
            if isinstance(phoneme_dict, str) or isinstance(phoneme_dict, pathlib.Path)
            else phoneme_dict
        )
        self.phoneme_list = sorted({pron for prons in phoneme_dict.values() for pron in prons})

        # ascii letters
        self.ascii_letter_dict = {
            x: ascii_letter_prefix + x for x in get_grapheme_character_set(locale="en-US", case=ascii_letter_case)
        }
        self.ascii_letter_list = sorted(self.ascii_letter_dict)
        self.ascii_letter_case = ascii_letter_case
        self.punctuation = get_ipa_punctuation_list('ja-JP')

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

        if word_segmenter == "janome":
            try:
                from janome.tokenizer import Tokenizer
            except ImportError as e:
                logging.error(e)

            # Cut sentences into words to improve polyphone disambiguation
            self.word_segmenter = Tokenizer().tokenize
        else:
            self.word_segmenter = lambda x: [x]

    @staticmethod
    def _parse_ja_phoneme_dict(
        phoneme_dict_path: Union[str, pathlib.Path], phoneme_prefix: str
    ) -> Dict[str, List[str]]:
        """Loads prondict dict file, and generates a set of all valid symbols."""
        g2p_dict = defaultdict(list)
        with open(phoneme_dict_path, 'r') as file:
            for line in file:
                # skip empty lines and comment lines starting with `;;;`.
                if line.startswith(";;;") or len(line.strip()) == 0:
                    continue

                word, pronunciation = line.rstrip().split(maxsplit=1)

                # add a prefix to distinguish phoneme symbols from non-phoneme symbols.
                pronunciation_with_prefix = [phoneme_prefix + pron for pron in pronunciation]
                g2p_dict[word] = pronunciation_with_prefix

        return g2p_dict

    def __call__(self, text: str) -> List[str]:
        """
        This forward pass function translates Japanese characters into IPA phoneme sequences.

        For example, The text "こんにちは" would be converted as a list,
        `['k', 'o', 'n', 'n', 'i', 't', 'ʃ', 'i', 'h', 'a']`
        """
        text = set_grapheme_case(text, case=self.ascii_letter_case)

        words_list = self.word_segmenter(text)
        phoneme_seq = []
        for token in words_list:
            word = str(token).split("\t")[0]
            if word in self.phoneme_dict.keys():
                phoneme_seq += self.phoneme_dict[word]
            elif word in self.punctuation:
                phoneme_seq += word
            else:
                logging.warning(f"{word} not found in the pronunciation dictionary. Returning graphemes instead.")
                phoneme_seq += [c for c in word]
        return phoneme_seq
