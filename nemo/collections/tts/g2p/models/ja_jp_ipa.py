# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Union

import pyopenjtalk

from nemo.collections.common.tokenizers.text_to_speech.ipa_lexicon import (
    GRAPHEME_CHARACTER_SETS,
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


class JapaneseKatakanaAccentG2p(BaseG2p):
    """Japanese G2P module that converts text to Kana with pitch accent markers.

    Converts Japanese text to katakana with pitch accent (0=low, 1=high) before each mora.
    Implements Japanese pitch accent rules for entire word chains.

    Japanese pitch accent rules:
    - acc=0 (Heiban 平板): L-H-H-H... (first mora low, rest high)
    - acc=1 (Atamadaka 頭高): H-L-L-L... (first mora high, rest low)
    - acc=N (2 ≤ N < mora count, Nakadaka 中高): L-H-H...-H-L... (low, then high, drops after Nth mora)
    - acc=N (N >= mora count, Odaka 尾高): L-H-H-H (drop at end or after)

    chain_flag handling:
    - chain_flag=0 or -1: Start of new word chain (use this word's acc)
    - chain_flag=1: Continuation of chain (ignore this word's acc)
    - Entire chain treated as single word with first word's acc and total mora count

    Output format: [pitch, kana_char(s), pitch, kana_char(s), ...] where pitch (0/1) precedes each mora
    """

    def __init__(
        self,
        ascii_letter_prefix: str = "",
        ascii_letter_case: str = "lower",
        word_tokenize_func=None,
        apply_to_oov_word=None,
        mapping_file: Optional[str] = None,
    ):
        if pyopenjtalk is None:
            raise ImportError("pyopenjtalk is required. Install with: pip install pyopenjtalk")
        if ascii_letter_prefix is None:
            ascii_letter_prefix = ""

        self.ascii_letter_case = ascii_letter_case
        # Load Japanese katakana grapheme set
        ja_graphemes = GRAPHEME_CHARACTER_SETS.get("ja-JP", [])

        pitch_markers = ['0', '1']
        self.phoneme_list = sorted(list(ja_graphemes) + pitch_markers)

        # ASCII letters handling
        self.ascii_letter_dict = {
            x: ascii_letter_prefix + x for x in get_grapheme_character_set(locale="en-US", case=ascii_letter_case)
        }
        self.ascii_letter_list = sorted(self.ascii_letter_dict)

        self.punctuation = get_ipa_punctuation_list('ja-JP')

        super().__init__(
            word_tokenize_func=word_tokenize_func,
            apply_to_oov_word=apply_to_oov_word,
            mapping_file=mapping_file,
        )

    @staticmethod
    def _split_katakana_to_moras(katakana: str) -> List[str]:
        """Split Mora pattern: [main_katakana][small_katakana]? | [standalone_small] | [choonpu]"""
        mora_pattern = r'[ア-ンヴ][ャュョァィゥェォヮ]?|[ァィゥェォヵヶッャュョヮ]|ー'
        return re.findall(mora_pattern, katakana)

    def _get_pitch_pattern(self, acc: int, total_mora: int) -> List[int]:
        """Calculate pitch pattern for entire word chain.

        Args:
            acc: Accent nucleus position from first word in chain
            total_mora: Total mora count of entire chain

        Returns:
            List of pitch values (0=low, 1=high) for each mora
        """
        if total_mora == 0:
            return []

        if acc == 0:  # Heiban: L-H-H-H...
            return [0] + [1] * (total_mora - 1)

        if acc == 1:  # Atamadaka: H-L-L-L...
            return [1] + [0] * (total_mora - 1)

        if acc >= total_mora:  # Odaka: L-H-H-H
            return [0] + [1] * (total_mora - 1)

        # Nakadaka: L-H...H-L...L (drop after acc-th mora)
        return [0] + [1] * (acc - 1) + [0] * (total_mora - acc)

    def _process_chain(self, chain: List[Dict], result: List[str]) -> None:
        if not chain:
            return

        # Find chain starter
        chain_starter_idx = 0
        for i, word in enumerate(chain):
            if word['chain_flag'] != 1:
                chain_starter_idx = i
                break

        chain_acc = chain[chain_starter_idx]['acc']

        # Split all words into moras
        all_moras = []
        for word in chain:
            moras = self._split_katakana_to_moras(word['pron'])
            all_moras.extend(moras)

        # Calculate pitch pattern using chain starter's accent
        total_mora = len(all_moras)
        pitch_pattern = self._get_pitch_pattern(chain_acc, total_mora)

        # Build output
        for mora, pitch in zip(all_moras, pitch_pattern):
            result.append(str(pitch))
            result.extend(mora)

    def __call__(self, text: str) -> List[str]:
        """Convert Japanese text to kana with pitch accent markers.

        For example, The text "こんにちは" would be converted as a list,
        `['0', 'コ', '1', 'ン', '1', 'ニ', '1', 'チ', '1', 'ワ']`
        """
        text = set_grapheme_case(text, case=self.ascii_letter_case)

        # njd (Nihongo Jisho Data): List of word dictionaries with linguistic features
        njd = pyopenjtalk.run_frontend(text)

        result = []
        current_chain = []
        punctuation = self.punctuation

        for idx, word in enumerate(njd):
            if not isinstance(word, dict):
                continue

            pron = word.get('pron', '')
            pos = word.get('pos', '')
            string = word.get('string', '')
            chain_flag = word.get('chain_flag', 0)
            mora_size = word.get('mora_size', 0)
            acc = word.get('acc', 0)

            string = unicodedata.normalize('NFKC', string)

            # Handle English letters
            if string and all(c in self.ascii_letter_dict for c in string):
                if current_chain:
                    self._process_chain(current_chain, result)
                    current_chain = []

                result.extend(list(string))
                continue

            # Handle punctuation
            if pos in ('記号', '補助記号'):
                if current_chain:
                    self._process_chain(current_chain, result)
                    current_chain = []
                if string.isspace():
                    result.append(' ')
                elif string in punctuation:
                    result.append(string)
                continue

            if not pron or mora_size == 0:
                continue

            # Add word to current chain
            current_chain.append(
                {
                    'pron': pron,
                    'acc': acc,
                    'mora_size': mora_size,
                    'chain_flag': chain_flag,
                }
            )

            # Check if next word continues chain
            next_has_chain = (
                idx + 1 < len(njd) and isinstance(njd[idx + 1], dict) and njd[idx + 1].get('chain_flag', 0) == 1
            )

            # If next word doesn't continue chain, process current chain
            if not next_has_chain:
                self._process_chain(current_chain, result)
                current_chain = []

        # Process any remaining chain
        if current_chain:
            self._process_chain(current_chain, result)
        return result
