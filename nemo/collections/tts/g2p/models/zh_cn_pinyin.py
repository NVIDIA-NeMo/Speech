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
from collections import defaultdict
from typing import Optional

from nemo.collections.tts.g2p.models.base import BaseG2p
from nemo.utils import logging


class ChineseG2p(BaseG2p):
    def __init__(
        self,
        phoneme_dict=None,
        word_tokenize_func=None,
        apply_to_oov_word=None,
        mapping_file: Optional[str] = None,
        word_segmenter: Optional[str] = None,
    ):
        """Chinese G2P module. This module first converts Chinese characters into pinyin sequences using pypinyin, then pinyin sequences would
           be further converted into phoneme sequences using pinyin_dict_nv_22.10.txt dict file. For Chinese and English bilingual sentences, the English words
           would be converted into letters.
        Args:
            phoneme_dict (str, Path, Dict): Path to pinyin_dict_nv_22.10.txt dict file.
            word_tokenize_func: Function for tokenizing text to words.
                It has to return List[Tuple[Union[str, List[str]], bool]] where every tuple denotes word representation and flag whether to leave unchanged or not.
                It is expected that unchangeable word representation will be represented as List[str], other cases are represented as str.
                It is useful to mark word as unchangeable which is already in phoneme representation.
            apply_to_oov_word: Function that will be applied to out of phoneme_dict word.
            word_segmenter: method that will be applied to segment utterances into words for better polyphone disambiguation.
        """
        assert phoneme_dict is not None, "Please set the phoneme_dict path."
        assert word_segmenter in [
            None,
            'jieba',
        ], f"{word_segmenter} is not supported now. Please choose correct word_segmenter."

        phoneme_dict = (
            self._parse_as_pinyin_dict(phoneme_dict)
            if isinstance(phoneme_dict, str) or isinstance(phoneme_dict, pathlib.Path)
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
        self.tones = {'1': '#1', '2': '#2', '3': '#3', '4': '#4', '5': '#5'}

        if word_segmenter == "jieba":
            try:
                import jieba
            except ImportError as e:
                logging.error(e)

            # Cut sentences into words to improve polyphone disambiguation
            self.word_segmenter = jieba.cut
        else:
            self.word_segmenter = lambda x: [x]

        try:
            from pypinyin import Style, lazy_pinyin
            from pypinyin_dict.pinyin_data import cc_cedict
        except ImportError as e:
            logging.error(e)

        # replace pypinyin default dict with cc_cedict.txt for polyphone disambiguation
        cc_cedict.load()

        self._lazy_pinyin = lazy_pinyin
        self._Style = Style

    @staticmethod
    def _parse_as_pinyin_dict(phoneme_dict_path):
        """Loads pinyin dict file, and generates a set of all valid symbols."""
        g2p_dict = defaultdict(list)
        with open(phoneme_dict_path, 'r') as file:
            for line in file:
                parts = line.split('\t')
                # let the key be lowercased, since pypinyin would give lower representation
                pinyin = parts[0].lower()
                pronunciation = parts[1].split()
                pronunciation_with_sharp = ['#' + pron for pron in pronunciation]
                g2p_dict[pinyin] = pronunciation_with_sharp
        return g2p_dict

    def __call__(self, text):
        """
        errors func handle below is to process the bilingual situation,
        where English words would be split into letters.
        e.g. 我今天去了Apple Store, 买了一个iPhone。
        would return a list
        ['wo3', 'jin1', 'tian1', 'qu4', 'le5', 'A', 'p', 'p', 'l', 'e',
        ' ', 'S', 't', 'o', 'r', 'e', ',', ' ', 'mai3', 'le5', 'yi2',
        'ge4', 'i', 'P', 'h', 'o', 'n', 'e', '。']
        """
        pinyin_seq = []
        words_list = self.word_segmenter(text)

        for word in words_list:
            pinyin_seq += self._lazy_pinyin(
                word,
                style=self._Style.TONE3,
                neutral_tone_with_five=True,
                errors=lambda en_words: [letter for letter in en_words],
            )
        phoneme_seq = []
        for pinyin in pinyin_seq:
            if pinyin[-1] in self.tones:
                assert pinyin[:-1] in self.phoneme_dict, pinyin[:-1]
                phoneme_seq += self.phoneme_dict[pinyin[:-1]]
                phoneme_seq.append(self.tones[pinyin[-1]])
            # All pinyin would end up with a number in 1-5, which represents tones of the pinyin.
            # For symbols which are not pinyin, e.g. English letters, Chinese puncts, we directly
            # use them as inputs.
            else:
                phoneme_seq.append(pinyin)
        return phoneme_seq
