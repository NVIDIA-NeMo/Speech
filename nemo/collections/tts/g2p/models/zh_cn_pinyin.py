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
from typing import Dict, List, Optional, Union

from nemo.collections.common.tokenizers.text_to_speech.ipa_lexicon import get_grapheme_character_set
from nemo.collections.tts.g2p.models.base import BaseG2p
from nemo.collections.tts.g2p.utils import set_grapheme_case
from nemo.utils import logging


class ChineseG2p(BaseG2p):
    def __init__(
        self,
        phoneme_dict: Union[str, pathlib.Path, Dict[str, List[str]]],
        phoneme_prefix: str = "#",
        phoneme_case: str = "upper",
        tone_prefix: str = "#",
        ascii_letter_prefix: str = "",
        ascii_letter_case: str = "lower",
        word_tokenize_func=None,
        apply_to_oov_word=None,
        mapping_file: Optional[str] = None,
        word_segmenter: Optional[str] = None,
    ):
        """
        Chinese G2P module. This module first converts Chinese characters into pinyin sequences using pypinyin, then
            pinyin sequences would be further converted into phoneme sequences by looking them up in the `phoneme_dict`.
            This G2P module also works with Chinese/English bilingual sentences where English words would be converted
            into letters. It is advised to attach prefix symbols for Chinese phonemes and tones to discriminate them
            from English letters to avoid any potential symbol set overlaps.
        Args:
            phoneme_dict (str, Path, Dict): Path to pinyin_dict_nv_22.10.txt dict file or a dict object.
            phoneme_prefix (str): Prepend a special symbol to any phonemes in order to distinguish phonemes from
                graphemes because there may be overlaps between the two sets. Phoneme dictionary typically applies
                uppercase initials and finals. It is suggested to choose a prefix that
                is not used or preserved somewhere else. Default to "#".
            phoneme_case (str): Specify the case chosen from `"lower"`, `"upper"`, or `"mixed"`, and process the
                cases of Chinese phonemes. Default to `"upper"`.
            tone_prefix (str): Prepend a special symbol to any tone digits. Default to "#".
            ascii_letter_prefix (str): Prepend a special symbol to any ASCII letters. Default to "".
            ascii_letter_case (str): Specify the case chosen from `"lower"`, `"upper"`, or `"mixed"`, and process the
                cases of non-Chinese words. Default to `"lower"`.
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
            'jieba',
            'pkuseg',
        ], f"{word_segmenter} is not supported now. Please choose correct word_segmenter."

        if phoneme_prefix is None:
            phoneme_prefix = ""
        if tone_prefix is None:
            tone_prefix = ""
        if ascii_letter_prefix is None:
            ascii_letter_prefix = ""

        # phonemes
        phoneme_dict = (
            self._parse_as_pinyin_dict(phoneme_dict, phoneme_prefix, phoneme_case)
            if isinstance(phoneme_dict, str) or isinstance(phoneme_dict, pathlib.Path)
            else phoneme_dict
        )
        self.phoneme_list = sorted({pron for prons in phoneme_dict.values() for pron in prons})

        # tones
        self.tone_dict = {str(x): tone_prefix + str(x) for x in range(1, 6)}
        self.tone_list = sorted(self.tone_dict.values())

        # ascii letters
        self.ascii_letter_dict = {
            x: ascii_letter_prefix + x for x in get_grapheme_character_set(locale="en-US", case=ascii_letter_case)
        }
        self.ascii_letter_list = sorted(self.ascii_letter_dict.values())
        self.ascii_letter_case = ascii_letter_case

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

        if word_segmenter == "jieba":
            try:
                import jieba
            except ImportError as e:
                logging.error(e)

            # Cut sentences into words to improve polyphone disambiguation
            self.word_segmenter = jieba.cut
        elif word_segmenter == "pkuseg":
            try:
                import pkuseg
            except ImportError as e:
                logging.error(
                    "`pkuseg` is not the default word segmenter for Chinese in NeMo. Please install it manually by running `pip install pkuseg`, or choose `jieba` segmenter instead."
                )

            self.word_segmenter = pkuseg.pkuseg().cut
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
    def _parse_as_pinyin_dict(
        phoneme_dict_path: Union[str, pathlib.Path], phoneme_prefix: str, phoneme_case: str
    ) -> Dict[str, List[str]]:
        """Loads pinyin dict file, and generates a set of all valid symbols."""
        g2p_dict = defaultdict(list)
        with open(phoneme_dict_path, 'r') as file:
            for line in file:
                # skip empty lines and comment lines starting with `;;;`.
                if line.startswith(";;;") or len(line.strip()) == 0:
                    continue

                parts = line.split('\t')
                # Convert the cases of Chinese syllables loaded from the dictionary to lowercase to match the lowercase
                # Chinese syllable outputs generated by the function `pypinyin.lazy_pinyin`. Note that the function
                # `pypinyin.lazy_pinyin` preserves the cases of ASCII letters.
                syllable = parts[0].lower()
                pronunciation = set_grapheme_case(parts[1], case=phoneme_case).split()

                # add a prefix to distinguish phoneme symbols from non-phoneme symbols.
                pronunciation_with_prefix = [phoneme_prefix + pron for pron in pronunciation]
                g2p_dict[syllable] = pronunciation_with_prefix

        return g2p_dict

    def __call__(self, text: str) -> List[str]:
        """
        This forward pass function translates Chinese characters into pinyin sequences and then converts the pinyin
        into phonemes. It is primarily designed to process texts containing with Chinese characters, but we have
        extended its support to handle texts that include both Chinese and English. This extension was mainly
        necessitated by the limited availability of bilingual datasets. The `errors` argument used in the
        `pypinyin.lazy_pinyin` function below is used to process non-Chinese words, where each English word is split
        into letters.

        For example, The text "我今天去了Apple Store, 买了一个iPhone。" would be converted as a list,
        `['wo3', 'jin1', 'tian1', 'qu4', 'le5', 'A', 'p', 'p', 'l', 'e', ' ', 'S', 't', 'o', 'r', 'e', ',', ' ', 'mai3',
         'le5', 'yi2', 'ge4', 'i', 'P', 'h', 'o', 'n', 'e', '。']`
        """
        err = False
        text = set_grapheme_case(text, case=self.ascii_letter_case)

        pinyin_seq = []
        words_list = self.word_segmenter(text)

        # TODO @xueyang: add a g2p process for non-pinyin words by customizing a function for `errors` argument. For
        #  example, add a dict look up for English words.
        for word in words_list:
            pinyin_seq += self._lazy_pinyin(
                word,
                style=self._Style.TONE3,
                neutral_tone_with_five=True,
                errors=lambda en_words: [letter for letter in en_words],
            )
        phoneme_seq = []
        for pinyin in pinyin_seq:
            # only pinyin has tones while non-pinyin doesn't.
            tone_hyp = pinyin[-1]
            if tone_hyp in self.tone_dict:
                syllable = pinyin[:-1]
                # TODO: skipping the syllable that does not exist in the dictionary will lead to deletion errors in the
                #   synthesized speech. Even though this case is uncommon, it should be fixed in future.
                if syllable not in self.phoneme_dict:
                    err = True
                    logging.error(
                        f"Syllable <{syllable}> does not exist in the dictionary. You should expect symbol "
                        f"deletion risks!!"
                    )
                    continue
                phoneme_seq += self.phoneme_dict[syllable]
                phoneme_seq.append(self.tone_dict[tone_hyp])
            # All pinyin would end up with a number in 1-5, which represents tones of the pinyin.
            # For symbols which are not pinyin, such as English letters and Chinese punctuations, we directly
            # use them as inputs.
            elif tone_hyp in self.ascii_letter_dict:
                phoneme_seq.append(self.ascii_letter_dict[tone_hyp])
            else:
                phoneme_seq.append(pinyin)
        if err:
            logging.error(f"|{text}| contained unknown syllables")
        return phoneme_seq
