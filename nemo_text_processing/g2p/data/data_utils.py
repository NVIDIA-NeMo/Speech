# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


import csv
import re
import string
import unicodedata
from builtins import str as unicode
from typing import List, Tuple

__all__ = [
    "read_wordids",
    "chinese_text_preprocessing",
    "english_text_preprocessing",
    "german_text_preprocessing",
    "any_locale_text_preprocessing",
    "spanish_text_preprocessing",
    "any_locale_word_tokenize",
    "english_word_tokenize",
]

# +
# Derived from LJSpeech
_synoglyphs = {
    "'": ['’'],
    '"': ['”', '“'],
}
SYNOGLYPH2ASCII = {g: asc for asc, glyphs in _synoglyphs.items() for g in glyphs}

# Example of parsing by groups via _WORDS_RE_EN.
# Regular expression pattern groups:
#   1st group -- valid english words,
#   2nd group -- any substring starts from | to | (mustn't be nested), useful when you want to leave sequence unchanged,
#   3rd group -- punctuation marks or whitespaces.
# Text (first line) and mask of groups for every char (second line).
# config file must contain |EY1 EY1|, B, C, D, E, F, and G.

# define char set based on https://en.wikipedia.org/wiki/List_of_Unicode_characters
LATIN_ALPHABET_BASIC = "A-Za-z"
ACCENTED_CHARS = "À-ÖØ-öø-ÿ"
LATIN_CHARS_ALL = f"{LATIN_ALPHABET_BASIC}{ACCENTED_CHARS}"
_WORDS_RE_EN = re.compile(
    fr"([{LATIN_ALPHABET_BASIC}]+(?:[{LATIN_ALPHABET_BASIC}\-']*[{LATIN_ALPHABET_BASIC}]+)*)|(\|[^|]*\|)|([^{LATIN_ALPHABET_BASIC}|]+)"
)
_WORDS_RE_ANY_LOCALE = re.compile(
    fr"([{LATIN_CHARS_ALL}]+(?:[{LATIN_CHARS_ALL}\-']*[{LATIN_CHARS_ALL}]+)*)|(\|[^|]*\|)|([^{LATIN_CHARS_ALL}|]+)"
)

# -


def read_wordids(wordid_map: str):
    """
    Reads wordid file from WikiHomograph dataset,
    https://github.com/google-research-datasets/WikipediaHomographData/blob/master/data/wordids.tsv

    Args:
        wordid_map: path to wordids.tsv
    Returns:
        homograph_dict: a dictionary of graphemes with corresponding word_id - ipa_form pairs
        wordid_to_idx: word id to label id mapping
    """
    homograph_dict = {}
    wordid_to_idx = {}

    with open(wordid_map, "r", encoding="utf-8") as f:
        tsv_file = csv.reader(f, delimiter="\t")

        for i, line in enumerate(tsv_file):
            if i == 0:
                continue

            grapheme = line[0]
            word_id = line[1]
            ipa_form = line[3]
            wordid_to_idx[word_id] = len(wordid_to_idx)
            if grapheme not in homograph_dict:
                homograph_dict[grapheme] = {}
            homograph_dict[grapheme][word_id] = ipa_form
    return homograph_dict, wordid_to_idx


def get_wordid_to_nemo(wordid_to_nemo_cmu_file: str = "../../../scripts/tts_dataset_files/wordid_to_nemo_cmu.tsv"):
    """
    WikiHomograph and NeMo use slightly different phoneme sets, this function reads WikiHomograph word_ids to NeMo
    IPA heteronyms mapping
    """
    wordid_to_nemo_cmu = {}
    with open(wordid_to_nemo_cmu_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip().split("\t")
            wordid_to_nemo_cmu[line[0]] = line[1]
    return wordid_to_nemo_cmu


def remove_punctuation(text: str, exclude: List[str] = None):
    all_punct_marks = string.punctuation

    if exclude is not None:
        for p in exclude:
            all_punct_marks = all_punct_marks.replace(p, "")
    text = re.sub("[" + all_punct_marks + "]", " ", text)

    text = re.sub(r" +", " ", text)
    return text.strip()


def english_text_preprocessing(text, lower=True):
    text = unicode(text)
    text = ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn')
    text = ''.join(char if char not in SYNOGLYPH2ASCII else SYNOGLYPH2ASCII[char] for char in text)

    if lower:
        text = text.lower()

    return text


def _word_tokenize(words: List[Tuple[str, str, str]]) -> List[Tuple[List[str], bool]]:
    """
    Process a list of words and attach indicators showing if each word is unchangeable or not. Each word representation
    can be one of valid word, any substring starting from | to | (unchangeable word), or punctuation marks including
    whitespaces. This function will split unchanged strings by whitespaces and return them as `List[str]`. For example,

    .. code-block:: python
        [
            ('Hello', '', ''),  # valid word
            ('', '', ' '),  # punctuation mark
            ('World', '', ''),  # valid word
            ('', '', ' '),  # punctuation mark
            ('', '|NVIDIA unchanged|', ''),  # unchangeable word
            ('', '', '!')  # punctuation mark
        ]

    will be converted into,

    .. code-block:: python
        [
            (["hello"], False),
            ([" "], False),
            (["world"], False),
            ([" "], False),
            (["nvidia", "unchanged"], True),
            (["!"], False)
        ]

    Args:
        words (List[str]): a list of tuples like `(maybe_word, maybe_without_changes, maybe_punct)` where each element
            corresponds to a non-overlapping match of either `_WORDS_RE_EN` or `_WORDS_RE_ANY_LOCALE`.

    Returns: List[Tuple[List[str], bool]], a list of tuples like `(a list of words, is_unchanged)`.

    """
    result = []
    for word in words:
        maybe_word, maybe_without_changes, maybe_punct = word

        without_changes = False
        if maybe_word != '':
            token = [maybe_word.lower()]
        elif maybe_punct != '':
            token = [maybe_punct]
        elif maybe_without_changes != '':
            without_changes = True
            token = maybe_without_changes[1:-1].split(" ")
        else:
            raise ValueError(
                f"This is not expected. Found empty string: <{word}>. "
                f"Please validate your regular expression pattern '_WORDS_RE_EN' or '_WORDS_RE_ANY_LOCALE'."
            )

        result.append((token, without_changes))

    return result


def english_word_tokenize(text):
    words = _WORDS_RE_EN.findall(text)
    return _word_tokenize(words)


def any_locale_word_tokenize(text):
    words = _WORDS_RE_ANY_LOCALE.findall(text)
    return _word_tokenize(words)


def any_locale_text_preprocessing(text):
    return text.lower()


def german_text_preprocessing(text):
    return text.lower()


def spanish_text_preprocessing(text):
    return text.lower()


def chinese_text_preprocessing(text):
    return text.lower()
