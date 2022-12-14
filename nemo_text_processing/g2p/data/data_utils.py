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
from typing import List

__all__ = ['read_wordids']

# +
# Derived from LJSpeech
_synoglyphs = {
    "'": ['’'],
    '"': ['”', '“'],
}
SYNOGLYPH2ASCII = {g: asc for asc, glyphs in _synoglyphs.items() for g in glyphs}

# Example of parsing by groups via _WORDS_RE.
# Groups:
# 1st group -- valid english words,
# 2nd group -- any substring starts from | to | (mustn't be nested), useful when you want to leave sequence unchanged,
# 3rd group -- punctuation marks.
# Text (first line) and mask of groups for every char (second line).
# config file must contain |EY1 EY1|, B, C, D, E, F, and G.

# define char set based on https://en.wikipedia.org/wiki/List_of_Unicode_characters
LATIN_ALPHABET_BASIC = "A-Za-z"
ACCENTED_CHARS = "À-ÖØ-öø-ÿ"
LATIN_CHARS_ALL = f"{LATIN_ALPHABET_BASIC}{ACCENTED_CHARS}"
_WORDS_RE = re.compile(
    fr"([{LATIN_ALPHABET_BASIC}]+(?:[{LATIN_ALPHABET_BASIC}\-']*[{LATIN_ALPHABET_BASIC}]+)*)|(\|[^|]*\|)|([^{LATIN_ALPHABET_BASIC}|]+)"
)
_WORDS_RE_IPA = re.compile(
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


def _word_tokenize(words):
    """
    Convert text (str) to List[Tuple[Union[str, List[str]], bool]] where every tuple denotes word representation and
    flag whether to leave unchanged or not.
    Word can be one of: valid english word, any substring starts from | to | (unchangeable word) or punctuation marks.
    This function expects that unchangeable word is carefully divided by spaces (e.g. HH AH L OW).
    Unchangeable word will be splitted by space and represented as List[str], other cases are represented as str.
    """
    result = []
    for word in words:
        maybe_word, maybe_without_changes, maybe_punct = word

        if maybe_word != '':
            without_changes = False
            result.append((maybe_word.lower(), without_changes))
        elif maybe_punct != '':
            without_changes = False
            result.append((maybe_punct, without_changes))
        elif maybe_without_changes != '':
            without_changes = True
            result.append((maybe_without_changes[1:-1].split(" "), without_changes))
    return result


def english_word_tokenize(text):
    words = _WORDS_RE.findall(text)
    return _word_tokenize(words)


def ipa_word_tokenize(text):
    words = _WORDS_RE_IPA.findall(text)
    return _word_tokenize(words)


def ipa_text_preprocessing(text):
    return text.lower()


def german_text_preprocessing(text):
    return text.lower()


def spanish_text_preprocessing(text):
    return text.lower()


def chinese_text_preprocessing(text):
    return text.lower()
