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
import os
import re
import string
import unicodedata
from builtins import str as unicode
from typing import Dict, List, Tuple, Union

__all__ = [
    "read_wordids",
    "chinese_text_preprocessing",
    "english_text_preprocessing",
    "any_locale_text_preprocessing",
    "spanish_text_preprocessing",
    "any_locale_word_tokenize",
    "english_word_tokenize",
    "LATIN_CHARS_ALL",
    "set_grapheme_case",
    "normalize_unicode_text",
    "GRAPHEME_CASE_UPPER",
    "GRAPHEME_CASE_LOWER",
    "GRAPHEME_CASE_MIXED",
    "get_heteronym_spans",
]

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

# define grapheme cases.
GRAPHEME_CASE_UPPER = "upper"
GRAPHEME_CASE_LOWER = "lower"
GRAPHEME_CASE_MIXED = "mixed"


def read_wordids(wordid_map: str):
    """
    Reads wordid file from WikiHomograph dataset,
    https://github.com/google-research-datasets/WikipediaHomographData/blob/master/data/wordids.tsv

    Args:
        wordid_map: path to wordids.tsv
    Returns:
        data_dict: a dictionary of graphemes with corresponding word_id - ipa_form pairs
        wordid_to_idx: word id to label id mapping
    """
    if not os.path.exists(wordid_map):
        raise ValueError(f"{wordid_map} not found")

    data_dict = {}
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
            if grapheme not in data_dict:
                data_dict[grapheme] = {}
            data_dict[grapheme][word_id] = ipa_form
    return data_dict, wordid_to_idx


def get_wordid_to_phonemes(wordid_to_phonemes_file: str, to_lower: bool = True):
    """
    WikiHomograph and NeMo use slightly different phoneme sets, this function reads WikiHomograph word_ids to NeMo
    IPA heteronyms mapping.

    Args:
        wordid_to_phonemes_file: Path to a file with mapping from wordid predicted by the model to phonemes, e.g.,
            NeMo/scripts/tts_dataset_files/wordid_to_ipa-0.7b_nv22.10.tsv
        to_lower: set to True to lower case wordid
    """
    if not os.path.exists(wordid_to_phonemes_file):
        raise ValueError(f"{wordid_to_phonemes_file} not found")

    wordid_to_nemo_cmu = {}
    with open(wordid_to_phonemes_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if to_lower:
                line = line.lower()
            line = line.strip().split("  ")
            wordid_to_nemo_cmu[line[0]] = line[1]
    return wordid_to_nemo_cmu


def remove_punctuation(text: str, remove_spaces: bool = False, do_lower: bool = False, exclude: List[str] = None):
    """
    Remove punctuation marks form text

    Args:
        text: input text
        remove_spaces: set to True to remove spaces
        do_lower: set to True to lower case the text
        exclude: specify list of punctuation marks keep in the output, e.g., exclude=["'", "."]

    Return:
        processed text with punctuation marks removed
    """
    all_punct_marks = string.punctuation

    if exclude is not None:
        for p in exclude:
            all_punct_marks = all_punct_marks.replace(p, "")

    text = re.sub("[" + all_punct_marks + "]", " ", text)

    text = re.sub(r" +", " ", text)
    if remove_spaces:
        text = text.replace(" ", "").replace("\u00A0", "").strip()

    if do_lower:
        text = text.lower()
    return text.strip()


def english_text_preprocessing(text, lower=True):
    text = unicode(text)
    text = ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn')
    text = ''.join(char if char not in SYNOGLYPH2ASCII else SYNOGLYPH2ASCII[char] for char in text)

    if lower:
        text = text.lower()

    return text


def any_locale_text_preprocessing(text: str) -> str:
    """
    Normalize unicode text with "NFC", and convert right single quotation mark (U+2019, decimal 8217) as an apostrophe.

    Args:
        text (str): the original input sentence.

    Returns: normalized text (str).
    """
    res = []
    for c in normalize_unicode_text(text):
        if c in ['’']:  # right single quotation mark (U+2019, decimal 8217) as an apostrophe
            res.append("'")
        else:
            res.append(c)

    return ''.join(res)


def normalize_unicode_text(text: str) -> str:
    """
    TODO @xueyang: Apply NFC form may be too aggressive since it would ignore some accented characters that do not exist
      in predefined German alphabet (nemo.collections.common.tokenizers.text_to_speech.ipa_lexicon.IPA_CHARACTER_SETS),
      such as 'é'. This is not expected. A better solution is to add an extra normalization with NFD to discard the
      diacritics and consider 'é' and 'e' produce similar pronunciations.

    Note that the tokenizer needs to run `unicodedata.normalize("NFC", x)` before calling `encode` function,
    especially for the characters that have diacritics, such as 'ö' in the German alphabet. 'ö' can be encoded as
    b'\xc3\xb6' (one char) as well as b'o\xcc\x88' (two chars). Without the normalization of composing two chars
    together and without a complete predefined set of diacritics, when the tokenizer reads the input sentence
    char-by-char, it would skip the combining diaeresis b'\xcc\x88', resulting in indistinguishable pronunciations
    for 'ö' and 'o'.

    Args:
        text (str): the original input sentence.

    Returns:
        NFC normalized sentence (str).
    """
    # normalize word with NFC form
    if not unicodedata.is_normalized("NFC", text):
        text = unicodedata.normalize("NFC", text)

    return text


def _word_tokenize(words: List[Tuple[str, str, str]], is_lower: bool = False) -> List[Tuple[List[str], bool]]:
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
            (["Hello"], False),
            ([" "], False),
            (["World"], False),
            ([" "], False),
            (["NVIDIA", "unchanged"], True),
            (["!"], False)
        ]

    Args:
        words (List[str]): a list of tuples like `(maybe_word, maybe_without_changes, maybe_punct)` where each element
            corresponds to a non-overlapping match of either `_WORDS_RE_EN` or `_WORDS_RE_ANY_LOCALE`.
        is_lower (bool): a flag to trigger lowercase all words. By default, it is False.

    Returns: List[Tuple[List[str], bool]], a list of tuples like `(a list of words, is_unchanged)`.

    """
    result = []
    for word in words:
        maybe_word, maybe_without_changes, maybe_punct = word

        without_changes = False
        if maybe_word != '':
            if is_lower:
                token = [maybe_word.lower()]
            else:
                token = [maybe_word]
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


def english_word_tokenize(text: str) -> List[Tuple[List[str], bool]]:
    words = _WORDS_RE_EN.findall(text)
    return _word_tokenize(words, is_lower=True)


def any_locale_word_tokenize(text: str) -> List[Tuple[List[str], bool]]:
    words = _WORDS_RE_ANY_LOCALE.findall(text)
    return _word_tokenize(words)


# TODO @xueyang: deprecate language-specific text preprocessing and use any_locale_text_preprocessing.
def spanish_text_preprocessing(text):
    return text.lower()


def chinese_text_preprocessing(text):
    return text.lower()


def get_heteronym_spans(sentences: List[str], supported_heteronyms: Union[Dict, List]):
    """
    Find heteronyms in sentences and returns span indices

    Args:
        sentences: sentences to find heteronyms in
        supported_heteronyms: heteronyms to look for

    Return:
        start_end: List[Tuple[int]] - start-end indices that indicate location of found heteronym in the sentence
        heteronyms: List[List[str]] - heteronyms found in sentences, each sentence can contain more than one heteronym
    """
    start_end = []
    heteronyms = []
    for sent in sentences:
        cur_start_end = []
        cur_heteronyms = []
        start_idx = 0
        for word in sent.lower().split():
            word_by_hyphen = word.split("-")
            for sub_word in word_by_hyphen:
                no_punct_word = remove_punctuation(sub_word, do_lower=True, remove_spaces=False)
                if no_punct_word in supported_heteronyms:
                    start_idx = sent.lower().index(no_punct_word, start_idx)
                    end_idx = start_idx + len(no_punct_word)
                    cur_start_end.append((start_idx, end_idx))
                    cur_heteronyms.append(no_punct_word)
                    start_idx = end_idx
                else:
                    start_idx += len(sub_word) + 1
        heteronyms.append(cur_heteronyms)
        start_end.append(cur_start_end)
    return start_end, heteronyms


def set_grapheme_case(text: str, case: str = "upper") -> str:
    if case == "upper":
        text_new = text.upper()
    elif case == "lower":
        text_new = text.lower()
    elif case == "mixed":  # keep as-is, mix-cases
        text_new = text
    else:
        raise ValueError(f"Case <{case}> is not supported. Please specify either 'upper', 'lower', or 'mixed'.")

    return text_new
