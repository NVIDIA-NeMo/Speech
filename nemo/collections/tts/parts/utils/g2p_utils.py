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


import csv
import os
import re
import string
from typing import Dict, List, Union

__all__ = [
    "read_wordids",
    "set_grapheme_case",
    "GRAPHEME_CASE_UPPER",
    "GRAPHEME_CASE_LOWER",
    "GRAPHEME_CASE_MIXED",
    "get_heteronym_spans",
]


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
