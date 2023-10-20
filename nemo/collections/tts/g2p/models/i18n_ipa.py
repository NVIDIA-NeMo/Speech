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
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from nemo.collections.common.tokenizers.text_to_speech.ipa_lexicon import validate_locale
from nemo.collections.common.tokenizers.text_to_speech.tokenizer_utils import (
    LATIN_CHARS_ALL,
    any_locale_word_tokenize,
    english_word_tokenize,
    normalize_unicode_text,
)
from nemo.collections.tts.g2p.models.base import BaseG2p
from nemo.collections.tts.g2p.utils import GRAPHEME_CASE_MIXED, GRAPHEME_CASE_UPPER, set_grapheme_case
from nemo.utils import logging
from nemo.utils.decorators import experimental


@experimental
class IpaG2p(BaseG2p):
    # fmt: off
    STRESS_SYMBOLS = ["ˈ", "ˌ"]
    # Regex for roman characters, accented characters, and locale-agnostic numbers/digits
    CHAR_REGEX = re.compile(fr"[{LATIN_CHARS_ALL}\d]")
    PUNCT_REGEX = re.compile(fr"[^{LATIN_CHARS_ALL}\d]")
    # fmt: on

    def __init__(
        self,
        phoneme_dict: Union[str, pathlib.Path, Dict[str, List[List[str]]]],
        locale: str = "en-US",
        apply_to_oov_word: Optional[Callable[[str], str]] = None,
        ignore_ambiguous_words: bool = True,
        heteronyms: Optional[Union[str, pathlib.Path, List[str]]] = None,
        use_chars: bool = False,
        phoneme_probability: Optional[float] = None,
        use_stresses: Optional[bool] = True,
        grapheme_case: Optional[str] = GRAPHEME_CASE_UPPER,
        grapheme_prefix: Optional[str] = "",
        mapping_file: Optional[str] = None,
    ) -> None:
        """
        Generic IPA G2P module. This module converts words from graphemes to International Phonetic Alphabet
        representations. Optionally, it can ignore heteronyms, ambiguous words, or words marked as unchangeable
        by `word_tokenize_func` (see code for details). Ignored words are left unchanged or passed through
        `apply_to_oov_word` for handling.

        Args:
            phoneme_dict (str, Path, or Dict): Path to file in CMUdict format or an IPA dict object with CMUdict-like
                entries. For example,
                a dictionary file: scripts/tts_dataset_files/ipa_cmudict-0.7b_nv22.06.txt;
                a dictionary object: {..., "Wire": [["ˈ", "w", "a", "ɪ", "ɚ"], ["ˈ", "w", "a", "ɪ", "ɹ"]], ...}.
            locale (str): Locale used to determine a locale-specific tokenization logic. Currently, it supports "en-US",
                "de-DE", and "es-ES". Defaults to "en-US". Specify None if implementing custom logic for a new locale.
            apply_to_oov_word (Callable): Function that deals with the out-of-vocabulary (OOV) words that do not exist
                in the `phoneme_dict`.
            ignore_ambiguous_words (bool): Whether to handle word via phoneme_dict with ambiguous phoneme sequences.
                Defaults to True.
            heteronyms (str, Path, List[str]): Path to file that includes heteronyms (one word entry per line), or a
                list of words.
            use_chars (bool): Whether to include chars/graphemes in the token list. It is True if `phoneme_probability`
                is not None or if `apply_to_oov_word` function ever returns graphemes.
            phoneme_probability (Optional[float]): The probability (0.0 <= ε <= 1.0) that is used to balance the action
                that a word in a sentence is whether transliterated into a sequence of phonemes, or kept as a sequence
                of graphemes. If a random number for a word is greater than ε, then the word is kept as graphemes;
                otherwise, the word is transliterated as phonemes. Defaults to None which is equivalent to setting it
                to 1.0, meaning always transliterating the word into phonemes. Note that this code path is only run if
                the word can be transliterated into phonemes, otherwise, if a word does not have an entry in the g2p
                dict, it will be kept as graphemes. If a word has multiple pronunciations as shown in the g2p dict and
                `ignore_ambiguous_words` is True, it will be kept as graphemes as well.
            use_stresses (Optional[bool]): Whether to include the stress symbols (ˈ and ˌ).
            grapheme_case (Optional[str]): Trigger converting all graphemes to uppercase, lowercase, or keeping them as
                original mix-cases. You may want to use this feature to distinguish the grapheme set from the phoneme
                set if there is an overlap in between. Defaults to `upper` because phoneme set only uses lowercase
                symbols. You could explicitly prepend `grapheme_prefix` to distinguish them.
            grapheme_prefix (Optional[str]): Prepend a special symbol to any graphemes in order to distinguish graphemes
                from phonemes because there may be overlaps between the two set. It is suggested to choose a prefix that
                is not used or preserved somewhere else. "#" could be a good candidate. Default to "".
            TODO @borisfom: add docstring for newly added `mapping_file` argument.
        """
        self.use_stresses = use_stresses
        self.grapheme_case = grapheme_case
        self.grapheme_prefix = grapheme_prefix
        self.phoneme_probability = phoneme_probability
        self.locale = locale
        self._rng = random.Random()

        if locale is not None:
            validate_locale(locale)

        if not use_chars and self.phoneme_probability is not None:
            self.use_chars = True
            logging.warning(
                "phoneme_probability was not None, characters will be enabled even though "
                "use_chars was set to False."
            )
        else:
            self.use_chars = use_chars

        phoneme_dict_obj = self._parse_phoneme_dict(phoneme_dict)

        # verify if phoneme dict obj is empty
        if phoneme_dict_obj:
            _phoneme_dict, self.symbols = self._normalize_dict(phoneme_dict_obj)
        else:
            raise ValueError(f"{phoneme_dict} contains no entries!")

        if apply_to_oov_word is None:
            logging.warning(
                "apply_to_oov_word=None, This means that some of words will remain unchanged "
                "if they are not handled by any of the rules in self.parse_one_word(). "
                "This may be intended if phonemes and chars are both valid inputs, otherwise, "
                "you may see unexpected deletions in your input."
            )

        # word_tokenize_func returns a List[Tuple[List[str], bool]] where every tuple denotes
        # a word representation (a list tokens) and a flag indicating whether to process the word or
        # leave it unchanged.
        if locale == "en-US":
            word_tokenize_func = english_word_tokenize
        else:
            word_tokenize_func = any_locale_word_tokenize

        super().__init__(
            phoneme_dict=_phoneme_dict,
            word_tokenize_func=word_tokenize_func,
            apply_to_oov_word=apply_to_oov_word,
            mapping_file=mapping_file,
        )

        self.ignore_ambiguous_words = ignore_ambiguous_words
        if isinstance(heteronyms, str) or isinstance(heteronyms, pathlib.Path):
            self.heteronyms = set(self._parse_file_by_lines(heteronyms))
        elif isinstance(heteronyms, list) and all(isinstance(het, str) for het in heteronyms):
            self.heteronyms = set(heteronyms)
        else:
            self.heteronyms = None

        if self.heteronyms:
            self.heteronyms = {set_grapheme_case(het, case=self.grapheme_case) for het in self.heteronyms}

    @staticmethod
    def _parse_phoneme_dict(
        phoneme_dict: Union[str, pathlib.Path, Dict[str, List[List[str]]]]
    ) -> Dict[str, List[List[str]]]:
        """
        parse an input IPA dictionary and save it as a dict object.

        Args:
            phoneme_dict (Union[str, pathlib.Path, dict]): Path to file in CMUdict format or an IPA dict object with
                CMUdict-like entries. For example,
                a dictionary file: scripts/tts_dataset_files/ipa_cmudict-0.7b_nv22.06.txt;
                a dictionary object: {..., "Wire": [["ˈ", "w", "a", "ɪ", "ɚ"], ["ˈ", "w", "a", "ɪ", "ɹ"]], ...}.

        Returns: a dict object (Dict[str, List[List[str]]]).
        """
        if isinstance(phoneme_dict, str) or isinstance(phoneme_dict, pathlib.Path):
            # load the dictionary file where there may exist a digit suffix after a word, e.g. "Word(2)", which
            # represents the pronunciation variant of that word.
            phoneme_dict_obj = defaultdict(list)
            _alt_re = re.compile(r"\([0-9]+\)")
            with open(phoneme_dict, "r", encoding="utf-8") as fdict:
                for line in fdict:
                    # skip the empty lines
                    if len(line) == 0:
                        continue

                    # Note that latin character pattern should be consistent with
                    # nemo.collections.tts.g2p.data.data_utils.LATIN_CHARS_ALL. It is advised to extend its character
                    # coverage if adding the support of new languages.
                    # TODO @xueyang: unify hardcoded range of characters with LATIN_CHARS_ALL to avoid duplicates.
                    line = normalize_unicode_text(line)

                    if (
                        'A' <= line[0] <= 'Z'
                        or 'a' <= line[0] <= 'z'
                        or 'À' <= line[0] <= 'Ö'
                        or 'Ø' <= line[0] <= 'ö'
                        or 'ø' <= line[0] <= 'ÿ'
                        or line[0] == "'"
                    ):
                        parts = line.strip().split(maxsplit=1)
                        word = re.sub(_alt_re, "", parts[0])
                        prons = re.sub(r"\s+", "", parts[1])
                        phoneme_dict_obj[word].append(list(prons))
        else:
            # Load phoneme_dict as dictionary object
            logging.info("Loading phoneme_dict as a Dict object, and validating its entry format.")

            phoneme_dict_obj = {}
            for word, prons in phoneme_dict.items():
                # validate dict entry format
                assert isinstance(
                    prons, list
                ), f"Pronunciation type <{type(prons)}> is not supported. Please convert to <list>."

                # normalize word with NFC form
                word = normalize_unicode_text(word)

                # normalize phonemes with NFC form
                prons = [[normalize_unicode_text(p) for p in pron] for pron in prons]

                phoneme_dict_obj.update({word: prons})

        return phoneme_dict_obj

    def replace_dict(self, phoneme_dict: Union[str, pathlib.Path, Dict[str, List[List[str]]]]):
        """
        Replace model's phoneme dictionary with a custom one
        """
        self.phoneme_dict = self._parse_phoneme_dict(phoneme_dict)

    @staticmethod
    def _parse_file_by_lines(p: Union[str, pathlib.Path]) -> List[str]:
        with open(p, 'r') as f:
            return [line.rstrip() for line in f.readlines()]

    def _prepend_prefix_for_one_word(self, word: str) -> List[str]:
        return [f"{self.grapheme_prefix}{character}" for character in word]

    def _normalize_dict(self, phoneme_dict_obj: Dict[str, List[List[str]]]) -> Tuple[Dict[str, List[List[str]]], Set]:
        """
        Parse a python dict object according to the decision on word cases and removal of lexical stress markers.

        Args:
            phoneme_dict_obj (Dict[str, List[List[str]]]): a dictionary object.
                e.g. {..., "Wire": [["ˈ", "w", "a", "ɪ", "ɚ"], ["ˈ", "w", "a", "ɪ", "ɹ"]], ...}

        Returns:
            g2p_dict (dict): processed dict.
            symbols (set): a IPA phoneme set, or its union with grapheme set.

        """
        g2p_dict = defaultdict(list)
        symbols = set()
        for word, prons in phoneme_dict_obj.items():
            # process word
            # update word cases.
            word_new = set_grapheme_case(word, case=self.grapheme_case)

            # add grapheme symbols if `use_chars=True`.
            if self.use_chars:
                # remove punctuations within a word. Punctuations can exist at the start, middle, and end of a word.
                word_no_punct = self.PUNCT_REGEX.sub('', word_new)

                # add prefix to distinguish graphemes from phonemes.
                symbols.update(self._prepend_prefix_for_one_word(word_no_punct))

            # process IPA pronunciations
            # update phoneme symbols by removing lexical stress markers if `use_stresses=False`.
            prons_new = list()
            if not self.use_stresses:
                for pron in prons:
                    prons_new.append([symbol for symbol in pron if symbol not in self.STRESS_SYMBOLS])
            else:
                prons_new = prons

            # update symbols
            for pron in prons_new:
                symbols.update(pron)  # This will insert each char individually

            # update dict entry
            g2p_dict[word_new] = prons_new

            # duplicate word entries if grapheme_case is mixed. Even though grapheme_case is set to mixed, the words in
            # the original input text and the g2p_dict remain unchanged, so they could still be either lowercase,
            # uppercase, or mixed-case as defined in `set_grapheme_case` func. When mapping an uppercase word, e.g.
            # "HELLO", into phonemes using the g2p_dict with {"Hello": [["həˈɫoʊ"]]}, "HELLO" can't find its
            # pronunciations in the g2p_dict due to the case-mismatch of the words. Augmenting the g2p_dict with its
            # uppercase word entry, e.g. {"Hello": [["həˈɫoʊ"]], "HELLO": [["həˈɫoʊ"]]} would provide possibility to
            # find "HELLO"'s pronunciations rather than directly considering it as an OOV.
            if self.grapheme_case == GRAPHEME_CASE_MIXED and not word_new.isupper():
                g2p_dict[word_new.upper()] = prons_new

        return g2p_dict, symbols

    def replace_symbols(self, symbols, keep_alternate=True):
        """Replaces the vocabulary of symbols with the one given.
        Also filters out any entries with illegal graphemes or phonemes according to the new vocab.

        Args:
            symbols (List, Set): User-provided set of valid symbols, both graphemes and phonemes
            keep_alternate (bool): Whether to keep the other pronunciation(s) of a word if not all contain
                illegal phonemes (and the word doesn't contain illegal graphemes).
                Warning: this may change a word from being ambiguous to having only one valid pronunciation.
                Defaults to True.
        """
        new_symbols = set(symbols)

        # Keep track of what will need to be deleted or (if keep_alternate=True) replaced
        deletion_words = []
        replacement_dict = {}

        for word, prons in self.phoneme_dict.items():
            # Check for illegal grapheme in the word itself
            word_graphemes = set(self._prepend_prefix_for_one_word(set_grapheme_case(word, self.grapheme_case)))
            word_diff = word_graphemes - new_symbols
            if word_diff:
                deletion_words.append(word)
                continue

            # Check for illegal phonemes in the pronunciation(s)
            legal_prons = []
            for pron in prons:
                pron_diff = set(pron) - new_symbols
                if not pron_diff:
                    legal_prons.append(pron)

            # Check if at least one pronunciation was illegal
            if len(legal_prons) != len(prons):
                if not keep_alternate:  # Remove the word and entry fully
                    deletion_words.append(word)
                else:  # Need to check if all prons were illegal
                    if not legal_prons:
                        deletion_words.append(word)
                    else:
                        replacement_dict[word] = legal_prons

        # Update pronunciation dictionary as needed
        for del_word in deletion_words:
            del self.phoneme_dict[del_word]

        if keep_alternate:
            self.phoneme_dict.update(replacement_dict)

        self.symbols = new_symbols

    def is_unique_in_phoneme_dict(self, word: str) -> bool:
        return len(self.phoneme_dict[word]) == 1

    def parse_one_word(self, word: str) -> Tuple[List[str], bool]:
        """Returns parsed `word` and `status` (bool: False if word wasn't handled, True otherwise).
        """
        word = set_grapheme_case(word, case=self.grapheme_case)

        # Punctuation (assumes other chars have been stripped)
        if self.CHAR_REGEX.search(word) is None:
            return list(word), True

        # Keep graphemes of a word with a probability.
        if self.phoneme_probability is not None and self._rng.random() > self.phoneme_probability:
            return self._prepend_prefix_for_one_word(word), True

        # Heteronyms
        if self.heteronyms and word in self.heteronyms:
            return self._prepend_prefix_for_one_word(word), True

        # special cases for en-US when transliterating a word into a list of phonemes.
        # TODO @xueyang: add special cases for any other languages upon new findings.
        if self.locale == "en-US":
            # `'s` suffix (with apostrophe) - not in phoneme dict
            if len(word) > 2 and (word.endswith("'s") or word.endswith("'S")):
                word_found = None
                if (word not in self.phoneme_dict) and (word.upper() not in self.phoneme_dict):
                    if word[:-2] in self.phoneme_dict:
                        word_found = word[:-2]
                    elif word[:-2].upper() in self.phoneme_dict:
                        word_found = word[:-2].upper()

                if word_found is not None and (
                    not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word_found)
                ):
                    if word_found[-1] in ['T', 't']:
                        # for example, "airport's" doesn't exist in the dict while "airport" exists. So append a phoneme
                        # /s/ at the end of "airport"'s first pronunciation.
                        return self.phoneme_dict[word_found][0] + ["s"], True
                    elif word_found[-1] in ['S', 's']:
                        # for example, "jones's" doesn't exist in the dict while "jones" exists. So append two phonemes,
                        # /ɪ/ and /z/ at the end of "jones"'s first pronunciation.
                        return self.phoneme_dict[word_found][0] + ["ɪ", "z"], True
                    else:
                        return self.phoneme_dict[word_found][0] + ["z"], True

            # `s` suffix (without apostrophe) - not in phoneme dict
            if len(word) > 1 and (word.endswith("s") or word.endswith("S")):
                word_found = None
                if (word not in self.phoneme_dict) and (word.upper() not in self.phoneme_dict):
                    if word[:-1] in self.phoneme_dict:
                        word_found = word[:-1]
                    elif word[:-1].upper() in self.phoneme_dict:
                        word_found = word[:-1].upper()

                if word_found is not None and (
                    not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word_found)
                ):
                    if word_found[-1] in ['T', 't']:
                        # for example, "airports" doesn't exist in the dict while "airport" exists. So append a phoneme
                        # /s/ at the end of "airport"'s first pronunciation.
                        return self.phoneme_dict[word_found][0] + ["s"], True
                    else:
                        return self.phoneme_dict[word_found][0] + ["z"], True

        if self.locale == "fr-FR":
            # contracted prefix (with apostrophe) - not in phoneme dict
            contractions_g = ['l', 'c', 'd', 'j', 'm', 'n', 'qu', 's', 't', 'puisqu', 'lorsqu', 'jusqu']
            contractions_p = ['l', 's', 'd', 'ʒ', 'm', 'n', 'k', 's', 't', 'pyisk', 'loʁsk', 'ʒysk']

            for cont_g, cont_p in zip(contractions_g, contractions_p):
                starter = cont_g + "'"
                if len(word) > 2 and (word.startswith(starter) or word.startswith(starter.upper())):
                    word_found = None
                    if (word not in self.phoneme_dict) and (word.upper() not in self.phoneme_dict):
                        start_index = len(starter)
                        if word[start_index:] in self.phoneme_dict:
                            word_found = word[start_index:]
                        elif word[start_index:].upper() in self.phoneme_dict:
                            word_found = word[start_index:].upper()

                    if word_found is not None and (
                        not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word_found)
                    ):
                        return [c for c in cont_p] + self.phoneme_dict[word_found][0], True

        # For the words that have a single pronunciation, directly look it up in the phoneme_dict; for the
        # words that have multiple pronunciation variants, if we don't want to ignore them, then directly choose their
        # first pronunciation variant as the target phonemes.
        # TODO @xueyang: this is a temporary solution, but it is not optimal if always choosing the first pronunciation
        #  variant as the target if a word has multiple pronunciation variants. We need explore better approach to
        #  select its optimal pronunciation variant aligning with its reference audio.
        if word in self.phoneme_dict and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word)):
            return self.phoneme_dict[word][0], True

        if (
            self.grapheme_case == GRAPHEME_CASE_MIXED
            and word not in self.phoneme_dict
            and word.upper() in self.phoneme_dict
        ):
            word = word.upper()
            if not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word):
                return self.phoneme_dict[word][0], True

        if self.apply_to_oov_word is not None:
            return self.apply_to_oov_word(word), True
        else:
            return self._prepend_prefix_for_one_word(word), False

    def __call__(self, text: str) -> List[str]:
        text = normalize_unicode_text(text)

        if self.heteronym_model is not None:
            try:
                text = self.heteronym_model.disambiguate(sentences=[text])[1][0]
            except Exception as e:
                logging.warning(f"Heteronym model failed {e}, skipping")

        words_list_of_tuple = self.word_tokenize_func(text)

        prons = []
        for words, without_changes in words_list_of_tuple:
            if without_changes:
                # for example: (["NVIDIA", "unchanged"], True). "NVIDIA" is considered as a single token.
                prons.extend([f"{self.grapheme_prefix}{word}" for word in words])
            else:
                assert (
                    len(words) == 1
                ), f"{words} should only have a single item when `without_changes` is False, but found {len(words)}."

                word = words[0]
                pron, is_handled = self.parse_one_word(word)

                # If `is_handled` is False, then the only possible case is that the word is an OOV. The OOV may have a
                # hyphen so that it doesn't show up in the g2p dictionary. We need split it into sub-words by a hyphen,
                # and parse the sub-words again just in case any sub-word exists in the g2p dictionary.
                if not is_handled:
                    subwords_by_hyphen = word.split("-")
                    if len(subwords_by_hyphen) > 1:
                        pron = []  # reset the previous pron
                        for sub_word in subwords_by_hyphen:
                            p, _ = self.parse_one_word(sub_word)
                            pron.extend(p)
                            pron.append("-")
                        pron.pop()  # remove the redundant hyphen that is previously appended at the end of the word.

                prons.extend(pron)

        return prons
