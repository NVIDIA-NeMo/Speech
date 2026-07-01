import json
import pathlib
import random
import re
import unicodedata
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

from nemo.collections.tts.g2p.models.base import BaseG2p
from nemo.utils import logging

_URDU_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+")
_WHITESPACE = re.compile(r"\s+")


def urdu_word_tokenize(text):
    tokens = _WHITESPACE.split(text.strip())
    result = []
    for token in tokens:
        if not token:
            continue
        if _URDU_CHAR_PATTERN.fullmatch(token):
            result.append(([token], False))
        else:
            result.append(([token], True))
    return result


class UrduIpaG2p(BaseG2p):
    STRESS_SYMBOLS = ["\u02c8", "\u02cc"]

    def __init__(
        self,
        phoneme_dict,
        apply_to_oov_word=None,
        ignore_ambiguous_words=True,
        heteronyms=None,
        use_stresses=True,
        phoneme_probability=None,
        max_phrase_len=4,
        mapping_file=None,
    ):
        self.use_stresses = use_stresses
        self.ignore_ambiguous_words = ignore_ambiguous_words
        self.phoneme_probability = phoneme_probability
        self.max_phrase_len = max_phrase_len
        self._rng = random.Random()
        parsed_dict = self._parse_urdu_json_dict(phoneme_dict, use_stresses, self.STRESS_SYMBOLS)
        if not parsed_dict:
            raise ValueError("UrduIpaG2p: phoneme_dict contains no valid entries!")
        if apply_to_oov_word is None:
            logging.warning("apply_to_oov_word=None. OOV words returned as grapheme characters.")
        if isinstance(heteronyms, (str, pathlib.Path)):
            with open(heteronyms, "r", encoding="utf-8") as f:
                self.heteronyms = set(line.rstrip() for line in f)
        elif isinstance(heteronyms, list):
            self.heteronyms = set(heteronyms)
        else:
            self.heteronyms = None
        super().__init__(
            phoneme_dict=parsed_dict,
            word_tokenize_func=urdu_word_tokenize,
            apply_to_oov_word=apply_to_oov_word,
            mapping_file=mapping_file,
        )
        logging.info(f"UrduIpaG2p: loaded {len(self.phoneme_dict)} entries.")

    @staticmethod
    def _parse_urdu_json_dict(phoneme_dict, use_stresses, stress_symbols):
        if isinstance(phoneme_dict, (str, pathlib.Path)):
            with open(phoneme_dict, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = phoneme_dict
        result = defaultdict(list)
        for grapheme, ipa in raw.items():
            grapheme = unicodedata.normalize("NFC", grapheme.strip())
            variants = [ipa] if isinstance(ipa, str) else ipa
            for variant in variants:
                variant = unicodedata.normalize("NFC", variant.strip())
                tokens = variant.split()
                if not use_stresses:
                    tokens = ["".join(c for c in t if c not in stress_symbols) for t in tokens]
                    tokens = [t for t in tokens if t]
                if tokens:
                    result[grapheme].append(tokens)
        return dict(result)

    def is_unique_in_phoneme_dict(self, word):
        return len(self.phoneme_dict[word]) == 1

    def parse_one_word(self, word):
        if self.phoneme_probability is not None and self._rng.random() > self.phoneme_probability:
            return list(word), True
        if self.heteronyms and word in self.heteronyms:
            return list(word), True
        if word in self.phoneme_dict and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word)):
            return self.phoneme_dict[word][0], True
        if self.apply_to_oov_word is not None:
            return self.apply_to_oov_word(word), True
        return list(word), False

    def __call__(self, text):
        text = unicodedata.normalize("NFC", text)
        token_tuples = self.word_tokenize_func(text)
        words, unchanged_mask = [], []
        for tokens, without_changes in token_tuples:
            for tok in tokens:
                words.append(tok)
                unchanged_mask.append(without_changes)
        prons = []
        i = 0
        while i < len(words):
            if unchanged_mask[i]:
                prons.append(words[i])
                i += 1
                continue
            matched = False
            for phrase_len in range(min(self.max_phrase_len, len(words) - i), 0, -1):
                phrase = " ".join(words[i : i + phrase_len])
                if phrase in self.phoneme_dict and (
                    not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(phrase)
                ):
                    prons.extend(self.phoneme_dict[phrase][0])
                    i += phrase_len
                    matched = True
                    break
            if not matched:
                word = words[i]
                pron, is_handled = self.parse_one_word(word)
                if not is_handled:
                    subwords = word.split("-")
                    if len(subwords) > 1:
                        pron = []
                        for sub in subwords:
                            p, _ = self.parse_one_word(sub)
                            pron.extend(p)
                            pron.append("-")
                        pron.pop()
                prons.extend(pron)
                i += 1
        return prons
