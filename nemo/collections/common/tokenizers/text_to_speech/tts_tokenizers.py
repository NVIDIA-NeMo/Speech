# -*- coding: utf-8 -*-
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

import itertools
import string
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Optional

from nemo.collections.common.tokenizers.text_to_speech.ipa_lexicon import (
    get_grapheme_character_set,
    get_ipa_punctuation_list,
    validate_locale,
)
from nemo.collections.common.tokenizers.text_to_speech.tokenizer_utils import (
    any_locale_text_preprocessing,
    chinese_text_preprocessing,
    english_text_preprocessing,
    french_text_preprocessing,
    italian_text_preprocessing,
    japanese_text_preprocessing,
    spanish_text_preprocessing,
    vietnamese_text_preprocessing,
)
from nemo.utils import logging
from nemo.utils.decorators import experimental


class BaseTokenizer(ABC):
    PAD, BLANK, OOV = '<pad>', '<blank>', '<oov>'

    def __init__(self, tokens, *, pad=PAD, blank=BLANK, oov=OOV, sep='', add_blank_at=None):
        """Abstract class for creating an arbitrary tokenizer to convert string to list of int tokens.
        Args:
            tokens: List of tokens.
            pad: Pad token as string.
            blank: Blank token as string.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
        """
        super().__init__()

        tokens = list(tokens)
        # TODO @xueyang: in general, IDs of pad, sil, blank, and oov are preserved ahead instead of dynamically
        #  assigned according to the number of tokens. The downside of using dynamical assignment leads to different IDs
        #  for each.
        self.pad, tokens = len(tokens), tokens + [pad]  # Padding

        if add_blank_at is not None:
            self.blank, tokens = len(tokens), tokens + [blank]  # Reserved for blank from asr-model
        else:
            # use add_blank_at=None only for ASR where blank is added automatically, disable blank here
            self.blank = None

        self.oov, tokens = len(tokens), tokens + [oov]  # Out Of Vocabulary

        if add_blank_at == "last":
            tokens[-1], tokens[-2] = tokens[-2], tokens[-1]
            self.oov, self.blank = self.blank, self.oov

        self.tokens = tokens
        self.sep = sep

        self._util_ids = {self.pad, self.blank, self.oov}
        self._token2id = {l: i for i, l in enumerate(tokens)}
        self._id2token = tokens

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Turns str text into int tokens."""
        pass

    def decode(self, tokens: List[int]) -> str:
        """Turns ints tokens into str text."""
        return self.sep.join(self._id2token[t] for t in tokens if t not in self._util_ids)


class BaseCharsTokenizer(BaseTokenizer):
    # fmt: off
    # TODO @xueyang: unify definition of the default PUNCT_LIST and import from ipa_lexicon.py
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    # fmt: on

    def __init__(
        self,
        chars,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=lambda x: x,
    ):
        """Base class for char-based tokenizer.
        Args:
            chars: string that represents all possible characters.
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
        """

        tokens = []
        self.space, tokens = len(tokens), tokens + [' ']  # Space
        tokens.extend(chars)
        if apostrophe:
            tokens.append("'")  # Apostrophe for saving "don't" and "Joe's"

        if punct:
            if non_default_punct_list is not None:
                self.PUNCT_LIST = non_default_punct_list
            tokens.extend(self.PUNCT_LIST)

        super().__init__(tokens, add_blank_at=add_blank_at)

        self.punct = punct
        self.pad_with_space = pad_with_space

        self.text_preprocessing_func = text_preprocessing_func

    def encode(self, text):
        """See base class."""
        cs, space, tokens = [], self.tokens[self.space], set(self.tokens)

        text = self.text_preprocessing_func(text)
        for c in text:
            # Add a whitespace if the current char is a whitespace while the previous char is not a whitespace.
            if c == space and len(cs) > 0 and cs[-1] != space:
                cs.append(c)
            # Add the current char that is an alphanumeric or an apostrophe.
            elif (c.isalnum() or c == "'") and c in tokens:
                cs.append(c)
            # Add a punctuation that has a single char.
            elif (c in self.PUNCT_LIST) and self.punct:
                cs.append(c)
            # Warn about unknown char
            elif c != space:
                logging.warning(f"Text: [{text}] contains unknown char: [{c}]. Symbol will be skipped.")

        # Remove trailing spaces
        if cs:
            while cs[-1] == space:
                cs.pop()

        if self.pad_with_space:
            cs = [space] + cs + [space]

        return [self._token2id[p] for p in cs]


class EnglishCharsTokenizer(BaseCharsTokenizer):
    def __init__(
        self,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=english_text_preprocessing,
    ):
        """English char-based tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Basically, it replaces all non-unicode characters with unicode ones and apply lower() function.
        """
        super().__init__(
            chars=string.ascii_lowercase,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=text_preprocessing_func,
        )


class VietnameseCharsTokenizer(BaseCharsTokenizer):

    _LOCALE = "vi-VN"
    _CHARSET_STR = get_grapheme_character_set(locale=_LOCALE, case="mixed")

    def __init__(
        self,
        chars=_CHARSET_STR,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=vietnamese_text_preprocessing,
    ):
        """Vietnamese grapheme tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
            if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer. By default, it
            would keep any word lowercase.
        """
        super().__init__(
            chars=chars,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=vietnamese_text_preprocessing,
        )


class GermanCharsTokenizer(BaseCharsTokenizer):

    _LOCALE = "de-DE"
    _PUNCT_LIST = get_ipa_punctuation_list(_LOCALE)
    _CHARSET_STR = get_grapheme_character_set(locale=_LOCALE, case="mixed")

    def __init__(
        self,
        chars=_CHARSET_STR,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=_PUNCT_LIST,
        text_preprocessing_func=any_locale_text_preprocessing,
    ):
        """German grapheme-based tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer. By default, it
            would keep any word unchanged.
        """
        super().__init__(
            chars=chars,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=text_preprocessing_func,
        )


class SpanishCharsTokenizer(BaseCharsTokenizer):

    PUNCT_LIST = get_ipa_punctuation_list("es-ES")

    def __init__(
        self,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
    ):
        """Spanish grapheme tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
        """

        es_alphabet = "abcdefghijklmnopqrstuvwxyzáéíñóúü"
        super().__init__(
            chars=es_alphabet,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=spanish_text_preprocessing,
        )


class FrenchCharsTokenizer(BaseCharsTokenizer):

    PUNCT_LIST = get_ipa_punctuation_list("fr-FR")

    def __init__(
        self,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
    ):
        """French grapheme tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
            if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
        """

        fr_alphabet = get_grapheme_character_set(locale="fr-FR", case="lower")
        super().__init__(
            chars=fr_alphabet,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=french_text_preprocessing,
        )


class ItalianCharsTokenizer(BaseCharsTokenizer):
    PUNCT_LIST = get_ipa_punctuation_list("it-IT")

    def __init__(
        self, punct=True, apostrophe=True, add_blank_at=None, pad_with_space=False, non_default_punct_list=None
    ):
        """Italian grapheme tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
            if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
        """

        it_alphabet = "abcdefghijklmnopqrstuvwxyzàèéìòùó"
        super().__init__(
            chars=it_alphabet,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=italian_text_preprocessing,
        )


class GermanPhonemesTokenizer(BaseCharsTokenizer):
    # fmt: off
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    # fmt: on

    def __init__(
        self,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=any_locale_text_preprocessing,
    ):
        """Deutsch phoneme-based tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Currently, it only applies lower() function.
        """

        de_ipa = "abdefhijklmnoprstuvwxyzçðøŋœɐɑɒɔəɛɜɡɪɹɾʃʊʌʒː̃"
        de_suprasegmentals = "12"
        super().__init__(
            chars=de_ipa + de_suprasegmentals,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=text_preprocessing_func,
        )

    def encode(self, text):
        """See base class."""
        cs, space, tokens = [], self.tokens[self.space], set(self.tokens)

        text = self.text_preprocessing_func(text)
        for c in text:
            # Add space if last one isn't one
            if c == space and len(cs) > 0 and cs[-1] != space:
                cs.append(c)
            # Add next char
            elif (c.isalnum() or c == "'" or c == "\u0303") and c in tokens:
                cs.append(c)
            # Add punct
            elif (c in self.PUNCT_LIST) and self.punct:
                cs.append(c)
            # Warn about unknown char
            elif c != space:
                logging.warning(f"Text: [{text}] contains unknown char: [{c}]. Symbol will be skipped.")

        # Remove trailing spaces
        while cs[-1] == space:
            cs.pop()

        if self.pad_with_space:
            cs = [space] + cs + [space]

        return [self._token2id[p] for p in cs]


class ItalianPhonemesTokenizer(BaseCharsTokenizer):
    # fmt: off
    PUNCT_LIST = (
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
        '„', '“', '”', '‘', '’', '‒', '—', '«', '»', '‹', '›', '_',
    )
    # fmt: on

    def __init__(
        self,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=italian_text_preprocessing,
    ):
        """Italian phoneme-based tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Currently, it only applies lower() function.
        """

        it_ipa = "abcdefghijklmnopqrstuvwxyzàèéìòùóæɐɑɔəɚɜɬɹʌʔᵻðŋɛɡɣɪɲɾʃʊʎʒʝβθd͡'t͡'øɒɕɓçɖɘɝɞɟʄɡɠɢʛɦɧħɥʜɨɬɫɮʟɱɯɰɳɵɸœɶʘɺɻʀʁɽʂʈʧʉʋⱱɤʍχʏʑʐʔʡʕʢǀǁǂᵻʃ'ː"
        super().__init__(
            chars=it_ipa,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=text_preprocessing_func,
        )

    def encode(self, text):
        """See base class."""
        cs, space, tokens = [], self.tokens[self.space], set(self.tokens)

        text = self.text_preprocessing_func(text)
        for c in text:
            # Add space if last one isn't one
            if c == space and len(cs) > 0 and cs[-1] != space:
                cs.append(c)
            # Add next char
            elif (c.isalnum() or c == "'" or c == "\u0303") and c in tokens:
                cs.append(c)
            # Add punct
            elif (c in self.PUNCT_LIST) and self.punct:
                cs.append(c)
            # Warn about unknown char
            elif c != space:
                logging.warning(f"Text: [{text}] contains unknown char: [{c}]. Symbol will be skipped.")

        # Remove trailing spaces
        while cs[-1] == space:
            cs.pop()

        if self.pad_with_space:
            cs = [space] + cs + [space]

        return [self._token2id[p] for p in cs]


class EnglishPhonemesTokenizer(BaseTokenizer):
    # fmt: off
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    VOWELS = (
        'AA', 'AE', 'AH', 'AO', 'AW',
        'AY', 'EH', 'ER', 'EY', 'IH',
        'IY', 'OW', 'OY', 'UH', 'UW',
    )
    CONSONANTS = (
        'B', 'CH', 'D', 'DH', 'F', 'G',
        'HH', 'JH', 'K', 'L', 'M', 'N',
        'NG', 'P', 'R', 'S', 'SH', 'T',
        'TH', 'V', 'W', 'Y', 'Z', 'ZH',
    )
    # fmt: on

    def __init__(
        self,
        g2p,
        punct=True,
        non_default_punct_list=None,
        stresses=False,
        chars=False,
        *,
        space=' ',
        silence=None,
        apostrophe=True,
        oov=BaseTokenizer.OOV,
        sep='|',  # To be able to distinguish between 2/3 letters codes.
        add_blank_at=None,
        pad_with_space=False,
        text_preprocessing_func=lambda text: english_text_preprocessing(text, lower=False),
    ):
        """English phoneme-based tokenizer.
        Args:
            g2p: Grapheme to phoneme module.
            punct: Whether to reserve grapheme for basic punctuation or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            stresses: Whether to use phonemes codes with stresses (0-2) or not.
            chars: Whether to additionally use chars together with phonemes. It is useful if g2p module can return chars too.
            space: Space token as string.
            silence: Silence token as string (will be disabled if it is None).
            apostrophe: Whether to use apostrophe or not.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Basically, it replaces all non-unicode characters with unicode ones.
             Note that lower() function shouldn't be applied here, in case the text contains phonemes (it will be handled by g2p).
        """

        self.phoneme_probability = None
        if hasattr(g2p, "phoneme_probability"):
            self.phoneme_probability = g2p.phoneme_probability
        tokens = []
        self.space, tokens = len(tokens), tokens + [space]  # Space

        if silence is not None:
            self.silence, tokens = len(tokens), tokens + [silence]  # Silence

        tokens.extend(self.CONSONANTS)
        vowels = list(self.VOWELS)

        if stresses:
            vowels = [f'{p}{s}' for p, s in itertools.product(vowels, (0, 1, 2))]
        tokens.extend(vowels)

        if chars or self.phoneme_probability is not None:
            if not chars:
                logging.warning(
                    "phoneme_probability was not None, characters will be enabled even though "
                    "chars was set to False."
                )
            tokens.extend(string.ascii_lowercase)

        if apostrophe:
            tokens.append("'")  # Apostrophe

        if punct:
            if non_default_punct_list is not None:
                self.PUNCT_LIST = non_default_punct_list
            tokens.extend(self.PUNCT_LIST)

        super().__init__(tokens, oov=oov, sep=sep, add_blank_at=add_blank_at)

        self.chars = chars if self.phoneme_probability is None else True
        self.punct = punct
        self.stresses = stresses
        self.pad_with_space = pad_with_space

        self.text_preprocessing_func = text_preprocessing_func
        self.g2p = g2p

    def encode(self, text):
        """See base class for more information."""

        text = self.text_preprocessing_func(text)
        g2p_text = self.g2p(text)  # TODO: handle infer
        return self.encode_from_g2p(g2p_text, text)

    def encode_from_g2p(self, g2p_text: List[str], raw_text: Optional[str] = None):
        """
        Encodes text that has already been run through G2P.
        Called for encoding to tokens after text preprocessing and G2P.

        Args:
            g2p_text: G2P's output, could be a mixture of phonemes and graphemes,
                e.g. "see OOV" -> ['S', 'IY1', ' ', 'O', 'O', 'V']
            raw_text: original raw input
        """
        ps, space, tokens = [], self.tokens[self.space], set(self.tokens)
        for p in g2p_text:  # noqa
            # Remove stress
            if p.isalnum() and len(p) == 3 and not self.stresses:
                p = p[:2]

            # Add space if last one isn't one
            if p == space and len(ps) > 0 and ps[-1] != space:
                ps.append(p)
            # Add next phoneme or char (if chars=True)
            elif (p.isalnum() or p == "'") and p in tokens:
                ps.append(p)
            # Add punct
            elif (p in self.PUNCT_LIST) and self.punct:
                ps.append(p)
            # Warn about unknown char/phoneme
            elif p != space:
                message = f"Text: [{''.join(g2p_text)}] contains unknown char/phoneme: [{p}]."
                if raw_text is not None:
                    message += f"Original text: [{raw_text}]. Symbol will be skipped."
                logging.warning(message)

        # Remove trailing spaces
        if ps:
            while ps[-1] == space:
                ps.pop()

        if self.pad_with_space:
            ps = [space] + ps + [space]

        return [self._token2id[p] for p in ps]

    @contextmanager
    def set_phone_prob(self, prob):
        if hasattr(self.g2p, "phoneme_probability"):
            self.g2p.phoneme_probability = prob
        try:
            yield
        finally:
            if hasattr(self.g2p, "phoneme_probability"):
                self.g2p.phoneme_probability = self.phoneme_probability


@experimental
class IPATokenizer(BaseTokenizer):
    def __init__(
        self,
        g2p,
        locale="en-US",
        punct=True,
        non_default_punct_list=None,
        fixed_vocab=None,
        *,
        space=' ',
        silence=None,
        apostrophe=False,
        oov=BaseTokenizer.OOV,
        sep='|',  # To be able to distinguish between symbols
        add_blank_at=None,
        pad_with_space=False,
    ):
        """General-purpose IPA-based tokenizer.
        Args:
            g2p: Grapheme to phoneme module, should be IpaG2p or some subclass thereof.
            locale: Locale used to determine default text processing logic and punctuation.
                Supports ["en-US", "de-DE", "es-ES", "fr-FR"]. Defaults to "en-US".
                Specify None if implementing custom logic for a new locale.
            punct: Whether to reserve grapheme for basic punctuation or not.
            non_default_punct_list: List of punctuation marks which will be used instead default, if any.
            fixed_vocab: List of valid grapheme/phoneme tokens for the model.
                Set only if overriding the default vocab generation process (reading from G2P dict).
                If set, any dataset entries that have unincluded graphemes will be filtered out, and any words whose
                pronunciations have unincluded phonemes will be treated as OOV.
                Please make sure that the grapheme prefixes and cases are consistent with the G2P module's settings.
                Defaults to None, which means default vocab generation is used.
            space: Space token as string.
            silence: Silence token as string (will be disabled if it is None).
            apostrophe: Whether to use apostrophe or not.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
                if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
        """
        if not hasattr(g2p, "symbols"):
            logging.error(
                f"Please make sure the G2P module passed into the IPATokenizer has a `symbols` attribute. "
                f"This is required in order to build the tokenizer vocabulary.\n"
                f"Expected e.g. IpaG2p, found {type(g2p)}"
            )
            raise ValueError("G2P modules passed into the IPATokenizer must have `symbols` defined.")

        if locale is not None:
            validate_locale(locale)

        self.phoneme_probability = None
        if hasattr(g2p, "phoneme_probability"):
            self.phoneme_probability = g2p.phoneme_probability

        if locale == "en-US":
            self.text_preprocessing_func = lambda text: english_text_preprocessing(text, lower=False)
        else:
            self.text_preprocessing_func = any_locale_text_preprocessing

        # Build tokens list if fixed_vocab isn't set
        if fixed_vocab:
            tokens = {self.text_preprocessing_func(c) for c in fixed_vocab}
            self.set_fixed_vocab = True  # Used to check whether dataset entries need filtering

            if g2p.symbols == tokens:
                logging.info(
                    "Did not replace G2P valid symbol set since the given set is equivalent to the existing one."
                )
                self.set_fixed_vocab = False
            else:
                g2p.replace_symbols(tokens)
        else:
            tokens = set(g2p.symbols)
            self.set_fixed_vocab = False

        if apostrophe:
            tokens.add("'")

        if punct:
            if non_default_punct_list is not None:
                self.punct_list = non_default_punct_list
            else:
                self.punct_list = get_ipa_punctuation_list(locale)

            tokens.update(self.punct_list)

        # Sort to ensure that vocab is in the same order every time
        tokens = sorted(list(tokens))

        if space in g2p.symbols:
            self.space = tokens.index(space)
        else:
            self.space, tokens = len(tokens), tokens + [space]

        if silence is not None:
            self.silence, tokens = len(tokens), tokens + [silence]

        super().__init__(tokens, oov=oov, sep=sep, add_blank_at=add_blank_at)

        self.tokens_set = set(self.tokens)  # To save some repeated work when filtering entries

        self.punct = punct
        self.pad_with_space = pad_with_space

        self.g2p = g2p

    def encode(self, text: str) -> List[int]:
        """See base class for more information."""
        # normalize the input text with "NFC" form.
        text = self.text_preprocessing_func(text)

        # transliterate the text into phoneme sequences and/or grapheme sequences.
        g2p_text = self.g2p(text)

        return self.encode_from_g2p(g2p_text, text)

    def encode_from_g2p(self, g2p_text: List[str], raw_text: Optional[str] = None) -> List[int]:
        """
        Tokenize the `g2p_text` that has been already run through G2P. Each item in the `g2p_text` would be encoded as
        one of the integer IDs predefined in `self._token2id`. Note that this function should be called after
        `self.text_preprocessing_func` and `self.g2p` functions

        Args:
            g2p_text (List[str]): a sequence of tokens from G2P's output. It could be a sequence of phonemes, a sequence
                of graphemes, or a mixture of both. For example, `['ˈ', 's', 'i', ' ', '#O', '#O', '#V']`, which is the
                G2P's output of the text "see OOV", where '#' is prepended to each grapheme in order to distinguish
                graphemes from phonemes if there are overlaps in between. The prefix '#' can be customized in
                `nemo.collections.tts.g2p.models.i18n_ipa.IpaG2p.grapheme_prefix`.
            raw_text (str): the original text after calling `self.text_preprocessing_func`. It is optional. It is only
                used to deliver a warning message that some graphemes from the original text are skipped.

        Returns: a list of integer IDs that tokenize the `g2p_text`.
        """
        ps, space, tokens = [], self.tokens[self.space], set(self.tokens)
        for p in g2p_text:
            if p == space and len(ps) > 0 and ps[-1] != space:
                # Add space if last token isn't one
                ps.append(p)
            elif p in tokens:
                # Add next phoneme or char (if chars=True)
                ps.append(p)
            elif (p in self.punct_list) and self.punct:
                # Add punct
                ps.append(p)
            elif p != space:
                message = f"Text: [{''.join(g2p_text)}] contains unknown char/phoneme: [{p}]."
                if raw_text is not None:
                    message += f"Original text: [{raw_text}]. Symbol will be skipped."
                logging.warning(message)

        # Remove trailing spaces
        if ps:
            while ps[-1] == space:
                ps.pop()

        if self.pad_with_space:
            ps = [space] + ps + [space]

        # Token index lookups
        return [self._token2id[p] for p in ps]

    @contextmanager
    def set_phone_prob(self, prob):
        if hasattr(self.g2p, "phoneme_probability"):
            self.g2p.phoneme_probability = prob
        try:
            yield
        finally:
            if hasattr(self.g2p, "phoneme_probability"):
                self.g2p.phoneme_probability = self.phoneme_probability


class ChinesePhonemesTokenizer(BaseTokenizer):
    # fmt: off
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    ZH_PUNCT_LIST = list("，。？！；：、‘’“”（）【】「」《》") + list(PUNCT_LIST)

    def __init__(
        self,
        g2p,
        punct=True,
        non_default_punct_list=None,
        *,
        space=' ',
        silence=None,
        apostrophe=True,
        sep='|',  # To be able to distinguish between 2/3 letters codes.
        add_blank_at=None,
        pad_with_space=False,
        text_preprocessing_func=chinese_text_preprocessing,
    ):
        """Chinese phoneme-based tokenizer.
        Note: This tokenizer for now covers Chinese phonemes/tones and English letters because our dataset contains
              both Chinese and English graphemes.
        Args:
            g2p: Grapheme to phoneme module.
            punct: Whether to reserve grapheme for basic punctuation or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            space: Space token as string.
            silence: Silence token as string (will be disabled if it is None).
            apostrophe: Whether to use apostrophe or not.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Basically, it replaces all non-unicode characters with unicode ones.
             Note that lower() function shouldn't be applied here, in case the text contains phonemes (it will be handled by g2p).
        """
        tokens = []
        self.space, tokens = len(tokens), tokens + [space]  # Space

        if silence is not None:
            self.silence, tokens = len(tokens), tokens + [silence]  # Silence

        self.phoneme_list = g2p.phoneme_list
        self.tone_list = g2p.tone_list
        self.ascii_letter_list = g2p.ascii_letter_list

        tokens.extend(self.phoneme_list)
        tokens.extend(self.tone_list)
        tokens.extend(self.ascii_letter_list)

        self.text_preprocessing_func = text_preprocessing_func

        if apostrophe:
            tokens.append("'")  # Apostrophe

        if punct:
            if non_default_punct_list is not None:
                self.PUNCT_LIST = non_default_punct_list
            else:
                self.PUNCT_LIST = list(self.ZH_PUNCT_LIST)
            tokens.extend(self.PUNCT_LIST)

        super().__init__(tokens, sep=sep, add_blank_at=add_blank_at)

        self.punct = punct
        self.pad_with_space = pad_with_space
        self.g2p = g2p

    def encode(self, text: str) -> List[int]:
        """See base class for more information."""
        text = self.text_preprocessing_func(text)
        g2p_text = self.g2p(text)
        return self.encode_from_g2p(g2p_text, text)

    def encode_from_g2p(self, g2p_text: List[str], raw_text: Optional[str] = None):
        """
        Encodes text that has already been run through G2Pr.
        Called for encoding to tokens after text preprocessing and G2P.

        Args:
            g2p_text: G2P's output, could be a mixture of Chinese phonemes and English letters.
            raw_text: original raw input
        """
        ps, space, tokens = [], self.tokens[self.space], set(self.tokens)
        for p in g2p_text:  # noqa
            # Add space if last one isn't one
            if p == space and len(ps) > 0 and ps[-1] != space:
                ps.append(p)
            # Add next phoneme or tone or ascii letter or apostrophe.
            elif (p.isalnum() or p == "'" or p in self.phoneme_list + self.tone_list + self.ascii_letter_list) and p in tokens:
                ps.append(p)
            # Add punctuation
            elif (p in self.PUNCT_LIST) and self.punct:
                ps.append(p)
            # Warn about unknown char/phoneme
            elif p != space:
                message = f"Text: [{' '.join(g2p_text)}] contains unknown char/phoneme: [{p}]."
                if raw_text is not None:
                    message += f"Original text: [{raw_text}]. Symbol will be skipped."
                logging.warning(message)

        # Remove trailing spaces
        if ps:
            while ps[-1] == space:
                ps.pop()

        if self.pad_with_space:
            ps = [space] + ps + [space]

        return [self._token2id[p] for p in ps]


class JapanesePhonemeTokenizer(BaseTokenizer):

    JA_PUNCT_LIST = get_ipa_punctuation_list("ja-JP")

    def __init__(
        self,
        g2p,
        punct=True,
        non_default_punct_list=None,
        *,
        space=' ',
        silence=None,
        apostrophe=True,
        sep='|',  # To be able to distinguish between 2/3 letters codes.
        add_blank_at=None,
        pad_with_space=False,
        text_preprocessing_func=japanese_text_preprocessing,
    ):
        """Japanese phoneme-based tokenizer.
        Note: This tokenizer for now covers Japanese phonemes
        Args:
            g2p: Grapheme to phoneme module.
            punct: Whether to reserve grapheme for basic punctuation or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            space: Space token as string.
            silence: Silence token as string (will be disabled if it is None).
            apostrophe: Whether to use apostrophe or not.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Basically, it replaces all non-unicode characters with unicode ones.
             Note that lower() function shouldn't be applied here, in case the text contains phonemes (it will be handled by g2p).
        """
        tokens = []
        self.space, tokens = len(tokens), tokens + [space]  # Space

        if silence is not None:
            self.silence, tokens = len(tokens), tokens + [silence]  # Silence

        self.phoneme_list = g2p.phoneme_list
        self.ascii_letter_list = g2p.ascii_letter_list

        tokens.extend(self.phoneme_list)
        tokens.extend(self.ascii_letter_list)

        self.text_preprocessing_func = text_preprocessing_func

        if apostrophe:
            tokens.append("'")  # Apostrophe

        if punct:
            if non_default_punct_list is not None:
                self.PUNCT_LIST = non_default_punct_list
            else:
                self.PUNCT_LIST = list(self.JA_PUNCT_LIST)
            tokens.extend(self.PUNCT_LIST)

        super().__init__(tokens, sep=sep, add_blank_at=add_blank_at)

        self.punct = punct
        self.pad_with_space = pad_with_space
        self.g2p = g2p

    def encode(self, text: str) -> List[int]:
        """See base class for more information."""
        text = self.text_preprocessing_func(text)
        g2p_text = self.g2p(text)
        return self.encode_from_g2p(g2p_text, text)

    def encode_from_g2p(self, g2p_text: List[str], raw_text: Optional[str] = None):
        """
        Encodes text that has already been run through G2P.
        Called for encoding to tokens after text preprocessing and G2P.

        Args:
            g2p_text: G2P's output, could be a mixture of Chinese phonemes and English letters.
            raw_text: original raw input
        """
        ps, space, tokens = [], self.tokens[self.space], set(self.tokens)
        for p in g2p_text:  # noqa
            # Add space if last one isn't one
            if p == space and len(ps) > 0 and ps[-1] != space:
                ps.append(p)
            # Add next phoneme or tone or ascii letter or apostrophe.
            elif (p.isalnum() or p == "'" or p in self.phoneme_list + self.ascii_letter_list) and p in tokens:
                ps.append(p)
            # Add punctuation
            elif (p in self.PUNCT_LIST) and self.punct:
                ps.append(p)
            # Warn about unknown char/phoneme
            elif p != space:
                message = f"Text: [{' '.join(g2p_text)}] contains unknown char/phoneme: [{p}]."
                if raw_text is not None:
                    message += f"Original text: [{raw_text}]. Symbol will be skipped."
                logging.warning(message)

        # Remove trailing spaces
        if ps:
            while ps[-1] == space:
                ps.pop()

        if self.pad_with_space:
            ps = [space] + ps + [space]

        return [self._token2id[p] for p in ps]
