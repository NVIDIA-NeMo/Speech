# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
A collection of simple character based parsers. These parser handle cleaning and tokenization by default.
We currently support English.
"""

import string
from typing import List, Optional

from nemo.collections.common.parts.preprocessing import cleaners


class CharParser:
    """Functor for parsing raw strings into list of int tokens.

    Examples:
        >>> parser = CharParser(['a', 'b', 'c'])
        >>> parser('abc')
        [0, 1, 2]
    """

    def __init__(
        self,
        labels: List[str],
        *,
        unk_id: int = -1,
        blank_id: int = -1,
        do_normalize: bool = True,
        do_lowercase: bool = True,
        do_tokenize: bool = True,
    ):
        """Creates simple mapping char parser.

        Args:
            labels: List of labels to allocate indexes for. Essentially,
                this is a id to str mapping.
            unk_id: Index to choose for OOV words (default: -1).
            blank_id: Index to filter out from final list of tokens
                (default: -1).
            do_normalize: True if apply normalization step before tokenizing
                (default: True).
            do_lowercase: True if apply lowercasing at normalizing step
                (default: True).
        """

        self._labels = labels
        self._unk_id = unk_id
        self._blank_id = blank_id
        self._do_normalize = do_normalize
        self._do_lowercase = do_lowercase
        self._do_tokenize = do_tokenize

        self._labels_map = {label: index for index, label in enumerate(labels)}
        self._special_labels = set([label for label in labels if len(label) > 1])

    def __call__(self, text: str) -> Optional[List[int]]:
        if self._do_normalize:
            text = self._normalize(text)
            if text is None:
                return None

        if not self._do_tokenize:
            return text

        text_tokens = self._tokenize(text)
        return text_tokens

    def _normalize(self, text: str) -> Optional[str]:
        text = text.strip()

        if self._do_lowercase:
            text = text.lower()

        return text

    def _tokenize(self, text: str) -> List[int]:
        tokens = []
        # Split by word for find special labels.
        for word_id, word in enumerate(text.split(' ')):
            if word_id != 0:  # Not first word - so we insert space before.
                tokens.append(self._labels_map.get(' ', self._unk_id))

            if word in self._special_labels:
                tokens.append(self._labels_map[word])
                continue

            for char in word:
                tokens.append(self._labels_map.get(char, self._unk_id))

        # If unk_id == blank_id, OOV tokens are removed.
        tokens = [token for token in tokens if token != self._blank_id]

        return tokens

    def decode(self, str_input):
        r_map = {}
        for k, v in self._labels_map.items():
            r_map[v] = k
        r_map[len(self._labels_map)] = "<BOS>"
        r_map[len(self._labels_map) + 1] = "<EOS>"
        r_map[len(self._labels_map) + 2] = "<P>"

        out = []
        for i in str_input:
            # Skip OOV
            if i not in r_map:
                continue
            out.append(r_map[i.item()])

        return "".join(out)


class ENCharParser(CharParser):
    """Incorporates english-specific parsing logic."""

    PUNCTUATION_TO_REPLACE = {'+': 'plus', '&': 'and', '%': 'percent'}

    def __init__(self, abbreviation_version=None, make_table=True, *args, **kwargs):
        """Creates english-specific mapping char parser.

        This class overrides normalizing implementation.

        Args:
            *args: Positional args to pass to `CharParser` constructor.
            **kwargs: Key-value args to pass to `CharParser` constructor.
        """

        super().__init__(*args, **kwargs)

        self._table = None
        if make_table:
            self._table = self.__make_trans_table()
        self.abbreviation_version = abbreviation_version

    def __make_trans_table(self):
        punctuation = string.punctuation

        for char in self.PUNCTUATION_TO_REPLACE:
            punctuation = punctuation.replace(char, '')

        for label in self._labels:
            punctuation = punctuation.replace(label, '')

        table = str.maketrans(punctuation, ' ' * len(punctuation))

        return table

    def _normalize(self, text: str) -> Optional[str]:
        # noinspection PyBroadException
        try:
            text = cleaners.clean_text(
                string=text,
                table=self._table,
                punctuation_to_replace=self.PUNCTUATION_TO_REPLACE,
                abbreviation_version=self.abbreviation_version,
            )
        except Exception:
            return None

        return text


class RUCharParser(CharParser):
    """Incorporates russian-specific parsing logic."""

    PUNCTUATION_TO_REPLACE = {'+': 'плюс', 'ё': 'е'}

    def __init__(self, *args, **kwargs):
        """Creates cyrillic-specific mapping char parser.
        This class overrides normalizing implementation.
        Args:
            *args: Positional args to pass to `CharParser` constructor.
            **kwargs: Key-value args to pass to `CharParser` constructor.
        """

        super().__init__(*args, **kwargs)

        self._table = self.__make_trans_table()

    def __make_trans_table(self):
        punctuation = string.punctuation

        for char in self.PUNCTUATION_TO_REPLACE:
            punctuation = punctuation.replace(char, '')

        for label in self._labels:
            punctuation = punctuation.replace(label, '')

        table = str.maketrans(punctuation, ' ' * len(punctuation))

        return table

    def _normalize(self, text: str) -> Optional[str]:
        # noinspection PyBroadException
        try:
            text = cleaners.clean_text(
                string=text, table=self._table, punctuation_to_replace=self.PUNCTUATION_TO_REPLACE,
            )
        except Exception:
            return None

        return text


NAME_TO_PARSER = {'base': CharParser, 'en': ENCharParser, 'ru': RUCharParser}


def make_parser(labels: Optional[List[str]] = None, name: str = 'base', **kwargs,) -> CharParser:
    """Creates parser from labels, set of arguments and concise parser name.

    Args:
        labels: List of labels to allocate indexes for. If set to
            None then labels would be ascii table list. Essentially, this is an
            id to str mapping (default: None).
        name: Concise name of parser to create (default: 'base').
            (default: -1).
        **kwargs: Other set of kwargs to pass to parser constructor.

    Returns:
        Instance of `CharParser`.

    Raises:
        ValueError: For invalid parser name.

    Examples:
        >>> type(make_parser(['a', 'b', 'c'], 'en'))
        ENCharParser
    """

    if name not in NAME_TO_PARSER:
        raise ValueError('Invalid parser name.')

    if labels is None:
        labels = list(string.printable)

    parser_type = NAME_TO_PARSER[name]
    parser = parser_type(labels=labels, **kwargs)

    return parser
