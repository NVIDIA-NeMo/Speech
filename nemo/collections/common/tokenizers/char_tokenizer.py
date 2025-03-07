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
import json
import os
import warnings
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Dict, List, NewType, Optional, Union

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = ['CharTokenizer']


NUMBER_OF_CHARACTERS_READ_BUFFER_SIZE = 10**7


class SpecialTokenString(Enum):
    MASK = 'mask'
    BOS = 'bos'
    EOS = 'eos'
    PAD = 'pad'
    SEP = 'sep'
    CLS = 'cls'
    UNK = 'unk'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


SpecialTokenStringType = NewType('SpecialTokenString', SpecialTokenString)


class CharTokenizer(TokenizerSpec):
    rf"""
    Each character is a token.
    Args:
        vocab_file: path to file with vocabulary for a tokenizer. The file consists of valid Python string literals 
            separated by the new line character. Such literals must contain 1 character. Examples of valid Python 
            literals: ``'a'``, ``'\n'``, ``"'"``, ``'ж'``, ``'\u8976'``. Optionally the first line in the file can be a
            JSON dictionary of special tokens. The keys of the special tokens dictionary are ``'mask_token'``,
            ``'bos_token'`` and so on. Some special tokens names can be omitted in the special tokens dictionary line.
            A file ``vocab_file`` has to be in ``'utf-8'`` encoding.
        mask_token: mask token. The following is applicable to all special tokens. Parameter ``mask_token`` is used
            for adding mask token to vocabulary or for modification of mask token present in special tokens dictionary
            in the first line of file ``vocab_file``. Parameter ``mask_token`` can be either of type ``bool`` or a 
            ``str`` of length 1. 
            
            If ``mask_token`` is ``bool`` it has to be ``False``. If ``mask_token`` is ``True`` an exception is raised.
            If ``mask_token`` is ``False`` and ``mask_token`` is present in special tokens dictionary in vocabulary
            file ``vocab_file``, then ``mask_token`` is remove from special tokens dictionary.
            
            If the parameter ``mask_token`` is a string, then such strings in the input sequence are interpreted as
            mask tokens.
        bos_token: the beginning of sequence token. See more in ``mask_token`` parameter description.
        eos_token: the end of sequence token. Usually equal to sep_token. See more in ``mask_token`` parameter 
            description.
        pad_token: token to use for padding. See more in ``mask_token`` parameter description.
        sep_token: token used for separating sequences. See more in ``mask_token`` parameter description.
        cls_token: class token. Usually equal to bos_token. See more in ``mask_token`` parameter description.
        unk_token: token to use for unknown tokens. If the parameter ``unk_token`` is set and there is a character
            in the input of ``text_to_ids`` of ``text_to_tokens`` methods which is not in the vocabulary, then
            such an unknown character is tokenized into ``unk_token``. If the parameter ``unk_token`` is ``False``,
            then unknown tokens are discarded. See more in ``mask_token`` parameter description.
        special_token_to_prepend: special token to prepend to the output of ``text_to_ids`` of ``text_to_tokens``
            methods. This option can be used if you decide to add EOS and BOS tokens to the input on the stage of
            tokenization. Possible options are: {[None] + [e.value for e in SpecialTokenString]}.
        special_token_to_append: special token to append to the output of ``text_to_ids`` of ``text_to_tokens``
            methods. See more in the description of ``special_token_to_prepend`` parameter.
        special_tokens_to_remove_while_decoding: which special tokens are remove before detokenization. If this
            parameter equals ``'all'``, then all special tokens are removed. The parameter
            ``special_tokens_to_remove_while_decoding`` can also be a list of values from this set
            {set(e.value for e in SpecialTokenString)}.
    """

    def __init__(
        self,
        vocab_file: str,
        mask_token: Optional[Union[str, bool]] = None,
        bos_token: Optional[Union[str, bool]] = None,
        eos_token: Optional[Union[str, bool]] = None,
        pad_token: Optional[Union[str, bool]] = None,
        sep_token: Optional[Union[str, bool]] = None,
        cls_token: Optional[Union[str, bool]] = None,
        unk_token: Optional[Union[str, bool]] = None,
        special_token_to_prepend: Optional[SpecialTokenStringType] = None,
        special_token_to_append: Optional[SpecialTokenStringType] = None,
        special_tokens_to_remove_while_decoding: Union[List[SpecialTokenStringType], str] = 'all',
    ):
        vocab_file = Path(vocab_file).expanduser()
        with vocab_file.open(encoding='utf-8') as f:
            first_line = f.readline()
            if first_line[0] == '{':
                special_tokens_dict = json.loads(first_line)
                self.check_special_tokens_dict_from_file(special_tokens_dict, vocab_file)
                vocab_list = f.readlines()
            else:
                special_tokens_dict = {}
                vocab_list = [first_line] + f.readlines()
        special_tokens_dict = self.update_special_tokens_dict(
            special_tokens_dict, mask_token, bos_token, eos_token, pad_token, sep_token, cls_token, unk_token
        )
        for e in SpecialTokenString:
            name = e.value + '_token'
            setattr(self, name, special_tokens_dict[name] if name in special_tokens_dict else None)
        for k, v in special_tokens_dict.items():
            setattr(self, k, v)
        for value, name in [
            (special_token_to_prepend, 'special_token_to_prepend'),
            (special_token_to_append, 'special_token_to_append'),
        ]:
            self.check_special_token_name(name, value, special_tokens_dict)
            setattr(self, name, value + '_token' if isinstance(value, str) else value)
        self.vocab = {}
        count = 0
        for v in special_tokens_dict.values():
            self.vocab[v] = count
            count += 1
        for i, token in enumerate(vocab_list):

            if token[1:-1].startswith("\\"):
                token = token[1:-1].encode('ascii').decode('unicode_escape')[0]
            else:
                token = token[1:-1][0]

            self.check_token_from_file(token, vocab_file, i)
            if token not in self.vocab:
                self.vocab[token] = count
                count += 1
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.check_special_tokens_to_remove_while_decoding(
            special_tokens_to_remove_while_decoding, special_tokens_dict
        )
        self.special_token_ids_to_remove_while_decoding = (
            self.tokens_to_ids([v for v in special_tokens_dict.values()])
            if special_tokens_to_remove_while_decoding == 'all'
            else [getattr(self, e + '_id') for e in special_tokens_to_remove_while_decoding]
        )

    @classmethod
    def check_special_tokens_dict_from_file(cls, special_tokens_dict, vocab_file):
        for k, v in special_tokens_dict.items():
            if k[-6:] != '_token' or not SpecialTokenString.has_value(k[:-6]):
                raise ValueError(
                    f"Unsupported key {repr(k)} in special tokens dictionary in vocabulary file {vocab_file} "
                    f"(first line). Supported keys are {[e.value + '_token' for e in SpecialTokenString]}."
                )
            if not isinstance(v, str):
                raise ValueError(
                    f"Values of special tokens dictionary in vocabulary file {vocab_file} (first line) has to belong "
                    f"to type `str`, whereas type of item '{k}' value {repr(v)} is `{type(v)}`."
                )
            elif len(v) == 0:
                raise ValueError(
                    f"Values of special tokens dictionary in vocabulary file {vocab_file} (first line) has to not "
                    f"empty strings, whereas value of item '{k}' is an empty string."
                )
        cls.check_special_tokens_dict_for_duplicate_values(
            special_tokens_dict, f"Loaded from vocabulary file {vocab_file}"
        )

    @staticmethod
    def check_special_tokens_dict_for_duplicate_values(special_tokens_dict, err_msg_prefix):
        if len(special_tokens_dict) != len(set(special_tokens_dict.values())):
            tokens_with_equal_values = []
            duplicate_values = []
            for k, v in list(reversed(list(special_tokens_dict.items())))[:-1]:
                tokens = [k]
                for kk, vv in special_tokens_dict.items():
                    if kk == k:
                        break
                    if v == vv:
                        tokens.append(kk)
                if len(tokens) > 1:
                    duplicate_values.append(v)
                    tokens_with_equal_values.append(tokens)
            if duplicate_values:
                dup_values_msg = '. '.join(
                    [f"Tokens {t} have value '{v}'" for t, v in zip(tokens_with_equal_values, duplicate_values)]
                )
                raise ValueError(
                    err_msg_prefix + f" special tokens dictionary has duplicate values. " + dup_values_msg
                )

    @classmethod
    def update_special_tokens_dict(
        cls,
        init_special_tokens_dict: Dict[str, str],
        mask_token: Optional[Union[str, bool]] = None,
        bos_token: Optional[Union[str, bool]] = None,
        eos_token: Optional[Union[str, bool]] = None,
        pad_token: Optional[Union[str, bool]] = None,
        sep_token: Optional[Union[str, bool]] = None,
        cls_token: Optional[Union[str, bool]] = None,
        unk_token: Optional[Union[str, bool]] = None,
    ):
        special_tokens_dict = init_special_tokens_dict.copy()
        for value, name in zip(
            [pad_token, unk_token, bos_token, eos_token, sep_token, mask_token, cls_token],
            ['pad_token', 'unk_token', 'bos_token', 'eos_token', 'sep_token', 'mask_token', 'cls_token'],
        ):
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        raise ValueError(
                            f"If `CharTokenizer` constructor parameter `{name}` is `bool` it has to be `False`"
                        )
                    else:
                        if name in special_tokens_dict:
                            del special_tokens_dict[name]
                        else:
                            warnings.warn(
                                f"Cannot remove special token `{name}` since it is not in special tokens dictionary "
                                f"{special_tokens_dict}."
                            )
                elif not isinstance(value, str):
                    raise ValueError(
                        f"`CharTokenizer` constructor parameter `{name}` has to be either `False` or belong to type "
                        f"`str`, whereas type of `{name}` is `{type(value)}`."
                    )
                else:
                    special_tokens_dict[name] = value
        cls.check_special_tokens_dict_for_duplicate_values(
            special_tokens_dict,
            "After updating special tokens dictionary with tokens passed in `CharTokenizer` constructor parameters",
        )
        return special_tokens_dict

    @staticmethod
    def check_token_from_file(token, vocab_file, line_i):
        if not isinstance(token, str) or isinstance(token, str) and len(token) != 1:
            raise ValueError(
                f"Each line in vocabulary have to be a Python string literal containing 1 character. "
                f"Encountered {repr(token)} on line {line_i} in file {vocab_file}."
            )

    @staticmethod
    def check_special_token_name(parameter_name, value, special_tokens_dict):
        if value is not None:
            if not SpecialTokenString.has_value(value):
                raise ValueError(
                    f"Value {repr(value)} of parameter `{parameter_name}` is wrong. Supported values are "
                    f"{[e.value for e in SpecialTokenString]}."
                )
            elif value + '_token' not in special_tokens_dict:
                raise ValueError(
                    f"You should provide `{value + '_token'}` parameter to `CharTokenizer` constructor if "
                    f"you wish to pass token {repr(value)} in parameter `{parameter_name}`."
                )

    @staticmethod
    def check_special_tokens_to_remove_while_decoding(special_tokens_to_remove_while_decoding, special_tokens_dict):
        if isinstance(special_tokens_to_remove_while_decoding, list):
            for i, value in enumerate(special_tokens_to_remove_while_decoding):
                if not SpecialTokenString.has_value(value):
                    raise ValueError(
                        f'Wrong element with value {repr(value)} in position {i} of parameter '
                        f'`special_tokens_to_remove_while_decoding` of `CharTokenizer` constructor. Supported values '
                        f'are {[e.value for e in SpecialTokenString]}.'
                    )
                elif value + '_token' not in special_tokens_dict:
                    raise ValueError(
                        f"You should provide `{value + '_token'}` parameter to `CharTokenizer` constructor if "
                        f"you wish to pass token {repr(value)} in parameter `special_tokens_to_remove_while_decoding`. "
                        f"`{value + '_token'}` was detected in position {i} in "
                        f"`special_tokens_to_remove_while_decoding`."
                    )
        elif (
            isinstance(special_tokens_to_remove_while_decoding, str)
            and special_tokens_to_remove_while_decoding != 'all'
            or not isinstance(special_tokens_to_remove_while_decoding, str)
        ):
            raise ValueError(
                f"Parameter `special_tokens_to_remove_while_decoding` of `CharTokenizer` constructor has to be "
                f"equal to a string 'all' or be a list of values from set {set(e.value for e in SpecialTokenString)} "
                f"whereas `special_tokens_to_remove_while_decoding={repr(special_tokens_to_remove_while_decoding)}`"
            )

    def text_to_tokens(self, text: str) -> List[str]:
        token_candidates = [char for char in text]
        tokens = []
        if self.special_token_to_prepend is not None:
            tokens.append(getattr(self, self.special_token_to_prepend))
        for i, token in enumerate(token_candidates):
            if token in self.vocab:
                tokens.append(token)
            elif self.unk_token is not None:
                tokens.append(self.unk_token)
            else:
                warnings.warn(
                    f"Character {repr(token)} in position {i} is not present in vocabulary and no `<UNK>` token was "
                    f"set. Character {repr(token)} is discarded."
                )
        if self.special_token_to_append is not None:
            tokens.append(getattr(self, self.special_token_to_append))
        return tokens

    def tokens_to_text(self, tokens: List[str]) -> str:
        return self.ids_to_text(self.tokens_to_ids(tokens))

    def text_to_ids(self, text: str) -> List[int]:
        ids = [self.vocab[token] for token in self.text_to_tokens(text)]
        return ids

    def ids_to_text(self, ids: List[int]) -> str:
        ids_ = [id_ for id_ in ids if id_ not in self.special_token_ids_to_remove_while_decoding]
        return "".join(self.ids_to_tokens(ids_))

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab[token] for token in tokens]

    def token_to_id(self, token: str) -> int:
        return self.vocab[token]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.inv_vocab[id] for id in ids]

    @staticmethod
    def check_special_token_id_getting(special_token, id_name):
        if special_token is None:
            token_param = id_name[:-3] + '_token'
            raise ValueError(
                f"Cannot return `{id_name}` since `{token_param}` is not set. To obtain `{id_name}` you need to pass "
                f"parameter `{token_param}` to `CharTokenizer` constructor."
            )

    @property
    def pad_id(self):
        self.check_special_token_id_getting(self.pad_token, 'pad_id')
        return self.vocab[self.pad_token]

    @property
    def bos_id(self):
        self.check_special_token_id_getting(self.bos_token, 'bos_id')
        return self.vocab[self.bos_token]

    @property
    def eos_id(self):
        self.check_special_token_id_getting(self.eos_token, 'eos_id')
        return self.vocab[self.eos_token]

    @property
    def unk_id(self):
        self.check_special_token_id_getting(self.unk_token, 'unk_id')
        return self.vocab[self.unk_token]

    @property
    def mask_id(self):
        self.check_special_token_id_getting(self.mask_token, 'mask_id')
        return self.vocab[self.mask_token]

    @property
    def sep_id(self):
        self.check_special_token_id_getting(self.sep_token, 'sep_id')
        return self.vocab[self.sep_token]

    @property
    def cls_id(self):
        self.check_special_token_id_getting(self.cls_token, 'cls_id')
        return self.vocab[self.cls_token]

    @staticmethod
    def create_special_tokens_dict(
        mask_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        unk_token: Optional[str] = None,
    ):
        special_tokens_dict = {}
        for value, name in zip(
            [pad_token, unk_token, bos_token, eos_token, sep_token, mask_token, cls_token],
            ['pad_token', 'unk_token', 'bos_token', 'eos_token', 'sep_token', 'mask_token', 'cls_token'],
        ):
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(
                        f"The type of parameter `{name}` has to be `None` or `str`, found `{type(value)}`"
                    )
                elif len(value) == 0:
                    raise ValueError(f"If the parameter `{name}` is `str`, then its length has to be nonzero.")
                elif value in special_tokens_dict.values():
                    other_name = None
                    for k, v in special_tokens_dict.items():
                        if v == value:
                            other_name = k
                    raise ValueError(
                        f"The value {repr(value)} of special token `{name}` is the same as the value of special token "
                        f"`{other_name}`."
                    )
                special_tokens_dict[name] = value
        return special_tokens_dict

    @staticmethod
    def check_characters_to_exclude_from_vocabulary(characters_to_exclude_from_vocabulary):
        for i, char in enumerate(characters_to_exclude_from_vocabulary):
            if not isinstance(char, str):
                raise ValueError(
                    f"Character to exclude from vocabulary has to `str`, whereas an element in position {i} is of "
                    f"type `{type(char)}`."
                )
            elif len(char) != 1:
                raise ValueError(
                    f"A length of an element of `characters_to_exclude_from_vocabulary` parameter has to be 1. "
                    f"The length of an element in position {i} is {len(char)}."
                )

    @staticmethod
    def check_text_and_text_file_name(text, text_file_name):
        if text is None and text_file_name is None:
            raise ValueError(
                f'Exactly one of parameters `text` and `text_file_name` should be provided whereas both parameters '
                f'are `None`.'
            )
        if text is not None and text_file_name is not None:
            raise ValueError(
                f"Exactly one of parameters `text` and `text_file_name` has to be provided, whereas both parameters "
                f"are not `None`."
            )
        if text is not None:
            if not isinstance(text, str):
                raise ValueError(
                    f"Parameter `text` has to be of type `str`, whereas it belongs to type `{type(text)}`."
                )

    @classmethod
    def build_vocab(
        cls,
        save_path: Union[str, bytes, os.PathLike],
        text: Optional[str] = None,
        text_file_name: Optional[Union[str, bytes, os.PathLike]] = None,
        characters_to_exclude: Optional[List[str]] = None,
        vocab_size: int = None,
        mask_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        unk_token: Optional[str] = None,
    ):
        """
        Creates character vocabulary and saves it to file ``save_path``. You should provide one of parameters ``text``
        and ``text_file_name``. The format of created character vocabulary file is following:
        ```
        {['mask_token': "ANY NON EMPTY STRING", ]['bos_token': "ANY NON EMPTY STRING", ] and so on}
        ' '
        'e'
        ...
        ```
        The first line is a JSON which contains special tokens. This special token are set using parameters
        ``mas_token``, ``bos_token``, ``eos_token``, ``pad_token``, ``sep_token``, ``cls_token``, ``unk_token``.
        Other lines in created vocabulary file are Python string literals containing one character each.

        Args:
            save_path: path to the output text file. If ``save_path`` parent directory does not exist it will be created
            text: string which characters are used for vocabulary creation.
            text_file_name: path to a file which characters are used for vocabulary creation. Use this parameter if
                the text in file is too large to be loaded in memory.
            characters_to_exclude: a list of characters which will not be added to vocabulary.
            vocab_size: vocabulary size. If this parameter is set only most frequent ``vocab_size`` characters are added
                to vocabulary.
            mask_token: mask token
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token. Usually equal to sep_token.
            pad_token: token to use for padding.
            sep_token: token used for separating sequences.
            cls_token: class token. Usually equal to bos_token.
            unk_token: token to use for unknown tokens. If the parameter ``unk_token`` is set and there is a character
                in the input of ``text_to_ids`` of ``text_to_tokens`` methods which is not in the vocabulary, then
                such an unknown character is tokenized into ``unk_token``. If the parameter ``unk_token`` is ``False``,
                then unknown tokens are discarded.
        """
        special_tokens_dict = cls.create_special_tokens_dict(
            mask_token, bos_token, eos_token, pad_token, sep_token, cls_token, unk_token
        )
        if characters_to_exclude is None:
            characters_to_exclude = []
        else:
            cls.check_characters_to_exclude_from_vocabulary(characters_to_exclude)
        cls.check_text_and_text_file_name(text, text_file_name)
        if text is not None:
            counter = Counter(text)
        else:
            assert text_file_name is not None
            text_file_name = Path(text_file_name).expanduser()
            counter = Counter()
            with text_file_name.open(encoding='utf-8') as f:
                while True:
                    segment = f.read(NUMBER_OF_CHARACTERS_READ_BUFFER_SIZE)
                    if not segment:
                        break
                    counter.update(segment)
        for char in characters_to_exclude:
            if char in counter:
                del counter[char]
        save_path = Path(save_path).expanduser()
        save_path.parent.mkdir(exist_ok=True, parents=True)
        with save_path.open('w', encoding='utf-8') as f:
            f.write(json.dumps(special_tokens_dict) + '\n')
            if vocab_size is None:
                for c, _ in sorted(counter.items(), key=lambda x: -x[1]):
                    f.write(repr(c) + '\n')
            else:
                vocab_size -= len(special_tokens_dict)
                for i, (c, _) in enumerate(sorted(counter.items(), key=lambda x: -x[1])):
                    if i < vocab_size:
                        f.write(repr(c) + '\n')
                    else:
                        break
