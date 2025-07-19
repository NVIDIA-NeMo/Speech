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

import os
import re
from typing import Dict, List, Optional, Union

import numpy as np
import sentencepiece
import torch

from nemo.collections.common.parts.utils import if_exist
from nemo.collections.common.tokenizers.chat_template_mixin import ChatTemplateMixin
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

__all__ = ['SentencePieceTokenizer', 'create_spt_model']


class SentencePieceTokenizer(TokenizerSpec, ChatTemplateMixin):
    """Sentencepiecetokenizer https://github.com/google/sentencepiece.

    Args:
        model_path: path to sentence piece tokenizer model. To create the model use create_spt_model()
        special_tokens: either list of special tokens or dictionary of token name to token value
        legacy: when set to True, the previous behavior of the SentecePiece wrapper will be restored,
            including the possibility to add special tokens inside wrapper.
        ignore_extra_whitespaces: whether to ignore extra whitespaces in the input text while encoding.
            Note:
            This is done for the current models tokenizers that don't handle extra whitespaces as by default tokenizer
            learned to ignore it. To check if the tokenizer by default ignores extra whitespaces refer to
            `self.removed_extra_spaces` attribute of the tokenizer. We added a parameter to process_asr_tokenizer.py
            for upcoming models to handle it inbuilt.
    """

    def __init__(
        self,
        model_path: str,
        special_tokens: Optional[Union[Dict[str, str], List[str]]] = None,
        legacy: bool = False,
        ignore_extra_whitespaces: bool = True,
        chat_template: Optional[Dict] = None,
        trim_spm_separator_after_special_token=True,
        spm_separator='▁',
    ):
        self.chat_template = chat_template
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"model_path: {model_path} is invalid")
        self.tokenizer = sentencepiece.SentencePieceProcessor()
        self.tokenizer.Load(model_path)

        self.original_vocab_size = self.tokenizer.get_piece_size()
        self.vocab_size = self.tokenizer.get_piece_size()
        self.legacy = legacy
        self.ignore_extra_whitespaces = ignore_extra_whitespaces
        # using special symbol for extra_space token, so it is not likely to be in the vocabulary
        self.extra_space_token = '☯'
        self.special_token_to_id = {}
        self.id_to_special_token = {}
        self.trim_spm_separator_after_special_token = trim_spm_separator_after_special_token
        self.spm_separator = spm_separator
        self.spm_separator_id = self.tokenizer.piece_to_id(spm_separator)

        if special_tokens:
            if not self.legacy:
                raise ValueError(
                    "Special tokens must be None when legacy is set to False. Provide special tokens at train time."
                )
            self.add_special_tokens(special_tokens)

        self.removed_extra_spaces = self.tokenizer.encode_as_pieces('x  y') == self.tokenizer.encode_as_pieces('x y')
        self.space_sensitive = self.text_to_tokens('x y') != self.text_to_tokens('x') + self.text_to_tokens('y')

    def text_to_tokens(self, text):
        """Converts input text to a list of tokens.

        If legacy mode is enabled, handles special tokens separately.

        Args:
            text: The input string to tokenize.

        Returns:
            A list of string tokens.
        """
        if self.removed_extra_spaces and not self.ignore_extra_whitespaces:
            text = re.sub(r'(?<= )(?= )|^ | $', f' {self.extra_space_token} ', text)
        if self.legacy:
            tokens = []
            cur_idx = 0

            while 1:
                st_indices = {}

                for token in self.special_token_to_id:
                    try:
                        st_indices[token] = text[cur_idx:].index(token)
                    except ValueError:
                        continue

                if len(st_indices) == 0:
                    break

                next_special_token = min(st_indices, key=st_indices.get)
                next_start_idx = cur_idx + st_indices[next_special_token]
                # tokens between the last special token and the next special token
                text_tokens = self.tokenizer.encode_as_pieces(text[cur_idx:next_start_idx])
                # Chat-templates insert a space between a special token and first word (e.g.
                # "[INST] who") which is tokenized as <inst-id> <space-id> <who-id> instead of
                # <inst-id> <who-id>.
                if (
                    self.trim_spm_separator_after_special_token
                    and len(tokens) > 0
                    and tokens[-1] in self.special_token_to_id
                    and len(text_tokens) > 0
                    and text_tokens[0] == self.spm_separator
                ):
                    text_tokens.pop(0)
                # Add the text tokens between the last special token and this one
                tokens.extend(text_tokens)
                # add the next special token
                tokens.append(next_special_token)
                # increment
                cur_idx = next_start_idx + len(next_special_token)

            tokens.extend(self.tokenizer.encode_as_pieces(text[cur_idx:]))

        else:
            tokens = self.tokenizer.encode_as_pieces(text)

        if self.removed_extra_spaces and not self.ignore_extra_whitespaces:
            tokens = list(filter(lambda x: x != self.extra_space_token, tokens))
        return tokens

    def text_to_ids(self, text, sample_alpha=None):
        """Converts input text to a list of token IDs.

        Handles chat formatting or raw string tokenization depending on input type.

        Args:
            text: A string or list representing chat template inputs.
            sample_alpha: Optional float to enable subword sampling for data augmentation.

        Returns:
            A list of token IDs.
        """
        if isinstance(text, str):
            return self._text_to_ids(text, sample_alpha)
        elif isinstance(text, list):
            return self.apply_chat_template(text)
        else:
            raise ValueError(f"Expected either str or list input, but got {type(text)}")

    def _text_to_ids(self, text, sample_alpha=None):
        """Internal method to convert text to token IDs, handling optional sampling and special token logic.

        Args:
            text: Input string.
            sample_alpha: Optional alpha value for stochastic subword sampling.

        Returns:
            A list of token IDs.
        """
        if self.removed_extra_spaces and not self.ignore_extra_whitespaces:
            text = re.sub(r'(?<= )(?= )|^ | $', f' {self.extra_space_token} ', text).rstrip()
        if self.legacy:
            ids = []
            cur_idx = 0

            # Account for special tokens
            while 1:
                st_indices = {}

                for token in self.special_token_to_id:
                    try:
                        st_indices[token] = text[cur_idx:].index(token)
                    except ValueError:
                        continue

                if len(st_indices) == 0:
                    break

                next_special_token = min(st_indices, key=st_indices.get)
                next_start_idx = cur_idx + st_indices[next_special_token]
                # tokens between the last special token and the next special token
                text_tokens = self.tokenizer.encode(text[cur_idx:next_start_idx])
                # Chat-templates insert a space between a special token and first word (e.g.
                # "[INST] who") which is tokenized as <inst-id> <space-id> <who-id> instead of
                # <inst-id> <who-id>.
                if (
                    self.trim_spm_separator_after_special_token
                    and len(ids) > 0
                    and ids[-1] in self.id_to_special_token
                    and len(text_tokens) > 0
                    and text_tokens[0] == self.spm_separator_id
                ):
                    text_tokens.pop(0)
                # Add the text tokens between the last special token and this one
                ids.extend(text_tokens)
                # add the next special token
                ids.append(self.special_token_to_id[next_special_token])
                # increment
                cur_idx = next_start_idx + len(next_special_token)

            if self.removed_extra_spaces and not self.ignore_extra_whitespaces:
                ids.extend(self._text_to_ids_extra_space(text[cur_idx:]))
            else:
                ids.extend(self.tokenizer.encode_as_ids(text[cur_idx:]))
            return ids

        if self.removed_extra_spaces and not self.ignore_extra_whitespaces:
            return self._text_to_ids_extra_space(text, sample_alpha)

        if sample_alpha is not None:
            return self.tokenizer.encode_as_ids(text, enable_sampling=True, alpha=sample_alpha, nbest_size=-1)
        else:
            return self.tokenizer.encode_as_ids(text)

    def _text_to_ids_extra_space(self, text, sample_alpha=None):
        """Tokenizes text while preserving extra space tokens for legacy mode.

        Args:
            text: Input string.
            sample_alpha: Optional alpha value for subword sampling.

        Returns:
            A list of token IDs with preserved extra space markers.
        """
        ids = []
        encoding_kwargs = {}
        if sample_alpha is not None:
            encoding_kwargs = {'enable_sampling': True, 'alpha': sample_alpha, 'nbest_size': -1}
        for part in text.split(self.extra_space_token):
            if not part:
                continue
            part += self.extra_space_token
            part_ids = self.tokenizer.encode_as_ids(part, **encoding_kwargs)
            ids.extend(part_ids[:-1])

        return ids

    def tokens_to_text(self, tokens):
        """Converts a list of tokens back to the corresponding string.

        Args:
            tokens: A list of string tokens or a tensor/array of token IDs.

        Returns:
            The decoded string.
        """
        if isinstance(tokens, (np.ndarray, torch.Tensor)):
            tokens = tokens.tolist()

        return self.tokenizer.decode_pieces(tokens)

    def ids_to_text(self, ids):
        """Decodes a list of token IDs into a string, handling special tokens if in legacy mode.

        Args:
            ids: A list or tensor/array of token IDs.

        Returns:
            The decoded string.
        """
        if isinstance(ids, (np.ndarray, torch.Tensor)):
            ids = ids.tolist()

        if self.legacy:
            text = ""
            last_i = 0

            for i, id in enumerate(ids):
                if id in self.id_to_special_token:
                    text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                    text += self.id_to_special_token[id] + " "
                    last_i = i + 1

            text += self.tokenizer.decode_ids(ids[last_i:])
            return text.strip()

        return self.tokenizer.decode_ids(ids)

    def token_to_id(self, token):
        """Gets the ID corresponding to a token.

        Args:
            token: Token string.

        Returns:
            Token ID as an integer.
        """
        if self.legacy and token in self.special_token_to_id:
            return self.special_token_to_id[token]

        return self.tokenizer.piece_to_id(token)

    def ids_to_tokens(self, ids):
        """Converts a list of token IDs into corresponding token strings.

        Args:
            ids: A list or array/tensor of token IDs.

        Returns:
            List of string tokens.
        """
        if isinstance(ids, (np.ndarray, torch.Tensor)):
            ids = ids.tolist()
        tokens = []
        for id in ids:
            if id >= self.original_vocab_size:
                tokens.append(self.id_to_special_token[id])
            else:
                tokens.append(self.tokenizer.id_to_piece(id))
        return tokens

    def tokens_to_ids(self, tokens: Union[str, List[str]], tokens_to_skip: List[str] = []) -> Union[int, List[int]]:
        """Converts one or more tokens into their respective IDs, skipping any specified tokens.

        Args:
            tokens: A string or list of token strings.
            tokens_to_skip: List of tokens to ignore during conversion.

        Returns:
            A single ID or list of IDs.
        """
        if isinstance(tokens, str):
            tokens = [tokens]
        ids = []
        for token in tokens:
            if token not in tokens_to_skip:
                ids.append(self.token_to_id(token))
        return ids

    def add_special_tokens(self, special_tokens):
        """Adds new special tokens to the tokenizer's vocabulary (only if legacy=True).

        Args:
            special_tokens: List or dict of special tokens to add.

        Raises:
            AttributeError: If not in legacy mode.
            ValueError: If the input is not a list or dictionary.
        """
        if not self.legacy:
            raise AttributeError("Special Token addition does not work when legacy is set to False.")

        if isinstance(special_tokens, list):
            for token in special_tokens:
                if (
                    self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id()
                    and token not in self.special_token_to_id
                ):
                    self.special_token_to_id[token] = self.vocab_size
                    self.id_to_special_token[self.vocab_size] = token
                    self.vocab_size += 1
                elif self.tokenizer.piece_to_id(token) != self.tokenizer.unk_id():
                    self.special_token_to_id[token] = self.tokenizer.piece_to_id(token)
                    self.id_to_special_token[self.special_token_to_id[token]] = token

        elif isinstance(special_tokens, dict):
            for token_name, token in special_tokens.items():
                setattr(self, token_name, token)
                if (
                    self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id()
                    and token not in self.special_token_to_id
                ):
                    self.special_token_to_id[token] = self.vocab_size
                    self.id_to_special_token[self.vocab_size] = token
                    self.vocab_size += 1
                elif self.tokenizer.piece_to_id(token) != self.tokenizer.unk_id():
                    self.special_token_to_id[token] = self.tokenizer.piece_to_id(token)
                    self.id_to_special_token[self.special_token_to_id[token]] = token
        else:
            raise ValueError(f"Expected special_tokens to be a list or a dict {str(type(special_tokens))}")

    @property
    def pad_id(self):
        """Returns the ID for the padding token."""
        if self.legacy:
            pad_id = self.tokens_to_ids([self.pad_token])[0]
        else:
            pad_id = self.tokenizer.pad_id()
        return pad_id

    @property
    def bos_id(self):
        """Returns the ID for the beginning-of-sequence token."""
        if self.legacy:
            bos_id = self.tokens_to_ids([self.bos_token])[0]
        else:
            bos_id = self.tokenizer.bos_id()
        return bos_id

    @property
    def eos_id(self):
        """Returns the ID for the end-of-sequence token."""
        if self.legacy:
            eos_id = self.tokens_to_ids([self.eos_token])[0]
        else:
            eos_id = self.tokenizer.eos_id()
        return eos_id

    @property
    def sep_id(self):
        """Returns the ID for the separator token (only in legacy mode)."""
        if self.legacy:
            return self.tokens_to_ids([self.sep_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def cls_id(self):
        """Returns the ID for the classification token (only in legacy mode)."""
        if self.legacy:
            return self.tokens_to_ids([self.cls_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def mask_id(self):
        """Returns the ID for the mask token (only in legacy mode)."""
        if self.legacy:
            return self.tokens_to_ids([self.mask_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def unk_id(self):
        """Returns the ID for the unknown token."""
        return self.tokenizer.unk_id()

    @property
    def additional_special_tokens_ids(self):
        """Returns a list of the additional special tokens (excluding bos, eos, pad, unk).

        Used to return sentinel tokens for e.g. T5.
        """
        special_tokens = set(
            [self.bos_token, self.eos_token, self.pad_token, self.mask_token, self.cls_token, self.sep_token]
        )
        return [v for k, v in self.special_token_to_id.items() if k not in special_tokens]

    @property
    def vocab(self):
        """Returns the combined vocabulary list, including base and special tokens."""
        main_vocab = [self.tokenizer.id_to_piece(id) for id in range(self.tokenizer.get_piece_size())]
        special_tokens = [
            self.id_to_special_token[self.original_vocab_size + i]
            for i in range(self.vocab_size - self.original_vocab_size)
        ]
        return main_vocab + special_tokens


def create_spt_model(
    data_file: str,
    vocab_size: int,
    sample_size: int,
    do_lower_case: bool,
    tokenizer_type: str = 'unigram',
    output_dir: Optional[str] = None,
    character_coverage: float = 1.0,
    train_extremely_large_corpus: bool = False,
    max_sentencepiece_length: int = -1,
    bos: bool = False,
    eos: bool = False,
    pad: bool = False,
    control_symbols: List[str] = None,
    user_defined_symbols: List[str] = None,
    byte_fallback: bool = False,
    split_digits: bool = False,
    split_by_whitespace: bool = True,
    split_by_unicode_script: bool = True,
    remove_extra_whitespaces: bool = False,
):
    """Creates sentence piece tokenizer model from data file.

    Args:
        data_file: data file
        vocab_size: vocabulary size
        sample_size: maximum size of sentences the trainer loads
        do_lower_case: if text should be lower cased before tokenizer model is created
        character_coverage: float value between 0 and 1 (as a percentage). For languages with a vast charset,
            can be < 1.0, but for all other languages, it should be set as 1.0
        output_dir: folder to save created tokenizer model. If not specified will store at data_file/../spt folder
        train_extremely_large_corpus: If training on huge datasets, pass this flag to allow SentencePiece
            to build the tokenizer.
        max_sentencepiece_length: Limits the maximum length of the SentencePiece subword that can be constructed.
            By default, no limit is placed.
        bos: when True, bos token "<s>" is added to the vocabulary.
        eos: when True, eos token "</s>" is added to the vocabulary.
        pad: when True, pad token "<pad>" is added to the vocabulary.
        control_symbols: control symbols to add to tokenizer, as defined by sentencepiece.
            These tokens get removed at decode time and are not encoded from the text - can only be added to the input
            programatically.
        user_defined_symbols: user symbols to add to tokenizer, as defined by sentencepiece.
            These tokens remain in the decoded text and are encoded automatically when present in the input text.
        byte_fallback: If <unk>, fallback to a byte sequence of the character.
        split_digits: If true, digits are split into individual tokens.
        split_by_whitespace: Whether to respect white space while creating subwords.
            If False, will learn merges across whitespace.
        split_by_unicode_script: Whether to include multiple Unicode scripts.
            Ex. is Arabic diacritics which are considered part of the letter (عِدَّةُ).
        remove_extra_whitespaces: Whether to remove leading, trailing, and duplicate internal whitespace.
            If true, will skip double spaces during encoding.
    """

    if not data_file or not os.path.exists(data_file):
        raise ValueError(f"data_file must be valid file path, but got {data_file}")
    data_dir = os.path.dirname(data_file)
    vocab = []
    special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
    if not output_dir:
        output_dir = f'{data_dir}/spt'
    if if_exist(output_dir, ['tokenizer.model']):
        logging.info(f"tokenizer model {output_dir}/tokenizer.model already exists")
        return f'{output_dir}/tokenizer.model', f'{output_dir}/vocab.txt'
    logging.info(f'Processing {data_file} and store at {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    cmd = (
        f"--input={data_file} --model_prefix={output_dir}/tokenizer "
        f"--vocab_size={vocab_size} "
        f"--shuffle_input_sentence=true --hard_vocab_limit=false "
        f"--model_type={tokenizer_type} "
        f"--character_coverage={character_coverage}"
    )

    pad_id = 3
    if not bos:
        pad_id -= 1
        cmd += " --bos_id=-1"

    if not eos:
        pad_id -= 1
        cmd += " --eos_id=-1"

    if pad:
        cmd += f" --pad_id={pad_id}"

    if control_symbols:
        control_string = (",").join(control_symbols)
        cmd += f" --control_symbols={control_string}"
        special_tokens += control_symbols

    if user_defined_symbols:
        user_string = (",").join(user_defined_symbols)
        cmd += f" --user_defined_symbols={user_string}"
        special_tokens += user_defined_symbols

    if do_lower_case:
        cmd += " --normalization_rule_name=nmt_nfkc_cf"

    if sample_size > 0:
        cmd += f" --input_sentence_size={sample_size}"

    if train_extremely_large_corpus:
        cmd += " --train_extremely_large_corpus=true"

    if max_sentencepiece_length >= 0:
        cmd += f" --max_sentencepiece_length={max_sentencepiece_length}"

    if byte_fallback:
        cmd += " --byte_fallback=true"

    if split_digits:
        cmd += " --split_digits=true"

    if not split_by_whitespace:
        cmd += " --split_by_whitespace=false"

    if not split_by_unicode_script:
        cmd += " --split_by_unicode_script=false"

    if not remove_extra_whitespaces:
        cmd += " --remove_extra_whitespaces=false"

    sentencepiece.SentencePieceTrainer.Train(cmd)

    # Add BERT control symbols
    tokens = []

    # Encoding arg is added for compatibility with systems which enforce
    # ASCII encoding in Python. Sentencepiece always uses Unicode (UTF8).
    with open(f"{output_dir}/tokenizer.vocab", "r", encoding="utf8") as f:
        # Read tokens from each line and parse for vocab
        for line in f:
            piece = line.split("\t")[0]
            if piece in special_tokens:
                # skip special tokens
                continue
            token = piece[1:] if piece.startswith("▁") else f"##{piece}"

            if len(token) > 0:
                tokens.append(token)
            else:
                tokens.append(piece[0])

    vocab.extend(tokens)

    # Save vocabulary to output file
    vocab_file = f'{output_dir}/vocab.txt'
    with open(vocab_file, "w", encoding="utf8") as f:
        for token in vocab:
            f.write(f"{token}\n")
    return f'{output_dir}/tokenizer.model', vocab_file
