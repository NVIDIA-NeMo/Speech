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
    """
    Sentencepiecetokenizer https://github.com/google/sentencepiece.

    Args:
        model_path: path to sentence piece tokenizer model. To create the model use create_spt_model()
        special_tokens: either list of special tokens or dictionary of token name to token value
        legacy: when set to True, the previous behavior of the SentecePiece wrapper will be restored,
            including the possibility to add special tokens inside wrapper.
    """

    def __init__(
        self,
        model_path: str,
        special_tokens: Optional[Union[Dict[str, str], List[str]]] = None,
        legacy: bool = False,
        chat_template: Optional[Dict] = None,
    ):
        self.chat_template = chat_template
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"model_path: {model_path} is invalid")
        self.tokenizer = sentencepiece.SentencePieceProcessor()
        self.tokenizer.Load(model_path)

        self.original_vocab_size = self.tokenizer.get_piece_size()
        self.vocab_size = self.tokenizer.get_piece_size()
        self.legacy = legacy
        self.special_token_to_id = {}
        self.id_to_special_token = {}
        if special_tokens:
            if not self.legacy:
                raise ValueError(
                    "Special tokens must be None when legacy is set to False. Provide special tokens at train time."
                )
            self.add_special_tokens(special_tokens)
        self.space_sensitive = self.text_to_tokens('x y') != self.text_to_tokens('x') + self.text_to_tokens('y')

    def text_to_tokens(self, text):
        if self.legacy:
            tokens = []
            idx = 0
            last_idx = 0

            while 1:
                indices = {}

                for token in self.special_token_to_id:
                    try:
                        indices[token] = text[idx:].index(token)
                    except ValueError:
                        continue

                if len(indices) == 0:
                    break

                next_token = min(indices, key=indices.get)
                next_idx = idx + indices[next_token]

                tokens.extend(self.tokenizer.encode_as_pieces(text[idx:next_idx]))
                tokens.append(next_token)
                idx = next_idx + len(next_token)

            tokens.extend(self.tokenizer.encode_as_pieces(text[idx:]))
            return tokens

        return self.tokenizer.encode_as_pieces(text)

    def text_to_ids(self, text, sample_alpha=None):
        if isinstance(text, str):
            return self._text_to_ids(text, sample_alpha)
        elif isinstance(text, list):
            return self.apply_chat_template(text)
        else:
            raise ValueError(f"Expected either str or list input, but got {type(text)}")

    def _text_to_ids(self, text, sample_alpha=None):
        if self.legacy:
            ids = []
            idx = 0
            last_idx = 0

            while 1:
                indices = {}

                for token in self.special_token_to_id:
                    try:
                        indices[token] = text[idx:].index(token)
                    except ValueError:
                        continue

                if len(indices) == 0:
                    break

                next_token = min(indices, key=indices.get)
                next_idx = idx + indices[next_token]

                ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
                ids.append(self.special_token_to_id[next_token])
                idx = next_idx + len(next_token)

            ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
            return ids

        if sample_alpha is not None:
            return self.tokenizer.encode_as_ids(text, enable_sampling=True, alpha=sample_alpha, nbest_size=-1)
        else:
            return self.tokenizer.encode_as_ids(text)

    def tokens_to_text(self, tokens):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        return self.tokenizer.decode_pieces(tokens)

    def ids_to_text(self, ids):
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
        if self.legacy and token in self.special_token_to_id:
            return self.special_token_to_id[token]

        return self.tokenizer.piece_to_id(token)

    def ids_to_tokens(self, ids):
        tokens = []
        for id in ids:
            if id >= self.original_vocab_size:
                tokens.append(self.id_to_special_token[id])
            else:
                tokens.append(self.tokenizer.id_to_piece(id))
        return tokens

    def tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            tokens = [tokens]
        ids = []
        for token in tokens:
            ids.append(self.token_to_id(token))
        return ids

    def add_special_tokens(self, special_tokens):
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

    @property
    def pad_id(self):
        if self.legacy:
            pad_id = self.tokens_to_ids([self.pad_token])[0]
        else:
            pad_id = self.tokenizer.pad_id()
        return pad_id

    @property
    def bos_id(self):
        if self.legacy:
            bos_id = self.tokens_to_ids([self.bos_token])[0]
        else:
            bos_id = self.tokenizer.bos_id()
        return bos_id

    @property
    def eos_id(self):
        if self.legacy:
            eos_id = self.tokens_to_ids([self.eos_token])[0]
        else:
            eos_id = self.tokenizer.eos_id()
        return eos_id

    @property
    def sep_id(self):
        if self.legacy:
            return self.tokens_to_ids([self.sep_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def cls_id(self):
        if self.legacy:
            return self.tokens_to_ids([self.cls_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def mask_id(self):
        if self.legacy:
            return self.tokens_to_ids([self.mask_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def unk_id(self):
        return self.tokenizer.unk_id()

    @property
    def additional_special_tokens_ids(self):
        """Returns a list of the additional special tokens (excluding bos, eos, pad, unk). Used to return sentinel tokens for e.g. T5."""
        special_tokens = set(
            [self.bos_token, self.eos_token, self.pad_token, self.mask_token, self.cls_token, self.sep_token]
        )
        return [v for k, v in self.special_token_to_id.items() if k not in special_tokens]

    @property
    def vocab(self):
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
):
    """
    Creates sentence piece tokenizer model from data file.

    Args:
        data_file: data file
        vocab_size: vocabulary size
        sample_size: maximum size of sentences the trainer loads
        do_lower_case: if text should be lower cased before tokenizer model is created
        character_coverage: float value between 0 and 1 (as a percentage). For languages with a vast charset,
            can be < 1.0, but for all other languages, it should be set as 1.0
        output_dir: folder to save created tokenizer model. If not specified will store model at data_file/../spt folder
        train_extremely_large_corpus: If training on huge datasets, pass this flag to allow SentencePiece
            to build the tokenizer.
        max_sentencepiece_length: Limits the maximum length of the SentencePiece subword that can be constructed.
            By default, no limit is placed.
        bos: when True, bos token "<s>" is added to the vocabulary.
        eos: when True, eos token "</s>" is added to the vocabulary.
        pad: when True, pad token "<pad>" is added to the vocabulary.
        control_symbols: control symbols to add to tokenizer, as defined by sentencepiece.
            These tokens get removed at decode time and are not encoded from the text - can only be added to the input programatically.
        user_defined_symbols: user symbols to add to tokenizer, as defined by sentencepiece.
            These tokens remain in the decoded text and are encoded automatically when present in the input text.
        byte_fallback: If <unk>, fallback to a byte sequence of the character.
        split_digits: If true, digits are split into individual tokens.
        split_by_whitespace: Whether to respect white space while creating subwords. If False, will learn merges across whitespace.
        split_by_unicode_script: Whether to include multiple Unicode scripts. Ex. is Arabic diacritics which are considered part of the letter (عِدَّةُ)
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
