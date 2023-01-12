# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import pickle
from typing import List

import numpy

from nemo.collections.common.tokenizers.column_coder import ColumnCodes
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

__all__ = ['TabularTokenizer']

END_OF_TEXT = '<|endoftext|>'
NEW_LINE = '\n'


def find_index_of(list_input, item):
    output = -1
    try:
        output = list_input.index(item)
    except ValueError:
        pass
    return output


class TabularTokenizer(TokenizerSpec):
    def __init__(self, coder, special_tokens=[END_OF_TEXT, NEW_LINE], delimiter=','):
        if isinstance(coder, ColumnCodes):
            self.code_column: ColumnCodes = coder
        else:
            with open(coder, 'rb') as handle:
                self.code_column: ColumnCodes = pickle.load(handle)
        self.num_columns = len(self.code_column.columns)
        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.add_special_tokens(special_tokens)
        self.delimiter = delimiter
        self.eod_id = self.special_tokens[END_OF_TEXT]
        self.eos_id = self.eod_id
        self.bos_id = self.eos_id

    def __len__(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return max(self.special_tokens_decoder.keys()) + 1

    def text_to_ids(self, text):
        return self.encode(text)

    def ids_to_text(self, token_ids):
        return self.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def eor(self):
        return self.special_tokens[NEW_LINE]

    def add_special_tokens(self, special_tokens):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last
            index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        new = dict(
            (tok, self.code_column.vocab_size + i)
            for i, tok in enumerate(special_tokens)
            if tok not in self.special_tokens
        )
        self.special_tokens.update(new)
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}

    def text_to_tokens(self, text):
        """ Tokenize a string. """
        tokens = []
        rows = text.split(NEW_LINE)
        num_rows = len(rows)
        for row_id in range(num_rows):
            row = rows[row_id]
            if row == '':
                continue
            fields = row.split(self.delimiter)
            for f in fields:
                splits = f.split(END_OF_TEXT)
                if len(splits) == 1:
                    tokens.append(f.strip())
                elif len(splits) == 2:
                    if splits[0] != '':
                        tokens.append(splits[0].strip())
                    tokens.append(END_OF_TEXT)
                    if splits[1] != '':
                        tokens.append(splits[1].strip())
                else:
                    raise ValueError("delimiter error")
            if row_id != num_rows - 1:
                tokens.append(NEW_LINE)
        return tokens

    def tokens_to_ids(self, tokens: List[str]):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        cindex = 0
        if NEW_LINE in tokens:
            idd = tokens.index(NEW_LINE)
            cindex = (self.num_columns - idd) % self.num_columns
        for token in tokens:

            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                index = cindex % self.num_columns
                column = self.code_column.columns[index]
                ids.extend(self.code_column.encode(column, token))
                cindex += 1
        return ids

    def ids_to_tokens(self, ids, skip_special_tokens=False):
        """Converts a sequence of ids in Tabular tokens using the vocab."""
        tokens = []
        sizes = self.code_column.sizes
        ids_size = sum(sizes)
        cindex = 0
        eor_pos = find_index_of(ids, self.eor)
        eod_pos = find_index_of(ids, self.eod)
        if eor_pos >= 0 and eod_pos >= 0:
            idd = min(eor_pos, eod_pos)
            cindex = (ids_size - idd) % ids_size
        elif eor_pos >= 0 and eod_pos < 0:
            idd = eor_pos
            cindex = (ids_size - idd) % ids_size
        elif eod_pos >= 0 and eor_pos < 0:
            idd = eod_pos
            cindex = (ids_size - idd) % ids_size
        cum_sizes = numpy.cumsum(sizes)
        old_column_index = -1
        token_ids = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                index = cindex % ids_size
                column_index = numpy.where(index < cum_sizes)[0][0]
                column = self.code_column.columns[column_index]
                if old_column_index != column_index:
                    token_ids = [i]
                    old_column_index = column_index
                else:
                    token_ids.append(i)
                if len(token_ids) == sizes[column_index]:
                    tokens.append(self.code_column.decode(column, token_ids))
                cindex += 1
        return tokens

    def encode(self, text):
        return self.tokens_to_ids(self.text_to_tokens(text))

    def decode(self, token_ids):
        tokens = self.ids_to_tokens(token_ids, skip_special_tokens=False)
        return self.tokens_to_text(tokens)

    def tokens_to_text(self, tokens):
        all_lines = []
        line = []
        for token in tokens:
            if token == END_OF_TEXT or token == NEW_LINE:
                if len(line) != 0:
                    line_text = self.delimiter.join(line)
                    all_lines.append(line_text)
                all_lines.append(token)
                line = []
            else:
                line.append(token)
        if len(line) != 0:
            # remaining items
            line_text = self.delimiter.join(line)
            all_lines.append(line_text)
        text = "".join(all_lines)
        return text
