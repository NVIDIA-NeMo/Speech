# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Mapping, Optional

import datasets
import numpy as np
import torch

# hack to avoid the "not enough disk space" error in some slurm cluster
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset
from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ['GPTEmbeddingDataset', 'GPTRerankerDataset']


class GPTEmbeddingDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        max_num_samples: int = None,
        seed: int = 1234,
        index_mapping_dir: str = None,
        virtual_tokens: int = 0,
        memmap_workers: Optional[int] = None,
        truncation_method: str = 'right',
        special_tokens: Optional[Mapping[str, str]] = None,  # special tokens, a dictory of {token_type: token}
        data_type: str = 'train',  # train, query or doc
    ):
        """
        file_path: Path to a JSONL dataset with (query,pos_doc,neg_doc) triplets in jsonl format.
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        index_mapping_dir: Directory to save the index mapping to. If None, will write to the same folder as the dataset.
        truncation_method: Truncation from which position. Options: ['left', 'right']
        special_tokens: special tokens for the chat prompts, a dictionary of {token_type: token}. Default: {'system_turn_start': '<extra_id_0>', 'turn_start': '<extra_id_1>', 'label_start': '<extra_id_2>', 'end_of_turn': '\n', "end_of_name": "\n"}
        """
        # TODO: lot of copy-paste from GPTSFDDataset, should refactor both to use a common base class (@adithyare)
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.index_mapping_dir = index_mapping_dir
        self.virtual_tokens = virtual_tokens
        self.truncation_method = truncation_method
        if special_tokens is None:
            self.special_tokens = {
                "system_turn_start": "<extra_id_0>",
                "turn_start": "<extra_id_1>",
                "label_start": "<extra_id_2>",
                "end_of_turn": "\n",
                "end_of_name": "\n",
            }
        else:
            self.special_tokens = special_tokens
        self.data_type = data_type

        self.indexed_dataset = JSONLMemMapDataset(
            dataset_paths=[file_path],
            tokenizer=None,
            header_lines=0,
            index_mapping_dir=index_mapping_dir,
            workers=memmap_workers,
        )

        # Will be None after this call if `max_num_samples` is None
        self.samples_mapping = None
        self._build_samples_mapping()

    def _build_samples_mapping(self):
        if self.max_num_samples is not None:
            self.samples_mapping = get_samples_mapping(
                indexed_dataset=self.indexed_dataset,
                data_prefix=self.file_path,
                num_epochs=None,
                max_num_samples=self.max_num_samples,
                max_seq_length=self.max_seq_length - 2,
                short_seq_prob=0,
                seed=self.seed,
                name=self.file_path.split('/')[-1],
                binary_head=False,
                index_mapping_dir=self.index_mapping_dir,
            )
        else:
            self.samples_mapping = None

    def __len__(self):
        if self.max_num_samples is None:
            return len(self.indexed_dataset)
        else:
            assert self.samples_mapping is not None
            return len(self.samples_mapping)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, np.uint32):
                idx = idx.item()

        assert idx < len(self.indexed_dataset)
        # idx may < 0 because we pad_samples_to_global_batch_size, e.g. id = -1
        if idx < 0:
            idx = len(self) + idx
            auto_gen_idx = True
        else:
            auto_gen_idx = False
        try:
            example = self.indexed_dataset[idx]
            if auto_gen_idx:
                example['__AUTOGENERATED__'] = True
        except Exception as e:
            logging.error(f"Error while loading example {idx} from dataset {self.file_path}")
            raise e
        return self._process_example(example)

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        metadata = {k: v for k, v in example.items()}
        if self.data_type == 'train':
            q = self.tokenizer.text_to_ids("query: " + example['query'].strip())
            d = self.tokenizer.text_to_ids("passage: " + example['pos_doc'].strip())
            nd = self.tokenizer.text_to_ids("passage: " + example['neg_doc'].strip())
        elif self.data_type == 'query':
            q = self.tokenizer.text_to_ids("query: " + example['query'].strip())
            d, nd = None, None
            assert "query_id" in example, "query_id is required for query dataset"
            assert "doc_id" in example, "doc_id is required for query dataset"
        elif self.data_type == 'doc':
            d = self.tokenizer.text_to_ids("passage: " + example['pos_doc'].strip())
            assert "doc_id" in example, "doc_id is required for doc dataset"
            q, nd = None, None
        else:
            raise ValueError(f"Invalid data type: {self.data_type}")

        q = q if q is not None else []
        d = d if d is not None else []
        nd = nd if nd is not None else []

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens for ptuning (if used)
            q = [self.tokenizer.eos_id] * self.virtual_tokens + q  # type: ignore
            d = [self.tokenizer.eos_id] * self.virtual_tokens + d  # type: ignore
            nd = [self.tokenizer.eos_id] * self.virtual_tokens + nd  # type: ignore

        if self.add_bos:
            q = [self.tokenizer.bos_id] + q  # type: ignore
            d = [self.tokenizer.bos_id] + d  # type: ignore
            nd = [self.tokenizer.bos_id] + nd  # type: ignore

        # TODO: (@adithyare) should probably add a warning before truncation
        q = q[: self.max_seq_length - 1]
        d = d[: self.max_seq_length - 1]
        nd = nd[: self.max_seq_length - 1]

        if self.add_eos:
            q = q + [self.tokenizer.eos_id]  # type: ignore
            d = d + [self.tokenizer.eos_id]  # type: ignore
            nd = nd + [self.tokenizer.eos_id]  # type: ignore

        processed_example = {
            'query': q,
            'pos_doc': d,
            'neg_doc': nd,
            'metadata': metadata,
        }

        return processed_example

    def _maybe_cast_to_list(self, x):
        if isinstance(x, np.ndarray):
            return [item.tolist() for item in x]
        return x

    def _ceil_to_nearest(self, n, m):
        return (n + m - 1) // m * m

    def _collate_item(self, item, max_length, pad_id):
        item = self._maybe_cast_to_list(item)
        # max_length = max([len(x) for x in item]) if item else 0
        # here [0] should be tokenizer.pad_id
        item = [x + [pad_id] * (max_length - len(x)) for x in item]
        return item

    @torch.no_grad()
    def _create_attention_mask(self, max_length):
        """Create `attention_mask`.
        Args:
            input_ids: A 1D tensor that holds the indices of tokens.
        """
        # seq_length = len(input_ids)
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5
        return attention_mask

    def collate_fn(self, batch):
        input_ids = []
        metadata = []
        lengths = []
        max_length = -1
        for item in batch:
            metadata.append(item['metadata'])
            if self.data_type == 'train':
                input_ids.append(item['query'])
                lengths.append(len(item['query']))
                input_ids.append(item['pos_doc'])
                lengths.append(len(item['pos_doc']))
                input_ids.append(item['neg_doc'])
                lengths.append(len(item['neg_doc']))
                max_length = max(max_length, len(item['query']), len(item['pos_doc']), len(item['neg_doc']))
            elif self.data_type == 'query':
                input_ids.append(item['query'])
                lengths.append(len(item['query']))
                max_length = max(max_length, len(item['query']))
            elif self.data_type == 'doc':
                input_ids.append(item['pos_doc'])
                lengths.append(len(item['pos_doc']))
                max_length = max(max_length, len(item['pos_doc']))
            else:
                raise ValueError(f"Invalid data type: {self.data_type}")

        max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, 16))
        assert max_length <= self.max_seq_length

        attention_mask = [self._create_attention_mask(max_length) for _ in input_ids]
        attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in input_ids]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        )
        lengths = torch.LongTensor(lengths) - 1  # subtract 1 to account for the eos token

        processed_batch = {
            'tokens': input_ids,
            'attention_mask': attention_mask,
            'loss_mask': lengths,
            'position_ids': position_ids,
            'metadata': metadata,
        }

        return processed_batch


class GPTRerankerDataset(GPTEmbeddingDataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        max_num_samples: int = None,
        seed: int = 1234,
        index_mapping_dir: str = None,
        virtual_tokens: int = 0,
        memmap_workers: Optional[int] = None,
        truncation_method: str = 'right',
        special_tokens: Optional[Mapping[str, str]] = None,  # special tokens, a dictory of {token_type: token}
        data_type: str = 'train',  # train, query or doc
    ):
        """
        file_path: Path to a JSONL dataset with (query,pos_doc,neg_doc) triplets in jsonl format.
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        index_mapping_dir: Directory to save the index mapping to. If None, will write to the same folder as the dataset.
        truncation_method: Truncation from which position. Options: ['left', 'right']
        special_tokens: special tokens for the chat prompts, a dictionary of {token_type: token}. Default: {'system_turn_start': '<extra_id_0>', 'turn_start': '<extra_id_1>', 'label_start': '<extra_id_2>', 'end_of_turn': '\n', "end_of_name": "\n"}
        """
        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            max_num_samples=max_num_samples,
            seed=seed,
            index_mapping_dir=index_mapping_dir,
            virtual_tokens=virtual_tokens,
            memmap_workers=memmap_workers,
            truncation_method=truncation_method,
            special_tokens=special_tokens,
            data_type=data_type,
        )

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        metadata = {k: v for k, v in example.items()}
        if self.data_type == 'train':
            qd = self.tokenizer.text_to_ids(
                "query: " + example['query'].strip() + " passage: " + example['pos_doc'].strip()
            )
            qnd = self.tokenizer.text_to_ids(
                "query: " + example['query'].strip() + " passage: " + example['neg_doc'].strip()
            )
        else:
            qd = self.tokenizer.text_to_ids(
                "query: " + example['query'].strip() + " passage: " + example['pos_doc'].strip()
            )
            qnd = []

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens for ptuning (if used)
            qd = [self.tokenizer.eos_id] * self.virtual_tokens + qd  # type: ignore
            qnd = [self.tokenizer.eos_id] * self.virtual_tokens + qnd  # type: ignore

        if self.add_bos:
            qd = [self.tokenizer.bos_id] + qd  # type: ignore
            qnd = [self.tokenizer.bos_id] + qnd  # type: ignore

        # TODO: (@adithyare) should probably add a warning before truncation
        qd = qd[: self.max_seq_length - 1]
        qnd = qnd[: self.max_seq_length - 1]

        if self.add_eos:
            qd = qd + [self.tokenizer.eos_id]  # type: ignore
            qnd = qnd + [self.tokenizer.eos_id]  # type: ignore

        processed_example = {
            'query_pos_doc': qd,
            'query_neg_doc': qnd,
            'metadata': metadata,
        }

        return processed_example

    def collate_fn(self, batch):
        input_ids = []
        metadata = []
        lengths = []
        max_length = -1
        for item in batch:
            metadata.append(item['metadata'])
            if self.data_type == 'train':
                input_ids.append(item['query_pos_doc'])
                lengths.append(len(item['query_pos_doc']))
                input_ids.append(item['query_neg_doc'])
                lengths.append(len(item['query_neg_doc']))
                max_length = max(max_length, len(item['query_pos_doc']), len(item['query_neg_doc']))
            else:
                input_ids.append(item['query_pos_doc'])
                lengths.append(len(item['query_pos_doc']))
                max_length = max(max_length, len(item['query_pos_doc']))

        max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, 16))
        assert max_length <= self.max_seq_length

        attention_mask = [self._create_attention_mask(max_length) for _ in input_ids]
        attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in input_ids]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        )
        lengths = torch.LongTensor(lengths) - 1  # subtract 1 to account for the eos token

        processed_batch = {
            'tokens': input_ids,
            'attention_mask': attention_mask,
            'loss_mask': lengths,
            'position_ids': position_ids,
            'metadata': metadata,
        }

        return processed_batch
