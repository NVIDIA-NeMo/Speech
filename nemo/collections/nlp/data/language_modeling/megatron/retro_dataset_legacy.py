# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""RETRO Style dataset."""

import os
from typing import List

import numpy as np
import torch

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import (
    _build_index_mappings,
    get_indexed_dataset_,
)
from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import (
    KNNIndex,
    MMapRetrievalIndexedDataset,
)
from nemo.core import Dataset
from nemo.utils import logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

__all__ = [
    "RETRODataset",
    "build_train_valid_test_datasets",
    "MockRETRODataset",
    "build_mock_train_valid_test_datasets",
]


class RETRODataset(Dataset):
    """
    Dataset for RETRO model.

    It constructs single data record from the training/retrieval indexed retrieval dataset and knn index file.
    The KNN index file maps data chunk id to K-nearest neighbors in the the retrieval dataset chunk ids.
    First, it loads a long sequence (2048) from training dataset. Then for each chunk in the sequence, it finds the kNN 
    chunks from the retrieval dataset using the KNN index. Lastly, compute the masks based on pad id.
    """

    def __init__(
        self,
        cfg,
        trainer,
        tokenizer,
        name: str,
        data_prefix: str,
        documents,  # document ids in the indexed_dataset used for this dataset
        indexed_dataset: MMapRetrievalIndexedDataset,
        num_samples: int,  # number of data samples,  max_steps * global_batch_size
        seq_length: int,  # input seq length
        seed: int,
        knn_index: KNNIndex,
        retrieval_index: MMapRetrievalIndexedDataset,
    ):
        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        super().__init__()
        self.name = name
        self.indexed_dataset: MMapRetrievalIndexedDataset = indexed_dataset
        self.knn_index: KNNIndex = knn_index
        self.retrieval_index: MMapRetrievalIndexedDataset = retrieval_index
        self.chunk_size = self.indexed_dataset.chunk_size

        # make sure seq_length is a multiple of chunk_size
        assert seq_length % self.chunk_size == 0
        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

        assert self.retrieval_index._index.retrieval_db
        self._validate_pad_id()

        # save index mappings to a configurable dir
        self.index_mapping_dir = cfg.data.get('index_mapping_dir', None)
        self.neighbors = cfg.data.get('neighbors', self.knn_index.K)
        # the number of neighbors cannot exceed the max number of neighbors in the index
        assert self.neighbors <= self.knn_index.K
        # create index_mapping_dir on rank 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.index_mapping_dir is not None and not os.path.isdir(self.index_mapping_dir):
                    os.makedirs(self.index_mapping_dir)
            torch.distributed.barrier()

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            self.name,
            data_prefix,
            documents,
            self.indexed_dataset.sizes,
            num_samples,
            seq_length,
            seed,
            index_mapping_dir=self.index_mapping_dir,
        )
        if len(self.doc_idx) > np.iinfo('int32').max:
            raise "number of epochs exceeds the maximum number for int32 used by sample_idx"
        self.padding_context = np.ones(2 * self.chunk_size, dtype=self.retrieval_index._index.dtype) * self.pad_id

    def _validate_pad_id(self):
        # validate the pad_id matches the dataset pad_id
        ptr, size = self.retrieval_index._index[0]
        ptr += size * np.dtype(self.retrieval_index._index.dtype).itemsize
        # padded chunk_size of pad_ids at the end of the doc
        retrieval_paddings = np.frombuffer(
            self.retrieval_index._bin_buffer,
            dtype=self.retrieval_index._index.dtype,
            count=self.chunk_size,
            offset=ptr,
        )
        assert (retrieval_paddings == self.pad_id).all()

        ptr, size = self.indexed_dataset._index[0]
        ptr += (size - 1) * np.dtype(self.indexed_dataset._index.dtype).itemsize
        data_paddings = np.frombuffer(
            self.indexed_dataset._bin_buffer, dtype=self.indexed_dataset._index.dtype, count=1, offset=ptr
        )
        # the last element is either a padding or an eos
        assert (data_paddings == self.pad_id).all() or (data_paddings == self.eos_id).all()

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def _get_chunks(self, chunk_id: int, num_chunks: int, chunks: List):
        """
        starting from chunk_id, loop for num_chunks, get the 
        KNN chunk ids from retrieval dataset, and get the chunk token ids,
        put them into the chunks list
        """
        for i in range(chunk_id, chunk_id + num_chunks):
            knn = self.knn_index.get_KNN_chunk_ids(i)
            for rid in knn[: self.neighbors]:
                if rid < 0:
                    # no neighbor, just pad it
                    one_chunk = self.padding_context
                else:
                    one_chunk = self.retrieval_index.get_chunk(rid)
                chunks.append(one_chunk)

    def _get_text(self, idx: int) -> np.ndarray:
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(
                self.doc_idx[doc_index_f], offset=offset_f, length=offset_l - offset_f + 1
            )
            chunk_id = self.indexed_dataset.get_chunk_id(self.doc_idx[doc_index_f], offset_f)
            num_chunks = (offset_l - offset_f) // self.chunk_size
            chunks = []
            self._get_chunks(chunk_id, num_chunks, chunks)
            chunks = np.stack(chunks, axis=0).reshape(num_chunks, self.neighbors, -1).astype(np.int64)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)]
            num_chunks = (self.indexed_dataset._index.sizes[self.doc_idx[doc_index_f]] - offset_f) // self.chunk_size
            total_chunks = num_chunks
            chunks = []
            chunk_id = self.indexed_dataset.get_chunk_id(self.doc_idx[doc_index_f], offset_f)
            self._get_chunks(chunk_id, num_chunks, chunks)
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
                chunk_id = self.indexed_dataset.get_chunk_id(self.doc_idx[i], 0)
                num_chunks = self.indexed_dataset._index.sizes[self.doc_idx[i]] // self.chunk_size
                total_chunks += num_chunks
                self._get_chunks(chunk_id, num_chunks, chunks)
                # And finally add the relevant portion of last document.
            chunk_id = self.indexed_dataset.get_chunk_id(self.doc_idx[doc_index_l], 0)
            num_chunks = (offset_l) // self.chunk_size
            total_chunks += num_chunks
            self._get_chunks(chunk_id, num_chunks, chunks)
            sample_list.append(self.indexed_dataset.get(self.doc_idx[doc_index_l], length=offset_l + 1))
            sample = np.concatenate(sample_list)
            chunks = np.stack(chunks, axis=0).reshape(total_chunks, self.neighbors, -1).astype(np.int64)
        return sample.astype(np.int64), chunks

    def __getitem__(self, idx):
        text, retrieved = self._get_text(idx)
        text = torch.from_numpy(text)
        retrieved = torch.from_numpy(retrieved)
        tokens = text[:-1].contiguous()
        labels = text[1:].contiguous()
        hidden_mask = tokens != self.pad_id
        context_mask = retrieved != self.pad_id
        return {
            'tokens': tokens,
            'labels': labels,
            'tokens_mask': hidden_mask,
            'loss_mask': hidden_mask,
            'retrieved_emb_mask': context_mask,
            'retrieved_ids': retrieved,
        }


def build_train_valid_test_datasets(
    cfg,
    trainer,
    data_prefix: List[str],
    data_impl: str,
    splits_string: str,
    train_valid_test_num_samples,
    seq_length: int,
    seed: int,
    skip_warmup: bool,
    tokenizer,
    retrieval_prefix: str,
    knn_map_path: List[str],
):
    """Build train, valid, and test RETRO datasets.
       There is one to one mapping between data_prefix and knn_map_path.
       Currently only supports one retrieval dataset.
    """
    # make sure there is one to one mapping  between data_prefix and knn_map_path
    assert len(data_prefix) == len(knn_map_path)

    # Single dataset.
    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(
            cfg,
            trainer,
            data_prefix[0],
            data_impl,
            splits_string,
            train_valid_test_num_samples,
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            retrieval_prefix,
            knn_map_path[0],
        )

    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples)
    prefixes, weights, datasets_train_valid_test_num_samples = output
    train_n, valid_n, test_n = map(sum, zip(*datasets_train_valid_test_num_samples))

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            cfg,
            trainer,
            prefixes[i],
            data_impl,
            splits_string,
            datasets_train_valid_test_num_samples[i],
            seq_length,
            seed,
            skip_warmup,
            tokenizer,
            retrieval_prefix,
            knn_map_path[i],
        )
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

    # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights, train_n)
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights, valid_n)
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights, test_n)

    return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_train_valid_test_datasets(
    cfg,
    trainer,
    data_prefix: str,
    data_impl: str,
    splits_string: str,
    train_valid_test_num_samples,
    seq_length: int,
    seed: int,
    skip_warmup: bool,
    tokenizer,
    retrieval_prefix: str,
    knn_map_path: str,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset: MMapRetrievalIndexedDataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)
    knn_index: KNNIndex = KNNIndex(knn_map_path, skip_warmup)
    retrieval_index: MMapRetrievalIndexedDataset = get_indexed_dataset_(retrieval_prefix, data_impl, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    logging.info(' > dataset split:')

    def print_split_stats(name, index):
        logging.info('    {}:'.format(name))
        logging.info(
            '     document indices in [{}, {}) total of {} '
            'documents'.format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
        )

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32)
            dataset = RETRODataset(
                cfg,
                trainer,
                tokenizer,
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
                knn_index,
                retrieval_index,
            )
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


class MockRETRODataset(torch.utils.data.Dataset):
    def __init__(self, cfg, trainer, tokenizer, name, size):
        super().__init__()
        self.name = name
        self.tokenizer = tokenizer
        self._cfg = cfg
        self.size = size
        seed_val = parallel_state.get_data_parallel_rank() * 131 + 97
        torch.manual_seed(seed_val)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        vocab_size = self.tokenizer.vocab_size

        neighbors = self._cfg.data.neighbors
        input_length = self._cfg.data.seq_length
        chunks = input_length // self._cfg.chunk_size
        chunk_size = self._cfg.chunk_size
        pad_id = self.tokenizer.pad_id

        all_tokens = torch.randint(0, vocab_size, (input_length + 1,))
        # make sure the eod happens at the end of each chunk, can add paddings to it
        # e.g. [..., id, id, pad, pad, pad, eod]  each has chunk_size, each sentence
        # has length of multiple of chunk_size
        hidden = all_tokens[:-1]
        labels = all_tokens[1:]

        hidden_mask = hidden != pad_id
        # to mask out the token ids [id, id,  eod, id, pad, eod, id, id]
        # so attention is not across eod, mask should be:
        # [false, true,  true, true,  true, true,  true,  true]
        # [false, false, true, true,  true, true,  true,  true]
        # [false, false, false,true,  true, true,  true,  true]
        # [true,  true,  true, false, true, true,  true,  true]
        # [true,  true,  true, true,  true, true,  true,  true]
        # [true,  true,  true, false, true, false, true,  true]
        # [true,  true,  true, true,  true, true,  false, true]
        # [true,  true,  true, true,  true, true,  false, false]
        retrieved = torch.randint(0, vocab_size, (chunks, neighbors, 2 * chunk_size))

        context_mask = retrieved != pad_id

        return {
            'tokens': hidden,
            'labels': labels,
            'tokens_mask': hidden_mask,
            'loss_mask': hidden_mask,
            'retrieved_emb_mask': context_mask,
            'retrieved_ids': retrieved,
        }


def build_mock_train_valid_test_datasets(
    cfg, trainer, splits_string, tokenizer, mock_data_size,
):
    """Build train, valid, and test datasets."""

    splits = get_train_valid_test_split_(splits_string, mock_data_size)

    # Print stats about the splits.
    logging.info(' > dataset split:')

    def print_split_stats(name, index):
        logging.info('    {}:'.format(name))
        logging.info(
            '     document indices in [{}, {}) total of {} '
            'documents'.format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
        )

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            dataset = MockRETRODataset(cfg, trainer, tokenizer, name, splits[index + 1] - splits[index],)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)
