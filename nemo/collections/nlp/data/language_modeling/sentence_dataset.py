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

"""Pytorch Dataset with sentences packed into batches by length."""
import io
import json
import logging
import pickle
from collections import OrderedDict
from typing import Any

import braceexpand
import numpy as np
import webdataset as wds
from torch.utils.data import IterableDataset

from nemo.collections.nlp.data.data_utils.data_preprocessing import dataset_to_ids
from nemo.core import Dataset
from nemo.utils.distributed import webdataset_split_by_workers

__all__ = ['SentenceDataset', 'TarredSentenceDataset']


class SentenceDataset(Dataset):
    def __init__(
        self,
        tokenizer: Any,
        dataset: Any,
        tokens_in_batch: int = 1024,
        clean: bool = False,
        cache_ids: bool = False,
        max_seq_length: int = 512,
        min_seq_length: int = 1,
    ):

        self.tokenizer = tokenizer
        self.tokens_in_batch = tokens_in_batch

        ids = dataset_to_ids(dataset, tokenizer, cache_ids=cache_ids)
        if clean:
            ids = self.clean(ids, max_tokens=max_seq_length, min_tokens=min_seq_length)
        self.batch_sent_ids, self.batch_elem_lengths = self.pack_data_into_batches(ids)
        self.batches = self.pad_batches(ids)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        ids = self.batches[idx]
        mask = (ids != self.tokenizer.pad_id).astype(np.int32)
        return ids, mask

    def pad_batches(self, ids):
        """
        Augments source and target ids in the batches with padding symbol
        to make the lengths of all sentences in the batches equal.
        """

        batches = []
        for batch_elem_len, batch_sent_ids in zip(self.batch_elem_lengths, self.batch_sent_ids):
            batch = self.tokenizer.pad_id * np.ones((len(batch_sent_ids), batch_elem_len), dtype=np.int64)
            for i, sentence_idx in enumerate(batch_sent_ids):
                batch[i][: len(ids[sentence_idx])] = ids[sentence_idx]
            batches.append(batch)
        return batches

    def pack_data_into_batches(self, ids):
        """
        Takes two lists of source and target sentences, sorts them, and packs
        into batches to minimize the use of padding tokens. Returns a list of
        batches where each batch contains indices of sentences included into it
        """

        # create buckets sorted by the number of src tokens
        # each bucket is also sorted by the number of tgt tokens
        buckets = {}
        for i, line_ids in enumerate(ids):
            len_ = len(line_ids)
            if len_ not in buckets:
                buckets[len_] = [i]
            else:
                buckets[len_].append(i)

        for b_idx in buckets:
            buckets[b_idx] = sorted(buckets[b_idx])

        buckets = OrderedDict(sorted(buckets.items()))

        batches = []
        batch_elem_lengths = []
        curr_batch = []
        len_of_longest_sent = 0
        for sent_len, bucket in buckets.items():
            for sent_i in bucket:
                if sent_len * (len(curr_batch) + 1) > self.tokens_in_batch:
                    if not curr_batch:
                        raise ValueError(
                            f"The limitation on number of tokens in batch {self.tokens_in_batch} is too strong."
                            f"Several sentences contain {sent_len} tokens."
                        )
                    batches.append(curr_batch)
                    batch_elem_lengths.append(sent_len)
                    curr_batch = []
                curr_batch.append(sent_i)
            len_of_longest_sent = sent_len
        if curr_batch:
            batches.append(curr_batch)
            batch_elem_lengths.append(len_of_longest_sent)
        return batches, batch_elem_lengths

    def clean(self, ids, max_tokens=None, min_tokens=None):
        """
        Cleans source and target sentences to get rid of noisy data.
        Specifically, a pair of sentences is removed if
          -- either source or target is longer than *max_tokens*
          -- either source or target is shorter than *min_tokens*
          -- absolute difference between source and target is larger than
             *max_tokens_diff*
          -- one sentence is *max_tokens_ratio* times longer than the other
        """

        ids_ = []
        for i in range(len(ids)):
            len_ = len(ids[i])
            if (max_tokens is not None and len_ > max_tokens) or (min_tokens is not None and len_ < min_tokens):
                continue
            ids_.append(ids[i])
        return ids_


class TarredSentenceDataset(IterableDataset):
    """
    A similar Dataset to the SentenceDataset, but which loads tarred tokenized pickle files.
    Accepts a single JSON metadata file containing the total number of batches
    as well as the path(s) to the tarball(s) containing the wav files.
    Valid formats for the text_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/text.tar' or 'path/to/text_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['text_1.tar', 'text_2.tar', ...].
    Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
    This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
    Supported opening braces - { <=> (, [, < and the special tag _OP_.
    Supported closing braces - } <=> ), ], > and the special tag _CL_.
    For SLURM based tasks, we suggest the use of the special tags for ease of use.
    See the WebDataset documentation for more information about accepted data and input formats.
    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    Additionally, please note that the len() of this DataLayer is assumed to be the number of tokens
    of the text data. An incorrect manifest length may lead to some DataLoader issues down the line.
    Args:
        text_tar_filepaths: Either a list of tokenized text tarball filepaths, or a
            string (can be brace-expandable).
        metadata_path (str): Path to the metadata manifest.
        encoder_tokenizer: Autokenizer wrapped BPE tokenizer model, such as SentencePiece
        decoder_tokenizer: Autokenizer wrapped BPE tokenizer model, such as SentencePiece
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                .. warning::
                    Replicated strategy allows every node to sample the entire set of available tarfiles,
                    and therefore more than one node may sample the same tarfile, and even sample the same
                    data points! As such, there is no assured guarantee that all samples in the dataset will be
                    sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                    occasions (when the number of shards is not divisible with ``world_size``), will not sample
                    the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                    or test datasets.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        reverse_lang_direction (bool): When True, swaps the source and target directions when returning minibatches.
    """

    def __init__(
        self,
        text_tar_filepaths: str,
        metadata_path: str,
        tokenizer: str,
        shuffle_n: int = 1,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
    ):
        super(TarredSentenceDataset, self).__init__()

        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_id

        valid_shard_strategies = ['scatter', 'replicate']
        if shard_strategy not in valid_shard_strategies:
            raise ValueError(
                f"Invalid shard strategy of type {type(shard_strategy)} "
                f"{repr(shard_strategy) if len(repr(shard_strategy)) < 100 else repr(shard_strategy)[:100] + '...'}! "
                f"Allowed values are: {valid_shard_strategies}."
            )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.metadata = metadata

        if isinstance(text_tar_filepaths, str):
            # Replace '(', '[', '<' and '_OP_' with '{'
            brace_keys_open = ['(', '[', '<', '_OP_']
            for bkey in brace_keys_open:
                if bkey in text_tar_filepaths:
                    text_tar_filepaths = text_tar_filepaths.replace(bkey, "{")

            # Replace ')', ']', '>' and '_CL_' with '}'
            brace_keys_close = [')', ']', '>', '_CL_']
            for bkey in brace_keys_close:
                if bkey in text_tar_filepaths:
                    text_tar_filepaths = text_tar_filepaths.replace(bkey, "}")

        if isinstance(text_tar_filepaths, str):
            # Brace expand
            text_tar_filepaths = list(braceexpand.braceexpand(text_tar_filepaths))

        if shard_strategy == 'scatter':
            logging.info("Tarred dataset shards will be scattered evenly across all nodes.")
            if len(text_tar_filepaths) % world_size != 0:
                logging.warning(
                    f"Number of shards in tarred dataset ({len(text_tar_filepaths)}) is not divisible "
                    f"by number of distributed workers ({world_size}). "
                    f"Some shards will not be used ({len(text_tar_filepaths) % world_size})."
                )
            batches_per_tar = self.metadata['num_batches'] // len(text_tar_filepaths)
            begin_idx = (len(text_tar_filepaths) // world_size) * global_rank
            end_idx = begin_idx + (len(text_tar_filepaths) // world_size)
            logging.info('Begin Index : %d' % (begin_idx))
            logging.info('End Index : %d' % (end_idx))
            text_tar_filepaths = text_tar_filepaths[begin_idx:end_idx]
            logging.info(
                "Partitioning tarred dataset: process (%d) taking shards [%d, %d)", global_rank, begin_idx, end_idx
            )
            self.length = batches_per_tar * len(text_tar_filepaths) * world_size

        elif shard_strategy == 'replicate':
            logging.info("All tarred dataset shards will be replicated across all nodes.")
            self.length = self.metadata['num_batches']

        else:
            raise ValueError(f"Invalid shard strategy ! Allowed values are : {valid_shard_strategies}")

        self.tarpath = text_tar_filepaths

        # Put together WebDataset
        self._dataset = wds.DataPipeline(
            wds.SimpleShardList(text_tar_filepaths),
            webdataset_split_by_workers,
            wds.shuffle(shuffle_n),
            wds.tarfile_to_samples(),
            wds.rename(pkl='pkl', key='__key__'),
            wds.to_tuple('pkl', 'key'),
            wds.map(self._build_sample),
        )

    def _build_sample(self, fname):
        # Load file
        pkl_file, _ = fname
        pkl_file = io.BytesIO(pkl_file)
        data = pickle.load(pkl_file)  # loads np.int64 vector
        pkl_file.close()
        ids = data["src"]
        mask = (ids != self.pad_id).astype(np.int32)
        return ids, mask

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return self.length
