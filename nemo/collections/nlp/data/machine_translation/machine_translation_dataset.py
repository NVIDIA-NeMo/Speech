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

"""Pytorch Dataset for training Neural Machine Translation."""

import io
import json
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional

import braceexpand
import numpy as np
import webdataset as wd
from torch.utils.data import IterableDataset

from nemo.collections.nlp.data.data_utils.data_preprocessing import dataset_to_ids
from nemo.core import Dataset
from nemo.utils import logging

__all__ = ['TranslationDataset', 'TarredTranslationDataset']


@dataclass
class TranslationDataConfig:
    src_file_name: Optional[Any] = None  # Any = str or List[str]
    tgt_file_name: Optional[Any] = None  # Any = str or List[str]
    use_tarred_dataset: bool = False
    tar_files: Optional[Any] = None  # Any = str or List[str]
    metadata_file: Optional[Any] = None  # Any = str or List[str]
    lines_per_dataset_fragment: Optional[int] = 1000000
    num_batches_per_tarfile: Optional[int] = 1000
    shard_strategy: Optional[str] = 'scatter'
    tokens_in_batch: int = 512
    clean: bool = False
    max_seq_length: int = 512
    min_seq_length: int = 1
    cache_ids: bool = False
    cache_data_per_node: bool = False
    use_cache: bool = False
    shuffle: bool = False
    num_samples: int = -1
    drop_last: bool = False
    pin_memory: bool = False
    num_workers: int = 8
    reverse_lang_direction: bool = False
    load_from_tarred_dataset: bool = False
    metadata_path: Optional[str] = None
    tar_shuffle_n: int = 100
    n_preproc_jobs: int = -2
    tar_file_prefix: str = 'parallel'
    concat_sampling_technique: Optional[str] = 'temperature'
    concat_sampling_temperature: Optional[int] = 5
    concat_sampling_probabilities: Optional[List[float]] = None


class TranslationDataset(Dataset):
    def __init__(
        self,
        dataset_src: str,
        dataset_tgt: str,
        tokens_in_batch: int = 1024,
        clean: bool = False,
        max_seq_length: int = 512,
        min_seq_length: int = 1,
        max_seq_length_diff: int = 512,
        max_seq_length_ratio: int = 512,
        cache_ids: bool = False,
        cache_data_per_node: bool = False,
        use_cache: bool = False,
        reverse_lang_direction: bool = False,
        prepend_id: int = None,
        add_bos_eos_to_encoder: bool = True,
    ):
        self.dataset_src = dataset_src
        self.dataset_tgt = dataset_tgt
        self.tokens_in_batch = tokens_in_batch
        self.cache_ids = cache_ids
        self.use_cache = use_cache
        self.clean = clean
        self.cache_data_per_node = cache_data_per_node
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.max_seq_length_diff = max_seq_length_diff
        self.max_seq_length_ratio = max_seq_length_ratio
        self.reverse_lang_direction = reverse_lang_direction
        self.prepend_id = prepend_id
        self.add_bos_eos_to_encoder = add_bos_eos_to_encoder

        # deprecation warnings for cache_ids, use_cache, and cache_data_per_node
        if self.cache_ids is True or self.use_cache is True or self.cache_data_per_node is True:
            logging.warning(
                'Deprecation warning. self.cache_ids, self.use_cache, and self.cache_data_per_node will be removed. Data caching to be done with tarred datasets moving forward.'
            )

    def batchify(self, tokenizer_src, tokenizer_tgt):
        src_ids = dataset_to_ids(
            self.dataset_src,
            tokenizer_src,
            cache_ids=self.cache_ids,
            cache_data_per_node=self.cache_data_per_node,
            use_cache=self.use_cache,
            add_bos_eos=self.add_bos_eos_to_encoder,
            remove_trailing_newline=True,
        )
        tgt_ids = dataset_to_ids(
            self.dataset_tgt,
            tokenizer_tgt,
            cache_ids=self.cache_ids,
            cache_data_per_node=self.cache_data_per_node,
            use_cache=self.use_cache,
            remove_trailing_newline=True,
        )
        if self.clean:
            src_ids, tgt_ids = self.clean_src_and_target(
                src_ids,
                tgt_ids,
                max_tokens=self.max_seq_length,
                min_tokens=self.min_seq_length,
                max_tokens_diff=self.max_seq_length_diff,
                max_tokens_ratio=self.max_seq_length_ratio,
            )
        self.src_pad_id = tokenizer_src.pad_id
        self.tgt_pad_id = tokenizer_tgt.pad_id

        self.batch_indices = self.pack_data_into_batches(src_ids, tgt_ids)
        self.batches = self.pad_batches(src_ids, tgt_ids, self.batch_indices)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        src_ids = self.batches[idx]["src"]
        tgt = self.batches[idx]["tgt"]
        if self.reverse_lang_direction:
            src_ids, tgt = tgt, src_ids
        labels = tgt[:, 1:]
        tgt_ids = tgt[:, :-1]
        if self.prepend_id:
            src_ids = np.insert(src_ids, 0, self.prepend_id, axis=-1)
        src_mask = (src_ids != self.src_pad_id).astype(np.int32)
        tgt_mask = (tgt_ids != self.tgt_pad_id).astype(np.int32)
        return src_ids, src_mask, tgt_ids, tgt_mask, labels

    def pad_batches(self, src_ids, tgt_ids, batch_indices):
        """
        Augments source and target ids in the batches with padding symbol
        to make the lengths of all sentences in the batches equal.
        """

        batches = {}
        for batch_idx, b in enumerate(batch_indices):
            src_len = max([len(src_ids[i]) for i in b])
            tgt_len = max([len(tgt_ids[i]) for i in b])
            src_ids_ = self.src_pad_id * np.ones((len(b), src_len), dtype=np.int)
            tgt_ids_ = self.tgt_pad_id * np.ones((len(b), tgt_len), dtype=np.int)
            for i, sentence_idx in enumerate(b):
                src_ids_[i][: len(src_ids[sentence_idx])] = src_ids[sentence_idx]
                tgt_ids_[i][: len(tgt_ids[sentence_idx])] = tgt_ids[sentence_idx]
            batches[batch_idx] = {"src": src_ids_, "tgt": tgt_ids_}
        return batches

    def pack_data_into_batches(self, src_ids, tgt_ids):
        """
        Takes two lists of source and target sentences, sorts them, and packs
        into batches to minimize the use of padding tokens. Returns a list of
        batches where each batch contains indices of sentences included into it
        """

        # create buckets sorted by the number of src tokens
        # each bucket is also sorted by the number of tgt tokens
        buckets = {}
        for i, src_id in enumerate(src_ids):
            src_len, tgt_len = len(src_id), len(tgt_ids[i])
            if src_len not in buckets:
                buckets[src_len] = [(tgt_len, i)]
            else:
                buckets[src_len].append((tgt_len, i))

        for b_idx in buckets:
            buckets[b_idx] = sorted(buckets[b_idx])

        buckets = OrderedDict(sorted(buckets.items()))
        indices = list(buckets.keys())

        batches = [[]]
        num_batches = 0
        batch_size = 0
        i = 0
        src_len = 0
        tgt_len = 0

        while i < len(buckets):
            while buckets[indices[i]]:

                i_src = max(src_len, indices[i])
                i_tgt = max(tgt_len, buckets[indices[i]][0][0])

                try:
                    ip1_src = max(src_len, indices[i + 1])
                    ip1_tgt = max(tgt_len, buckets[indices[i + 1]][0][0])
                except IndexError:
                    ip1_src = i_src + 1
                    ip1_tgt = i_tgt + 1

                if i_src + i_tgt <= ip1_src + ip1_tgt:
                    src_len = i_src
                    tgt_len = i_tgt
                    _, idx = buckets[indices[i]].pop(0)
                else:
                    src_len = ip1_src
                    tgt_len = ip1_tgt
                    _, idx = buckets[indices[i + 1]].pop(0)

                batches[num_batches].append(idx)
                batch_size += 1

                if batch_size * (src_len + tgt_len) > self.tokens_in_batch:

                    num_examples_to_split = len(batches[num_batches])
                    batches_to_evict = 8 * ((num_examples_to_split - 1) // 8)

                    if batches_to_evict == 0:
                        batches_to_evict = num_examples_to_split

                    batches.append(batches[num_batches][batches_to_evict:])
                    batches[num_batches] = batches[num_batches][:batches_to_evict]
                    batch_size = num_examples_to_split - batches_to_evict

                    num_batches += 1
                    if batch_size > 0:
                        src_len = max([len(src_ids[j]) for j in batches[num_batches]])
                        tgt_len = max([len(tgt_ids[j]) for j in batches[num_batches]])
                    else:
                        src_len = 0
                        tgt_len = 0
                    break

            if not buckets[indices[i]]:
                i = i + 1

        if not batches[-1]:
            batches.pop(-1)

        return batches

    def clean_src_and_target(
        self,
        src_ids,
        tgt_ids,
        max_tokens=None,
        min_tokens=None,
        max_tokens_diff=None,
        max_tokens_ratio=None,
        filter_equal_src_and_dest=False,
    ):
        """
        Cleans source and target sentences to get rid of noisy data.
        Specifically, a pair of sentences is removed if
          -- either source or target is longer than *max_tokens*
          -- either source or target is shorter than *min_tokens*
          -- absolute difference between source and target is larger than
             *max_tokens_diff*
          -- one sentence is *max_tokens_ratio* times longer than the other
        """

        if len(src_ids) != len(tgt_ids):
            raise ValueError("Source and target corpora have different lengths!")
        src_ids_, tgt_ids_ = [], []
        for i in range(len(src_ids)):
            src_len, tgt_len = len(src_ids[i]), len(tgt_ids[i])
            if (
                (max_tokens is not None and (src_len > max_tokens or tgt_len > max_tokens))
                or (min_tokens is not None and (src_len < min_tokens or tgt_len < min_tokens))
                or (filter_equal_src_and_dest and src_ids[i] == tgt_ids[i])
                or (max_tokens_diff is not None and np.abs(src_len - tgt_len) > max_tokens_diff)
            ):
                continue
            if max_tokens_ratio is not None:
                ratio = max(src_len - 2, 1) / max(tgt_len - 2, 1)
                if ratio > max_tokens_ratio or ratio < (1 / max_tokens_ratio):
                    continue
            src_ids_.append(src_ids[i])
            tgt_ids_.append(tgt_ids[i])
        return src_ids_, tgt_ids_


class TarredTranslationDataset(IterableDataset):
    """
    A similar Dataset to the TranslationDataset, but which loads tarred tokenized pickle files.
    Accepts a single JSON metadata file containing the total number of batches
    as well as the path(s) to the tarball(s) containing the pickled parallel dataset batch files.
    Valid formats for the text_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/text.tar' or 'path/to/text_{1..100}.tar', or
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
        encoder_tokenizer: Autokenizer wrapped BPE tokenizer model, such as YTTM
        decoder_tokenizer: Autokenizer wrapped BPE tokenizer model, such as YTTM
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
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 1.
        reverse_lang_direction (bool): When True, swaps the source and target directions when returning minibatches.
        prepend_id (int): Prepends the specificed token id to the start of every source sentence. Defaults to None.
    """

    def __init__(
        self,
        text_tar_filepaths: str,
        metadata_path: str,
        encoder_tokenizer: str,
        decoder_tokenizer: str,
        shuffle_n: int = 1,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 1,
        reverse_lang_direction: bool = False,
        prepend_id: int = None,
    ):
        super(TarredTranslationDataset, self).__init__()

        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.reverse_lang_direction = reverse_lang_direction
        self.src_pad_id = encoder_tokenizer.pad_id
        self.tgt_pad_id = decoder_tokenizer.pad_id
        self.prepend_id = prepend_id

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
        self._dataset = wd.WebDataset(urls=text_tar_filepaths, nodesplitter=None)

        if shuffle_n > 0:
            self._dataset = self._dataset.shuffle(shuffle_n, initial=shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")

        self._dataset = self._dataset.rename(pkl='pkl', key='__key__').to_tuple('pkl', 'key').map(f=self._build_sample)

    def _build_sample(self, fname):
        # Load file
        pkl_file, _ = fname
        pkl_file = io.BytesIO(pkl_file)
        data = pickle.load(pkl_file)  # loads np.int64 vector
        pkl_file.close()
        src_ids = data["src"]
        tgt = data["tgt"]
        if self.reverse_lang_direction:
            src_ids, tgt = tgt, src_ids
        labels = tgt[:, 1:]
        tgt_ids = tgt[:, :-1]
        if self.prepend_id:
            src_ids = np.insert(src_ids, 0, self.prepend_id, axis=-1)
        src_mask = (src_ids != self.src_pad_id).astype(np.int32)
        tgt_mask = (tgt_ids != self.tgt_pad_id).astype(np.int32)
        return src_ids, src_mask, tgt_ids, tgt_mask, labels

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return self.length
