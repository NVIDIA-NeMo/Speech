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

import io
import json
from typing import Optional

import braceexpand
import numpy as np
import webdataset as wds
from torch.utils.data import Dataset, IterableDataset

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils import dataset_to_ids
from nemo.utils import logging
from nemo.utils.distributed import webdataset_split_by_workers

__all__ = ['L2RLanguageModelingDataset', 'TarredL2RLanguageModelingDataset']


class L2RLanguageModelingDataset(Dataset):
    """
    Dataset for training and evaluating left-to-right language models.

    Args:
        tokenizer: tokenizer, such as WordTokenizer or CharTokenizer
        dataset: path to data
        max_seq_length: maximum sequence length (in tokens) of input tensors
        batch_step: distance (in tokens) between two successive sequences of
            the text. By default, it is equal to max_seq_length which corresponds
            to splitting text into disjoint segments covering full dataset
        use_cache: bool value, defaults to False. Determines whether the preprocessed,
            tokenized dataset should be cached into a pickle file. If true, cache is saved
            at the path provided in `dataset`.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        dataset: str,
        max_seq_length: Optional[int] = 512,
        batch_step: Optional[int] = None,
        use_cache: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_step = batch_step or self.max_seq_length
        ids = dataset_to_ids(dataset, tokenizer, cache_ids=use_cache, add_bos_eos=False)
        self.ids = np.array([j for i in ids for j in i])

    def __len__(self):
        return (len(self.ids) - self.max_seq_length) // self.batch_step

    def __getitem__(self, idx):
        left = idx * self.batch_step
        right = left + self.max_seq_length
        src_ids = self.ids[left:right]
        labels = self.ids[left + 1 : right + 1]
        src_mask = (src_ids != self.tokenizer.pad_id).astype(np.float32)
        return src_ids, src_mask, labels


class TarredL2RLanguageModelingDataset(IterableDataset):
    """
    A similar Dataset to the L2RLanguageModelingDataset, but which loads tarred tokenized numpy files.
    Accepts a single JSON metadata manifest file as well as the path(s) to the tarball(s) containing the wav files.
    The manifest should contain information such as the number of shards, the number of tokens in the corpus,
    and the number of tokens contained within each shard of the tarfile(s).

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
        tokenizer: tokenizer, such as WordTokenizer or CharTokenizer
        dataset: path to data
        max_seq_length: maximum sequence length (in tokens) of input tensors
        batch_step: distance (in tokens) between two successive sequences of
            the text. By default, it is equal to max_seq_length which corresponds
            to splitting text into disjoint segments covering full dataset
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
    """

    def __init__(
        self,
        text_tar_filepaths: str,
        metadata_path: str,
        tokenizer,
        max_seq_length: int = 512,
        batch_step: int = None,
        shuffle_n: int = 1,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
    ):
        super(TarredL2RLanguageModelingDataset, self).__init__()

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_step = batch_step or self.max_seq_length

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
            logging.info("All tarred dataset shards will be scattered evenly across all nodes.")

            if len(text_tar_filepaths) % world_size != 0:
                logging.warning(
                    f"Number of shards in tarred dataset ({len(text_tar_filepaths)}) is not divisible "
                    f"by number of distributed workers ({world_size})."
                )

            begin_idx = (len(text_tar_filepaths) // world_size) * global_rank
            end_idx = begin_idx + (len(text_tar_filepaths) // world_size)
            text_tar_filepaths = text_tar_filepaths[begin_idx:end_idx]
            logging.info(
                "Partitioning tarred dataset: process (%d) taking shards [%d, %d)", global_rank, begin_idx, end_idx
            )

        elif shard_strategy == 'replicate':
            logging.info("All tarred dataset shards will be replicated across all nodes.")

        else:
            raise ValueError(f"Invalid shard strategy ! Allowed values are : {valid_shard_strategies}")

        self.tarpath = text_tar_filepaths

        # Put together WebDataset
        self._dataset = wds.DataPipeline(
            wds.SimpleShardList(text_tar_filepaths),
            webdataset_split_by_workers,
            wds.shuffle(shuffle_n),
            wds.tarfile_to_samples(),
            wds.rename(npy='npy', key='__key__'),
            wds.to_tuple('npy', 'key'),
            wds.map(self._build_sample),
        )

    def _build_sample(self, tup):
        # Load file
        npy, filepath = tup
        npy = io.BytesIO(npy)
        data = np.load(npy)  # loads np.int64 vector
        npy.close()

        # Select random contiguous subsegment
        idx = np.random.randint(0, (len(data) - self.max_seq_length) // self.batch_step)

        # Slice of data chunk
        left = idx * self.batch_step
        right = left + self.max_seq_length
        data = data[left : right + 1]

        # Create batch
        src_ids = data[:-1]
        labels = data[1:]
        src_mask = (src_ids != self.tokenizer.pad_id).astype(np.float32)
        return src_ids, src_mask, labels

    def __iter__(self):
        # We need to wrap an infinite generator since the actual files
        # within the tar files contains large chunks of contiguous data.
        # This prevents PTL from early exiting the train loop after exhausting
        # all of the files in one iteration (though the actual dataset is many
        # times larger due to each file containing a large chunk of data).
        dl_iter = iter(self._dataset)
        while True:
            try:
                batch = next(dl_iter)
                yield batch
            except StopIteration:
                dl_iter = iter(self._dataset)
                continue

    def __len__(self):
        return (self.metadata['num_text'] - self.max_seq_length) // self.batch_step
