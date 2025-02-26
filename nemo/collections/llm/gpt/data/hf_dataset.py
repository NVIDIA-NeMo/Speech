# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import lightning.pytorch as pl
import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging


def clean_split(name):
    """removes split from name

    Args:
        name (str): partition name (e.g. "train[:100]")

    Returns:
        str: return partition name without any selector (e.g. "train").
    """
    if '[' in name:
        return name.split('[')[0]
    return name


def make_dataset_splits(dataset, split, split_aliases):
    """
    Given a dataset (e.g. from datasets.load_dataset or datasets.Dataset.from_dict) it
    returns a dictionary containing the corresponding dataset splits.

    For example:

    $ ds = load_dataset("dataset-id")
    $ ans = make_dataset_splits(ds)

    # `ds` contains the following
    $ print(ds)
    > DatasetDict({
    >    train: Dataset({
    >        features: ['id', 'title', 'context', 'question', 'answers'],
    >        num_rows: 87599
    >    })
    >    validation: Dataset({
    >        features: ['id', 'title', 'context', 'question', 'answers'],
    >        num_rows: 10570
    >    })
    > })

    # In this case the value of `ans` (returned value) will be:
    $ print(ans)
    > {
    >    "train": Dataset .. (with 87599 rows),
    >    "val": Dataset .. (with 10570 rows),
    > }
    """
    valid_split_names = ['train', 'test', 'val']
    dataset_splits = {_split: None for _split in valid_split_names}

    alias_to_split = {}
    for split_name, _split_aliases in split_aliases.items():
        assert split_name in valid_split_names
        for alias in _split_aliases:
            alias_to_split[alias] = split_name

    if isinstance(dataset, Dataset):
        assert isinstance(split, str), "Expected split to be a string, but got " + str(type(split))
        split = clean_split(split)
        dataset_splits[split] = dataset
    elif isinstance(dataset, DatasetDict):
        dataset_split_names = dataset.keys()
        logging.info(f"HF dataset has the following splits: {dataset_split_names}")
        for alias_split_name, split in dataset.items():
            split_name = alias_to_split[alias_split_name]
            assert dataset_splits[split_name] is None
            dataset_splits[split_name] = split
    elif isinstance(split, list):
        logging.info(f"Loaded HF dataset will use {str(split)} splits.")
        assert isinstance(dataset, list)
        for i, alias_split_name in enumerate(map(clean_split, split)):
            split_name = alias_to_split[alias_split_name]
            assert dataset_splits[split_name] is None
            dataset_splits[split_name] = dataset[i]
    elif isinstance(split, str):
        logging.info("Loaded HF dataset has a single split.")
        assert not isinstance(dataset, list)
        alias_split_name = split
        if '+' in alias_split_name:
            raise ValueError("Split concatenation not supported")
        elif '[' in alias_split_name:
            alias_split_name = alias_split_name.split('[')[0]
        split_name = alias_to_split[alias_split_name]
        assert dataset_splits[split_name] is None
        dataset_splits[split_name] = dataset
    else:
        raise ValueError("Expected split name to be None, str or a list")

    assert set(valid_split_names) == set(dataset_splits.keys()), dataset_splits.keys()
    num_init_splits = sum(map(lambda x: x is not None, dataset_splits.values()))
    assert num_init_splits > 0, f"Expected at least one split to have been initialized {num_init_splits}"
    return dataset_splits


def has_dist_env_init_or_rank_env_var():
    """returns whether it runs on a dist-environment"""
    env_vars = ['LOCAL_RANK', 'GLOBAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    return dist.is_initialized() or any(map(lambda x: x in os.environ, env_vars))


class HFDatasetDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for loading and managing datasets from the `datasets` library.

    Args:
        path_or_dataset (str | Dataset | DatasetDict): The dataset name from HF or a preloaded dataset.
        split (str | list, optional): The dataset split(s) to load (e.g., "train" or ["train", "validation"]).
            Defaults to None.
        collate_fn (callable, optional): Custom function for batching data; defaults to a padding-based collation.
            Defaults to None.
        num_workers (int, optional): Number of workers for data loading. Defaults to 2.
        pin_memory (bool, optional): Whether to use pinned memory for faster GPU transfers. Defaults to True.
        persistent_workers (bool, optional): Whether to keep worker threads alive between epochs. Defaults to True.
        seq_length (int, optional): Maximum sequence length for tokenized inputs. Defaults to 1024.
        micro_batch_size (int, optional): Batch size per device. Defaults to 2.
        global_batch_size (int, optional): Total batch size across all devices. Defaults to 2.
        pad_token_id (int, optional): Token ID used for padding sequences. Defaults to 0.
        use_mcore_sampler (bool, optional): Whether to use NVIDIA MCore sampler for efficient data loading.
            Defaults to False.
        use_dist_sampler (bool, optional): Whether to enable distributed sampling. Defaults to False.
        mcore_dataloader_type (str, optional): Dataloader type when using MCore sampling. Defaults to 'cyclic'.
        train_aliases (list, optional): Alternative names for the training split. Defaults to ["train", "training"].
        test_aliases (list, optional): Alternative names for the test split. Defaults to ["test", "testing"].
        val_aliases (list, optional): Alternative names for the validation split.
            Defaults to ["val", "validation", "valid", "eval"].
        **kwargs: Additional arguments passed to `datasets.load_dataset`.

    Raises:
        ValueError: If `path_or_dataset` is not a valid dataset type (str, Dataset, or DatasetDict).

    Examples:
        Load a single split (train) from a dataset:
        ```python
        data_module = HFDatasetDataModule("rajpurkar/squad", split="train")
        ```

        Load multiple splits (train and validation):
        ```python
        data_module = HFDatasetDataModule("rajpurkar/squad", split=["train", "validation"])
        ```

        Use a preloaded dataset:
        ```python
        from datasets import load_dataset
        dataset = load_dataset("imdb")
        data_module = HFDatasetDataModule(dataset, split="train")
        ```

    Notes:
        - If neither `use_dist_sampler` nor `use_mcore_sampler` are enabled, but a distributed
        environment is detected, HFDatasetDataModule will use a distributed-sampler automatically.
        - If no collation function is provided, a default function with padding using `pad_token_id` is applied.
    """

    def __init__(
        self,
        path_or_dataset,
        split=None,
        collate_fn=None,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        seq_length=1024,
        micro_batch_size=2,
        global_batch_size=2,
        pad_token_id=0,
        use_mcore_sampler=False,
        use_dist_sampler=False,
        mcore_dataloader_type='cyclic',
        train_aliases=["train", "training"],
        test_aliases=["test", "testing"],
        val_aliases=["val", "validation", "valid", "eval"],
        **kwargs,
    ) -> None:
        super().__init__()
        assert pad_token_id is not None
        # A dataset usually will have several splits (e.g. train, val, test, etc).
        # We map synonym names to canonical names (train, test, val).
        # A synonym can be a prefix/suffixed word e.g. train <> training.
        split_aliases = {'train': train_aliases, 'test': test_aliases, 'val': val_aliases}

        # self.dataset_splits will hold the actual dataset for each split.
        if isinstance(path_or_dataset, str):
            logging.info(f"Loading HF dataset from {path_or_dataset}, this may take a moment.")
            dataset = load_dataset(path_or_dataset, split=split, **kwargs)
        elif isinstance(path_or_dataset, Dataset) or isinstance(path_or_dataset, DatasetDict):
            logging.info(f"Using passed HF dataset {str(path_or_dataset)}")
            dataset = path_or_dataset
        else:
            raise ValueError(
                "Expected `path_or_dataset` to be str, Dataset, DatasetDict, but got " + str(type(path_or_dataset))
            )

        self.dataset_splits = make_dataset_splits(dataset, split, split_aliases)

        if collate_fn is None:
            self._collate_fn = lambda x: HFDatasetDataModule.collate_fn(x, pad_token_id=self.pad_token_id)
        else:
            self._collate_fn = collate_fn

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.pad_token_id = pad_token_id

        self.use_mcore_sampler = use_mcore_sampler
        self.mcore_dataloader_type = mcore_dataloader_type
        self.use_dist_sampler = use_dist_sampler

    @staticmethod
    def from_dict(dataset_dict, split, **kwargs):
        """wraps Dataset's from_dict method"""
        dataset = Dataset.from_dict(dataset_dict)
        return HFDatasetDataModule(path_or_dataset=dataset, split=split, **kwargs)

    @staticmethod
    def collate_fn(batch, pad_token_id=0):
        """Default batch collator"""

        def batchify(tensor):
            if tensor.ndim == 1:
                return tensor.unsqueeze_(0)
            return tensor

        def extract_key_from_dicts(batch, key):
            return list(map(lambda x: x[key], batch))

        def pad_within_micro(batch, pad_token_id):
            max_len = max(map(len, batch))
            return [item + [pad_token_id] * (max_len - len(item)) for item in batch]

        return {
            key: batchify(
                torch.LongTensor(
                    pad_within_micro(
                        extract_key_from_dicts(batch, key),
                        pad_token_id if key != 'loss_mask' else 0,
                    )
                )
            )
            for key in batch[0].keys()
        }

    def setup(self, stage: str):
        """setups sampler"""
        # Turn-on dist-sampler if the user is running inside a dist-env.
        if not self.use_dist_sampler and not self.use_mcore_sampler and has_dist_env_init_or_rank_env_var():
            self.use_dist_sampler = True
            logging.info("Turning on distributed data sampler")
        elif self.use_mcore_sampler:
            self.mcore_data_sampler = MegatronDataSampler(
                seq_len=self.seq_length,
                micro_batch_size=self.micro_batch_size,
                global_batch_size=self.global_batch_size,
                dataloader_type=self.mcore_dataloader_type,
            )

    def get_data_sampler(self, dataset):
        """returns the data sampler"""
        if self.use_dist_sampler:
            return DistributedSampler(dataset)
        elif self.use_mcore_sampler:
            return self.mcore_data_sampler
        else:
            return None

    def _make_dataloader(self, dataset, collate_fn=None):
        """Dataloader creator"""
        assert dataset is not None

        if collate_fn is None:
            collate_fn = lambda x: HFDatasetDataModule.collate_fn(x, pad_token_id=self.pad_token_id)

        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
            batch_size=self.micro_batch_size,
            sampler=self.get_data_sampler(dataset),
        )

    @property
    def train(self):
        """Returns the training partition"""
        return self.dataset_splits['train']

    @property
    def val(self):
        """Returns the validation partition"""
        return self.dataset_splits['val']

    @property
    def test(self):
        """Returns the test partition"""
        return self.dataset_splits['test']

    def train_dataloader(self):
        """Returns the train dataloader"""
        return self._make_dataloader(self.train, self._collate_fn)

    def val_dataloader(self):
        """Returns the validation dataloader"""
        return self._make_dataloader(self.val, self._collate_fn)

    def test_dataloader(self):
        """Returns the test dataloader"""
        return self._make_dataloader(self.test, self._collate_fn)

    def map(self, function=None, split_names=None, **kwargs):
        """Maps a function to the dataset"""
        if isinstance(split_names, str):
            dataset_splits = {split_names: self.dataset_splits[split_names]}
        elif isinstance(split_names, list):
            dataset_splits = {k: self.dataset_splits[k] for k in split_names}
        else:
            dataset_splits = self.dataset_splits

        for split_name, subset in dataset_splits.items():
            if subset is None:
                continue
            dataset_splits[split_name] = subset.map(function, **kwargs)


class SquadHFDataModule(HFDatasetDataModule):
    """
    A data module for handling the SQuAD dataset using HFDatasetDataModule.

    This class is responsible for tokenizing and formatting the SQuAD dataset for training
    language models. It extends `HFDatasetDataModule` and implements a prompt-based
    formatting function suitable for causal language modeling.

    Attributes:
        tokenizer: A tokenizer instance used to convert text into token IDs.
    """

    def __init__(self, tokenizer, **kwargs):
        """
        Initializes the SquadHFDataModule.

        Args:
            tokenizer: A tokenizer instance for processing text data.
            **kwargs: Additional arguments passed to the parent class (`HFDatasetDataModule`).
        """
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

    def formatting_prompts_func(self, example):
        """
        Formats a given example into a structured prompt for training.

        This method converts a dataset example (containing context, question, and answer)
        into a structured format, tokenizes it, and prepares input IDs and labels for
        training a language model.

        Args:
            example (dict): A dictionary containing the following keys:
                - 'context': The passage from which the question is derived.
                - 'question': The question about the passage.
                - 'answers': A dictionary with a 'text' key containing the answer(s).

        Returns:
            dict: A dictionary containing:
                - 'input_ids': Tokenized input sequence (excluding the last token).
                - 'labels': Tokenized output sequence (excluding the first token).
                - 'loss_mask': A mask indicating which tokens contribute to the loss.
        """
        formatted_text = [
            f"Context: {example['context']} Question: {example['question']} Answer:",
            f" {example['answers']['text'][0].strip()}",
        ]
        context_ids, answer_ids = list(map(self.tokenizer.text_to_ids, formatted_text))
        if len(context_ids) > 0 and context_ids[0] != self.tokenizer.bos_id:
            context_ids.insert(0, self.tokenizer.bos_id)
        if len(answer_ids) > 0 and answer_ids[-1] != self.tokenizer.eos_id:
            answer_ids.append(self.tokenizer.eos_id)

        return dict(
            labels=(context_ids + answer_ids)[1:],
            input_ids=(context_ids + answer_ids)[:-1],
            loss_mask=[0] * (len(context_ids) - 1) + [1] * len(answer_ids),
        )

    def setup(self, stage):
        """
        Prepares the dataset for training and applies formatting.

        Args:
            stage (str): The stage of training.
        """
        super().setup(stage)

        self.map(
            self.formatting_prompts_func,
            batched=False,
            batch_size=2,
            remove_columns=["id", "title", "context", "question", 'answers'],
        )
