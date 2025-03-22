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

from nemo.utils import logging


def clean_split(name):
    """removes split from name

    Args:
        name (str): partition name (e.g. "train[:100]")

    Returns:
        str: return partition name without any selector (e.g. "train").
    """
    if "[" in name:
        name = name.split("[")[0]
    if '+' in name:
        name = name.split('+')[0]
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
    valid_split_names = ["train", "test", "val"]
    dataset_splits = {_split: None for _split in valid_split_names}

    alias_to_split = {}
    for split_name, _split_aliases in split_aliases.items():
        assert split_name in valid_split_names
        for alias in _split_aliases:
            alias_to_split[alias] = split_name
    for name in valid_split_names:
        alias_to_split[name] = name

    if isinstance(dataset, Dataset):
        assert isinstance(split, str), "Expected split to be a string, but got {}".format(type(split))
        split = clean_split(split)
        split = alias_to_split[split]
        dataset_splits[split] = dataset
    elif isinstance(dataset, DatasetDict):
        dataset_split_names = dataset.keys()
        logging.info("HF dataset has the following splits: {}".format(dataset_split_names))
        for alias_split_name, split in dataset.items():
            split_name = alias_to_split[alias_split_name]
            assert dataset_splits[split_name] is None
            dataset_splits[split_name] = split
    elif isinstance(split, list):
        logging.info("Loaded HF dataset will use {} splits.".format(split))
        assert isinstance(dataset, list)
        for i, alias_split_name in enumerate(map(clean_split, split)):
            split_name = alias_to_split[alias_split_name]
            assert dataset_splits[split_name] is None
            dataset_splits[split_name] = dataset[i]
    elif isinstance(split, str):
        logging.info("Loaded HF dataset has a single split.")
        assert not isinstance(dataset, list)
        alias_split_name = split
        if "+" in alias_split_name:
            raise ValueError("Split concatenation not supported")
        elif "[" in alias_split_name:
            alias_split_name = alias_split_name.split("[")[0]
        split_name = alias_to_split[alias_split_name]
        assert dataset_splits[split_name] is None
        dataset_splits[split_name] = dataset
    else:
        raise ValueError("Expected split name to be None, str or a list")

    assert set(valid_split_names) == set(dataset_splits.keys()), dataset_splits.keys()
    num_init_splits = sum(map(lambda x: x is not None, dataset_splits.values()))
    assert num_init_splits > 0, "Expected at least one split to have been initialized {}".format(num_init_splits)
    return dataset_splits


def has_dist_env_init_or_rank_env_var():
    """returns whether it runs on a dist-environment"""
    return dist.is_initialized() or int(os.environ.get('WORLD_SIZE', '0')) > 1


def batchify(tensor):
    """Ensures that the input tensor has at least two dimensions by adding an extra batch dimension if necessary.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be batchified.

    Returns
    -------
    torch.Tensor
        The tensor with an extra dimension added if it was originally 1-dimensional.
        Otherwise, the tensor is returned as-is.
    """
    if tensor.ndim == 1:
        return tensor.unsqueeze_(0)
    return tensor


def extract_key_from_dicts(batch, key):
    """Extracts the value of the given key from each dictionary in a list of dictionaries.

    Parameters
    ----------
    batch : List[dict]
        A list of dictionaries.
    key : str
        The key whose values are to be extracted from each dictionary.

    Returns
    -------
    List
        A list of values associated with the specified key, in the same order as
        the dictionaries in the input batch.
    """
    return list(map(lambda x: x[key], batch))


def pad_within_micro(batch, pad_token_id):
    """Pads each list in a batch of lists to the same length with a specified token.

    Parameters
    ----------
    batch : List[List[int]]
        A batch of sequences (e.g., token IDs), where each sequence is a list of integers.
    pad_token_id : int
        The token ID to use for padding shorter sequences.

    Returns
    -------
    List[List[int]]
        A batch of sequences where each inner list has been padded with the pad token
        to match the length of the longest sequence in the batch.
    """
    max_len = max(map(len, batch))
    return [item + [pad_token_id] * (max_len - len(item)) for item in batch]


class HFDatasetDataModule(pl.LightningDataModule):
    """HFDatasetDataModule wraps HF's load_dataset (datasets library)
    so that it can be used within NeMo.
    Users can select whether to use an mcore-sampler via use_mcore_sampler arg.

    Usage examples:

    - loading a single split (train) from a dataset
    llm.HFDatasetDataModule("rajpurkar/squad", split="train")

    - loading multiple splits (train, validation) from a dataset
    llm.HFDatasetDataModule("rajpurkar/squad", split=["train", "validation"])
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
        use_dist_sampler=False,
        train_aliases=["train", "training"],
        test_aliases=["test", "testing"],
        val_aliases=["val", "validation", "valid", "eval"],
        **kwargs,
    ) -> None:
        super().__init__()
        # Removed support for mcore-sampler post 25.02.
        if 'use_mcore_sampler' in kwargs:
            kwargs.pop('use_mcore_sampler')
        if 'mcore_dataloader_type' in kwargs:
            kwargs.pop('mcore_dataloader_type')
        assert pad_token_id is not None
        # A dataset usually will have several splits (e.g. train, val, test, etc).
        # We map synonym names to canonical names (train, test, val).
        # A synonym can be a prefix/suffixed word e.g. train <> training.
        split_aliases = {'train': train_aliases, 'test': test_aliases, 'val': val_aliases}

        # self.dataset_splits will hold the actual dataset for each split.
        if isinstance(path_or_dataset, str):
            logging.info(f"Loading HF dataset from {path_or_dataset}")
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
        self.use_dist_sampler = use_dist_sampler

    @staticmethod
    def from_dict(dataset_dict, split, **kwargs):
        """Creates a Dataset from a dictionary"""
        dataset = Dataset.from_dict(dataset_dict)
        return HFDatasetDataModule(path_or_dataset=dataset, split=split, **kwargs)

    @staticmethod
    def collate_fn(batch, pad_token_id=0):
        """Collate for VLM data"""
        return {
            key: batchify(
                torch.LongTensor(
                    pad_within_micro(
                        extract_key_from_dicts(batch, key),
                        pad_token_id,
                    )
                )
            )
            for key in batch[0].keys()
        }

    def setup(self, stage: str):
        """setups sampler"""
        # Turn-on dist-sampler if the user is running inside a dist-env.
        if not self.use_dist_sampler and has_dist_env_init_or_rank_env_var():
            self.use_dist_sampler = True
            logging.info("Turning on distributed data sampler")

    def get_data_sampler(self, dataset):
        """returns the data sampler"""
        if self.use_dist_sampler:
            return DistributedSampler(dataset)
        else:
            return None

    def _make_dataloader(self, dataset, collate_fn=None):
        """Creates a dataloader"""
        assert dataset is not None

        if collate_fn is None:

            def collate_fn(x):
                return HFDatasetDataModule.collate_fn(x, pad_token_id=self.pad_token_id)

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
        """Train data split"""
        return self.dataset_splits['train']

    @property
    def val(self):
        """Validation data split"""
        return self.dataset_splits['val']

    @property
    def test(self):
        """Testing data split"""
        return self.dataset_splits['test']

    def train_dataloader(self):
        """Creates a dataloader for the train split"""
        return self._make_dataloader(self.train, self._collate_fn)

    def val_dataloader(self):
        """Creates a dataloader for the validation split"""
        return self._make_dataloader(self.val, self._collate_fn)

    def test_dataloader(self):
        """Creates a dataloader for the test split"""
        return self._make_dataloader(self.test, self._collate_fn)

    def map(self, function=None, split_names=None, **kwargs):
        """Maps a function to all/selected splits
        Additional arguments can be passed down to dataset's map via kwargs"""
        if isinstance(split_names, str):
            split_names = [split_names]
        elif isinstance(split_names, list):
            pass
        elif split_names is None:
            split_names = self.dataset_splits.keys()
        else:
            raise ValueError("split_names must None/str/list")

        for split_name in split_names:
            if not self.dataset_splits[split_name] is None:
                self.dataset_splits[split_name] = self.dataset_splits[split_name].map(function, **kwargs)
