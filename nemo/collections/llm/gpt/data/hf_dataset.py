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

import datasets.dataset_dict
import lightning.pytorch as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging


def make_dataset_splits(path, split, split_aliases, kwargs):
    """
    Loads a dataset with datasets.load_dataset and
    returns a dictionary containing all dataset splits.

    For example:

    ans = make_dataset_splits("dataset-id")
        $ ds = load_dataset("dataset-id")
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

    In this case the value of `ans` (returned value) will be:
    $ print(ans)
    > {
    >    "train": Dataset .. (with 87599 rows),
    >    "val": Dataset .. (with 10570 rows),
    > }
    """
    dataset = load_dataset(path, split=split, **kwargs)

    split_names = ['train', 'test', 'val']
    dataset_splits = {split: None for split in split_names}

    alias_to_split = {}
    for split_name, _split_aliases in split_aliases.items():
        assert split_name in split_names
        for alias in _split_aliases:
            alias_to_split[alias] = split_name

    if isinstance(dataset, datasets.dataset_dict.DatasetDict):
        dataset_split_names = dataset.keys()
        logging.info(f"HF dataset has the following splits: {dataset_split_names}")
        for alias_split_name, split in dataset.items():
            split_name = alias_to_split[alias_split_name]
            assert dataset_splits[split_name] is None
            dataset_splits[split_name] = split
    elif isinstance(split, list):
        logging.info(f"Loaded HF dataset will use " + str(split) + " splits.")
        assert isinstance(dataset, list)
        for i, alias_split_name in enumerate(split):
            split_name = alias_to_split[alias_split_name]
            assert dataset_splits[split_name] is None
            dataset_splits[split_name] = dataset[i]
    elif isinstance(split, str):
        logging.info(f"Loaded HF dataset has a single split.")
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

    assert (
        sum(map(lambda x: x is not None, dataset_splits.values())) > 0
    ), "Expected at least one dataset to have been initialized"
    return dataset_splits


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
        path,
        collate_fn=None,
        split=None,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        seq_length=1024,
        micro_batch_size=2,
        global_batch_size=2,
        pad_token_id=0,
        use_mcore_sampler=False,
        mcore_dataloader_type='cyclic',
        train_aliases=["train", "training"],
        test_aliases=["test", "testing"],
        val_aliases=["val", "validation", "valid", "eval"],
        **kwargs,
    ) -> None:
        super().__init__()
        assert pad_token_id is not None

        logging.info(f"Loading HF dataset from {path}")

        # A dataset usually will have several splits (e.g. train, val, test, etc).
        # We map synonym names to canonical names (train, test, val).
        # A synonym can be a prefix/suffixed word e.g. train <> training.
        split_aliases = {'train': train_aliases, 'test': test_aliases, 'val': val_aliases}

        # self.dataset_splits will hold the actual dataset for each split.
        self.dataset_splits = make_dataset_splits(path, split, split_aliases, kwargs)

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

    @staticmethod
    def collate_fn(batch, pad_token_id=0):
        def batchify(tensor):
            if tensor.ndim == 1:
                return tensor.unsqueeze_(0)
            return tensor

        def extract_key_from_dicts(batch, key):
            return list(map(lambda x: x[key], batch))

        def pad_within_micro(batch, pad_token_id):
            max_len = max(map(len, batch))
            return [item + [pad_token_id] * (max_len - len(item)) for item in batch]

        keys = list(filter(lambda x: x in batch[0], ['tokens', 'labels', 'position_ids', 'loss_mask']))
        return {
            key: batchify(
                torch.LongTensor(
                    pad_within_micro(
                        extract_key_from_dicts(batch, key),
                        pad_token_id,
                    )
                )
            )
            for key in keys
        }

    def setup(self, stage: str):
        if not self.use_mcore_sampler:
            return
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            dataloader_type=self.mcore_dataloader_type,
        )

    def _make_dataloader(self, dataset, collate_fn=None):
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
        )

    @property
    def train(self):
        return self.dataset_splits['train']

    @property
    def val(self):
        return self.dataset_splits['val']

    @property
    def test(self):
        return self.dataset_splits['test']

    def train_dataloader(self):
        return self._make_dataloader(self.train, self._collate_fn)

    def val_dataloader(self):
        return self._make_dataloader(self.val, self._collate_fn)

    def test_dataloader(self):
        return self._make_dataloader(self.test, self._collate_fn)

    def map(self, function=None, split_names=None, **kwargs):
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
