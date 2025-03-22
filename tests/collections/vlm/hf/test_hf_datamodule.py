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

import pytest
import torch
from datasets import Dataset, DatasetDict

from nemo.collections.vlm.hf.data.hf_dataset import (
    HFDatasetDataModule,
    batchify,
    clean_split,
    extract_key_from_dicts,
    make_dataset_splits,
    pad_within_micro,
)


@pytest.fixture
def simple_dataset():
    """
    Returns a small HuggingFace Dataset for illustration.
    """
    data = {
        "input_ids": [[10, 20, 30], [40, 50, 60]],
        "labels": [[1], [0]],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def simple_dataset_dict(simple_dataset):
    """
    Returns a HuggingFace DatasetDict with three splits: train, validation, test.
    """
    return DatasetDict(
        {
            "train": simple_dataset,
            "validation": simple_dataset,
            "test": simple_dataset,
        }
    )


def test_clean_split_no_brackets():
    assert clean_split("train") == "train"
    assert clean_split("validation") == "validation"
    assert clean_split("test") == "test"


def test_clean_split_with_brackets():
    assert clean_split("train[:100]") == "train"
    assert clean_split("val[0:50]") == "val"


def test_clean_split_with_plus_sign():
    assert clean_split("train+validation") == "train"


def test_make_dataset_splits_single_dataset(simple_dataset):
    """
    dataset is a single Dataset, split is a string (e.g. "train").
    """
    split_aliases = {
        "train": ["train", "training"],
        "test": ["test", "testing"],
        "val": ["val", "validation", "valid", "eval"],
    }
    result = make_dataset_splits(simple_dataset, "train", split_aliases)
    assert result["train"] is simple_dataset
    assert result["val"] is None
    assert result["test"] is None


def test_make_dataset_splits_single_dataset_alias(simple_dataset):
    """
    dataset is a single Dataset, split is a string "training" -> maps to "train".
    """
    split_aliases = {
        "train": ["train", "training"],
        "test": ["test", "testing"],
        "val": ["val", "validation", "valid", "eval"],
    }
    result = make_dataset_splits(simple_dataset, "training[:10]", split_aliases)
    # training[:10] -> clean_split -> "training" -> alias_to_split -> "train"
    assert result["train"] is not None
    assert result["val"] is None
    assert result["test"] is None


def test_make_dataset_splits_single_dataset_plus_sign(simple_dataset):
    """
    dataset is a single Dataset, but we pass a plus-sign in the split => raises ValueError.
    """
    split_aliases = {
        "train": ["train", "training"],
        "test": ["test", "testing"],
        "val": ["val", "validation", "valid", "eval"],
    }
    with pytest.raises(ValueError, match="Split concatenation not supported"):
        _ = make_dataset_splits(None, "train+test", split_aliases)


def test_make_dataset_splits_dataset_dict(simple_dataset_dict):
    """
    dataset is a DatasetDict with both "train" and "validation", "test" inside.
    """
    split_aliases = {
        "train": ["train", "training"],
        "test": ["test", "testing"],
        "val": ["val", "validation", "valid", "eval"],
    }
    result = make_dataset_splits(simple_dataset_dict, None, split_aliases)
    assert result["train"] is not None
    assert result["val"] is not None
    assert result["test"] is not None


def test_make_dataset_splits_list_of_splits(simple_dataset):
    """
    Simulate the scenario in which HF's load_dataset(path, split=[...]) returns multiple Datasets as a list.
    """
    from datasets import DatasetDict

    # Suppose we have a user-supplied list for the split argument: ["train[:1]", "test[:1]"]
    # And we have an actual list of 2 dataset objects:
    ds_list = [simple_dataset, simple_dataset]
    split_aliases = {
        "train": ["train", "training"],
        "test": ["test", "testing"],
        "val": ["val", "validation", "valid", "eval"],
    }
    result = make_dataset_splits(ds_list, ["train[:1]", "test[:1]"], split_aliases)
    assert result["train"] is not None
    assert result["test"] is not None
    assert result["val"] is None


def test_pad_within_micro():
    batch = [
        [10, 20, 30],
        [11, 21],
    ]
    padded = pad_within_micro(batch, pad_token_id=0)
    assert len(padded[0]) == len(padded[1])
    assert padded[1][-1] == 0


def test_batchify():
    single_dim = torch.tensor([1, 2, 3])
    two_dim = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # single input expands dim
    batch1 = batchify(single_dim)
    assert batch1.shape == (1, 3)

    # already 2D stays
    batch2 = batchify(two_dim)
    assert batch2.shape == (2, 3)


def test_extract_key_from_dicts():
    batch = [
        {"foo": 1, "bar": 2},
        {"foo": 3, "bar": 4},
    ]
    foo_vals = extract_key_from_dicts(batch, "foo")
    assert foo_vals == [1, 3]

    bar_vals = extract_key_from_dicts(batch, "bar")
    assert bar_vals == [2, 4]


def test_hfdatamodule_init_from_dataset(simple_dataset):
    """
    Tests initializing HFDatasetDataModule directly from a Dataset object.
    """
    dm = HFDatasetDataModule(path_or_dataset=simple_dataset, split="train")
    # Should produce train=dataset, val=None, test=None
    assert dm.train is not None
    assert dm.val is None
    assert dm.test is None

    # Dataloaders
    train_dl = dm.train_dataloader()
    for batch in train_dl:
        assert "input_ids" in batch
        assert "labels" in batch
        break


def test_hfdatamodule_init_from_datasetdict(simple_dataset_dict):
    """
    Tests initializing HFDatasetDataModule from a DatasetDict object.
    """
    dm = HFDatasetDataModule(path_or_dataset=simple_dataset_dict, split=None)
    # train, val, test should not be None
    assert dm.train is not None
    assert dm.val is not None
    assert dm.test is not None

    # Dataloaders
    train_dl = dm.train_dataloader()
    for batch in train_dl:
        assert "input_ids" in batch
        assert "labels" in batch
        break


def test_hfdatamodule_collate_fn(simple_dataset_dict):
    """
    Test the default collate function from the module.
    """
    dm = HFDatasetDataModule(path_or_dataset=simple_dataset_dict)
    collate_fn = dm._collate_fn

    batch = [
        {"input_ids": [1, 2, 3], "labels": [9]},
        {"input_ids": [4, 5], "labels": [8]},
    ]
    output = collate_fn(batch)
    # Expect 2D for each key
    assert "input_ids" in output and "labels" in output
    assert output["input_ids"].shape == (1, 2, 3) or output["input_ids"].shape == (2, 3)
    # # Some dimension expansions can happen if batchify was used, so shape might be (2, 3).
    # # Also verify padding
    # assert torch.all(output["input_ids"][:, 2] != 0) or torch.all(output["input_ids"][:, -1] != 0)


def test_hfdatamodule_map(simple_dataset_dict):
    """
    Test the `.map()` method: applying a function to all or select splits.
    """
    dm = HFDatasetDataModule(path_or_dataset=simple_dataset_dict)

    def add_one(example):
        example["input_ids"] = [x + 1 for x in example["input_ids"]]
        print(example)
        return example

    # Map to train split only
    dm.map(add_one, split_names="train")
    print(dm.dataset_splits['train'][0])
    # Confirm the first row "input_ids" is incremented
    assert dm.train[0]["input_ids"][0] == 11, dm.train[0]  # was 10 originally in simple_dataset fixture

    # Map to all splits
    dm.map(add_one)  # now modifies train, val, and test
    assert dm.train[0]["input_ids"][0] == 12
    assert dm.val[0]["input_ids"][0] == 11  # val was not previously mapped
    assert dm.test[0]["input_ids"][0] == 11
