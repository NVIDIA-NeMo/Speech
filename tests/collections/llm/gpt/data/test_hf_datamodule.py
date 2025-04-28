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

from unittest.mock import MagicMock

import pytest
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

from nemo.collections import llm
from nemo.collections.llm.gpt.data.hf_dataset import (
    SquadHFDataModule,
    batchify,
    extract_key_from_dicts,
    make_dataset_splits,
    pad_within_micro,
)

SQUAD_HF_CACHE = "/home/TestData/lite/hf_cache/squad/"
SQUAD_NEMO_CACHE = "/home/TestData/lite/nemo_cache/squad"


def test_load_single_split():
    ds = llm.HFDatasetDataModule(
        path_or_dataset=SQUAD_HF_CACHE,
        split='train',
        seq_length=512,
        micro_batch_size=2,
    )
    from datasets.arrow_dataset import Dataset

    assert isinstance(ds.dataset_splits, dict)
    assert len(ds.dataset_splits) == 3
    assert 'train' in ds.dataset_splits
    assert ds.dataset_splits['train'] is not None
    assert ds.train is not None
    assert isinstance(ds.dataset_splits['train'], Dataset)
    assert 'val' in ds.dataset_splits
    assert ds.dataset_splits['val'] is None
    assert ds.val is None
    assert 'test' in ds.dataset_splits
    assert ds.dataset_splits['test'] is None
    assert ds.test is None


def test_load_single_split_with_subset():
    ds = llm.HFDatasetDataModule(
        path_or_dataset=SQUAD_HF_CACHE,
        split='train[:10]',
        seq_length=512,
        micro_batch_size=2,
    )
    from datasets.arrow_dataset import Dataset

    assert isinstance(ds.dataset_splits, dict)
    assert len(ds.dataset_splits) == 3
    assert 'train' in ds.dataset_splits
    assert ds.dataset_splits['train'] is not None
    assert ds.train is not None
    assert isinstance(ds.dataset_splits['train'], Dataset)
    assert 'val' in ds.dataset_splits
    assert ds.dataset_splits['val'] is None
    assert ds.val is None
    assert 'test' in ds.dataset_splits
    assert ds.dataset_splits['test'] is None
    assert ds.test is None


def test_load_nonexistent_split():
    exception_msg = ''
    expected_msg = '''Unknown split "this_split_name_should_not_exist". Should be one of ['train', 'validation'].'''
    try:
        llm.HFDatasetDataModule(
            path_or_dataset=SQUAD_HF_CACHE,
            split='this_split_name_should_not_exist',
            seq_length=512,
            micro_batch_size=2,
        )
    except ValueError as e:
        exception_msg = str(e)
    assert exception_msg == expected_msg, exception_msg


def test_load_multiple_split():
    ds = llm.HFDatasetDataModule(
        path_or_dataset=SQUAD_HF_CACHE,
        split=['train', 'validation'],
        seq_length=512,
        micro_batch_size=2,
    )
    from datasets.arrow_dataset import Dataset

    assert isinstance(ds.dataset_splits, dict)
    assert len(ds.dataset_splits) == 3
    assert 'train' in ds.dataset_splits
    assert ds.dataset_splits['train'] is not None
    assert ds.train is not None
    assert isinstance(ds.dataset_splits['train'], Dataset)
    assert isinstance(ds.train, Dataset)
    assert 'val' in ds.dataset_splits
    assert ds.dataset_splits['val'] is not None
    assert ds.val is not None
    assert isinstance(ds.dataset_splits['val'], Dataset)
    assert isinstance(ds.val, Dataset)
    assert 'test' in ds.dataset_splits
    assert ds.dataset_splits['test'] is None
    assert ds.test is None


def test_load_multiple_split_with_subset():
    ds = llm.HFDatasetDataModule(
        path_or_dataset=SQUAD_HF_CACHE,
        split=['train[:100]', 'validation'],
        seq_length=512,
        micro_batch_size=2,
    )
    from datasets.arrow_dataset import Dataset

    assert isinstance(ds.dataset_splits, dict)
    assert len(ds.dataset_splits) == 3
    assert 'train' in ds.dataset_splits
    assert ds.dataset_splits['train'] is not None
    assert ds.train is not None
    assert isinstance(ds.dataset_splits['train'], Dataset)
    assert isinstance(ds.train, Dataset)
    assert 'val' in ds.dataset_splits
    assert ds.dataset_splits['val'] is not None
    assert ds.val is not None
    assert isinstance(ds.dataset_splits['val'], Dataset)
    assert isinstance(ds.val, Dataset)
    assert 'test' in ds.dataset_splits
    assert ds.dataset_splits['test'] is None
    assert ds.test is None


def test_validate_dataset_asset_accessibility_file_does_not_exist():
    raised_exception = False
    try:
        llm.HFDatasetDataModule(
            path_or_dataset="/this/path/should/not/exist/",
            seq_length=512,
            micro_batch_size=2,
        )
    except FileNotFoundError:
        raised_exception = True

    assert raised_exception == True, "Expected to raise a FileNotFoundError"


def test_validate_dataset_asset_accessibility_file_is_none():
    exception_msg = ''
    expected_msg = "Expected `path_or_dataset` to be str, Dataset, DatasetDict, but got <class 'NoneType'>"
    try:
        llm.HFDatasetDataModule(
            path_or_dataset=None,
            seq_length=512,
            micro_batch_size=2,
        )
    except ValueError as e:
        exception_msg = str(e)

    assert exception_msg == expected_msg, exception_msg


@pytest.fixture
def sample_dataset():
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "id": ["1", "2"],
                    "title": ["Title1", "Title2"],
                    "context": ["This is a context.", "Another context."],
                    "question": ["What is this?", "What about this?"],
                    "answers": [{"text": ["A context"]}, {"text": ["Another"]}],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "id": ["3"],
                    "title": ["Title3"],
                    "context": ["Validation context."],
                    "question": ["What is validation?"],
                    "answers": [{"text": ["Validation answer"]}],
                }
            ),
        }
    )


@pytest.fixture
def data_module(sample_dataset):
    return llm.HFDatasetDataModule(path_or_dataset=sample_dataset, split=["train", "validation"])


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.text_to_ids.side_effect = lambda text: [ord(c) for c in text]  # Mock character-based token IDs
    tokenizer.bos_id = 1
    tokenizer.eos_id = 2
    return tokenizer


@pytest.fixture
def squad_data_module(mock_tokenizer, sample_dataset):
    return SquadHFDataModule(tokenizer=mock_tokenizer, path_or_dataset=sample_dataset, split=["train", "validation"])


def test_dataset_splits(data_module):
    assert data_module.train is not None
    assert data_module.val is not None
    assert data_module.test is None  # No test split in sample dataset


def test_dataloaders(data_module):
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)


def test_formatting_prompts_func(squad_data_module):
    example = {"context": "This is a context.", "question": "What is this?", "answers": {"text": ["A context"]}}
    result = squad_data_module.formatting_prompts_func(example)

    assert "input_ids" in result
    assert "labels" in result
    assert "loss_mask" in result
    assert len(result["input_ids"]) == len(result["labels"])


def test_make_dataset_splits_single_dataset():
    data = {"id": [1, 2, 3], "text": ["a", "b", "c"]}
    dataset = Dataset.from_dict(data)
    split_aliases = {"train": ["train"], "val": ["validation"], "test": ["test"]}

    result = make_dataset_splits(dataset, "train", split_aliases)

    assert result["train"] is not None
    assert result["val"] is None
    assert result["test"] is None
    assert len(result["train"]) == 3


def test_make_dataset_splits_dataset_dict():
    data_train = Dataset.from_dict({"id": [1, 2, 3], "text": ["a", "b", "c"]})
    data_val = Dataset.from_dict({"id": [4, 5], "text": ["d", "e"]})
    dataset = DatasetDict({"train": data_train, "validation": data_val})
    split_aliases = {"train": ["train"], "val": ["validation"], "test": ["test"]}

    result = make_dataset_splits(dataset, None, split_aliases)

    assert result["train"] is not None
    assert result["val"] is not None
    assert result["test"] is None
    assert len(result["train"]) == 3
    assert len(result["val"]) == 2


def test_make_dataset_splits_invalid_split():
    data = {"id": [1, 2, 3], "text": ["a", "b", "c"]}
    dataset = Dataset.from_dict(data)
    split_aliases = {"train": ["train"], "val": ["validation"], "test": ["test"]}

    with pytest.raises(KeyError):
        make_dataset_splits(dataset, "invalid_split", split_aliases)


def test_make_dataset_splits_with_list():
    data_train = Dataset.from_dict({"id": [1, 2, 3], "text": ["a", "b", "c"]})
    data_val = Dataset.from_dict({"id": [4, 5], "text": ["d", "e"]})
    dataset = [data_train, data_val]
    split_aliases = {"train": ["train"], "val": ["validation"], "test": ["test"]}

    result = make_dataset_splits(dataset, ["train", "validation"], split_aliases)

    assert result["train"] is not None
    assert result["val"] is not None
    assert result["test"] is None
    assert len(result["train"]) == 3
    assert len(result["val"]) == 2


def test_collate_fn():
    # Create a minimal HFDatasetDataModule instance with required parameters
    dm = llm.HFDatasetDataModule(
        path_or_dataset=Dataset.from_dict({"dummy": [0]}),  # Minimal dummy dataset
        split="train",
    )
    batch = [{"id": [1], "token_ids": [1, 2, 3]}, {"id": [2], "token_ids": [123]}]
    result = dm.collate_fn(batch)
    assert isinstance(result, dict)
    assert "id" in result
    assert "token_ids" in result
    assert isinstance(result["id"], torch.Tensor)
    assert result["id"].ndim == 2
    assert result["id"].shape[0] == 2
    assert result["id"].shape[1] == 1
    assert isinstance(result["token_ids"], torch.Tensor)
    assert result["token_ids"].ndim == 2
    assert result["token_ids"].shape[0] == 2
    assert result["token_ids"].shape[1] == 3


def test_batchify():
    batch = torch.Tensor(128)
    output = batchify(batch)
    assert isinstance(output, torch.Tensor)
    assert output.ndim == 2
    assert output.shape[0] == 1
    assert output.shape[1] == 128


def test_extract_key_from_dicts():
    dicts = [{"key": "value1"}, {"key": "value2"}, {"key": "value3"}]
    result = extract_key_from_dicts(dicts, "key")
    assert result == ["value1", "value2", "value3"]


def test_pad_within_micro():
    data = [[1, 2], [3, 4, 5], [6]]
    padded_data = pad_within_micro(data, pad_token_id=0)
    assert len(padded_data) == 3
    assert all(len(row) == 3 for row in padded_data)
    assert padded_data[0] == [1, 2, 0]
    assert padded_data[1] == [3, 4, 5]
    assert padded_data[2] == [6, 0, 0]


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.bos_id = 1
    tokenizer.eos_id = 2

    def mock_text_to_ids(text):
        # Return a deterministic list of token ids for the test
        return [1, 11, 22, 2]

    tokenizer.text_to_ids.side_effect = mock_text_to_ids
    return tokenizer


@pytest.fixture
def mock_trainer():
    """Mock Trainer object that can provide max_steps attribute."""
    trainer = MagicMock()
    trainer.max_steps = 42  # Example value; adjust as needed
    return trainer


def test_squad_data_module_no_download(mock_trainer):
    """Test that SquadDataModule uses the dataset_root path and does not download data."""
    data_module = llm.SquadDataModule(dataset_root=SQUAD_NEMO_CACHE, force_redownload=False, delete_raw=False)
    data_module.trainer = mock_trainer
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Verify it used the mock path
    assert str(data_module.dataset_root) == SQUAD_NEMO_CACHE


def test_squad_data_module_with_pth_dataloader_init(mock_tokenizer, mock_trainer):
    """
    Test that SquadDataModuleWithPthDataloader can be instantiated without errors, and that it
    creates a PyTorch DataLoader (not some other type)."""

    class SquadDataModuleWithPthDataloader(llm.SquadDataModule):
        """Creates a squad dataset with a PT dataloader"""

        def _create_dataloader(self, dataset, mode, **kwargs):
            return DataLoader(
                dataset,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=dataset.collate_fn,
                batch_size=self.micro_batch_size,
                **kwargs,
            )

    dm = SquadDataModuleWithPthDataloader(
        dataset_root=SQUAD_NEMO_CACHE,
        tokenizer=mock_tokenizer,
        seq_length=512,
        micro_batch_size=4,
        num_workers=0,
        force_redownload=False,
        delete_raw=False,
        dataset_kwargs={
            "pad_to_max_length": True,
            "sanity_check_dist_workers": False,
        },
    )
    # Verify it used the mock path
    assert str(dm.dataset_root) == SQUAD_NEMO_CACHE

    # Check type
    assert isinstance(dm, SquadDataModuleWithPthDataloader)
    dm.trainer = mock_trainer
    # print(dm.trainer.max_steps)
    # Typically you call dm.prepare_data() + dm.setup()
    # But since it's a small example, we can just ensure it doesn't raise errors.
    dm.prepare_data()
    dm.setup('fit')

    # # Make sure we get a torch DataLoader back
    train_loader = dm.train_dataloader()
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    # Optionally, check if we can actually fetch a single batch
    batch = next(iter(train_loader), None)
    # Depending on your environment, that might be empty or a dict
    # Check that the batch is not None
    assert batch is not None
    expected_batch = {
        'tokens': [4, 512],
        'labels': [4, 512],
        'loss_mask': [4, 512],
        'position_ids': [4, 512],
        'contexts': [4, 512],
        'context_lengths': [4],
        'answers': [4, 512],
        'metadata': 4,
        'token_count': 4,
        'attention_mask': [4, 1, 512, 512],
    }
    assert isinstance(batch, dict)
    for key, val in expected_batch.items():
        batch_val = batch.pop(key)
        if isinstance(batch_val, list):
            assert len(batch_val) == val, (key, val, batch_val)
        else:
            assert list(batch_val.size()) == val, (key, val, batch_val)
    assert len(batch) == 0


@pytest.fixture
def sample_hf_dataset():
    data = {
        "id": [1, 2, 3],
        "title": ["title1", "title2", "title3"],
        "context": ["context1", "context2", "context3"],
        "question": ["question1", "question2", "question3"],
        "answers": [["ans1"], ["ans2"], ["ans3"]],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def sample_hf_dataset_dict(sample_hf_dataset):
    return DatasetDict(
        {
            "train": sample_hf_dataset,
            "validation": sample_hf_dataset,  # alias -> val in our clean_split
        }
    )


@pytest.fixture
def split_aliases():
    return {
        "train": ["train"],
        "val": ["validation", "val"],
        "test": ["test"],
    }


def test_single_dataset_str_split(sample_hf_dataset, split_aliases):
    """Covers the branch: dataset is Dataset, split is str."""
    splits = make_dataset_splits(dataset=sample_hf_dataset, split="train", split_aliases=split_aliases)
    assert splits["train"] is not None
    assert splits["val"] is None
    assert splits["test"] is None


def test_dataset_dict_splits(sample_hf_dataset_dict, split_aliases):
    """Covers the branch: dataset is DatasetDict."""
    splits = make_dataset_splits(
        dataset=sample_hf_dataset_dict, split=None, split_aliases=split_aliases  # Not used in this branch
    )
    assert splits["train"] is not None
    assert splits["val"] is not None
    assert splits["test"] is None


def test_list_dataset_splits(sample_hf_dataset, split_aliases):
    """Covers the branch: dataset is list, split is list."""
    # Suppose we have two splits in a list
    dataset_list = [sample_hf_dataset, sample_hf_dataset]
    splits_list = ["train", "validation"]  # alias will map validation -> val
    splits = make_dataset_splits(dataset=dataset_list, split=splits_list, split_aliases=split_aliases)
    assert splits["train"] is not None
    assert splits["val"] is not None
    assert splits["test"] is None


def test_single_dataset_with_plus_sign(sample_hf_dataset, split_aliases):
    """Covers the branch that raises ValueError if '+' in split string."""
    splits = make_dataset_splits(dataset=sample_hf_dataset, split="train+val", split_aliases=split_aliases)
    assert splits["train"] is not None
    assert splits["val"] is None
    assert splits["test"] is None


def test_single_dataset_with_bracket(sample_hf_dataset, split_aliases):
    """
    Covers the branch that strips bracket notation
    and uses the cleaned alias (e.g. 'validation[10:20]' -> 'validation').
    """
    splits = make_dataset_splits(dataset=sample_hf_dataset, split="validation[10:20]", split_aliases=split_aliases)
    assert splits["val"] is not None
    assert splits["train"] is None
    assert splits["test"] is None


def test_invalid_split_type_raises(sample_hf_dataset, split_aliases):
    with pytest.raises(AssertionError, match="Expected split to be a string, but got <class 'int'>"):
        make_dataset_splits(dataset=sample_hf_dataset, split=123, split_aliases=split_aliases)  # Invalid type


def test_no_alias_match_raises(sample_hf_dataset_dict, split_aliases):
    """If alias in the dataset dict isn't in split_aliases, we get a KeyError."""
    # Make a dataset dict with a split name not in our alias map
    wrong_dataset_dict = DatasetDict({"custom_split": sample_hf_dataset_dict["train"]})
    with pytest.raises(KeyError):
        make_dataset_splits(dataset=wrong_dataset_dict, split=None, split_aliases=split_aliases)
