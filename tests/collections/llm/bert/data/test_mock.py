# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.llm.bert.data.mock import BERTMockDataModule, _MockBERTDataset


@pytest.fixture
def mock_tokenizer():
    """
    Provides a mock tokenizer that simulates BERT tokenizer behavior.
    """
    tokenizer = MagicMock()
    tokenizer.tokenizer = tokenizer
    tokenizer.vocab_size = 1024
    return tokenizer


@pytest.fixture
def mock_data_module():
    """Fixture to create and set up a mock BERT data module."""
    dm = BERTMockDataModule(
        seq_length=16,
        micro_batch_size=2,
        global_batch_size=4,
        num_train_samples=20,
        num_val_samples=5,
        num_test_samples=5,
    )
    dm.setup()
    return dm


def test_mock_bert_dataset_length(mock_tokenizer):
    """Ensure the dataset's length matches the configured number of samples."""
    ds = _MockBERTDataset(
        tokenizer=mock_tokenizer,
        name="train",
        num_samples=100,
        seq_length=16,
    )
    assert len(ds) == 100


def test_mock_bert_dataset_item_shapes(mock_tokenizer):
    """Check that a sample has the expected keys and shapes."""
    seq_length = 16
    ds = _MockBERTDataset(
        tokenizer=mock_tokenizer,
        name="train",
        num_samples=1,
        seq_length=seq_length,
    )
    sample = ds[0]

    expected_keys = {"text", "types", "labels", "is_random", "padding_mask", "loss_mask", "truncated"}
    assert set(sample.keys()) == expected_keys

    assert len(sample["text"]) == seq_length
    assert len(sample["types"]) == seq_length
    assert len(sample["labels"]) == seq_length
    assert len(sample["padding_mask"]) == seq_length
    assert len(sample["loss_mask"]) == seq_length
    assert isinstance(sample["is_random"], int)
    assert isinstance(sample["truncated"], int)


def test_data_module_train_dataloader(mock_data_module):
    """Check the train dataloader returns batches of the expected shape."""
    train_dl = mock_data_module.train_dataloader()
    batch = next(iter(train_dl))

    assert isinstance(batch, dict)
    expected_keys = {"text", "types", "labels", "is_random", "padding_mask", "loss_mask", "truncated"}
    assert set(batch.keys()) == expected_keys

    # Check shapes (assuming batch_size=1 after DataLoader collation)
    assert batch["text"].shape == torch.Size([1, 16])
    assert batch["types"].shape == torch.Size([1, 16])
    assert batch["labels"].shape == torch.Size([1, 16])
    assert batch["padding_mask"].shape == torch.Size([1, 16])
    assert batch["loss_mask"].shape == torch.Size([1, 16])


def test_data_module_val_dataloader(mock_data_module):
    """Check the val dataloader returns a non-empty dataset."""
    val_dl = mock_data_module.val_dataloader()
    val_batch = next(iter(val_dl))
    assert val_batch["text"].shape == torch.Size([1, 16])


def test_data_module_test_dataloader(mock_data_module):
    """Check the test dataloader returns a non-empty dataset."""
    test_dl = mock_data_module.test_dataloader()
    test_batch = next(iter(test_dl))
    assert test_batch["text"].shape == torch.Size([1, 16])


def test_mock_bert_dataset_deterministic(mock_tokenizer):
    """Test that the dataset generates deterministic outputs for the same seed."""
    ds1 = _MockBERTDataset(
        tokenizer=mock_tokenizer,
        name="train",
        num_samples=10,
        seq_length=16,
        seed=42,
    )

    ds2 = _MockBERTDataset(
        tokenizer=mock_tokenizer,
        name="train",
        num_samples=10,
        seq_length=16,
        seed=42,
    )

    # Check if same index returns same data
    sample1 = ds1[0]
    sample2 = ds2[0]
    assert torch.equal(sample1["text"], sample2["text"])
    assert torch.equal(sample1["labels"], sample2["labels"])


def test_data_module_batch_sizes(mock_tokenizer):
    """Test that the data module handles different batch sizes correctly."""
    dm = BERTMockDataModule(
        seq_length=16,
        tokenizer=mock_tokenizer,
        micro_batch_size=2,
        global_batch_size=8,
        num_train_samples=20,
    )

    assert dm.micro_batch_size == 2
    assert dm.global_batch_size == 8

    # Test with rampup_batch_size
    dm_rampup = BERTMockDataModule(
        seq_length=16,
        tokenizer=mock_tokenizer,
        micro_batch_size=2,
        global_batch_size=8,
        rampup_batch_size=[4, 2, 1000],
        num_train_samples=20,
    )

    assert dm_rampup.data_sampler is not None
