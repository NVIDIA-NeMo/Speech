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


from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo.collections.llm.gpt.data.hf_dataset import HFMockDataModule, _MockGPTDataset


@pytest.fixture
def mock_tokenizer():
    """
    Provides a mock tokenizer that simulates huggingface's tokenizer behavior.
    """
    tokenizer = MagicMock()
    tokenizer.tokenizer = tokenizer
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = "<|endoftext|>"

    def tokenizer_call(text, max_length, truncation=True):
        return {
            "input_ids": [[101, 102, 103]],
            "attention_mask": [[1, 1, 1]],
        }

    # Replace __call__ with a MagicMock and set the side_effect
    tokenizer.__call__ = MagicMock(side_effect=tokenizer_call)
    tokenizer.vocab_size = 1024
    return tokenizer


@pytest.fixture
def mock_data_module():
    """Fixture to create and set up a mock data module."""
    dm = HFMockDataModule(
        seq_length=16,
        micro_batch_size=2,
        num_train_samples=20,
        num_val_samples=5,
        num_test_samples=5,
        vocab_size=1024,
        create_attention_mask=True,
    )
    dm.setup()
    return dm


def test_mock_gpt_dataset_length(mock_tokenizer):
    """Ensure the dataset's length matches the configured number of samples."""
    ds = _MockGPTDataset(
        vocab_size=1024,
        name="train",
        num_samples=100,
        seq_length=16,
        create_attention_mask=False,
    )
    assert len(ds) == 100


def test_mock_gpt_dataset_item_shapes(mock_tokenizer):
    """Check that a sample has the expected keys and shapes."""
    seq_length = 16
    ds = _MockGPTDataset(
        vocab_size=1024,
        name="train",
        num_samples=1,
        seq_length=seq_length,
        create_attention_mask=True,
    )
    sample = ds[0]
    assert "input_ids" in sample
    assert "labels" in sample
    assert "loss_mask" in sample
    assert "position_ids" in sample
    assert "attention_mask" in sample
    assert len(sample["input_ids"]) == seq_length
    assert len(sample["labels"]) == seq_length
    assert len(sample["loss_mask"]) == seq_length
    assert len(sample["position_ids"]) == seq_length
    # Attention mask is [1, seq_length, seq_length] if used
    assert len(sample["attention_mask"]) == 1
    assert len(sample["attention_mask"][0]) == seq_length
    assert len(sample["attention_mask"][0][0]) == seq_length


def test_data_module_train_dataloader(mock_data_module):
    """Check the train dataloader returns batches of the expected shape."""
    train_dl = mock_data_module.train_dataloader()
    batch = next(iter(train_dl))
    assert isinstance(batch, dict)
    assert set(["input_ids", "labels", "loss_mask", "position_ids"]).issubset(batch.keys())
    assert batch["input_ids"].shape == torch.Size([2, 16])
    assert batch["labels"].shape == torch.Size([2, 16])
    # Attention mask may be optional, check if included
    if "attention_mask" in batch:
        # Should be [2, 1, seq_length, seq_length]
        assert batch["attention_mask"].shape == torch.Size([2, 1, 16, 16])


def test_data_module_val_dataloader(mock_data_module):
    """Check the val dataloader returns a non-empty dataset."""
    val_dl = mock_data_module.val_dataloader()
    val_batch = next(iter(val_dl))
    assert val_batch["input_ids"].shape == torch.Size([2, 16])


def test_data_module_test_dataloader(mock_data_module):
    """Check the test dataloader returns a non-empty dataset."""
    test_dl = mock_data_module.test_dataloader()
    test_batch = next(iter(test_dl))
    assert test_batch["input_ids"].shape == torch.Size([2, 16])
