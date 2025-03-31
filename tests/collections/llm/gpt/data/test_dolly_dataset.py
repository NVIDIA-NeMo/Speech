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

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset, DatasetDict

from nemo.collections.llm import DollyDataModule


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.text_to_ids.side_effect = lambda text: [ord(c) for c in text]  # Mock character-based token IDs
    tokenizer.bos_id = 1
    tokenizer.eos_id = 2
    return tokenizer


@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    trainer.global_step = 0
    trainer.max_steps = 1000
    return trainer


@pytest.fixture
def mock_sampler():
    sampler = MagicMock()
    sampler.init_global_step = 0
    return sampler


@pytest.fixture
def sample_dolly_dataset():
    dataset_len = 30
    dataset = Dataset.from_dict(
        {
            "context": ["Process mining is a family of techniques..."] * dataset_len,
            "instruction": ["What is process mining?"] * dataset_len,
            "response": ["Process mining brings together data mining."] * dataset_len,
            "category": ["information_extraction"] * dataset_len,
        }
    )
    return DatasetDict({'train': dataset})


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def dolly_data_module(mock_tokenizer, temp_dataset_dir, sample_dolly_dataset, mock_trainer, mock_sampler):
    # Patch the _download_data method directly
    with patch.object(DollyDataModule, '_download_data', return_value=sample_dolly_dataset):
        with patch('nemo.collections.llm.gpt.data.core.get_dataset_root', return_value=temp_dataset_dir):
            data_module = DollyDataModule(
                tokenizer=mock_tokenizer,
                seq_length=512,
                micro_batch_size=2,
                global_batch_size=4,
                force_redownload=True,
            )
            data_module.dataset_root = temp_dataset_dir
            data_module.trainer = mock_trainer
            data_module.data_sampler = mock_sampler

            yield data_module


def test_dolly_data_module_initialization(dolly_data_module):
    assert dolly_data_module.seq_length == 512
    assert dolly_data_module.micro_batch_size == 2
    assert dolly_data_module.global_batch_size == 4
    assert dolly_data_module.force_redownload is True
    assert dolly_data_module.delete_raw is True


def test_preprocess_and_split_data(dolly_data_module, temp_dataset_dir, sample_dolly_dataset):

    # Call the preprocessing function
    dolly_data_module._preprocess_and_split_data(sample_dolly_dataset)

    # Check if files were created
    assert (temp_dataset_dir / "training.jsonl").exists()
    assert (temp_dataset_dir / "validation.jsonl").exists()
    assert (temp_dataset_dir / "test.jsonl").exists()

    # Check content of training file
    with open(temp_dataset_dir / "training.jsonl", "r") as f:
        lines = f.readlines()
        assert len(lines) > 0
        data = json.loads(lines[0])
        assert "input" in data
        assert "output" in data


def test_dataloaders(dolly_data_module, mock_trainer):
    dolly_data_module.prepare_data()

    train_loader = dolly_data_module.train_dataloader()
    val_loader = dolly_data_module.val_dataloader()
    test_loader = dolly_data_module.test_dataloader()

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)


def test_force_redownload(dolly_data_module, temp_dataset_dir):
    # First prepare data
    dolly_data_module.prepare_data()

    # Create a marker file to simulate existing data
    marker_file = temp_dataset_dir / "training.jsonl"
    assert marker_file.exists()

    # Store original file stats and content
    original_mtime = marker_file.stat().st_mtime
    with open(marker_file, "r") as f:
        f.read()

    # Set force_redownload to True and prepare again
    dolly_data_module.force_redownload = True
    dolly_data_module.prepare_data()

    # Check if files were recreated
    assert marker_file.exists()

    # Verify the file is different
    new_mtime = marker_file.stat().st_mtime
    assert new_mtime > original_mtime, "File modification time should be newer after redownload"


def test_delete_raw(dolly_data_module, temp_dataset_dir, sample_dolly_dataset):
    # First prepare data
    dolly_data_module.prepare_data()

    # Create a mock raw data file
    raw_file = temp_dataset_dir / "raw_data.txt"
    raw_file.touch()

    # Set delete_raw to True and prepare again
    dolly_data_module.delete_raw = True
    dolly_data_module._preprocess_and_split_data(sample_dolly_dataset)

    # Check if raw file was deleted
    assert not raw_file.exists()
