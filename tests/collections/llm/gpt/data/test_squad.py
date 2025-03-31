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

from nemo.collections.llm import SquadDataModule


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
def sample_squad_dataset():
    dataset_len = 30
    train_dataset = Dataset.from_dict(
        {
            "context": ["The quick brown fox jumps over the lazy dog."] * dataset_len,
            "question": ["What does the fox do?"] * dataset_len,
            "answers": [{"text": ["jumps over the lazy dog"], "answer_start": [20]} for _ in range(dataset_len)],
            "id": [str(i) for i in range(dataset_len)],
        }
    )

    validation_dataset = Dataset.from_dict(
        {
            "context": ["A computer is an electronic device."] * dataset_len,
            "question": ["What is a computer?"] * dataset_len,
            "answers": [{"text": ["an electronic device"], "answer_start": [15]} for _ in range(dataset_len)],
            "id": [str(i) for i in range(dataset_len)],
        }
    )

    return DatasetDict({'train': train_dataset, 'validation': validation_dataset})


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def squad_data_module(mock_tokenizer, temp_dataset_dir, sample_squad_dataset, mock_trainer, mock_sampler):
    # Patch the _download_data method directly
    with patch.object(SquadDataModule, '_download_data', return_value=sample_squad_dataset):
        with patch('nemo.collections.llm.gpt.data.core.get_dataset_root', return_value=temp_dataset_dir):
            data_module = SquadDataModule(
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


def test_squad_data_module_initialization(squad_data_module):
    assert squad_data_module.seq_length == 512
    assert squad_data_module.micro_batch_size == 2
    assert squad_data_module.global_batch_size == 4
    assert squad_data_module.force_redownload is True
    assert squad_data_module.delete_raw is True


def test_preprocess_and_split_data(squad_data_module, temp_dataset_dir, sample_squad_dataset):
    # Call the preprocessing function
    squad_data_module._preprocess_and_split_data(sample_squad_dataset)

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
        assert data["input"].startswith("Context:")
        assert "Question:" in data["input"]
        assert data["input"].endswith("Answer:")


@pytest.mark.pleasefixme
def test_dataloaders(squad_data_module, mock_trainer):
    squad_data_module.prepare_data()

    train_loader = squad_data_module.train_dataloader()
    val_loader = squad_data_module.val_dataloader()
    test_loader = squad_data_module.test_dataloader()

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)


@pytest.mark.pleasefixme
def test_force_redownload(squad_data_module, temp_dataset_dir):
    # First prepare data
    squad_data_module.prepare_data()

    # Create a marker file to simulate existing data
    marker_file = temp_dataset_dir / "training.jsonl"
    assert marker_file.exists()

    # Store original file stats
    original_mtime = marker_file.stat().st_mtime

    # Set force_redownload to True and prepare again
    squad_data_module.force_redownload = True
    squad_data_module.prepare_data()

    # Check if files were recreated
    assert marker_file.exists()
    new_mtime = marker_file.stat().st_mtime
    assert new_mtime > original_mtime, "File modification time should be newer after redownload"


def test_delete_raw(squad_data_module, temp_dataset_dir, sample_squad_dataset):
    # First prepare data
    squad_data_module.prepare_data()

    # Create a mock raw data file
    raw_file = temp_dataset_dir / "raw_data.txt"
    raw_file.touch()

    # Set delete_raw to True and prepare again
    squad_data_module.delete_raw = True
    squad_data_module._preprocess_and_split_data(sample_squad_dataset)

    # Check if raw file was deleted
    assert not raw_file.exists()
