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

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict

from nemo.collections.llm.gpt.data.mlperf_govreport import MLPerfGovReportDataModule


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.text_to_ids.side_effect = lambda text: [ord(c) for c in text]
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
def sample_govreport_dataset():
    dataset_len = 30
    dataset = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4, 5]] * dataset_len, "labels": [[1, 2, -100, 4, 5]] * dataset_len}
    )
    return DatasetDict({'train': dataset, 'validation': dataset})


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def govreport_data_module(mock_tokenizer, temp_dataset_dir, sample_govreport_dataset, mock_trainer, mock_sampler):
    # Patch the _download_data method directly
    with patch.object(MLPerfGovReportDataModule, '_download_data', return_value=sample_govreport_dataset):
        with patch('nemo.collections.llm.gpt.data.core.get_dataset_root', return_value=temp_dataset_dir):
            mock_packed_sequence_specs = MagicMock(
                packed_sequence_size=2048, pad_cu_seqlens=False  # Disable pad_cu_seqlens to avoid metadata requirement
            )

            data_module = MLPerfGovReportDataModule(
                tokenizer=mock_tokenizer,
                seq_length=2048,
                micro_batch_size=1,
                global_batch_size=4,
                force_redownload=True,
                packed_sequence_specs=mock_packed_sequence_specs,
            )
            data_module.dataset_root = temp_dataset_dir
            data_module.trainer = mock_trainer
            data_module.data_sampler = mock_sampler

            yield data_module


def test_govreport_data_module_initialization(govreport_data_module):
    assert govreport_data_module.seq_length == 2048
    assert govreport_data_module.micro_batch_size == 1
    assert govreport_data_module.global_batch_size == 4
    assert govreport_data_module.force_redownload is True
    assert govreport_data_module.delete_raw is True


def test_preprocess_and_split_data(govreport_data_module, temp_dataset_dir, sample_govreport_dataset):
    # Call the preprocessing function
    govreport_data_module._preprocess_and_split_data(sample_govreport_dataset)

    # Check if files were created
    assert (temp_dataset_dir / "training.npy").exists()
    assert (temp_dataset_dir / "validation.npy").exists()
    assert (temp_dataset_dir / "test.npy").exists()

    # Check content of training file
    data = np.load(temp_dataset_dir / "training.npy", allow_pickle=True)
    assert len(data) > 0
    assert "input_ids" in data[0]
    assert "loss_mask" in data[0]
    assert "seq_start_id" in data[0]
    assert isinstance(data[0]["loss_mask"], list)
    assert data[0]["seq_start_id"] == [0]


def test_prepare_data(govreport_data_module, temp_dataset_dir):
    govreport_data_module.prepare_data()

    # Check if files were created
    assert (temp_dataset_dir / "training.npy").exists()
    assert (temp_dataset_dir / "validation.npy").exists()
    assert (temp_dataset_dir / "test.npy").exists()


def test_dataloaders(govreport_data_module, mock_trainer):
    govreport_data_module.prepare_data()

    train_loader = govreport_data_module.train_dataloader()
    val_loader = govreport_data_module.val_dataloader()
    test_loader = govreport_data_module.test_dataloader()

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)


def test_force_redownload(govreport_data_module, temp_dataset_dir):
    # First prepare data
    govreport_data_module.prepare_data()

    # Create a marker file to simulate existing data
    marker_file = temp_dataset_dir / "training.npy"
    assert marker_file.exists()

    # Store original file stats
    original_mtime = marker_file.stat().st_mtime

    # Set force_redownload to True and prepare again
    govreport_data_module.force_redownload = True
    govreport_data_module.prepare_data()

    # Check if files were recreated
    assert marker_file.exists()

    # Verify the file is different
    new_mtime = marker_file.stat().st_mtime
    assert new_mtime > original_mtime, "File modification time should be newer after redownload"


def test_delete_raw(govreport_data_module, temp_dataset_dir, sample_govreport_dataset):
    # First prepare data
    govreport_data_module.prepare_data()

    # Create a mock raw data file
    raw_file = temp_dataset_dir / "raw_data.txt"
    raw_file.touch()

    # Set delete_raw to True and prepare again
    govreport_data_module.delete_raw = True
    govreport_data_module._preprocess_and_split_data(sample_govreport_dataset)

    # Check if raw file was deleted
    assert not raw_file.exists()


def test_invalid_packed_sequence_size():
    with pytest.raises(ValueError):
        MLPerfGovReportDataModule(seq_length=2048, packed_sequence_specs=MagicMock(packed_sequence_size=1024))
