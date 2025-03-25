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

from datasets import Dataset

from nemo.collections.llm.gpt.data.chat import ChatDataModule


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
def sample_chat_dataset():
    return Dataset.from_dict(
        {
            "conversations": [
                [
                    {"from": "human", "value": "Hello, how are you?"},
                    {"from": "assistant", "value": "I'm doing well, thank you! How can I help you today?"},
                ],
                [
                    {"from": "human", "value": "What's the weather like?"},
                    {"from": "assistant", "value": "I don't have access to real-time weather information."},
                ],
            ]
        }
    )


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def chat_data_module(mock_tokenizer, temp_dataset_dir):
    with patch('nemo.collections.llm.gpt.data.core.create_sft_dataset') as mock_create_dataset:
        mock_create_dataset.return_value = MagicMock()
        data_module = ChatDataModule(
            tokenizer=mock_tokenizer,
            seq_length=512,
            micro_batch_size=2,
            global_batch_size=4,
            dataset_root=temp_dataset_dir,
        )
        return data_module


def test_chat_data_module_initialization(chat_data_module):
    assert chat_data_module.seq_length == 512
    assert chat_data_module.micro_batch_size == 2
    assert chat_data_module.global_batch_size == 4


def test_create_dataset(chat_data_module, temp_dataset_dir):
    # Create a sample chat dataset file
    dataset_path = temp_dataset_dir / "chat_dataset.jsonl"
    with open(dataset_path, "w") as f:
        json.dump(
            {"conversations": [[{"from": "human", "value": "Hello"}, {"from": "assistant", "value": "Hi there!"}]]}, f
        )

    # Test dataset creation
    dataset = chat_data_module._create_dataset(str(dataset_path))
    assert dataset is not None

    # Test with is_test=True
    test_dataset = chat_data_module._create_dataset(str(dataset_path), is_test=True)
    assert test_dataset is not None
