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
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset, DatasetDict

from nemo.collections.llm.gpt.data.reranker import CustomReRankerDataModule, ReRankerDataset, SpecterReRankerDataModule


@pytest.fixture
def sample_data():
    return [
        {
            "question": "What is NeMo?",
            "pos_doc": ["NeMo is NVIDIA's framework for conversational AI"],  # Always use list
            "neg_doc": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3", "Wrong answer 4", "Wrong answer 5"],
        },
        {
            "question": "What is PyTorch?",
            "pos_doc": ["PyTorch is a machine learning framework"],  # Always use list
            "neg_doc": [
                "Incorrect PyTorch description 1",
                "Incorrect PyTorch description 2",
                "Incorrect PyTorch description 3",
                "Incorrect PyTorch description 4",
            ],
        },
    ] * 20


@pytest.fixture
def temp_data_files(sample_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create main data file
        main_file = Path(tmpdir) / "train.json"
        with open(main_file, 'w') as f:
            json.dump(sample_data, f)

        # Create validation data file with consistent format
        val_data = [
            {
                "question": "What is NeMo?",
                "pos_doc": ["NeMo is NVIDIA's framework for conversational AI"],
                "neg_doc": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3", "Wrong answer 4"],
            }
        ]
        val_file = Path(tmpdir) / "val.json"
        with open(val_file, 'w') as f:
            json.dump(val_data, f)

        yield str(main_file), str(val_file)


class MockTokenizer:
    def __init__(self):
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0

    def text_to_ids(self, text):
        # Simple mock implementation that converts text to a list of integers
        return [ord(c) % 10 for c in text]


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


def test_reranker_dataset_initialization(temp_data_files, mock_tokenizer):
    main_file, _ = temp_data_files
    dataset = ReRankerDataset(
        file_path=main_file,
        tokenizer=mock_tokenizer,
        max_seq_length=128,
        num_hard_negatives=4,
        negative_sample_strategy="first",
    )
    assert dataset.max_seq_length == 128
    assert dataset.num_hard_negatives == 4
    assert dataset.negative_sample_strategy == "first"
    assert dataset.question_key == "question"
    assert dataset.pos_key == "pos_doc"
    assert dataset.neg_key == "neg_doc"


def test_reranker_dataset_getitem(temp_data_files, mock_tokenizer):
    main_file, _ = temp_data_files
    dataset = ReRankerDataset(
        file_path=main_file,
        tokenizer=mock_tokenizer,
        max_seq_length=128,
        num_hard_negatives=4,
        negative_sample_strategy="first",
    )

    item = dataset[0]
    assert "positive" in item
    assert "negatives" in item
    assert "metadata" in item
    assert len(item["negatives"]) == 4  # num_hard_negatives
    assert isinstance(item["positive"], list)
    assert all(isinstance(n, list) for n in item["negatives"])


def test_reranker_dataset_collate_fn(temp_data_files, mock_tokenizer):
    main_file, _ = temp_data_files
    dataset = ReRankerDataset(
        file_path=main_file,
        tokenizer=mock_tokenizer,
        max_seq_length=128,
        num_hard_negatives=4,
        negative_sample_strategy="first",
    )

    batch = [dataset[0], dataset[1]]
    collated = dataset.collate_fn(batch)

    assert "input_ids" in collated
    assert "attention_mask" in collated
    assert "token_type_ids" in collated
    assert "position_ids" in collated
    assert "metadata" in collated

    # Check shapes
    batch_size = len(batch)
    num_examples = batch_size * (1 + dataset.num_hard_negatives)  # positive + negatives
    assert collated["input_ids"].shape[0] == num_examples
    assert collated["attention_mask"].shape[0] == num_examples
    assert collated["token_type_ids"].shape[0] == num_examples
    assert collated["position_ids"].shape[0] == batch_size


def test_reranker_datamodule_initialization(temp_data_files):
    main_file, val_file = temp_data_files
    data_module = CustomReRankerDataModule(
        data_root=main_file,
        val_root=val_file,
        seq_length=128,
        micro_batch_size=2,
        dataset_kwargs={"num_hard_negatives": 4},
    )
    assert data_module.val_ratio == 0
    assert data_module.train_ratio == 0.99
    assert data_module.query_key == "question"
    assert data_module.pos_doc_key == "pos_doc"
    assert data_module.neg_doc_key == "neg_doc"


def test_reranker_datamodule_multiple_data_roots(sample_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two data files
        file1 = Path(tmpdir) / "data1.json"
        file2 = Path(tmpdir) / "data2.json"
        val_ratio = 0.05
        test_ratio = 0.05
        train_ratio = 1 - val_ratio - test_ratio

        with open(file1, 'w') as f:
            json.dump(sample_data, f)
        with open(file2, 'w') as f:
            json.dump(sample_data, f)

        data_module = CustomReRankerDataModule(
            data_root=[str(file1), str(file2)],
            seq_length=128,
            micro_batch_size=2,
            force_redownload=True,
            test_ratio=0.05,
            val_ratio=0.05,
            dataset_kwargs={"num_hard_negatives": 4},
        )
        data_module.prepare_data()

        # Verify that data from both files was combined
        with open(data_module.dataset_root / "training.jsonl", 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2 * train_ratio * len(sample_data)


@pytest.mark.parametrize(
    "val_ratio,test_ratio",
    [
        (0.3, 0.2),
        (0.0, 0.5),
        (0.5, 0.0),
    ],
)
def test_reranker_datamodule_different_split_ratios(temp_data_files, val_ratio, test_ratio):
    main_file, additional_file = temp_data_files

    if val_ratio == 0.0:
        val_file = additional_file
    else:
        val_file = None

    if test_ratio == 0.0:
        test_file = additional_file
    else:
        test_file = None

    data_module = CustomReRankerDataModule(
        data_root=main_file,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seq_length=128,
        micro_batch_size=2,
        force_redownload=True,
        val_root=val_file,
        test_root=test_file,
        dataset_kwargs={"num_hard_negatives": 4},
    )
    data_module.prepare_data()

    assert data_module.train_ratio == 1 - val_ratio - test_ratio


def test_reranker_datamodule_invalid_data_path():
    with pytest.raises(AssertionError):
        CustomReRankerDataModule(
            data_root="nonexistent_file.json",
            seq_length=128,
            micro_batch_size=2,
            dataset_kwargs={"num_hard_negatives": 4},
        )


def test_reranker_dataset_negative_sampling_strategies(temp_data_files, mock_tokenizer):
    main_file, _ = temp_data_files
    dataset = ReRankerDataset(
        file_path=main_file,
        tokenizer=mock_tokenizer,
        max_seq_length=128,
        num_hard_negatives=4,
        negative_sample_strategy="random",
    )

    # Test random sampling
    item = dataset[0]
    assert len(item["negatives"]) == 4

    # Test first sampling
    dataset.negative_sample_strategy = "first"
    item = dataset[0]
    assert len(item["negatives"]) == 4


def test_reranker_dataset_sequence_length_handling(temp_data_files, mock_tokenizer):
    main_file, _ = temp_data_files
    max_seq_length = 50
    dataset = ReRankerDataset(
        file_path=main_file,
        tokenizer=mock_tokenizer,
        max_seq_length=max_seq_length,
        num_hard_negatives=4,
    )

    item = dataset[0]
    assert len(item["positive"]) <= max_seq_length
    assert all(len(n) <= max_seq_length for n in item["negatives"])


@pytest.fixture
def mock_specter_dataset():
    """Mock dataset that mimics the structure of the SPECTER dataset."""
    train_data = [
        {
            'anchor': 'What is NeMo?',
            'positive': 'NeMo is NVIDIA\'s framework for conversational AI',
            'negative': 'Wrong answer about NeMo',
        },
        {
            'anchor': 'What is PyTorch?',
            'positive': 'PyTorch is a machine learning framework',
            'negative': 'Incorrect PyTorch description',
        },
    ] * 10  # Multiply to create more examples

    train_dataset = Dataset.from_dict(
        {
            'anchor': [item['anchor'] for item in train_data],
            'positive': [item['positive'] for item in train_data],
            'negative': [item['negative'] for item in train_data],
        }
    )

    return DatasetDict({'train': train_dataset})


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def specter_data_module(mock_tokenizer, temp_dataset_dir, mock_specter_dataset):
    """Fixture that creates a SpecterReRankerDataModule with mocked download functionality."""
    with patch.object(SpecterReRankerDataModule, '_download_data', return_value=mock_specter_dataset):
        with patch('nemo.collections.llm.gpt.data.reranker.get_dataset_root', return_value=temp_dataset_dir):
            data_module = SpecterReRankerDataModule(
                seq_length=512,
                tokenizer=mock_tokenizer,
                micro_batch_size=4,
                global_batch_size=8,
                force_redownload=True,
                delete_raw=True,
                seed=1234,
                memmap_workers=1,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
            )
            data_module.dataset_root = temp_dataset_dir
            yield data_module


def test_specter_reranker_datamodule_initialization(specter_data_module):
    """Test initialization of SpecterReRankerDataModule with default parameters."""
    assert specter_data_module.seq_length == 512
    assert specter_data_module.micro_batch_size == 4
    assert specter_data_module.global_batch_size == 8
    assert specter_data_module.force_redownload is False
    assert specter_data_module.delete_raw is True
    assert specter_data_module.seed == 1234
    assert specter_data_module.memmap_workers == 1
    assert specter_data_module.num_workers == 0
    assert specter_data_module.pin_memory is True
    assert specter_data_module.persistent_workers is False


def test_specter_reranker_datamodule_prepare_data(specter_data_module, temp_dataset_dir):
    """Test data preparation and splitting functionality."""
    # Call prepare_data which should trigger download and preprocessing
    specter_data_module.prepare_data()

    # Verify that the dataset files were created
    assert (temp_dataset_dir / "training.jsonl").exists()
    assert (temp_dataset_dir / "validation.jsonl").exists()
    assert (temp_dataset_dir / "test.jsonl").exists()

    # Verify the content of the training file
    with open(temp_dataset_dir / "training.jsonl", 'r') as f:
        lines = f.readlines()
        assert len(lines) > 0
        first_line = json.loads(lines[0])
        assert "question" in first_line
        assert "pos_doc" in first_line
        assert "neg_doc" in first_line
        assert isinstance(first_line["neg_doc"], list)


def test_specter_reranker_datamodule_dataset_creation(specter_data_module):
    """Test dataset creation with the SpecterReRankerDataModule."""
    # Prepare data and create datasets
    specter_data_module.prepare_data()
    train_dataset = specter_data_module._create_dataset(
        specter_data_module.dataset_root / "training.jsonl", num_hard_negatives=1
    )

    # Test dataset properties
    assert train_dataset.max_seq_length == 512
    assert train_dataset.num_hard_negatives == 1
    assert train_dataset.question_key == "question"


def test_specter_reranker_datamodule_split_ratios(specter_data_module):
    """Test that the dataset is split according to the specified ratios."""
    specter_data_module.prepare_data()

    # Count lines in each split file
    with open(specter_data_module.dataset_root / "training.jsonl", 'r') as f:
        train_lines = len(f.readlines())
    with open(specter_data_module.dataset_root / "validation.jsonl", 'r') as f:
        val_lines = len(f.readlines())
    with open(specter_data_module.dataset_root / "test.jsonl", 'r') as f:
        test_lines = len(f.readlines())

    total_lines = train_lines + val_lines + test_lines

    # Verify approximate split ratios (allowing for small rounding differences)
    assert abs(train_lines / total_lines - 0.80) < 0.05  # 80% training
    assert abs(val_lines / total_lines - 0.15) < 0.05  # 15% validation
    assert abs(test_lines / total_lines - 0.05) < 0.05  # 5% test
