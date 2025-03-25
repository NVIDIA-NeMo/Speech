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

import pytest

from nemo.collections.llm.gpt.data.retrieval import CustomRetrievalDataModule


@pytest.fixture
def sample_data():
    return [
        {
            "question": "What is NeMo?",
            "pos_doc": ["NeMo is NVIDIA's framework for conversational AI"],  # Always use list
            "neg_doc": ["Wrong answer 1", "Wrong answer 2"],
        },
        {
            "question": "What is PyTorch?",
            "pos_doc": ["PyTorch is a machine learning framework"],  # Always use list
            "neg_doc": ["Incorrect PyTorch description"],
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
                "neg_doc": ["Wrong answer 1"],
            }
        ]
        val_file = Path(tmpdir) / "val.json"
        with open(val_file, 'w') as f:
            json.dump(val_data, f)

        yield str(main_file), str(val_file)


def test_initialization(temp_data_files):
    main_file, val_file = temp_data_files
    data_module = CustomRetrievalDataModule(data_root=main_file, val_root=val_file, seq_length=128, micro_batch_size=2)
    assert data_module.val_ratio == 0
    assert data_module.train_ratio == 0.99
    assert data_module.query_key == "question"
    assert data_module.pos_doc_key == "pos_doc"
    assert data_module.neg_doc_key == "neg_doc"


def test_multiple_data_roots(sample_data):
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

        data_module = CustomRetrievalDataModule(
            data_root=[str(file1), str(file2)],
            seq_length=128,
            micro_batch_size=2,
            force_redownload=True,
            test_ratio=0.05,
            val_ratio=0.05,
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
def test_different_split_ratios(temp_data_files, val_ratio, test_ratio):
    main_file, additional_file = temp_data_files

    if val_ratio == 0.0:
        val_file = additional_file
    else:
        val_file = None

    if test_ratio == 0.0:
        test_file = additional_file
    else:
        test_file = None

    data_module = CustomRetrievalDataModule(
        data_root=main_file,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seq_length=128,
        micro_batch_size=2,
        force_redownload=True,
        val_root=val_file,
        test_root=test_file,
    )
    data_module.prepare_data()

    assert data_module.train_ratio == 1 - val_ratio - test_ratio


def test_invalid_data_path():
    with pytest.raises(AssertionError):
        CustomRetrievalDataModule(data_root="nonexistent_file.json", seq_length=128, micro_batch_size=2)
