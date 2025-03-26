# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Union

import pytest
import yaml

from nemo.collections.llm.gpt.data.megatron.hyena.config import (
    Evo2BlendedDatasetConfig,
    infer_global_batch_size,
    parse_dataset_config,
)


@contextmanager
def change_dir(new_dir: Union[str, Path]):
    """
    Context manager for temporarily changing the working directory using os.

    Args:
        new_dir (Union[str, Path]): The directory to change to

    Yields:
        str: The new working directory path

    Example:
        with change_dir('/path/to/dir'):
            # Do some work in the new directory
            ...
        # Original directory is restored
    """
    prev_dir = os.getcwd()
    new_dir = os.path.expanduser(str(new_dir))
    try:
        os.chdir(new_dir)
        yield new_dir
    finally:
        os.chdir(prev_dir)


@pytest.fixture
def temp_dataset_config():
    # Create a temporary directory for the dataset path
    temp_dir = tempfile.TemporaryDirectory()
    dataset_path = temp_dir.name

    # Create a temporary YAML file for the dataset configuration
    temp_yaml = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    dataset_config_path = temp_yaml.name

    # Define dataset configuration content
    dataset_config_content = [
        {"dataset_prefix": "dataset1", "dataset_weight": 0.5, "dataset_split": "train"},
        {"dataset_prefix": "dataset2", "dataset_weight": 0.5, "dataset_split": "train"},
        {"dataset_prefix": "dataset1", "dataset_weight": 0.6, "dataset_split": "validation"},
        {"dataset_prefix": "dataset2", "dataset_weight": 0.6, "dataset_split": "validation"},
        {"dataset_prefix": "dataset2", "dataset_weight": 0.2, "dataset_split": "test"},
    ]

    # Write the dataset configuration content to the YAML file
    with open(dataset_config_path, "w") as yaml_file:
        yaml.dump(dataset_config_content, yaml_file)

    # Create dummy dataset files in the temporary directory
    for dataset in dataset_config_content:
        dataset_file = Path(dataset_path) / f"{dataset['dataset_prefix']}.txt"
        dataset_file.touch()

    yield dataset_config_path, dataset_path

    # Clean up temporary files and directories
    temp_yaml.close()
    os.remove(dataset_config_path)
    temp_dir.cleanup()


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create temporary dataset files for testing."""
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    (dataset_dir / "dataset.bin").touch()
    return dataset_dir


def test_valid_absolute_path(tmp_dataset):
    """Test configuration with valid absolute path."""
    config = Evo2BlendedDatasetConfig(
        dataset_prefix=str(tmp_dataset / "dataset"), dataset_weight=0.5, dataset_split="train"
    )
    assert config.dataset_prefix == str(tmp_dataset / "dataset")
    assert config.dataset_weight == 0.5
    assert config.dataset_split == "train"


def test_valid_relative_path(tmp_dataset):
    """Test configuration with valid relative path and base data path."""
    config = Evo2BlendedDatasetConfig(
        dataset_path=str(tmp_dataset), dataset_prefix="dataset", dataset_weight=0.5, dataset_split="validation"
    )
    assert config.dataset_prefix == str(tmp_dataset / "dataset")


def test_invalid_relative_path_without_base():
    """Test relative path fails without base data path."""
    with pytest.raises(ValueError, match=f"dataset_prefix file does not exist: {Path('dataset').resolve()}"):
        Evo2BlendedDatasetConfig(dataset_prefix="dataset", dataset_weight=0.5, dataset_split="train")


def test_valid_relative_path_without_base(tmp_dataset: Path):
    """Test relative path in current workdir does not fail without base data path."""
    # changing temporary cwd since Path(dataset_prefix).resolve() will resolve relative paths to the current working directory
    with change_dir(tmp_dataset):
        Evo2BlendedDatasetConfig(dataset_prefix="dataset", dataset_weight=0.5, dataset_split="train")


def test_nonexistent_parent_path(tmp_path):
    """Test configuration fails with nonexistent parent directory."""
    invalid_path = tmp_path / "nonexistent" / "dataset"
    with pytest.raises(ValueError, match="parent path does not exist"):
        Evo2BlendedDatasetConfig(dataset_prefix=str(invalid_path), dataset_weight=0.5, dataset_split="train")


def test_nonexistent_dataset_file(tmp_dataset):
    """Test configuration fails with nonexistent dataset file."""
    invalid_path = tmp_dataset / "nonexistent_dataset"
    with pytest.raises(ValueError, match="dataset_prefix file does not exist"):
        Evo2BlendedDatasetConfig(dataset_prefix=str(invalid_path), dataset_weight=0.5, dataset_split="train")


def test_path_resolution(tmp_dataset):
    """Test proper path resolution with different input formats."""
    relative_path = Path("dataset")
    absolute_path = tmp_dataset / "dataset"

    config1 = Evo2BlendedDatasetConfig(
        dataset_path=str(tmp_dataset), dataset_prefix=str(relative_path), dataset_weight=0.5, dataset_split="train"
    )
    # changing temporary cwd since Path(dataset_prefix).resolve() will resolve relative paths to the current working directory
    with change_dir(tmp_dataset):
        config2 = Evo2BlendedDatasetConfig(
            dataset_prefix=str(absolute_path), dataset_weight=0.5, dataset_split="train"
        )

    assert config1.dataset_prefix == config2.dataset_prefix


def test_parse_dataset_config(temp_dataset_config):
    dataset_config_path, dataset_path = temp_dataset_config

    # Call the function to test
    result = parse_dataset_config(dataset_config_path, dataset_path)

    print(result)
    # Define the expected result
    expected_result = defaultdict(
        list,
        {
            "train": [0.5, str(Path(dataset_path) / "dataset1"), 0.5, str(Path(dataset_path) / "dataset2")],
            "validation": [0.5, str(Path(dataset_path) / "dataset1"), 0.5, str(Path(dataset_path) / "dataset2")],
            "test": [
                1.0,
                str(Path(dataset_path) / "dataset2"),
            ],
        },
    )

    # Assert the result matches the expected result
    assert result == expected_result


def test_infer_global_batch_size_validation():
    # Test non-integer inputs
    with pytest.raises(ValueError, match="All arguments must be of type int"):
        infer_global_batch_size(1.0, 1, 1, 1, 1, 1, 1)

    # Test all non-positive inputs
    with pytest.raises(ValueError, match="micro_batch_size must be greater than 0"):
        infer_global_batch_size(0, 1, 1, 1, 1, 1, 1)

    with pytest.raises(ValueError, match="num_nodes must be greater than 0"):
        infer_global_batch_size(1, 0, 1, 1, 1, 1, 1)

    with pytest.raises(ValueError, match="devices must be greater than 0"):
        infer_global_batch_size(1, 1, 0, 1, 1, 1, 1)

    with pytest.raises(ValueError, match="accumulate_grad_batches must be greater than 0"):
        infer_global_batch_size(1, 1, 1, 0, 1, 1, 1)

    with pytest.raises(ValueError, match="tensor_model_parallel_size must be greater than 0"):
        infer_global_batch_size(1, 1, 1, 1, 0, 1, 1)

    with pytest.raises(ValueError, match="pipeline_model_parallel_size must be greater than 0"):
        infer_global_batch_size(1, 1, 1, 1, 1, 0, 1)

    with pytest.raises(ValueError, match="context_model_parallel_size must be greater than 0"):
        infer_global_batch_size(1, 1, 1, 1, 1, 1, 0)


def test_infer_global_batch_size_calculation():
    # Test world_size divisibility error
    with pytest.raises(ValueError, match="world_size must be divisible by"):
        infer_global_batch_size(1, 1, 3, 1, 2, 1, 1)  # 3 devices not divisible by TP=2

    # Test successful calculation
    result = infer_global_batch_size(
        micro_batch_size=2,
        num_nodes=1,
        devices=4,
        accumulate_grad_batches=2,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_model_parallel_size=1,
    )
    # world_size = 1 * 4 = 4
    # model_parallel_size = 2 * 1 * 1 = 2
    # data_parallel_size = 4 // 2 = 2
    # global_batch_size = 2 * 2 * 2 = 8
    assert result == 8
