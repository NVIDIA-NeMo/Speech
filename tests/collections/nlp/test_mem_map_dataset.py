# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import csv
import json
import os

import pytest

from nemo.collections.nlp.data.language_modeling import text_memmap_dataset


@pytest.fixture
def jsonl_file(tmp_path):
    # Create a temporary file path
    file_path = tmp_path / "data.jsonl"

    # Generate data to write to the JSONL file
    data = [
        {"name": "John", "age": 30},
        {"name": "Jane", "age": 25},
        {"name": "Bob", "age": 35},
    ]

    # Write data to the JSONL file
    with open(file_path, mode="w") as file:
        for item in data:
            json.dump(item, file)
            file.write("\n")

    # Provide the file path to the test function
    yield str(file_path)

    # Optional: Clean up the temporary file after the test
    file_path.unlink()


@pytest.fixture
def csv_file(tmp_path):
    # Create a temporary file path
    file_path = tmp_path / "data.csv"

    # Generate data to write to the CSV file
    data = [["ID", "Name"], [1, "John"], [2, "Jane"], [3, "Bob"]]

    # Write data to the CSV file
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    # Provide the file path to the test function
    yield str(file_path)

    # Optional: Clean up the temporary file after the test
    file_path.unlink()


def test_jsonl_mem_map_dataset(jsonl_file):
    """Test for JSONL memory-mapped datasets."""

    indexed_dataset = text_memmap_dataset.JSONLMemMapDataset(dataset_paths=[jsonl_file], header_lines=0)
    assert indexed_dataset[0] == {"name": "John", "age": 30}
    assert indexed_dataset[1] == {"name": "Jane", "age": 25}
    assert indexed_dataset[2] == {"name": "Bob", "age": 35}


def test_csv_mem_map_dataset(csv_file):
    """Test for CSV memory-mapped datasets."""

    indexed_dataset = text_memmap_dataset.CSVMemMapDataset(dataset_paths=[csv_file], data_col=1, header_lines=1)
    assert indexed_dataset[0].strip() == "John"
    assert indexed_dataset[1].strip() == "Jane"
    assert indexed_dataset[2].strip() == "Bob"


@pytest.mark.parametrize(
    "dataset_class", [text_memmap_dataset.JSONLMemMapDataset, text_memmap_dataset.CSVMemMapDataset],
)
@pytest.mark.parametrize("use_alternative_index_mapping_dir", [True, False])
@pytest.mark.parametrize("relative_index_fn", [True, False])
def test_mem_map_dataset_index_mapping_dir(
    tmp_path, dataset_class, jsonl_file, use_alternative_index_mapping_dir, relative_index_fn,
):
    """Test for index_mapping_dir."""
    if relative_index_fn:
        jsonl_file = os.path.relpath(jsonl_file)
    else:
        jsonl_file = os.path.abspath(jsonl_file)

    if use_alternative_index_mapping_dir:
        index_mapping_dir = tmp_path / "subdir"
        dataset_class(dataset_paths=[jsonl_file], header_lines=0, index_mapping_dir=str(index_mapping_dir))
        # Index files should not be created in default location.
        assert not os.path.isfile(f"{jsonl_file}.idx.npy")
        assert not os.path.isfile(f"{jsonl_file}.idx.info")
        if relative_index_fn:
            # Remove leading ".." sequences.
            while jsonl_file.startswith(("../")):
                jsonl_file = jsonl_file.lstrip("../")
        idx_fn = f"{str(index_mapping_dir)}/{jsonl_file}.idx"
        assert os.path.isfile(f"{idx_fn}.npy")
        assert os.path.isfile(f"{idx_fn}.info")
    else:
        text_memmap_dataset.JSONLMemMapDataset(dataset_paths=[jsonl_file], header_lines=0)
        assert os.path.isfile(f"{jsonl_file}.idx.npy")
        assert os.path.isfile(f"{jsonl_file}.idx.info")
