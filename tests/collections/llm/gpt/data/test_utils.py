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
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from nemo.collections.llm.gpt.data.utils import (
    IGNORE_INDEX,
    _add_speaker_and_signal,
    _build_memmap_index_files,
    _get_header_conversation_type_mask_role,
    _JSONLMemMapDataset,
    _mask_targets,
    _OnlineSampleMapping,
    _response_value_formater,
    _TextMemMapDataset,
    build_index_files,
    build_index_from_memdata,
)


# Test fixtures
@pytest.fixture
def sample_jsonl_file(tmp_path):
    file_path = tmp_path / "test.jsonl"
    data = [
        {"text": "line1", "label": 1},
        {"text": "line2", "label": 2},
    ]
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return str(file_path)


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.text_to_ids.return_value = [1, 2, 3, 4, 5]
    return tokenizer


@pytest.fixture
def special_tokens():
    return {
        'turn_start': '<turn>',
        'end_of_turn': '</s>',
        'end_of_name': '</n>',
        'system_turn_start': '<system>',
        'label_start': '<label>',
    }


class TestTextMemMapDataset:
    def test_initialization(self, sample_jsonl_file, mock_tokenizer):
        dataset = _TextMemMapDataset(
            dataset_paths=[sample_jsonl_file],
            tokenizer=mock_tokenizer,
            header_lines=0,
        )
        assert len(dataset) > 0
        assert dataset.tokenizer == mock_tokenizer

    def test_getitem(self, sample_jsonl_file, mock_tokenizer):
        dataset = _TextMemMapDataset(
            dataset_paths=[sample_jsonl_file],
            tokenizer=mock_tokenizer,
        )
        item = dataset[0]
        assert isinstance(item, list)

    def test_load_file(self, sample_jsonl_file):
        dataset = _TextMemMapDataset(
            dataset_paths=[sample_jsonl_file],
            header_lines=0,
        )
        mdata, midx = dataset.load_file(sample_jsonl_file)
        assert isinstance(mdata, np.memmap)
        assert isinstance(midx, np.ndarray)

    def test_multiple_files(self, tmp_path):
        # Create multiple test files
        files = []
        for i in range(2):
            file_path = tmp_path / f"test{i}.txt"
            with open(file_path, "w") as f:
                f.write(f"content{i}\n")
            files.append(str(file_path))

        dataset = _TextMemMapDataset(
            dataset_paths=files,
            header_lines=0,
        )
        assert len(dataset.mdata_midx_list) == 2


class TestJSONLMemMapDataset:
    def test_initialization(self, sample_jsonl_file, mock_tokenizer):
        dataset = _JSONLMemMapDataset(
            dataset_paths=[sample_jsonl_file],
            tokenizer=mock_tokenizer,
        )
        assert len(dataset) > 0

    def test_build_data_from_text(self, sample_jsonl_file):
        dataset = _JSONLMemMapDataset(
            dataset_paths=[sample_jsonl_file],
        )
        json_str = '{"key": "value"}'
        result = dataset._build_data_from_text(json_str)
        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_invalid_json(self, sample_jsonl_file):
        dataset = _JSONLMemMapDataset(
            dataset_paths=[sample_jsonl_file],
        )
        with pytest.raises(json.JSONDecodeError):
            dataset._build_data_from_text("invalid json")


def test_mask_targets(mock_tokenizer, special_tokens):
    target = torch.ones(100, dtype=torch.long)
    tokenized_lens = [20, 30, 50]
    speakers = ["user", "assistant", "user"]
    header_len = 10
    s_ids = [torch.ones(l) for l in tokenized_lens]

    _mask_targets(
        target=target,
        tokenized_lens=tokenized_lens,
        speakers=speakers,
        header_len=header_len,
        s_ids=s_ids,
        tokenizer=mock_tokenizer,
        mask_role="user",
        gtype="TEXT_TO_VALUE",
        name_end_token_ids=5,
        special_tokens=special_tokens,
        label_start_ids=[4],
        num_turn_start_tokens=1,
    )

    assert (target == IGNORE_INDEX).any()


def test_get_header_conversation_type_mask_role(special_tokens):
    source = {
        "system": "System prompt",
        "conversations": [
            {"from": "user", "value": "Hello"},
            {"from": "assistant", "value": "Hi"},
        ],
        "type": "TEXT_TO_VALUE",
        "mask": "user",
    }

    header, conversation, data_type, mask_role = _get_header_conversation_type_mask_role(source, special_tokens)

    assert isinstance(header, str)
    assert isinstance(conversation, str)
    assert data_type == "TEXT_TO_VALUE"
    assert mask_role == "user"


def test_add_speaker_and_signal(special_tokens):
    header = "System: "
    source = [{"from": "user", "value": "Hello", "label": "positive"}, {"from": "assistant", "value": "Hi"}]

    result = _add_speaker_and_signal(
        header=header, source=source, mask_role="user", gtype="TEXT_TO_VALUE", special_tokens=special_tokens
    )

    assert isinstance(result, str)
    assert special_tokens['turn_start'] in result
    assert special_tokens['end_of_turn'] in result


def test_response_value_formater(special_tokens):
    # Test with string label
    result = _response_value_formater("test_label", special_tokens['label_start'], special_tokens['end_of_name'])
    assert isinstance(result, str)
    assert result.startswith(special_tokens['label_start'])
    assert result.endswith(special_tokens['end_of_name'])

    # Test with None label
    result = _response_value_formater(None, special_tokens['label_start'], special_tokens['end_of_name'])
    assert result == ''

    # Test with invalid label type
    with pytest.raises(ValueError):
        _response_value_formater(123, special_tokens['label_start'], special_tokens['end_of_name'])


def test_build_index_files(tmp_path):
    # Create test file
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("line1\nline2\n")

    build_index_files(
        dataset_paths=[str(file_path)],
        newline_int=10,
        workers=1,
        build_index_fn=build_index_from_memdata,
        index_mapping_dir=str(tmp_path),
    )


def test_build_memmap_index_files(tmp_path):
    # Create test file
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("line1\nline2\n")

    result = _build_memmap_index_files(
        newline_int=10, build_index_fn=build_index_from_memdata, fn=str(file_path), index_mapping_dir=str(tmp_path)
    )

    assert result == True
    # Test that calling again returns False (files exist)
    result = _build_memmap_index_files(
        newline_int=10, build_index_fn=build_index_from_memdata, fn=str(file_path), index_mapping_dir=str(tmp_path)
    )
    assert result == False


@pytest.mark.parametrize(
    "dataset_size,num_samples,block_size,shuffle",
    [
        (100, 50, 10, True),
        (100, 150, 20, False),
        (100, 100, None, True),
    ],
)
def test_online_sample_mapping_variations(dataset_size, num_samples, block_size, shuffle):
    mapping = _OnlineSampleMapping(
        dataset_size=dataset_size, num_samples=num_samples, block_size=block_size, shuffle=shuffle, seed=42
    )

    assert len(mapping) == num_samples
    # Test block size is correctly set
    if block_size is None:
        assert mapping.block_size == dataset_size
    else:
        assert mapping.block_size == min(block_size, dataset_size)


def test_online_sample_mapping_cache():
    mapping = _OnlineSampleMapping(dataset_size=100, num_samples=50, block_size=10, cache_maxsize=2, shuffle=True)

    # Test cache behavior
    block1 = mapping.get_sample_block(0)
    block2 = mapping.get_sample_block(1)
    block3 = mapping.get_sample_block(2)

    # Check that blocks are cached and return the same values
    np.testing.assert_array_equal(mapping.get_sample_block(0), block1)
    np.testing.assert_array_equal(mapping.get_sample_block(1), block2)
    np.testing.assert_array_equal(mapping.get_sample_block(2), block3)


if __name__ == "__main__":
    pytest.main([__file__])
