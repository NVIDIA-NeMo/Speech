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

import numpy as np
import pytest
import torch

from nemo.collections.llm.gpt.data.core import (
    GPTSFTChatDataset,
    GPTSFTDataset,
    GPTSFTPackedDataset,
    create_sft_dataset,
)


class MockTokenizer:
    def __init__(self):
        self.eos_id = 2
        self.bos_id = 1
        self.space_sensitive = False

    def text_to_ids(self, text):
        # Simple mock implementation - converts each character to its ASCII value
        return [ord(c) % 10 for c in text]


@pytest.fixture
def sample_data():
    return [
        {"input": "What is machine learning?", "output": "Machine learning is a subset of AI."},
        {
            "input": "Define neural networks.",
            "output": "Neural networks are computing systems inspired by biological brains.",
        },
    ]


@pytest.fixture
def temp_jsonl_file(sample_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    yield Path(f.name)
    Path(f.name).unlink()


@pytest.fixture
def tokenizer():
    return MockTokenizer()


def test_gpt_sft_dataset_initialization(temp_jsonl_file, tokenizer):
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=512,
        prompt_template="{input} {output}",
        truncation_field="input",
        label_key="output",
    )

    assert len(dataset) == 2
    assert dataset.max_seq_length == 512
    assert dataset.tokenizer == tokenizer


def test_create_sft_dataset(temp_jsonl_file, tokenizer):
    dataset = create_sft_dataset(
        path=temp_jsonl_file,
        tokenizer=tokenizer,
        seq_length=512,
        prompt_template="{input} {output}",
    )

    assert isinstance(dataset, GPTSFTDataset)
    assert dataset.max_seq_length == 512


def test_dataset_getitem(temp_jsonl_file, tokenizer):
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=512,
        prompt_template="{input} {output}",
        truncation_field="input",
        label_key="output",
    )

    item = dataset[0]
    assert 'input_ids' in item
    assert 'answer_start_idx' in item
    assert 'context_ids' in item
    assert 'answer_ids' in item
    assert 'metadata' in item
    assert isinstance(item['input_ids'], list)


def test_dataset_collate_fn(temp_jsonl_file, tokenizer):
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=512,
        prompt_template="{input} {output}",
        truncation_field="input",
        label_key="output",
    )

    batch = [dataset[0], dataset[1]]
    collated = dataset.collate_fn(batch)

    assert isinstance(collated, dict)
    assert 'tokens' in collated
    assert 'labels' in collated
    assert 'loss_mask' in collated
    assert 'position_ids' in collated
    assert isinstance(collated['tokens'], torch.Tensor)
    assert isinstance(collated['labels'], torch.Tensor)
    assert isinstance(collated['loss_mask'], torch.Tensor)


def test_dataset_truncation(temp_jsonl_file, tokenizer):
    # Test with very small max_seq_length to force truncation
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=10,
        prompt_template="{input} {output}",
        truncation_field="input",
        label_key="output",
    )

    item = dataset[0]
    assert len(item['input_ids']) <= 10


def test_dataset_padding(temp_jsonl_file, tokenizer):
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=512,
        prompt_template="{input} {output}",
        truncation_field="input",
        label_key="output",
        pad_to_max_length=True,
    )

    batch = [dataset[0], dataset[1]]
    collated = dataset.collate_fn(batch)

    assert collated['tokens'].size(1) == 512
    assert collated['labels'].size(1) == 512


@pytest.fixture
def chat_sample_data():
    return [
        {
            "conversations": [
                {"from": "System", "value": "You are a helpful assistant."},
                {"from": "User", "value": "What is Python?"},
                {"from": "Assistant", "value": "Python is a programming language."},
            ],
            "system": "user",
        },
        {
            "conversations": [
                {"from": "User", "value": "How do I print in Python?"},
                {"from": "Assistant", "value": "Use the print() function."},
            ],
            "system": "user",
        },
    ]


@pytest.fixture
def temp_chat_jsonl_file(chat_sample_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in chat_sample_data:
            f.write(json.dumps(item) + '\n')
    yield Path(f.name)
    Path(f.name).unlink()


@pytest.fixture
def temp_npy_file():
    # Create a structured array that matches the expected format
    dtype = np.dtype([('input_ids', np.int64, (5,)), ('seq_start_id', np.int64, (2,)), ('loss_mask', np.int64, (5,))])

    # Create the data with the correct structure
    data = np.array(
        [
            (
                np.array([1, 2, 3, 4, 5]),  # input_ids
                np.array([0, 0]),  # seq_start_id
                np.array([1, 1, 1, 1, 1]),  # loss_mask
            )
        ],
        dtype=dtype,
    )

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f, data)
    yield Path(f.name)
    Path(f.name).unlink()


def test_multiple_truncation(temp_jsonl_file, tokenizer):
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=10,
        prompt_template="{input} {output}",
        truncation_field="input,output",
        label_key="output",
    )

    template_ids = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    template_ids_keys = ['input', 'output']
    context_ids, label_ids = dataset._multiple_truncation(template_ids, template_ids_keys)

    assert len(context_ids) <= dataset.max_seq_length
    assert len(label_ids) <= dataset.max_seq_length


def test_truncation_methods(temp_jsonl_file, tokenizer):
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=10,
        prompt_template="{input} {output}",
        truncation_field="input",
        label_key="output",
    )

    # Test left truncation
    dataset.truncation_method = 'left'
    ids = [1, 2, 3, 4, 5]
    truncated = dataset._truncation(ids, 3)
    assert truncated == [3, 4, 5]

    # Test right truncation
    dataset.truncation_method = 'right'
    truncated = dataset._truncation(ids, 3)
    assert truncated == [1, 2, 3]


def test_separate_template(temp_jsonl_file, tokenizer):
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=512,
        prompt_template="Question: {input}\nAnswer: {output}",
        truncation_field="input",
        label_key="output",
    )

    template_strings, template_keys = dataset._separate_template(["What is ML?", "ML is AI."])
    assert len(template_strings) == len(template_keys)
    assert "Question:" in template_strings[0]
    assert "Answer:" in template_strings[-2]


def test_chat_dataset(temp_chat_jsonl_file, tokenizer):
    dataset = create_sft_dataset(path=temp_chat_jsonl_file, tokenizer=tokenizer, seq_length=512, chat=True)

    assert isinstance(dataset, GPTSFTChatDataset)
    item = dataset[0]
    assert 'input_ids' in item
    assert 'mask' in item

    # Test collate_fn for chat dataset
    batch = [dataset[0], dataset[0]]
    collated = dataset.collate_fn(batch)
    assert 'tokens' in collated
    assert 'labels' in collated
    assert 'loss_mask' in collated


def test_packed_dataset(temp_npy_file, tokenizer):
    # Create metadata file
    metadata = [{"max_samples_per_bin": 2, "dataset_max_seqlen": 512}]
    metadata_file = Path("temp_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    try:
        dataset = create_sft_dataset(
            path=temp_npy_file,
            tokenizer=tokenizer,
            seq_length=512,
            pack_metadata_file_path=metadata_file,
            pad_cu_seqlens=True,
            pad_to_max_length=True,
        )

        assert isinstance(dataset, GPTSFTPackedDataset)

        # Test getting an item
        item = dataset[0]
        assert 'input_ids' in item
        assert 'seq_boundaries' in item
        assert 'loss_mask' in item

        # Verify the shapes and types
        assert isinstance(item['input_ids'], np.ndarray)
        assert isinstance(item['loss_mask'], np.ndarray)

        # Test negative indexing
        item_neg = dataset[-1]
        assert 'input_ids' in item_neg
        assert 'seq_boundaries' in item_neg
        assert 'loss_mask' in item_neg

    finally:
        if metadata_file.exists():
            metadata_file.unlink()


def test_packed_dataset_no_pad_cu_seqlens(temp_npy_file, tokenizer):
    """Test packed dataset without padding cu_seqlens"""
    dataset = create_sft_dataset(
        path=temp_npy_file, tokenizer=tokenizer, seq_length=512, pad_cu_seqlens=False, pad_to_max_length=False
    )

    assert isinstance(dataset, GPTSFTPackedDataset)

    # Test collate_fn
    batch = [dataset[0], dataset[0]]
    collated = dataset.collate_fn(batch)

    # Verify that cu_seqlens is not padded
    assert 'cu_seqlens' in collated
    assert isinstance(collated['cu_seqlens'], torch.IntTensor)
    # The shape should be smaller than when padding is enabled


def test_packed_dataset_invalid_data(temp_npy_file, tokenizer, caplog):
    """Test packed dataset with invalid data"""
    import numpy as np

    # Create invalid data
    invalid_data = np.array([1, 2, 3])  # Wrong format

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f, invalid_data)
        invalid_file = Path(f.name)

    try:
        with pytest.raises(Exception):
            dataset = create_sft_dataset(path=invalid_file, tokenizer=tokenizer, seq_length=512)
            batch = [dataset[0], dataset[0]]
            dataset.collate_fn(batch)
    finally:
        invalid_file.unlink()


def test_virtual_tokens(temp_jsonl_file, tokenizer):
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=512,
        prompt_template="{input} {output}",
        truncation_field="input",
        label_key="output",
        virtual_tokens=2,
    )

    item = dataset[0]
    assert len(item['context_ids']) >= 2  # Should include virtual tokens


def test_ceil_to_power_2(temp_jsonl_file, tokenizer):
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=512,
        prompt_template="{input} {output}",
        truncation_field="input",
        label_key="output",
        ceil_to_power_2=True,
    )

    batch = [dataset[0], dataset[1]]
    collated = dataset.collate_fn(batch)
    # Check if padded length is a power of 2
    padded_length = collated['tokens'].size(1)
    assert (padded_length & (padded_length - 1) == 0) and padded_length != 0


def test_attention_mask_from_fusion(temp_jsonl_file, tokenizer):
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=512,
        prompt_template="{input} {output}",
        truncation_field="input",
        label_key="output",
        get_attention_mask_from_fusion=True,
    )

    batch = [dataset[0], dataset[1]]
    collated = dataset.collate_fn(batch)
    assert 'attention_mask' not in collated


def test_output_original_text(temp_jsonl_file, tokenizer):
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=512,
        prompt_template="{input} {output}",
        truncation_field="input",
        label_key="output",
        output_original_text=True,
    )

    item = dataset[0]
    assert 'input' in item['metadata']
    assert 'output' in item['metadata']


def test_negative_indexing(temp_jsonl_file, tokenizer):
    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=512,
        prompt_template="{input} {output}",
        truncation_field="input",
        label_key="output",
    )

    item = dataset[-1]  # Should get last item
    assert 'input_ids' in item


def test_special_tokens(temp_jsonl_file, tokenizer):
    special_tokens = {
        "system_turn_start": "<sys>",
        "turn_start": "<turn>",
        "label_start": "<label>",
        "end_of_turn": "\n",
        "end_of_name": "\n",
    }

    dataset = GPTSFTDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=512,
        prompt_template="{input} {output}",
        truncation_field="input",
        label_key="output",
        special_tokens=special_tokens,
    )

    assert dataset.special_tokens == special_tokens
