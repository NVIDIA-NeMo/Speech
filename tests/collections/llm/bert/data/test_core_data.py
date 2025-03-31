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
import torch

from nemo.collections.llm.bert.data.core import BertEmbeddingDataset, create_sft_dataset, get_dataset_root


class MockTokenizer:
    def __init__(self):
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

    def text_to_ids(self, text):
        # Simple mock implementation that returns sequential tokens
        return [3, 4, 5]  # Dummy tokens


@pytest.fixture
def sample_data():
    return [
        {
            "query": "what is machine learning?",
            "pos_doc": "Machine learning is a subset of AI",
            "neg_doc": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3", "Wrong answer 4"],
            "query_id": "q1",
            "doc_id": "d1",
        },
        {
            "query": "what is deep learning?",
            "pos_doc": "Deep learning uses neural networks",
            "neg_doc": ["Wrong answer 5", "Wrong answer 6", "Wrong answer 7", "Wrong answer 8"],
            "query_id": "q2",
            "doc_id": "d2",
        },
    ]


@pytest.fixture
def temp_jsonl_file(sample_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    yield Path(f.name)
    Path(f.name).unlink()


def test_get_dataset_root():
    root = get_dataset_root("test_dataset")
    assert root.exists()
    assert root.is_dir()


def test_bert_embedding_dataset_train(temp_jsonl_file):
    tokenizer = MockTokenizer()
    dataset = BertEmbeddingDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=128,
        data_type='train',
        num_hard_negatives=2,
    )

    # Test dataset length
    assert len(dataset) == 2

    # Test single item retrieval
    item = dataset[0]
    assert 'query' in item
    assert 'pos_doc' in item
    assert 'neg_doc' in item
    assert 'metadata' in item
    assert len(item['neg_doc']) == 2  # num_hard_negatives=2

    # Test batch collation
    batch = dataset._collate_fn([dataset[0], dataset[1]])
    assert isinstance(batch['input_ids'], torch.Tensor)
    assert isinstance(batch['attention_mask'], torch.Tensor)
    assert isinstance(batch['token_type_ids'], torch.Tensor)
    assert isinstance(batch['position_ids'], torch.Tensor)


def test_bert_embedding_dataset_query(temp_jsonl_file):
    tokenizer = MockTokenizer()
    dataset = BertEmbeddingDataset(
        file_path=str(temp_jsonl_file), tokenizer=tokenizer, max_seq_length=128, data_type='query'
    )

    item = dataset[0]
    assert 'query' in item
    assert 'metadata' in item
    assert 'query_id' in item['metadata']


def test_bert_embedding_dataset_doc(temp_jsonl_file):
    tokenizer = MockTokenizer()
    dataset = BertEmbeddingDataset(
        file_path=str(temp_jsonl_file), tokenizer=tokenizer, max_seq_length=128, data_type='doc'
    )

    item = dataset[0]
    assert 'pos_doc' in item
    assert 'metadata' in item
    assert 'doc_id' in item['metadata']


def test_create_sft_dataset(temp_jsonl_file):
    tokenizer = MockTokenizer()
    dataset = create_sft_dataset(path=temp_jsonl_file, tokenizer=tokenizer, seq_length=128, data_type='train')

    assert isinstance(dataset, BertEmbeddingDataset)
    assert len(dataset) > 0


def test_truncation_methods(temp_jsonl_file):
    tokenizer = MockTokenizer()

    # Test right truncation
    dataset_right = BertEmbeddingDataset(
        file_path=str(temp_jsonl_file), tokenizer=tokenizer, truncation_method='right'
    )

    # Test left truncation
    dataset_left = BertEmbeddingDataset(file_path=str(temp_jsonl_file), tokenizer=tokenizer, truncation_method='left')

    # Test invalid truncation method
    with pytest.raises(AssertionError):
        BertEmbeddingDataset(file_path=str(temp_jsonl_file), tokenizer=tokenizer, truncation_method='invalid')


def test_collate_fn_all_data_types(temp_jsonl_file):
    """Test _collate_fn for all data types (train, query, doc) with different batch configurations"""
    tokenizer = MockTokenizer()

    # Test train data type
    dataset_train = BertEmbeddingDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=128,
        data_type='train',
        num_hard_negatives=2,
    )

    batch_train = [dataset_train[0], dataset_train[1]]
    collated_train = dataset_train._collate_fn(batch_train)

    # For train, we expect query + positive doc + num_hard_negatives docs per item
    expected_train_items = len(batch_train) * (1 + 1 + 2)  # query + pos_doc + 2 neg_docs
    assert collated_train['input_ids'].shape[0] == expected_train_items
    assert collated_train['attention_mask'].shape[0] == expected_train_items
    assert collated_train['token_type_ids'].shape[0] == expected_train_items
    assert len(collated_train['metadata']) == len(batch_train)

    # Test query data type
    dataset_query = BertEmbeddingDataset(
        file_path=str(temp_jsonl_file), tokenizer=tokenizer, max_seq_length=128, data_type='query'
    )

    batch_query = [dataset_query[0], dataset_query[1]]
    collated_query = dataset_query._collate_fn(batch_query)

    # For query, we expect only query embeddings
    assert collated_query['input_ids'].shape[0] == len(batch_query)
    assert collated_query['attention_mask'].shape[0] == len(batch_query)
    assert collated_query['token_type_ids'].shape[0] == len(batch_query)
    assert len(collated_query['metadata']) == len(batch_query)

    # Test doc data type
    dataset_doc = BertEmbeddingDataset(
        file_path=str(temp_jsonl_file), tokenizer=tokenizer, max_seq_length=128, data_type='doc'
    )

    batch_doc = [dataset_doc[0], dataset_doc[1]]
    collated_doc = dataset_doc._collate_fn(batch_doc)

    # For doc, we expect only document embeddings
    assert collated_doc['input_ids'].shape[0] == len(batch_doc)
    assert collated_doc['attention_mask'].shape[0] == len(batch_doc)
    assert collated_doc['token_type_ids'].shape[0] == len(batch_doc)
    assert len(collated_doc['metadata']) == len(batch_doc)


def test_collate_fn_attention_mask(temp_jsonl_file):
    """Test attention mask generation in _collate_fn for different truncation methods"""
    tokenizer = MockTokenizer()

    # Test right truncation
    dataset_right = BertEmbeddingDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=128,
        data_type='train',
        truncation_method='right',
    )

    batch = [dataset_right[0]]
    collated_right = dataset_right._collate_fn(batch)

    # For right truncation, attention mask should be 1s from the left
    first_mask = collated_right['attention_mask'][0]
    assert (first_mask[:3] == 1).all()  # First tokens should be attended to
    assert (first_mask[-3:] == 0).all()  # Last tokens should be masked (padding)

    # Test left truncation
    dataset_left = BertEmbeddingDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=128,
        data_type='train',
        truncation_method='left',
    )

    collated_left = dataset_left._collate_fn(batch)

    # For left truncation, attention mask should be 1s from the right
    first_mask = collated_left['attention_mask'][0]
    assert (first_mask[-3:] == 1).all()  # Last tokens should be attended to
    assert (first_mask[:3] == 0).all()  # First tokens should be masked (padding)


def test_collate_fn_batch_size_handling(temp_jsonl_file):
    """Test _collate_fn with different batch sizes"""
    tokenizer = MockTokenizer()
    dataset = BertEmbeddingDataset(
        file_path=str(temp_jsonl_file),
        tokenizer=tokenizer,
        max_seq_length=128,
        data_type='train',
        num_hard_negatives=2,
    )

    # Test with single item batch
    single_batch = [dataset[0]]
    collated_single = dataset._collate_fn(single_batch)
    expected_single_items = 1 * (1 + 1 + 2)  # 1 item * (query + pos_doc + 2 neg_docs)
    assert collated_single['input_ids'].shape[0] == expected_single_items

    # Test with multiple item batch
    multi_batch = [dataset[0], dataset[1]]
    collated_multi = dataset._collate_fn(multi_batch)
    expected_multi_items = 2 * (1 + 1 + 2)  # 2 items * (query + pos_doc + 2 neg_docs)
    assert collated_multi['input_ids'].shape[0] == expected_multi_items


def test_collate_fn_metadata_handling(temp_jsonl_file):
    """Test metadata handling in _collate_fn"""
    tokenizer = MockTokenizer()
    dataset = BertEmbeddingDataset(
        file_path=str(temp_jsonl_file), tokenizer=tokenizer, max_seq_length=128, data_type='train'
    )

    batch = [dataset[0], dataset[1]]
    collated = dataset._collate_fn(batch)

    # Verify metadata is preserved
    assert len(collated['metadata']) == len(batch)
    for item_metadata in collated['metadata']:
        assert 'query' in item_metadata
        assert 'pos_doc' in item_metadata
        assert 'neg_doc' in item_metadata
        assert 'query_id' in item_metadata
        assert 'doc_id' in item_metadata
