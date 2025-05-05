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
import pytest
import torch

from nemo.collections.llm.bert.model.embedding import (
    BertEmbeddingHead,
    BertEmbeddingLargeConfig,
    BertEmbeddingMiniConfig,
)


class TestBertEmbeddingHead:
    @pytest.fixture
    def embedding_head(self):
        return BertEmbeddingHead(word_embedding_dimension=768)

    def test_embedding_head_forward(self, embedding_head):
        batch_size = 2
        seq_length = 4
        hidden_dim = 768

        token_embeddings = torch.randn(seq_length, batch_size, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_length)

        output = embedding_head(token_embeddings, attention_mask)

        assert output.shape == (batch_size, hidden_dim)
        # Check if output vectors are normalized (L2 norm should be close to 1)
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_embedding_head_masked_tokens(self, embedding_head):
        batch_size = 2
        seq_length = 4
        hidden_dim = 768

        token_embeddings = torch.randn(seq_length, batch_size, hidden_dim)
        # Create mask where some tokens are masked (0)
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float)

        output = embedding_head(token_embeddings, attention_mask)
        assert output.shape == (batch_size, hidden_dim)


class TestBertEmbeddingConfig:
    def test_large_config(self):
        config = BertEmbeddingLargeConfig()
        assert config.num_layers == 24
        assert config.hidden_size == 1024
        assert config.intermediate_size == 4096
        assert config.num_attention_heads == 16

    def test_mini_config(self):
        config = BertEmbeddingMiniConfig()
        assert config.num_layers == 6
        assert config.hidden_size == 384
        assert config.intermediate_size == 1536
        assert config.num_attention_heads == 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
class TestBertEmbeddingModel:
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.vocab_size = 30522  # Base BERT vocab size
        return tokenizer

    @pytest.fixture
    def model_config(self):
        return BertEmbeddingMiniConfig()  # Using mini config for faster testing

    @pytest.fixture
    def model(self, model_config, mock_tokenizer):
        model = BertEmbeddingModel(config=model_config, tokenizer=mock_tokenizer)
        model.configure_model()
        return model


from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo.collections.llm.bert.model.embedding import bert_embedding_data_step


def test_bert_embedding_data_step():
    # Setup mock data
    batch_size = 2
    seq_length = 10
    mock_batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length),
        "token_type_ids": torch.zeros(batch_size, seq_length),
        "labels": torch.tensor([1, 0]),  # This should not be moved to cuda
    }

    # Create a mock iterator that returns our batch
    mock_iterator = iter([mock_batch])

    # Mock CUDA movement
    def mock_cuda(non_blocking=True):
        return torch.ones_like(mock_batch["attention_mask"])

    for tensor in mock_batch.values():
        if isinstance(tensor, torch.Tensor):
            tensor.cuda = MagicMock(side_effect=mock_cuda)

    # Mock pipeline first stage check
    with patch('megatron.core.parallel_state.is_pipeline_first_stage', return_value=True):
        # Mock context parallel rank function to return the same batch
        with patch('megatron.core.utils.get_batch_on_this_cp_rank', side_effect=lambda x: x):
            with patch('megatron.core.parallel_state.get_context_parallel_world_size', return_value=1):
                result = bert_embedding_data_step(mock_iterator)

    # Verify the output contains the expected keys
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "token_type_ids" in result
    assert "labels" in result  # Should be None in the result

    # Verify cuda was called for required tensors
    mock_batch["attention_mask"].cuda.assert_called_once()
    mock_batch["token_type_ids"].cuda.assert_called_once()
    mock_batch["input_ids"].cuda.assert_called_once()

    # Verify labels were not moved to cuda (should be None)
    assert result["labels"] is None


def test_bert_embedding_data_step_tuple_input():
    # Test the case where input is a tuple of (batch, _, _)
    batch_size = 2
    seq_length = 10
    mock_batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length),
        "token_type_ids": torch.zeros(batch_size, seq_length),
    }

    # Create a mock iterator that returns a tuple
    mock_iterator = iter([(mock_batch, None, None)])

    # Mock CUDA movement
    def mock_cuda(non_blocking=True):
        return torch.ones_like(mock_batch["attention_mask"])

    for tensor in mock_batch.values():
        if isinstance(tensor, torch.Tensor):
            tensor.cuda = MagicMock(side_effect=mock_cuda)

    # Mock pipeline first stage check
    with patch('megatron.core.parallel_state.is_pipeline_first_stage', return_value=True):
        with patch('megatron.core.utils.get_batch_on_this_cp_rank', side_effect=lambda x: x):
            with patch('megatron.core.parallel_state.get_context_parallel_world_size', return_value=1):
                result = bert_embedding_data_step(mock_iterator)
    # Verify the output structure
    assert isinstance(result, dict)
    assert all(key in result for key in ["input_ids", "attention_mask", "token_type_ids"])


from unittest.mock import MagicMock

import pytest
import torch

from nemo.collections.llm.bert.model.embedding import bert_embedding_forward_step


def test_bert_embedding_forward_step():
    # Setup mock data
    batch_size = 2
    seq_length = 10
    hidden_size = 768

    # Create test batch
    batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length),
        "token_type_ids": torch.zeros(batch_size, seq_length),
        "extra_key": torch.ones(batch_size),  # This should be ignored
    }

    # Create mock model
    mock_model = MagicMock()
    # Mock the config attribute
    mock_model.config = MagicMock()
    mock_model.config.num_tokentypes = 2  # Set to test token type handling

    # Mock the forward pass to return a tensor
    expected_output = torch.randn(batch_size, hidden_size)
    mock_model.__call__ = MagicMock(return_value=expected_output)

    # Test standard forward pass
    bert_embedding_forward_step(mock_model, batch)


from unittest.mock import MagicMock, patch

import pytest

from nemo.collections.llm.bert.loss import BERTInBatchExclusiveHardNegativesRankingLoss
from nemo.collections.llm.bert.model.embedding import BertEmbeddingModel


def test_training_loss_reduction_initialization():
    # Create a basic config
    config = BertEmbeddingMiniConfig()

    # Create model instance with mock components
    model = BertEmbeddingModel(config=config, tokenizer=MagicMock())

    # Get the training loss reduction
    loss_reduction = model.training_loss_reduction

    # Verify it's the correct type
    assert isinstance(loss_reduction, BERTInBatchExclusiveHardNegativesRankingLoss)

    # Verify the configuration parameters were passed correctly
    assert loss_reduction.num_hard_negatives == config.num_hard_negatives
    assert loss_reduction.scale == config.ce_loss_scale


def test_validation_loss_reduction_initialization():
    # Create a basic config
    config = BertEmbeddingMiniConfig()

    # Create model instance with mock components
    model = BertEmbeddingModel(config=config, tokenizer=MagicMock())

    # Get the training loss reduction
    loss_reduction = model.validation_loss_reduction

    # Verify it's the correct type
    assert isinstance(loss_reduction, BERTInBatchExclusiveHardNegativesRankingLoss)

    # Verify the configuration parameters were passed correctly
    assert loss_reduction.num_hard_negatives == config.num_hard_negatives
    assert loss_reduction.scale == config.ce_loss_scale
