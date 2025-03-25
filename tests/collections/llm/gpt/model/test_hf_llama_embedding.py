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
from unittest.mock import Mock, patch

import pytest
import torch

from nemo.collections.llm.gpt.model.hf_llama_embedding import (
    LlamaBidirectionalConfig,
    LlamaBidirectionalForSequenceClassification,
    LlamaBidirectionalHFAdapter,
    LlamaBidirectionalModel,
    Pooling,
    get_llama_bidirectional_hf_model,
    pool,
)


@pytest.fixture
def sample_hidden_states():
    return torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [0.0, 0.0]]]
    )  # batch_size=2, seq_len=3, hidden_dim=2


@pytest.fixture
def sample_attention_mask():
    return torch.tensor([[1, 1, 1], [1, 1, 0]])  # all tokens are valid  # last token is padding


def test_pool_avg(sample_hidden_states, sample_attention_mask):
    result = pool(sample_hidden_states, sample_attention_mask, "avg")
    expected_first = torch.tensor([3.0, 4.0])  # average of all tokens in first sequence
    expected_second = torch.tensor([8.0, 9.0])  # average of first two tokens in second sequence

    assert torch.allclose(result[0], expected_first)
    assert torch.allclose(result[1], expected_second)


def test_pool_cls(sample_hidden_states, sample_attention_mask):
    result = pool(sample_hidden_states, sample_attention_mask, "cls")
    assert torch.allclose(result[0], sample_hidden_states[0][0])
    assert torch.allclose(result[1], sample_hidden_states[1][0])


def test_pool_last(sample_hidden_states, sample_attention_mask):
    result = pool(sample_hidden_states, sample_attention_mask, "last")
    # For the first sequence, should take last token
    # For the second sequence, should take last non-padding token
    assert torch.allclose(result[0], sample_hidden_states[0][-1])
    assert torch.allclose(result[1], sample_hidden_states[1][1])


def test_pool_invalid():
    with pytest.raises(ValueError):
        pool(torch.randn(2, 3, 4), torch.ones(2, 3), "invalid_pool_type")


class TestPoolingModule:
    def test_pooling_avg(self, sample_hidden_states, sample_attention_mask):
        pooling = Pooling("avg")
        result = pooling(sample_hidden_states, sample_attention_mask)
        assert result.shape == (2, 2)  # batch_size=2, hidden_dim=2

    def test_pooling_cls(self, sample_hidden_states, sample_attention_mask):
        pooling = Pooling("cls")
        result = pooling(sample_hidden_states, sample_attention_mask)
        assert torch.allclose(result, sample_hidden_states[:, 0, :])

    @pytest.mark.parametrize("pool_type", ["avg", "cls", "cls__left", "last", "last__right"])
    def test_batch_size_one(self, pool_type):
        pooling = Pooling(pool_type)
        hidden_states = torch.randn(1, 3, 2)
        attention_mask = torch.ones(1, 3)

        result = pooling(hidden_states, attention_mask)
        assert result.shape == (1, 2)


class TestLlamaBidirectionalHFAdapter:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.device = torch.device("cpu")
        return model

    @pytest.fixture
    def adapter(self, mock_model):
        pooling = Pooling("avg")
        return LlamaBidirectionalHFAdapter(model=mock_model, normalize=True, pooling_module=pooling)

    def test_forward(self, adapter, sample_hidden_states, sample_attention_mask):
        adapter.model.return_value = {"last_hidden_state": sample_hidden_states}

        result = adapter(input_ids=torch.ones(2, 3).long(), attention_mask=sample_attention_mask)

        assert result.shape == (2, 2)
        # Check if normalization was applied
        assert torch.allclose(torch.norm(result, dim=1), torch.ones(2))


@patch('nemo.collections.llm.gpt.model.hf_llama_embedding.AutoModel')
@patch('nemo.collections.llm.gpt.model.hf_llama_embedding.AutoTokenizer')
def test_get_llama_bidirectional_hf_model(mock_tokenizer_cls, mock_model_cls):
    # Setup mocks
    mock_tokenizer = Mock()
    mock_tokenizer.padding_side = "right"
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    mock_model = Mock()
    mock_model_cls.from_pretrained.return_value = mock_model

    # Test function
    model, tokenizer = get_llama_bidirectional_hf_model(
        model_name_or_path="dummy_path", normalize=True, pooling_mode="avg"
    )

    assert isinstance(model, LlamaBidirectionalHFAdapter)
    assert tokenizer == mock_tokenizer


class TestLlamaBidirectionalForSequenceClassification:
    @pytest.fixture
    def config(self):
        return LlamaBidirectionalConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            num_labels=3,
            pooling="avg",
            temperature=1.0,
        )

    @pytest.fixture
    def model(self, config):
        return LlamaBidirectionalForSequenceClassification(config)

    def test_model_initialization(self, model, config):
        assert isinstance(model.model, LlamaBidirectionalModel)
        assert model.config.num_labels == 3
        assert model.config.pooling == "avg"
        assert model.config.temperature == 1.0

    def test_forward_classification(self, model):
        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.randint(0, 3, (batch_size,))

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Check output structure
        assert outputs.loss is not None
        assert outputs.logits.shape == (batch_size, 3)  # (batch_size, num_labels)

    def test_forward_regression(self):
        # Create config for regression (num_labels=1)
        config = LlamaBidirectionalConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            num_labels=1,
            pooling="avg",
            temperature=1.0,
        )
        model = LlamaBidirectionalForSequenceClassification(config)

        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.randn(batch_size, 1)  # Continuous values for regression

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        assert outputs.loss is not None
        assert outputs.logits.shape == (batch_size, 1)

    def test_forward_multi_label(self):
        # Create config for multi-label classification
        config = LlamaBidirectionalConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            num_labels=3,
            pooling="avg",
            temperature=1.0,
            problem_type="multi_label_classification",
        )
        model = LlamaBidirectionalForSequenceClassification(config)

        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.randint(0, 2, (batch_size, 3)).float()  # Binary labels for each class

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        assert outputs.loss is not None
        assert outputs.logits.shape == (batch_size, 3)

    def test_different_pooling_types(self, config):
        for pooling in ["avg", "cls", "last"]:
            config.pooling = pooling
            model = LlamaBidirectionalForSequenceClassification(config)

            batch_size = 2
            seq_length = 4
            input_ids = torch.randint(0, 100, (batch_size, seq_length))
            attention_mask = torch.ones(batch_size, seq_length)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            assert outputs.logits.shape == (batch_size, config.num_labels)

    def test_forward_without_labels(self, model):
        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert outputs.loss is None
        assert outputs.logits.shape == (batch_size, 3)

    def test_temperature_scaling(self, config):
        # Test with different temperature values
        temperatures = [0.5, 1.0, 2.0]
        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        for temp in temperatures:
            config.temperature = temp
            model = LlamaBidirectionalForSequenceClassification(config)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # The logits should be scaled by the temperature
            assert outputs.logits.shape == (batch_size, config.num_labels)

    @pytest.mark.parametrize("return_dict", [True, False])
    def test_return_dict_option(self, model, return_dict):
        batch_size = 2
        seq_length = 4
        input_ids = torch.randint(0, 100, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)

        if return_dict:
            assert hasattr(outputs, "logits")
        else:
            assert isinstance(outputs, tuple)
            assert len(outputs) > 0
