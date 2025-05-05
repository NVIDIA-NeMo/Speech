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
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo.collections.llm.bert.model.base import (
    BertConfig,
    BertModel,
    bert_forward_step,
    get_batch_on_this_cp_rank,
    get_packed_seq_params,
)


class TestBertBase:
    @pytest.fixture
    def sample_batch(self):
        return {
            "text": torch.randint(0, 1000, (32, 128)),  # [batch_size, seq_len]
            "padding_mask": torch.ones(32, 128),  # [batch_size, seq_len]
            "labels": torch.randint(0, 1000, (32, 128)),  # [batch_size, seq_len]
            "loss_mask": torch.ones(32, 128),  # [batch_size, seq_len]
            "types": torch.zeros(32, 128),  # [batch_size, seq_len]
        }

    @pytest.fixture
    def sample_packed_batch(self):
        return {
            "cu_seqlens": torch.tensor([0, 10, 25, 35, -1, -1]),
            "max_seqlen": torch.tensor(15),
        }

    @pytest.fixture
    def basic_config(self):
        return BertConfig(
            num_layers=6,
            hidden_size=768,
            num_attention_heads=12,
        )

    def test_get_batch_on_this_cp_rank_no_cp(self, sample_batch):
        with patch('megatron.core.parallel_state') as mock_parallel_state:
            mock_parallel_state.get_context_parallel_world_size.return_value = 1

            result = get_batch_on_this_cp_rank(sample_batch)

            # When context parallel size is 1, should return original batch unchanged
            assert result == sample_batch

    def test_get_packed_seq_params(self, sample_packed_batch):
        params = get_packed_seq_params(sample_packed_batch)

        # Check that cu_seqlens was properly trimmed using cu_seqlens_argmin
        assert params.cu_seqlens_q.shape[0] == 4  # Should trim at index 4
        assert params.max_seqlen_q == sample_packed_batch["max_seqlen"]
        assert params.qkv_format == "thd"

    def test_bert_config_initialization(self, basic_config):
        assert basic_config.num_layers == 6
        assert basic_config.hidden_size == 768
        assert basic_config.num_attention_heads == 12
        assert basic_config.bert_type == "megatron"  # default value
        assert basic_config.add_pooler is True  # default value

    def test_bert_model_initialization(self, basic_config):
        tokenizer = MagicMock()
        tokenizer.vocab_size = 30000

        model = BertModel(config=basic_config, tokenizer=tokenizer)

        assert model.config == basic_config
        assert model.tokenizer == tokenizer
        assert hasattr(model, "optim")

    def test_bert_forward_step(self, basic_config, sample_batch):
        model = BertModel(config=basic_config, tokenizer=None)
        model.module = MagicMock()

        bert_forward_step(model, sample_batch)

        # Verify model was called with correct arguments
        model.module.assert_called_once()
        call_args = model.module.call_args[1]
        assert "input_ids" in call_args
        assert "attention_mask" in call_args
        assert "lm_labels" in call_args
        assert "loss_mask" in call_args

    def test_bert_forward_step_with_tokentypes(self, basic_config, sample_batch):
        basic_config.num_tokentypes = 2
        model = BertModel(config=basic_config, tokenizer=None)
        model.module = MagicMock()

        result = bert_forward_step(model, sample_batch)

        # Verify tokentype_ids was included in forward call
        call_args = model.module.call_args[1]
        assert "tokentype_ids" in call_args

    def test_bert_forward_step_with_packed_seqs(self, basic_config, sample_batch):
        model = BertModel(config=basic_config, tokenizer=None)
        model.module = MagicMock()

        # Add packed sequence params to batch
        sample_batch["cu_seqlens"] = torch.tensor([0, 10, 20])

        result = bert_forward_step(model, sample_batch)

        # Verify packed_seq_params was included
        call_args = model.module.call_args[1]
        assert "packed_seq_params" in call_args

    def test_bert_model_training_step(self, basic_config):
        model = BertModel(config=basic_config, tokenizer=None)
        model.forward_step = MagicMock()

        batch = {"dummy": "batch"}
        output = model.training_step(batch)

        model.forward_step.assert_called_once_with(batch)

    def test_bert_model_validation_step(self, basic_config):
        model = BertModel(config=basic_config, tokenizer=None)
        model.forward_step = MagicMock()

        batch = {"dummy": "batch"}
        output = model.validation_step(batch)

        model.forward_step.assert_called_once_with(batch)

    def test_get_batch_with_context_parallel(self, sample_batch):
        with patch('megatron.core.parallel_state') as mock_parallel_state:
            mock_parallel_state.get_context_parallel_world_size.return_value = 2
            mock_parallel_state.get_context_parallel_rank.return_value = 0

            result = get_batch_on_this_cp_rank(sample_batch)

            # Verify batch was properly split for context parallel processing
            for key, val in result.items():
                if val is not None:
                    if key != "attention_mask":
                        assert val.shape[1] == sample_batch[key].shape[1]
                    else:
                        assert val.shape[2] == sample_batch[key].shape[2]
