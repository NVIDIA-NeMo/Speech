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

from nemo.collections.llm.bert.model.bert import (
    HuggingFaceBertBaseConfig,
    HuggingFaceBertLargeConfig,
    MegatronBertBaseConfig,
    MegatronBertLargeConfig,
)


def test_huggingface_bert_base_config():
    config = HuggingFaceBertBaseConfig()
    assert config.bert_type == 'huggingface'
    assert config.num_layers == 12
    assert config.hidden_size == 768
    assert config.ffn_hidden_size == 3072
    assert config.num_attention_heads == 12


def test_huggingface_bert_large_config():
    config = HuggingFaceBertLargeConfig()
    assert config.bert_type == 'huggingface'
    assert config.num_layers == 24
    assert config.hidden_size == 1024
    assert config.ffn_hidden_size == 4096
    assert config.num_attention_heads == 16


def test_megatron_bert_base_config():
    config = MegatronBertBaseConfig()
    assert config.bert_type == 'megatron'
    assert config.num_layers == 12
    assert config.hidden_size == 768
    assert config.ffn_hidden_size == 3072
    assert config.num_attention_heads == 12


def test_megatron_bert_large_config():
    config = MegatronBertLargeConfig()
    assert config.bert_type == 'megatron'
    assert config.num_layers == 24
    assert config.hidden_size == 1024
    assert config.ffn_hidden_size == 4096
    assert config.num_attention_heads == 16


from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

from nemo.collections.llm.bert.model.bert import (
    _export_embedding,
    _export_qkv,
    _export_qkv_bias,
    _import_embedding,
    _import_embedding_2,
    _import_output_bias,
    _import_qkv,
    _import_qkv_2,
    _import_qkv_bias,
    _import_qkv_bias_2,
)


@dataclass
class MockConfig:
    num_attention_heads: int = 12
    hidden_size: int = 768
    kv_channels: int = 64
    make_vocab_size_divisible_by: int = 128
    vocab_size: int = 30522


class TestBertTransforms:
    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.target.config = MockConfig()
        ctx.source.config = MockConfig()
        return ctx

    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        num_attention_heads: int = 12
        hidden_size: int = 768
        kv_channels: int = 64
        make_vocab_size_divisible_by: int = 128
        vocab_size: int = 30522

    class TestBertTransforms:
        @pytest.fixture
        def mock_ctx(self):
            ctx = MagicMock()
            ctx.target.config = MockConfig()
            ctx.source.config = MockConfig()
            return ctx

        def test_import_qkv(self, mock_ctx):
            hidden_size = 768
            head_size = 64
            num_heads = 12

            q = torch.randn(num_heads * head_size, hidden_size)
            k = torch.randn(num_heads * head_size, hidden_size)
            v = torch.randn(num_heads * head_size, hidden_size)

            # Test both import functions
            for transform_fn in [_import_qkv.transform, _import_qkv_2.transform]:
                result = transform_fn(mock_ctx, q, k, v)

                # Check output shape
                expected_shape = (3 * num_heads * head_size, hidden_size)
                assert result.shape == expected_shape

        def test_import_qkv_bias(self, mock_ctx):
            head_size = 64
            num_heads = 12

            qb = torch.randn(num_heads * head_size)
            kb = torch.randn(num_heads * head_size)
            vb = torch.randn(num_heads * head_size)

            # Test both bias import functions
            for transform_fn in [_import_qkv_bias.transform, _import_qkv_bias_2.transform]:
                result = transform_fn(mock_ctx, qb, kb, vb)

                # Check output shape
                expected_shape = (3 * num_heads * head_size,)
                assert result.shape == expected_shape

        def test_import_embedding(self, mock_ctx):
            vocab_size = 30000  # Less than divisible_by * n
            hidden_size = 768
            embedding = torch.randn(vocab_size, hidden_size)

            # Test both embedding import functions
            for transform_fn in [_import_embedding.transform, _import_embedding_2.transform]:
                result = transform_fn(mock_ctx, embedding)

                # Check padding
                expected_padded_size = int(torch.ceil(torch.tensor(vocab_size) / 128) * 128)
                assert result.shape == (expected_padded_size, hidden_size)
                # Check original values preserved
                torch.testing.assert_close(result[:vocab_size], embedding)
                # Check padding is zeros
                assert torch.all(result[vocab_size:] == 0)

        def test_import_output_bias(self, mock_ctx):
            vocab_size = 30000
            bias = torch.randn(vocab_size)

            result = _import_output_bias.transform(mock_ctx, bias)

            # Check padding
            expected_padded_size = int(torch.ceil(torch.tensor(vocab_size) / 128) * 128)
            assert result.shape == (expected_padded_size,)
            # Check original values preserved
            torch.testing.assert_close(result[:vocab_size], bias)
            # Check padding is zeros
            assert torch.all(result[vocab_size:] == 0)

        def test_export_qkv(self, mock_ctx):
            hidden_size = 768
            head_size = 64
            num_heads = 12

            # Create input tensor with shape [3 * num_heads * head_size, hidden_size]
            linear_qkv = torch.randn(3 * num_heads * head_size, hidden_size)

            q_proj, k_proj, v_proj = _export_qkv.transform(mock_ctx, linear_qkv)

            # Check output shapes
            assert q_proj.shape == (num_heads * head_size, hidden_size)
            assert k_proj.shape == (num_heads * head_size, hidden_size)
            assert v_proj.shape == (num_heads * head_size, hidden_size)

        def test_export_qkv_bias(self, mock_ctx):
            head_size = 64
            num_heads = 12

            # Create input bias tensor
            qkv_bias = torch.randn(3 * num_heads * head_size)

            q_bias, k_bias, v_bias = _export_qkv_bias.transform(mock_ctx, qkv_bias)

            # Check output shapes
            assert q_bias.shape == (num_heads * head_size,)
            assert k_bias.shape == (num_heads * head_size,)
            assert v_bias.shape == (num_heads * head_size,)

        def test_export_embedding(self, mock_ctx):
            vocab_size = 30522
            hidden_size = 768
            padded_vocab_size = 30720  # Next multiple of 128

            # Create padded embedding tensor
            embedding = torch.randn(padded_vocab_size, hidden_size)

            result = _export_embedding.transform(mock_ctx, embedding)

            # Check output shape matches vocab_size
            assert result.shape == (vocab_size, hidden_size)
            # Check values preserved
            torch.testing.assert_close(result, embedding[:vocab_size])
