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

import torch.nn.functional as F

from nemo.collections.llm.gpt.model.llama_nemotron import (
    Llama31Nemotron70BConfig,
    Llama31NemotronNano8BConfig,
    Llama31NemotronUltra253BConfig,
    Llama33NemotronSuper49BConfig,
)


def test_llama31_nemotron_nano_8b_config():
    config = Llama31NemotronNano8BConfig()
    assert config.num_layers == 32
    assert config.hidden_size == 4096
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 14336
    assert config.kv_channels == 128
    assert config.rotary_base == 500_000
    assert config.seq_length == 131072
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.gated_linear_unit is True
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False


def test_llama31_nemotron_70b_config():
    config = Llama31Nemotron70BConfig()
    assert config.num_layers == 80
    assert config.hidden_size == 8192
    assert config.num_attention_heads == 64
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 28672
    assert config.kv_channels == 128
    assert config.rotary_base == 500_000
    assert config.seq_length == 131072
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.gated_linear_unit is True
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False
    assert config.make_vocab_size_divisible_by == 128


def test_llama33_nemotron_super_49b_config():
    config = Llama33NemotronSuper49BConfig()
    assert config.num_layers == 80
    assert config.hidden_size == 8192
    assert config.num_attention_heads == 64
    assert config.ffn_hidden_size == 28672
    assert config.rotary_base == 500_000
    assert config.seq_length == 131072
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.gated_linear_unit is True
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False
    assert config.make_vocab_size_divisible_by == 128
    assert config.heterogeneous_layers_config_encoded_json is not None
    assert config.transformer_layer_spec is not None


def test_llama33_nemotron_ultra_253b_config():
    config = Llama31NemotronUltra253BConfig()
    assert config.num_layers == 162
    assert config.hidden_size == 16384
    assert config.num_attention_heads == 128
    assert config.rotary_base == 500_000
    assert config.seq_length == 131072
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.gated_linear_unit is True
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False
    assert config.make_vocab_size_divisible_by == 128
    assert config.heterogeneous_layers_config_encoded_json is not None
    assert config.transformer_layer_spec is not None
