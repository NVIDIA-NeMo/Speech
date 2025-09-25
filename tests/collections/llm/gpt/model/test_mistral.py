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

from nemo.collections.llm.gpt.model.mistral import (
    MistralConfig7B,
    MistralNeMoConfig12B,
    MistralNeMoConfig123B,
    MistralSmall3Config24B,
)


def test_mistral_config7b():
    config = MistralConfig7B()
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.gated_linear_unit is True
    assert config.num_layers == 32
    assert config.hidden_size == 4096
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 14336
    assert config.seq_length == 32768
    assert config.attention_dropout == 0.0
    assert config.hidden_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False
    assert config.init_method_std == 0.02
    assert config.layernorm_epsilon == 1e-5
    assert config.window_size == [4096, 0]


def test_mistral_nemo_config_12b():
    config = MistralNeMoConfig12B()
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.gated_linear_unit is True
    assert config.num_layers == 40
    assert config.hidden_size == 5120
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 14336
    assert config.seq_length == 4096
    assert config.attention_dropout == 0.0
    assert config.hidden_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False
    assert config.init_method_std == 0.02
    assert config.layernorm_epsilon == 1e-5
    assert config.window_size is None
    assert config.rotary_percent == 1.0
    assert config.rotary_base == 1000000.0
    assert config.kv_channels == 128


def test_mistral_nemo_config_123b():
    config = MistralNeMoConfig123B()
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.gated_linear_unit is True
    assert config.num_layers == 88
    assert config.hidden_size == 12288
    assert config.num_attention_heads == 96
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 28672
    assert config.seq_length == 4096
    assert config.attention_dropout == 0.0
    assert config.hidden_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False
    assert config.init_method_std == 0.02
    assert config.layernorm_epsilon == 1e-5
    assert config.window_size is None
    assert config.rotary_percent == 1.0
    assert config.rotary_base == 1000000.0
    assert config.kv_channels == 128


def test_mistral_small3_config_24b():
    config = MistralSmall3Config24B()
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.gated_linear_unit is True
    assert config.num_layers == 40
    assert config.hidden_size == 5120
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 32768
    assert config.seq_length == 32768
    assert config.attention_dropout == 0.0
    assert config.hidden_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False
    assert config.init_method_std == 0.02
    assert config.layernorm_epsilon == 1e-5
    assert config.window_size is None
    assert config.rotary_percent == 1.0
    assert config.rotary_base == 100000000.0
    assert config.kv_channels == 128
