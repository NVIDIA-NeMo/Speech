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

from nemo.collections.llm.gpt.model.gemma3 import Gemma3Config1B, Gemma3Config4B, Gemma3Config12B, Gemma3Config27B


def test_gemma3_1b_config():
    config = Gemma3Config1B()
    assert config.num_layers == 26
    assert config.hidden_size == 1152
    assert config.num_attention_heads == 4
    assert config.num_query_groups == 1
    assert config.kv_channels == 256
    assert config.ffn_hidden_size == 6912
    assert config.window_size == 512
    assert config.rotary_base == (10_000, 1_000_000)
    assert config.rope_scaling_factor == 1.0
    assert config.seq_length == 32768
    assert config.normalization == "RMSNorm"
    assert config.layernorm_zero_centered_gamma is True
    assert config.layernorm_epsilon == 1e-6
    assert config.gated_linear_unit is True
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.share_embeddings_and_output_weights is True
    assert config.is_vision_language is False
    assert config.vocab_size == 262_144


def test_gemma3_4b_config():
    config = Gemma3Config4B()
    assert config.num_layers == 34
    assert config.hidden_size == 2560
    assert config.num_attention_heads == 8
    assert config.num_query_groups == 4
    assert config.kv_channels == 256
    assert config.ffn_hidden_size == 10240
    assert config.window_size == 1024
    assert config.rotary_base == (10_000, 1_000_000)
    assert config.rope_scaling_factor == 8.0
    assert config.seq_length == 131072
    assert config.normalization == "RMSNorm"
    assert config.layernorm_zero_centered_gamma is True
    assert config.layernorm_epsilon == 1e-6
    assert config.gated_linear_unit is True
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.share_embeddings_and_output_weights is True
    assert config.is_vision_language is True


def test_gemma3_12b_config():
    config = Gemma3Config12B()
    assert config.num_layers == 48
    assert config.hidden_size == 3840
    assert config.num_attention_heads == 16
    assert config.num_query_groups == 8
    assert config.kv_channels == 256
    assert config.ffn_hidden_size == 15360
    assert config.window_size == 1024
    assert config.rotary_base == (10_000, 1_000_000)
    assert config.rope_scaling_factor == 8.0
    assert config.seq_length == 131072
    assert config.normalization == "RMSNorm"
    assert config.layernorm_zero_centered_gamma is True
    assert config.layernorm_epsilon == 1e-6
    assert config.gated_linear_unit is True
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.share_embeddings_and_output_weights is True
    assert config.is_vision_language is True


def test_gemma3_27b_config():
    config = Gemma3Config27B()
    assert config.num_layers == 62
    assert config.hidden_size == 5376
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 16
    assert config.kv_channels == 128
    assert config.softmax_scale == 1.0 / (168**0.5)  # Special scaling for 27B model
    assert config.ffn_hidden_size == 21504
    assert config.window_size == 1024
    assert config.rotary_base == (10_000, 1_000_000)
    assert config.rope_scaling_factor == 8.0
    assert config.seq_length == 131072
    assert config.normalization == "RMSNorm"
    assert config.layernorm_zero_centered_gamma is True
    assert config.layernorm_epsilon == 1e-6
    assert config.gated_linear_unit is True
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.share_embeddings_and_output_weights is True
    assert config.is_vision_language is True
