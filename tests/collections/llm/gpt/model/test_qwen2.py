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

from nemo.collections.llm.gpt.model.qwen2 import (
    Qwen2Config,
    Qwen2Config1P5B,
    Qwen2Config7B,
    Qwen2Config72B,
    Qwen2Config500M,
    Qwen25Config3B,
)


def test_qwen2_config():
    config = Qwen2Config(num_layers=24, hidden_size=896, num_attention_heads=14)
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.gated_linear_unit is True
    assert config.add_bias_linear is False
    assert config.add_qkv_bias is True
    assert config.seq_length == 4096
    assert config.init_method_std == 0.02
    assert config.hidden_dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.vocab_size == 151936
    assert config.share_embeddings_and_output_weights is False
    assert config.layernorm_epsilon == 1e-6
    assert config.rotary_base == 1000000.0
    assert config.position_embedding_type == "rope"


def test_qwen2_config_500m():
    config = Qwen2Config500M()
    assert config.num_layers == 24
    assert config.hidden_size == 896
    assert config.num_attention_heads == 14
    assert config.num_query_groups == 2
    assert config.ffn_hidden_size == 4864


def test_qwen2_config_1p5b():
    config = Qwen2Config1P5B()
    assert config.num_layers == 28
    assert config.hidden_size == 1536
    assert config.num_attention_heads == 12
    assert config.num_query_groups == 2
    assert config.ffn_hidden_size == 8960


def test_qwen25_config_3B():
    config = Qwen25Config3B()
    assert config.num_layers == 36
    assert config.hidden_size == 2048
    assert config.num_attention_heads == 16
    assert config.num_query_groups == 2
    assert config.ffn_hidden_size == 11008


def test_qwen2_config_7b():
    config = Qwen2Config7B()
    assert config.num_layers == 28
    assert config.hidden_size == 3584
    assert config.num_attention_heads == 28
    assert config.num_query_groups == 4
    assert config.ffn_hidden_size == 18944
    assert config.vocab_size == 152064


def test_qwen2_config_72b():
    config = Qwen2Config72B()
    assert config.num_layers == 80
    assert config.hidden_size == 8192
    assert config.num_attention_heads == 64
    assert config.num_query_groups == 8
    assert config.ffn_hidden_size == 29568
    assert config.vocab_size == 152064
    assert config.layernorm_epsilon == 1e-5
    assert config.vocab_size == 152064
