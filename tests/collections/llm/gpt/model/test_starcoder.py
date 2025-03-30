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

import torch.nn.functional as F

from nemo.collections.llm.gpt.model.starcoder import StarcoderConfig, StarcoderConfig15B


def test_starcoder_config():
    config = StarcoderConfig(num_layers=40, num_attention_heads=48, hidden_size=6144)
    assert config.normalization == "LayerNorm"
    assert config.activation_func == F.gelu
    assert config.add_bias_linear is True
    assert config.seq_length == 8192
    assert config.position_embedding_type == "learned_absolute"
    assert config.hidden_dropout == 0.2
    assert config.attention_dropout == 0.2
    assert config.init_method_std == 0.01
    assert config.layernorm_epsilon == 1e-5
    assert config.share_embeddings_and_output_weights is False
    assert config.kv_channels == 6144 // 48
    assert config.num_query_groups == 1
    assert config.attention_softmax_in_fp32 is True
    assert config.bias_activation_fusion is True
    assert config.bias_dropout_fusion is True


def test_starcoder_config_15b():
    config = StarcoderConfig15B()
    assert config.num_layers == 40
    assert config.hidden_size == 6144
    assert config.ffn_hidden_size == 24576
    assert config.num_attention_heads == 48
    assert config.init_method_std == 0.02
