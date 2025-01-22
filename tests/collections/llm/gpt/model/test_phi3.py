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

from nemo.collections.llm.gpt.model.phi3mini import Phi3Config, Phi3ConfigMini


def test_Phi3_config():
    config = Phi3Config(
        num_layers=32, hidden_size=3072, num_attention_heads=32, num_query_groups=32, ffn_hidden_size=8192
    )
    assert config.normalization == "RMSNorm"
    assert config.activation_func == F.silu
    assert config.gated_linear_unit is True
    assert config.position_embedding_type == "rope"
    assert config.add_bias_linear is False
    assert config.seq_length == 4096
    assert config.attention_dropout == 0.0
    assert config.hidden_dropout == 0.0
    assert config.share_embeddings_and_output_weights is False


# individual model config tests below...
def test_phi3configmini():
    config = Phi3ConfigMini()
    assert config.num_layers == 32
    assert config.hidden_size == 3072
    assert config.ffn_hidden_size == 8192
    assert config.num_attention_heads == 32
    assert config.num_query_groups == 32
    assert config.rotary_base == 10000.0
    assert config.vocab_size == 32064
