# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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

from nemo.collections import llm
from nemo.collections.llm.gpt.model.hyena import (
    HuggingFaceSavannaHyenaImporter,
    Hyena1bConfig,
    Hyena7bARCLongContextConfig,
    Hyena7bConfig,
    Hyena40bARCLongContextConfig,
    Hyena40bConfig,
    HyenaConfig,
    HyenaNV1bConfig,
    HyenaNV7bConfig,
    HyenaNV40bConfig,
    HyenaNVTestConfig,
    HyenaTestConfig,
)


def test_hyena_base_config():
    config = HyenaConfig()
    assert config.num_layers == 2
    assert config.hidden_size == 1024
    assert config.num_attention_heads == 8
    assert config.seq_length == 2048
    assert config.position_embedding_type == "rope"
    assert config.make_vocab_size_divisible_by == 128
    assert config.num_groups_hyena == None
    assert config.num_groups_hyena_medium == None
    assert config.num_groups_hyena_short == None
    assert config.hybrid_attention_ratio == 0.0
    assert config.hybrid_mlp_ratio == 0.0
    assert config.gated_linear_unit == True


def test_hyena_7b_config():
    config = Hyena7bConfig()
    assert config.num_layers == 32
    assert config.hidden_size == 4096
    assert config.num_attention_heads == 32
    assert config.seq_length == 8192
    assert config.num_groups_hyena == 4096
    assert config.num_groups_hyena_medium == 256
    assert config.num_groups_hyena_short == 256
    assert config.hybrid_override_pattern == "SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*"


def test_hyena_nv_7b_config():
    config = HyenaNV7bConfig()
    assert config.num_layers == 32
    assert config.hidden_size == 4096
    assert config.num_attention_heads == 32
    assert config.seq_length == 8192
    assert config.tokenizer_library == "byte-level"
    assert config.mapping_type == "base"
    assert config.use_short_conv_bias is True


def test_hyena_1b_config():
    config = Hyena1bConfig()
    assert config.num_layers == 25
    assert config.hidden_size == 1920
    assert config.num_attention_heads == 15
    assert config.seq_length == 8192
    assert config.num_groups_hyena == 1920
    assert config.num_groups_hyena_medium == 128
    assert config.num_groups_hyena_short == 128


def test_hyena_nv_1b_config():
    config = HyenaNV1bConfig()
    assert config.num_layers == 25
    assert config.hidden_size == 1920
    assert config.ffn_hidden_size == 5120
    assert config.num_attention_heads == 15
    assert config.seq_length == 8192
    assert config.tokenizer_library == "byte-level"


def test_hyena_40b_config():
    config = Hyena40bConfig()
    assert config.num_layers == 50
    assert config.hidden_size == 8192
    assert config.num_attention_heads == 64
    assert config.seq_length == 8192
    assert config.num_groups_hyena == 8192
    assert config.num_groups_hyena_medium == 512
    assert config.num_groups_hyena_short == 512


def test_hyena_nv_40b_config():
    config = HyenaNV40bConfig()
    assert config.num_layers == 50
    assert config.hidden_size == 8192
    assert config.ffn_hidden_size == 22528
    assert config.num_attention_heads == 64
    assert config.seq_length == 8192
    assert config.tokenizer_library == "byte-level"


def test_hyena_7b_arc_long_context_config():
    config = Hyena7bARCLongContextConfig()
    assert config.num_layers == 32
    assert config.hidden_size == 4096
    assert config.ffn_hidden_size == 11264
    assert config.num_attention_heads == 32
    assert config.seq_length == 1048576  # ~1M or 2**20
    assert config.tokenizer_library == "byte-level"


def test_hyena_40b_arc_long_context_config():
    config = Hyena40bARCLongContextConfig()
    assert config.num_layers == 50
    assert config.hidden_size == 8192
    assert config.ffn_hidden_size == 22528
    assert config.num_attention_heads == 64
    assert config.seq_length == 1048576  # ~1M or 2**20
    assert config.tokenizer_library == "byte-level"


def test_hyena_test_config():
    config = HyenaTestConfig()
    assert config.num_layers == 4
    assert config.hidden_size == 4096
    assert config.num_attention_heads == 32
    assert config.seq_length == 8192
    assert config.hybrid_override_pattern == "SDH*"


def test_hyena_nv_test_config():
    config = HyenaNVTestConfig()
    assert config.num_layers == 4
    assert config.hidden_size == 4096
    assert config.ffn_hidden_size == 11008
    assert config.num_attention_heads == 32
    assert config.seq_length == 8192
    assert config.tokenizer_library == "byte-level"


@pytest.mark.pleasefixme
def test_convert_hyena():
    from huggingface_hub.utils import RepositoryNotFoundError

    evo2_config = llm.Hyena1bConfig()
    model_ckpt = "dummy_model_deosnt_exist"
    exporter = HuggingFaceSavannaHyenaImporter(model_ckpt, model_config=evo2_config)

    with pytest.raises(RepositoryNotFoundError):
        exporter.apply("dummy_output_deosnt_exist")
