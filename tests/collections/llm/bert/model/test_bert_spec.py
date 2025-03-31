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
from unittest.mock import MagicMock

import pytest
import torch
from megatron.core import parallel_state

from nemo.collections.llm.bert.model.bert_spec import (
    TransformerLayerWithPostLNSupport,
    get_bert_layer_local_spec_postln,
    get_bert_layer_with_transformer_engine_spec_postln,
)


class TestBertSpec:
    @pytest.fixture(autouse=True)
    def setup_parallel_state(self):
        """Initialize parallel state for testing"""
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(world_size=1, rank=0)
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        yield
        parallel_state.destroy_model_parallel()

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.hidden_size = 768
        config.layernorm_epsilon = 1e-5
        config.bias_dropout_fusion = True
        config.tensor_model_parallel_size = 1
        config.pipeline_model_parallel_size = 1
        config.sequence_parallel = False
        config.gradient_accumulation_fusion = False
        return config

    @pytest.fixture
    def mock_submodules_config(self):
        config = MagicMock()
        # Mock the necessary layer norms
        config.post_att_layernorm = MagicMock()
        config.post_mlp_layernorm = MagicMock()
        config.input_layernorm = MagicMock()
        config.self_attention = MagicMock()
        config.self_attn_bda = MagicMock()
        config.pre_cross_attn_layernorm = MagicMock()
        config.cross_attention = MagicMock()
        config.cross_attn_bda = MagicMock()
        config.pre_mlp_layernorm = MagicMock()
        config.mlp = MagicMock()
        config.mlp_bda = MagicMock()
        return config
