# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import torch

try:

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

from nemo.collections.nlp.models.language_modeling.megatron.falcon.falcon_decoder_layer import FalconTransformerLayer
from nemo.collections.nlp.models.language_modeling.megatron.falcon.falcon_spec import get_falcon_layer_spec


@pytest.mark.run_only_on('GPU')
class TestParallelFalconTransformerLayer:
    def setup_method(self, method):
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.parallel_falcon_transformer_layer = FalconTransformerLayer(
            transformer_config, get_falcon_layer_spec().submodules
        )

    @pytest.mark.unit
    def test_constructor(self):
        parallel_falcon_transformer_layer = self.parallel_falcon_transformer_layer
        assert isinstance(parallel_falcon_transformer_layer, FalconTransformerLayer)
        assert parallel_falcon_transformer_layer.layer_number == 1

        num_weights = sum([p.numel() for p in parallel_falcon_transformer_layer.parameters()])
        assert num_weights == 1884

    @pytest.mark.unit
    def test_gpu_forward(self):
        parallel_transformer_layer = self.parallel_falcon_transformer_layer
        config: TransformerConfig = parallel_transformer_layer.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_layer.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
        hidden_states, context = parallel_transformer_layer(hidden_states=hidden_states, attention_mask=attention_mask)
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size
