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

import pytest


@pytest.mark.parametrize(
    'input_layer_names,expected_model_prefix',
    [
        (
            [
                'model.embedding.word_embeddings.weight',
                'model.decoder.layers.0.self_attention.linear_proj.weight',
                'model.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight',
                'model.decoder.layers.0.self_attention.linear_qkv.weight',
                'model.decoder.layers.0.mlp.linear_fc1.layer_norm_weight',
                'model.decoder.layers.0.mlp.linear_fc1.weight',
                'model.decoder.layers.0.mlp.linear_fc2.weight',
            ],
            'model.',
        )
    ],
)
@pytest.mark.run_only_on('CPU')
@pytest.mark.unit
def test_get_layer_prefix_is_mcore(input_layer_names, expected_model_prefix):
    try:
        from nemo.export.trt_llm.converter.model_to_trt_llm_ckpt import get_layer_prefix
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
    model_prefix, _ = get_layer_prefix(input_layer_names, is_mcore=True)
    assert model_prefix == expected_model_prefix
