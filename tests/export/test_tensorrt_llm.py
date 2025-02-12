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

import re

import pytest


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_get_nemo_to_trtllm_conversion_dict_on_nemo_model():
    try:
        from nemo.export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")

    dummy_state = object()
    model_state_dict = {
        'model.embedding.word_embeddings.weight': dummy_state,
        'model.decoder.layers.0.self_attention.linear_proj.weight': dummy_state,
    }
    nemo_model_conversion_dict = TensorRTLLM.get_nemo_to_trtllm_conversion_dict(model_state_dict)

    # Check that every key starts with 'model.' and not 'model..' by using a regex
    # This pattern ensures:
    #   - The key starts with 'model.'
    #   - Immediately after 'model.', there must be at least one character that is NOT a '.'
    #     (preventing the 'model..' scenario)
    pattern = re.compile(r'^model\.[^.].*')
    for key in nemo_model_conversion_dict.keys():
        assert pattern.match(key), f"Key '{key}' does not properly start with 'model.'"


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_get_nemo_to_trtllm_conversion_dict_on_mcore_model():
    try:
        from megatron.core.export.trtllm.model_to_trllm_mapping.default_conversion_dict import DEFAULT_CONVERSION_DICT

        from nemo.export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")

    dummy_state = object()
    model_state_dict = {
        'embedding.word_embeddings.weight': dummy_state,
        'decoder.layers.0.self_attention.linear_proj.weight': dummy_state,
    }
    nemo_model_conversion_dict = TensorRTLLM.get_nemo_to_trtllm_conversion_dict(model_state_dict)

    # This is essentially a no-op
    assert nemo_model_conversion_dict == DEFAULT_CONVERSION_DICT
