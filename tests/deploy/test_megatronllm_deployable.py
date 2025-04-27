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


from unittest.mock import MagicMock, patch

import pytest

from nemo.deploy.nlp.megatronllm_deployable import MegatronLLMDeployableNemo2


@pytest.fixture
def mock_model_and_tokenizer():
    """Fixture to mock the model and tokenizer setup."""
    with patch('nemo.collections.llm.inference.setup_model_and_tokenizer') as mock_setup:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenizer.tokenizer = MagicMock()
        mock_tokenizer.tokenizer.tokenizer.chat_template = "{{messages}}"
        mock_tokenizer.tokenizer.tokenizer.bos_token = "<bos>"
        mock_tokenizer.tokenizer.tokenizer.eos_token = "<eos>"
        mock_setup.return_value = (mock_model, mock_tokenizer)
        yield mock_setup


@pytest.fixture
def deployable(mock_model_and_tokenizer):
    """Fixture to create a deployable instance with mocked dependencies."""
    return MegatronLLMDeployableNemo2(
        nemo_checkpoint_filepath="dummy.nemo",
        num_devices=1,
        num_nodes=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )


@pytest.mark.run_only_on("GPU")
def test_initialization(deployable, mock_model_and_tokenizer):
    """Test initialization of the deployable class."""
    assert deployable.nemo_checkpoint_filepath == "dummy.nemo"
    mock_model_and_tokenizer.assert_called_once()


@pytest.mark.run_only_on("GPU")
def test_generate(deployable):
    """Test text generation functionality."""
    prompts = ["Hello", "World"]
    max_batch_size = 4
    random_seed = 42

    # Mock the inference.generate function
    with patch('nemo.collections.llm.inference.generate') as mock_generate:
        mock_generate.return_value = [MagicMock(generated_text="Generated text")]
        results = deployable.generate(prompts, max_batch_size, random_seed=random_seed)

        assert len(results) == 1
        mock_generate.assert_called_once()


@pytest.mark.run_only_on("GPU")
def test_apply_chat_template(deployable):
    """Test chat template application."""
    messages = [{"role": "user", "content": "Hello"}]
    template = deployable.apply_chat_template(messages)
    assert isinstance(template, str)
    assert messages[0]["content"] in template


@pytest.mark.run_only_on("GPU")
def test_remove_eos_token(deployable):
    """Test EOS token removal."""
    texts = ["Hello<eos>", "World", "Test<eos>"]
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == ["Hello", "World", "Test"]


@pytest.mark.run_only_on("GPU")
def test_str_to_dict(deployable):
    """Test string to dictionary conversion."""
    json_str = '{"key": "value"}'
    result = deployable.str_to_dict(json_str)
    assert isinstance(result, dict)
    assert result["key"] == "value"


@pytest.mark.run_only_on("GPU")
def test_triton_input_output(deployable):
    """Test Triton input and output tensor definitions."""
    inputs = deployable.get_triton_input
    outputs = deployable.get_triton_output

    assert len(inputs) == 9  # Number of input tensors
    assert len(outputs) == 2  # Number of output tensors

    # Check input tensor names
    input_names = [tensor.name for tensor in inputs]
    assert "prompts" in input_names
    assert "max_length" in input_names
    assert "max_batch_size" in input_names

    # Check output tensor names
    output_names = [tensor.name for tensor in outputs]
    assert "sentences" in output_names
    assert "log_probs" in output_names
