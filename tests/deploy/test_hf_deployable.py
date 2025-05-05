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

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo.deploy.nlp.hf_deployable import HuggingFaceLLMDeploy


@pytest.fixture
def mock_model():
    model = MagicMock(spec=AutoModelForCausalLM)
    model.generate = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    model.cuda = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.batch_decode = MagicMock(return_value=["Generated text"])
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    return tokenizer


@pytest.fixture
def mock_peft_model():
    with patch("nemo.deploy.nlp.hf_deployable.PeftModel") as mock:
        mock.from_pretrained.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_distributed():
    with patch("torch.distributed") as mock:
        mock.is_initialized.return_value = True
        mock.get_world_size.return_value = 2
        mock.get_rank.return_value = 1
        mock.broadcast = MagicMock(return_value=torch.tensor([0]))
        yield mock


@pytest.fixture
def mock_torch_cuda():
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.Tensor.cuda', return_value=torch.tensor([[1, 2, 3]])):
            yield


class MockRequest:
    def __init__(self, data):
        self.data = data
        self.span = None

    def __getitem__(self, key):
        return self.data[key]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()


class TestHuggingFaceLLMDeploy:

    def test_initialization_invalid_task(self):
        with pytest.raises(AssertionError):
            HuggingFaceLLMDeploy(hf_model_id_path="test/model", task="invalid-task")

    def test_initialization_no_model(self):
        with pytest.raises(ValueError):
            HuggingFaceLLMDeploy(task="text-generation")

    def test_initialization_with_model_and_tokenizer(self):
        model = MagicMock(spec=AutoModelForCausalLM)
        tokenizer = MagicMock(spec=AutoTokenizer)
        deployer = HuggingFaceLLMDeploy(model=model, tokenizer=tokenizer, task="text-generation")
        assert deployer.model == model
        assert deployer.tokenizer == tokenizer
        assert deployer.task == "text-generation"

    def test_initialization_with_model_path(self, mock_model, mock_tokenizer):
        with (
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            deployer = HuggingFaceLLMDeploy(hf_model_id_path="test/model", task="text-generation")
            assert deployer.model == mock_model
            assert deployer.tokenizer == mock_tokenizer

    def test_initialization_with_peft_model(self, mock_model, mock_tokenizer, mock_peft_model):
        with (
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        ):
            deployer = HuggingFaceLLMDeploy(
                hf_model_id_path="test/model", hf_peft_model_id_path="test/peft_model", task="text-generation"
            )
            assert deployer.model == mock_peft_model.from_pretrained.return_value

    def test_triton_input_output_config(self):
        deployer = HuggingFaceLLMDeploy(model=MagicMock(), tokenizer=MagicMock(), task="text-generation")

        inputs = deployer.get_triton_input
        outputs = deployer.get_triton_output

        assert len(inputs) == 10  # Verify number of input tensors
        assert len(outputs) == 3  # Verify number of output tensors

        # Verify required input tensor names
        assert any(tensor.name == "prompts" for tensor in inputs)
        assert any(tensor.name == "max_length" for tensor in inputs)

        # Verify output tensor names
        assert any(tensor.name == "sentences" for tensor in outputs)
        assert any(tensor.name == "logits" for tensor in outputs)
        assert any(tensor.name == "scores" for tensor in outputs)

    def test_generate_without_model(self):
        deployer = HuggingFaceLLMDeploy(model=MagicMock(), tokenizer=MagicMock(), task="text-generation")
        deployer.model = None
        with pytest.raises(RuntimeError):
            deployer.generate(text_inputs=["test prompt"])

    def test_generate_with_model(self, mock_model, mock_tokenizer, mock_torch_cuda):
        deployer = HuggingFaceLLMDeploy(model=mock_model, tokenizer=mock_tokenizer, task="text-generation")
        output = deployer.generate(text_inputs=["test prompt"])
        assert output == ["Generated text"]
        mock_model.generate.assert_called_once()
        mock_tokenizer.batch_decode.assert_called_once()

    def test_generate_with_output_logits_and_scores(self, mock_model, mock_tokenizer, mock_torch_cuda):
        mock_model.generate.return_value = {
            "sequences": torch.tensor([[1, 2, 3]]),
            "logits": torch.tensor([1.0]),
            "scores": torch.tensor([0.5]),
        }
        deployer = HuggingFaceLLMDeploy(model=mock_model, tokenizer=mock_tokenizer, task="text-generation")
        output = deployer.generate(
            text_inputs=["test prompt"], output_logits=True, output_scores=True, return_dict_in_generate=True
        )
        assert isinstance(output, dict)
        assert "sentences" in output
        assert "logits" in output
        assert "scores" in output

    def test_triton_infer_fn(self, mock_model, mock_tokenizer):
        deployer = HuggingFaceLLMDeploy(model=mock_model, tokenizer=mock_tokenizer, task="text-generation")
        request_data = {
            "prompts": np.array(["test prompt"]),
            "temperature": np.array([[1.0]]),
            "top_k": np.array([[1]]),
            "top_p": np.array([[0.0]]),
            "max_length": np.array([[10]]),
            "output_logits": np.array([[False]]),
            "output_scores": np.array([[False]]),
        }
        requests = [MockRequest(request_data)]
        output = deployer.triton_infer_fn(requests)
        assert "sentences" in output[0]
        assert isinstance(output[0]["sentences"], np.ndarray)

    def test_triton_infer_fn_with_error(self, mock_model, mock_tokenizer):
        deployer = HuggingFaceLLMDeploy(model=mock_model, tokenizer=mock_tokenizer, task="text-generation")
        mock_model.generate.side_effect = Exception("Test error")
        request_data = {
            "prompts": np.array(["test prompt"]),
            "temperature": np.array([[1.0]]),
            "top_k": np.array([[1]]),
            "top_p": np.array([[0.0]]),
            "max_length": np.array([[10]]),
            "output_logits": np.array([[False]]),
            "output_scores": np.array([[False]]),
        }
        requests = [MockRequest(request_data)]
        output = deployer.triton_infer_fn(requests)
        assert "sentences" in output[0]
        assert "An error occurred" in str(output[0]["sentences"][0])
