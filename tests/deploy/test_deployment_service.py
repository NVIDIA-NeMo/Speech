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

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nemo.deploy.service.fastapi_interface_to_pytriton import (
    CompletionRequest,
    TritonSettings,
    _helper_fun,
    app,
    convert_numpy,
    dict_to_str,
    query_llm_async,
)
from nemo.deploy.service.rest_model_api import CompletionRequest as RestCompletionRequest
from nemo.deploy.service.rest_model_api import TritonSettings as RestTritonSettings
from nemo.deploy.service.rest_model_api import app as rest_app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_triton_settings():
    with patch('nemo.deploy.service.fastapi_interface_to_pytriton.TritonSettings') as mock:
        instance = mock.return_value
        instance.triton_service_port = 8000
        instance.triton_service_ip = "localhost"
        yield instance


@pytest.fixture
def rest_client():
    return TestClient(rest_app)


@pytest.fixture
def mock_rest_triton_settings():
    with patch('nemo.deploy.service.rest_model_api.TritonSettings') as mock:
        instance = mock.return_value
        instance.triton_service_port = 8080
        instance.triton_service_ip = "localhost"
        instance.triton_request_timeout = 60
        instance.openai_format_response = False
        instance.output_generation_logits = False
        yield instance


class TestTritonSettings:
    def test_default_values(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = TritonSettings()
            assert settings.triton_service_port == 8000
            assert settings.triton_service_ip == "0.0.0.0"

    def test_custom_values(self):
        with patch.dict(os.environ, {'TRITON_PORT': '9000', 'TRITON_HTTP_ADDRESS': '127.0.0.1'}, clear=True):
            settings = TritonSettings()
            assert settings.triton_service_port == 9000
            assert settings.triton_service_ip == "127.0.0.1"


class TestCompletionRequest:
    def test_default_values(self):
        request = CompletionRequest(model="test_model")
        assert request.model == "test_model"
        assert request.prompt == "hello"
        assert request.messages == [{}]
        assert request.max_tokens == 512
        assert request.temperature == 1.0
        assert request.top_p == 0.0
        assert request.top_k == 0
        assert request.logprobs is None

    def test_greedy_params(self):
        request = CompletionRequest(model="test_model", temperature=0.0, top_p=0.0)
        assert request.top_k == 1


class TestHealthEndpoints:
    def test_health_check(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestUtilityFunctions:
    def test_convert_numpy(self):
        # Test with numpy array
        arr = np.array([1, 2, 3])
        assert convert_numpy(arr) == [1, 2, 3]

        # Test with nested dictionary
        nested = {"a": np.array([1, 2]), "b": {"c": np.array([3, 4])}}
        assert convert_numpy(nested) == {"a": [1, 2], "b": {"c": [3, 4]}}

        # Test with list
        lst = [np.array([1, 2]), np.array([3, 4])]
        assert convert_numpy(lst) == [[1, 2], [3, 4]]

    def test_dict_to_str(self):
        test_dict = {"key": "value", "number": 42}
        result = dict_to_str(test_dict)
        assert isinstance(result, str)
        assert json.loads(result) == test_dict


class TestLLMQueryFunctions:
    def test_helper_fun(self):
        mock_nq = MagicMock()
        mock_nq.query_llm.return_value = {"test": "response"}

        with patch('nemo.deploy.service.fastapi_interface_to_pytriton.NemoQueryLLMPyTorch', return_value=mock_nq):
            result = _helper_fun(
                url="http://test",
                model="test_model",
                prompts=["test prompt"],
                temperature=0.7,
                top_k=10,
                top_p=0.9,
                compute_logprob=True,
                max_length=100,
                apply_chat_template=False,
            )
            assert result == {"test": "response"}
            mock_nq.query_llm.assert_called_once()

    def test_query_llm_async(self):
        mock_result = {"test": "response"}
        with patch('nemo.deploy.service.fastapi_interface_to_pytriton._helper_fun', return_value=mock_result):
            # Create an event loop and run the async function
            import asyncio

            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                query_llm_async(
                    url="http://test",
                    model="test_model",
                    prompts=["test prompt"],
                    temperature=0.7,
                    top_k=10,
                    top_p=0.9,
                    compute_logprob=True,
                    max_length=100,
                    apply_chat_template=False,
                )
            )
            assert result == mock_result


class TestAPIEndpoints:
    def test_completions_v1(self, client):
        mock_output = {
            "choices": [
                {
                    "text": [["test response"]],
                    "logprobs": {"token_logprobs": [[1.0, 2.0]], "top_logprobs": [[{"a": 0.5}, {"b": 0.5}]]},
                }
            ]
        }

        with patch('nemo.deploy.service.fastapi_interface_to_pytriton.query_llm_async', return_value=mock_output):
            response = client.post(
                "/v1/completions/", json={"model": "test_model", "prompt": "test prompt", "logprobs": 1}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["text"] == "test response"
            assert "logprobs" in data["choices"][0]

    def test_chat_completions_v1(self, client):
        mock_output = {"choices": [{"text": [["test response"]]}]}

        with patch('nemo.deploy.service.fastapi_interface_to_pytriton.query_llm_async', return_value=mock_output):
            response = client.post(
                "/v1/chat/completions/",
                json={"model": "test_model", "messages": [{"role": "user", "content": "test message"}]},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert data["choices"][0]["message"]["content"] == "test response"


class TestRestTritonSettings:
    def test_default_values(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = RestTritonSettings()
            assert settings.triton_service_port == 8080
            assert settings.triton_service_ip == "0.0.0.0"
            assert settings.triton_request_timeout == 60
            assert settings.openai_format_response is False
            assert settings.output_generation_logits is False

    def test_custom_values(self):
        with patch.dict(
            os.environ,
            {
                'TRITON_PORT': '9000',
                'TRITON_HTTP_ADDRESS': '127.0.0.1',
                'TRITON_REQUEST_TIMEOUT': '120',
                'OPENAI_FORMAT_RESPONSE': 'True',
                'OUTPUT_GENERATION_LOGITS': 'True',
            },
            clear=True,
        ):
            settings = RestTritonSettings()
            assert settings.triton_service_port == 9000
            assert settings.triton_service_ip == "127.0.0.1"
            assert settings.triton_request_timeout == 120
            assert settings.openai_format_response is True
            assert settings.output_generation_logits is True


class TestRestCompletionRequest:
    def test_default_values(self):
        request = RestCompletionRequest(model="test_model", prompt="test prompt")
        assert request.model == "test_model"
        assert request.prompt == "test prompt"
        assert request.max_tokens == 512
        assert request.temperature == 1.0
        assert request.top_p == 0.0
        assert request.top_k == 1
        assert request.stream is False
        assert request.stop is None
        assert request.frequency_penalty == 1.0


class TestRestHealthEndpoints:
    def test_health_check(self, rest_client):
        response = rest_client.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_triton_health_success(self, rest_client):
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            response = rest_client.get("/v1/triton_health")
            assert response.status_code == 200
            assert response.json() == {"status": "Triton server is reachable and ready"}


class TestRestCompletionsEndpoint:
    def test_completions_success(self, rest_client):
        mock_output = [["test response"]]
        with patch('nemo.deploy.service.rest_model_api.NemoQueryLLM') as mock_llm:
            mock_instance = mock_llm.return_value
            mock_instance.query_llm.return_value = mock_output

            response = rest_client.post(
                "/v1/completions/",
                json={
                    "model": "test_model",
                    "prompt": "test prompt",
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 10,
                },
            )
            assert response.status_code == 200
            assert response.json() == {"output": "test response"}

    def test_completions_standard_format(self, rest_client, mock_rest_triton_settings):
        mock_output = [["test response"]]
        mock_rest_triton_settings.openai_format_response = False

        with patch('nemo.deploy.service.rest_model_api.NemoQueryLLM') as mock_llm:
            mock_instance = mock_llm.return_value
            mock_instance.query_llm.return_value = mock_output

            response = rest_client.post("/v1/completions/", json={"model": "test_model", "prompt": "test prompt"})
            assert response.status_code == 200
            assert response.json() == {"output": "test response"}

    def test_completions_error_handling(self, rest_client):
        with patch('nemo.deploy.service.rest_model_api.NemoQueryLLM') as mock_llm:
            mock_instance = mock_llm.return_value
            mock_instance.query_llm.side_effect = Exception("Test error")

            response = rest_client.post("/v1/completions/", json={"model": "test_model", "prompt": "test prompt"})
            assert response.status_code == 200
            assert response.json() == {"error": "An exception occurred"}
