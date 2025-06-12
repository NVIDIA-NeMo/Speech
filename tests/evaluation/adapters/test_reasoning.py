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


from typing import Any, Generator

import pytest
import requests

from nemo.collections.llm.evaluation.adapters.server import create_server_process
from nemo.collections.llm.evaluation.api import AdapterConfig


@pytest.fixture
def adapter_server(fake_openai_endpoint) -> Generator[AdapterConfig, Any, Any]:
    # Create serializable configuration
    adapter_config = AdapterConfig(
        api_url="http://localhost:3300/v1/chat/completions",
        use_reasoning=True,
        end_reasoning_token="</think>",
    )

    # Create server process and get a reference instance for config
    p, adapter_config = create_server_process(adapter_config)

    yield adapter_config

    p.terminate()


@pytest.mark.parametrize(
    "input_content,expected_content",
    [
        (
            "Let me think about this...\n<think>This is my reasoning process that should be removed</think>\nHere's my final answer.",
            "Here's my final answer.",
        ),
        (
            "No reasoning tokens in this response.",
            "No reasoning tokens in this response.",
        ),
        (
            "<think>First I'll analyze the problem\nThen I'll solve it step by step</think>Here's the solution.",
            "Here's the solution.",
        ),
    ],
)
def test_reasoning_responses(
    adapter_server,
    fake_openai_endpoint,
    input_content,
    expected_content,
):

    url = f"http://localhost:{adapter_server.local_port}"
    # We parametrize the response of the openai fake server.
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": input_content,
                }
            }
        ]
    }
    data = {
        "prompt": "This is a test prompt",
        "max_tokens": 100,
        "temperature": 0.5,
        "fake_response": response_data,
    }
    response = requests.post(url, json=data)

    assert response.status_code == 200
    cleaned_data = response.json()
    cleaned_content = cleaned_data["choices"][0]["message"]["content"]
    assert cleaned_content == expected_content


def test_multiple_choices(
    adapter_server,
    fake_openai_endpoint,
):
    # Given: A response with multiple choices containing reasoning tokens
    url = f"http://localhost:{adapter_server.local_port}"
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>Reasoning 1</think>Answer 1",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>Reasoning 2</think>Answer 2",
                }
            },
        ]
    }
    data = {
        "prompt": "This is a test prompt",
        "max_tokens": 100,
        "temperature": 0.5,
        "fake_response": response_data,
    }
    response = requests.post(url, json=data)

    # Then: The reasoning tokens should be removed from all choices
    assert response.status_code == 200
    cleaned_data = response.json()
    assert cleaned_data["choices"][0]["message"]["content"] == "Answer 1"
    assert cleaned_data["choices"][1]["message"]["content"] == "Answer 2"


def test_non_assistant_role(
    adapter_server,
    fake_openai_endpoint,
):
    # Given: A response with a non-assistant role message
    url = f"http://localhost:{adapter_server.local_port}"
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "system",
                    "content": "<think>This should not be processed</think>System message",
                }
            }
        ]
    }
    data = {
        "prompt": "This is a test prompt",
        "max_tokens": 100,
        "temperature": 0.5,
        "fake_response": response_data,
    }
    response = requests.post(url, json=data)

    # Then: The content should remain unchanged
    cleaned_data = response.json()
    assert (
        cleaned_data["choices"][0]["message"]["content"] == "<think>This should not be processed</think>System message"
    )
