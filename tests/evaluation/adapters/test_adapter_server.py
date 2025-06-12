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
        max_logged_responses=1,
        max_logged_requests=1,
    )

    # Create server process and get a reference instance for config
    p, adapter_config = create_server_process(adapter_config)

    yield adapter_config

    p.terminate()


def test_adapter_server_post_request(adapter_server, capfd):

    url = f"http://localhost:{adapter_server.local_port}"
    data = {
        "prompt": "This is a test prompt",
        "max_tokens": 100,
        "temperature": 0.5,
    }

    response = requests.post(url, json=data)
    assert response.status_code == 200
    assert "choices" in response.json()
    assert len(response.json()["choices"]) > 0

    response = requests.post(url, json=data)
    assert response.status_code == 200
    assert "choices" in response.json()
    assert len(response.json()["choices"]) > 0
    # We also test that reasoning has gone
    assert "</think>" not in response.json()["choices"][0]["message"]["content"]
