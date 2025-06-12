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


"""Utilities for adapter testing."""

import multiprocessing

import pytest
from flask import Flask, jsonify, request

from nemo.collections.llm.evaluation.adapters.utils import wait_for_server

DEFAULT_FAKE_RESPONSE = {
    "object": "chat.completion",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "This is a fake LLM response</think>This survives reasoning",
            }
        }
    ],
}


def create_and_run_fake_endpoint():
    """Create and run a fake OpenAI API endpoint."""
    app = Flask(__name__)

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completion():
        data = request.json
        if "fake_response" in data:
            response = data["fake_response"]
        else:
            response = DEFAULT_FAKE_RESPONSE
        return jsonify(response)

    app.run(host="localhost", port=3300)


def create_fake_endpoint_process():
    """Create a process running a fake OpenAI endpoint.

    Returns:
        The multiprocessing.Process object running the endpoint.
    """
    p = multiprocessing.Process(target=create_and_run_fake_endpoint)
    p.start()

    # Wait for the server to be ready
    if not wait_for_server("localhost", 3300):
        p.terminate()
        pytest.fail("Fake OpenAI endpoint did not start within the timeout period")

    return p
