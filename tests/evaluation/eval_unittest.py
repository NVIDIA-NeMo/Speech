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

import pytest
from pytest_httpserver import HTTPServer

from nemo.collections.llm.api import evaluate
from nemo.collections.llm.evaluation.api import ConfigParams, EvaluationConfig, EvaluationTarget


@pytest.fixture(scope="session")
def httpserver_listen_address():
    return ("127.0.0.1", 8000)


@pytest.mark.parametrize(
    "params",
    [
        {
            "top_p": 0.1,
            "temperature": 0.001,
        },
        {"limit_samples": 10},
        {"limit_samples": 0.1},
        {"max_new_tokens": 64},
        {"max_retries": 10, "parallelism": 16, "request_timeout": 100},
        {"task": "my_task", "extra": {"num_fewshot": 5, "tokenizer": "my_tokenizer"}},
    ],
)
def test_configuration(params: dict):
    eval_config = EvaluationConfig(type="custom", params=params)
    assert isinstance(eval_config.params, ConfigParams)
    assert eval_config.type == "custom"
    for param_name, param_value in params.items():
        assert getattr(eval_config.params, param_name) == param_value


def test_default_none_tokenizer():
    eval_config = EvaluationConfig(type="custom", params={"extra": {"num_fewshot": 5}})
    assert eval_config.type == "custom"
    assert eval_config.params.extra["tokenizer"] is None
    assert eval_config.params.extra["num_fewshot"] == 5


def test_evaluation(httpserver: HTTPServer):
    httpserver.expect_request("/v1/triton_health").respond_with_json(
        {"status": "Triton server is reachable and ready"}
    )
    httpserver.expect_request("/v1/completions/", method="POST").respond_with_json(
        {
            'id': 'cmpl-123456',
            'object': 'text_completion',
            'created': 1234567,
            'model': 'triton_model',
            'choices': [
                {
                    'text': ' Janet eats 3 eggs and bakes 4 eggs, so she has 16 - 3 - 4 = <<16-3-4=9>>9 eggs left.\n'
                    'She sells 9 eggs for $2 each, so she makes 9 x 2 = <<9*2=18>>18 dollars.\n#### 18'
                }
            ],
        },
    )
    target_config = EvaluationTarget(
        api_endpoint={"url": "http://localhost:8000/v1/completions/", "type": "completions"}
    )
    eval_config = EvaluationConfig(
        type="gsm8k",
        params=ConfigParams(
            extra={"tokenizer": "Qwen/Qwen2.5-0.5B", "num_fewshot": 13}, limit_samples=1, parallelism=1
        ),
    )

    results = evaluate(target_cfg=target_config, eval_cfg=eval_config)
    assert (
        results['tasks']['gsm8k']['metrics']['exact_match__strict-match']['scores']['exact_match__strict-match'][
            'value'
        ]
        == 1.0
    )
