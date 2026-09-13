# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from __future__ import annotations

from types import SimpleNamespace

import benchmark_model as benchmark  # noqa: E402
import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams


@pytest.mark.asyncio
async def test_model_benchmark_uses_service_protocol_for_one_token_chunks():
    meta = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda _text, add_special_tokens=False: [10, 11, 12]),
        speaker_id="eng",
        speaker_embedding=None,
        prompt_len=2,
        text_eos_id=99,
    )
    params = SamplingParams(max_tokens=32, output_kind=RequestOutputKind.DELTA)
    inputs, observe_output = benchmark.build_streaming_request(
        "ignored",
        meta,
        params,
        max_new_tokens=32,
        tokens_per_chunk=1,
    )

    expected = [
        ({"text_token": [10], "text_token_start": 0}, 1),
        ({"text_token": [11], "text_token_start": 1}, 1),
        ({"text_token": [12], "text_token_start": 2}, 1),
        ({"text_token": [99], "text_token_start": 3}, 1),
    ]
    for info, max_tokens in expected:
        engine_input = await anext(inputs)
        assert engine_input.prompt["additional_information"]["text_token"] == info["text_token"]
        assert engine_input.prompt["additional_information"]["text_token_start"] == info["text_token_start"]
        assert engine_input.sampling_params.max_tokens == max_tokens
        observe_output(
            SimpleNamespace(
                stage_id=0,
                outputs=[SimpleNamespace(token_ids=[1] * max_tokens, finish_reason="length")],
            )
        )

    tail = await anext(inputs)
    assert tail.prompt["additional_information"] == {"text_token": []}
    assert tail.sampling_params.max_tokens == 28
