# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Known-speaker prompt length must follow the request's effective ``context_text``.

The model assembles ``[task? | speaker | context_text | target text prefill]`` from the request context,
falling back to ``"[EN]"`` when the context is missing or empty. The serving adapter must advertise the same
number of placeholder rows, otherwise vLLM truncates the prefill or the coverage assertion fails.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from easymagpie_vllm_omni.serving_adapter import _build_adapter_cls
from vllm import SamplingParams
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

if TYPE_CHECKING:
    from pathlib import Path

_SPEAKER_ROWS = 5
_TASK_ROWS = 1
_TEXT_PREFILL = 2
_DEFAULT_CONTEXT_ROWS = len("[EN]")


def _tokenize(text: str) -> list[int]:
    """One synthetic token per character, so lengths are easy to reason about."""
    return list(range(len(text)))


def _expected_rows(context_text: str) -> int:
    return _TASK_ROWS + _SPEAKER_ROWS + len(context_text) + _TEXT_PREFILL


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    (tmp_path / "config.json").write_text(json.dumps({"num_task_embeddings": 1}))
    (tmp_path / "speaker_embeddings").mkdir()
    torch.save({"speaker_encoding": torch.zeros(_SPEAKER_ROWS, 3)}, tmp_path / "speaker_embeddings" / "demon.pt")
    return tmp_path


@pytest.fixture
def adapter(model_dir: Path) -> Any:
    stage = SimpleNamespace(engine_args=SimpleNamespace(model_stage="easymagpie", scheduler_cls=None))
    instance = _build_adapter_cls()(SimpleNamespace(engine_client=SimpleNamespace(stage_configs=[stage])))
    instance._model_path_cache = str(model_dir)
    instance._model_config_cache = {"text_vocab_size": 100}
    instance._tokenize = lambda: _tokenize
    instance._text_stream_metadata = lambda: (99, _TEXT_PREFILL)
    instance._model_tokenizer = lambda: SimpleNamespace(encode=lambda *_args, **_kwargs: [10, 11, 12])
    instance._acoustic_info = lambda _request: {}
    return instance


@pytest.mark.parametrize(
    ("extra_params", "effective_context"),
    [
        (None, "[EN]"),
        ({"context_text": "[EN]"}, "[EN]"),
        ({"context_text": "Speaker: a longer custom context"}, "Speaker: a longer custom context"),
        ({"context_text": "x"}, "x"),
        ({"context_text": ""}, "[EN]"),
        ({"context_text": None}, "[EN]"),
    ],
)
@pytest.mark.asyncio
async def test_http_known_speaker_prompt_len_follows_effective_context(
    adapter: Any, extra_params: dict | None, effective_context: str
) -> None:
    request = OpenAICreateSpeechRequest(input="hello", voice="demon", extra_params=extra_params)
    prepared = await adapter.build(request, [], False)
    assert len(prepared.prompt["prompt_token_ids"]) == _expected_rows(effective_context)


@pytest.mark.parametrize(
    ("context_text", "effective_context"),
    [
        ("[EN]", "[EN]"),
        ("Speaker: a longer custom context", "Speaker: a longer custom context"),
        ("x", "x"),
        ("", "[EN]"),
    ],
)
def test_streaming_known_speaker_prompt_len_follows_effective_context(
    adapter: Any, context_text: str, effective_context: str
) -> None:
    request = OpenAICreateSpeechRequest(input="", voice="demon", extra_params={"context_text": context_text})
    spec = adapter.build_streaming_spec(request)
    assert len(spec.prefill_prompt["prompt_token_ids"]) == _expected_rows(effective_context)


@pytest.mark.asyncio
async def test_same_speaker_different_contexts_do_not_share_a_stale_length(adapter: Any) -> None:
    contexts = ["[EN]", "a much longer context than default", "y", "[EN]"]
    for context in contexts:
        request = OpenAICreateSpeechRequest(input="hello", voice="demon", extra_params={"context_text": context})
        prepared = await adapter.build(request, [], False)
        assert len(prepared.prompt["prompt_token_ids"]) == _expected_rows(context)


@pytest.mark.asyncio
async def test_async_history_bound_uses_context_aware_prompt_len(adapter: Any) -> None:
    args = adapter.ctx.engine_client.stage_configs[0].engine_args
    args.scheduler_cls = "easymagpie_vllm_omni.cfg_async_scheduler.EasyMagpieCFGAsyncScheduler"
    args.max_model_len = _expected_rows("[EN]") + 100
    adapter.ctx.engine_client.default_sampling_params_list = [SamplingParams(max_tokens=100)]

    fits = OpenAICreateSpeechRequest(input="hello", voice="demon")
    await adapter.build(fits, [], False)

    too_long = OpenAICreateSpeechRequest(input="hello", voice="demon", extra_params={"context_text": "[EN] more"})
    with pytest.raises(ValueError, match="full history exceeds max_model_len"):
        await adapter.build(too_long, [], False)


@pytest.mark.parametrize(("context_text", "effective_context"), [("Speaker: custom", "Speaker: custom"), ("", "[EN]")])
def test_get_prompt_len_accepts_request_context(model_dir: Path, context_text: str, effective_context: str) -> None:
    plen = EasyMagpieTTSForConditionalGeneration.get_prompt_len(
        "demon", str(model_dir), tokenize=_tokenize, context_text=context_text
    )
    assert plen == _TASK_ROWS + _SPEAKER_ROWS + len(effective_context)


@pytest.mark.asyncio
async def test_reference_audio_empty_context_uses_default_rows() -> None:
    class FakeServer:
        @staticmethod
        def _validate_ref_audio_format(ref_audio):
            return None

        @staticmethod
        async def _resolve_ref_audio(ref_audio):
            return [0.25, -0.5], 16000

    adapter = _build_adapter_cls()(SimpleNamespace(engine_client=None, server=FakeServer()))
    adapter._arch = lambda: SimpleNamespace(
        num_task_embeddings=1,
        audio_input_token_id=7,
        codec_input_sample_rate=16000,
        ensure_reference_audio_available=lambda: None,
    )
    adapter._text_stream_metadata = lambda: (99, _TEXT_PREFILL)
    adapter._model_tokenizer = lambda: SimpleNamespace(encode=lambda *_args, **_kwargs: [10, 11, 12])
    adapter._tokenize = lambda: _tokenize
    adapter._acoustic_info = lambda _request: {}
    request = SimpleNamespace(
        input="hello", voice=None, ref_audio="data:audio/wav;base64,AAAA", extra_params={"context_text": ""}
    )

    prepared = await adapter.build(request, sampling_params_list=[], has_inline_ref_audio=True)

    assert len(prepared.prompt["prompt_token_ids"]) == _TASK_ROWS + 1 + _DEFAULT_CONTEXT_ROWS + _TEXT_PREFILL
