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
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from easymagpie_vllm_omni.serving_adapter import _build_adapter_cls
from easymagpie_vllm_omni.serving_stream import EasyMagpieStreamingSpeechHandler
from vllm import SamplingParams
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path


@pytest.fixture
def adapter(tmp_path: Path) -> Any:
    """Build a serving adapter with a synthetic CFG-capable converted model.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        A serving adapter configured for a synthetic registered speaker.
    """
    stage = SimpleNamespace(
        engine_args=SimpleNamespace(
            model_stage="easymagpie",
            scheduler_cls="easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGScheduler",
        )
    )
    instance = _build_adapter_cls()(SimpleNamespace(engine_client=SimpleNamespace(stage_configs=[stage])))
    instance._model_path_cache = str(tmp_path)
    instance._model_config_cache = {
        "acoustic_predictor_type": "acoustic_codes_predictor",
        "cfg_capability": {"version": 1, "context_cas_disabled": True},
        "cfg_unk_token_id": 3,
        "text_vocab_size": 100,
    }
    instance._text_stream_metadata = lambda: (99, 0)
    instance._prompt_len = lambda _speaker: 2
    instance._model_tokenizer = lambda: SimpleNamespace(encode=lambda *_args, **_kwargs: [10, 11])
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("scale", [0, 1, 2.5])
async def test_http_cfg_settings_reach_prompt(
    adapter: Any,
    scale: float,
) -> None:
    """Verify HTTP CFG settings reach prompt.

    Args:
        adapter: Wrapped codec transfer adapter or adapter test fixture.
        scale: Guidance scale to exercise.
    """
    request = OpenAICreateSpeechRequest(
        input="hello",
        voice="demon",
        extra_params={
            "use_cfg": True,
            "cfg_scale": scale,
            "temperature": 0,
            "top_k": 12,
        },
    )
    assert adapter.validate(request) is None
    prepared = await adapter.build(request, [], False)
    info = prepared.prompt["additional_information"]
    assert {key: info[key] for key in request.extra_params} == request.extra_params
    assert info["speaker_id"] == "demon"


@pytest.mark.parametrize(
    "extra,error",
    [
        ({"use_cfg": "true"}, "boolean"),
        ({"cfg_scale": float("inf")}, "finite"),
        ({"temperature": "0.7"}, "finite"),
        ({"top_k": True}, "positive integer"),
        ({"top_k": 1.5}, "positive integer"),
    ],
)
def test_invalid_acoustic_settings_rejected(
    adapter: Any,
    extra: Sequence[str],
    error: str | None,
) -> None:
    """Verify invalid acoustic settings rejected.

    Args:
        adapter: Wrapped codec transfer adapter or adapter test fixture.
        extra: Additional command-line arguments.
        error: Expected error message, or None for success.
    """
    request = OpenAICreateSpeechRequest(input="hello", extra_params=extra)
    assert error in adapter.validate(request)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("unsupported", ["profile", "artifact", "reference"])
def test_cfg_rejected_before_generation(
    adapter: Any,
    streaming: bool,
    unsupported: str,
) -> None:
    """Verify CFG rejected before generation.

    Args:
        adapter: Wrapped codec transfer adapter or adapter test fixture.
        streaming: Whether to build an incremental streaming request.
        unsupported: Unsupported request setting to exercise.
    """
    request = OpenAICreateSpeechRequest(input="hello", extra_params={"use_cfg": True})
    if unsupported == "profile":
        adapter.ctx.engine_client.stage_configs[0].engine_args.scheduler_cls = "other.Scheduler"
        expected = "easymagpie_cfg"
    elif unsupported == "artifact":
        adapter._model_config_cache.pop("cfg_capability")
        expected = "reconvert"
    else:
        request.ref_audio = "data:audio/wav;base64,AAAA"
        expected = "registered speaker"
    if streaming:
        with pytest.raises(ValueError, match=expected):
            adapter.build_streaming_spec(request)
    else:
        assert expected in adapter.validate(request)


def test_streaming_spec_carries_cfg_and_conditional_bound(adapter: Any) -> None:
    """Verify streaming spec carries CFG and conditional bound.

    Args:
        adapter: Wrapped codec transfer adapter or adapter test fixture.
    """
    request = OpenAICreateSpeechRequest(
        input="",
        voice="demon",
        max_new_tokens=128,
        extra_params={
            "use_cfg": True,
            "cfg_scale": 1,
            "temperature": 0.1,
            "top_k": 20,
        },
    )
    info = adapter.build_streaming_spec(request).prefill_prompt["additional_information"]
    for key, value in request.extra_params.items():
        assert info[key] == value
    assert info["max_stream_tokens"] == 128
    request.extra_params = {"use_cfg": False}
    info = adapter.build_streaming_spec(request).prefill_prompt["additional_information"]
    assert info["use_cfg"] is False
    assert info["max_stream_tokens"] == 128


@pytest.mark.asyncio
async def test_websocket_config_preserves_extra_params() -> None:
    """Verify WebSocket config preserves extra params."""

    class Socket:
        """Test double for Socket behavior."""

        async def receive_text(self) -> str:
            """Return the next configured JSON session message.

            Returns:
                The next configured JSON message.
            """
            return json.dumps(
                {
                    "type": "session.config",
                    "stream_audio": True,
                    "response_format": "pcm",
                    "extra_params": {"use_cfg": True, "cfg_scale": 2.5},
                }
            )

    handler = EasyMagpieStreamingSpeechHandler(SimpleNamespace(_tts_model_type="easymagpie"))
    config = await handler._receive_config(Socket())
    assert config.extra_params == {"use_cfg": True, "cfg_scale": 2.5}


@pytest.mark.asyncio
@pytest.mark.parametrize("use_cfg", [True, False])
async def test_websocket_handler_delivers_settings_and_bound_to_engine(
    adapter: Any,
    use_cfg: bool,
) -> None:
    """Verify WebSocket handler delivers settings and bound to engine.

    Args:
        adapter: Wrapped codec transfer adapter or adapter test fixture.
        use_cfg: Whether the request enables classifier-free guidance.
    """
    chunks = []
    adapter._model_tokenizer = lambda: SimpleNamespace(
        text_vocab_size=100,
        incremental_encoder=lambda: SimpleNamespace(clean=True, finish=lambda: []),
    )
    engine = adapter.ctx.engine_client
    engine.default_sampling_params_list = [SamplingParams(max_tokens=32), SamplingParams(max_tokens=32)]

    async def outputs(prompt: AsyncIterator[Any]) -> AsyncIterator[SimpleNamespace]:
        """Outputs.

        Args:
            prompt: Prepared model prompt.

        Yields:
            Incremental input or public engine output for the current request.
        """
        async for chunk in prompt:
            chunks.append(chunk)
            yield SimpleNamespace(stage_id=0, outputs=[SimpleNamespace(token_ids=[1], finish_reason="length")])

    engine.generate = lambda **kwargs: outputs(kwargs["prompt"])

    async def pcm(
        generator: AsyncIterator[Any],
        _request_id: str,
    ) -> AsyncIterator[bytes]:
        """Pcm.

        Args:
            generator: Public output iterator.
            _request_id: Public request identity, unused by this test double.

        Yields:
            PCM audio fragments.
        """
        async for _output in generator:
            yield b"\x01\x02"

    service = SimpleNamespace(
        _tts_model_type="easymagpie",
        engine_client=engine,
        _get_tts_adapter=lambda: adapter,
        _generate_pcm_chunks=pcm,
    )
    extra = {"use_cfg": use_cfg, "cfg_scale": 1.5, "temperature": 0.1, "top_k": 10}
    socket = _Socket(
        [
            {
                "type": "session.config",
                "voice": "demon",
                "stream_audio": True,
                "response_format": "pcm",
                "max_new_tokens": 24,
                "extra_params": extra,
            },
            {"type": "input.tokens", "tokens": [10, 11]},
            {"type": "input.done"},
        ]
    )
    await EasyMagpieStreamingSpeechHandler(service).handle_session(socket)
    assert socket.messages[-1] == {"type": "session.done", "total_sentences": 1}
    info = chunks[0].prompt["additional_information"]
    assert {key: info[key] for key in extra} == extra
    assert info["max_stream_tokens"] == 24
    assert sum(chunk.sampling_params.max_tokens for chunk in chunks) == 24
    assert socket.audio


@pytest.mark.asyncio
async def test_websocket_invalid_cfg_rejected_without_engine_request(adapter: Any) -> None:
    """Verify WebSocket invalid CFG rejected without engine request.

    Args:
        adapter: Wrapped codec transfer adapter or adapter test fixture.
    """
    socket = _Socket(
        [
            {
                "type": "session.config",
                "stream_audio": True,
                "response_format": "pcm",
                "extra_params": {"use_cfg": "true"},
            },
        ]
    )
    service = SimpleNamespace(_tts_model_type="easymagpie", _get_tts_adapter=lambda: adapter)
    await EasyMagpieStreamingSpeechHandler(service).handle_session(socket)
    assert socket.messages[-1]["type"] == "error"
    assert "boolean" in socket.messages[-1]["message"]
    assert not socket.audio


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,error",
    [
        ([], "JSON object"),
        ({"type": "input.text"}, "Expected session.config"),
        ({"type": "session.config", "extra_params": []}, "Invalid session config"),
        ({"type": "session.config", "stream_audio": True, "response_format": "wav"}, "Invalid session config"),
    ],
)
async def test_websocket_config_validation_preserved(
    payload: dict[str, Any],
    error: str | None,
) -> None:
    """Verify WebSocket config validation preserved.

    Args:
        payload: Session configuration payload.
        error: Expected error message, or None for success.
    """
    socket = _Socket([payload])
    handler = EasyMagpieStreamingSpeechHandler(SimpleNamespace(_tts_model_type="easymagpie"))
    assert await handler._receive_config(socket) is None
    assert error in socket.messages[-1]["message"]


@pytest.mark.asyncio
async def test_http_cfg_defaults_and_conditional_opt_in(adapter: Any) -> None:
    """Verify HTTP CFG defaults and conditional opt in.

    Args:
        adapter: Wrapped codec transfer adapter or adapter test fixture.
    """
    request = OpenAICreateSpeechRequest(input="hello", extra_params={"use_cfg": True})
    prepared = await adapter.build(request, [], False)
    info = prepared.prompt["additional_information"]
    assert info["cfg_scale"] == 2.5
    assert info["temperature"] == 0.7
    assert info["top_k"] == 80

    request.extra_params = None
    adapter._model_config_cache.pop("cfg_capability")
    assert adapter.validate(request) is None
    prepared = await adapter.build(request, [], False)
    assert "use_cfg" not in prepared.prompt["additional_information"]


def test_http_rejects_conditional_raw_audio_on_cfg_pipeline(adapter: Any) -> None:
    """Verify HTTP rejects conditional raw audio on CFG pipeline.

    Args:
        adapter: Wrapped codec transfer adapter or adapter test fixture.
    """
    request = OpenAICreateSpeechRequest(input="hello", ref_audio="data:audio/wav;base64,AAAA")
    assert "registered speaker" in adapter.validate(request)


def test_http_rejects_malformed_cfg_capability(adapter: Any) -> None:
    """Verify HTTP rejects malformed CFG capability.

    Args:
        adapter: Wrapped codec transfer adapter or adapter test fixture.
    """
    adapter._model_config_cache["cfg_capability"]["version"] = 99
    request = OpenAICreateSpeechRequest(input="hello", extra_params={"use_cfg": True})
    assert "reconvert" in adapter.validate(request)


def test_conditional_stream_on_original_pipeline_remains_unbounded(adapter: Any) -> None:
    """Verify conditional stream on original pipeline remains unbounded.

    Args:
        adapter: Wrapped codec transfer adapter or adapter test fixture.
    """
    adapter.ctx.engine_client.stage_configs[0].engine_args.scheduler_cls = "other.Scheduler"
    spec = adapter.build_streaming_spec(OpenAICreateSpeechRequest(input=""))
    assert "max_stream_tokens" not in spec.prefill_prompt["additional_information"]
    assert "use_cfg" not in spec.prefill_prompt["additional_information"]


class _Socket:
    """Test double for Socket behavior."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        """Initialize Socket state.

        Args:
            messages: JSON input events returned by the socket test double.
        """
        self.incoming = iter(messages)
        self.messages = []
        self.audio = []

    async def accept(self) -> None:
        """Record acceptance of the test WebSocket connection."""
        pass

    async def receive_text(self) -> str:
        """Return the next configured JSON session message.

        Returns:
            The next configured JSON message.
        """
        return json.dumps(next(self.incoming))

    async def send_json(self, message: str) -> None:
        """Record a JSON message sent by the streaming handler.

        Args:
            message: Message sent over the socket or expected in the failure.
        """
        self.messages.append(message)

    async def send_bytes(self, data: bytes) -> None:
        """Record PCM audio sent by the streaming handler.

        Args:
            data: PCM audio bytes.
        """
        self.audio.append(data)
