# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ASGI WebSocket/frontend lifetimes with a recording engine, without a model."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import aclosing
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from easymagpie_vllm_omni.serving_stream import EasyMagpieStreamingSpeechHandler
from starlette.websockets import WebSocket
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

if TYPE_CHECKING:
    from vllm.engine.protocol import StreamingInput


@pytest.mark.asyncio
async def test_engine_failure_is_reported_while_websocket_input_is_idle() -> None:
    """Verify an engine failure is reported and cleaned up without waiting for further client input."""
    engine = _RecordingEngine(fail=True)
    session = _Session(engine)
    task = asyncio.create_task(session.handler.handle_session(session.websocket))
    try:
        await session.configure()
        await session.message({"type": "input.tokens", "tokens": [2, 3, 4]})
        await asyncio.wait_for(engine.started.wait(), 1)
        # The client need not send another input message to learn of failure.
        await asyncio.wait_for(asyncio.shield(task), 0.2)
        assert any(event.get("type") == "error" for event in session.json_messages)
        assert engine.aborted and not engine.active
        assert all(task.done() for task in engine.producers)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2, 8, 32])
async def test_concurrent_asgi_sessions_preserve_text_eos_tail_and_terminal_order(count: int) -> None:
    """Verify concurrent sessions each submit their full text with the EOS tail and end in terminal order.

    Args:
        count: Number of concurrent WebSocket sessions sharing one engine.
    """
    engine = _RecordingEngine()
    sessions = [_Session(engine) for _ in range(count)]

    async def run(session: _Session) -> None:
        """Drive one session through configuration, text input, and completion, then check its events.

        Args:
            session: Session to drive.
        """
        await session.configure()
        await session.message({"type": "input.tokens", "tokens": [2, 3, 4]})
        await session.message({"type": "input.tokens", "tokens": [5, 6]})
        await session.message({"type": "input.done"})
        await session.handler.handle_session(session.websocket)
        assert session.audio
        types = [event["type"] for event in session.json_messages]
        assert types == ["audio.start", "audio.done", "session.done"]
        assert session.json_messages[-2]["total_bytes"] == sum(map(len, session.audio))
        assert session.json_messages[-2]["talker_frames"] == 32

    await asyncio.wait_for(asyncio.gather(*(run(session) for session in sessions)), 3)
    assert len(engine.segments) == count and not engine.active and not engine.aborted
    assert all(task.done() for task in engine.producers)
    for segments in engine.segments.values():
        text = [token for segment in segments for token in segment.prompt["additional_information"]["text_token"]]
        assert text == [2, 3, 4, 5, 6, 9]
        assert sum(segment.sampling_params.max_tokens for segment in segments) == 32
        assert segments[-1].prompt["additional_information"]["text_token"] == []
        assert segments[0].prompt["additional_information"]["max_stream_tokens"] == 32


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["disconnect", "timeout", "cancel", "slow_output_disconnect"])
async def test_asgi_interruption_joins_audio_and_input_producer(action: str) -> None:
    """Verify disconnect, idle timeout, or cancellation aborts the request once and joins all producers.

    Args:
        action: Interruption applied to the running session.
    """
    engine = _RecordingEngine()
    session = _Session(engine)
    if action == "slow_output_disconnect":
        session.audio_release.clear()
    else:
        engine.output_release.clear()
    if action == "timeout":
        session.handler._idle_timeout = 0.05
    task = asyncio.create_task(session.handler.handle_session(session.websocket))
    try:
        await session.configure()
        await session.message({"type": "input.tokens", "tokens": [2, 3, 4]})
        await asyncio.wait_for(engine.started.wait(), 1)
        if action == "slow_output_disconnect":
            await asyncio.wait_for(session.audio_started.wait(), 1)
        if action in ("disconnect", "slow_output_disconnect"):
            session.incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
        elif action == "cancel":
            task.cancel()
        result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
        if action != "cancel":
            assert result == [None]
        assert len(engine.aborted) == 1
        assert not engine.active
        assert all(task.done() for task in engine.producers)
        assert not any(event["type"] == "session.done" for event in session.json_messages)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("after_audio", [False, True])
async def test_disconnect_after_input_done_joins_pending_audio_and_input_producer(after_audio: bool) -> None:
    """Verify a disconnect after ``input.done`` aborts the request once and joins pending audio and input.

    Args:
        after_audio: Whether the disconnect happens while an audio send is blocked rather than engine output.
    """
    engine = _RecordingEngine()
    session = _Session(engine)
    if after_audio:
        session.audio_release.clear()
    else:
        engine.output_release.clear()
    task = asyncio.create_task(session.handler.handle_session(session.websocket))
    try:
        await session.configure()
        await session.message({"type": "input.tokens", "tokens": [2, 3, 4]})
        await session.message({"type": "input.done"})
        await asyncio.wait_for(engine.started.wait(), 1)
        if after_audio:
            await asyncio.wait_for(session.audio_started.wait(), 1)
        session.incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
        # Shield leaves cancellation/cleanup to finally if the regression hangs.
        await asyncio.wait_for(asyncio.shield(task), 0.2)
        assert len(engine.aborted) == 1
        assert not engine.active
        assert all(task.done() for task in engine.producers)
        assert not any(event["type"] == "session.done" for event in session.json_messages)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_engine_ending_the_utterance_before_input_done_stops_input_and_completes_session() -> None:
    """Acoustic EOS ends the utterance; later text is ignored and the session still completes normally."""
    engine = _RecordingEngine(end_after_segments=1)
    session = _Session(engine)
    task = asyncio.create_task(session.handler.handle_session(session.websocket))
    try:
        await session.configure()
        await session.message({"type": "input.tokens", "tokens": [2, 3, 4]})
        await asyncio.wait_for(engine.ended.wait(), 1)
        # AsyncOmni does not cancel the input task after normal completion: input must end by itself.
        await asyncio.wait_for(asyncio.gather(*engine.producers), 1)
        await session.message({"type": "input.tokens", "tokens": [5, 6]})
        await session.message({"type": "input.done"})
        await asyncio.wait_for(asyncio.shield(task), 1)
        assert [event["type"] for event in session.json_messages] == ["audio.start", "audio.done", "session.done"]
        assert session.audio
        (segments,) = engine.segments.values()
        assert len(segments) == 1
        assert not engine.aborted and not engine.active
    finally:
        task.cancel()
        for producer in engine.producers:
            producer.cancel()
        await asyncio.gather(task, *engine.producers, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["disconnect", "cancel"])
async def test_disconnect_during_pcm_cleanup_joins_iterator_before_aborting_engine(action: str) -> None:
    """Verify an interruption during PCM iterator cleanup waits for that cleanup before aborting the engine.

    Args:
        action: Interruption applied while the PCM iterator is closing, ``"disconnect"`` or ``"cancel"``.
    """
    engine = _RecordingEngine()
    session = _Session(engine)
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_finished = asyncio.Event()
    original_pcm = session.handler._speech_service._generate_pcm_chunks

    async def pcm(
        generator: AsyncGenerator[SimpleNamespace, None],
        request_id: str,
    ) -> AsyncGenerator[bytes, None]:
        """Forward PCM chunks and block in cleanup until the test releases it.

        Args:
            generator: Engine output stream.
            request_id: Request identifier forwarded to the original PCM generator.

        Yields:
            PCM chunks from the original generator.
        """
        async with aclosing(original_pcm(generator, request_id)) as chunks:
            try:
                async for chunk in chunks:
                    yield chunk
            finally:
                cleanup_started.set()
                await cleanup_release.wait()
                cleanup_finished.set()

    async def send(event: dict[str, Any]) -> None:
        """Fail binary audio sends and forward other ASGI messages to the session.

        Args:
            event: Outgoing ASGI WebSocket message.

        Raises:
            OSError: If the message carries audio bytes.
        """
        if event.get("bytes") is not None:
            raise OSError("controlled connection loss during audio send")
        await session.send(event)

    session.handler._speech_service._generate_pcm_chunks = pcm
    session.websocket._send = send
    task = asyncio.create_task(session.handler.handle_session(session.websocket))
    try:
        await session.configure()
        await session.message({"type": "input.tokens", "tokens": [2, 3, 4]})
        await session.message({"type": "input.done"})
        await asyncio.wait_for(cleanup_started.wait(), 1)
        if action == "disconnect":
            session.incoming.put_nowait({"type": "websocket.disconnect", "code": 1006})
        else:
            task.cancel()
        # Interruption occurs while the failed send is closing its iterator.
        await asyncio.sleep(0.02)
        assert not task.done()
        assert not engine.aborted
        cleanup_release.set()
        if action == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), 1)
        else:
            await asyncio.wait_for(asyncio.shield(task), 1)
        assert cleanup_finished.is_set()
        assert not engine.active
        assert all(producer.done() for producer in engine.producers)
        assert len(engine.aborted) == 1
    finally:
        cleanup_release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class _RecordingEngine:
    """Engine double that records streamed input segments and emits synthetic stage outputs."""

    default_sampling_params_list = [
        SamplingParams(max_tokens=32, output_kind=RequestOutputKind.DELTA),
        SamplingParams(max_tokens=32, output_kind=RequestOutputKind.DELTA),
    ]

    def __init__(
        self,
        *,
        fail: bool = False,
        end_after_segments: int | None = None,
    ) -> None:
        """Initialize the recording state.

        Args:
            fail: Whether ``generate`` fails after receiving its first segment.
            end_after_segments: Number of input segments after which the utterance ends, or None to never end
                it early.
        """
        self.fail = fail
        # Simulates Stage 0 ending the utterance at acoustic EOS after this many input segments.
        self.end_after_segments = end_after_segments
        self.ended = asyncio.Event()
        self.started = asyncio.Event()
        self.active = set()
        self.aborted = []
        self.segments = defaultdict(list)
        self.producers = []
        self.output_release = asyncio.Event()
        self.output_release.set()

    async def abort(self, request_id: str) -> None:
        """Record an aborted request.

        Args:
            request_id: Identifier of the aborted request.
        """
        self.aborted.append(request_id)

    async def generate(
        self,
        *,
        prompt: AsyncGenerator[StreamingInput, None],
        request_id: str,
        **kwargs: Any,
    ) -> AsyncGenerator[SimpleNamespace, None]:
        """Consume streamed input segments and emit one stage-0 and one stage-1 output per segment.

        Args:
            prompt: Streaming input segments for the request.
            request_id: Identifier of the request.
            **kwargs: Remaining ``generate`` arguments; ``output_modalities`` must be ``["audio"]``.

        Yields:
            A stage-0 output with the segment's token budget, then a stage-1 output with PCM bytes.

        Raises:
            RuntimeError: If the engine is configured to fail.
            Exception: Any error raised while consuming the input segments.
        """
        assert kwargs["output_modalities"] == ["audio"]
        self.active.add(request_id)
        queue = asyncio.Queue()
        done = object()

        async def produce() -> None:
            """Record input segments and forward them, an error, and a completion marker to the queue."""
            try:
                async with aclosing(prompt):
                    async for segment in prompt:
                        self.segments[request_id].append(segment)
                        queue.put_nowait(segment)
            except Exception as error:
                queue.put_nowait(error)
            finally:
                queue.put_nowait(done)

        producer = asyncio.create_task(produce())
        self.producers.append(producer)
        utterance_ended = False
        try:
            while True:
                segment = await queue.get()
                if segment is done:
                    break
                if isinstance(segment, Exception):
                    raise segment
                self.started.set()
                if self.fail:
                    raise RuntimeError("controlled CPU engine failure")
                await self.output_release.wait()
                utterance_ended = len(self.segments[request_id]) == self.end_after_segments
                yield SimpleNamespace(
                    stage_id=0,
                    outputs=[
                        SimpleNamespace(
                            token_ids=[0] * segment.sampling_params.max_tokens,
                            finish_reason="stop" if utterance_ended else "length",
                        )
                    ],
                )
                yield SimpleNamespace(stage_id=1, pcm=b"\x01\x00\x02\x00")
                await asyncio.sleep(0)
                if utterance_ended:
                    break
        finally:
            if not utterance_ended:
                # A normally completed AsyncOmni request leaves its input task running.
                producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)
            self.active.discard(request_id)
            self.ended.set()


class _Session:
    """In-memory ASGI WebSocket session wired to a streaming handler and an engine double."""

    def __init__(self, engine: _RecordingEngine) -> None:
        """Build the WebSocket, synthetic streaming spec, and speech service around the engine.

        Args:
            engine: Engine double serving the session.
        """
        self.incoming = asyncio.Queue()
        self.incoming.put_nowait({"type": "websocket.connect"})
        self.json_messages = []
        self.audio = []
        self.audio_release = asyncio.Event()
        self.audio_release.set()
        self.audio_started = asyncio.Event()
        self.websocket = WebSocket({"type": "websocket", "path": "/speech"}, self.incoming.get, self.send)
        encoder = SimpleNamespace(clean=True, finish=lambda: [], push=lambda text: [2] * len(text))
        spec = SimpleNamespace(
            prefill_prompt={"prompt_token_ids": [0], "additional_information": {"use_cfg": True}},
            tokenizer=SimpleNamespace(text_vocab_size=128, incremental_encoder=lambda: encoder),
            text_eos_id=9,
            text_prefill_num=2,
            sample_rate=32000,
        )
        adapter = SimpleNamespace(build_streaming_spec=lambda request: spec)

        async def pcm(
            generator: AsyncGenerator[SimpleNamespace, None],
            request_id: str,
        ) -> AsyncGenerator[bytes, None]:
            """Yield the PCM bytes of stage-1 outputs.

            Args:
                generator: Engine output stream.
                request_id: Unused request identifier.

            Yields:
                PCM bytes of each stage-1 output.
            """
            async with aclosing(generator):
                async for output in generator:
                    if output.stage_id == 1:
                        yield output.pcm

        service = SimpleNamespace(
            _tts_model_type="easymagpie",
            engine_client=engine,
            _get_tts_adapter=lambda: adapter,
            _generate_pcm_chunks=pcm,
        )
        self.handler = EasyMagpieStreamingSpeechHandler(service)

    async def send(self, event: dict[str, Any]) -> None:
        """Record outgoing JSON messages and audio, waiting for audio release before storing bytes.

        Args:
            event: Outgoing ASGI WebSocket message.
        """
        if event["type"] == "websocket.send":
            if "text" in event:
                self.json_messages.append(json.loads(event["text"]))
            elif "bytes" in event:
                self.audio_started.set()
                await self.audio_release.wait()
                self.audio.append(event["bytes"])

    async def message(self, value: dict[str, Any]) -> None:
        """Queue a client text message.

        Args:
            value: JSON-serializable message payload.
        """
        self.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps(value)})

    async def configure(self) -> None:
        """Queue a streaming PCM session configuration with a 32-token limit."""
        await self.message(
            {
                "type": "session.config",
                "voice": "eng",
                "stream_audio": True,
                "response_format": "pcm",
                "max_new_tokens": 32,
            }
        )
