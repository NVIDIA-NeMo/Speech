# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Attach CTC word timestamps to chat completions served by ``vllm serve``.

A chat completion request opts into CTC timestamps with vLLM's own extension field,
``"mm_processor_kwargs": {"capture_ctc_timestamps": true}``, so the engine keeps the
audio's alignment inputs while it generates. Start the server with::

    vllm serve ... --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware

and :func:`ctc_timestamp_middleware` aligns the finished transcript of such a request
through :func:`align_async`, the same worker method offline callers reach, and adds
the result to the response as a top-level ``ctc_timestamps`` field in the offline
format: ``words``, ``diarization``, ``speaker_tag_to_diarization_speaker`` and
``error``, which names why ``words`` is empty. Score diarization (DER) with ``diarization``,
never with ``words``. Under data parallelism, every engine core is asked and the one that
holds the request's capture answers. vLLM's chat route serves the request itself, API-key
check included; every other request passes through untouched.

Speaker tags reach the aligner only when the model writes them and the request keeps
them with ``"skip_special_tokens": false``. The middleware does not read the prompt: a
transcript without tags is aligned as one speaker, with a warning. With ``"stream": true``,
text deltas stream immediately and ``ctc_timestamps`` accompanies the final choice's
``finish_reason``, before any final usage chunk and ``[DONE]``. Only stream completion
waits for alignment. Requests with ``n`` above 1 are refused. A request that fails,
is cancelled or cannot be aligned releases its capture.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing
from http import HTTPStatus
from typing import Any

import anyio
from starlette.responses import StreamingResponse

from nemo.collections.speechlm2.vllm.salm.ctc_timestamps import (
    _empty_result,
    align_async,
    ctc_adapter_path,
    ctc_timestamp_config,
    release_captures_async,
)
from nemo.utils import logging
from nemo.utils.nemo_logging import LogMode

_CHAT_PATH = "/v1/chat/completions"
_OPT_IN = "capture_ctc_timestamps"
_SSE_BOUNDARY = re.compile(rb"\r?\n\r?\n")

# The errors a prepare reply reports for a request when its engine core holds no capture of it.
_NOT_HELD = frozenset({"no_capture", "not_enabled"})


async def ctc_timestamp_middleware(request: Any, call_next: Any) -> Any:
    """Attach CTC timestamps to the chat completions that opted in; pass every other request to vLLM.

    Args:
        request (Any): The incoming Starlette request.
        call_next (Any): The rest of the app, vLLM's routes included.

    Returns:
        Any: vLLM's response, with ``ctc_timestamps`` added for an opted-in request.
    """
    engine = getattr(request.app.state, "engine_client", None)
    path = request.scope["path"].removeprefix(request.scope.get("root_path", ""))
    if (
        request.method != "POST"
        or path != _CHAT_PATH
        or engine is None
        or not ctc_adapter_path(ctc_timestamp_config(engine.model_config))
    ):
        return await call_next(request)
    # Reading the body here keeps it replayable for vLLM's route.
    payload = _opted_in(await request.body())
    if payload is None:
        return await call_next(request)
    num_completion_choices = payload.get("n")
    if isinstance(num_completion_choices, int) and num_completion_choices > 1:
        return _error("CTC timestamps do not support n > 1.")

    request_id = _engine_request_id(request, payload)
    try:
        response = await call_next(request)
        if response.status_code != HTTPStatus.OK:
            await _release(engine, request_id)
            return response
        if payload.get("stream") and response.headers.get("content-type", "").startswith("text/event-stream"):
            return _TimestampStreamingResponse(response, engine, request_id)
        completion = json.loads(b"".join([chunk async for chunk in response.body_iterator]))
        headers = {name: value for name, value in response.headers.items() if name.lower() != "content-length"}
        if not isinstance(completion, dict):
            # vLLM's chat route answers null for a request whose client disconnected; there is nothing to align.
            await _release(engine, request_id)
            return _json(completion, response.status_code, headers)
        request_id = _reported_request_id(completion, request_id)
        text = completion["choices"][0]["message"]["content"] or ""
        (result,) = await align_async(_capture_owner_rpc(engine), [(request_id, text)], require_enabled=False)
    except BaseException as error:
        # Alignment releases the capture; a request that stops before it releases it here.
        await _release(engine, request_id)
        if not isinstance(error, Exception):
            raise
        logging.error(
            "[NeMoSpeechLM] CTC timestamps for chat request %s failed: %s", request_id, error, exc_info=error
        )
        return _error(error, "InternalServerError", HTTPStatus.INTERNAL_SERVER_ERROR)

    completion["ctc_timestamps"] = result
    return _json(completion, response.status_code, headers)


class _TimestampStreamingResponse(StreamingResponse):
    """Close the alignment iterator even if ASGI send fails, or the client disconnects before the body starts."""

    def __init__(self, response: Any, engine: Any, request_id: str):
        self._started = False
        self._upstream = response.body_iterator
        self._engine = engine
        self._request_id = request_id
        headers = {name: value for name, value in response.headers.items() if name.lower() != "content-length"}
        super().__init__(
            self._body(), status_code=response.status_code, headers=headers, background=response.background
        )

    async def _body(self) -> AsyncIterator[bytes]:
        self._started = True
        async with aclosing(_stream_with_timestamps(self._upstream, self._engine, self._request_id)) as stream:
            async for chunk in stream:
                yield chunk

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()
                if not self._started:
                    await _release(self._engine, self._request_id)
                    await self._upstream.aclose()


async def _stream_with_timestamps(body: AsyncIterator[bytes], engine: Any, request_id: str) -> AsyncIterator[bytes]:
    """Forward text immediately; defer the finish marker, trailing usage and DONE until alignment completes.

    Keep only transcript text and terminal metadata, not the stream. A final text/logprobs
    delta can share an event with ``finish_reason``; split it so that alignment delays no text.
    Alignment failures use the normal timestamp error field because HTTP headers are already sent.
    Upstream errors retain vLLM's error event instead of fabricating a successful completion.
    """
    text: list[str] = []
    terminal = None
    trailing: list[bytes] = []
    failed = False
    aligned = False
    try:
        async with aclosing(_sse_events(body)) as events:
            async for event in events:
                data = _sse_data(event)
                if data == "[DONE]":
                    if terminal is not None and not failed:
                        try:
                            (result,) = await align_async(
                                _capture_owner_rpc(engine), [(request_id, "".join(text))], require_enabled=False
                            )
                            aligned = True  # The worker already released the capture.
                        except Exception as error:
                            logging.error(
                                "[NeMoSpeechLM] CTC timestamps for streaming chat request %s failed: %s",
                                request_id,
                                error,
                                exc_info=error,
                            )
                            result = _empty_result("alignment_failed")
                        terminal["ctc_timestamps"] = result
                        yield _sse_json(terminal)
                    for tail in trailing:
                        yield tail
                    yield event
                    return
                if data is None:
                    # SSE comments/keepalives carry no completion data.
                    yield event
                    continue
                chunk = json.loads(data)
                if "error" in chunk:
                    failed = True
                    terminal = None
                    yield event
                    continue
                request_id = _reported_request_id(chunk, request_id)
                if terminal is not None:
                    trailing.append(event)
                    continue
                choices = chunk.get("choices") or []
                if failed or not choices:
                    yield event
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                text.append(delta.get("content") or "")
                if choice.get("finish_reason") is None:
                    yield event
                    continue
                terminal = chunk
                if delta or choice.get("logprobs") is not None:
                    partial = {**choice, "finish_reason": None}
                    if "stop_reason" in partial:
                        partial["stop_reason"] = None
                    yield _sse_json({**chunk, "choices": [partial]})
                    terminal["choices"] = [{**choice, "delta": {}, "logprobs": None}]
                    if "token_ids" in choice:
                        terminal["choices"][0]["token_ids"] = []
            raise ValueError("vLLM's chat stream ended before [DONE].")
    except Exception as error:
        logging.error("[NeMoSpeechLM] Streaming chat request %s failed: %s", request_id, error, exc_info=error)
        yield b"data: " + _error(error, "InternalServerError", HTTPStatus.INTERNAL_SERVER_ERROR).body + b"\n\n"
        yield b"data: [DONE]\n\n"
    finally:
        # Starlette cancels the stream's AnyIO scope on disconnect. Shield cleanup so
        # cancellation cannot strand an owned capture while other requests keep decoding.
        with anyio.CancelScope(shield=True):
            if not aligned:
                await _release(engine, request_id)
            await body.aclose()


async def _sse_events(body: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Frame SSE events across arbitrary HTTP fragments, including split UTF-8 and CRLF delimiters."""
    pending = b""
    async for chunk in body:
        start = max(0, len(pending) - 3)
        pending += chunk.encode("utf-8") if isinstance(chunk, str) else chunk
        while match := _SSE_BOUNDARY.search(pending, start):
            yield pending[: match.end()]
            pending = pending[match.end() :]
            start = 0
    if pending:
        raise ValueError("vLLM's chat stream ended with an incomplete SSE event.")


def _sse_data(event: bytes) -> str | None:
    """Read SSE data fields after framing, keeping comments and other fields out of the JSON payload."""
    fields = [line[5:].removeprefix(b" ") for line in event.splitlines() if line.startswith(b"data:")]
    return b"\n".join(fields).decode("utf-8") if fields else None


def _sse_json(chunk: dict) -> bytes:
    return ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8")


def _opted_in(body: bytes) -> dict | None:
    """The request's JSON when it opts into capture, read as the processor reads the flag; else ``None``."""
    # A quick filter, not the decision: a body that never mentions the flag cannot opt in,
    # so most requests skip parsing their megabytes of base64 audio. The parsed check below decides.
    if _OPT_IN.encode() not in body:
        return None
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    kwargs = payload.get("mm_processor_kwargs") if isinstance(payload, dict) else None
    return payload if isinstance(kwargs, dict) and bool(kwargs.get(_OPT_IN, False)) else None


def _engine_request_id(request: Any, payload: dict) -> str:
    """The engine request id vLLM's chat route will use, set up front so a failure can release its capture.

    vLLM takes ``chatcmpl-`` plus the ``X-Request-Id`` header, else the body's
    ``request_id``. With neither, the middleware adds the header.
    """
    base = request.headers.get("x-request-id") or payload.get("request_id")
    if not base:
        base = f"ctc-{uuid.uuid4().hex}"
        request.scope["headers"] = [*request.scope["headers"], (b"x-request-id", base.encode("latin-1"))]
    return f"chatcmpl-{base}"


def _reported_request_id(completion: dict, predicted: str) -> str:
    """The engine request id vLLM reports as the completion's ``id``, which alignment uses.

    It equals the prediction while vLLM keeps its request id scheme. If that ever
    changes, alignment still finds the capture, but a request that fails before its
    completion arrives releases nothing, so the first mismatch is logged.
    """
    reported = completion.get("id") or predicted
    if reported != predicted:
        logging.warning(
            "[NeMoSpeechLM] vLLM reported chat request %s, not the predicted %s (logged once): its request id "
            "scheme changed, so a request that fails before its completion arrives keeps its CTC capture, which "
            "counts against NEMO_CTC_TIMESTAMP_RETAIN_GB.",
            reported,
            predicted,
            mode=LogMode.ONCE,
        )
    return reported


def _capture_owner_rpc(engine: Any) -> Any:
    """``engine.collective_rpc`` for preparing one request, answered by the engine core that holds its capture.

    Under data parallelism, vLLM's ``collective_rpc`` runs on every engine core but returns only
    the first core's replies (``DPLBAsyncMPClient.call_utility_async``), while a request's capture
    lives on the core that served it, so this collects each core's replies and keeps the owner's.
    A single engine keeps vLLM's own path. Releasing needs no owner: vLLM's ``collective_rpc``
    reaches every core.
    """
    core = getattr(engine, "engine_core", None)
    identities = list(getattr(core, "core_engines", None) or ())
    if len(identities) < 2:
        return engine.collective_rpc

    async def rpc(method: str, timeout: float | None = None, args: tuple = (), kwargs: dict | None = None) -> list:
        replies = await asyncio.gather(
            *(
                core._call_utility_async("collective_rpc", method, timeout, args, kwargs, engine=identity)
                for identity in identities
            )
        )
        return [_owner_reply(replies)]

    return rpc


def _owner_reply(core_replies: list[list]) -> Any:
    """The prepare reply of the engine core that holds the request's capture.

    Of a core's per-worker replies, only the worker that holds captures answers with ``errors``.
    A core without the request's capture reports ``no_capture``, or ``not_enabled`` without
    timestamps; the owner reports anything else. Ownership decides, not words or batches, so the
    diarization and error of an empty transcript or a failed alignment come from the owner too.
    With no owner, the first core's reply stands, as in vLLM.

    Raises:
        RuntimeError: More than one core holds a capture of the request.
    """
    holders = [reply for replies in core_replies for reply in replies if isinstance(reply, dict) and "errors" in reply]
    owners = [reply for reply in holders if not _NOT_HELD.issuperset(reply["errors"])]
    if len(owners) > 1:
        raise RuntimeError(f"{len(owners)} data-parallel engine cores hold a CTC capture of this request.")
    return (owners or holders or core_replies[0])[0]


async def _release(engine: Any, request_id: str) -> None:
    """Release a capture that alignment will not, without masking the error that led here."""
    await asyncio.gather(release_captures_async(engine.collective_rpc, [request_id]), return_exceptions=True)


def _error(
    error: Exception | str, err_type: str = "BadRequestError", status_code: HTTPStatus = HTTPStatus.BAD_REQUEST
) -> Any:
    """An OpenAI-style error response, in vLLM's format; for an exception, vLLM picks the status from its type."""
    from vllm.entrypoints.serve.exception_handling.error_response import create_error_response

    response = create_error_response(error, err_type, status_code)
    return _json(response.model_dump(), response.error.code)


def _json(content: Any, status_code: int = HTTPStatus.OK, headers: dict | None = None) -> Any:
    from starlette.responses import JSONResponse

    return JSONResponse(content=content, status_code=int(status_code), headers=headers)
