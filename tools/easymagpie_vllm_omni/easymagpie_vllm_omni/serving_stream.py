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
"""Incremental text-input WebSocket serving for EasyMagpieTTS."""
from __future__ import annotations

import asyncio
import copy
import json
from contextlib import aclosing
from typing import Any, cast

from fastapi import WebSocket, WebSocketDisconnect
from vllm.engine.protocol import StreamingInput
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech_stream import OmniStreamingSpeechHandler
from vllm_omni.entrypoints.utils import coerce_param_message_types

logger = init_logger(__name__)

_MODEL_TYPE = "easymagpie"
# Segment completion includes time spent queued behind other requests. Keep
# this comfortably above the multi-second queueing observed at concurrency 32,
# while still bounding a genuinely stalled engine request.
_DEFAULT_PACE_TIMEOUT_S = 30.0
_MAX_TOKEN_CHUNK_SIZE = 256
_MAX_INPUT_MESSAGE_SIZE = 128 * 1024
_QUEUE_DEPTH = 8
_DONE = object()


def _sampling_params_with_max_tokens(params: Any, max_tokens: int) -> Any:
    cloned = copy.deepcopy(params)
    cloned.max_tokens = max(1, int(max_tokens))
    return cloned


class EasyMagpieInputStream:
    """Turn queued text-token chunks into one resumable engine input stream."""

    def __init__(
        self,
        *,
        prefill_prompt: dict[str, Any],
        sampling_params: Any,
        text_eos_id: int,
        max_new_tokens: int,
        pace_timeout_s: float = _DEFAULT_PACE_TIMEOUT_S,
        text_prefill_num: int = 0,
        queue_depth: int = _QUEUE_DEPTH,
        coalesce_queued_tokens: bool = True,
    ) -> None:
        self.prefill_prompt = prefill_prompt
        self.sampling_params = sampling_params
        self.text_eos_id = int(text_eos_id)
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.text_prefill_num = max(0, int(text_prefill_num))
        self.pace_timeout_s = max(0.0, float(pace_timeout_s))
        self.coalesce_queued_tokens = coalesce_queued_tokens
        self._input_queue: asyncio.Queue[list[int] | object] = asyncio.Queue(maxsize=max(1, queue_depth))
        self._segment_completions: asyncio.Queue[None] = asyncio.Queue()
        self._finished = False
        self._received_first_update = False
        self._pending_first_tokens: list[int] = []
        self.observed_output_frames = 0

    @property
    def finished(self) -> bool:
        return self._finished

    async def put_tokens(self, token_ids: list[int]) -> None:
        if self._finished:
            raise RuntimeError("Cannot append tokens after input.done")
        if not token_ids:
            return
        normalized = [int(token_id) for token_id in token_ids]
        if not self._received_first_update:
            self._pending_first_tokens.extend(normalized)
            if len(self._pending_first_tokens) < self.text_prefill_num:
                return
            normalized = self._pending_first_tokens
            self._pending_first_tokens = []
        await self._input_queue.put(normalized)
        self._received_first_update = True

    async def finish(self) -> None:
        if not self._finished:
            if self._pending_first_tokens:
                raise ValueError(
                    f"first input update must contain at least {self.text_prefill_num} text tokens; "
                    f"received {len(self._pending_first_tokens)}"
                )
            self._finished = True
            await self._input_queue.put(_DONE)

    def observe_output(self, output: Any) -> None:
        """Track generated frames and release the next text update at segment completion."""
        if getattr(output, "stage_id", None) != 0:
            return
        outputs = getattr(output, "outputs", None) or []
        if not outputs:
            return
        token_ids = getattr(outputs[0], "token_ids", None) or []
        finish_reason = getattr(outputs[0], "finish_reason", None)
        self.observed_output_frames += len(token_ids)
        if finish_reason is not None:
            self._segment_completions.put_nowait(None)

    async def _wait_for_segment_completion(self) -> None:
        if self.pace_timeout_s <= 0:
            return
        try:
            await asyncio.wait_for(self._segment_completions.get(), timeout=self.pace_timeout_s)
        except asyncio.TimeoutError as error:
            raise TimeoutError("Timed out waiting for stage-0 segment completion") from error

    def _accumulate_queued_tokens(self, token_ids: list[int]) -> tuple[list[int], bool]:
        """Coalesce text received while stage 0 was producing prior frames."""
        if not self.coalesce_queued_tokens:
            return token_ids, False
        done = False
        while not self._input_queue.empty():
            item = self._input_queue.get_nowait()
            if item is _DONE:
                done = True
                break
            token_ids.extend(cast(list[int], item))
        return token_ids, done

    async def inputs(self):
        """Yield token-bearing prefill, updates, EOS, then the acoustic tail."""
        first_item = await self._input_queue.get()
        if first_item is _DONE:
            return
        # Cumulative index of the next text id in the model's buffer. Each segment
        # is tagged with the absolute ``text_token_start`` of its first id so the
        # model absorbs it exactly once regardless of async segment lookahead (see
        # EasyMagpieTTSForConditionalGeneration._preprocess_decode).
        text_token_start = 0

        first_token_ids = cast(list[int], first_item)
        first_prompt = copy.deepcopy(self.prefill_prompt)
        first_info = first_prompt.setdefault("additional_information", {})
        first_info["text_token"] = first_token_ids
        first_info["text_token_start"] = text_token_start
        if self.text_prefill_num:
            first_info["text_prefill_num"] = self.text_prefill_num
            first_info["prefill_text_tokens"] = first_token_ids[: self.text_prefill_num]
            first_required_frames = 1 + len(first_token_ids) - self.text_prefill_num
        else:
            first_required_frames = len(first_token_ids)
        yield StreamingInput(
            prompt=first_prompt,
            sampling_params=_sampling_params_with_max_tokens(self.sampling_params, first_required_frames),
        )
        text_token_start += len(first_token_ids)

        input_done = False
        while True:
            item = await self._input_queue.get()
            if item is _DONE:
                break
            token_ids = cast(list[int], item)
            await self._wait_for_segment_completion()
            token_ids, input_done = self._accumulate_queued_tokens(token_ids)
            if input_done:
                token_ids.append(self.text_eos_id)
            yield StreamingInput(
                prompt={
                    "prompt_token_ids": [0],
                    "additional_information": {"text_token": token_ids, "text_token_start": text_token_start},
                },
                sampling_params=_sampling_params_with_max_tokens(self.sampling_params, len(token_ids)),
            )
            text_token_start += len(token_ids)
            if input_done:
                break

        await self._wait_for_segment_completion()
        if not input_done:
            yield StreamingInput(
                prompt={
                    "prompt_token_ids": [0],
                    "additional_information": {
                        "text_token": [self.text_eos_id],
                        "text_token_start": text_token_start,
                    },
                },
                sampling_params=_sampling_params_with_max_tokens(self.sampling_params, 1),
            )
            text_token_start += 1
            await self._wait_for_segment_completion()

        tail_max_tokens = self.max_new_tokens - self.observed_output_frames
        yield StreamingInput(
            prompt={
                "prompt_token_ids": [0],
                "additional_information": {"text_token": []},
            },
            sampling_params=_sampling_params_with_max_tokens(
                self.sampling_params,
                tail_max_tokens,
            ),
        )


class EasyMagpieStreamingSpeechHandler(OmniStreamingSpeechHandler):
    """Use resumable EasyMagpie requests while preserving the generic handler."""

    async def handle_session(self, websocket: WebSocket) -> None:
        if getattr(self._speech_service, "_tts_model_type", None) != _MODEL_TYPE:
            await super().handle_session(websocket)
            return

        await websocket.accept()
        request_id: str | None = None
        audio_task: asyncio.Task[int] | None = None
        input_stream: EasyMagpieInputStream | None = None
        completed = False
        try:
            config = await self._receive_config(websocket)
            if config is None:
                return
            if not config.stream_audio or config.response_format != "pcm":
                await self._send_error(
                    websocket,
                    "Incremental EasyMagpie input requires stream_audio=true and response_format='pcm'.",
                )
                return
            if config.word_timestamps:
                await self._send_error(
                    websocket, "word_timestamps is not supported with incremental EasyMagpie input."
                )
                return
            if config.model and hasattr(self._speech_service, "_check_model"):
                error = await self._speech_service._check_model(
                    OpenAICreateSpeechRequest(input="ping", model=config.model)
                )
                if error is not None:
                    await self._send_error(websocket, str(error))
                    return

            adapter = self._speech_service._get_tts_adapter()
            if adapter is None or not hasattr(adapter, "build_streaming_spec"):
                await self._send_error(websocket, "EasyMagpie incremental serving adapter is unavailable.")
                return

            request = OpenAICreateSpeechRequest(
                input="",
                model=config.model,
                voice=config.voice,
                response_format="pcm",
                max_new_tokens=config.max_new_tokens,
                stream=True,
            )
            spec = adapter.build_streaming_spec(request)
            text_encoder = spec.tokenizer.incremental_encoder()
            sampling_params_list = list(self._speech_service.engine_client.default_sampling_params_list)
            sampling_params_list = coerce_param_message_types(sampling_params_list, is_streaming=True)
            stage0_params = sampling_params_list[0]
            max_new_tokens = config.max_new_tokens or getattr(stage0_params, "max_tokens", 2048)
            input_stream = EasyMagpieInputStream(
                prefill_prompt=spec.prefill_prompt,
                sampling_params=stage0_params,
                text_eos_id=spec.text_eos_id,
                max_new_tokens=max_new_tokens,
                text_prefill_num=getattr(spec, "text_prefill_num", 0),
            )
            request_id = f"speech-stream-{random_uuid()}"
            generator = self._speech_service.engine_client.generate(
                prompt=input_stream.inputs(),
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                # Stage 0 is marked final_output only to expose pacing deltas;
                # completion must still be governed solely by stage-1 audio.
                output_modalities=["audio"],
            )

            async def observed_generator():
                async for output in generator:
                    input_stream.observe_output(output)
                    yield output

            await websocket.send_json(
                {
                    "type": "audio.start",
                    "sentence_index": 0,
                    "sentence_text": "",
                    "format": "pcm",
                    "sample_rate": spec.sample_rate,
                }
            )

            async def send_audio() -> int:
                total_bytes = 0
                async with aclosing(
                    self._speech_service._generate_pcm_chunks(observed_generator(), request_id)
                ) as chunks:
                    async for chunk in chunks:
                        total_bytes += len(chunk)
                        await websocket.send_bytes(chunk)
                return total_bytes

            audio_task = asyncio.create_task(send_audio())
            text_parts: list[str] = []
            input_token_count = 0
            while True:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=self._idle_timeout)
                if audio_task.done():
                    audio_task.result()
                if len(raw) > _MAX_INPUT_MESSAGE_SIZE:
                    await self._send_error(websocket, "Input message too large")
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON message")
                    continue
                if not isinstance(msg, dict):
                    await self._send_error(websocket, "WebSocket messages must be JSON objects")
                    continue

                msg_type = msg.get("type")
                if msg_type == "input.tokens":
                    tokens = msg.get("tokens")
                    text_vocab_size = int(spec.tokenizer.text_vocab_size)
                    if (
                        not isinstance(tokens, list)
                        or not tokens
                        or len(tokens) > _MAX_TOKEN_CHUNK_SIZE
                        or any(
                            isinstance(token, bool)
                            or not isinstance(token, int)
                            or token < 0
                            or token >= text_vocab_size
                            for token in tokens
                        )
                    ):
                        await self._send_error(
                            websocket,
                            f"input.tokens requires 1-{_MAX_TOKEN_CHUNK_SIZE} integer token IDs in "
                            f"[0, {text_vocab_size}).",
                        )
                        continue
                    if not text_encoder.clean:
                        await self._send_error(
                            websocket,
                            "input.tokens cannot be used while an input.text marker or IPA span is incomplete.",
                        )
                        continue
                    await input_stream.put_tokens(tokens)
                    input_token_count += len(tokens)
                elif msg_type == "input.text":
                    text = msg.get("text")
                    if not isinstance(text, str):
                        await self._send_error(websocket, "input.text requires a string value")
                        continue
                    tokens = text_encoder.push(text)
                    text_parts.append(text)
                    if tokens:
                        await input_stream.put_tokens(tokens)
                        input_token_count += len(tokens)
                elif msg_type == "input.done":
                    tokens = text_encoder.finish()
                    if tokens:
                        await input_stream.put_tokens(tokens)
                        input_token_count += len(tokens)
                    if input_token_count == 0:
                        await self._send_error(websocket, "No text or token input was provided.")
                        return
                    await input_stream.finish()
                    break
                else:
                    await self._send_error(websocket, f"Unknown message type: {msg_type}")

            total_bytes = await audio_task
            await websocket.send_json(
                {
                    "type": "audio.done",
                    "sentence_index": 0,
                    "sentence_text": "".join(text_parts),
                    "total_bytes": total_bytes,
                    "talker_frames": input_stream.observed_output_frames,
                    "text_tokens": input_token_count,
                    "error": False,
                }
            )
            await websocket.send_json({"type": "session.done", "total_sentences": 1})
            completed = True
        except (WebSocketDisconnect, asyncio.TimeoutError):
            # Client disconnects and input timeouts are expected terminal conditions.
            pass
        except Exception as error:
            logger.exception("Incremental EasyMagpie generation failed for %s", request_id)
            await self._send_error(websocket, f"Incremental EasyMagpie generation failed: {error}")
        finally:
            if audio_task is not None and not audio_task.done():
                audio_task.cancel()
                await asyncio.gather(audio_task, return_exceptions=True)
            if request_id is not None and not completed:
                try:
                    await self._speech_service.engine_client.abort(request_id)
                except Exception:
                    logger.debug("Failed to abort incremental request %s", request_id, exc_info=True)
