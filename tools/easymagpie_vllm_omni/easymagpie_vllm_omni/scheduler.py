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
"""Streaming scheduler that propagates EasyMagpie request metadata.

Configure it on a single-stage deployment with::

    "scheduler_cls": "easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler"
"""
from __future__ import annotations

import threading
from types import MethodType

import torch
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter


class EasyMagpieARAsyncScheduler(OmniARAsyncScheduler):
    """Keep incremental EasyMagpie prompts schedulable between segments."""

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        super()._update_request_as_session(session, update)

        # Defensive guard: if a resumed session has every token already computed
        # (``num_computed_tokens >= num_tokens``), the upstream scheduler computes
        # ``num_new_tokens == 0`` and trips ``assert num_new_tokens > 0``. Roll
        # back one token so there is always something to recompute and sample from
        # — the same "recompute the last token" corrective vLLM applies on a full
        # prompt cache hit (see Scheduler._update_waiting_for_remote_kv).
        if session.num_computed_tokens >= session.num_tokens:
            session.num_computed_tokens = session.num_tokens - 1


def _codec_payload_frames(info, num_quantizers: int) -> int:
    """Return the number of time-major acoustic rows in a connector payload."""
    codes = info.get("codes", {}) if isinstance(info, dict) else {}
    audio = codes.get("audio") if isinstance(codes, dict) else None
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        return 0
    if audio.ndim == 2:
        return int(audio.shape[0])
    if audio.ndim == 1 and audio.numel() % num_quantizers == 0:
        return int(audio.numel() // num_quantizers)
    raise ValueError(f"invalid native codec payload shape: {tuple(audio.shape)}")


def _set_segment_finished(outputs, request_id: str, value: bool) -> None:
    """Set the streaming-boundary bit on one scheduler output."""
    if not isinstance(outputs, dict):
        return
    for client_outputs in outputs.values():
        for output in getattr(client_outputs, "outputs", ()):
            if getattr(output, "request_id", None) == request_id:
                output.is_segment_finished = value


def _poll_native_codec_chunk(adapter: OmniChunkTransferAdapter, entry) -> bool:
    """Serialize codec cache-position updates with the scheduler."""
    with adapter._easymagpie_chunk_lock:
        return OmniChunkTransferAdapter._poll_single_request(adapter, entry)


def _commit_native_codec_chunk(adapter: OmniChunkTransferAdapter, request: Request, result, **kwargs) -> bool:
    """Append new frames at the existing codec cache position."""
    payload, _ = result
    if not payload:
        return OmniChunkTransferAdapter._commit_received_chunk(adapter, request, result, **kwargs)

    old_computed = request.num_computed_tokens
    old_prompt = list(request.prompt_token_ids or [])[:old_computed]
    old_all_tokens = list(request._all_token_ids)[:old_computed]

    # Each chunk is a snapshot of new frames. Controls without codes must
    # replace consumed audio and any prewarm payload in the worker input.
    payload.setdefault("meta", {})["replace_runtime_additional_information"] = True
    frames = _codec_payload_frames(payload, adapter._easymagpie_num_quantizers)
    received = OmniChunkTransferAdapter._commit_received_chunk(adapter, request, result, **kwargs)
    placeholders = [0] * frames
    request.prompt_token_ids = old_prompt + placeholders
    request._all_token_ids[:] = old_all_tokens + placeholders
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.num_computed_tokens = old_computed
    request.update_block_hashes()
    return received


def _codec_segment_resets_state(request: Request) -> bool:
    """Whether the completed Stage-1 segment is a dialogue utterance boundary."""
    payload = getattr(request, "additional_information", None)
    if payload is None:
        payload = getattr(request, "model_intermediate_buffer", None)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    codec_streaming = meta.get("codec_streaming") if isinstance(meta, dict) else None
    if isinstance(codec_streaming, torch.Tensor):
        return codec_streaming.numel() == 1 and not bool(codec_streaming.item())
    return codec_streaming is False


class EasyMagpieCodecScheduler(OmniGenerationScheduler):
    """Keep each Stage-1 stream on one append-only native vLLM request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        adapter = self.chunk_transfer_adapter
        if adapter is None:
            raise ValueError("the native EasyMagpie codec requires async_chunk")
        config = self.vllm_config.model_config.hf_config
        num_quantizers = int(getattr(config, "num_stacked_codebooks", 0))
        if num_quantizers <= 0:
            raise ValueError("native EasyMagpie codec config has no stacked codebooks")
        adapter._easymagpie_chunk_lock = threading.Lock()
        adapter._easymagpie_num_quantizers = num_quantizers
        adapter._poll_single_request = MethodType(_poll_native_codec_chunk, adapter)
        adapter._commit_received_chunk = MethodType(_commit_native_codec_chunk, adapter)
        self._easymagpie_reset_on_update: set[str] = set()

    def add_request(self, request: Request) -> None:
        # The terminal text update also prewarms Stage 1. Wait for the
        # connector's terminal chunk so queued audio can drain first.
        if not request.resumable and request.request_id in self.requests:
            return
        super().add_request(request)

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """Resume a text chunk, or reset Stage 1 for a new dialogue reply.

        Every incremental text update prewarms downstream stages again. vLLM
        turns the duplicate Stage-1 request into a streaming update, but the
        generation scheduler's default handler replaces its prompt and resets
        ``num_computed_tokens``. For the native codec the update carries no new
        codec input; it only signals that another upstream segment is coming.
        Keep the append-only prompt position and vLLM-managed codec state intact.
        """
        reset_on_update = getattr(self, "_easymagpie_reset_on_update", set())
        if session.request_id in reset_on_update:
            reset_on_update.discard(session.request_id)
            self.chunk_transfer_adapter.segment_finished_requests.discard(session.request_id)
            return super()._update_request_as_session(session, update)

        prompt_token_ids = session.prompt_token_ids
        all_token_ids = list(session._all_token_ids)
        num_prompt_tokens = session.num_prompt_tokens
        num_computed_tokens = session.num_computed_tokens
        additional_information = session.additional_information

        super()._update_request_as_session(session, update)

        session.prompt_token_ids = prompt_token_ids
        session._all_token_ids[:] = all_token_ids
        session.num_prompt_tokens = num_prompt_tokens
        session.num_computed_tokens = num_computed_tokens
        session.additional_information = additional_information
        session.update_block_hashes()

    def _handle_stopped_request(self, request: Request) -> bool:
        adapter = self.chunk_transfer_adapter
        segment_boundary = request.request_id in adapter.segment_finished_requests
        reset_codec = _codec_segment_resets_state(request)
        if reset_codec:
            # Arm the reset before the base handler consumes a queued update;
            # it calls our _update_request_as_session override synchronously.
            self._easymagpie_reset_on_update.add(request.request_id)

        finished = super()._handle_stopped_request(request)
        if finished:
            self._easymagpie_reset_on_update.discard(request.request_id)
        else:
            segment_requests = getattr(self, "_easymagpie_segment_requests", None)
            if segment_requests is not None:
                segment_requests.add(request.request_id)
            if not reset_codec and request.num_computed_tokens > 0:
                stopped_sessions = getattr(self, "_easymagpie_stopped_sessions", None)
                if stopped_sessions is not None:
                    stopped_sessions.append(request)

        if segment_boundary:
            # TODO(vLLM-Omni upstream): metadata-only boundaries become ready
            # without scheduling tokens, so the adapter never consumes these
            # readiness entries. Clear them exactly when Stage 1 consumes the
            # boundary; otherwise the next turn schedules its prewarm token
            # against the previous codec payload.
            adapter.segment_finished_requests.discard(request.request_id)
            adapter._finished_load_reqs.discard(request.request_id)
            adapter.requests_with_ready_chunks.discard(request.request_id)
        return finished

    def _resume_codec_after_segment(self, session: Request) -> None:
        """Keep a resumable codec request on the worker's cached-request path."""
        waiting_for_input = session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
        if session in self.waiting:
            self.waiting.remove_requests((session,))
        if session in self.kv_holding_waiting:
            self.kv_holding_waiting.remove_requests((session,))
        self.deferred_waiting.discard(session)
        if waiting_for_input:
            self.num_waiting_for_streaming_input -= 1

        session.status = RequestStatus.RUNNING
        if session not in self.running:
            self.running.append(session)
        self.chunk_transfer_adapter.segment_finished_requests.discard(session.request_id)

    def update_from_output(self, scheduler_output, model_runner_output):
        # A segment finish must reach the output processor, but Stage 1 must not
        # be re-admitted through the generation scheduler's ``scheduled_new``
        # path afterward. That path recreates the worker batch row, losing the
        # codec's recurrent cache even when ``num_computed_tokens`` is retained.
        # Move resumable sessions with decoded frames back to ``running`` after the base method
        # has emitted the finish and removed them. Their next codec frames are
        # then scheduled as cached tokens against the same state pages.
        self._easymagpie_stopped_sessions = []
        self._easymagpie_segment_requests = set()
        try:
            outputs = super().update_from_output(scheduler_output, model_runner_output)
            for request_id in self._easymagpie_segment_requests:
                _set_segment_finished(outputs, request_id, True)
            for session in self._easymagpie_stopped_sessions:
                self._resume_codec_after_segment(session)
        finally:
            self._easymagpie_stopped_sessions = None
            self._easymagpie_segment_requests = None
        return outputs

    def schedule(self, *args, **kwargs):
        with self.chunk_transfer_adapter._easymagpie_chunk_lock:
            return super().schedule(*args, **kwargs)
