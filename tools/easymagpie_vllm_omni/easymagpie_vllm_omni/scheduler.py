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

import copy
import threading
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information

if TYPE_CHECKING:
    from collections.abc import Iterable


class EasyMagpieARAsyncScheduler(OmniARAsyncScheduler):
    """Forward each chunk's token limit and additional information.

    This class also works around a bug in vLLM-Omni's async segment-stop
    handling that deadlocks paced streaming sessions. On a resumable segment
    stop, ``OmniARScheduler.update_from_output`` does::

        request.async_tokens_to_discard = 1        # hardcoded
        request.num_output_placeholders = 0

    i.e. it assumes exactly one async token is in flight and, unlike omni's own
    *resume* path, it never rolls ``num_computed_tokens`` back for the tokens it
    is about to discard. Combined with vLLM 0.24's async accounting (a discarded
    token returns early from ``AsyncScheduler._update_request_with_output``
    without decrementing ``num_output_placeholders``) this leaves the re-admitted
    session in an unschedulable state:

    * a leaked placeholder (``placeholders>0`` with ``num_computed==num_tokens``)
      permanently trips the scheduler's async skip-optimisation, or
    * ``num_computed_tokens == num_tokens`` with ``placeholders==0`` yields
      ``num_new_tokens==0``.

    Either way the request is never scheduled again and paced clients hang.

    The fix mirrors omni's resume path: snapshot the *true* number of in-flight
    async tokens at the moment of the stop, then after ``update_from_output`` set
    ``async_tokens_to_discard`` to that count (0 when nothing is in flight, so no
    spurious discard) and roll ``num_computed_tokens`` back by the same amount.

    TODO(upstream): fix ``OmniARScheduler.update_from_output`` directly so the
    segment-stop branch uses ``async_tokens_to_discard = num_output_placeholders``
    and ``num_computed_tokens -= num_output_placeholders`` (matching the resume
    branch), then drop this override.
    """

    def add_request(self, request: Request) -> None:
        session = self.requests.get(request.request_id)
        if session is None and request.resumable and self.vllm_config.model_config.stage_id == 0:
            # Record the initial prefill boundary once. Later text updates append
            # placeholders that vLLM can label as prefill, although they must consume
            # new text using the retained model state. The model uses this boundary
            # to dispatch those rows to decode; otherwise it slices past the initial
            # speaker/context embeddings and fails the prefill-length assertion.
            info = deserialize_additional_information(request.additional_information)
            if "_stream_prompt_len" in info:
                raise ValueError("Streaming prompt boundary is reserved for the scheduler")
            info["_stream_prompt_len"] = request.num_prompt_tokens
            request.additional_information = serialize_additional_information(info)
        if (
            session is not None
            and session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
            and session.resumable
            and not request.resumable
            and self.chunk_transfer_adapter is not None
        ):
            # TODO(vLLM/vLLM-Omni upstream): a final non-resumable update
            # received while the existing session is already waiting is turned
            # directly into FINISHED_ABORTED by Scheduler.add_request(). That
            # bypasses the Stage-0 output path, so async-chunk never publishes
            # its terminal payload and every downstream stage waits forever.
            # Upstream must publish terminal connector state before freeing the
            # waiting session; remove this bridge once it does. A shallow copy
            # lets the sender observe a terminal Request without mutating the
            # live session before vLLM performs its normal final-update cleanup.
            terminal_request = copy.copy(session)
            terminal_request.resumable = False
            terminal_request.status = RequestStatus.FINISHED_STOPPED
            self.chunk_transfer_adapter.save_async(
                multimodal_output=None,
                request=terminal_request,
                is_segment_finished=False,
            )

        super().add_request(request)

    def _update_request_with_output(self, request: Request, new_token_ids):
        new_token_ids, stopped = super()._update_request_with_output(request, new_token_ids)
        if stopped:
            # After super() has decremented the placeholder for the stopping
            # token, ``num_output_placeholders`` is the number of *other* async
            # tokens still in flight for this request — the value omni's stop
            # handler should have used but overwrites with a hardcoded 1. Record
            # it so update_from_output can restore the correct accounting. Only
            # tracked while inside update_from_output, so there is no per-step
            # cost beyond the (rare) segment stops themselves.
            pending = getattr(self, "_emp_stopped_this_step", None)
            if pending is not None:
                pending.append((request, request.num_output_placeholders))
        return new_token_ids, stopped

    def update_from_output(self, scheduler_output, model_runner_output):
        self._emp_stopped_this_step = []
        try:
            outputs = super().update_from_output(scheduler_output, model_runner_output)
            for request, snap in self._emp_stopped_this_step:
                # Only correct resumable stops where omni actually armed a discard.
                if getattr(request, "async_tokens_to_discard", 0) > 0:
                    request.async_tokens_to_discard = snap
                    if snap > 0:
                        request.num_computed_tokens -= snap
                request_id = getattr(request, "request_id", None)
                if request_id is not None and not getattr(request, "resumable", False):
                    _set_segment_finished(outputs, request_id, False)
        finally:
            self._emp_stopped_this_step = None
        return outputs

    def _handle_stopped_request(self, request: Request) -> bool:
        # The input engine queues ``None`` after the final StreamingInput but
        # leaves the existing session's ``resumable`` flag set. Clear it before
        # the base handler consumes the sentinel so the chunk-transfer adapter
        # emits a true terminal payload and releases request-persistent codec
        # state. An empty queue still means "waiting for more websocket input".
        streaming_queue = getattr(request, "streaming_queue", None)
        if getattr(request, "resumable", False) and streaming_queue and streaming_queue[0] is None:
            request.resumable = False
        return super()._handle_stopped_request(request)

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        outstanding_async_tokens = getattr(session, "num_output_placeholders", 0)
        super()._update_request_as_session(session, update)

        # Upstream hardcodes one discard on resume even when multiple async
        # outputs are outstanding. Its rollback is otherwise correct, so retain
        # it and replace only the discard count with the captured real value.
        if outstanding_async_tokens > 0 and getattr(session, "async_tokens_to_discard", 0) > 0:
            session.async_tokens_to_discard = outstanding_async_tokens

        new_max_tokens = getattr(update, "max_tokens", None)
        if new_max_tokens is not None:
            session.max_tokens = new_max_tokens

        if self.vllm_config.model_config.stage_id == 0:
            new_info = getattr(update, "additional_information", None)
            if new_info is not None:
                session.additional_information = new_info

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


def _poll_native_codec_chunk_unlocked(adapter: OmniChunkTransferAdapter, request: Request) -> bool:
    """Receive a chunk without resetting the vLLM state-cache position."""
    old_num_computed_tokens = request.num_computed_tokens
    # Async-chunk prewarm may install one unscheduled placeholder before the
    # first real payload. Only tokens with materialized state are retained.
    old_prompt = list(request.prompt_token_ids or [])[:old_num_computed_tokens]
    old_all_token_ids = list(request._all_token_ids)[:old_num_computed_tokens]

    received = OmniChunkTransferAdapter._poll_single_request(adapter, request)
    if not received:
        segment_boundary_received = request.request_id in adapter.segment_finished_requests
        request.prompt_token_ids = old_prompt
        request._all_token_ids[:] = old_all_token_ids
        request.num_prompt_tokens = len(old_prompt)
        request.num_computed_tokens = old_num_computed_tokens
        request.update_block_hashes()

        # TODO(vLLM-Omni upstream): `_poll_single_request()` consumes an empty
        # `is_segment_finished` payload, records the boundary, and then returns
        # False before publishing scheduler readiness through
        # `_finished_load_reqs`. This leaves Stage 1 parked in
        # WAITING_FOR_CHUNK forever. Upstream should treat
        # `payload_segment_finished` like `payload_finished` in its early-return
        # condition; remove this private-state bridge once that fix is available.
        if segment_boundary_received:
            adapter._finished_load_reqs.add(request.request_id)
        return segment_boundary_received

    frames = _codec_payload_frames(request.additional_information, adapter._easymagpie_num_quantizers)
    placeholders = [0] * frames
    request.prompt_token_ids = old_prompt + placeholders
    request._all_token_ids[:] = old_all_token_ids + placeholders
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.num_computed_tokens = old_num_computed_tokens
    request.update_block_hashes()
    return True


def _poll_native_codec_chunk(adapter: OmniChunkTransferAdapter, request: Request) -> bool:
    """Publish connector readiness only after the request payload is coherent."""
    with adapter._easymagpie_chunk_lock:
        return _poll_native_codec_chunk_unlocked(adapter, request)


def _codec_segment_resets_state(request: Request) -> bool:
    """Whether the completed Stage-1 segment is a dialogue utterance boundary."""
    payload = getattr(request, "additional_information", None)
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
        # The legacy connector queue can preempt a partly decoded stream back
        # through scheduled_new_reqs, replaying its cumulative placeholders
        # against only the latest chunk. Hold excess streams on the cached
        # request path instead, preserving their recurrent codec state.
        if adapter._active_window <= 0:
            adapter._active_window = adapter.scheduler_max_num_seqs
        self._easymagpie_reset_on_update: set[str] = set()
        self._easymagpie_pending_terminal: set[str] = set()

    def add_request(self, request: Request) -> None:
        """Admit a request or apply its next streaming update.

        Args:
            request: Incoming engine request.
        """
        session = self.requests.get(request.request_id)
        if request.request_id in self._easymagpie_pending_terminal:
            # Connector completion can overtake downstream prewarm submission.
            # Ignore remaining prewarms until the input iterator's final update;
            # they cannot start another codec stream with the same identity.
            if not request.resumable:
                self._easymagpie_pending_terminal.discard(request.request_id)
            return
        if session is None:
            # Prewarm tokens estimate talker input length, not codec frames.
            # A stream outside the active window has not polled its connector
            # yet; leave it empty so it cannot run before its first real chunk.
            request.prompt_token_ids = []
            request._all_token_ids[:] = []
            request.num_prompt_tokens = 0
            request.update_block_hashes()
        if session is not None and session.streaming_queue is not None and not request.resumable:
            # The frontend sentinel can overtake queued acoustic chunks. Only
            # the ordered connector terminal payload may end the codec stream;
            # queuing None here would stop at an earlier segment boundary.
            session._easymagpie_terminal_prewarm_seen = True
            return
        super().add_request(request)

    def _free_request(
        self,
        request: Request,
        delay_free_blocks: bool = False,
    ) -> dict[str, Any] | None:
        """Free a codec request and clear its connector ownership.

        Args:
            request: Incoming engine request.
            delay_free_blocks: Whether cache blocks must remain reserved after completion.

        Returns:
            Optional cache-transfer metadata returned by the base scheduler.
        """
        if (
            getattr(request, "streaming_queue", None) is not None
            and request.status == RequestStatus.FINISHED_STOPPED
            and not getattr(request, "_easymagpie_terminal_prewarm_seen", False)
        ):
            self._easymagpie_pending_terminal.add(request.request_id)
        try:
            return super()._free_request(request, delay_free_blocks)
        finally:
            # Generation output cleanup covers normal finishes, but the base
            # abort path leaves the connector's active-stream entry behind.
            self.chunk_transfer_adapter.cleanup_receiver(request.request_id)

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        """Finish selected requests and release their paired histories.

        Args:
            request_ids: Request identities to finish, or None for all requests.
            finished_status: Terminal status applied to the selected requests.

        Returns:
            Public request identities and their client indices for completed aborts.
        """
        if request_ids is None:
            self._easymagpie_pending_terminal.clear()
        elif isinstance(request_ids, str):
            self._easymagpie_pending_terminal.discard(request_ids)
        else:
            request_ids = tuple(request_ids)
            self._easymagpie_pending_terminal.difference_update(request_ids)
        return super().finish_requests(request_ids, finished_status)

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
            if not reset_codec:
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
        if session in self.skipped_waiting:
            self.skipped_waiting.remove_requests((session,))
        if waiting_for_input:
            self.num_waiting_for_streaming_input -= 1

        if session.num_computed_tokens == 0:
            # A text/phoneme warmup can finish a segment before emitting any
            # codec frames. No worker request exists yet: the first real codec
            # payload must take scheduled_new_reqs, not the cached-request path.
            session.status = RequestStatus.WAITING
            self.waiting.add_request(session)
        else:
            session.status = RequestStatus.RUNNING
            if session not in self.running:
                self.running.append(session)
        self.chunk_transfer_adapter.segment_finished_requests.discard(session.request_id)

    def update_from_output(self, scheduler_output, model_runner_output):
        # A segment finish must reach the output processor, but Stage 1 must not
        # be re-admitted through the generation scheduler's ``scheduled_new``
        # path afterward. That path recreates the worker batch row, losing the
        # codec's recurrent cache even when ``num_computed_tokens`` is retained.
        # Move resumable segment stops back to ``running`` after the base method
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
