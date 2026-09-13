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
"""Tests for the vLLM-Omni 0.24 async scheduler compatibility layer."""
from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.scheduler import (
    EasyMagpieARAsyncScheduler,
    EasyMagpieCodecScheduler,
    _poll_native_codec_chunk,
)
from vllm.v1.request import RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter


def test_no_stop_is_inert_for_non_resumable_requests(monkeypatch):
    """A plain HTTP request never hits a segment stop, so the override must pass
    ``super()`` through unchanged and leave request accounting untouched."""
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(
        async_tokens_to_discard=0,
        num_computed_tokens=20,
        num_output_placeholders=0,
    )

    def fake_update_request_with_output(self, req, new_token_ids):
        return new_token_ids, False  # no stop this step

    def fake_update_from_output(self, scheduler_output, model_runner_output):
        self._update_request_with_output(request, [7])
        return "outputs"

    monkeypatch.setattr(OmniARAsyncScheduler, "_update_request_with_output", fake_update_request_with_output)
    monkeypatch.setattr(OmniARAsyncScheduler, "update_from_output", fake_update_from_output)

    outputs = scheduler.update_from_output(None, None)

    assert outputs == "outputs"
    assert request.async_tokens_to_discard == 0
    assert request.num_computed_tokens == 20
    assert request.num_output_placeholders == 0


def test_terminal_stop_without_discard_is_inert(monkeypatch):
    """HTTP requests end on a normal audio-EOS stop (not a resumable segment
    stop), so omni arms no discard and the override must not roll anything back."""
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(
        async_tokens_to_discard=0,
        num_computed_tokens=20,
        num_output_placeholders=0,
    )

    def fake_update_request_with_output(self, req, new_token_ids):
        req.num_output_placeholders = 0  # terminal stop, nothing else in flight
        return new_token_ids, True

    def fake_update_from_output(self, scheduler_output, model_runner_output):
        self._update_request_with_output(request, [7])
        return "outputs"  # omni does not arm a discard for a terminal stop

    monkeypatch.setattr(OmniARAsyncScheduler, "_update_request_with_output", fake_update_request_with_output)
    monkeypatch.setattr(OmniARAsyncScheduler, "update_from_output", fake_update_from_output)

    outputs = scheduler.update_from_output(None, None)

    assert outputs == "outputs"
    assert request.async_tokens_to_discard == 0
    assert request.num_computed_tokens == 20


@pytest.mark.parametrize("remaining_placeholders", [0, 1, 2])
def test_segment_stop_discards_and_rolls_back_exact_inflight_count(monkeypatch, remaining_placeholders):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(
        async_tokens_to_discard=0,
        num_computed_tokens=20,
        num_output_placeholders=remaining_placeholders + 1,
    )

    def fake_update_request_with_output(self, req, new_token_ids):
        req.num_output_placeholders -= 1  # stopping token returned
        return new_token_ids, True

    def fake_update_from_output(self, scheduler_output, model_runner_output):
        self._update_request_with_output(request, [7])
        request.async_tokens_to_discard = 1
        request.num_output_placeholders = 0
        return "outputs"

    monkeypatch.setattr(OmniARAsyncScheduler, "_update_request_with_output", fake_update_request_with_output)
    monkeypatch.setattr(OmniARAsyncScheduler, "update_from_output", fake_update_from_output)

    outputs = scheduler.update_from_output(None, None)

    assert outputs == "outputs"
    assert request.async_tokens_to_discard == remaining_placeholders
    assert request.num_computed_tokens == 20 - remaining_placeholders


def test_final_streaming_sentinel_marks_session_non_resumable(monkeypatch):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(resumable=True, streaming_queue=deque([None]))

    def fake_handle_stopped_request(self, req):
        assert req.resumable is False
        return True

    monkeypatch.setattr(OmniARAsyncScheduler, "_handle_stopped_request", fake_handle_stopped_request)

    assert scheduler._handle_stopped_request(request) is True
    assert request.resumable is False


def test_empty_streaming_queue_remains_resumable_while_waiting(monkeypatch):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(resumable=True, streaming_queue=deque())
    monkeypatch.setattr(OmniARAsyncScheduler, "_handle_stopped_request", lambda self, req: False)

    assert scheduler._handle_stopped_request(request) is False
    assert request.resumable is True


def test_resume_uses_exact_discard_count_and_forwards_chunk_metadata(monkeypatch):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0))
    session = SimpleNamespace(
        async_tokens_to_discard=0,
        num_computed_tokens=20,
        num_output_placeholders=2,
        num_tokens=23,
        max_tokens=1,
        additional_information={"text_token": [1]},
    )
    update = SimpleNamespace(max_tokens=5, additional_information={"text_token": [2, 3]})

    def fake_update_request_as_session(self, req, streaming_update):
        req.async_tokens_to_discard = 1
        req.num_computed_tokens -= req.num_output_placeholders
        req.num_output_placeholders = 0

    monkeypatch.setattr(OmniARAsyncScheduler, "_update_request_as_session", fake_update_request_as_session)

    scheduler._update_request_as_session(session, update)

    assert session.async_tokens_to_discard == 2
    assert session.num_computed_tokens == 18
    assert session.max_tokens == 5
    assert session.additional_information == {"text_token": [2, 3]}


def test_native_codec_chunk_appends_prompt_without_resetting_state(monkeypatch):
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter._easymagpie_num_quantizers = 2
    adapter._easymagpie_chunk_lock = threading.Lock()
    adapter.get_req_chunk = {"request": 1}
    request = SimpleNamespace(
        prompt_token_ids=[0, 0],
        request_id="request",
        _all_token_ids=[0, 0],
        num_computed_tokens=2,
        num_prompt_tokens=2,
        additional_information=None,
        update_block_hashes=lambda: None,
    )

    def fake_poll(self, req):
        req.prompt_token_ids = [0]
        req._all_token_ids[:] = []
        req.num_computed_tokens = 0
        req.additional_information = {"codes": {"audio": torch.ones((3, 2), dtype=torch.long)}}
        return True

    monkeypatch.setattr(OmniChunkTransferAdapter, "_poll_single_request", fake_poll)

    assert _poll_native_codec_chunk(adapter, request) is True
    assert request.prompt_token_ids == [0, 0, 0, 0, 0]
    assert request._all_token_ids == [0, 0, 0, 0, 0]
    assert request.num_prompt_tokens == 5
    assert request.num_computed_tokens == 2
    assert request.additional_information["codes"]["audio"].shape == (3, 2)

    prewarm_request = SimpleNamespace(
        prompt_token_ids=[0],
        request_id="request",
        _all_token_ids=[0],
        num_computed_tokens=0,
        num_prompt_tokens=1,
        additional_information=None,
        update_block_hashes=lambda: None,
    )
    assert _poll_native_codec_chunk(adapter, prewarm_request) is True
    assert prewarm_request.prompt_token_ids == [0, 0, 0]
    assert prewarm_request._all_token_ids == [0, 0, 0]
    assert prewarm_request.num_computed_tokens == 0


def test_native_codec_empty_segment_marker_does_not_reset_state(monkeypatch):
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter._easymagpie_num_quantizers = 2
    adapter._easymagpie_chunk_lock = threading.Lock()
    request = SimpleNamespace(
        prompt_token_ids=[0, 0, 0],
        request_id="request",
        _all_token_ids=[0, 0, 0],
        num_computed_tokens=3,
        num_prompt_tokens=3,
        additional_information={"codes": {"audio": torch.ones((3, 2), dtype=torch.long)}},
        update_block_hashes=lambda: None,
    )

    def fake_poll(self, req):
        # The base adapter resets these fields before deciding that an empty
        # non-terminal segment marker is not a schedulable codec chunk.
        req.prompt_token_ids = []
        req._all_token_ids[:] = []
        req.num_prompt_tokens = 0
        req.num_computed_tokens = 0
        req.additional_information = {"meta": {"is_segment_finished": True}}
        return False

    monkeypatch.setattr(OmniChunkTransferAdapter, "_poll_single_request", fake_poll)

    assert _poll_native_codec_chunk(adapter, request) is False
    assert request.prompt_token_ids == [0, 0, 0]
    assert request._all_token_ids == [0, 0, 0]
    assert request.num_prompt_tokens == 3
    assert request.num_computed_tokens == 3


def test_native_codec_streaming_update_preserves_state_and_resumes_polling():
    scheduler = object.__new__(EasyMagpieCodecScheduler)
    scheduler.chunk_transfer_adapter = SimpleNamespace(segment_finished_requests={"request"})
    scheduler.num_waiting_for_streaming_input = 1
    scheduler.log_stats = False

    class _SkippedWaiting(list):
        def remove_requests(self, requests):
            for request in requests:
                self.remove(request)

    original_prompt = [0, 0, 0, 0, 0]
    original_all_tokens = [0, 0, 0, 0, 0]
    session = SimpleNamespace(
        request_id="request",
        prompt_token_ids=original_prompt,
        _all_token_ids=original_all_tokens,
        num_prompt_tokens=5,
        num_computed_tokens=5,
        additional_information={"codes": {"audio": torch.ones((2, 2), dtype=torch.long)}},
        arrival_time=1.0,
        sampling_params="old",
        _output_token_ids=[],
        update_block_hashes=lambda: None,
        status=RequestStatus.WAITING_FOR_STREAMING_REQ,
    )
    scheduler.skipped_waiting = _SkippedWaiting([session])
    enqueued = []
    scheduler._enqueue_waiting_request = enqueued.append
    update = SimpleNamespace(
        prompt_token_ids=[0],
        additional_information=None,
        arrival_time=2.0,
        sampling_params="new",
    )

    scheduler._update_request_as_session(session, update)

    assert session.prompt_token_ids is original_prompt
    assert session._all_token_ids is original_all_tokens
    assert session.num_prompt_tokens == 5
    assert session.num_computed_tokens == 5
    assert session.additional_information["codes"]["audio"].shape == (2, 2)
    assert session.arrival_time == 2.0
    assert session.sampling_params == "new"
    assert session.status == RequestStatus.WAITING
    assert scheduler.num_waiting_for_streaming_input == 0
    assert scheduler.chunk_transfer_adapter.segment_finished_requests == set()
    assert scheduler.skipped_waiting == []
    assert enqueued == [session]


@pytest.mark.parametrize(
    ("status", "queue_name", "num_waiting"),
    [
        (RequestStatus.WAITING, "waiting", 0),
        (RequestStatus.WAITING_FOR_STREAMING_REQ, "skipped_waiting", 1),
    ],
)
def test_native_codec_segment_resume_stays_on_cached_request_path(status, queue_name, num_waiting):
    scheduler = object.__new__(EasyMagpieCodecScheduler)
    scheduler.chunk_transfer_adapter = SimpleNamespace(segment_finished_requests={"request"})
    scheduler.num_waiting_for_streaming_input = num_waiting
    scheduler.running = []

    class _RequestQueue(list):
        def remove_requests(self, requests):
            for request in requests:
                self.remove(request)

    session = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[0] * 6,
        _all_token_ids=[0] * 6,
        num_prompt_tokens=6,
        num_computed_tokens=6,
        status=status,
    )
    scheduler.waiting = _RequestQueue()
    scheduler.skipped_waiting = _RequestQueue()
    getattr(scheduler, queue_name).append(session)

    scheduler._resume_codec_after_segment(session)

    assert scheduler.waiting == []
    assert scheduler.skipped_waiting == []
    assert scheduler.running == [session]
    assert session.status == RequestStatus.RUNNING
    assert session.prompt_token_ids == [0] * 6
    assert session._all_token_ids == [0] * 6
    assert session.num_prompt_tokens == 6
    assert session.num_computed_tokens == 6
    assert scheduler.num_waiting_for_streaming_input == 0
    assert scheduler.chunk_transfer_adapter.segment_finished_requests == set()
