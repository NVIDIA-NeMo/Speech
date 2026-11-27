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
"""Tests for the EasyMagpie streaming schedulers."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.scheduler import (
    EasyMagpieCodecScheduler,
    _codec_segment_resets_state,
    _commit_native_codec_chunk,
    _set_segment_finished,
)
from vllm.v1.request import RequestStatus
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter


def test_scheduler_output_marks_only_the_matching_segment():
    matching = SimpleNamespace(request_id="matching", is_segment_finished=False)
    other = SimpleNamespace(request_id="other", is_segment_finished=False)
    outputs = {0: SimpleNamespace(outputs=[matching, other])}

    _set_segment_finished(outputs, "matching", True)

    assert matching.is_segment_finished is True
    assert other.is_segment_finished is False


@pytest.mark.parametrize("computed", [0, 2])
def test_native_codec_chunk_appends_prompt_without_resetting_state(computed):
    adapter, request = _codec_receiver(computed)
    payload = {"codes": {"audio": torch.ones((3, 2), dtype=torch.long)}}

    assert _receive_codec_chunk(adapter, request, payload)
    assert request.prompt_token_ids == request._all_token_ids == [0] * (computed + 3)
    assert request.num_prompt_tokens == computed + 3
    assert request.num_computed_tokens == computed
    assert request.additional_information["codes"]["audio"].shape == (3, 2)


@pytest.mark.parametrize("boundary", ["is_segment_finished", "finished"])
def test_native_codec_control_marker_never_replays_consumed_frames(boundary):
    adapter, request = _codec_receiver(3)
    request.additional_information = {"codes": {"audio": torch.ones((3, 2), dtype=torch.long)}}
    request.model_intermediate_buffer = request.additional_information

    assert _receive_codec_chunk(adapter, request, {"meta": {boundary: True}})
    assert request.prompt_token_ids == request._all_token_ids == [0, 0, 0]
    assert request.num_prompt_tokens == request.num_computed_tokens == 3
    assert "codes" not in request.additional_information
    assert "codes" not in request.model_intermediate_buffer
    assert adapter._finished_load_reqs == {"request"}
    assert request.resumable == (boundary == "is_segment_finished")


def _receive_codec_chunk(adapter, request, payload):
    return _commit_native_codec_chunk(
        adapter, request, (payload, 1), stage_id=1, req_id="request", chunk_id=1, connector_get_key="request_0_1"
    )


def _codec_receiver(computed):
    adapter = object.__new__(OmniChunkTransferAdapter)
    adapter._easymagpie_num_quantizers = 2
    adapter.model_mode = "generation"
    adapter.get_req_chunk = {"request": 1}
    adapter.segment_finished_requests = set()
    adapter.upstream_exhausted_requests = set()
    adapter._finished_load_reqs = set()
    request = SimpleNamespace(
        prompt_token_ids=[0] * max(1, computed),
        _all_token_ids=[0] * max(1, computed),
        request_id="request",
        resumable=True,
        num_computed_tokens=computed,
        num_prompt_tokens=max(1, computed),
        additional_information=None,
        update_block_hashes=lambda: None,
    )
    return adapter, request


def test_codec_segment_stop_consumes_late_boundary_readiness(monkeypatch):
    scheduler = object.__new__(EasyMagpieCodecScheduler)
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        segment_finished_requests={"request"},
        _finished_load_reqs={"request"},
        requests_with_ready_chunks={"request"},
    )
    scheduler._easymagpie_reset_on_update = set()
    scheduler._easymagpie_segment_requests = set()
    scheduler._easymagpie_stopped_sessions = []
    request = SimpleNamespace(
        request_id="request",
        additional_information={"meta": {"codec_streaming": False}},
    )
    monkeypatch.setattr(OmniGenerationScheduler, "_handle_stopped_request", lambda self, request: False)

    assert scheduler._handle_stopped_request(request) is False

    assert scheduler.chunk_transfer_adapter.segment_finished_requests == set()
    assert scheduler.chunk_transfer_adapter._finished_load_reqs == set()
    assert scheduler.chunk_transfer_adapter.requests_with_ready_chunks == set()


def test_dialogue_boundary_resets_codec_before_consuming_queued_update(monkeypatch):
    scheduler = object.__new__(EasyMagpieCodecScheduler)
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        segment_finished_requests={"request"},
        _finished_load_reqs={"request"},
        requests_with_ready_chunks={"request"},
    )
    scheduler._easymagpie_reset_on_update = set()
    scheduler._easymagpie_stopped_sessions = []
    scheduler._easymagpie_segment_requests = set()
    request = SimpleNamespace(
        request_id="request",
        additional_information={"meta": {"codec_streaming": torch.tensor(False)}},
    )
    update = SimpleNamespace()
    base_updates = []

    def fake_base_update(self, session, streaming_update):
        base_updates.append((session, streaming_update))

    def fake_base_stop(self, session):
        # OmniGenerationScheduler consumes an already queued update synchronously.
        self._update_request_as_session(session, update)
        return False

    monkeypatch.setattr(OmniGenerationScheduler, "_update_request_as_session", fake_base_update)
    monkeypatch.setattr(OmniGenerationScheduler, "_handle_stopped_request", fake_base_stop)

    assert _codec_segment_resets_state(request) is True
    assert scheduler._handle_stopped_request(request) is False
    assert base_updates == [(request, update)]
    assert scheduler._easymagpie_reset_on_update == set()
    assert scheduler._easymagpie_stopped_sessions == []
    assert scheduler._easymagpie_segment_requests == {"request"}
    assert scheduler.chunk_transfer_adapter.segment_finished_requests == set()


@pytest.mark.parametrize("computed", [0, 3])
def test_native_codec_segment_caches_only_previously_admitted_requests(monkeypatch, computed):
    scheduler = object.__new__(EasyMagpieCodecScheduler)
    scheduler.chunk_transfer_adapter = SimpleNamespace(segment_finished_requests=set())
    scheduler._easymagpie_reset_on_update = set()
    output = SimpleNamespace(request_id="request", is_segment_finished=False)
    request = SimpleNamespace(request_id="request", additional_information={}, num_computed_tokens=computed)

    def fake_base_stop(self, session):
        return False

    def fake_base_update(self, scheduler_output, model_runner_output):
        self._handle_stopped_request(request)
        return {0: SimpleNamespace(outputs=[output])}

    monkeypatch.setattr(OmniGenerationScheduler, "_handle_stopped_request", fake_base_stop)
    monkeypatch.setattr(OmniGenerationScheduler, "update_from_output", fake_base_update)
    resumed = []
    monkeypatch.setattr(
        EasyMagpieCodecScheduler, "_resume_codec_after_segment", lambda self, session: resumed.append(session)
    )

    scheduler.update_from_output(None, None)

    assert output.is_segment_finished is True
    assert resumed == ([request] if computed else [])
    assert scheduler._easymagpie_segment_requests is None


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
    scheduler.deferred_waiting = _SkippedWaiting([session])
    scheduler.deferred_waiting.discard = scheduler.deferred_waiting.remove
    scheduler.kv_holding_waiting = _SkippedWaiting([session])
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
    assert scheduler.deferred_waiting == []
    assert scheduler.kv_holding_waiting == []
    assert enqueued == [session]


@pytest.mark.parametrize(
    ("status", "queue_name", "num_waiting"),
    [
        (RequestStatus.WAITING, "waiting", 0),
        (RequestStatus.WAITING_FOR_STREAMING_REQ, "kv_holding_waiting", 1),
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
    scheduler.kv_holding_waiting = _RequestQueue()
    scheduler.deferred_waiting = _RequestQueue()
    scheduler.deferred_waiting.discard = scheduler.deferred_waiting.remove
    scheduler.deferred_waiting.append(session)
    getattr(scheduler, queue_name).append(session)

    scheduler._resume_codec_after_segment(session)

    assert scheduler.waiting == []
    assert scheduler.kv_holding_waiting == []
    assert scheduler.deferred_waiting == []
    assert scheduler.running == [session]
    assert session.status == RequestStatus.RUNNING
    assert session.prompt_token_ids == [0] * 6
    assert session._all_token_ids == [0] * 6
    assert session.num_prompt_tokens == 6
    assert session.num_computed_tokens == 6
    assert scheduler.num_waiting_for_streaming_input == 0
    assert scheduler.chunk_transfer_adapter.segment_finished_requests == set()
