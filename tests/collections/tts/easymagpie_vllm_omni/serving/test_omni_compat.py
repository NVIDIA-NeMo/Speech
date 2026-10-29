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
import asyncio
from types import SimpleNamespace

import pytest
from easymagpie_vllm_omni.omni_compat import patch_streaming_lifecycle, preserve_resumable_segment
from vllm_omni.engine.orchestrator import Orchestrator


def test_resumable_segment_is_not_treated_as_request_completion():
    output = SimpleNamespace(finished=True)
    request_state = SimpleNamespace(streaming=SimpleNamespace(enabled=True, segment_finished=True))

    assert preserve_resumable_segment(output, request_state) is True
    assert output.finished is False
    assert output.is_segment_finished is True


def test_resumable_segment_marker_survives_vllm_finished_rewrite():
    output = SimpleNamespace(finished=False)
    request_state = SimpleNamespace(streaming=SimpleNamespace(enabled=True, segment_finished=True))

    assert preserve_resumable_segment(output, request_state) is True
    assert output.finished is False
    assert output.is_segment_finished is True


def test_terminal_request_completion_is_unchanged():
    output = SimpleNamespace(finished=True)
    request_state = SimpleNamespace(streaming=SimpleNamespace(enabled=True, segment_finished=False))

    assert preserve_resumable_segment(output, request_state) is False
    assert output.finished is True
    assert not hasattr(output, "is_segment_finished")


@pytest.mark.asyncio
async def test_patch_marks_segment_before_delegating_and_is_idempotent(monkeypatch):
    calls = []

    async def original(self, stage_id, replica_id, outputs):
        calls.append((stage_id, replica_id, outputs))

    monkeypatch.setattr(Orchestrator, "_handle_processed_outputs", original)
    patch_streaming_lifecycle()
    patched = Orchestrator._handle_processed_outputs
    patch_streaming_lifecycle()
    assert Orchestrator._handle_processed_outputs is patched

    output = SimpleNamespace(request_id="request", finished=True)
    request_state = SimpleNamespace(streaming=SimpleNamespace(enabled=True, segment_finished=True))
    orchestrator = SimpleNamespace(request_states={"request": request_state})

    await patched(orchestrator, 1, 0, [output])

    assert output.finished is False
    assert output.is_segment_finished is True
    assert calls == [(1, 0, [output])]


@pytest.mark.asyncio
async def test_terminal_update_publishes_completion_when_stages_are_already_waiting(monkeypatch):
    calls = []

    async def original(self, msg):
        calls.append(msg)

    monkeypatch.setattr(Orchestrator, "_handle_streaming_update", original)
    patch_streaming_lifecycle()
    patched = Orchestrator._handle_streaming_update

    request_state = SimpleNamespace(
        final_stage_id=1, final_output_stage_ids={1}, finished_final_output_stage_ids={1}, stage_submit_ts={1: 42.0}
    )
    final_pool = SimpleNamespace(
        stage_client=SimpleNamespace(final_output_type="audio", audio_sample_rate=22050),
        get_bound_replica_id=lambda request_id: 3,
    )
    cleanups = []

    async def cleanup(request_ids):
        cleanups.append(request_ids)

    orchestrator = SimpleNamespace(
        request_states={"request": request_state},
        stage_pools=[None, final_pool],
        output_async_queue=asyncio.Queue(),
        _cleanup_request_ids=cleanup,
    )
    msg = SimpleNamespace(request_id="request", prompt=SimpleNamespace(resumable=False))

    await patched(orchestrator, msg)

    assert calls == [msg]
    assert not orchestrator.output_async_queue.empty()
    result = orchestrator.output_async_queue.get_nowait()
    assert result.request_id == "request"
    assert result.stage_id == 1
    assert result.replica_id == 3
    assert result.finished is True
    assert result.stage_submit_ts == 42.0
    audio = result.engine_outputs.outputs[0].multimodal_output
    assert audio["audio"].numel() == 0
    assert audio["sr"] == 22050
    assert cleanups == [["request"]]


@pytest.mark.asyncio
@pytest.mark.parametrize('final_stages', [{1}, {0, 1}])
@pytest.mark.parametrize('real_terminal_output', [False, True])
async def test_terminal_update_waits_for_codec_acknowledgement(monkeypatch, final_stages, real_terminal_output):
    calls = []

    async def original_update(self, msg):
        calls.append('update')

    async def original_outputs(self, stage_id, replica_id, outputs):
        calls.append('outputs')
        for output in outputs:
            await self.output_async_queue.put(output)
            await self._cleanup_request_ids([output.request_id])

    monkeypatch.setattr(Orchestrator, '_handle_streaming_update', original_update)
    monkeypatch.setattr(Orchestrator, '_handle_processed_outputs', original_outputs)
    patch_streaming_lifecycle()
    state = SimpleNamespace(
        final_stage_id=1,
        final_output_stage_ids=final_stages,
        finished_final_output_stage_ids=set(),
        stage_submit_ts={1: 42.0},
        streaming=SimpleNamespace(enabled=True, segment_finished=False),
    )
    pool = SimpleNamespace(
        stage_client=SimpleNamespace(final_output_type='audio', audio_sample_rate=32000),
        get_bound_replica_id=lambda rid: 0,
    )
    cleanups = []

    async def cleanup(ids):
        cleanups.append(ids)
        for rid in ids:
            orchestrator.request_states.pop(rid, None)

    orchestrator = SimpleNamespace(
        request_states={'request': state},
        stage_pools=[None, pool],
        output_async_queue=asyncio.Queue(),
        _cleanup_request_ids=cleanup,
    )
    await Orchestrator._handle_streaming_update(
        orchestrator, SimpleNamespace(request_id='request', prompt=SimpleNamespace(resumable=False))
    )
    # Text completion is not proof the codec has consumed its queued chunks.
    assert orchestrator.output_async_queue.empty()
    assert cleanups == []
    state.finished_final_output_stage_ids.add(0)
    await Orchestrator._handle_processed_outputs(orchestrator, 0, 0, [])
    assert orchestrator.output_async_queue.empty()
    # Upstream records this raw terminal acknowledgement before processing
    # outputs, even if the output processor drops the empty terminal output.
    state.finished_final_output_stage_ids.add(1)
    terminal = SimpleNamespace(request_id='request', finished=True)
    await Orchestrator._handle_processed_outputs(orchestrator, 1, 0, [terminal] if real_terminal_output else [])
    result = orchestrator.output_async_queue.get_nowait()
    assert result.finished is True
    if real_terminal_output:
        assert result is terminal
    assert orchestrator.output_async_queue.empty()
    assert orchestrator._easymagpie_terminal_requests == set()
    assert cleanups == [['request']]
    assert calls == ['update', 'outputs', 'outputs']
