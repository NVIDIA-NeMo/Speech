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
from types import SimpleNamespace

import pytest
from easymagpie_vllm_omni.omni_compat import patch_streaming_lifecycle, preserve_resumable_segment
from vllm_omni.engine.orchestrator import Orchestrator


def _request_state(*, finished):
    return SimpleNamespace(
        streaming=SimpleNamespace(
            enabled=True,
            segment=lambda stage_id: SimpleNamespace(finished=finished and stage_id == 1),
        )
    )


def test_resumable_segment_is_not_treated_as_request_completion():
    output = SimpleNamespace(finished=True)
    request_state = _request_state(finished=True)

    assert preserve_resumable_segment(output, request_state, stage_id=1) is True
    assert output.finished is False
    assert output.is_segment_finished is True


def test_resumable_segment_marker_survives_vllm_finished_rewrite():
    output = SimpleNamespace(finished=False)
    request_state = _request_state(finished=True)

    assert preserve_resumable_segment(output, request_state, stage_id=1) is True
    assert output.finished is False
    assert output.is_segment_finished is True


def test_terminal_request_completion_is_unchanged():
    output = SimpleNamespace(finished=True)
    request_state = _request_state(finished=False)

    assert preserve_resumable_segment(output, request_state, stage_id=1) is False
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
    request_state = _request_state(finished=True)
    orchestrator = SimpleNamespace(request_states={"request": request_state})

    await patched(orchestrator, 1, 0, [output])

    assert output.finished is False
    assert output.is_segment_finished is True
    assert calls == [(1, 0, [output])]
