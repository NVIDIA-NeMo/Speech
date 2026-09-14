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
"""Compatibility fixes for the pinned vLLM-Omni streaming lifecycle."""
from __future__ import annotations

from functools import wraps


def preserve_resumable_segment(output, request_state) -> bool:
    """Expose a per-turn stop without completing the whole streaming request."""
    streaming = getattr(request_state, "streaming", None)
    # vLLM clears ``finished`` on resumable outputs, so use the raw per-segment boundary.
    is_segment = bool(getattr(streaming, "enabled", False)) and bool(getattr(streaming, "segment_finished", False))
    if not is_segment:
        return False

    output.finished = False
    output.is_segment_finished = True
    return True


def patch_streaming_lifecycle() -> None:
    """Patch segment and request completion for the pinned vLLM-Omni lifecycle."""
    from vllm_omni.engine.messages import OutputMessage
    from vllm_omni.engine.orchestrator import (
        Orchestrator,
        _build_terminal_empty_output,
        _infer_stage_audio_sample_rate,
    )

    original_outputs = Orchestrator._handle_processed_outputs
    if not getattr(original_outputs, "_easymagpie_patched", False):

        @wraps(original_outputs)
        async def patched_outputs(self, stage_id, replica_id, outputs):
            for output in outputs:
                request_state = self.request_states.get(output.request_id)
                if request_state is not None:
                    preserve_resumable_segment(output, request_state)
            return await original_outputs(self, stage_id, replica_id, outputs)

        patched_outputs._easymagpie_patched = True
        Orchestrator._handle_processed_outputs = patched_outputs

    original_update = Orchestrator._handle_streaming_update
    if getattr(original_update, "_easymagpie_terminal_patched", False):
        return

    @wraps(original_update)
    async def patched_update(self, msg):
        await original_update(self, msg)
        if bool(getattr(msg.prompt, "resumable", False)):
            return

        request_state = self.request_states.get(msg.request_id)
        if request_state is None:
            return

        # TODO(vLLM-Omni upstream): a final streaming update can arrive after
        # every stage is already WAITING_FOR_STREAMING_REQ. Each scheduler then
        # removes its request without a model step, so no processed output ever
        # reaches Orchestrator._route_output() and AsyncOmni.generate() waits
        # forever. The orchestrator owns the session and must publish terminal
        # frontend state after submitting the non-resumable update. Remove this
        # private-state bridge once upstream handles that lifecycle explicitly.
        final_stage_id = request_state.final_stage_id
        final_pool = self.stage_pools[final_stage_id]
        terminal_output = _build_terminal_empty_output(
            msg.request_id,
            final_output_type=getattr(final_pool.stage_client, "final_output_type", None),
            audio_sample_rate=_infer_stage_audio_sample_rate(final_pool),
        )
        await self.output_async_queue.put(
            OutputMessage(
                request_id=msg.request_id,
                stage_id=final_stage_id,
                replica_id=final_pool.get_bound_replica_id(msg.request_id) or 0,
                engine_outputs=terminal_output,
                metrics=None,
                finished=True,
                stage_submit_ts=request_state.stage_submit_ts.get(final_stage_id),
            )
        )
        await self._cleanup_request_ids([msg.request_id])

    patched_update._easymagpie_terminal_patched = True
    Orchestrator._handle_streaming_update = patched_update
