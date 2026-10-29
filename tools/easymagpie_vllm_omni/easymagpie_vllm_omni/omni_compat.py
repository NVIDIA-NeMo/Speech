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
    from vllm_omni.engine.orchestrator import Orchestrator

    original_outputs = Orchestrator._handle_processed_outputs
    if not getattr(original_outputs, "_easymagpie_patched", False):

        @wraps(original_outputs)
        async def patched_outputs(self, stage_id, replica_id, outputs):
            for output in outputs:
                request_state = self.request_states.get(output.request_id)
                if request_state is not None:
                    preserve_resumable_segment(output, request_state)
            result = await original_outputs(self, stage_id, replica_id, outputs)
            # Raw terminal acknowledgements are recorded before this callback.
            # Process the real audio outputs first; only then bridge a terminal
            # acknowledgement dropped by the output processor.
            for request_id in list(getattr(self, '_easymagpie_terminal_requests', ())):
                await _complete_drained_stream(self, request_id)
            return result

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

        # Text input ending does not mean the downstream codec has drained.
        # Keep the request live until every selected output stage acknowledges
        # its terminal connector payload.
        pending = getattr(self, '_easymagpie_terminal_requests', None)
        if pending is None:
            pending = self._easymagpie_terminal_requests = set()
        pending.add(msg.request_id)
        await _complete_drained_stream(self, msg.request_id)

    patched_update._easymagpie_terminal_patched = True
    Orchestrator._handle_streaming_update = patched_update


async def _complete_drained_stream(orchestrator, request_id) -> None:
    """Bridge a dropped terminal output only after downstream acknowledgement."""
    pending = orchestrator._easymagpie_terminal_requests
    state = orchestrator.request_states.get(request_id)
    if state is None:
        pending.discard(request_id)
        return
    stages = state.final_output_stage_ids or {state.final_stage_id}
    if not stages.issubset(state.finished_final_output_stage_ids):
        return

    from vllm_omni.engine.messages import OutputMessage
    from vllm_omni.engine.orchestrator import _build_terminal_empty_output, _infer_stage_audio_sample_rate

    # vLLM may drop the terminal output after removing a streaming session.
    # Its raw acknowledgement still proves the codec consumed the final chunk.
    pool = orchestrator.stage_pools[state.final_stage_id]
    terminal = _build_terminal_empty_output(
        request_id,
        final_output_type=getattr(pool.stage_client, 'final_output_type', None),
        audio_sample_rate=_infer_stage_audio_sample_rate(pool),
    )
    await orchestrator.output_async_queue.put(
        OutputMessage(
            request_id=request_id,
            stage_id=state.final_stage_id,
            replica_id=pool.get_bound_replica_id(request_id) or 0,
            engine_outputs=terminal,
            metrics=None,
            finished=True,
            stage_submit_ts=state.stage_submit_ts.get(state.final_stage_id),
        )
    )
    pending.discard(request_id)
    await orchestrator._cleanup_request_ids([request_id])
