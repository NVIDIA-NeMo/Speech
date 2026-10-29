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
"""vLLM-Omni 0.24 streaming-input compatibility classes.

vLLM-Omni 0.24 no longer merges a resumed request's
``additional_information`` into ``model_intermediate_buffer``. Consequently,
per-chunk EasyMagpie ``text_token`` payloads reach the scheduler but not the
model runner. The custom runner restores the merge performed by 0.21 while
preserving model-generated state such as ``decode_offset`` and ``text_tokens``.
"""
from __future__ import annotations

from typing import Any

import torch
from easymagpie_vllm_omni.cfg_pairing import build_cfg_pairing_layout, is_cfg_hidden_id
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.worker import gpu_ar_worker
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_ar_worker import GPUARWorker


def merge_streaming_additional_information(
    cached: dict[str, Any],
    incoming: dict[str, Any],
    accumulated_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Merge one streaming chunk without dropping persistent model state."""
    accumulated_keys = accumulated_keys or set()
    merged = dict(cached)

    for key, value in incoming.items():
        if not isinstance(value, dict):
            merged[key] = value
            continue

        old_value = merged.get(key)
        merged_sub = dict(old_value) if isinstance(old_value, dict) else {}
        for subkey, subvalue in value.items():
            if (key, subkey) in accumulated_keys and isinstance(subvalue, torch.Tensor):
                new_tensor = subvalue.detach().to("cpu").contiguous()
                old_tensor = merged_sub.get(subkey)
                merged_sub[subkey] = new_tensor if old_tensor is None else torch.cat((old_tensor, new_tensor), dim=0)
            else:
                merged_sub[subkey] = subvalue
        merged[key] = merged_sub

    meta = dict(merged.get("meta", {}))
    meta["num_processed_tokens"] = 0
    meta["resumable"] = True
    merged["meta"] = meta
    return merged


class EasyMagpieGPUARModelRunner(GPUARModelRunner):
    """GPU AR runner that restores streaming chunk metadata propagation."""

    def _preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors=None):
        result = super()._preprocess(scheduler_output, num_input_tokens, intermediate_tensors)
        # Input preparation may reorder requests. Build the mapping afterwards,
        # before forward, using unpadded physical rows and absolute positions.
        model = self.get_model()  # CUDA graph wrappers do not forward attribute assignments.
        request_ids = self.input_batch.req_ids
        graph_buffers = getattr(model, "cfg_graph_buffers", None)
        if not any(is_cfg_hidden_id(rid) for rid in request_ids):
            if hasattr(model, "cfg_pairing_layout"):
                model.cfg_pairing_layout = None
                model.cfg_guidance_layout = None
        else:
            counts = [scheduler_output.num_scheduled_tokens[rid] for rid in request_ids]
            metadata = {
                rid: getattr(self.requests[rid], "additional_information_cpu", None) or {} for rid in request_ids
            }
            model.cfg_pairing_layout = build_cfg_pairing_layout(
                request_ids,
                counts,
                self.input_batch.num_computed_tokens_cpu[: len(request_ids)],
                metadata,
            )
            model.cfg_guidance_layout = (
                model.cfg_pairing_layout if any(info.get("use_cfg") is True for info in metadata.values()) else None
            )
        if graph_buffers is not None:
            actual_tokens = sum(scheduler_output.num_scheduled_tokens[rid] for rid in request_ids)
            graph_buffers.update(model.cfg_guidance_layout, actual_tokens, num_input_tokens)
            # Preprocess writes live rows only. Clear padding which may retain a
            # completed request's audio/phoneme validity from a larger prior batch.
            model._dec_audio_valid[actual_tokens:num_input_tokens].zero_()
            if model.has_phoneme:
                model._dec_phoneme_valid[actual_tokens:num_input_tokens].zero_()
        return result

    def _request_needs_downstream_stage_payload(self, req_id):
        # Suppress private branch postprocessing/payload construction. Scheduler
        # filtering additionally protects terminal connector notifications.
        if is_cfg_hidden_id(req_id):
            return False
        return super()._request_needs_downstream_stage_payload(req_id)

    def _update_streaming_request(self, req_id, new_req_data):
        payload = getattr(new_req_data, "additional_information", None)
        incoming = deserialize_additional_information(payload)
        if isinstance(incoming, dict) and incoming:
            model = getattr(self, "model", None)
            accumulated_keys = getattr(model, "streaming_accumulated_keys", set())
            cached = self.model_intermediate_buffer.get(req_id, {})
            merged = merge_streaming_additional_information(cached, incoming, accumulated_keys)
            self.model_intermediate_buffer[req_id] = merged
            setattr(self.requests[req_id], "additional_information_cpu", merged)

        return super()._update_streaming_request(req_id, new_req_data)


class EasyMagpieGPUARWorker(GPUARWorker):
    """GPU AR worker that constructs :class:`EasyMagpieGPUARModelRunner`."""

    def init_device(self):
        # GPUARWorker hardcodes its module-level GPUARModelRunner symbol rather
        # than exposing a runner-class hook. Swap it only while the base method
        # constructs this worker's runner; each worker lives in its own process.
        original_runner_cls = gpu_ar_worker.GPUARModelRunner
        gpu_ar_worker.GPUARModelRunner = EasyMagpieGPUARModelRunner
        try:
            return super().init_device()
        finally:
            gpu_ar_worker.GPUARModelRunner = original_runner_cls
