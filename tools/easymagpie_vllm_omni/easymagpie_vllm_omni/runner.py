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
