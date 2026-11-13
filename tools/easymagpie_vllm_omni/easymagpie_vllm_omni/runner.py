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

from typing import TYPE_CHECKING, Any

import torch
import vllm
import vllm_omni
from easymagpie_vllm_omni.cfg_pairing import build_cfg_pairing_layout, is_cfg_hidden_id
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.worker import gpu_ar_worker
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_ar_worker import GPUARWorker

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ECConnectorOutput


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
    """Streaming metadata and strict, owned EasyMagpie worker feedback."""

    def __init__(self, *args: Any, **kwargs: Any):
        # These hooks are an adapter contract for the audited 0.24 runtime.
        if vllm.__version__ != "0.24.0" or vllm_omni.__version__ != "0.24.0":
            raise RuntimeError("EasyMagpie worker feedback requires vLLM and vLLM-Omni 0.24.0")
        super().__init__(*args, **kwargs)

    def execute_model(self, *args: Any, **kwargs: Any):
        self._check_feedback_health()
        return super().execute_model(*args, **kwargs)

    def sample_tokens(self, *args: Any, **kwargs: Any):
        self._check_feedback_health()
        return super().sample_tokens(*args, **kwargs)

    def _check_feedback_health(self):
        failure = getattr(self, "_feedback_failure", None)
        if failure is not None:
            raise RuntimeError(
                "EasyMagpie worker is unusable after feedback failure; recreate the engine"
            ) from failure

    def _update_intermediate_buffer(self, req_id: str, upd: dict) -> None:
        if not getattr(self.model, "strict_worker_feedback", False):
            return super()._update_intermediate_buffer(req_id, upd)
        if not isinstance(upd, dict) or not upd or req_id not in self.requests:
            return
        declared = getattr(self.model, "gpu_resident_buffer_keys", set())
        flat_keys = {key for key in declared if isinstance(key, str)}
        nested_keys = {key for key in declared if isinstance(key, tuple) and len(key) == 2}
        existing = self.model_intermediate_buffer.setdefault(req_id, {})
        for key, value in upd.items():
            if isinstance(value, dict):
                dest = existing.setdefault(key, {})
                for qualifier, item in value.items():
                    self._store_value(dest, qualifier, item, {q for k, q in nested_keys if k == key})
            else:
                # The base helper clones declared tensors on their original device.
                # Copies and the next preprocess read use the worker's current stream.
                self._store_value(existing, key, value, flat_keys)
        self.requests[req_id].additional_information_cpu = existing

    def _bookkeeping_sync(self, scheduler_output, sampler_output, *args, **kwargs):
        self._check_feedback_health()
        ids = self.input_batch.req_ids
        indices = {rid: i for i, rid in enumerate(ids)}
        sources, destinations = [], []
        for row, rid in enumerate(ids):
            if not is_cfg_hidden_id(rid):
                continue
            info = getattr(self.requests[rid], "additional_information_cpu", None) or {}
            public = info.get("_cfg_pair", {}).get("logical_id")
            if public not in indices:
                raise RuntimeError(f"CFG partner missing before bookkeeping: {rid}")
            source = indices[public]
            # Discard masks describe sampled request rows, not expanded token rows.
            mask = self.discard_request_mask.np
            if mask[row] != mask[source]:
                raise RuntimeError(f"CFG discard masks disagree: {public}, {rid}")
            if not mask[row]:
                sources.append(source)
                destinations.append(row)
        if destinations:
            tokens = sampler_output.sampled_token_ids
            staging = getattr(self, "_cfg_bookkeeping_indices", None)
            if staging is None:
                # These are sampled request rows, independent of graph token rows.
                capacity = getattr(self.input_batch, "max_num_reqs", len(ids))
                staging = self._cfg_bookkeeping_indices = _CFGBookkeepingIndices(capacity, tokens.device)
            src, dst = staging.stage(sources, destinations)
            tokens.index_copy_(0, dst, tokens.index_select(0, src))
        return super()._bookkeeping_sync(scheduler_output, sampler_output, *args, **kwargs)

    def _process_additional_information_updates(
        self,
        hidden_states,
        multimodal_outputs,
        num_scheduled_tokens_np,
        scheduler_output,
        combined_hidden_states=None,
        combined_multimodal_outputs=None,
        req_ids_filter=None,
        req_ids=None,
        query_start_loc_cpu=None,
    ) -> None:
        if not getattr(self.model, "strict_worker_feedback", False):
            return super()._process_additional_information_updates(
                hidden_states,
                multimodal_outputs,
                num_scheduled_tokens_np,
                scheduler_output,
                combined_hidden_states,
                combined_multimodal_outputs,
                req_ids_filter,
                req_ids,
                query_start_loc_cpu,
            )
        self._check_feedback_health()
        ids = list(self.input_batch.req_ids if req_ids is None else req_ids)
        starts = self.query_start_loc.cpu if query_start_loc_cpu is None else query_start_loc_cpu
        starts = starts() if callable(starts) else starts
        request_id, phase = None, "select"
        try:
            for index, request_id in enumerate(ids):
                # Private inputs use the guided model buffers; they never owned a
                # postprocess update. Public feedback is independent of output routing.
                if is_cfg_hidden_id(request_id):
                    continue
                if request_id not in self.requests:
                    raise RuntimeError(f"Feedback request state is missing: {request_id}")
                count = int(num_scheduled_tokens_np[index])
                if count == 0:
                    continue
                phase = f"postprocess span={int(starts[index])}:{int(starts[index]) + count}"
                if combined_hidden_states or combined_multimodal_outputs:
                    raise RuntimeError("EasyMagpie feedback requires scheduled spans, not prefix-cache payloads")
                start = int(starts[index])
                span = hidden_states[start : start + count]
                if span.shape[0] != count:
                    raise RuntimeError("Incomplete scheduled feedback span")
                info = self.model_intermediate_buffer.get(request_id, {})
                update = self.model.postprocess(
                    span,
                    multimodal_outputs=multimodal_outputs,
                    **{k: v for k, v in info.items() if k != "hidden_states"},
                )
                required = ["last_audio_codes"]
                if self.model.has_phoneme:
                    required.append("last_phoneme_token")
                for key in required:
                    value = update.get(key) if isinstance(update, dict) else None
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.ndim < 1
                        or value.shape[0] != 1
                        or not value.numel()
                    ):
                        raise RuntimeError(f"Missing or invalid required feedback: {key}")
                phase = "store " + ", ".join(required)
                self._update_intermediate_buffer(request_id, update)
        except Exception as error:
            self._feedback_failure = error
            error.add_note(f"EasyMagpie feedback failed for request={request_id!r}, phase={phase}, batch={ids!r}")
            raise

    def _build_omni_model_runner_output_from_snapshot(self, **kwargs):
        if getattr(self.model, "strict_worker_feedback", False) and not kwargs.get(
            "postprocess_already_applied", False
        ):
            self._process_additional_information_updates(
                kwargs["hidden_states"],
                kwargs["multimodal_outputs"],
                kwargs["num_scheduled_tokens_np"],
                kwargs["scheduler_output"],
                req_ids=kwargs["req_ids_output_copy"],
                query_start_loc_cpu=kwargs["query_start_loc_cpu"],
            )
            # This marker belongs to this result invocation, never runner-global.
            kwargs["postprocess_already_applied"] = True
        return super()._build_omni_model_runner_output_from_snapshot(**kwargs)

    def _preprocess(
        self,
        scheduler_output: SchedulerOutput,
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        """Stage CFG row ownership after the base runner reorders the input batch.

        Args:
            scheduler_output: Scheduled physical requests and token counts for this step.
            num_input_tokens: Padded physical token count used by the model invocation.
            intermediate_tensors: Intermediate tensors supplied by pipeline parallel execution.

        Returns:
            The base runner preprocessing result, with CFG ownership buffers staged in place.
        """
        model = self.get_model()
        if hasattr(model, "use_fixed_acoustic_sampling"):
            model.use_fixed_acoustic_sampling = True
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

    def _model_forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Bypass captured default constants for batches with custom settings.

        Args:
            *args: Model inputs forwarded to the Omni runner.
            **kwargs: Model inputs forwarded to the Omni runner.

        Returns:
            The model output for this batch.
        """
        if getattr(self.get_model(), "use_fixed_acoustic_sampling", True):
            return super()._model_forward(*args, **kwargs)
        context = get_forward_context()
        previous_mode = context.cudagraph_runtime_mode
        context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        try:
            return super()._model_forward(*args, **kwargs)
        finally:
            context.cudagraph_runtime_mode = previous_mode

    def _request_needs_downstream_stage_payload(self, req_id: str) -> bool:
        # Suppress private branch postprocessing/payload construction. Scheduler
        # filtering additionally protects terminal connector notifications.
        """Check whether a public request needs a codec-stage payload.

        Args:
            req_id: Request identity.

        Returns:
            Whether the request should produce a downstream codec payload.
        """
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


class _CFGBookkeepingIndices:
    """Bounded staging for sampled request-row indices, outside model graph capture.

    Upload and index consumers must use the same ordered worker CUDA stream.
    The device scratch may then be overwritten by the next upload only after
    earlier consumers finish. Host storage needs a separate copy-completion fence
    because a nonblocking upload may still read it after returning to Python.
    Two pinned slots permit overlap; only pending slot reuse waits on the host.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self._device = torch.empty(2 * capacity, dtype=torch.long, device=device)
        cuda = self._device.is_cuda
        self._host_slots = [
            torch.empty(2 * capacity, dtype=torch.long, device="cpu", pin_memory=cuda) for _ in range(2)
        ]
        self._host_views = [slot.numpy() for slot in self._host_slots]
        self._copy_events = [torch.cuda.Event() for _ in self._host_slots] if cuda else None
        self._next_slot = 0

    def stage(self, sources: list[int], destinations: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Upload both index lists together and return their device views."""
        count = len(sources)
        if count != len(destinations) or 2 * count > self._device.numel():
            raise ValueError("CFG bookkeeping indices exceed request capacity or disagree in length")
        slot = self._next_slot
        event = self._copy_events[slot] if self._copy_events is not None else None
        if event is not None and not event.query():
            event.synchronize()
        host = self._host_views[slot]
        host[:count] = sources
        host[count : 2 * count] = destinations
        self._device[: 2 * count].copy_(self._host_slots[slot][: 2 * count], non_blocking=True)
        if event is not None:
            event.record(torch.cuda.current_stream(self._device.device))
        self._next_slot = (slot + 1) % len(self._host_slots)
        return self._device[:count], self._device[count : 2 * count]
