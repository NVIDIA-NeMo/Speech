# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Synchronous V2 CFG scheduling and the restricted cache-pairing diagnostic."""
from __future__ import annotations

import copy
from dataclasses import dataclass

from easymagpie_vllm_omni.cfg import AcousticSamplingSettings
from easymagpie_vllm_omni.cfg_graph import supports_cfg_graphs
from easymagpie_vllm_omni.cfg_pairing import (
    CFG_PAIR_METADATA_KEY,
    CFG_PAIRING_PROBE_KEY,
    CFGPair,
    CFGPairingError,
    CFGPairingTransferFilter,
    filter_cfg_outputs,
    is_cfg_hidden_id,
)
from easymagpie_vllm_omni.config import EasyMagpieOmniArch
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.request import RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.request import OmniRequest


class EasyMagpieCFGPairingScheduler(OmniARScheduler):
    """Allocate real engine histories for two branches and expose one logical output.

    Admission is limited to one non-resumable, known-speaker request. Insufficient
    cache capacity or a split step aborts both branches before worker execution;
    this diagnostic deliberately does not preempt or replay Mamba state.
    """

    single_request = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_cfg_pairing_config(self.vllm_config, allow_decode_graphs=not self.single_request)
        self.cfg_pairs: dict[str, CFGPair] = {}
        self._cfg_step_pending = False
        if self.chunk_transfer_adapter is not None:
            self.chunk_transfer_adapter = CFGPairingTransferFilter(self.chunk_transfer_adapter)

    @property
    def cfg_pair(self) -> CFGPair | None:
        """The diagnostic's sole pair; mixed scheduling uses cfg_pairs explicitly."""
        return next(iter(self.cfg_pairs.values()), None)

    def add_request(self, request) -> None:
        if self.single_request and self.cfg_pairs:
            raise CFGPairingError("CFG pairing probe accepts only one logical request at a time")
        info = deserialize_additional_information(request.additional_information)
        settings = AcousticSamplingSettings.from_info(info)
        _validate_probe_request(
            request, info, self.max_num_scheduled_tokens, allow_conditional=not self.single_request
        )
        if request.request_id in self.requests:
            raise ValueError("Duplicate logical request ID")
        if settings.use_cfg:
            arch = EasyMagpieOmniArch.from_hf_config(self.vllm_config.model_config.hf_config)
            if arch.cfg_capability is None:
                raise ValueError(
                    "CFG requires cfg_capability and cfg_context_embedding; reconvert the NeMo checkpoint"
                )
            if not request.resumable and (info.get("text_token") or not (info.get("text") or info.get("text_tokens"))):
                raise ValueError("Initial CFG synthesis requires complete target text")
        if not settings.use_cfg and info.get(CFG_PAIRING_PROBE_KEY) is not True:
            self._validate_reservation([request])
            super().add_request(request)
            return
        pair = CFGPair.create(request.request_id)
        # Construct fresh Requests: shallow-copying a Request aliases token lists,
        # status/events, and other mutable state even if its request_id is changed.
        shadow = OmniRequest(
            request_id=pair.unconditional_id,
            external_req_id=pair.unconditional_id,
            client_index=request.client_index,
            prompt_token_ids=list(request.prompt_token_ids),
            sampling_params=copy.deepcopy(request.sampling_params),
            pooling_params=None,
            arrival_time=request.arrival_time,
            priority=request.priority,
            cache_salt=pair.unconditional_id,
            resumable=request.resumable,
            additional_information=serialize_additional_information({**info, **pair.metadata(pair.unconditional_id)}),
        )
        if hasattr(request, '_cfg_reservation_tokens'):
            shadow._cfg_reservation_tokens = request._cfg_reservation_tokens
        if not self.single_request:
            self._validate_reservation([request, shadow])
        original_info = request.additional_information
        request.additional_information = serialize_additional_information(
            {**info, **pair.metadata(pair.conditional_id)}
        )
        self.cfg_pairs[pair.conditional_id] = pair
        try:
            super().add_request(request)
            super().add_request(shadow)
        except Exception:
            self.finish_requests(pair.physical_ids, RequestStatus.FINISHED_ABORTED)
            request.additional_information = original_info
            raise

    def schedule(self, *args, **kwargs):
        if self._cfg_step_pending:
            raise CFGPairingError("CFG pairing requires synchronous output before the next step")
        try:
            waiting = self.waiting
            try:
                if not self.single_request:
                    self.waiting = self._admit_waiting_units()
                return super().schedule(*args, **kwargs)
            finally:
                if self.waiting is not waiting:
                    # Preserve FCFS order when a pause prevents consumption.
                    for request in reversed(list(self.waiting)):
                        waiting.prepend_request(request)
                    self.waiting = waiting
        except Exception:
            # The stock scheduler may have admitted only the first branch or
            # preempted one before our validation hook. Neither may reach forward.
            self.finish_requests(None, RequestStatus.FINISHED_ABORTED)
            raise

    def _update_after_schedule(self, scheduler_output) -> None:
        if scheduler_output.preempted_req_ids:
            raise CFGPairingError("CFG scheduling does not support preemption/recomputation")
        for pair in self.cfg_pairs.values():
            pair.validate_step(
                scheduler_output.num_scheduled_tokens,
                {rid: self.requests[rid].num_computed_tokens for rid in pair.physical_ids},
            )
        # Validate before the stock hook advances computed-token counters and
        # before metadata/block tables can be sent to a worker.
        super()._update_after_schedule(scheduler_output)
        self._cfg_step_pending = bool(scheduler_output.total_num_scheduled_tokens)

    def update_from_output(self, scheduler_output, model_runner_output):
        try:
            for pair in list(self.cfg_pairs.values()):
                if not all(rid in self.requests for rid in pair.physical_ids) or not any(
                    rid in scheduler_output.num_scheduled_tokens for rid in pair.physical_ids
                ):
                    continue
                # The public branch owns the engine continue/stop decision.
                # The model separately shares guided acoustic feedback.
                by_id = model_runner_output.req_id_to_index
                sampled = model_runner_output.sampled_token_ids
                if sampled:
                    copied = list(sampled)
                    copied[by_id[pair.unconditional_id]] = list(sampled[by_id[pair.conditional_id]])
                    model_runner_output = copy.copy(model_runner_output)
                    model_runner_output.sampled_token_ids = copied
            outputs = super().update_from_output(scheduler_output, model_runner_output)
            filter_cfg_outputs(outputs)
            for pair in list(self.cfg_pairs.values()):
                alive = [rid in self.requests for rid in pair.physical_ids]
                if any(alive) and not all(alive):
                    self.finish_requests(pair.physical_ids, RequestStatus.FINISHED_ABORTED)
                    raise CFGPairingError("CFG branch finished without its partner")
                if not any(alive):
                    self.cfg_pairs.pop(pair.conditional_id, None)
            return outputs
        except Exception:
            self.finish_requests(None, RequestStatus.FINISHED_ABORTED)
            raise
        finally:
            self._cfg_step_pending = False

    def finish_requests(self, request_ids, finished_status):
        if request_ids is None:
            ids = list(self.requests)
        elif isinstance(request_ids, str):
            ids = [request_ids]
        else:
            ids = list(request_ids)
        for pair in self.cfg_pairs.values():
            if set(ids).intersection(pair.physical_ids):
                ids = list(dict.fromkeys([*ids, *pair.physical_ids]))
        result = super().finish_requests(ids, finished_status)
        for pair in list(self.cfg_pairs.values()):
            if not any(rid in self.requests for rid in pair.physical_ids):
                self.cfg_pairs.pop(pair.conditional_id, None)
        # Worker-facing finished_req_ids still includes BOTH physical identities.
        # Hide the internal identity only from frontend abort notifications.
        return [(rid, client) for rid, client in result if not is_cfg_hidden_id(rid)]


class EasyMagpieCFGScheduler(EasyMagpieCFGPairingScheduler):
    """FCFS mixed CFG/conditional requests with whole-request cache reservation.

    A logical unit needs two sequence slots for CFG and one otherwise. Reserve
    its entire bounded history before admission so decode never needs preemption.
    This trades cache utilization for safe Mamba ownership and bounded scheduling.
    """

    single_request = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.policy.value != "fcfs" or self.cache_config.mamba_cache_mode != "none":
            raise ValueError("Mixed CFG scheduling requires FCFS and mamba_cache_mode=none")
        if self.max_num_scheduled_tokens < self.max_num_running_reqs:
            raise ValueError("Mixed CFG token budget must cover every physical decode sequence")
        self._cfg_streams: dict[str, _CFGTextStream] = {}

    def add_request(self, request):
        existing = self.requests.get(request.request_id)
        if existing is not None and request.request_id in self._cfg_streams:
            self._add_stream_update(request)
            return
        if not request.resumable:
            return super().add_request(request)
        info = deserialize_additional_information(request.additional_information)
        bound = info.get('max_stream_tokens')
        if type(bound) is not int or not 0 < bound <= self.max_model_len - request.num_prompt_tokens:
            raise ValueError('Streaming requires max_stream_tokens within the model history capacity')
        if request.max_tokens > bound:
            raise ValueError('Streaming segment exceeds max_stream_tokens')
        tokens = info.get('text_token')
        if not tokens or info.get('text_token_start') != 0 or info.get('text') or info.get('text_tokens'):
            raise ValueError('Initial streaming text requires text_token at text_token_start=0')
        warmup = info.get('text_prefill_num', 0)
        if type(warmup) is not int or warmup < 0 or warmup > len(tokens):
            raise ValueError('Streaming text must cover the initial text warmup')
        if request.max_tokens != len(tokens) - warmup + int(warmup > 0):
            raise ValueError('Streaming text must match its segment generation limit')
        if request.request_id in self.requests:
            raise ValueError('Duplicate logical request ID')
        if '_cfg_stream_prompt_len' in info:
            raise ValueError('Streaming prompt boundary is reserved for the scheduler')
        info['_cfg_stream_prompt_len'] = request.num_prompt_tokens
        request.additional_information = serialize_additional_information(info)
        request._cfg_reservation_tokens = request.num_prompt_tokens + bound
        super().add_request(request)
        self._cfg_streams[request.request_id] = _CFGTextStream(dict(info), list(tokens), bound, request.max_tokens)

    def update_from_output(self, scheduler_output, model_runner_output):
        streams = set(self._cfg_streams)
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        for client_outputs in outputs.values():
            for output in client_outputs.outputs:
                if output.request_id in streams and output.request_id not in self.requests:
                    output.is_segment_finished = False
        # Omni's resumable stop hardcodes one async discard. There are no
        # outstanding outputs in this explicitly synchronous scheduler.
        for request in self.requests.values():
            request.async_tokens_to_discard = 0
        self._prune_streams()
        return outputs

    def finish_requests(self, request_ids, finished_status):
        result = super().finish_requests(request_ids, finished_status)
        self._prune_streams()
        return result

    def _add_stream_update(self, request):
        stream = self._cfg_streams[request.request_id]
        session = self.requests[request.request_id]
        pair = self.cfg_pairs.get(request.request_id)
        ids = pair.physical_ids if pair else (request.request_id,)
        incoming = deserialize_additional_information(request.additional_information)
        if stream.closed:
            raise ValueError('Cannot update a closed text stream')
        if CFG_PAIR_METADATA_KEY in incoming:
            raise ValueError('CFG branch identity is reserved for the scheduler')
        allowed = {'text_token', 'text_token_start'}
        for key, value in incoming.items():
            if key not in allowed and (key not in stream.info or value != stream.info[key]):
                raise ValueError(f'Streaming cannot change request setting {key}')
        if request.mm_features or request.prompt_embeds is not None or request.lora_request is not None:
            raise ValueError('Streaming accepts registered-speaker text updates only')
        if not request.resumable:
            # The engine sends a final sentinel after exhausting its input iterator.
            # Waiting sessions need a terminal connector payload before freeing.
            if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                if self.chunk_transfer_adapter is not None:
                    terminal = copy.copy(session)
                    terminal.resumable = False
                    terminal.status = RequestStatus.FINISHED_STOPPED
                    self.chunk_transfer_adapter.save_async(None, terminal, False)
                self.finish_requests(ids, RequestStatus.FINISHED_ABORTED)
            else:
                for rid in ids:
                    self.requests[rid].streaming_queue.append(None)
                stream.closed = True
            return
        if request.prompt_token_ids != [0]:
            raise ValueError('Streaming continuation requires one placeholder token')
        tokens = list(incoming.get('text_token') or [])
        start = incoming.get('text_token_start')
        if tokens:
            if type(start) is not int or start < 0 or start > len(stream.text_tokens):
                raise ValueError('Invalid streaming text_token_start')
            if start < len(stream.text_tokens):
                if stream.text_tokens[start : start + len(tokens)] == tokens:
                    return  # Identical redelivery must not advance either history.
                raise ValueError('Conflicting or partially overlapping streaming text chunk')
        elif stream.tail_submitted:
            raise ValueError('Streaming acoustic tail has already been submitted')
        if tokens and request.max_tokens != len(tokens):
            raise ValueError('Streaming text must match its segment generation limit')
        if stream.submitted_tokens + request.max_tokens > stream.max_tokens:
            raise ValueError('Streaming segments exceed max_stream_tokens')
        if tokens and stream.tail_submitted:
            raise ValueError('Cannot append text after the acoustic tail')
        info = {**stream.info, **incoming}
        updates = []
        for rid in ids:
            physical = OmniRequest(
                request_id=rid,
                external_req_id=self.requests[rid].external_req_id,
                client_index=request.client_index,
                prompt_token_ids=[0],
                sampling_params=copy.deepcopy(request.sampling_params),
                pooling_params=None,
                resumable=True,
                additional_information=serialize_additional_information(
                    {**info, **(pair.metadata(rid) if pair else {})}
                ),
            )
            updates.append(physical)
        try:
            for physical in updates:
                # Bypass duplicate-ID admission: vLLM queues or resumes this update.
                OmniARScheduler.add_request(self, physical)
        except Exception:
            self.finish_requests(ids, RequestStatus.FINISHED_ABORTED)
            raise
        stream.text_tokens.extend(tokens)
        stream.submitted_tokens += request.max_tokens
        stream.tail_submitted = not tokens

    def _update_request_as_session(self, session, update):
        super()._update_request_as_session(session, update)
        session.max_tokens = update.max_tokens
        session.additional_information = update.additional_information
        session.async_tokens_to_discard = 0

    def _enqueue_waiting_request(self, request):
        # Keep parked streams in the same admission queue. Upstream puts them
        # in skipped_waiting, which otherwise bypasses whole-pair admission.
        self.waiting.add_request(request)

    def _handle_stopped_request(self, request):
        if request.resumable and request.streaming_queue and request.streaming_queue[0] is None:
            request.resumable = False
        return super()._handle_stopped_request(request)

    def _prune_streams(self):
        for rid in list(self._cfg_streams):
            if rid not in self.requests:
                self._cfg_streams.pop(rid)

    def _reservation_size(self, request):
        return getattr(
            request, '_cfg_reservation_tokens', min(request.num_prompt_tokens + request.max_tokens, self.max_model_len)
        )

    def _validate_reservation(self, requests):
        manager = self.kv_cache_manager
        needed = sum(
            manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=self._reservation_size(request),
                new_computed_blocks=manager.empty_kv_cache_blocks.blocks,
                num_encoder_tokens=0,
                total_computed_tokens=0,
                num_tokens_main_model=self._reservation_size(request),
            )
            for request in requests
        )
        # get_num_blocks_to_allocate may create empty per-request bookkeeping.
        for request in requests:
            manager.free(request)
        if needed + manager.watermark_blocks > manager.block_pool.num_gpu_blocks - 1:
            raise ValueError("Logical request's full cache reservation exceeds total cache capacity")

    def _admit_waiting_units(self):
        admitted = create_request_queue(self.policy)
        slots = self.max_num_running_reqs - len(self.running)
        tokens = self.max_num_scheduled_tokens - sum(r.num_tokens - r.num_computed_tokens for r in self.running)
        seen = set()
        # A parked stream owns its reservation. Resume owners before fresh
        # admissions so a new request waiting for those blocks cannot deadlock
        # the stream that must finish to release them. Fresh admission stays FCFS.
        for request in sorted(self.waiting, key=lambda r: not getattr(r, '_cfg_cache_reserved', False)):
            if request.request_id in seen:
                continue
            pair = self.cfg_pairs.get(request.request_id)
            if pair is None and is_cfg_hidden_id(request.request_id):
                pair = next(p for p in self.cfg_pairs.values() if p.unconditional_id == request.request_id)
            unit = [self.requests[rid] for rid in pair.physical_ids] if pair else [request]
            seen.update(r.request_id for r in unit)
            if any(r.status == RequestStatus.WAITING_FOR_STREAMING_REQ for r in unit):
                continue
            cost = sum(r.num_tokens - r.num_computed_tokens for r in unit)
            if len(unit) > slots or cost > tokens:
                break
            reserved = []
            try:
                for physical in unit:
                    if getattr(physical, '_cfg_cache_reserved', False):
                        reserved.append(physical)
                        continue
                    if (
                        self.kv_cache_manager.allocate_slots(
                            physical,
                            self._reservation_size(physical) - physical.num_computed_tokens,
                            has_scheduled_reqs=False,
                            reserved_blocks=self.kv_cache_manager.watermark_blocks,
                        )
                        is None
                    ):
                        break
                    reserved.append(physical)
            except Exception:
                for physical in unit:
                    self.kv_cache_manager.free(physical)
                raise
            if len(reserved) != len(unit):
                for physical in unit:
                    self.kv_cache_manager.free(physical)
                break
            self.waiting.remove_requests(unit)
            for physical in unit:
                physical._cfg_cache_reserved = True
                admitted.add_request(physical)
            slots -= len(unit)
            tokens -= cost
        return admitted


def validate_cfg_pairing_config(config, *, allow_decode_graphs: bool = False) -> None:
    """Fail at startup when the cache probe's execution assumptions do not hold."""
    scheduler, model, cache = config.scheduler_config, config.model_config, config.cache_config
    parallel = config.parallel_config
    compilation = getattr(config, "compilation_config", None)
    decode_graphs = (
        allow_decode_graphs
        and getattr(model.hf_config, "acoustic_predictor_type", None) == "acoustic_codes_predictor"
        and getattr(model.hf_config, "cfg_capability", None) is not None
        and supports_cfg_graphs(compilation)
    )
    if (
        (not model.enforce_eager and not decode_graphs)
        or scheduler.async_scheduling
        or scheduler.enable_chunked_prefill
        or cache.enable_prefix_caching
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or config.speculative_config is not None
    ):
        raise ValueError(
            "CFG pairing probe requires eager (or synthesis-only FULL_DECODE_ONLY without compilation / "
            "FULL_AND_PIECEWISE with VLLM_COMPILE), "
            "synchronous, unchunked execution, no prefix cache or TP/PP"
        )
    if scheduler.max_num_seqs < 2 or scheduler.max_num_batched_tokens < 2:
        raise ValueError("CFG pairing probe requires capacity for two physical sequences")
    if getattr(model.hf_config, "acoustic_predictor_type", None) != "acoustic_codes_predictor":
        raise ValueError("CFG pairing probe requires the V2 acoustic predictor")
    if config.kv_transfer_config is not None or config.ec_transfer_config is not None:
        raise ValueError("CFG pairing probe does not support distributed cache transfer")


def _validate_probe_request(request, info: dict, token_budget: int, *, allow_conditional: bool = False) -> None:
    settings = AcousticSamplingSettings.from_info(info)
    if not allow_conditional and info.get(CFG_PAIRING_PROBE_KEY) is not True and not settings.use_cfg:
        raise ValueError("Set _cfg_pairing_probe=true for the cache probe, or use_cfg=true for guided synthesis")
    if settings.use_cfg and info.get(CFG_PAIRING_PROBE_KEY):
        raise ValueError("Choose either _cfg_pairing_probe or use_cfg")
    if CFG_PAIR_METADATA_KEY in info or is_cfg_hidden_id(request.request_id):
        raise ValueError("CFG branch identity is reserved for the scheduler")
    if (
        (request.resumable and (not allow_conditional or info.get(CFG_PAIRING_PROBE_KEY)))
        or request.mm_features
        or request.prompt_embeds is not None
        or request.lora_request is not None
        or request.pooling_params is not None
        or request.use_structured_output
        or request.kv_transfer_params
        or not info.get("speaker_id")
    ):
        raise ValueError("CFG pairing probe requires non-resumable known-speaker text input without extra model state")
    if (
        allow_conditional
        and not request.resumable
        and (info.get("text_token") or not (info.get("text") or info.get("text_tokens")))
    ):
        if info.get(CFG_PAIRING_PROBE_KEY) is not True:
            raise ValueError("Mixed CFG scheduling requires complete target text")
    branches = 2 if settings.use_cfg or info.get(CFG_PAIRING_PROBE_KEY) is True else 1
    if not request.prompt_token_ids or branches * len(request.prompt_token_ids) > token_budget:
        raise ValueError("CFG pairing probe token budget must fit both complete prompts in one step")


@dataclass
class _CFGTextStream:
    info: dict
    text_tokens: list[int]
    max_tokens: int
    submitted_tokens: int
    tail_submitted: bool = False
    closed: bool = False
