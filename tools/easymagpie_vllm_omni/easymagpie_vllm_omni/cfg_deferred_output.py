# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned, ordered CFG output construction for the pinned V1 runner.

Explicitly select ``EasyMagpieCFGDeferredWorker`` to use this experimental path.
The ordinary worker remains the async-inline reference. Native sampling and its
sampled-token copy are unchanged; only the payload builder moves off-thread.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock, RLock
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner, EasyMagpieGPUARWorker
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner, _clone_cuda_tensor_payload, is_pin_memory_available


@dataclass
class OutputSnapshot:
    """Own a payload until its copy completes, and CPU storage until consumption."""

    payload: Any
    sources: list
    ready: Any = None
    device: Any = None
    pinned: bool = False
    destinations: list = field(default_factory=list)
    metadata: Any = None

    def wait(self):
        if self.ready is not None:
            with torch.cuda.device(self.device):
                self.ready.synchronize()
        self.sources.clear()
        self.ready = None

    @property
    def nbytes(self):
        return _payload_bytes(self.payload) + _payload_bytes(self.sources) + _payload_bytes(self.metadata)


class OrderedOutputQueue:
    """Bound output work through native consumption, including completed jobs."""

    def __init__(self, max_jobs=2, max_bytes=256 * 1024 * 1024):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cfg-output')
        self._lock = Lock()
        self._jobs = {}
        self._sequence = 0
        self._closed = False
        self._failure = None
        self.max_jobs = max_jobs
        self.max_bytes = max_bytes
        self.peak_jobs = 0
        self.peak_bytes = 0

    def check_health(self):
        with self._lock:
            self._check_health_locked()

    def check_capacity(self):
        with self._lock:
            self._check_capacity_locked()

    def fail(self, error):
        with self._lock:
            if self._failure is None:
                self._failure = error

    def submit(self, snapshot, build):
        with self._lock:
            # Admission and shutdown share one lock. A preflight capacity check
            # cannot reserve a slot or protect against a concurrent close/failure.
            self._check_capacity_locked()
            size = snapshot.nbytes
            if sum(job.nbytes for job in self._jobs.values()) + size > self.max_bytes:
                raise RuntimeError('CFG output snapshot-byte budget exceeded')
            sequence = self._sequence
            self._sequence += 1
            job = _OutputJob(sequence, snapshot, size)
            self._jobs[sequence] = job
            self.peak_jobs = max(self.peak_jobs, len(self._jobs))
            self.peak_bytes = max(self.peak_bytes, sum(j.nbytes for j in self._jobs.values()))
            try:
                job.future = self._executor.submit(self._build, job, build)
            except BaseException as error:
                self._jobs.pop(sequence)
                if self._failure is None:
                    self._failure = error
                # No build was accepted. The submitting runner still owns the
                # snapshot and must quiesce its copy before releasing storage.
                raise
        return job

    def _build(self, job, build):
        try:
            # Even a failed predecessor cannot cancel an enqueued CUDA copy.
            job.snapshot.wait()
            with self._lock:
                failure = self._failure
            if failure is not None:
                raise RuntimeError('CFG predecessor output failed') from failure
            with torch.inference_mode():
                output = build()
            with self._lock:
                job.nbytes = _payload_bytes(output) + job.snapshot.nbytes
                total = sum(j.nbytes for j in self._jobs.values())
                self.peak_bytes = max(self.peak_bytes, total)
                if total > self.max_bytes:
                    raise RuntimeError('CFG built-output byte budget exceeded')
            return output
        except BaseException as error:
            self.fail(error)
            raise

    def retire(self, job):
        with self._lock:
            self._jobs.pop(job.sequence, None)
        job.snapshot = None
        job.future = None
        job.native = None

    @property
    def failure(self):
        with self._lock:
            return self._failure

    @property
    def pending(self):
        with self._lock:
            return len(self._jobs), sum(j.nbytes for j in self._jobs.values())

    def shutdown(self):
        with self._lock:
            self._closed = True
            jobs = list(self._jobs.values())
        # No queue lock is held during waits or joins. Never cancel copy jobs.
        self._executor.shutdown(wait=True)
        errors = []
        for job in jobs:
            try:
                if job.result is not None:
                    job.result.drain()
                else:
                    job.future.result()
                    self.retire(job)
            except BaseException as error:
                errors.append(error)
                # An output failure is not a copy-lifetime failure. Recheck both
                # fences without rebuilding or retrying native finalization, and
                # retire storage only if quiescence can actually be established.
                try:
                    if job.result is not None:
                        job.result.drain()
                    else:
                        job.snapshot.wait()
                        self.retire(job)
                except BaseException as drain_error:
                    error.add_note(f'CFG shutdown drain also failed: {drain_error!r}')
        if errors:
            raise RuntimeError('CFG output shutdown failed; executor teardown required') from errors[0]

    def _check_health_locked(self):
        if self._failure is not None:
            raise RuntimeError('CFG output worker failed; recreate the engine') from self._failure
        if self._closed:
            raise RuntimeError('CFG output worker is shutting down')

    def _check_capacity_locked(self):
        self._check_health_locked()
        if len(self._jobs) >= self.max_jobs:
            raise RuntimeError('CFG output pending-job budget exceeded')


class DeferredCFGResult(AsyncModelRunnerOutput):
    """Single-consumer adapter retaining native sampled-token finalization."""

    def __init__(self, queue, job, native):
        self._queue = queue
        self._job = job
        self._lock = RLock()
        self._consumed = False
        job.native = native
        job.result = self

    def get_output(self):
        with self._lock:
            if self._consumed:
                raise RuntimeError('CFG output has already been consumed')
            self._consumed = True
            job = self._job
            try:
                output = job.future.result()
                job.native._model_runner_output = output
                output = job.native.get_output()
            except BaseException as error:
                self._queue.fail(error)
                # A build failure still must drain the native sampled-token copy.
                # Retain the job on a fence error; process teardown owns recovery.
                try:
                    job.native.async_copy_ready_event.synchronize()
                except BaseException as fence_error:
                    error.add_note(f'Native sampled-token drain also failed: {fence_error!r}')
                    raise error from fence_error
                raise
            else:
                self._queue.retire(job)
                return output

    def drain(self):
        # Keep the decision and materialization in one critical section: native
        # result consumption and runner teardown may reach this result together.
        with self._lock:
            if not self._consumed:
                self.get_output()
            elif self._job.native is not None:
                # The payload and sampled-token copies have independent fences.
                # A healthy sampled-token event cannot retire a failed payload DMA.
                self._job.snapshot.wait()
                self._job.native.async_copy_ready_event.synchronize()
                self._queue.retire(self._job)


class EasyMagpieCFGDeferredModelRunner(EasyMagpieGPUARModelRunner):
    """Restricted eager CFG runner with one ordered, owned output builder."""

    def __init__(self, *args, **kwargs):
        self._cfg_lifecycle_lock = RLock()
        self._cfg_stopping = False
        self._cfg_shutdown_complete = False
        self._cfg_shutdown_error = None
        super().__init__(*args, **kwargs)
        self._validate_deferred_profile()
        self._cfg_output_queue = OrderedOutputQueue()
        self._cfg_capture = None
        self._cfg_failed_snapshots = []

    def _validate_deferred_profile(self):
        scheduler = self.vllm_config.scheduler_config
        from easymagpie_vllm_omni.cfg_async_scheduler import EasyMagpieCFGAsyncScheduler

        selected = scheduler.scheduler_cls
        approved_name = f'{EasyMagpieCFGAsyncScheduler.__module__}.{EasyMagpieCFGAsyncScheduler.__qualname__}'
        if selected is not EasyMagpieCFGAsyncScheduler and selected != approved_name:
            raise ValueError('Deferred CFG output requires the explicit CFG async scheduler')
        if not self.use_async_scheduling or not self.model_config.enforce_eager or not self._async_chunk:
            raise ValueError('Deferred CFG output requires eager async scheduling and async_chunk')
        if self.omni_prefix_cache is not None or self.speculative_config is not None:
            raise ValueError('Deferred CFG output excludes prefix caching and speculative decoding')
        if self.model_config.enable_return_routed_experts or self._should_accumulate_full_payload_output():
            raise ValueError('Deferred CFG output excludes routed experts and full-payload accumulation')
        if (
            self.vllm_config.kv_transfer_config is not None
            or self.vllm_config.parallel_config.tensor_parallel_size != 1
        ):
            raise ValueError('Deferred CFG output requires single-rank execution without KV transfer')
        if getattr(self.model_config, 'engine_output_type', None) == 'audio':
            raise ValueError('Deferred CFG output requires the two-stage codec pipeline')

    def _check_feedback_health(self):
        super()._check_feedback_health()
        queue = getattr(self, '_cfg_output_queue', None)
        if queue is not None:
            queue.check_health()

    def _should_use_async_omni_output(self):
        # Capture at the inline hook. Native Omni's per-result threads are not
        # used; native sampled-token copy/finalization stays exactly upstream.
        return False

    def execute_model(self, *args, **kwargs):
        with self._cfg_lifecycle_lock:
            self._check_inference_admission()
            return super().execute_model(*args, **kwargs)

    def sample_tokens(self, *args, **kwargs):
        with self._cfg_lifecycle_lock:
            self._check_inference_admission()
            if self.execute_model_state is None:
                # Native empty-work collection creates no payload job. Late
                # connector signals must still drain when both slots are owned
                # by building or completed-but-unconsumed outputs.
                self._cfg_output_queue.check_health()
            else:
                self._cfg_output_queue.check_capacity()
            captured = []
            self._cfg_capture = captured
            try:
                native = super().sample_tokens(*args, **kwargs)
                if not captured:
                    return native  # Native empty-work/notification-only response.
                if len(captured) != 1 or not isinstance(native, AsyncGPUModelRunnerOutput):
                    raise RuntimeError('Pinned CFG native output contract changed')
                return DeferredCFGResult(self._cfg_output_queue, captured[0], native)
            except BaseException as error:
                self._cfg_output_queue.fail(error)
                raise
            finally:
                self._cfg_capture = None

    def _build_omni_model_runner_output_from_snapshot(self, **kwargs):
        if self._cfg_capture is None:
            raise RuntimeError('Deferred CFG builder invoked outside native sample_tokens')
        self._process_additional_information_updates(
            kwargs['hidden_states'],
            kwargs['multimodal_outputs'],
            kwargs['num_scheduled_tokens_np'],
            kwargs['scheduler_output'],
            req_ids=kwargs['req_ids_output_copy'],
            query_start_loc_cpu=kwargs['query_start_loc_cpu'],
        )
        routing = tuple(
            rid for rid in kwargs['req_ids_output_copy'] if self._request_needs_downstream_stage_payload(rid)
        )
        connector = _owned_cpu_value(self.get_omni_connector_output())
        snapshot = self._capture_cfg_payload(kwargs['multimodal_outputs'])
        try:
            # Preserve original sequence length for native payload slicing without
            # retaining/copying the backbone hidden-state storage.
            hidden_length = kwargs['hidden_states'].shape[0]
            build_args = {
                key: _owned_cpu_value(value)
                for key, value in kwargs.items()
                if key not in {'hidden_states', 'staged_hidden_states_cpu', 'multimodal_outputs'}
            }
            build_args.update(
                hidden_states=torch.empty((hidden_length, 0)),
                staged_hidden_states_cpu=None,
                multimodal_outputs=snapshot.payload,
                postprocess_already_applied=True,
            )
            # The closure retains scheduler metadata and connector values too.
            # Charge their tensor/array storage before enqueue and until native
            # consumption, even when the builder does not put them in its output.
            snapshot.metadata = {
                'arguments': {key: value for key, value in build_args.items() if key != 'multimodal_outputs'},
                'connector': connector,
            }
            builder = _IsolatedCFGBuilder(self.model_config.engine_output_type, routing, connector)
            job = self._cfg_output_queue.submit(
                snapshot, lambda: builder._build_omni_model_runner_output_from_snapshot(**build_args)
            )
        except BaseException as error:
            # A rejected job has no queue owner. Retain its storage before
            # waiting, so a failed DMA fence cannot drop the last references.
            self._cfg_failed_snapshots.append(snapshot)
            try:
                snapshot.wait()
            except BaseException as fence_error:
                error.add_note(f'Rejected CFG payload drain also failed: {fence_error!r}')
                raise error from fence_error
            self._cfg_failed_snapshots.remove(snapshot)
            raise
        self._cfg_capture.append(job)
        return OmniModelRunnerOutput(
            req_ids=list(build_args['req_ids_output_copy']),
            req_id_to_index=dict(build_args['req_id_to_index_output_copy']),
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=None,
        )

    def _capture_cfg_payload(self, payload):
        snapshot = OutputSnapshot(None, [], device=self.device, pinned=is_pin_memory_available())
        stream = self._get_or_create_omni_payload_copy_stream()
        producer = torch.cuda.current_stream()
        try:
            # Copy non-tensor metadata as well: native cloning alone shares numpy
            # arrays and arbitrary mutable leaves. CUDA tensors clone on producer.
            cloned = _clone_cuda_tensor_payload(_owned_metadata(payload), snapshot.sources)
            snapshot.payload = cloned
            if snapshot.sources:
                snapshot.ready = torch.cuda.Event()
                with torch.cuda.stream(stream):
                    stream.wait_stream(producer)
                    snapshot.payload = _copy_owned_to_cpu(cloned, snapshot)
                    snapshot.ready.record(stream)
            return snapshot
        except BaseException as error:
            # Preserve partially enqueued copies and clone references on a failed
            # fence. Such a worker requires process teardown, never reuse.
            self._cfg_failed_snapshots.append(snapshot)
            # Cloning may fail before wait_stream establishes a producer/copy
            # dependency. Quiesce both streams independently before releasing any
            # partial storage, even when one of their fences fails.
            fence_error = None
            for name, pending_stream in (('producer', producer), ('copy', stream)):
                try:
                    pending_stream.synchronize()
                except BaseException as drain_error:
                    error.add_note(f'Partial CFG {name} drain also failed: {drain_error!r}')
                    if fence_error is None:
                        fence_error = drain_error
            if fence_error is not None:
                raise error from fence_error
            snapshot.sources.clear()
            self._cfg_failed_snapshots.remove(snapshot)
            raise

    def shutdown(self):
        # Close inference admission before waiting for an active call. That call
        # must finish attaching its native sampled-token copy to the captured job
        # before queue teardown can decide which resources are safe to retire.
        self._cfg_stopping = True
        with self._cfg_lifecycle_lock:
            if self._cfg_shutdown_complete:
                if self._cfg_shutdown_error is not None:
                    raise self._cfg_shutdown_error
                return
            # Builders and native result consumers never acquire this lifecycle
            # lock, so joining/draining them cannot wait on the inference owner.
            queue = getattr(self, '_cfg_output_queue', None)
            output_error = self._cfg_shutdown_error
            if queue is not None:
                try:
                    queue.shutdown()
                except BaseException as error:
                    if queue.pending[0] or getattr(self, '_cfg_failed_snapshots', []):
                        # Unknown copy lifetime: leave model/streams/connectors alive
                        # for the supervised process teardown, never recycle them.
                        raise
                    output_error = error
                # A consumer may already have observed the failure and safely
                # drained/retired its job. Successful draining cannot erase that
                # failure or let runner teardown report a healthy shutdown.
                failure = queue.failure
                if output_error is None and failure is not None:
                    output_error = RuntimeError('CFG output shutdown failed; executor teardown required')
                    output_error.__cause__ = failure
            if getattr(self, '_cfg_failed_snapshots', []):
                raise RuntimeError('CFG copy quiescence is unknown; process teardown required')
            # Keep the first output error even if base cleanup needs another
            # attempt after a partial failure. A drained queue cannot replay it.
            self._cfg_shutdown_error = output_error
            try:
                super().shutdown()
            except BaseException as cleanup_error:
                if output_error is None:
                    raise
                output_error.add_note(f'Base runner shutdown also failed: {cleanup_error!r}')
                raise output_error
            self._cfg_shutdown_error = output_error
            self._cfg_shutdown_complete = True
            if output_error is not None:
                raise output_error

    def _check_inference_admission(self):
        if self._cfg_stopping:
            raise RuntimeError('CFG output worker is shutting down')


class EasyMagpieCFGDeferredWorker(EasyMagpieGPUARWorker):
    """Explicit opt-in; existing worker classes never activate deferred output."""

    runner_cls = EasyMagpieCFGDeferredModelRunner


@dataclass
class _OutputJob:
    sequence: int
    snapshot: OutputSnapshot
    nbytes: int
    future: Any = None
    native: Any = None
    result: Any = None


class _IsolatedCFGBuilder(GPUARModelRunner):
    """Minimal immutable dependencies for reachable native builder helpers."""

    def __init__(self, output_type, routing, connector):
        self.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type=output_type))
        self.model = SimpleNamespace(omni_pooler_payload_include_hidden=False)
        self.omni_prefix_cache = None
        self.supports_mm_inputs = False
        self._async_chunk = True
        self._routing = frozenset(routing)
        self._connector = connector

    def _request_needs_downstream_stage_payload(self, req_id):
        return req_id in self._routing

    def _stage_deferred_prefix_cache_mm_outputs(self, **kwargs):
        return None

    def _should_accumulate_full_payload_output(self):
        return False

    def _should_return_omni_routed_experts(self):
        return False

    def get_omni_connector_output(self):
        return self._connector


def _owned_metadata(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        return {k: _owned_metadata(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_owned_metadata(v) for v in value)
    return deepcopy(value)


def _owned_cpu_value(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError('Unexpected device tensor in CFG output metadata')
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _owned_cpu_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_owned_cpu_value(v) for v in value)
    return deepcopy(value)


def _payload_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_payload_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_payload_bytes(v) for v in value)
    if hasattr(value, '__dict__'):
        return _payload_bytes(vars(value))
    return 0


def _copy_owned_to_cpu(value, snapshot):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cuda':
            return value
        cpu = torch.empty_like(value, device='cpu', pin_memory=snapshot.pinned)
        # Retain each allocation before issuing its DMA, including partial trees
        # whose recursive construction can fail before returning a root payload.
        snapshot.destinations.append(cpu)
        cpu.copy_(value, non_blocking=True)
        return cpu
    if isinstance(value, dict):
        return {k: _copy_owned_to_cpu(v, snapshot) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_copy_owned_to_cpu(v, snapshot) for v in value)
    return value
