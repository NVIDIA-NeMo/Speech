# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned CFG payloads and bounded output jobs, using the pinned native builder."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from easymagpie_vllm_omni.cfg_deferred_output import (
    DeferredCFGResult,
    EasyMagpieCFGDeferredModelRunner,
    OrderedOutputQueue,
    OutputSnapshot,
    _IsolatedCFGBuilder,
    _owned_cpu_value,
)
from easymagpie_vllm_omni.cfg_pairing import CFGPair
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


def native_result():
    calls = []
    native = SimpleNamespace(async_copy_ready_event=SimpleNamespace(synchronize=lambda: calls.append('fence')))

    def finalize():
        calls.append('finalize')
        return native._model_runner_output

    native.get_output = finalize
    return native, calls


def snapshot(value=1):
    return OutputSnapshot(torch.tensor([value]), [])


def test_blocked_head_allows_second_job_but_serializes_builds_and_bounds_unconsumed():
    queue = OrderedOutputQueue()
    entered, release = Event(), Event()
    calls = []

    def build_a():
        calls.append('a-start')
        entered.set()
        assert release.wait(10)
        calls.append('a-end')
        return torch.tensor([1])

    try:
        a = queue.submit(snapshot(), build_a)
        assert entered.wait(10)
        b = queue.submit(snapshot(2), lambda: calls.append('b') or torch.tensor([2]))
        assert queue.pending[0] == 2
        assert not b.future.done() and calls == ['a-start']
        with pytest.raises(RuntimeError, match='pending-job budget'):
            queue.submit(snapshot(3), lambda: None)
        release.set()
        assert a.future.result(timeout=10).tolist() == [1]
        assert b.future.result(timeout=10).tolist() == [2]
        assert calls == ['a-start', 'a-end', 'b']
        assert queue.pending[0] == 2  # Built outputs still occupy the budget.
        assert queue.peak_jobs == 2 and queue.peak_bytes > 0
        for job in (a, b):
            native, seen = native_result()
            result = DeferredCFGResult(queue, job, native)
            assert result.get_output().numel() == 1 and seen == ['finalize']
            with pytest.raises(RuntimeError, match='already been consumed'):
                result.get_output()
        assert queue.pending == (0, 0)
    finally:
        release.set()
        queue.shutdown()


def test_cancelled_publication_does_not_release_pending_copy():
    entered, release = Event(), Event()
    queue = OrderedOutputQueue()
    source = snapshot()

    def wait():
        entered.set()
        assert release.wait(10)

    source.wait = wait
    try:
        job = queue.submit(source, lambda: torch.tensor([4]))
        assert entered.wait(10)
        native, calls = native_result()
        result = DeferredCFGResult(queue, job, native)
        # Native cancellation suppresses publication, not worker execution.
        with ThreadPoolExecutor(max_workers=1) as pool:
            drain = pool.submit(result.drain)
            assert not drain.done() and queue.pending[0] == 1
            release.set()
            drain.result(timeout=10)
        assert queue.pending == (0, 0) and calls == ['finalize']
    finally:
        release.set()
        queue.shutdown()


def test_original_builder_failure_latches_and_native_copy_still_drains():
    queue = OrderedOutputQueue()
    error = ValueError('build exploded')

    def fail():
        raise error

    job = queue.submit(snapshot(), fail)
    native, calls = native_result()
    result = DeferredCFGResult(queue, job, native)
    with pytest.raises(ValueError) as caught:
        result.get_output()
    assert caught.value is error and calls == ['fence']
    with pytest.raises(RuntimeError, match='recreate the engine') as caught:
        queue.check_health()
    assert caught.value.__cause__ is error
    queue.shutdown()
    assert queue.pending == (0, 0)


def test_shutdown_waits_for_active_copy_and_drains_built_unconsumed():
    queue = OrderedOutputQueue()
    entered, release = Event(), Event()

    def build():
        entered.set()
        assert release.wait(10)
        return torch.tensor([2])

    job = queue.submit(snapshot(), build)
    native, calls = native_result()
    DeferredCFGResult(queue, job, native)
    assert entered.wait(10)
    with ThreadPoolExecutor(max_workers=1) as pool:
        stopped = pool.submit(queue.shutdown)
        assert not stopped.done()
        release.set()
        stopped.result(timeout=10)
    assert calls == ['finalize'] and queue.pending == (0, 0)
    queue.shutdown()
    with pytest.raises(RuntimeError, match='shutting down'):
        queue.submit(snapshot(), lambda: None)


def test_snapshot_bytes_are_bounded_before_enqueue():
    queue = OrderedOutputQueue(max_bytes=1)
    try:
        with pytest.raises(RuntimeError, match='snapshot-byte budget'):
            queue.submit(snapshot(), lambda: None)
        assert queue.pending == (0, 0)
    finally:
        queue.shutdown()


def build_arguments():
    private = CFGPair.create('a').unconditional_id
    return dict(
        scheduler_output=SimpleNamespace(total_num_scheduled_tokens=3),
        hidden_states=torch.empty(3, 0),
        staged_hidden_states_cpu=None,
        multimodal_outputs={'codes': {'audio': torch.tensor([[1, 2], [3, 4], [9, 9]])}},
        req_ids_output_copy=['a', private],
        req_id_to_index_output_copy={'a': 0, private: 1},
        valid_sampled_token_ids=[[1], [1]],
        logprobs_lists=None,
        prompt_logprobs_dict={},
        num_nans_in_logits={},
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        kv_extracted_req_ids=[],
        num_scheduled_tokens_np=np.array([2, 1]),
        query_start_loc_cpu=np.array([0, 2, 3]),
        postprocess_already_applied=True,
    )


def test_isolated_native_builder_requires_no_live_routing_feedback_or_connector(monkeypatch):
    args = build_arguments()
    notification = SimpleNamespace(request_metadata={'a': {'count': [1]}})
    saved = _owned_cpu_value(args)
    builder = _IsolatedCFGBuilder('latent', ('a',), _owned_cpu_value(notification))

    def forbidden(*args, **kwargs):
        raise AssertionError('live inference state consulted')

    monkeypatch.setattr(GPUARModelRunner, '_process_additional_information_updates', forbidden)
    args['multimodal_outputs']['codes']['audio'].fill_(7)
    args['num_scheduled_tokens_np'].fill(0)
    args['req_ids_output_copy'].clear()
    notification.request_metadata['a']['count'][0] = 9
    output = builder._build_omni_model_runner_output_from_snapshot(**saved)
    assert output.inter_stage_outputs[0]['codes.audio'].tolist() == [[1, 2], [3, 4]]
    assert output.inter_stage_outputs[1] is None
    assert 'hidden' not in output.inter_stage_outputs[0]
    assert output.omni_connector_output.request_metadata == {'a': {'count': [1]}}
    assert not hasattr(builder, 'requests') and not hasattr(builder, 'model_intermediate_buffer')


def test_capture_commits_once_and_keeps_output_after_request_removal(monkeypatch):
    runner = _deferred_runner()
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    runner._cfg_capture = []
    runner.model_config = SimpleNamespace(engine_output_type='latent')
    calls = []
    monkeypatch.setattr(runner, '_process_additional_information_updates', lambda *a, **k: calls.append('feedback'))
    monkeypatch.setattr(runner, '_request_needs_downstream_stage_payload', lambda rid: rid == 'a')
    monkeypatch.setattr(runner, 'get_omni_connector_output', lambda: calls.append('connector') or None)
    monkeypatch.setattr(runner, '_capture_cfg_payload', lambda p: OutputSnapshot(_owned_cpu_value(p), []))
    args = build_arguments()
    try:
        placeholder = runner._build_omni_model_runner_output_from_snapshot(**args)
        assert calls == ['feedback', 'connector']
        assert placeholder.inter_stage_outputs is None
        args['multimodal_outputs']['codes']['audio'].fill_(8)
        monkeypatch.setattr(runner, '_request_needs_downstream_stage_payload', lambda rid: pytest.fail('live routing'))
        job = runner._cfg_capture[0]
        native, seen = native_result()
        output = DeferredCFGResult(queue, job, native).get_output()
        assert output.inter_stage_outputs[0]['codes.audio'].tolist() == [[1, 2], [3, 4]]
        assert calls == ['feedback', 'connector'] and seen == ['finalize']
    finally:
        queue.shutdown()


@pytest.mark.parametrize(
    'field,value',
    [
        ('use_async_scheduling', False),
        ('omni_prefix_cache', object()),
        ('speculative_config', object()),
    ],
)
def test_deferred_worker_rejects_unsupported_modes(field, value):
    runner = _deferred_runner()
    runner.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            scheduler_cls='easymagpie_vllm_omni.cfg_async_scheduler.EasyMagpieCFGAsyncScheduler'
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        kv_transfer_config=None,
    )
    runner.model_config = SimpleNamespace(
        enforce_eager=True, enable_return_routed_experts=False, engine_output_type='latent'
    )
    runner.use_async_scheduling = runner._async_chunk = True
    runner.omni_prefix_cache = runner.speculative_config = None
    setattr(runner, field, value)
    with pytest.raises(ValueError):
        runner._validate_deferred_profile()


def test_actual_cuda_snapshot_survives_producer_overwrite_and_metadata_mutation():
    runner = _deferred_runner()
    runner.device = torch.device('cuda:0')
    runner._cfg_failed_snapshots = []
    source = torch.arange(48, device='cuda').reshape(6, 8)[:, ::2]
    cpu = torch.tensor([5])
    array = np.array([6])
    expected = source.cpu().clone()
    captured = runner._capture_cfg_payload({'codes': source, 'nested': [cpu, array]})
    assert captured.sources and captured.ready is not None
    assert captured.payload['codes'].device.type == 'cpu'
    assert captured.payload['codes'].is_pinned() == captured.pinned
    source.fill_(99)
    cpu.fill_(88)
    array.fill(77)
    captured.wait()
    assert torch.equal(captured.payload['codes'], expected)
    assert captured.payload['nested'][0].tolist() == [5]
    assert captured.payload['nested'][1].tolist() == [6]
    assert not captured.sources and captured.ready is None
    captured.wait()


def test_actual_cuda_cancel_drain_retains_pending_copy_and_rejects_capacity_reuse():
    runner = _deferred_runner()
    runner.device = torch.device('cuda:0')
    runner._cfg_failed_snapshots = []
    source = torch.arange(4096, device='cuda')
    expected = source.cpu().clone()
    # Hold the actual copy stream behind finite device work. No mock fence can
    # establish whether the real destination or device clones are safe to retire.
    stream = runner._get_or_create_omni_payload_copy_stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
    captured = runner._capture_cfg_payload({'codes': source})
    ready = captured.ready
    assert ready is not None and not ready.query()
    assert captured.sources and captured.pinned and captured.payload['codes'].is_pinned()
    source.fill_(99)
    queue = OrderedOutputQueue(max_jobs=1)
    try:
        job = queue.submit(captured, lambda: captured.payload['codes'])
        native, calls = native_result()
        result = DeferredCFGResult(queue, job, native)
        with ThreadPoolExecutor(max_workers=1) as pool:
            drain = pool.submit(result.drain)
            assert not drain.done() and not ready.query()
            assert queue.pending[0] == 1 and captured.sources
            with pytest.raises(RuntimeError, match='pending-job budget'):
                queue.submit(snapshot(), lambda: None)
            drain.result(timeout=20)
        assert ready.query() and not captured.sources and captured.ready is None
        assert torch.equal(captured.payload['codes'], expected)
        assert queue.pending == (0, 0) and calls == ['finalize']
        replacement = queue.submit(snapshot(), lambda: torch.tensor([8]))
        native, _ = native_result()
        assert DeferredCFGResult(queue, replacement, native).get_output().tolist() == [8]
    finally:
        queue.shutdown()


def test_actual_cuda_partial_copy_failure_quiesces_before_releasing_clones(monkeypatch):
    import easymagpie_vllm_omni.cfg_deferred_output as deferred

    runner = _deferred_runner()
    runner.device = torch.device('cuda:0')
    runner._cfg_failed_snapshots = []
    source = torch.arange(4096, device='cuda')
    expected = source.cpu().clone()
    stream = runner._get_or_create_omni_payload_copy_stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
    original = deferred._copy_owned_to_cpu
    copies = []
    error = ValueError('Controlled failure after real CUDA copy enqueue')

    def copy_then_fail(value, captured):
        copied = original(value, captured)
        if isinstance(value, torch.Tensor) and value.is_cuda:
            fence = torch.cuda.Event()
            fence.record(torch.cuda.current_stream())
            assert not fence.query() and captured.sources
            copies.append((captured, copied, fence))
            raise error
        return copied

    monkeypatch.setattr(deferred, '_copy_owned_to_cpu', copy_then_fail)
    with pytest.raises(ValueError) as caught:
        runner._capture_cfg_payload({'codes': source})
    assert caught.value is error and len(copies) == 1
    captured, destination, fence = copies[0]
    assert fence.query() and not captured.sources and not runner._cfg_failed_snapshots
    assert destination.is_pinned() and torch.equal(destination, expected)


@pytest.mark.parametrize('phase', ['copy', 'build', 'built'])
def test_actual_cuda_shutdown_drains_queued_jobs_and_native_samples_before_base_destruction(monkeypatch, phase):
    from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    runner = _deferred_runner()
    runner.device = torch.device('cuda:0')
    runner._cfg_failed_snapshots = []
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    entered, release = Event(), Event()
    destroyed, captured, natives = [], [], []
    sampled_tokens = [torch.tensor([[index]], device='cuda') for index in range(2)]
    sampled_streams = [torch.cuda.Stream() for _ in range(2)]
    source = torch.arange(4096, device='cuda')
    expected = source.cpu().clone()
    stream = runner._get_or_create_omni_payload_copy_stream()
    if phase == 'copy':
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000_000)

    def build(item, first):
        if first:
            entered.set()
            if phase == 'build':
                assert release.wait(10)
        return SimpleNamespace(payload=item.payload)

    def destroy(base):
        assert queue.pending == (0, 0)
        assert all(not item.sources and item.ready is None for item in captured)
        assert all(not hasattr(native, '_sampled_token_ids') for native in natives)
        destroyed.append(True)

    monkeypatch.setattr(EasyMagpieGPUARModelRunner, 'shutdown', destroy)
    try:
        for index in range(2):
            item = runner._capture_cfg_payload({'codes': source})
            captured.append(item)
            if phase != 'copy':
                item.wait()
            job = queue.submit(item, lambda item=item, first=index == 0: build(item, first))
            native = AsyncGPUModelRunnerOutput(
                model_runner_output=SimpleNamespace(),
                sampled_token_ids=sampled_tokens[index],
                logprobs_tensors=None,
                invalid_req_indices=[],
                async_output_copy_stream=sampled_streams[index],
                vocab_size=2,
            )
            natives.append(native)
            DeferredCFGResult(queue, job, native)
            if phase == 'built':
                job.future.result(timeout=10)
        assert queue.pending[0] == 2
        if phase == 'copy':
            assert captured[0].ready is not None and not captured[0].ready.query() and captured[0].sources
        else:
            assert entered.wait(10)
        with ThreadPoolExecutor(max_workers=1) as pool:
            stopped = pool.submit(runner.shutdown)
            try:
                if phase in ('copy', 'build'):
                    assert not stopped.done() and not destroyed
            finally:
                release.set()
            stopped.result(timeout=20)
        assert destroyed == [True] and runner._cfg_shutdown_complete
        for index, (item, native) in enumerate(zip(captured, natives)):
            assert torch.equal(item.payload['codes'], expected)
            assert native._model_runner_output.sampled_token_ids == [[index]]
        runner.shutdown()
        assert destroyed == [True]
    finally:
        release.set()
        queue.shutdown()


@pytest.mark.parametrize('boundary', ['builder', 'finalization'])
def test_actual_cuda_failed_output_drains_both_fences_before_base_destruction(monkeypatch, boundary):
    from easymagpie_vllm_omni.runner import EasyMagpieGPUARModelRunner
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    runner = _deferred_runner()
    runner.device = torch.device('cuda:0')
    runner._cfg_failed_snapshots = []
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    source = torch.arange(4096, device='cuda')
    expected = source.cpu().clone()
    stream = runner._get_or_create_omni_payload_copy_stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
    captured = runner._capture_cfg_payload({'codes': source})
    payload_fence = captured.ready
    assert payload_fence is not None and not payload_fence.query()
    error = ValueError(f'Controlled real CUDA {boundary} failure')
    builds, destroyed = [], []

    def build():
        builds.append(True)
        if boundary == 'builder':
            raise error
        return SimpleNamespace(payload=captured.payload)

    job = queue.submit(captured, build)
    native = AsyncGPUModelRunnerOutput(
        model_runner_output=SimpleNamespace(),
        sampled_token_ids=torch.tensor([[1]], device='cuda'),
        logprobs_tensors=None,
        invalid_req_indices=[],
        async_output_copy_stream=torch.cuda.Stream(),
        vocab_size=2,
    )
    sampled_fence = native.async_copy_ready_event
    if boundary == 'finalization':
        original = native.get_output

        def finalize_then_fail():
            original()
            raise error

        native.get_output = finalize_then_fail
    DeferredCFGResult(queue, job, native)

    def destroy(base):
        assert payload_fence.query() and sampled_fence.query()
        assert queue.pending == (0, 0) and job.snapshot is None
        assert not captured.sources and captured.ready is None
        assert torch.equal(captured.payload['codes'], expected)
        destroyed.append(True)

    monkeypatch.setattr(EasyMagpieGPUARModelRunner, 'shutdown', destroy)
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match='shutdown failed') as stopped:
                runner.shutdown()
            assert stopped.value.__cause__ is error
        assert destroyed == [True] and builds == [True] and runner._cfg_shutdown_complete
    finally:
        queue.shutdown()


def test_native_sampling_hook_returns_owned_deferred_result(monkeypatch):
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    runner = _deferred_runner()
    runner.execute_model_state = object()
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    runner.model_config = SimpleNamespace(engine_output_type='latent')
    runner.model = SimpleNamespace(strict_worker_feedback=True)
    monkeypatch.setattr(runner, '_process_additional_information_updates', lambda *a, **k: None)
    monkeypatch.setattr(runner, '_request_needs_downstream_stage_payload', lambda rid: rid == 'a')
    monkeypatch.setattr(runner, 'get_omni_connector_output', lambda: None)
    monkeypatch.setattr(runner, '_capture_cfg_payload', lambda p: OutputSnapshot(_owned_cpu_value(p), []))
    native = object.__new__(AsyncGPUModelRunnerOutput)
    seen = []
    native.get_output = lambda: seen.append(native._model_runner_output) or native._model_runner_output
    native.async_copy_ready_event = SimpleNamespace(synchronize=lambda: None)

    def sample(base, *args, **kwargs):
        native._model_runner_output = base._build_omni_model_runner_output_from_snapshot(**build_arguments())
        assert native._model_runner_output.inter_stage_outputs is None
        return native

    monkeypatch.setattr(GPUARModelRunner, 'sample_tokens', sample)
    try:
        result = runner.sample_tokens(None)
        assert isinstance(result, DeferredCFGResult) and runner._cfg_capture is None
        output = result.get_output()
        assert seen == [output] and output.inter_stage_outputs[0]['codes.audio'].shape == (2, 2)
        runner.execute_model_state = None
        monkeypatch.setattr(GPUARModelRunner, 'sample_tokens', lambda *a, **k: 'empty-notification')
        assert runner.sample_tokens(None) == 'empty-notification'
    finally:
        queue.shutdown()


def test_explicit_deferred_deploy_retains_inline_profile():
    from conftest import EASYMAGPIE_ROOT
    from easymagpie_vllm_omni.pipeline import EASYMAGPIE_CFG_ASYNC_EAGER_PIPELINE
    from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

    workers = []
    for name in ('easymagpie_cfg_async_eager', 'easymagpie_cfg_async_deferred_eager'):
        deploy = load_deploy_config(EASYMAGPIE_ROOT / 'deploy' / f'{name}.yaml')
        acoustic, codec = merge_pipeline_deploy(EASYMAGPIE_CFG_ASYNC_EAGER_PIPELINE, deploy)
        assert acoustic.scheduler_cls.endswith('cfg_async_scheduler.EasyMagpieCFGAsyncScheduler')
        assert acoustic.yaml_engine_args['async_scheduling']
        assert not codec.yaml_engine_args['async_scheduling']
        workers.append(acoustic.yaml_engine_args['worker_cls'])
    assert workers == [
        'easymagpie_vllm_omni.runner.EasyMagpieGPUARWorker',
        'easymagpie_vllm_omni.cfg_deferred_output.EasyMagpieCFGDeferredWorker',
    ]


def test_native_engine_submits_b_while_a_builder_is_blocked(monkeypatch):
    from test_cfg_concurrency import _batch_output
    from test_cfg_engine_queue import engine
    from test_cfg_scheduler import _request
    from vllm.v1.executor.uniproc_executor import AsyncOutputFuture

    core = engine(monkeypatch)
    core.scheduler.add_request(_request('a'))
    queue = OrderedOutputQueue()
    entered, release = Event(), Event()
    results = []
    b_submitted = Event()

    def sample(grammar, non_block=False):
        assert non_block
        step = core.model_executor.steps[-1]
        number = len(results)
        output = _batch_output(step)

        def build():
            if number == 0:
                entered.set()
                assert release.wait(10)
            return output

        job = queue.submit(snapshot(), build)
        native, calls = native_result()
        future = AsyncOutputFuture(DeferredCFGResult(queue, job, native), single_value=True)
        results.append(future)
        if number == 1:
            b_submitted.set()
        return future

    monkeypatch.setattr(core.model_executor, 'sample_tokens', sample)
    try:
        assert core.step_fn() == (None, True)
        assert entered.wait(10)
        with ThreadPoolExecutor(max_workers=1) as pool:
            progress = pool.submit(core.step_fn)
            assert b_submitted.wait(10)
            assert not progress.done() and queue.pending[0] == 2
            assert len(core.scheduler._cfg_steps) == 2
            release.set()
            output, executed = progress.result(timeout=10)
        assert executed and len(core.model_executor.steps) == 2 and len(results) == 2
        assert queue.peak_jobs == 2
        assert len(core.scheduler._cfg_steps) == 1
        assert [r for r in core.model_executor.trace if r[0] == 'update'] == [('update', 0)]
        assert [o.request_id for o in output[0].outputs] == ['a']
        core.abort_requests(['a'])
        core.step_fn()
        assert queue.pending == (0, 0)
    finally:
        release.set()
        queue.shutdown()


def test_connector_collection_preserves_drains_levels_and_late_notifications():
    from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

    runner, pending = _connector_runner()
    first = _owned_cpu_value(runner.get_omni_connector_output())
    assert first.chunk_ready_req_ids == {'a'} and first.kv_sent_req_ids == ['sent']
    assert first.stage_recv_req_ids == {'received'} and first.has_pending_kv_work
    runner._chunk_finished_req_ids.add('late')
    runner._local_request_metadata['late'] = {'nested': [2]}
    second = _owned_cpu_value(runner.get_omni_connector_output())
    assert second.chunk_finished_req_ids == {'late'} and not second.chunk_ready_req_ids
    assert second.has_pending_kv_work  # Level, not a drained notification.
    assert first.request_metadata == {'a': {'nested': [1]}}
    pending[0] = False
    assert not OmniConnectorModelRunnerMixin._connector_output_has_signals(runner.get_omni_connector_output())


def test_finalization_failure_latches_and_shutdown_does_not_retry_builder():
    queue = OrderedOutputQueue()
    calls = []
    job = queue.submit(snapshot(), lambda: calls.append('build') or torch.tensor([1]))
    native, seen = native_result()
    error = ValueError('native finalize failed')

    def fail():
        raise error

    native.get_output = fail
    result = DeferredCFGResult(queue, job, native)
    with pytest.raises(ValueError) as caught:
        result.get_output()
    assert caught.value is error and seen == ['fence']
    queue.shutdown()
    assert calls == ['build'] and queue.pending == (0, 0)


def test_failed_copy_fence_prevents_base_runner_resource_destruction(monkeypatch):
    runner = _deferred_runner()
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    error = OSError('device fence failed')
    source = snapshot()

    def fail():
        raise error

    source.wait = fail
    job = queue.submit(source, lambda: pytest.fail('builder after failed copy'))
    native, seen = native_result()
    native.async_copy_ready_event.synchronize = fail
    DeferredCFGResult(queue, job, native)
    destroyed = []
    monkeypatch.setattr(GPUARModelRunner, 'shutdown', lambda *a: destroyed.append(True))
    with pytest.raises(RuntimeError, match='shutdown failed'):
        runner.shutdown()
    assert not destroyed and queue.pending[0] == 1
    assert job.snapshot is source  # No claim of safe storage retirement.


def test_failed_payload_copy_is_not_retired_by_a_healthy_sampled_token_fence(monkeypatch):
    runner = _deferred_runner()
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    source = snapshot()

    def fail_copy():
        raise OSError('payload DMA failed')

    source.wait = fail_copy
    job = queue.submit(source, lambda: pytest.fail('builder after failed DMA'))
    native, calls = native_result()
    result = DeferredCFGResult(queue, job, native)
    with pytest.raises(OSError, match='payload DMA'):
        result.get_output()
    destroyed = []
    monkeypatch.setattr(GPUARModelRunner, 'shutdown', lambda *a: destroyed.append(True))
    with pytest.raises(RuntimeError, match='shutdown failed'):
        runner.shutdown()
    assert not destroyed and queue.pending[0] == 1
    assert job.snapshot is source


@pytest.mark.parametrize('fence_fails', [False, True])
def test_rejected_snapshot_retains_storage_until_copy_quiescence(monkeypatch, fence_fails):
    runner = _deferred_runner()
    queue = runner._cfg_output_queue = OrderedOutputQueue(max_bytes=1)
    runner._cfg_capture = []
    runner._cfg_failed_snapshots = []
    runner.model_config = SimpleNamespace(engine_output_type='latent')
    source = snapshot()
    fence_error = OSError('rejected snapshot DMA failed')
    waits = []

    def wait():
        waits.append(True)
        if fence_fails:
            raise fence_error

    source.wait = wait
    monkeypatch.setattr(runner, '_process_additional_information_updates', lambda *a, **k: None)
    monkeypatch.setattr(runner, '_request_needs_downstream_stage_payload', lambda rid: rid == 'a')
    monkeypatch.setattr(runner, 'get_omni_connector_output', lambda: None)
    monkeypatch.setattr(runner, '_capture_cfg_payload', lambda payload: source)
    destroyed = []
    monkeypatch.setattr(GPUARModelRunner, 'shutdown', lambda *a: destroyed.append(True))
    try:
        with pytest.raises(RuntimeError, match='snapshot-byte budget') as caught:
            runner._build_omni_model_runner_output_from_snapshot(**build_arguments())
        assert waits == [True] and queue.pending == (0, 0)
        if fence_fails:
            assert caught.value.__cause__ is fence_error
            assert runner._cfg_failed_snapshots == [source]
            with pytest.raises(RuntimeError, match='copy quiescence is unknown'):
                runner.shutdown()
            assert not destroyed
        else:
            assert runner._cfg_failed_snapshots == []
            runner.shutdown()
            assert destroyed == [True]
    finally:
        queue.shutdown()


@pytest.mark.parametrize('owner', ['scheduler', 'connector'])
@pytest.mark.parametrize('max_bytes', [128, 1024 * 1024])
def test_capture_accounts_for_metadata_until_native_consumption(monkeypatch, owner, max_bytes):
    runner = _deferred_runner()
    queue = runner._cfg_output_queue = OrderedOutputQueue(max_bytes=max_bytes)
    runner._cfg_capture = []
    runner._cfg_failed_snapshots = []
    runner.model_config = SimpleNamespace(engine_output_type='latent')
    args = build_arguments()
    metadata = np.ones(4096, dtype=np.int64)
    connector = SimpleNamespace(request_metadata={'a': {'diagnostic': metadata}}) if owner == 'connector' else None
    if owner == 'scheduler':
        args['scheduler_output'].diagnostic = metadata
    monkeypatch.setattr(runner, '_process_additional_information_updates', lambda *a, **k: None)
    monkeypatch.setattr(runner, '_request_needs_downstream_stage_payload', lambda rid: rid == 'a')
    monkeypatch.setattr(runner, 'get_omni_connector_output', lambda: connector)
    monkeypatch.setattr(runner, '_capture_cfg_payload', lambda p: OutputSnapshot(_owned_cpu_value(p), []))
    try:
        if max_bytes < metadata.nbytes:
            with pytest.raises(RuntimeError, match='snapshot-byte budget'):
                runner._build_omni_model_runner_output_from_snapshot(**args)
            assert queue.pending == (0, 0)
        else:
            runner._build_omni_model_runner_output_from_snapshot(**args)
            job = runner._cfg_capture[0]
            assert job.nbytes >= metadata.nbytes
            job.future.result(timeout=10)
            assert queue.pending[0] == 1 and queue.pending[1] >= metadata.nbytes
            native, _ = native_result()
            DeferredCFGResult(queue, job, native).get_output()
            assert queue.pending == (0, 0)
    finally:
        queue.shutdown()


def test_executor_submission_failure_leaves_no_orphan_and_latches_original_error(monkeypatch):
    queue = OrderedOutputQueue()
    error = RuntimeError('output executor rejected submission')

    def reject(*args, **kwargs):
        raise error

    monkeypatch.setattr(queue._executor, 'submit', reject)
    try:
        with pytest.raises(RuntimeError) as caught:
            queue.submit(snapshot(), lambda: pytest.fail('rejected builder ran'))
        assert caught.value is error
        assert queue.pending == (0, 0)
        with pytest.raises(RuntimeError, match='recreate the engine') as caught:
            queue.check_health()
        assert caught.value.__cause__ is error
    finally:
        queue.shutdown()


def test_concurrent_shutdown_drains_finalize_native_output_once(monkeypatch):
    queue = OrderedOutputQueue()
    job = queue.submit(snapshot(), lambda: torch.tensor([1]))
    native, calls = native_result()
    result = DeferredCFGResult(queue, job, native)
    entered, release, second_started, duplicate = Event(), Event(), Event(), Event()
    original = result.get_output
    invocations = []

    def delayed_get():
        invocations.append(True)
        if len(invocations) > 1:
            duplicate.set()
        entered.set()
        assert release.wait(10)
        return original()

    monkeypatch.setattr(result, 'get_output', delayed_get)

    def second_drain():
        second_started.set()
        result.drain()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(result.drain)
            assert entered.wait(10)
            second = pool.submit(second_drain)
            assert second_started.wait(10)
            duplicate.wait(0.2)
            release.set()
            first.result(timeout=10)
            second.result(timeout=10)
        assert calls == ['finalize'] and len(invocations) == 1
        assert queue.pending == (0, 0)
    finally:
        release.set()
        queue.shutdown()


@pytest.mark.parametrize('representation', ['class', 'qualified_name', 'foreign_suffix'])
def test_deferred_profile_matches_exact_approved_scheduler(representation):
    from easymagpie_vllm_omni.cfg_async_scheduler import EasyMagpieCFGAsyncScheduler

    runner = _deferred_runner()
    approved = 'easymagpie_vllm_omni.cfg_async_scheduler.EasyMagpieCFGAsyncScheduler'
    selected = {
        'class': EasyMagpieCFGAsyncScheduler,
        'qualified_name': approved,
        'foreign_suffix': 'foreign.' + approved,
    }[representation]
    runner.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(scheduler_cls=selected),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        kv_transfer_config=None,
    )
    runner.model_config = SimpleNamespace(
        enforce_eager=True, enable_return_routed_experts=False, engine_output_type='latent'
    )
    runner.use_async_scheduling = runner._async_chunk = True
    runner.omni_prefix_cache = runner.speculative_config = None
    if representation == 'foreign_suffix':
        with pytest.raises(ValueError, match='explicit CFG async scheduler'):
            runner._validate_deferred_profile()
    else:
        runner._validate_deferred_profile()


@pytest.mark.parametrize('failure_phase', ['builder', 'finalization', 'unattached_builder'])
def test_shutdown_releases_quiescent_failed_jobs_before_resource_destruction(monkeypatch, failure_phase):
    runner = _deferred_runner()
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    runner._cfg_failed_snapshots = []
    error = ValueError('controlled output failure')
    trace = []

    def fail():
        trace.append(failure_phase)
        raise error

    job = queue.submit(snapshot(), fail if failure_phase != 'finalization' else lambda: torch.tensor([1]))
    native, calls = native_result()
    if failure_phase != 'unattached_builder':
        if failure_phase == 'finalization':
            native.get_output = fail
        DeferredCFGResult(queue, job, native)

    def destroy(*args):
        assert queue.pending == (0, 0)
        assert job.snapshot is None
        trace.append('destroy')

    monkeypatch.setattr(GPUARModelRunner, 'shutdown', destroy)
    with pytest.raises(RuntimeError, match='shutdown failed') as caught:
        runner.shutdown()
    assert caught.value.__cause__ is error
    assert trace == [failure_phase, 'destroy']
    assert queue.pending == (0, 0)
    assert calls == ([] if failure_phase == 'unattached_builder' else ['fence', 'fence'])


def test_deferred_result_uses_native_sampled_token_finalization():
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    queue = OrderedOutputQueue()
    native = object.__new__(AsyncGPUModelRunnerOutput)
    calls = []
    native.async_copy_ready_event = SimpleNamespace(synchronize=lambda: calls.append('sampled-copy-fence'))
    native.sampled_token_ids_cpu = torch.tensor([[7], [7]])
    native._sampled_token_ids = torch.tensor([[7], [7]])
    native._invalid_req_indices = [1]
    native._logprobs_tensors = native._logprobs_tensors_cpu = None
    native._routed_experts = native._routed_experts_cpu = None
    native.vocab_size = 32
    args = build_arguments()
    builder = _IsolatedCFGBuilder('latent', ('a',), None)
    job = queue.submit(snapshot(), lambda: builder._build_omni_model_runner_output_from_snapshot(**args))
    try:
        result = DeferredCFGResult(queue, job, native)
        output = result.get_output()
        assert output.sampled_token_ids == [[7], []]
        assert output.inter_stage_outputs[0]['codes.audio'].tolist() == [[1, 2], [3, 4]]
        assert output.inter_stage_outputs[1] is None
        assert calls == ['sampled-copy-fence']
        assert not hasattr(native, '_sampled_token_ids')
        assert queue.pending == (0, 0)
        result.drain()
        assert calls == ['sampled-copy-fence']
    finally:
        queue.shutdown()


@pytest.mark.parametrize('fence_fails', [False, True])
def test_capture_setup_failure_preserves_original_error_and_partial_storage(monkeypatch, fence_fails):
    import easymagpie_vllm_omni.cfg_deferred_output as deferred

    runner = _deferred_runner()
    runner.device = 'cpu'
    runner._cfg_failed_snapshots = []
    source = torch.tensor([1])
    setup_error = ValueError('partial payload setup failed')
    fence_error = OSError('partial payload drain failed')
    retained = []

    def clone(payload, sources):
        sources.append(source)
        raise setup_error

    def synchronize():
        retained.extend(runner._cfg_failed_snapshots)
        if fence_fails:
            raise fence_error

    monkeypatch.setattr(deferred, 'is_pin_memory_available', lambda: False)
    monkeypatch.setattr(deferred, '_clone_cuda_tensor_payload', clone)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(
        runner, '_get_or_create_omni_payload_copy_stream', lambda: SimpleNamespace(synchronize=synchronize)
    )
    with pytest.raises(ValueError) as caught:
        runner._capture_cfg_payload({'codes': source})
    assert caught.value is setup_error
    assert len(retained) == 1
    if fence_fails:
        assert caught.value.__cause__ is fence_error
        assert runner._cfg_failed_snapshots == retained
        assert retained[0].sources[0] is source
    else:
        assert runner._cfg_failed_snapshots == []
        assert retained[0].sources == []


def test_shutdown_waits_for_native_result_attachment_before_retiring_capture(monkeypatch):
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    runner = _deferred_runner()
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    captured, release, stopping, destroyed = Event(), Event(), Event(), Event()
    native = object.__new__(AsyncGPUModelRunnerOutput)
    finalized = []
    native.get_output = lambda: finalized.append(True) or native._model_runner_output
    native.async_copy_ready_event = SimpleNamespace(synchronize=lambda: None)

    def sample(base, *args, **kwargs):
        job = queue.submit(snapshot(), lambda: torch.tensor([7]))
        base._cfg_capture.append(job)
        captured.set()
        assert release.wait(10)
        assert not destroyed.is_set(), 'Runner resources destroyed before native result attachment'
        return native

    def stop():
        stopping.set()
        runner.shutdown()

    monkeypatch.setattr(GPUARModelRunner, 'sample_tokens', sample)
    monkeypatch.setattr(GPUARModelRunner, 'shutdown', lambda *a: destroyed.set())
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            sampling = pool.submit(runner.sample_tokens, None)
            assert captured.wait(10)
            shutdown = pool.submit(stop)
            assert stopping.wait(10)
            premature = destroyed.wait(0.2)
            release.set()
            result = sampling.result(timeout=10)
            shutdown.result(timeout=10)
        assert not premature
        assert finalized == [True] and queue.pending == (0, 0)
        assert result._consumed and destroyed.is_set()
    finally:
        release.set()
        queue.shutdown()


@pytest.mark.parametrize('operation', ['execute_model', 'sample_tokens'])
def test_shutdown_waits_for_active_inference_and_rejects_late_calls(monkeypatch, operation):
    runner = _deferred_runner()
    runner._cfg_output_queue = OrderedOutputQueue()
    entered, release, stopping, destroyed = Event(), Event(), Event(), Event()
    calls = []

    def inference(*args, **kwargs):
        calls.append(operation)
        entered.set()
        assert release.wait(10)
        assert not destroyed.is_set(), 'Runner resources destroyed during inference'
        return None

    def stop():
        stopping.set()
        runner.shutdown()

    monkeypatch.setattr(GPUARModelRunner, operation, inference)
    monkeypatch.setattr(GPUARModelRunner, 'shutdown', lambda *a: destroyed.set())
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            active = pool.submit(getattr(runner, operation), None)
            assert entered.wait(10)
            shutdown = pool.submit(stop)
            assert stopping.wait(10)
            premature = destroyed.wait(0.2)
            late = pool.submit(getattr(runner, operation), None)
            release.set()
            active.result(timeout=10)
            with pytest.raises(RuntimeError, match='shutting down'):
                late.result(timeout=10)
            shutdown.result(timeout=10)
        assert not premature
        with pytest.raises(RuntimeError, match='shutting down'):
            getattr(runner, operation)(None)
        assert calls == [operation]
    finally:
        release.set()
        runner._cfg_output_queue.shutdown()


@pytest.mark.parametrize('failure', [False, True])
def test_repeated_runner_shutdown_destroys_resources_once_and_preserves_failure(monkeypatch, failure):
    runner = _deferred_runner()
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    error = ValueError('builder failed during shutdown')
    destroyed = []

    def build():
        if failure:
            raise error
        return torch.tensor([1])

    job = queue.submit(snapshot(), build)
    native, calls = native_result()
    DeferredCFGResult(queue, job, native)
    monkeypatch.setattr(GPUARModelRunner, 'shutdown', lambda *a: destroyed.append(True))
    errors = []
    for _ in range(2):
        if failure:
            with pytest.raises(RuntimeError, match='shutdown failed') as caught:
                runner.shutdown()
            errors.append(caught.value)
        else:
            runner.shutdown()
    assert destroyed == [True]
    assert queue.pending == (0, 0)
    if failure:
        assert errors[0] is errors[1] and errors[0].__cause__ is error
    else:
        assert calls == ['finalize']


@pytest.mark.parametrize('phase', ['builder', 'finalization', 'sampling', 'retired'])
@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_runner_shutdown_preserves_previously_observed_output_failure(monkeypatch, phase, cleanup_fails):
    runner = _deferred_runner()
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    error = ValueError('output failure already observed by the consumer')
    cleanup_error = OSError('base cleanup needs another attempt')
    destroyed, builds = [], []

    def fail():
        raise error

    if phase == 'sampling':
        monkeypatch.setattr(GPUARModelRunner, 'sample_tokens', lambda *a, **k: fail())
        with pytest.raises(ValueError) as observed:
            runner.sample_tokens(None)
    else:

        def build():
            builds.append(True)
            if phase != 'finalization':
                fail()
            return torch.tensor([1])

        job = queue.submit(snapshot(), build)
        native, calls = native_result()
        if phase == 'finalization':
            native.get_output = fail
        result = DeferredCFGResult(queue, job, native)
        with pytest.raises(ValueError) as observed:
            result.get_output()
        assert calls == ['fence']
        if phase == 'retired':
            result.drain()
            assert queue.pending == (0, 0)
    assert observed.value is error

    def destroy(*args):
        assert queue.pending == (0, 0)
        destroyed.append(True)
        if cleanup_fails and len(destroyed) == 1:
            raise cleanup_error

    monkeypatch.setattr(GPUARModelRunner, 'shutdown', destroy)
    failures = []
    for _ in range(3):
        with pytest.raises(RuntimeError, match='shutdown failed') as stopped:
            runner.shutdown()
        assert stopped.value.__cause__ is error
        failures.append(stopped.value)
    assert failures[0] is failures[1] is failures[2]
    assert len(destroyed) == (2 if cleanup_fails else 1)
    assert builds == ([] if phase == 'sampling' else [True])
    if cleanup_fails:
        assert any(str(cleanup_error) in note for note in failures[0].__notes__)


@pytest.mark.parametrize('builder_fails', [False, True])
def test_incomplete_base_cleanup_can_retry_without_losing_original_output_failure(monkeypatch, builder_fails):
    runner = _deferred_runner()
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    build_error = ValueError('original builder error')
    cleanup_error = OSError('base cleanup incomplete')
    cleanup_calls = []

    def build():
        if builder_fails:
            raise build_error
        return torch.tensor([3])

    job = queue.submit(snapshot(), build)
    native, calls = native_result()
    DeferredCFGResult(queue, job, native)

    def cleanup(*args):
        cleanup_calls.append(True)
        if len(cleanup_calls) == 1:
            raise cleanup_error

    monkeypatch.setattr(GPUARModelRunner, 'shutdown', cleanup)
    with pytest.raises(RuntimeError if builder_fails else OSError) as caught:
        runner.shutdown()
    original = caught.value
    assert queue.pending == (0, 0)
    for _ in range(2):
        if builder_fails:
            with pytest.raises(RuntimeError, match='shutdown failed') as retry:
                runner.shutdown()
            assert retry.value is original and original.__cause__ is build_error
        else:
            runner.shutdown()
    assert cleanup_calls == [True, True]
    assert calls == (['fence', 'fence'] if builder_fails else ['finalize'])


@pytest.mark.parametrize('failed_stream', [None, 'producer', 'copy'])
def test_partial_clone_failure_drains_both_streams_before_releasing_storage(monkeypatch, failed_stream):
    import easymagpie_vllm_omni.cfg_deferred_output as deferred

    runner = _deferred_runner()
    runner.device = 'cpu'
    source = torch.tensor([1])
    runner._cfg_failed_snapshots = []
    setup_error = ValueError('clone failed before copy-stream dependency')
    fence_error = OSError('partial clone stream drain failed')
    calls, retained, destroyed = [], [], []

    def synchronize(name):
        calls.append(name)
        saved = runner._cfg_failed_snapshots[0]
        assert saved.sources[0] is source
        if not retained:
            retained.append(saved)
        if name == failed_stream:
            raise fence_error

    def clone(payload, sources):
        sources.append(source)
        raise setup_error

    monkeypatch.setattr(deferred, 'is_pin_memory_available', lambda: False)
    monkeypatch.setattr(deferred, '_clone_cuda_tensor_payload', clone)
    monkeypatch.setattr(
        torch.cuda, 'current_stream', lambda: SimpleNamespace(synchronize=lambda: synchronize('producer'))
    )
    monkeypatch.setattr(
        runner,
        '_get_or_create_omni_payload_copy_stream',
        lambda: SimpleNamespace(synchronize=lambda: synchronize('copy')),
    )
    monkeypatch.setattr(GPUARModelRunner, 'shutdown', lambda self: destroyed.append(True))
    with pytest.raises(ValueError) as caught:
        runner._capture_cfg_payload({'codes': source})
    assert caught.value is setup_error
    assert calls == ['producer', 'copy']
    if failed_stream is None:
        assert runner._cfg_failed_snapshots == [] and retained[0].sources == []
        runner.shutdown()
        assert destroyed == [True]
    else:
        assert caught.value.__cause__ is fence_error
        assert runner._cfg_failed_snapshots == retained
        assert retained[0].sources[0] is source
        with pytest.raises(RuntimeError, match='copy quiescence is unknown'):
            runner.shutdown()
        assert destroyed == []


@pytest.mark.parametrize('phase', ['building', 'built'])
def test_full_output_queue_preserves_native_empty_work_and_late_notifications(monkeypatch, phase):
    import vllm_omni.worker.gpu_ar_model_runner as native_runner

    runner, pending = _connector_runner()
    runner.execute_model_state = runner.kv_connector_output = None
    runner.use_async_scheduling = True
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    monkeypatch.setattr(native_runner, 'get_pp_group', lambda: SimpleNamespace(is_last_rank=True))
    entered, release = Event(), Event()

    def build():
        entered.set()
        assert release.wait(10)
        return torch.tensor([1])

    try:
        a = queue.submit(snapshot(), build)
        assert entered.wait(10)
        b = queue.submit(snapshot(2), lambda: torch.tensor([2]))
        if phase == 'built':
            release.set()
            a.future.result(timeout=10)
            b.future.result(timeout=10)
        before = queue.pending
        assert before[0] == 2
        first = runner.sample_tokens(None)
        assert first.req_ids == []
        assert first.omni_connector_output.chunk_ready_req_ids == {'a'}
        assert first.omni_connector_output.kv_sent_req_ids == ['sent']
        assert runner._cfg_capture is None and queue.pending == before
        runner._chunk_finished_req_ids.add('a')
        runner._local_request_metadata['a'] = {'terminal': True}
        terminal = runner.sample_tokens(None)
        assert terminal.omni_connector_output.chunk_finished_req_ids == {'a'}
        assert terminal.omni_connector_output.request_metadata == {'a': {'terminal': True}}
        assert terminal.omni_connector_output.chunk_ready_req_ids == set()
        assert terminal.omni_connector_output.has_pending_kv_work
        pending[0] = False
        empty = runner.sample_tokens(None)
        assert getattr(empty, 'omni_connector_output', None) is None
        assert queue.pending == before and runner._cfg_capture is None
        queue.check_health()
    finally:
        release.set()
        queue.shutdown()


def test_full_output_queue_still_rejects_payload_work_before_native_sampling(monkeypatch):
    runner = _deferred_runner()
    runner.execute_model_state = object()
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    monkeypatch.setattr(GPUARModelRunner, 'sample_tokens', lambda *a: pytest.fail('sampled with full queue'))
    try:
        for _ in range(2):
            queue.submit(snapshot(), lambda: torch.tensor([1])).future.result(timeout=10)
        with pytest.raises(RuntimeError, match='pending-job budget'):
            runner.sample_tokens(None)
        assert queue.pending[0] == 2
        queue.check_health()
    finally:
        queue.shutdown()


def test_empty_work_cannot_collect_notifications_after_output_failure(monkeypatch):
    runner, _ = _connector_runner()
    runner.execute_model_state = None
    queue = runner._cfg_output_queue = OrderedOutputQueue()
    error = ValueError('failed predecessor')
    queue.fail(error)
    monkeypatch.setattr(GPUARModelRunner, 'sample_tokens', lambda *a: pytest.fail('sampled after failure'))
    try:
        with pytest.raises(RuntimeError, match='recreate the engine') as caught:
            runner.sample_tokens(None)
        assert caught.value.__cause__ is error
        assert runner._finished_load_reqs == {'a'}
    finally:
        queue.shutdown()


def _connector_runner():
    runner = _deferred_runner()
    from threading import Lock

    runner._lock = Lock()
    runner._stage_id = 0
    runner._async_chunk = True
    runner._finished_load_reqs = {'a'}
    runner._chunk_finished_req_ids = set()
    runner._local_request_metadata = {'a': {'nested': [1]}}
    runner._local_stage_payload_cache = {}
    runner._send_side_request_payload = {}
    runner._request_ids_mapping = {}
    runner._chunk_ready_req_ids = set()
    runner._kv_sent_req_ids = ['sent']
    runner._stage_recv_req_ids = {'received'}
    runner._get_local_tp_group = lambda: None
    pending = [True]
    runner.has_pending_kv_work = lambda: pending[0]
    return runner, pending


def _deferred_runner():
    runner = object.__new__(EasyMagpieCFGDeferredModelRunner)
    runner._cfg_lifecycle_lock = RLock()
    runner._cfg_stopping = False
    runner._cfg_shutdown_complete = False
    runner._cfg_shutdown_error = None
    runner.execute_model_state = None
    return runner
