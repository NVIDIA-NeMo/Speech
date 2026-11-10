# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pinned EngineCore loop with standard controllable futures, no GPU executor."""
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from easymagpie_vllm_omni.cfg_async_scheduler import EasyMagpieCFGAsyncScheduler
from easymagpie_vllm_omni.cfg_scheduler import EasyMagpieCFGScheduler
from test_cfg_concurrency import _batch_output, _mixed_scheduler
from test_cfg_scheduler import _request
from vllm.config import VllmConfig
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import ModelRunnerOutput

_TRACE_CORES = []


@pytest.fixture(autouse=True)
def save_native_queue_trace(request):
    _TRACE_CORES.clear()
    yield
    destination = os.environ.get('P2_TRACE_DIR')
    if destination:
        out = Path(destination)
        out.mkdir(exist_ok=True)
        records = [
            dict(
                trace=core.model_executor.trace,
                queue_depth=core.batch_queue_size,
                unconsumed_records=len(getattr(core.scheduler, '_cfg_steps', ())),
                scheduled_fence=core.scheduler.sched_step_seq,
                consumed_fence=core.scheduler.processed_step_seq,
            )
            for core in _TRACE_CORES
        ]
        (out / (request.node.name + '.json')).write_text(json.dumps(records, indent=2) + '\n')


class ObservedFuture(Future):
    def __init__(self, label, trace):
        super().__init__()
        self.label, self.trace = label, trace
        self.entered = Event()

    def result(self, timeout=None):
        self.trace.append(('wait', self.label))
        self.entered.set()
        value = super().result(timeout=5 if timeout is None else timeout)
        self.trace.append(('materialized', self.label))
        return value


class ControlledExecutor:
    def __init__(self, config):
        self.trace = []
        self.steps, self.execution, self.sampling = [], [], []

    def execute_model(self, step, non_block=False):
        assert non_block
        index = len(self.steps)
        self.trace.append(('execute', index, tuple(step.num_scheduled_tokens), tuple(step.finished_req_ids)))
        self.steps.append(step)
        future = ObservedFuture(('execute', index), self.trace)
        self.execution.append(future)
        if not step.total_num_scheduled_tokens:
            future.set_result(ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[]))
        return future

    def sample_tokens(self, grammar, non_block=False):
        assert non_block
        index = len(self.steps) - 1
        self.trace.append(('sample', index))
        future = ObservedFuture(('sample', index), self.trace)
        self.sampling.append(future)
        return future


def engine(monkeypatch, async_mode=True):
    import vllm.plugins
    import vllm.v1.engine.core as module
    from vllm.multimodal import MULTIMODAL_REGISTRY

    template = _mixed_scheduler(max_num_seqs=5)
    config = template.vllm_config
    config.scheduler_config.async_scheduling = async_mode
    config.scheduler_config.scheduler_cls = (
        'easymagpie_vllm_omni.cfg_async_scheduler.EasyMagpieCFGAsyncScheduler'
        if async_mode
        else 'easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGScheduler'
    )
    config.parallel_config.data_parallel_rank_local = 1
    config.model_config.runner_type = 'generate'
    config.max_concurrent_batches = VllmConfig.max_concurrent_batches.fget(config)
    monkeypatch.setattr(vllm.plugins, 'load_general_plugins', lambda: None)
    monkeypatch.setattr(EngineCore, '_initialize_kv_caches', lambda *a: template.kv_cache_config)
    monkeypatch.setattr(
        module, 'StructuredOutputManager', lambda config: SimpleNamespace(should_advance=lambda r: False)
    )
    monkeypatch.setattr(module, 'resolve_kv_cache_block_sizes', lambda *a: (4, 4))
    monkeypatch.setattr(MULTIMODAL_REGISTRY, 'supports_multimodal_inputs', lambda *a: False)
    monkeypatch.setattr(MULTIMODAL_REGISTRY, 'engine_receiver_cache_from_config', lambda *a: None)
    core = EngineCore(config, ControlledExecutor, log_stats=False)
    _TRACE_CORES.append(core)
    core.log_error_detail = lambda *a: nullcontext()
    core.log_iteration_details = lambda *a: nullcontext()
    core._should_throttle_prefills = lambda: False
    original = core.scheduler.update_from_output

    def update(step, output):
        index = core.model_executor.steps.index(step)
        core.model_executor.trace.append(('update', index))
        return original(step, output)

    monkeypatch.setattr(core.scheduler, 'update_from_output', update)
    original_abort = core.abort_requests

    def abort(ids):
        core.model_executor.trace.append(('abort', tuple(ids)))
        return original_abort(ids)

    monkeypatch.setattr(core, 'abort_requests', abort)
    return core


def test_actual_constructor_selects_queue_and_scheduler(monkeypatch):
    core = engine(monkeypatch)
    assert type(core.scheduler) is EasyMagpieCFGAsyncScheduler
    assert core.batch_queue.maxlen == core.batch_queue_size == 2
    assert core.step_fn.__func__ is EngineCore.step_with_batch_queue
    sync = engine(monkeypatch, async_mode=False)
    assert type(sync.scheduler) is EasyMagpieCFGScheduler
    assert sync.batch_queue is None and sync.batch_queue_size == 1
    assert sync.step_fn.__func__ is EngineCore.step


def test_fifo_head_blocking_includes_popped_future(monkeypatch):
    core = engine(monkeypatch)
    core.scheduler.add_request(_request('a'))
    executor = core.model_executor
    assert core.step_fn() == (None, True)
    assert len(executor.steps) == 1 and not executor.sampling[0].done()
    with ThreadPoolExecutor(max_workers=1) as thread:
        pending = thread.submit(core.step_fn)
        assert executor.sampling[0].entered.wait(5)
        assert len(executor.steps) == 2
        assert len(core.batch_queue) == 1  # A is popped but still unconsumed.
        assert len(core.scheduler._cfg_steps) == 2
        executor.sampling[1].set_result(_batch_output(executor.steps[1]))
        assert not any(row[0] == 'update' for row in executor.trace)
        executor.sampling[0].set_result(_batch_output(executor.steps[0]))
        result, executed = pending.result(timeout=5)
    assert executed and [o.request_id for o in result[0].outputs] == ['a']
    assert [row for row in executor.trace if row[0] == 'update'] == [('update', 0)]
    result, executed = core.step_fn()  # C may be submitted after A, before B is consumed.
    assert len(executor.steps) == 3
    assert [row for row in executor.trace if row[0] == 'update'] == [('update', 0), ('update', 1)]
    assert core.scheduler.requests['a'].num_output_tokens == 2


def test_native_abort_between_materialization_and_apply(monkeypatch):
    core = engine(monkeypatch)
    core.scheduler.add_request(_request('cancel'))
    core.scheduler.add_request(_request('keep'))
    executor = core.model_executor
    core.step_fn()
    with ThreadPoolExecutor(max_workers=1) as thread:
        pending = thread.submit(core.step_fn)
        assert executor.sampling[0].entered.wait(5)
        core.aborts_queue.put(['cancel'])
        executor.sampling[0].set_result(_batch_output(executor.steps[0]))
        outputs, _ = pending.result(timeout=5)
    assert [o.request_id for o in outputs[0].outputs] == ['keep', 'cancel']
    cancel = next(o for o in outputs[0].outputs if o.request_id == 'cancel')
    assert cancel.finish_reason is not None and not cancel.new_token_ids
    assert core.scheduler.requests['keep'].num_output_tokens == 1
    assert 'cancel' not in core.scheduler.requests
    executor.sampling[1].set_result(_batch_output(executor.steps[1]))
    core.step_fn()
    assert core.scheduler.requests['keep'].num_output_tokens == 2


@pytest.mark.parametrize('failure', ['sampling', 'execution_none', 'execution'])
def test_original_future_failure_surfaces_without_retirement(monkeypatch, failure):
    core = engine(monkeypatch)
    core.scheduler.add_request(_request('a'))
    executor = core.model_executor
    core.step_fn()
    held = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    error = ValueError('controlled worker failure')
    if failure == 'sampling':
        executor.sampling[0].set_exception(error)
    elif failure == 'execution_none':
        executor.execution[0].set_exception(error)
        executor.sampling[0].set_result(None)
    else:
        executor.execution[0].set_exception(error)
        executor.sampling[0].set_exception(error)
    with pytest.raises(ValueError, match='controlled worker failure') as caught:
        core.step_fn()
    assert caught.value is error
    assert core.scheduler.processed_step_seq == 0
    assert core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    assert not any(row[0] == 'update' for row in executor.trace)
    with pytest.raises(Exception, match='two unconsumed'):
        core.step_fn()
    assert len(executor.steps) == 2


def test_no_eligible_work_drains_a_without_advancing_positions(monkeypatch):
    core = engine(monkeypatch)
    request = _request('a')
    request.max_tokens = request.sampling_params.max_tokens = 1
    core.scheduler.add_request(request)
    executor = core.model_executor
    core.step_fn()
    with ThreadPoolExecutor(max_workers=1) as thread:
        pending = thread.submit(core.step_fn)
        assert executor.sampling[0].entered.wait(5)
        assert len(executor.steps) == 2
        assert executor.steps[1].total_num_scheduled_tokens == 0
        assert len(executor.sampling) == 1
        assert core.scheduler.requests['a'].num_computed_tokens == 3
        executor.sampling[0].set_result(_batch_output(executor.steps[0]))
        pending.result(timeout=5)
    assert core.scheduler.processed_step_seq == 1
    core.step_fn()  # Cleanup C is queued before empty B is consumed.
    assert core.scheduler.processed_step_seq == 2
    core.step_fn()
    assert not core.scheduler._cfg_retiring


def test_abort_all_still_drains_nonempty_b_and_cleanup(monkeypatch):
    from vllm.v1.engine.core import EngineCoreProc

    core = engine(monkeypatch)
    core.engines_running = False
    core.scheduler.add_request(_request('a'))
    free = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    executor = core.model_executor
    core.step_fn()
    with ThreadPoolExecutor(max_workers=1) as thread:
        pending = thread.submit(core.step_fn)
        assert executor.sampling[0].entered.wait(5)
        core.aborts_queue.put(['a'])
        executor.sampling[0].set_result(_batch_output(executor.steps[0]))
        result, _ = pending.result(timeout=5)
    assert not core.scheduler.requests
    assert result[0].outputs[0].finish_reason is not None
    held = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    assert held < free
    executor.sampling[1].set_result(_batch_output(executor.steps[1]))
    core.step_fn()  # Cleanup C is submitted, retired B is consumed.
    assert core.scheduler.processed_step_seq == 2
    assert core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    assert EngineCoreProc.has_work(core)
    outputs, executed = core.step_fn()  # No live work: consume cleanup C.
    assert not any(client.outputs for client in outputs.values()) and not executed
    assert core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    assert not EngineCoreProc.has_work(core)
    assert core.scheduler.processed_step_seq == core.scheduler.sched_step_seq == 3


def test_a_uses_captured_mapping_after_b_admits_unrelated_prompt(monkeypatch):
    core = engine(monkeypatch)
    core.scheduler.add_request(_request('a'))
    executor = core.model_executor
    core.step_fn()
    core.scheduler.add_request(_request('plain', prompt=[0] * 5, info={'speaker_id': 'eng', 'text_tokens': [2]}))
    executor.sampling[0].set_result(_batch_output(executor.steps[0]))
    outputs, _ = core.step_fn()
    assert len(executor.steps[0].num_scheduled_tokens) == 2
    assert len(executor.steps[1].num_scheduled_tokens) == 3
    assert core.scheduler.cfg_confirmed_positions['a'] == 3
    assert core.scheduler.requests['a'].num_computed_tokens == 4
    assert core.scheduler.requests['plain'].num_output_tokens == 0
    assert [o.request_id for o in outputs[0].outputs] == ['a']
