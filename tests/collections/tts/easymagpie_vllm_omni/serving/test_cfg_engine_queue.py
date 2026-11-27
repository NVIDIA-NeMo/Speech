# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pinned EngineCore loop with standard controllable futures, no GPU executor."""
from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterator

import pytest
from easymagpie_vllm_omni.cfg_async_scheduler import EasyMagpieCFGAsyncScheduler
from easymagpie_vllm_omni.cfg_scheduler import EasyMagpieCFGScheduler
from test_cfg_concurrency import _batch_output, _mixed_scheduler
from test_cfg_scheduler import _request
from vllm.config import VllmConfig
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs

_TRACE_CORES = []


@pytest.fixture(autouse=True)
def save_native_queue_trace(request: pytest.FixtureRequest) -> Iterator[None]:
    """Record each engine core's executor trace and fences when ``P2_TRACE_DIR`` is set.

    Args:
        request: Pytest request for the running test, used to name the trace file.

    Yields:
        Control to the test after clearing the cores recorded by the previous test.
    """
    _TRACE_CORES.clear()
    yield
    destination = os.environ.get("P2_TRACE_DIR")
    if destination:
        out = Path(destination)
        out.mkdir(exist_ok=True)
        records = [
            dict(
                trace=core.model_executor.trace,
                queue_depth=core.batch_queue_size,
                unconsumed_records=len(getattr(core.scheduler, "_cfg_steps", ())),
                scheduled_fence=core.scheduler.sched_step_seq,
                consumed_fence=core.scheduler.processed_step_seq,
            )
            for core in _TRACE_CORES
        ]
        (out / (request.node.name + ".json")).write_text(json.dumps(records, indent=2) + "\n")


class ObservedFuture(Future):
    """Standard future that records when the engine waits on it and when it materializes."""

    def __init__(
        self,
        label: tuple[str, int],
        trace: list[tuple[Any, ...]],
    ) -> None:
        """Initialize an unresolved future with its trace label.

        Args:
            label: Future kind and executor step index.
            trace: Shared executor trace that receives wait and materialization events.
        """
        super().__init__()
        self.label, self.trace = label, trace
        self.entered = Event()

    def result(self, timeout: float | None = None) -> Any:
        """Record the wait, block for the result and record its materialization.

        Args:
            timeout: Seconds to wait; ``None`` uses a five-second bound instead of waiting forever.

        Returns:
            The value set on the future.
        """
        self.trace.append(("wait", self.label))
        self.entered.set()
        value = super().result(timeout=5 if timeout is None else timeout)
        self.trace.append(("materialized", self.label))
        return value


class ControlledExecutor:
    """Executor stand-in whose execution and sampling futures are resolved by the test."""

    def __init__(self, config: VllmConfig) -> None:
        """Initialize empty step, future and trace records.

        Args:
            config: Engine configuration passed by ``EngineCore``; unused.
        """
        self.trace = []
        self.steps, self.execution, self.sampling = [], [], []

    def execute_model(
        self,
        step: SchedulerOutput,
        non_block: bool = False,
    ) -> ObservedFuture:
        """Record a scheduled step and return its pending execution future.

        Args:
            step: Scheduled batch submitted by the engine.
            non_block: Whether the engine requests a future; must be true.

        Returns:
            An execution future, already resolved with an empty output when the step schedules no tokens.
        """
        assert non_block
        index = len(self.steps)
        self.trace.append(("execute", index, tuple(step.num_scheduled_tokens), tuple(step.finished_req_ids)))
        self.steps.append(step)
        future = ObservedFuture(("execute", index), self.trace)
        self.execution.append(future)
        if not step.total_num_scheduled_tokens:
            future.set_result(ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[]))
        return future

    def sample_tokens(
        self,
        grammar: GrammarOutput | None,
        non_block: bool = False,
    ) -> ObservedFuture:
        """Record a sampling request for the latest step and return its pending future.

        Args:
            grammar: Structured-output grammar for the step; unused.
            non_block: Whether the engine requests a future; must be true.

        Returns:
            A sampling future that the test resolves with the step's output.
        """
        assert non_block
        index = len(self.steps) - 1
        self.trace.append(("sample", index))
        future = ObservedFuture(("sample", index), self.trace)
        self.sampling.append(future)
        return future


def engine(
    monkeypatch: pytest.MonkeyPatch,
    async_mode: bool = True,
) -> EngineCore:
    """Build the actual ``EngineCore`` with the CFG scheduler and a controlled executor.

    Args:
        monkeypatch: Pytest fixture used to stub plugin loading, cache sizing and multimodal setup.
        async_mode: Whether to select the async scheduler and native batch queue.

    Returns:
        An engine core whose scheduler updates and aborts are recorded in the executor trace.
    """
    import vllm.plugins
    import vllm.v1.engine.core as module
    from vllm.multimodal import MULTIMODAL_REGISTRY

    template = _mixed_scheduler(max_num_seqs=5)
    config = template.vllm_config
    config.scheduler_config.async_scheduling = async_mode
    config.scheduler_config.scheduler_cls = (
        "easymagpie_vllm_omni.cfg_async_scheduler.EasyMagpieCFGAsyncScheduler"
        if async_mode
        else "easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGScheduler"
    )
    config.parallel_config.data_parallel_rank_local = 1
    config.model_config.runner_type = "generate"
    config.max_concurrent_batches = VllmConfig.max_concurrent_batches.fget(config)
    monkeypatch.setattr(vllm.plugins, "load_general_plugins", lambda: None)
    monkeypatch.setattr(EngineCore, "_initialize_kv_caches", lambda *a: template.kv_cache_config)
    monkeypatch.setattr(
        module, "StructuredOutputManager", lambda config: SimpleNamespace(should_advance=lambda r: False)
    )
    monkeypatch.setattr(module, "resolve_kv_cache_block_sizes", lambda *a: (4, 4))
    monkeypatch.setattr(MULTIMODAL_REGISTRY, "supports_multimodal_inputs", lambda *a: False)
    monkeypatch.setattr(MULTIMODAL_REGISTRY, "engine_receiver_cache_from_config", lambda *a: None)
    core = EngineCore(config, ControlledExecutor, log_stats=False)
    _TRACE_CORES.append(core)
    core.log_error_detail = lambda *a: nullcontext()
    core.log_iteration_details = lambda *a: nullcontext()
    core._should_throttle_prefills = lambda: False
    original = core.scheduler.update_from_output

    def update(
        step: SchedulerOutput,
        output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Record which step is applied, then apply it with the original scheduler method.

        Args:
            step: Scheduled batch whose result is applied.
            output: Model-runner output for the step.

        Returns:
            Engine outputs per client from the original update.
        """
        index = core.model_executor.steps.index(step)
        core.model_executor.trace.append(("update", index))
        return original(step, output)

    monkeypatch.setattr(core.scheduler, "update_from_output", update)
    original_abort = core.abort_requests

    def abort(ids: list[str]) -> None:
        """Record the aborted request IDs, then abort them with the original engine method.

        Args:
            ids: Request IDs to abort.
        """
        core.model_executor.trace.append(("abort", tuple(ids)))
        return original_abort(ids)

    monkeypatch.setattr(core, "abort_requests", abort)
    return core


def test_actual_constructor_selects_queue_and_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the engine constructor selects the batch queue and async scheduler, or the synchronous path.

    Args:
        monkeypatch: Pytest fixture used to build the engine.
    """
    core = engine(monkeypatch)
    assert type(core.scheduler) is EasyMagpieCFGAsyncScheduler
    assert core.batch_queue.maxlen == core.batch_queue_size == 2
    assert core.step_fn.__func__ is EngineCore.step_with_batch_queue
    sync = engine(monkeypatch, async_mode=False)
    assert type(sync.scheduler) is EasyMagpieCFGScheduler
    assert sync.batch_queue is None and sync.batch_queue_size == 1
    assert sync.step_fn.__func__ is EngineCore.step


def test_fifo_head_blocking_includes_popped_future(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the engine waits on the popped head future and applies results in FIFO order.

    Args:
        monkeypatch: Pytest fixture used to build the engine.
    """
    core = engine(monkeypatch)
    core.scheduler.add_request(_request("a"))
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
        assert not any(row[0] == "update" for row in executor.trace)
        executor.sampling[0].set_result(_batch_output(executor.steps[0]))
        result, executed = pending.result(timeout=5)
    assert executed and [o.request_id for o in result[0].outputs] == ["a"]
    assert [row for row in executor.trace if row[0] == "update"] == [("update", 0)]
    result, executed = core.step_fn()  # C may be submitted after A, before B is consumed.
    assert len(executor.steps) == 3
    assert [row for row in executor.trace if row[0] == "update"] == [("update", 0), ("update", 1)]
    assert core.scheduler.requests["a"].num_output_tokens == 2


def test_native_abort_between_materialization_and_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify an abort arriving after the result materializes finishes the request without its tokens.

    Args:
        monkeypatch: Pytest fixture used to build the engine.
    """
    core = engine(monkeypatch)
    core.scheduler.add_request(_request("cancel"))
    core.scheduler.add_request(_request("keep"))
    executor = core.model_executor
    core.step_fn()
    with ThreadPoolExecutor(max_workers=1) as thread:
        pending = thread.submit(core.step_fn)
        assert executor.sampling[0].entered.wait(5)
        core.aborts_queue.put(["cancel"])
        executor.sampling[0].set_result(_batch_output(executor.steps[0]))
        outputs, _ = pending.result(timeout=5)
    assert [o.request_id for o in outputs[0].outputs] == ["keep", "cancel"]
    cancel = next(o for o in outputs[0].outputs if o.request_id == "cancel")
    assert cancel.finish_reason is not None and not cancel.new_token_ids
    assert core.scheduler.requests["keep"].num_output_tokens == 1
    assert "cancel" not in core.scheduler.requests
    executor.sampling[1].set_result(_batch_output(executor.steps[1]))
    core.step_fn()
    assert core.scheduler.requests["keep"].num_output_tokens == 2


@pytest.mark.parametrize("failure", ["sampling", "execution_none", "execution"])
def test_original_future_failure_surfaces_without_retirement(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Verify a worker future failure is raised unchanged without retiring or applying the step.

    Args:
        monkeypatch: Pytest fixture used to build the engine.
        failure: Which future fails: sampling, execution with a ``None`` sample, or both.
    """
    core = engine(monkeypatch)
    core.scheduler.add_request(_request("a"))
    executor = core.model_executor
    core.step_fn()
    held = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    error = ValueError("controlled worker failure")
    if failure == "sampling":
        executor.sampling[0].set_exception(error)
    elif failure == "execution_none":
        executor.execution[0].set_exception(error)
        executor.sampling[0].set_result(None)
    else:
        executor.execution[0].set_exception(error)
        executor.sampling[0].set_exception(error)
    with pytest.raises(ValueError, match="controlled worker failure") as caught:
        core.step_fn()
    assert caught.value is error
    assert core.scheduler.processed_step_seq == 0
    assert core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    assert not any(row[0] == "update" for row in executor.trace)
    with pytest.raises(Exception, match="two unconsumed"):
        core.step_fn()
    assert len(executor.steps) == 2


def test_no_eligible_work_drains_a_without_advancing_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify an empty successor step drains the pending step without advancing positions.

    Args:
        monkeypatch: Pytest fixture used to build the engine.
    """
    core = engine(monkeypatch)
    request = _request("a")
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
        assert core.scheduler.requests["a"].num_computed_tokens == 3
        executor.sampling[0].set_result(_batch_output(executor.steps[0]))
        pending.result(timeout=5)
    assert core.scheduler.processed_step_seq == 1
    core.step_fn()  # Cleanup C is queued before empty B is consumed.
    assert core.scheduler.processed_step_seq == 2
    core.step_fn()
    assert not core.scheduler._cfg_retiring


def test_abort_all_still_drains_nonempty_b_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify aborting every request still drains the in-flight step and its cleanup step before freeing blocks.

    Args:
        monkeypatch: Pytest fixture used to build the engine.
    """
    from vllm.v1.engine.core import EngineCoreProc

    core = engine(monkeypatch)
    core.engines_running = False
    core.scheduler.add_request(_request("a"))
    free = core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    executor = core.model_executor
    core.step_fn()
    with ThreadPoolExecutor(max_workers=1) as thread:
        pending = thread.submit(core.step_fn)
        assert executor.sampling[0].entered.wait(5)
        core.aborts_queue.put(["a"])
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


def test_a_uses_captured_mapping_after_b_admits_unrelated_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a pending step applies with its captured mapping after the next step admits another prompt.

    Args:
        monkeypatch: Pytest fixture used to build the engine.
    """
    core = engine(monkeypatch)
    core.scheduler.add_request(_request("a"))
    executor = core.model_executor
    core.step_fn()
    core.scheduler.add_request(_request("plain", prompt=[0] * 5, info={"speaker_id": "eng", "text_tokens": [2]}))
    executor.sampling[0].set_result(_batch_output(executor.steps[0]))
    outputs, _ = core.step_fn()
    assert len(executor.steps[0].num_scheduled_tokens) == 2
    assert len(executor.steps[1].num_scheduled_tokens) == 3
    assert core.scheduler.cfg_confirmed_positions["a"] == 3
    assert core.scheduler.requests["a"].num_computed_tokens == 4
    assert core.scheduler.requests["plain"].num_output_tokens == 0
    assert [o.request_id for o in outputs[0].outputs] == ["a"]
