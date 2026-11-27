# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only native async CFG accounting with real hybrid cache blocks."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

import pytest
from easymagpie_vllm_omni.cfg_scheduler import EasyMagpieCFGScheduler
from test_cfg_concurrency import _batch_output, _mixed_scheduler
from test_cfg_scheduler import _request
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from easymagpie_vllm_omni.cfg_async_scheduler import EasyMagpieCFGAsyncScheduler
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


def async_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: Any,
) -> EasyMagpieCFGAsyncScheduler:
    """Build the mixed CFG test scheduler as a native async CFG scheduler.

    When the trace-directory environment variable is set, `schedule`, `update_from_output` and `finish_requests`
    are wrapped to write a JSON trace of step counters, request token counts, retiring pairs and free cache blocks
    after every call.

    Args:
        monkeypatch: Pytest fixture used to swap in the async scheduler class and install trace wrappers.
        **kwargs: Cache and budget options forwarded to `_mixed_scheduler`.

    Returns:
        The async CFG scheduler with `async_scheduling` enabled.
    """
    import test_cfg_concurrency
    from easymagpie_vllm_omni.cfg_async_scheduler import EasyMagpieCFGAsyncScheduler

    class Configured(EasyMagpieCFGAsyncScheduler):
        """Async CFG scheduler that forces `async_scheduling` on before construction."""

        def __init__(self, **params: Any) -> None:
            """Enable async scheduling on the supplied config and build the scheduler.

            Args:
                **params: Scheduler constructor arguments; must include `vllm_config`.
            """
            params["vllm_config"].scheduler_config.async_scheduling = True
            super().__init__(**params)

    with monkeypatch.context() as patch:
        patch.setattr(test_cfg_concurrency, "EasyMagpieCFGScheduler", Configured)
        scheduler = _mixed_scheduler(**kwargs)
    destination = os.environ.get("P2_TRACE_DIR")
    if destination:
        path = Path(destination)
        path.mkdir(exist_ok=True)
        output = path / ("counter_" + uuid4().hex + ".json")
        trace = []
        for name in ("schedule", "update_from_output", "finish_requests"):
            bound = getattr(scheduler, name)

            def record(
                *args: Any,
                _bound: Callable[..., Any] = bound,
                _name: str = name,
                **options: Any,
            ) -> Any:
                """Call the wrapped scheduler method and append a trace entry to the JSON trace file.

                Args:
                    *args: Positional arguments forwarded to the wrapped method.
                    _bound: Original bound scheduler method.
                    _name: Name of the wrapped method recorded as the trace event.
                    **options: Keyword arguments forwarded to the wrapped method.

                Returns:
                    The wrapped method's return value.
                """
                try:
                    return _bound(*args, **options)
                finally:
                    trace.append(
                        dict(
                            event=_name,
                            scheduled=scheduler.sched_step_seq,
                            consumed=scheduler.processed_step_seq,
                            requests={
                                rid: [r.num_computed_tokens, r.num_output_placeholders, r.num_output_tokens]
                                for rid, r in scheduler.requests.items()
                            },
                            retiring=dict(scheduler._cfg_retiring),
                            free_blocks=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                        )
                    )
                    output.write_text(
                        json.dumps(dict(test=os.environ.get("PYTEST_CURRENT_TEST"), trace=trace), indent=2) + "\n"
                    )

            monkeypatch.setattr(scheduler, name, record)
    return scheduler


def test_native_prefill_decode_placeholder_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify both pair branches move from output placeholders to output tokens as step results arrive.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request("a"))
    pair = scheduler.cfg_pairs["a"]
    a = scheduler.schedule()
    assert [
        (
            scheduler.requests[r].num_computed_tokens,
            scheduler.requests[r].num_output_placeholders,
            scheduler.requests[r].num_output_tokens,
        )
        for r in pair.physical_ids
    ] == [(3, 1, 0)] * 2
    b = scheduler.schedule()
    assert [
        (
            scheduler.requests[r].num_computed_tokens,
            scheduler.requests[r].num_output_placeholders,
            scheduler.requests[r].num_output_tokens,
        )
        for r in pair.physical_ids
    ] == [(4, 2, 0)] * 2
    scheduler.update_from_output(a, _batch_output(a))
    assert [
        (
            scheduler.requests[r].num_computed_tokens,
            scheduler.requests[r].num_output_placeholders,
            scheduler.requests[r].num_output_tokens,
        )
        for r in pair.physical_ids
    ] == [(4, 1, 1)] * 2
    scheduler.update_from_output(b, _batch_output(b))
    assert [
        (
            scheduler.requests[r].num_computed_tokens,
            scheduler.requests[r].num_output_placeholders,
            scheduler.requests[r].num_output_tokens,
        )
        for r in pair.physical_ids
    ] == [(4, 0, 2)] * 2


def test_abort_holds_blocks_until_cleanup_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify an aborted pair keeps its cache blocks and request id until the cleanup step result is applied.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch)
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_request("a"))
    ids = scheduler.cfg_pairs["a"].physical_ids
    a, b = scheduler.schedule(), scheduler.schedule()
    held = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    assert scheduler.finish_requests(ids[1], RequestStatus.FINISHED_ABORTED) == [("a", 0)]
    assert scheduler.finish_requests(ids[0], RequestStatus.FINISHED_ABORTED) == []
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    with pytest.raises(ValueError, match="retir"):
        scheduler.add_request(_request("a"))
    scheduler.update_from_output(a, _batch_output(a))
    cleanup = scheduler.schedule()
    assert cleanup.finished_req_ids == set(ids)
    scheduler.update_from_output(b, _batch_output(b))
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    scheduler.add_request(_request("a"))


def test_async_admits_resumable_then_final_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resumable chunks and a final sentinel are admitted by default while keeping one CFG pair.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    from test_cfg_streaming import _chunk

    scheduler = async_scheduler(monkeypatch, max_num_seqs=4, num_blocks=128)
    scheduler.add_request(_chunk(initial=True))
    assert "stream" in scheduler.cfg_pairs
    scheduler.add_request(_chunk())
    scheduler.add_request(_request("stream", prompt=[0], info={}, resumable=False))
    assert len(scheduler.cfg_pairs["stream"].physical_ids) == 2


@pytest.mark.parametrize("value", [None, "", "0", "false"])
def test_async_streaming_does_not_depend_on_environment(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    """Verify incremental input is admitted whatever the former opt-in variable holds.

    Args:
        monkeypatch: Pytest fixture used to set the variable and build the async scheduler.
        value: Value of the former ``EASYMAGPIE_ASYNC_STREAMING`` variable, or None to leave it unset.
    """
    from test_cfg_streaming import _chunk

    if value is None:
        monkeypatch.delenv("EASYMAGPIE_ASYNC_STREAMING", raising=False)
    else:
        monkeypatch.setenv("EASYMAGPIE_ASYNC_STREAMING", value)
    scheduler = async_scheduler(monkeypatch, max_num_seqs=4, num_blocks=128)
    scheduler.add_request(_chunk(initial=True))
    assert "stream" in scheduler.cfg_pairs


def test_async_admits_complete_text_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify complete-text requests are admitted alongside incremental input support.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request("a"))
    assert "a" in scheduler.cfg_pairs or "a" in scheduler.requests


def test_default_sync_profile_still_rejects_second_step() -> None:
    """Verify the default synchronous CFG scheduler refuses a second step before the first result."""
    scheduler = _mixed_scheduler()
    assert isinstance(scheduler, EasyMagpieCFGScheduler)
    scheduler.add_request(_request())
    scheduler.schedule()
    with pytest.raises(Exception, match="synchronous"):
        scheduler.schedule()


def test_native_async_mro_and_bounded_unconsumed_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the async scheduler MRO order and that at most two steps may be unconsumed.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    scheduler = async_scheduler(monkeypatch)
    mro = type(scheduler).__mro__
    assert mro.index(OmniARScheduler) < mro.index(AsyncScheduler) < mro.index(Scheduler)
    scheduler.add_request(_request("a"))
    a, b = scheduler.schedule(), scheduler.schedule()
    with pytest.raises(Exception, match="two unconsumed"):
        scheduler.schedule()
    scheduler.update_from_output(a, _batch_output(a))
    assert scheduler.cfg_confirmed_positions["a"] == 3
    assert scheduler.requests["a"].num_computed_tokens == 4
    scheduler.update_from_output(b, _batch_output(b))
    assert scheduler.cfg_confirmed_positions["a"] == 4


def test_pending_demand_admits_two_pairs_and_plain_with_physical_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a step with a pending result admits a new pair and a plain request within the token budget.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch, max_num_seqs=5, token_budget=10)
    scheduler.add_request(_request("first"))
    a = scheduler.schedule()
    scheduler.add_request(_request("second", prompt=[0, 0]))
    scheduler.add_request(_request("plain", prompt=[0] * 4, info={"speaker_id": "eng", "text_tokens": [2, 3]}))
    b = scheduler.schedule()
    assert b.total_num_scheduled_tokens == 10
    assert b.num_scheduled_tokens["first"] == 1
    assert b.num_scheduled_tokens["second"] == 2
    assert b.num_scheduled_tokens["plain"] == 4
    assert not b.preempted_req_ids
    scheduler.update_from_output(a, _batch_output(a))
    assert scheduler.requests["plain"].num_output_tokens == 0
    assert scheduler.cfg_confirmed_positions["first"] == 3
    scheduler.update_from_output(b, _batch_output(b))
    assert scheduler.requests["plain"].num_output_tokens == 1


def test_pending_token_pressure_does_not_admit_half_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a token budget too small for both branches leaves the new pair waiting.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch, max_num_seqs=4, token_budget=6)
    scheduler.add_request(_request("first"))
    a = scheduler.schedule()
    scheduler.add_request(_request("second"))
    b = scheduler.schedule()
    assert b.total_num_scheduled_tokens == 2 and "second" not in b.num_scheduled_tokens
    assert len(scheduler.waiting) == 2
    scheduler.update_from_output(a, _batch_output(a))
    scheduler.update_from_output(b, _batch_output(b))


def test_second_branch_reservation_failure_rolls_back_only_new_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify an unconditional-branch allocation failure frees only the new pair's blocks.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler and patch slot allocation.
    """
    scheduler = async_scheduler(monkeypatch, max_num_seqs=5)
    scheduler.add_request(_request("first"))
    a = scheduler.schedule()
    scheduler.add_request(_request("new"))
    pair = scheduler.cfg_pairs["new"]
    manager = scheduler.kv_cache_manager
    free = manager.block_pool.get_num_free_blocks()
    live_blocks = manager.get_blocks("first").get_block_ids()
    original = manager.allocate_slots

    def allocate(
        request: Request,
        *args: Any,
        **kwargs: Any,
    ) -> KVCacheBlocks | None:
        """Fail allocation for the new pair's unconditional branch and delegate otherwise.

        Args:
            request: Physical request asking for cache slots.
            *args: Positional arguments forwarded to the original allocator.
            **kwargs: Keyword arguments forwarded to the original allocator.

        Returns:
            None for the unconditional branch, otherwise the original allocator's result.
        """
        if request.request_id == pair.unconditional_id:
            return None
        return original(request, *args, **kwargs)

    monkeypatch.setattr(manager, "allocate_slots", allocate)
    b = scheduler.schedule()
    assert "new" not in b.num_scheduled_tokens
    assert manager.block_pool.get_num_free_blocks() == free
    assert manager.get_blocks("first").get_block_ids() == live_blocks
    assert all(not any(manager.get_blocks(rid).get_block_ids()) for rid in pair.physical_ids)
    scheduler.update_from_output(a, _batch_output(a))
    scheduler.update_from_output(b, _batch_output(b))


def test_cache_pressure_keeps_retired_blocks_unavailable_until_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify retired pair blocks stay unavailable to a waiting request until the cleanup result is applied.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch, max_num_seqs=4, num_blocks=10)
    manager = scheduler.kv_cache_manager
    free = manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_request("first"))
    a, b = scheduler.schedule(), scheduler.schedule()
    old_blocks = manager.get_blocks("first").get_block_ids()
    scheduler.finish_requests("first", RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(_request("second"))
    scheduler.update_from_output(a, _batch_output(a))
    cleanup = scheduler.schedule()
    assert not cleanup.num_scheduled_tokens
    scheduler.update_from_output(b, _batch_output(b))
    assert manager.block_pool.get_num_free_blocks() < free
    scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert manager.block_pool.get_num_free_blocks() == free
    next_step = scheduler.schedule()
    assert "second" in next_step.num_scheduled_tokens
    assert old_blocks  # Actual block identities were allocated and quarantined.


@pytest.mark.parametrize("allowance", [1, 2, 4])
def test_exact_generation_caps_and_empty_work(
    monkeypatch: pytest.MonkeyPatch,
    allowance: int,
) -> None:
    """Verify a pair stops at exactly `max_tokens` outputs and leaves no retiring state after cleanup.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
        allowance: Maximum number of generated tokens for the request.
    """
    scheduler = async_scheduler(monkeypatch)
    request = _request("a")
    request.max_tokens = request.sampling_params.max_tokens = allowance
    scheduler.add_request(request)
    a, b = scheduler.schedule(), scheduler.schedule()
    if allowance == 1:
        assert not b.num_scheduled_tokens
        assert all(r.num_computed_tokens == 3 for r in scheduler.running)
        scheduler.update_from_output(a, _batch_output(a))
        scheduler.update_from_output(b, _batch_output(b))
    else:
        scheduler.update_from_output(a, _batch_output(a))
        scheduler.update_from_output(b, _batch_output(b))
        for _ in range(allowance - 2):
            step = scheduler.schedule()
            scheduler.update_from_output(step, _batch_output(step))
    assert not scheduler.requests
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert not scheduler._cfg_retiring and not scheduler.deferred_frees


def test_terminal_a_with_b_pending_is_public_once_and_cleanup_fenced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a stop in step A is published once, B's late result is hidden, and blocks free after cleanup.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch)
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_request("a"))
    a, b = scheduler.schedule(), scheduler.schedule()
    out = scheduler.update_from_output(a, _batch_output(a, stopped={"a"}))
    assert [o.request_id for o in out[0].outputs] == ["a"]
    assert out[0].finished_requests == {"a"}
    cleanup = scheduler.schedule()
    late = scheduler.update_from_output(b, _batch_output(b))
    assert not any(client.outputs for client in late.values())
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() < free
    scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


def test_cancel_before_dispatch_and_repeated_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify aborting an undispatched pair is reported once and its id is reusable after cleanup.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request("a"))
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    assert scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED) == [("a", 0)]
    assert scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED) == []
    cleanup = scheduler.schedule()
    assert cleanup.total_num_scheduled_tokens == 0 and len(cleanup.finished_req_ids) == 2
    result = scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert [o.request_id for o in result[0].outputs] == ["a"]
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    scheduler.add_request(_request("a"))


@pytest.mark.parametrize(
    "corruption", ["missing_partner", "wrong_row", "extra_row", "placeholder", "no_output", "count"]
)
def test_malformed_result_latches_before_partial_application(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    """Verify a malformed step result is rejected before any state changes and latches the scheduler unusable.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
        corruption: Kind of corruption applied to the step result or scheduler state.
    """
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request("a"))
    step = scheduler.schedule()
    output = _batch_output(step)
    if corruption == "missing_partner":
        output.req_ids.pop()
    elif corruption == "wrong_row":
        output.req_id_to_index = {rid: 0 for rid in output.req_ids}
    elif corruption == "extra_row":
        output.req_ids.append("unknown")
    elif corruption == "placeholder":
        scheduler.requests["a"].num_output_placeholders += 1
    elif corruption == "no_output":
        output.sampled_token_ids = []
    else:
        step.num_scheduled_tokens["a"] = 99
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    with pytest.raises(Exception):
        scheduler.update_from_output(step, output)
    assert all(r.num_output_tokens == 0 for r in scheduler.requests.values())
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    assert scheduler.processed_step_seq == 0
    with pytest.raises(Exception, match="unusable"):
        scheduler.schedule()


def test_result_out_of_order_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify applying a newer step result before the oldest pending one raises without consuming a step.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request("a"))
    a, b = scheduler.schedule(), scheduler.schedule()
    with pytest.raises(Exception, match="oldest"):
        scheduler.update_from_output(b, _batch_output(b))
    assert scheduler.processed_step_seq == 0


def test_preemption_fails_before_resource_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify preempting a dispatched request raises without changing its tokens or cache blocks.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request("a"))
    step = scheduler.schedule()
    request = scheduler.requests["a"]
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    with pytest.raises(Exception, match="cannot preempt"):
        scheduler._preempt_request(request, 0)
    assert request.num_computed_tokens == 3
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


def test_full_history_bounds_and_oversize_reservations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify requests exceeding the model length or cache capacity are rejected without adding state.

    Args:
        monkeypatch: Pytest fixture used to build the async schedulers.
    """
    scheduler = async_scheduler(monkeypatch)
    request = _request("too_long", prompt=[0] * 62)
    with pytest.raises(ValueError, match="max_model_len"):
        scheduler.add_request(request)
    assert not scheduler.requests
    tiny = async_scheduler(monkeypatch, num_blocks=4)
    with pytest.raises(ValueError, match="cache"):
        tiny.add_request(_request("too_big"))
    assert not tiny.requests


@pytest.mark.parametrize(
    "feature",
    ["sync", "graphs", "chunked", "prefix", "tp", "pp", "speculation", "kv_transfer", "ec_transfer", "v2_runner"],
)
def test_async_config_rejects_invalid_policies_and_excluded_features(
    monkeypatch: pytest.MonkeyPatch,
    feature: str,
) -> None:
    """Verify config validation rejects each unsupported feature without changing the live config.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
        feature: Unsupported feature enabled on a copy of the scheduler config.
    """
    import copy

    scheduler = async_scheduler(monkeypatch)
    config = copy.deepcopy(scheduler.vllm_config)
    if feature == "sync":
        config.scheduler_config.async_scheduling = False
    elif feature == "graphs":
        config.model_config.enforce_eager = False
    elif feature == "chunked":
        config.scheduler_config.enable_chunked_prefill = True
    elif feature == "prefix":
        config.cache_config.enable_prefix_caching = True
    elif feature == "tp":
        config.parallel_config.tensor_parallel_size = 2
    elif feature == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    elif feature == "speculation":
        config.speculative_config = object()
    elif feature == "kv_transfer":
        config.kv_transfer_config = object()
    elif feature == "ec_transfer":
        config.ec_transfer_config = object()
    else:
        config.use_v2_model_runner = True
    with pytest.raises(ValueError):
        scheduler._validate_config(config)
    assert scheduler.vllm_config.scheduler_config.async_scheduling is True


def test_default_topology_remains_synchronous() -> None:
    """Verify the default CFG pipeline stage still uses the synchronous CFG scheduler."""
    from easymagpie_vllm_omni.pipeline import EASYMAGPIE_CFG_PIPELINE

    assert (
        EASYMAGPIE_CFG_PIPELINE.stages[0].scheduler_cls == "easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGScheduler"
    )


def test_exact_full_history_limit_does_not_submit_extra_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a request reaching the full-history limit gets no extra decode step and then finishes.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch, token_budget=128, num_blocks=128)
    scheduler.add_request(_request("a", prompt=[0] * 60))
    a, b = scheduler.schedule(), scheduler.schedule()
    assert a.num_scheduled_tokens["a"] == 60 and b.num_scheduled_tokens["a"] == 1
    scheduler.update_from_output(a, _batch_output(a))
    scheduler.update_from_output(b, _batch_output(b))
    c, d = scheduler.schedule(), scheduler.schedule()
    assert c.num_scheduled_tokens["a"] == d.num_scheduled_tokens["a"] == 1
    assert scheduler.requests["a"].num_computed_tokens == 63
    scheduler.update_from_output(c, _batch_output(c))
    empty = scheduler.schedule()
    assert not empty.num_scheduled_tokens
    scheduler.update_from_output(d, _batch_output(d))
    assert not scheduler.requests


def test_scheduler_application_error_keeps_resources_and_latches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify an error while applying a result keeps cache blocks held and latches the scheduler unusable.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler and inject the accounting failure.
    """
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request("a"))
    step = scheduler.schedule()
    held = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    original = AsyncScheduler._update_request_with_output
    count = []

    def fail_second(
        self,
        request: Request,
        tokens: list[int],
    ) -> tuple[list[int], bool]:
        """Raise on the second request update and delegate the others to the original method.

        Args:
            request: Physical request receiving sampled tokens.
            tokens: Sampled token ids for the request.

        Returns:
            The original method's new token ids and stop flag.

        Raises:
            RuntimeError: On the second call.
        """
        count.append(request.request_id)
        if len(count) == 2:
            raise RuntimeError("injected accounting failure")
        return original(self, request, tokens)

    monkeypatch.setattr(AsyncScheduler, "_update_request_with_output", fail_second)
    with pytest.raises(RuntimeError, match="injected accounting failure"):
        scheduler.update_from_output(step, _batch_output(step))
    assert scheduler.processed_step_seq == 0
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    assert scheduler.requests["a"].num_output_tokens == 1  # No unsafe rollback claim.
    with pytest.raises(Exception, match="unusable"):
        scheduler.add_request(_request("fresh"))
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)  # Logical teardown is permitted.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held


def test_successor_uses_distinct_actual_blocks_before_old_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a new pair admitted alongside a retiring pair's cleanup gets disjoint cache blocks.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch, max_num_seqs=4)
    manager = scheduler.kv_cache_manager
    scheduler.add_request(_request("old"))
    a, b = scheduler.schedule(), scheduler.schedule()
    old_ids = scheduler.cfg_pairs["old"].physical_ids
    old_blocks = [set(sum(manager.get_blocks(rid).get_block_ids(), [])) for rid in old_ids]
    scheduler.finish_requests("old", RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(a, _batch_output(a))
    scheduler.add_request(_request("new"))
    cleanup_and_prefill = scheduler.schedule()
    new_ids = scheduler.cfg_pairs["new"].physical_ids
    new_blocks = [set(sum(manager.get_blocks(rid).get_block_ids(), [])) for rid in new_ids]
    assert all(x.isdisjoint(y) for x in old_blocks for y in new_blocks)
    assert cleanup_and_prefill.finished_req_ids == set(old_ids)
    scheduler.update_from_output(b, _batch_output(b))
    scheduler.update_from_output(cleanup_and_prefill, _batch_output(cleanup_and_prefill))
    assert not scheduler._cfg_retiring


def test_sequence_slots_remain_atomic_while_first_result_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a waiting pair is not partly scheduled when only the running pair fits the sequence slots.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    scheduler = async_scheduler(monkeypatch, max_num_seqs=2)
    scheduler.add_request(_request("first"))
    a = scheduler.schedule()
    scheduler.add_request(_request("waiting"))
    b = scheduler.schedule()
    assert set(b.num_scheduled_tokens) == set(scheduler.cfg_pairs["first"].physical_ids)
    assert all(scheduler.requests[rid].num_computed_tokens == 0 for rid in scheduler.cfg_pairs["waiting"].physical_ids)
    scheduler.update_from_output(a, _batch_output(a))
    scheduler.update_from_output(b, _batch_output(b))


def test_allocator_pressure_cannot_preempt_half_of_submitted_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify allocation failure for a submitted pair raises and latches instead of preempting one branch.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler and patch slot allocation.
    """
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request("a"))
    a = scheduler.schedule()
    manager = scheduler.kv_cache_manager
    held = manager.block_pool.get_num_free_blocks()
    monkeypatch.setattr(manager, "allocate_slots", lambda *a, **k: None)
    with pytest.raises(Exception, match="cannot preempt"):
        scheduler.schedule()
    assert scheduler._cfg_failure is not None
    assert manager.block_pool.get_num_free_blocks() == held
    assert all(r.num_computed_tokens == 3 for r in scheduler.requests.values())
    assert len(scheduler._cfg_steps) == 1  # No successor batch can reach dispatch.


def test_async_streaming_late_empty_sentinel_after_cleanup_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a final stream sentinel arriving after the aborted stream is cleaned up adds no state.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """
    from test_cfg_streaming import _chunk

    scheduler = async_scheduler(monkeypatch, max_num_seqs=4, num_blocks=128)
    scheduler.add_request(_chunk(initial=True))
    scheduler.finish_requests("stream", RequestStatus.FINISHED_ABORTED)
    for _ in range(3):
        if not scheduler.requests:
            break
        step = scheduler.schedule()
        scheduler.update_from_output(step, _batch_output(step))
    scheduler._cfg_retiring.clear()
    terminal = _request(
        "stream", prompt=[0], info={"global_request_id": ["stream"], "omni_final_stage_id": 1}, resumable=False
    )
    scheduler.add_request(terminal)
    assert not scheduler.requests and not scheduler.cfg_pairs and not scheduler._cfg_streams


@pytest.mark.parametrize("retired", [False, True])
@pytest.mark.parametrize("kind", ["sentinel", "continuation_chunk", "final_update_with_tokens"])
def test_async_streaming_late_input_for_aborted_stream_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    retired: bool,
) -> None:
    """A client disconnect can race the input iterator: late input for an aborted stream must not kill the engine.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
        kind: Kind of late input sent for the aborted stream.
        retired: Whether the aborted stream finishes cleanup before the late input arrives.
    """
    from test_cfg_streaming import _chunk

    scheduler = async_scheduler(monkeypatch, max_num_seqs=4, num_blocks=128)
    scheduler.add_request(_chunk(initial=True))
    scheduler.finish_requests("stream", RequestStatus.FINISHED_ABORTED)
    assert "stream" in scheduler._cfg_retiring
    if retired:
        for _ in range(4):
            step = scheduler.schedule()
            scheduler.update_from_output(step, _batch_output(step))
        assert "stream" not in scheduler._cfg_retiring
    if kind == "sentinel":
        late = _request(
            "stream", prompt=[0], info={"global_request_id": ["stream"], "omni_final_stage_id": 1}, resumable=False
        )
    elif kind == "final_update_with_tokens":
        late = _request("stream", prompt=[0], info={"text_token": [4, 5], "text_token_start": 2}, resumable=False)
    else:
        late = _chunk()
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(late)
    assert not scheduler.requests and not scheduler.cfg_pairs and not scheduler._cfg_streams
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


@pytest.mark.parametrize("resumable", [True, False])
@pytest.mark.parametrize(
    "info",
    [
        {"text_token": [], "text_token_start": 3},
        {"text_token_start": 0},
        {"global_request_id": ["stream"], "omni_final_stage_id": 1, "text_token": [4], "text_token_start": 2},
        {"text_token": [4, 5]},
    ],
)
def test_async_streaming_drops_any_placeholder_input_for_removed_stream(
    monkeypatch: pytest.MonkeyPatch,
    info: dict[str, Any],
    resumable: bool,
) -> None:
    """Any placeholder-prompt input for an id that is not live is stream tail, never synthesis, and must not raise.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
        info: Request metadata carried by the late placeholder input.
        resumable: Whether the late input is marked resumable.
    """
    from test_cfg_streaming import _chunk

    scheduler = async_scheduler(monkeypatch, max_num_seqs=4, num_blocks=128)
    scheduler.add_request(_chunk(initial=True))
    scheduler.finish_requests("stream", RequestStatus.FINISHED_ABORTED)
    assert "stream" in scheduler._cfg_retiring
    scheduler.add_request(_request("stream", prompt=[0], info=dict(info), resumable=resumable))
    assert not scheduler.requests and not scheduler.cfg_pairs and not scheduler._cfg_streams


def test_async_streaming_still_rejects_placeholder_input_for_live_stream_without_stream_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drop applies only to ids that are not live; a real fresh request with a retiring id is still refused.

    Args:
        monkeypatch: Pytest fixture used to build the async scheduler.
    """

    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request("a"))
    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    assert "a" in scheduler._cfg_retiring
    with pytest.raises(ValueError, match="retiring"):
        scheduler.add_request(_request("a"))
