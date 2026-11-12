# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only native async CFG accounting with real hybrid cache blocks."""
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from easymagpie_vllm_omni.cfg_scheduler import EasyMagpieCFGScheduler
from test_cfg_concurrency import _batch_output, _mixed_scheduler
from test_cfg_scheduler import _request
from vllm.v1.request import RequestStatus


def async_scheduler(monkeypatch, **kwargs):
    import test_cfg_concurrency
    from easymagpie_vllm_omni.cfg_async_scheduler import EasyMagpieCFGAsyncScheduler

    class Configured(EasyMagpieCFGAsyncScheduler):
        def __init__(self, **params):
            params['vllm_config'].scheduler_config.async_scheduling = True
            super().__init__(**params)

    with monkeypatch.context() as patch:
        patch.setattr(test_cfg_concurrency, 'EasyMagpieCFGScheduler', Configured)
        scheduler = _mixed_scheduler(**kwargs)
    destination = os.environ.get('P2_TRACE_DIR')
    if destination:
        path = Path(destination)
        path.mkdir(exist_ok=True)
        output = path / ('counter_' + uuid4().hex + '.json')
        trace = []
        for name in ('schedule', 'update_from_output', 'finish_requests'):
            bound = getattr(scheduler, name)

            def record(*args, _bound=bound, _name=name, **options):
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
                        json.dumps(dict(test=os.environ.get('PYTEST_CURRENT_TEST'), trace=trace), indent=2) + '\n'
                    )

            monkeypatch.setattr(scheduler, name, record)
    return scheduler


def test_native_prefill_decode_placeholder_transitions(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request('a'))
    pair = scheduler.cfg_pairs['a']
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


def test_abort_holds_blocks_until_cleanup_result(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_request('a'))
    ids = scheduler.cfg_pairs['a'].physical_ids
    a, b = scheduler.schedule(), scheduler.schedule()
    held = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    assert scheduler.finish_requests(ids[1], RequestStatus.FINISHED_ABORTED) == [('a', 0)]
    assert scheduler.finish_requests(ids[0], RequestStatus.FINISHED_ABORTED) == []
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    with pytest.raises(ValueError, match='retir'):
        scheduler.add_request(_request('a'))
    scheduler.update_from_output(a, _batch_output(a))
    cleanup = scheduler.schedule()
    assert cleanup.finished_req_ids == set(ids)
    scheduler.update_from_output(b, _batch_output(b))
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    scheduler.add_request(_request('a'))


def test_async_rejects_resumable_without_mutation(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    with pytest.raises(ValueError, match='resumable'):
        scheduler.add_request(_request(resumable=True))
    assert not scheduler.requests


def test_default_sync_profile_still_rejects_second_step():
    scheduler = _mixed_scheduler()
    assert isinstance(scheduler, EasyMagpieCFGScheduler)
    scheduler.add_request(_request())
    scheduler.schedule()
    with pytest.raises(Exception, match='synchronous'):
        scheduler.schedule()


def test_native_async_mro_and_bounded_unconsumed_records(monkeypatch):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    scheduler = async_scheduler(monkeypatch)
    mro = type(scheduler).__mro__
    assert mro.index(OmniARScheduler) < mro.index(AsyncScheduler) < mro.index(Scheduler)
    scheduler.add_request(_request('a'))
    a, b = scheduler.schedule(), scheduler.schedule()
    with pytest.raises(Exception, match='two unconsumed'):
        scheduler.schedule()
    scheduler.update_from_output(a, _batch_output(a))
    assert scheduler.cfg_confirmed_positions['a'] == 3
    assert scheduler.requests['a'].num_computed_tokens == 4
    scheduler.update_from_output(b, _batch_output(b))
    assert scheduler.cfg_confirmed_positions['a'] == 4


def test_pending_demand_admits_two_pairs_and_plain_with_physical_budget(monkeypatch):
    scheduler = async_scheduler(monkeypatch, max_num_seqs=5, token_budget=10)
    scheduler.add_request(_request('first'))
    a = scheduler.schedule()
    scheduler.add_request(_request('second', prompt=[0, 0]))
    scheduler.add_request(_request('plain', prompt=[0] * 4, info={'speaker_id': 'eng', 'text_tokens': [2, 3]}))
    b = scheduler.schedule()
    assert b.total_num_scheduled_tokens == 10
    assert b.num_scheduled_tokens['first'] == 1
    assert b.num_scheduled_tokens['second'] == 2
    assert b.num_scheduled_tokens['plain'] == 4
    assert not b.preempted_req_ids
    scheduler.update_from_output(a, _batch_output(a))
    assert scheduler.requests['plain'].num_output_tokens == 0
    assert scheduler.cfg_confirmed_positions['first'] == 3
    scheduler.update_from_output(b, _batch_output(b))
    assert scheduler.requests['plain'].num_output_tokens == 1


def test_pending_token_pressure_does_not_admit_half_pair(monkeypatch):
    scheduler = async_scheduler(monkeypatch, max_num_seqs=4, token_budget=6)
    scheduler.add_request(_request('first'))
    a = scheduler.schedule()
    scheduler.add_request(_request('second'))
    b = scheduler.schedule()
    assert b.total_num_scheduled_tokens == 2 and 'second' not in b.num_scheduled_tokens
    assert len(scheduler.waiting) == 2
    scheduler.update_from_output(a, _batch_output(a))
    scheduler.update_from_output(b, _batch_output(b))


def test_second_branch_reservation_failure_rolls_back_only_new_pair(monkeypatch):
    scheduler = async_scheduler(monkeypatch, max_num_seqs=5)
    scheduler.add_request(_request('first'))
    a = scheduler.schedule()
    scheduler.add_request(_request('new'))
    pair = scheduler.cfg_pairs['new']
    manager = scheduler.kv_cache_manager
    free = manager.block_pool.get_num_free_blocks()
    live_blocks = manager.get_blocks('first').get_block_ids()
    original = manager.allocate_slots

    def allocate(request, *args, **kwargs):
        if request.request_id == pair.unconditional_id:
            return None
        return original(request, *args, **kwargs)

    monkeypatch.setattr(manager, 'allocate_slots', allocate)
    b = scheduler.schedule()
    assert 'new' not in b.num_scheduled_tokens
    assert manager.block_pool.get_num_free_blocks() == free
    assert manager.get_blocks('first').get_block_ids() == live_blocks
    assert all(not any(manager.get_blocks(rid).get_block_ids()) for rid in pair.physical_ids)
    scheduler.update_from_output(a, _batch_output(a))
    scheduler.update_from_output(b, _batch_output(b))


def test_cache_pressure_keeps_retired_blocks_unavailable_until_cleanup(monkeypatch):
    scheduler = async_scheduler(monkeypatch, max_num_seqs=4, num_blocks=10)
    manager = scheduler.kv_cache_manager
    free = manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_request('first'))
    a, b = scheduler.schedule(), scheduler.schedule()
    old_blocks = manager.get_blocks('first').get_block_ids()
    scheduler.finish_requests('first', RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(_request('second'))
    scheduler.update_from_output(a, _batch_output(a))
    cleanup = scheduler.schedule()
    assert not cleanup.num_scheduled_tokens
    scheduler.update_from_output(b, _batch_output(b))
    assert manager.block_pool.get_num_free_blocks() < free
    scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert manager.block_pool.get_num_free_blocks() == free
    next_step = scheduler.schedule()
    assert 'second' in next_step.num_scheduled_tokens
    assert old_blocks  # Actual block identities were allocated and quarantined.


@pytest.mark.parametrize('allowance', [1, 2, 4])
def test_exact_generation_caps_and_empty_work(monkeypatch, allowance):
    scheduler = async_scheduler(monkeypatch)
    request = _request('a')
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


def test_terminal_a_with_b_pending_is_public_once_and_cleanup_fenced(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_request('a'))
    a, b = scheduler.schedule(), scheduler.schedule()
    out = scheduler.update_from_output(a, _batch_output(a, stopped={'a'}))
    assert [o.request_id for o in out[0].outputs] == ['a']
    assert out[0].finished_requests == {'a'}
    cleanup = scheduler.schedule()
    late = scheduler.update_from_output(b, _batch_output(b))
    assert not any(client.outputs for client in late.values())
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() < free
    scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


def test_cancel_before_dispatch_and_repeated_abort(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request('a'))
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    assert scheduler.finish_requests('a', RequestStatus.FINISHED_ABORTED) == [('a', 0)]
    assert scheduler.finish_requests('a', RequestStatus.FINISHED_ABORTED) == []
    cleanup = scheduler.schedule()
    assert cleanup.total_num_scheduled_tokens == 0 and len(cleanup.finished_req_ids) == 2
    result = scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert [o.request_id for o in result[0].outputs] == ['a']
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    scheduler.add_request(_request('a'))


@pytest.mark.parametrize(
    'corruption', ['missing_partner', 'wrong_row', 'extra_row', 'placeholder', 'no_output', 'count']
)
def test_malformed_result_latches_before_partial_application(monkeypatch, corruption):
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request('a'))
    step = scheduler.schedule()
    output = _batch_output(step)
    if corruption == 'missing_partner':
        output.req_ids.pop()
    elif corruption == 'wrong_row':
        output.req_id_to_index = {rid: 0 for rid in output.req_ids}
    elif corruption == 'extra_row':
        output.req_ids.append('unknown')
    elif corruption == 'placeholder':
        scheduler.requests['a'].num_output_placeholders += 1
    elif corruption == 'no_output':
        output.sampled_token_ids = []
    else:
        step.num_scheduled_tokens['a'] = 99
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    with pytest.raises(Exception):
        scheduler.update_from_output(step, output)
    assert all(r.num_output_tokens == 0 for r in scheduler.requests.values())
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    assert scheduler.processed_step_seq == 0
    with pytest.raises(Exception, match='unusable'):
        scheduler.schedule()


def test_result_out_of_order_is_fatal(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request('a'))
    a, b = scheduler.schedule(), scheduler.schedule()
    with pytest.raises(Exception, match='oldest'):
        scheduler.update_from_output(b, _batch_output(b))
    assert scheduler.processed_step_seq == 0


def test_preemption_fails_before_resource_mutation(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request('a'))
    step = scheduler.schedule()
    request = scheduler.requests['a']
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    with pytest.raises(Exception, match='cannot preempt'):
        scheduler._preempt_request(request, 0)
    assert request.num_computed_tokens == 3
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


def test_full_history_bounds_and_oversize_reservations(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    request = _request('too_long', prompt=[0] * 62)
    with pytest.raises(ValueError, match='max_model_len'):
        scheduler.add_request(request)
    assert not scheduler.requests
    tiny = async_scheduler(monkeypatch, num_blocks=4)
    with pytest.raises(ValueError, match='cache'):
        tiny.add_request(_request('too_big'))
    assert not tiny.requests


@pytest.mark.parametrize(
    'feature',
    ['sync', 'graphs', 'chunked', 'prefix', 'tp', 'pp', 'speculation', 'kv_transfer', 'ec_transfer', 'v2_runner'],
)
def test_async_config_rejects_invalid_policies_and_excluded_features(monkeypatch, feature):
    import copy

    scheduler = async_scheduler(monkeypatch)
    config = copy.deepcopy(scheduler.vllm_config)
    if feature == 'sync':
        config.scheduler_config.async_scheduling = False
    elif feature == 'graphs':
        config.model_config.enforce_eager = False
    elif feature == 'chunked':
        config.scheduler_config.enable_chunked_prefill = True
    elif feature == 'prefix':
        config.cache_config.enable_prefix_caching = True
    elif feature == 'tp':
        config.parallel_config.tensor_parallel_size = 2
    elif feature == 'pp':
        config.parallel_config.pipeline_parallel_size = 2
    elif feature == 'speculation':
        config.speculative_config = object()
    elif feature == 'kv_transfer':
        config.kv_transfer_config = object()
    elif feature == 'ec_transfer':
        config.ec_transfer_config = object()
    else:
        config.use_v2_model_runner = True
    with pytest.raises(ValueError):
        scheduler._validate_config(config)
    assert scheduler.vllm_config.scheduler_config.async_scheduling is True


def test_default_topology_remains_synchronous():
    from easymagpie_vllm_omni.pipeline import EASYMAGPIE_CFG_PIPELINE

    assert (
        EASYMAGPIE_CFG_PIPELINE.stages[0].scheduler_cls == 'easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGScheduler'
    )


def test_exact_full_history_limit_does_not_submit_extra_decode(monkeypatch):
    scheduler = async_scheduler(monkeypatch, token_budget=128, num_blocks=128)
    scheduler.add_request(_request('a', prompt=[0] * 60))
    a, b = scheduler.schedule(), scheduler.schedule()
    assert a.num_scheduled_tokens['a'] == 60 and b.num_scheduled_tokens['a'] == 1
    scheduler.update_from_output(a, _batch_output(a))
    scheduler.update_from_output(b, _batch_output(b))
    c, d = scheduler.schedule(), scheduler.schedule()
    assert c.num_scheduled_tokens['a'] == d.num_scheduled_tokens['a'] == 1
    assert scheduler.requests['a'].num_computed_tokens == 63
    scheduler.update_from_output(c, _batch_output(c))
    empty = scheduler.schedule()
    assert not empty.num_scheduled_tokens
    scheduler.update_from_output(d, _batch_output(d))
    assert not scheduler.requests


def test_scheduler_application_error_keeps_resources_and_latches(monkeypatch):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request('a'))
    step = scheduler.schedule()
    held = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    original = AsyncScheduler._update_request_with_output
    count = []

    def fail_second(self, request, tokens):
        count.append(request.request_id)
        if len(count) == 2:
            raise RuntimeError('injected accounting failure')
        return original(self, request, tokens)

    monkeypatch.setattr(AsyncScheduler, '_update_request_with_output', fail_second)
    with pytest.raises(RuntimeError, match='injected accounting failure'):
        scheduler.update_from_output(step, _batch_output(step))
    assert scheduler.processed_step_seq == 0
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    assert scheduler.requests['a'].num_output_tokens == 1  # No unsafe rollback claim.
    with pytest.raises(Exception, match='unusable'):
        scheduler.add_request(_request('fresh'))
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)  # Logical teardown is permitted.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held


def test_successor_uses_distinct_actual_blocks_before_old_cleanup(monkeypatch):
    scheduler = async_scheduler(monkeypatch, max_num_seqs=4)
    manager = scheduler.kv_cache_manager
    scheduler.add_request(_request('old'))
    a, b = scheduler.schedule(), scheduler.schedule()
    old_ids = scheduler.cfg_pairs['old'].physical_ids
    old_blocks = [set(sum(manager.get_blocks(rid).get_block_ids(), [])) for rid in old_ids]
    scheduler.finish_requests('old', RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(a, _batch_output(a))
    scheduler.add_request(_request('new'))
    cleanup_and_prefill = scheduler.schedule()
    new_ids = scheduler.cfg_pairs['new'].physical_ids
    new_blocks = [set(sum(manager.get_blocks(rid).get_block_ids(), [])) for rid in new_ids]
    assert all(x.isdisjoint(y) for x in old_blocks for y in new_blocks)
    assert cleanup_and_prefill.finished_req_ids == set(old_ids)
    scheduler.update_from_output(b, _batch_output(b))
    scheduler.update_from_output(cleanup_and_prefill, _batch_output(cleanup_and_prefill))
    assert not scheduler._cfg_retiring


def test_sequence_slots_remain_atomic_while_first_result_pending(monkeypatch):
    scheduler = async_scheduler(monkeypatch, max_num_seqs=2)
    scheduler.add_request(_request('first'))
    a = scheduler.schedule()
    scheduler.add_request(_request('waiting'))
    b = scheduler.schedule()
    assert set(b.num_scheduled_tokens) == set(scheduler.cfg_pairs['first'].physical_ids)
    assert all(scheduler.requests[rid].num_computed_tokens == 0 for rid in scheduler.cfg_pairs['waiting'].physical_ids)
    scheduler.update_from_output(a, _batch_output(a))
    scheduler.update_from_output(b, _batch_output(b))


def test_allocator_pressure_cannot_preempt_half_of_submitted_pair(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    scheduler.add_request(_request('a'))
    a = scheduler.schedule()
    manager = scheduler.kv_cache_manager
    held = manager.block_pool.get_num_free_blocks()
    monkeypatch.setattr(manager, 'allocate_slots', lambda *a, **k: None)
    with pytest.raises(Exception, match='cannot preempt'):
        scheduler.schedule()
    assert scheduler._cfg_failure is not None
    assert manager.block_pool.get_num_free_blocks() == held
    assert all(r.num_computed_tokens == 3 for r in scheduler.requests.values())
    assert len(scheduler._cfg_steps) == 1  # No successor batch can reach dispatch.
