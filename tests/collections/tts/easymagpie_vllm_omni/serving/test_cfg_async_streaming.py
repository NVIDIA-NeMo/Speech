# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async CFG streaming protocol: native two-step lookahead with exact synchronous commits.

Admit through the existing synchronous input validator in this CPU harness. Streaming
histories keep the native lookahead; acoustic EOS ends the utterance, so speculative
work is only ever discarded for finished streams. These tests do not assert
model/codec numerical equivalence.
"""
import json
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import pytest
from easymagpie_vllm_omni.cfg_scheduler import EasyMagpieCFGScheduler
from test_cfg_async_scheduler import async_scheduler
from test_cfg_concurrency import _batch_output, _mixed_scheduler
from test_cfg_engine_queue import engine
from test_cfg_scheduler import _request
from test_cfg_stream_reference import _FIXTURE, _segment_request
from test_cfg_streaming import _chunk
from vllm.v1.request import RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information


@pytest.mark.parametrize('count', [1, 2, 8, 32])
def test_native_stream_cohort_drains_bounded_fifo_and_retires_every_history(monkeypatch, count):
    segments = json.loads(_FIXTURE.read_text())['segments']
    scheduler = async_scheduler(monkeypatch, max_num_seqs=2 * count, num_blocks=2048)
    baseline = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    expected = set()
    for index in range(count):
        rid = f'cohort-{index}'
        for segment in segments:
            request = _segment_request(segment)
            request.request_id = request.external_req_id = rid
            _admit(scheduler, request)
        _admit(scheduler, _request(rid, prompt=[0], info={}, resumable=False))
        expected.add(rid)
    pending, generated = deque(), Counter()
    maximum_depth = 0
    for _ in range(256):
        while len(pending) < 2:
            step = scheduler.schedule()
            pending.append(step)
            # Lookahead: at most the two batches the native queue holds.
            assert all(request.num_output_placeholders <= 2 for request in scheduler.requests.values())
        maximum_depth = max(maximum_depth, len(pending))
        step = pending.popleft()
        output = scheduler.update_from_output(step, _batch_output(step))
        for client in output.values():
            for request in client.outputs:
                assert request.request_id in expected
                generated[request.request_id] += len(request.new_token_ids)
        if not scheduler.requests and not scheduler._cfg_retiring:
            break
    else:
        pytest.fail('The bounded native cohort did not retire')
    while pending:
        step = pending.popleft()
        scheduler.update_from_output(step, _batch_output(step))
    assert maximum_depth == 2
    assert set(generated) == expected and set(generated.values()) == {10}
    assert not scheduler.cfg_pairs and not scheduler._cfg_streams and not scheduler._cfg_steps
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == baseline


def test_mixed_stream_prefill_decode_keeps_consumed_metadata_and_histories_distinct(monkeypatch):
    scheduler = async_scheduler(monkeypatch, max_num_seqs=4, num_blocks=128)
    _admit(scheduler, _chunk(initial=True))
    holder = scheduler.cfg_pairs['stream']
    first = scheduler.schedule()
    scheduler.update_from_output(first, _batch_output(first))
    before = _state(scheduler, holder)
    follower = _chunk(initial=True)
    follower.request_id = follower.external_req_id = 'follower'
    _admit(scheduler, follower)
    pair = scheduler.cfg_pairs['follower']
    step = scheduler.schedule()
    assert {step.num_scheduled_tokens[rid] for rid in holder.physical_ids} == {1}
    assert {step.num_scheduled_tokens[rid] for rid in pair.physical_ids} == {3}
    # Only the holder's actual scheduled step may advance its positions.
    assert _state(scheduler, holder) != before
    assert all(
        scheduler.requests[rid].num_output_placeholders == 1 for rid in (*holder.physical_ids, *pair.physical_ids)
    )
    for request_id in (*holder.physical_ids, *pair.physical_ids):
        info = deserialize_additional_information(scheduler.requests[request_id].additional_information)
        assert info['text_token'] == [2, 3]
        assert info['_cfg_pair']['logical_id'] == ('stream' if request_id in holder.physical_ids else 'follower')
    # The holder is at its segment limit; only the follower's next frame is submitted.
    successor = scheduler.schedule()
    assert successor.num_scheduled_tokens == dict.fromkeys(pair.physical_ids, 1)
    scheduler.update_from_output(step, _batch_output(step))
    scheduler.update_from_output(successor, _batch_output(successor))
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert not scheduler.requests and not scheduler.cfg_pairs and not scheduler._cfg_retiring


def test_stream_successor_shares_lookahead_batch_and_resumes_after_segment_limit(monkeypatch):
    scheduler = async_scheduler(monkeypatch, max_num_seqs=3)
    _admit(scheduler, _chunk(initial=True))
    pair = scheduler.cfg_pairs['stream']
    first = scheduler.schedule()
    _admit(scheduler, _chunk())
    scheduler.add_request(_request('plain', info={'speaker_id': 'eng', 'text_tokens': [2]}))
    second = scheduler.schedule()
    assert set(second.num_scheduled_tokens) == {*pair.physical_ids, 'plain'}
    assert all(scheduler.requests[rid].num_output_placeholders == 2 for rid in pair.physical_ids)
    assert all(scheduler.requests[rid].status == RequestStatus.RUNNING for rid in pair.physical_ids)
    assert all(len(scheduler.requests[rid].streaming_queue) == 1 for rid in pair.physical_ids)
    assert scheduler._cfg_streams['stream'].text_tokens == [2, 3, 4, 5]
    with pytest.raises(Exception, match='two unconsumed'):
        scheduler.schedule()
    scheduler.update_from_output(first, _batch_output(first))
    # The stream's segment limit is known: nothing is submitted past it.
    third = scheduler.schedule()
    assert set(third.num_scheduled_tokens) == {'plain'}
    scheduler.update_from_output(second, _batch_output(second))
    for rid in pair.physical_ids:
        request = scheduler.requests[rid]
        assert request.status == RequestStatus.WAITING and not request.streaming_queue
        assert request.num_output_placeholders == request.async_tokens_to_discard == 0
    scheduler.update_from_output(third, _batch_output(third))
    resumed = scheduler.schedule()
    assert {rid: resumed.num_scheduled_tokens[rid] for rid in pair.physical_ids} == dict.fromkeys(pair.physical_ids, 1)
    assert 'plain' in resumed.num_scheduled_tokens
    scheduler.update_from_output(resumed, _batch_output(resumed))


def test_stream_lookahead_keeps_physical_slots_until_the_stream_pauses(monkeypatch):
    scheduler = async_scheduler(monkeypatch, max_num_seqs=2)
    _admit(scheduler, _chunk(initial=True))
    pair = scheduler.cfg_pairs['stream']
    first = scheduler.schedule()
    scheduler.add_request(_request('waiting'))
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == dict.fromkeys(pair.physical_ids, 1) and not second.scheduled_new_reqs
    assert scheduler.requests['waiting'].num_computed_tokens == 0
    scheduler.update_from_output(first, _batch_output(first))
    assert len(scheduler.running) == 2
    scheduler.update_from_output(second, _batch_output(second))
    # A paused stream keeps its cache reservation but releases its sequence slots.
    assert all(scheduler.requests[rid].status == RequestStatus.WAITING_FOR_STREAMING_REQ for rid in pair.physical_ids)
    step = scheduler.schedule()
    assert set(step.num_scheduled_tokens) == set(scheduler.cfg_pairs['waiting'].physical_ids)


@pytest.mark.parametrize('queued', [False, True])
def test_resume_has_no_guessed_discard_and_keeps_original_prompt_boundary(monkeypatch, queued):
    scheduler = async_scheduler(monkeypatch)
    _admit(scheduler, _chunk(initial=True))
    pair = scheduler.cfg_pairs['stream']
    if queued:
        _admit(scheduler, _chunk())
    for _ in range(2):
        step = scheduler.schedule()
        scheduler.update_from_output(step, _batch_output(step))
    if not queued:
        assert all(r.status == RequestStatus.WAITING_FOR_STREAMING_REQ for r in scheduler.requests.values())
        _admit(scheduler, _chunk())
    assert all(r.num_computed_tokens == 4 and r.num_output_placeholders == 0 for r in scheduler.requests.values())
    assert all(r.async_tokens_to_discard == 0 for r in scheduler.requests.values())
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == dict.fromkeys(pair.physical_ids, 1)
    result = scheduler.update_from_output(step, _batch_output(step))
    assert result[0].outputs[0].new_token_ids == [0]
    for request in scheduler.requests.values():
        info = deserialize_additional_information(request.additional_information)
        assert info['text_token'] == [4, 5] and info['text_token_start'] == 2
        assert info['_stream_prompt_len'] == 3
        assert request.num_computed_tokens == 5
        assert request.num_output_tokens == 1 and request.num_output_placeholders == 0


def test_confirmation_does_not_reset_unrelated_discard_counter(monkeypatch):
    scheduler = async_scheduler(monkeypatch, max_num_seqs=3)
    _admit(scheduler, _chunk(initial=True))
    first = scheduler.schedule()
    scheduler.add_request(_request('unrelated', info={'speaker_id': 'eng', 'text_tokens': [2]}))
    scheduler.requests['unrelated'].async_tokens_to_discard = 7
    scheduler.update_from_output(first, _batch_output(first))
    assert scheduler.requests['unrelated'].async_tokens_to_discard == 7


def test_successor_past_segment_limit_fails_before_result_mutation(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    _admit(scheduler, _chunk(initial=True))
    first = scheduler.schedule()
    last = scheduler.schedule()
    scheduler.update_from_output(first, _batch_output(first))
    # Fault injection: a successor past the segment limit would carry its model
    # state into the next segment; Omni's position rewind cannot hide it.
    scheduler._cfg_steps.append(scheduler._cfg_steps[0])
    for request in scheduler.requests.values():
        request.num_output_placeholders = 2
    before = _state(scheduler, scheduler.cfg_pairs['stream'])
    with pytest.raises(Exception, match='unconfirmed'):
        scheduler.update_from_output(last, _batch_output(last))
    assert _state(scheduler, scheduler.cfg_pairs['stream']) == before
    assert scheduler.processed_step_seq == 1
    with pytest.raises(Exception, match='unusable'):
        scheduler.schedule()


def test_second_branch_resume_failure_latches_and_fences_resources(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    _admit(scheduler, _chunk(initial=True))
    for _ in range(2):
        step = scheduler.schedule()
        scheduler.update_from_output(step, _batch_output(step))
    hidden = scheduler.cfg_pairs['stream'].unconditional_id
    held = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    original = OmniARScheduler._update_request_as_session

    def fail_second(self, session, update):
        if session.request_id == hidden:
            raise RuntimeError('second branch resume failed')
        return original(self, session, update)

    monkeypatch.setattr(OmniARScheduler, '_update_request_as_session', fail_second)
    with pytest.raises(RuntimeError, match='second branch resume failed'):
        _admit(scheduler, _chunk())
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    assert not scheduler.requests and scheduler._cfg_retiring
    with pytest.raises(Exception, match='unusable'):
        scheduler.schedule()


def test_native_engine_overlaps_stream_steps_and_resumes_after_segment_limit(monkeypatch):
    core = engine(monkeypatch)
    scheduler, executor = core.scheduler, core.model_executor
    _admit(scheduler, _chunk(initial=True))
    pair = scheduler.cfg_pairs['stream']
    assert core.step_fn() == (None, True)
    with ThreadPoolExecutor(max_workers=1) as thread:
        pending = thread.submit(core.step_fn)
        assert executor.sampling[0].entered.wait(5)
        # The segment's last frame is submitted while the stream's first result is unconsumed.
        assert executor.steps[1].num_scheduled_tokens == dict.fromkeys(pair.physical_ids, 1)
        _admit(scheduler, _chunk())
        executor.sampling[0].set_result(_batch_output(executor.steps[0]))
        outputs, executed = pending.result(timeout=5)
    assert executed and outputs[0].outputs[0].request_id == 'stream'
    with ThreadPoolExecutor(max_workers=1) as thread:
        pending = thread.submit(core.step_fn)
        assert executor.sampling[1].entered.wait(5)
        # Nothing is submitted past the known segment limit, and no worker acknowledgement is awaited.
        assert not executor.steps[2].total_num_scheduled_tokens
        executor.sampling[1].set_result(_batch_output(executor.steps[1]))
        outputs, _ = pending.result(timeout=5)
    assert outputs[0].outputs[0].is_segment_finished
    core.step_fn()  # The queued text resumes both histories while the empty step drains.
    assert executor.steps[3].num_scheduled_tokens == dict.fromkeys(pair.physical_ids, 1)
    assert scheduler.processed_step_seq == 3


@pytest.mark.parametrize('queued', [False, True])
@pytest.mark.parametrize('early_stop', [False, True])
def test_internal_async_protocol_matches_fixed_synchronous_reference(monkeypatch, queued, early_stop):
    expected = _protocol_trace(_mixed_scheduler(), queued=queued, early_stop=early_stop)
    scheduler = async_scheduler(monkeypatch)
    actual = _protocol_trace(scheduler, queued=queued, early_stop=early_stop)
    assert actual == expected
    # The equivalence holds with real lookahead, not with an accidentally serialized schedule.
    assert scheduler.cfg_step_stats.overlapped_steps > 0


def test_waiting_admission_also_excludes_unconfirmed_histories(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    _admit(scheduler, _chunk(initial=True))
    scheduler.schedule()
    pair = scheduler.cfg_pairs['stream']
    # Fault fixture: a retained owner appears on the resumed admission path
    # before confirmation. It must not emit new/resumed worker metadata.
    for request in list(scheduler.running):
        request.status = RequestStatus.WAITING
        scheduler.waiting.add_request(request)
    scheduler.running.clear()
    before = _state(scheduler, pair)
    empty = scheduler.schedule()
    assert not empty.num_scheduled_tokens and not empty.scheduled_new_reqs
    assert not empty.scheduled_cached_reqs.req_ids
    assert _state(scheduler, pair) == before


def test_resume_rejects_pending_history_before_native_rewind(monkeypatch):
    from vllm.v1.request import StreamingUpdate

    scheduler = async_scheduler(monkeypatch)
    _admit(scheduler, _chunk(initial=True))
    scheduler.schedule()
    pair = scheduler.cfg_pairs['stream']
    before = _state(scheduler, pair)
    update = StreamingUpdate.from_request(_chunk())
    with pytest.raises(Exception, match='unconfirmed'):
        scheduler._update_request_as_session(scheduler.requests['stream'], update)
    assert _state(scheduler, pair) == before
    with pytest.raises(Exception, match='unusable'):
        scheduler.schedule()


def test_second_branch_queue_failure_latches_and_retains_pending_resources(monkeypatch):
    scheduler = async_scheduler(monkeypatch)
    _admit(scheduler, _chunk(initial=True))
    scheduler.schedule()
    hidden = scheduler.cfg_pairs['stream'].unconditional_id
    held = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    original = OmniARScheduler.add_request

    def fail_second(self, request):
        if request.request_id == hidden:
            raise RuntimeError('second branch queue failed')
        return original(self, request)

    monkeypatch.setattr(OmniARScheduler, 'add_request', fail_second)
    with pytest.raises(RuntimeError, match='second branch queue failed'):
        _admit(scheduler, _chunk())
    assert not scheduler.requests and scheduler._cfg_retiring
    assert len(scheduler._cfg_steps) == 1
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    with pytest.raises(Exception, match='unusable'):
        scheduler.schedule()


def test_pair_stays_gated_until_both_output_histories_are_applied(monkeypatch):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    scheduler = async_scheduler(monkeypatch)
    _admit(scheduler, _chunk(initial=True))
    step = scheduler.schedule()
    pair = scheduler.cfg_pairs['stream']
    original = AsyncScheduler._update_request_with_output
    seen = []

    def observe(self, request, tokens):
        result = original(self, request, tokens)
        seen.append(request.request_id)
        assert all(not scheduler._can_schedule_history(scheduler.requests[rid]) for rid in pair.physical_ids)
        return result

    monkeypatch.setattr(AsyncScheduler, '_update_request_with_output', observe)
    scheduler.update_from_output(step, _batch_output(step))
    assert set(seen) == set(pair.physical_ids)
    assert all(scheduler._can_schedule_history(scheduler.requests[rid]) for rid in pair.physical_ids)


@pytest.mark.parametrize('parked', [False, True])
def test_cancelled_stream_keeps_cleanup_fence_with_empty_batch_pending(monkeypatch, parked):
    scheduler = async_scheduler(monkeypatch)
    _admit(scheduler, _chunk(initial=True))
    free_before_admission = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    first = scheduler.schedule()
    held = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    if parked:
        scheduler.update_from_output(first, _batch_output(first))
        step = scheduler.schedule()
        scheduler.update_from_output(step, _batch_output(step))
    else:
        _admit(scheduler, _chunk())
    # Parked: nothing to run. Live: the stream's lookahead successor is in flight.
    second = scheduler.schedule()
    assert bool(second.num_scheduled_tokens) is not parked
    scheduler.finish_requests('stream', RequestStatus.FINISHED_ABORTED)
    assert not scheduler._cfg_streams and not scheduler.requests
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    if not parked:
        scheduler.update_from_output(first, _batch_output(first))
    cleanup = scheduler.schedule()
    assert len(cleanup.finished_req_ids) == 2
    scheduler.update_from_output(second, _batch_output(second))
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before_admission
    assert not scheduler._cfg_retiring and not scheduler.deferred_frees


def test_running_stream_demand_counts_before_admitting_another_whole_pair(monkeypatch):
    scheduler = async_scheduler(monkeypatch, max_num_seqs=4, token_budget=6)
    _admit(scheduler, _chunk(initial=True))
    first = scheduler.schedule()
    _admit(scheduler, _chunk(initial=True, request_id='other'))
    # FCFS: the running stream's next frame takes its share of the budget first.
    second = scheduler.schedule()
    assert set(second.num_scheduled_tokens) == set(scheduler.cfg_pairs['stream'].physical_ids)
    assert second.total_num_scheduled_tokens == 2
    assert len(scheduler.running) == 2
    scheduler.update_from_output(first, _batch_output(first))
    # At its segment limit the stream has no demand, so the whole other pair fits.
    third = scheduler.schedule()
    assert set(third.num_scheduled_tokens) == set(scheduler.cfg_pairs['other'].physical_ids)
    assert third.total_num_scheduled_tokens == 6
    scheduler.update_from_output(second, _batch_output(second))
    scheduler.update_from_output(third, _batch_output(third))
    fourth = scheduler.schedule()
    assert fourth.total_num_scheduled_tokens == 2
    assert set(fourth.num_scheduled_tokens) == set(scheduler.cfg_pairs['other'].physical_ids)


def test_invalid_and_duplicate_updates_leave_pending_pair_coherent(monkeypatch):
    from vllm_omni.engine.serialization import serialize_additional_information

    scheduler = async_scheduler(monkeypatch)
    _admit(scheduler, _chunk(initial=True))
    first = scheduler.schedule()
    _admit(scheduler, _chunk())
    pair = scheduler.cfg_pairs['stream']
    before = _state(scheduler, pair)
    _admit(scheduler, _chunk())
    invalid = _chunk()
    invalid.additional_information = serialize_additional_information({'text_token': [8], 'text_token_start': 3})
    with pytest.raises(ValueError, match='Conflicting'):
        _admit(scheduler, invalid)
    assert _state(scheduler, pair) == before
    assert all(len(r.streaming_queue) == 1 for r in scheduler.requests.values())
    assert scheduler._cfg_failure is None
    assert scheduler._cfg_streams['stream'].submitted_tokens == 4
    scheduler.update_from_output(first, _batch_output(first))
    assert scheduler.schedule().total_num_scheduled_tokens == 2


@pytest.mark.parametrize('frames', [0, 2, 4, 6])
@pytest.mark.parametrize('delayed_sender', [False, True])
def test_internal_async_parked_sentinel_flushes_codec_before_retirement(monkeypatch, frames, delayed_sender):
    import torch
    from easymagpie_vllm_omni.cfg_pairing import CFGPairingTransferFilter
    from test_stage_processors import _native_sender, _output

    scheduler = async_scheduler(monkeypatch)
    _admit(scheduler, _chunk(initial=True))
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    for _ in range(2):
        step = scheduler.schedule()
        scheduler.update_from_output(step, _batch_output(step))
    session = scheduler.requests['stream']
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    held = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    sender, payloads = _native_sender()
    scheduler.chunk_transfer_adapter = CFGPairingTransferFilter(sender)
    for value in range(1, frames + 1):
        sender.save_async(_output(value), session, is_segment_finished=value == frames)
    if not delayed_sender:
        while sender._pending_save_reqs:
            sender._send_single_request(sender._pending_save_reqs.popleft())
    _admit(scheduler, _request('stream', prompt=[0], info={}, resumable=False))
    assert not scheduler.requests and scheduler._cfg_retiring
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == held
    while sender._pending_save_reqs:
        sender._send_single_request(sender._pending_save_reqs.popleft())
    audio = [p.codes.audio for p in payloads if p.codes is not None and p.codes.audio is not None]
    actual = torch.cat([a.reshape(-1, 2) for a in audio]) if audio else torch.empty((0, 2), dtype=torch.long)
    expected = torch.tensor([[value, value + 100] for value in range(1, frames + 1)], dtype=torch.long).reshape(-1, 2)
    torch.testing.assert_close(actual, expected)
    assert sum(bool(p.meta.finished) for p in payloads) == 1
    assert payloads[-1].meta.finished
    # This fixture provides the real sender queue, not a receiver/polling loop.
    # Detach it after terminal delivery before scheduling worker cleanup.
    scheduler.chunk_transfer_adapter = None
    cleanup = scheduler.schedule()
    scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert not scheduler._cfg_retiring and not scheduler.deferred_frees
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


@pytest.mark.parametrize('count', [1, 4])
def test_stream_batches_keep_lookahead_and_every_running_history(monkeypatch, count):
    """Regression for the per-history gate that halved streaming batches under async scheduling."""
    scheduler = async_scheduler(monkeypatch, max_num_seqs=2 * count, num_blocks=512)
    for index in range(count):
        _admit(scheduler, _long_stream(f'stream-{index}', frames=8))
    physical = {rid for pair in scheduler.cfg_pairs.values() for rid in pair.physical_ids}
    pending = deque([scheduler.schedule()])
    assert set(pending[0].num_scheduled_tokens) == physical
    successors = []
    for _ in range(16):
        # Each successor is scheduled while its predecessor is still unconsumed.
        assert len(scheduler._cfg_steps) == 1
        step = scheduler.schedule()
        pending.append(step)
        scheduler.update_from_output(pending[0], _batch_output(pending.popleft()))
        if not step.num_scheduled_tokens:
            break
        successors.append(set(step.num_scheduled_tokens))
    while pending:
        scheduler.update_from_output(pending[0], _batch_output(pending.popleft()))
    # Prefill plus seven decode frames fill the eight-frame segment without a gap.
    assert successors == [physical] * 7
    assert all(r.status == RequestStatus.WAITING_FOR_STREAMING_REQ for r in scheduler.requests.values())
    assert all(r.num_output_placeholders == r.async_tokens_to_discard == 0 for r in scheduler.requests.values())
    assert all(r.num_computed_tokens == 10 for r in scheduler.requests.values())
    stats = scheduler.cfg_step_stats
    assert (stats.steps, stats.overlapped_steps) == (8, 7)
    assert stats.scheduled_rows == stats.running_rows == 8 * len(physical)
    assert stats.segment_pauses == count and stats.eos_stream_ends == 0


def test_acoustic_eos_discards_inflight_successor_and_retires_after_fence(monkeypatch):
    from easymagpie_vllm_omni.config import ASYNC_STREAMING_ENV

    monkeypatch.setenv(ASYNC_STREAMING_ENV, '1')
    scheduler = async_scheduler(monkeypatch, max_num_seqs=4, num_blocks=512)
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    _admit(scheduler, _long_stream('stream'))
    _admit(scheduler, _long_stream('other'))
    pair = scheduler.cfg_pairs['stream']
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert set(pair.physical_ids) <= set(second.num_scheduled_tokens)
    outputs = scheduler.update_from_output(first, _batch_output(first, stopped={'stream'}))
    public = {o.request_id: o for o in outputs[0].outputs}
    assert public['stream'].new_token_ids == [1] and not public['stream'].is_segment_finished
    assert public['stream'].finish_reason is not None and public['other'].finish_reason is None
    assert 'stream' not in scheduler._cfg_streams and 'stream' not in scheduler.cfg_pairs
    assert not set(pair.physical_ids) & set(scheduler.requests)
    assert set(pair.physical_ids) <= set(scheduler._cfg_retiring)
    cleanup = scheduler.schedule()
    assert set(pair.physical_ids) <= cleanup.finished_req_ids
    assert set(scheduler.cfg_pairs['other'].physical_ids) <= set(cleanup.num_scheduled_tokens)
    # The speculative successor of the stopped stream is discarded, never published.
    late = scheduler.update_from_output(second, _batch_output(second))
    assert [o.request_id for o in late[0].outputs] == ['other']
    scheduler.update_from_output(cleanup, _batch_output(cleanup))
    assert not set(pair.physical_ids) & set(scheduler._cfg_retiring)
    assert scheduler.cfg_step_stats.eos_stream_ends == 1
    assert scheduler.cfg_step_stats.discarded_rows == 2
    # Late text and the final sentinel for the ended stream are dropped without raising.
    scheduler.add_request(_chunk())
    scheduler.add_request(_request('stream', prompt=[0], info={}, resumable=False))
    assert 'stream' not in scheduler.requests and scheduler._cfg_failure is None
    scheduler.finish_requests('other', RequestStatus.FINISHED_ABORTED)
    for _ in range(4):
        step = scheduler.schedule()
        scheduler.update_from_output(step, _batch_output(step))
    assert not scheduler.requests and not scheduler._cfg_retiring and not scheduler.deferred_frees
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free


def _protocol_trace(scheduler, *, queued, early_stop):
    """Drive the fixed segments and record committed work and public outputs.

    Async keeps up to two batches in flight, like the native engine queue. A scheduled
    row is committed only when its result is applied to a live history; a speculative
    row for a stream that already ended is discarded. Text arrives either queued up
    front or, as a paced client sends it, one segment after each pause.
    """
    segments = json.loads(_FIXTURE.read_text())['segments']
    _admit(scheduler, _segment_request(segments[0]))
    pair = scheduler.cfg_pairs['stream']
    roles = {pair.conditional_id: 'conditional', pair.unconditional_id: 'unconditional'}
    inputs = deque(_segment_request(segment) for segment in segments[1:])
    inputs.append(_request('stream', prompt=[0], info={}, resumable=False))
    depth = 2 if hasattr(scheduler, '_cfg_steps') else 1
    pending, committed, published = deque(), [], []
    stop = early_stop
    for _ in range(64):
        while len(pending) < depth and scheduler.requests:
            step = scheduler.schedule()
            pending.append((step, _scheduled_rows(scheduler, step, roles)))
            if queued:
                while inputs:
                    _admit(scheduler, inputs.popleft())
            if not step.total_num_scheduled_tokens:
                break
        if not pending:
            break
        step, rows = pending.popleft()
        live = [row for rid, row in sorted(rows.items(), key=lambda item: item[1]) if rid in scheduler.requests]
        outputs = scheduler.update_from_output(step, _batch_output(step, stopped={'stream'} if stop else ()))
        stop = stop and 'stream' not in rows
        committed.extend(live)
        published.extend(
            (o.request_id, list(o.new_token_ids), o.finish_reason, o.is_segment_finished)
            for client in outputs.values()
            for o in client.outputs
        )
        parked = scheduler.requests and all(
            scheduler.requests[rid].status == RequestStatus.WAITING_FOR_STREAMING_REQ for rid in pair.physical_ids
        )
        if parked and not queued:
            _admit(scheduler, inputs.popleft())
    else:
        pytest.fail('The fixed stream did not finish')
    assert not scheduler.requests and not scheduler._cfg_streams
    cleanup = scheduler.schedule()
    assert cleanup.finished_req_ids == set(pair.physical_ids)
    # Omni publishes a session's terminal abort at the next applied result.
    outputs = scheduler.update_from_output(cleanup, _batch_output(cleanup))
    published.extend(
        (o.request_id, list(o.new_token_ids), o.finish_reason, o.is_segment_finished)
        for client in outputs.values()
        for o in client.outputs
    )
    if hasattr(scheduler, '_cfg_steps'):
        assert not scheduler._cfg_retiring and not scheduler.deferred_frees
    return committed, published


def _scheduled_rows(scheduler, step, roles):
    """Return each scheduled history's role, start position, token count and text metadata."""
    rows = {}
    for rid, count in step.num_scheduled_tokens.items():
        request = scheduler.requests[rid]
        info = deserialize_additional_information(request.additional_information)
        start = request.num_computed_tokens - count
        rows[rid] = (roles[rid], start, count, info['text_token'], info.get('text_token_start'))
    return rows


def _long_stream(request_id='stream', frames=8):
    """Build an initial text segment that generates ``frames`` acoustic frames."""
    request = _chunk(initial=True, request_id=request_id)
    info = deserialize_additional_information(request.additional_information)
    info.update(text_token=list(range(2, 2 + frames)), max_stream_tokens=frames + 4)
    request.additional_information = serialize_additional_information(info)
    request.max_tokens = request.sampling_params.max_tokens = frames
    return request


def _admit(scheduler, request):
    """Bypass only the public P6 rollout guard in this internal CPU harness."""
    if hasattr(scheduler, '_check_health'):
        scheduler._check_health()
    return EasyMagpieCFGScheduler.add_request(scheduler, request)


def _state(scheduler, pair):
    return {
        rid: (
            scheduler.requests[rid].num_computed_tokens,
            scheduler.requests[rid].num_output_placeholders,
            list(scheduler.requests[rid].prompt_token_ids),
            list(scheduler.requests[rid].output_token_ids),
            deserialize_additional_information(scheduler.requests[rid].additional_information),
            scheduler.kv_cache_manager.get_blocks(rid).get_block_ids(),
        )
        for rid in pair.physical_ids
    }
