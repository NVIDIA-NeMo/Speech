# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired incremental text against the pinned synchronous engine scheduler."""

import pytest
from test_cfg_concurrency import _batch_output, _mixed_scheduler, _plain
from test_cfg_scheduler import _request
from vllm.v1.request import RequestStatus
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information


def test_registered_speaker_cfg_stream_waits_and_resumes_both_histories():
    scheduler = _mixed_scheduler()
    scheduler.add_request(_chunk(initial=True))
    pair = scheduler.cfg_pair
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == dict.fromkeys(pair.physical_ids, 3)
    scheduler.update_from_output(first, _batch_output(first))
    second = scheduler.schedule()
    scheduler.update_from_output(second, _batch_output(second))
    assert all(scheduler.requests[rid].status == RequestStatus.WAITING_FOR_STREAMING_REQ for rid in pair.physical_ids)
    assert not scheduler.schedule().num_scheduled_tokens

    scheduler.add_request(_chunk())
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == dict.fromkeys(pair.physical_ids, 1)
    for rid in pair.physical_ids:
        info = deserialize_additional_information(scheduler.requests[rid].additional_information)
        assert info['text_token'] == [4, 5]
        assert info['text_token_start'] == 2
        assert info['_cfg_pair'] == pair.metadata(rid)['_cfg_pair']
    scheduler.update_from_output(resumed, _batch_output(resumed))
    assert all(scheduler.requests[rid].num_computed_tokens == 5 for rid in pair.physical_ids)


@pytest.mark.parametrize('queued', [False, True])
def test_stream_final_sentinel_completes_both_branches_and_allows_reuse(queued):
    scheduler = _mixed_scheduler()
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_chunk(initial=True))
    pair = scheduler.cfg_pair
    if queued:
        scheduler.add_request(_chunk())
        final = _chunk()
        final.resumable = False
        scheduler.add_request(final)
    for _ in range(4 if queued else 2):
        step = scheduler.schedule()
        outputs = scheduler.update_from_output(step, _batch_output(step))
    if not queued:
        final = _chunk()
        final.resumable = False
        scheduler.add_request(final)
    assert not scheduler.requests and not scheduler.cfg_pairs
    assert not scheduler._cfg_streams
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    assert all(not o.request_id.startswith('__') for o in outputs[0].outputs)
    if queued:
        assert all(not o.is_segment_finished for o in outputs[0].outputs)
    assert set(pair.physical_ids) <= scheduler.schedule().finished_req_ids
    scheduler.add_request(_chunk(initial=True))
    assert len(scheduler.schedule().num_scheduled_tokens) == 2


def test_stream_abort_while_waiting_and_late_output_preserves_other_request():
    scheduler = _mixed_scheduler()
    scheduler.add_request(_chunk(initial=True))
    scheduler.add_request(_plain('plain'))
    for _ in range(2):
        step = scheduler.schedule()
        scheduler.update_from_output(step, _batch_output(step))
    assert scheduler.finish_requests('stream', RequestStatus.FINISHED_ABORTED) == [('stream', 0)]
    scheduler.update_from_output(step, _batch_output(step))
    assert set(scheduler.requests) == {'plain'}
    assert not scheduler._cfg_streams
    with pytest.raises(ValueError):
        scheduler.add_request(_chunk())  # Late continuation cannot resurrect a stream.
    assert set(scheduler.requests) == {'plain'}


def test_duplicate_stream_chunk_does_not_queue_or_advance_histories():
    scheduler = _mixed_scheduler()
    scheduler.add_request(_chunk(initial=True))
    scheduler.add_request(_chunk())
    scheduler.add_request(_chunk())
    assert all(len(r.streaming_queue) == 1 for r in scheduler.requests.values())
    assert scheduler._cfg_streams['stream'].text_tokens == [2, 3, 4, 5]


def test_resume_keeps_reserved_cache_even_when_new_reservations_are_blocked(monkeypatch):
    scheduler = _mixed_scheduler()
    scheduler.add_request(_chunk(initial=True))
    for _ in range(2):
        step = scheduler.schedule()
        scheduler.update_from_output(step, _batch_output(step))
    blocks = {rid: scheduler.kv_cache_manager.get_blocks(rid) for rid in scheduler.requests}
    allocate = scheduler.kv_cache_manager.allocate_slots

    def only_decode(request, num_new_tokens, *args, **kwargs):
        return allocate(request, num_new_tokens, *args, **kwargs) if num_new_tokens == 1 else None

    monkeypatch.setattr(scheduler.kv_cache_manager, 'allocate_slots', only_decode)
    scheduler.add_request(_chunk())
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) == 2
    assert all(scheduler.kv_cache_manager.get_blocks(rid) == value for rid, value in blocks.items())


def test_resumed_pair_waits_atomically_when_only_one_sequence_slot_is_free():
    scheduler = _mixed_scheduler(max_num_seqs=3)
    scheduler.add_request(_chunk(initial=True))
    for _ in range(2):
        step = scheduler.schedule()
        scheduler.update_from_output(step, _batch_output(step))
    scheduler.add_request(_plain('plain-a'))
    scheduler.add_request(_plain('plain-b'))
    step = scheduler.schedule()
    scheduler.update_from_output(step, _batch_output(step))
    scheduler.add_request(_chunk())
    step = scheduler.schedule()
    assert set(step.num_scheduled_tokens) == {'plain-a', 'plain-b'}
    scheduler.update_from_output(step, _batch_output(step, stopped={'plain-a'}))
    step = scheduler.schedule()
    assert 'stream' in step.num_scheduled_tokens and len(step.num_scheduled_tokens) == 3


@pytest.mark.parametrize(
    'info',
    [
        {'text_token': [9], 'text_token_start': 1},
        {'text_token': [9], 'text_token_start': 5},
        {'text_token': [4]},
        {'text_token': [4], 'text_token_start': 2, 'use_cfg': False},
        {'text_token': [4], 'text_token_start': 2, 'cfg_scale': 1.0},
        {'text_token': [4], 'text_token_start': 2, 'speaker_id': 'other'},
        {'text_token': [4], 'text_token_start': 2, '_cfg_pair': {}},
    ],
)
def test_invalid_stream_update_is_atomic_and_preserves_pair(info):
    scheduler = _mixed_scheduler()
    scheduler.add_request(_chunk(initial=True))
    update = _chunk()
    update.additional_information = serialize_additional_information(info)
    with pytest.raises(ValueError):
        scheduler.add_request(update)
    assert len(scheduler.requests) == 2
    assert all(not r.streaming_queue for r in scheduler.requests.values())
    assert scheduler._cfg_streams['stream'].submitted_tokens == 2


@pytest.mark.parametrize('bound', [None, 0, -1, True, 5000])
def test_stream_requires_feasible_explicit_history_bound(bound):
    scheduler = _mixed_scheduler()
    request = _chunk(initial=True)
    info = deserialize_additional_information(request.additional_information)
    info['max_stream_tokens'] = bound
    request.additional_information = serialize_additional_information(info)
    with pytest.raises(ValueError, match='max_stream_tokens'):
        scheduler.add_request(request)
    assert not scheduler.requests


@pytest.mark.parametrize('initial', [False, True])
def test_stream_text_buffer_is_bounded_by_segment_generation_limit(initial):
    scheduler = _mixed_scheduler()
    if not initial:
        scheduler.add_request(_chunk(initial=True))
    request = _chunk(initial=initial)
    info = deserialize_additional_information(request.additional_information)
    info['text_token'] = list(range(100))
    request.additional_information = serialize_additional_information(info)
    with pytest.raises(ValueError, match='text.*generation limit'):
        scheduler.add_request(request)


def test_stream_generation_bound_and_pressure_cover_entire_history():
    scheduler = _mixed_scheduler(max_num_seqs=4, num_blocks=18)
    scheduler.add_request(_chunk(initial=True))
    scheduler.add_request(_chunk(initial=True, request_id='waiting'))
    for i in range(6):
        if i:
            update = _chunk()
            update.additional_information = serialize_additional_information(
                {'text_token': [10 + i, 20 + i], 'text_token_start': i * 2}
            )
            scheduler.add_request(update)
        for _ in range(2):
            step = scheduler.schedule()
            assert len(step.num_scheduled_tokens) == 2 and 'stream' in step.num_scheduled_tokens
            assert not step.preempted_req_ids
            scheduler.update_from_output(step, _batch_output(step))
        assert all(r.async_tokens_to_discard == 0 for r in scheduler.requests.values())
    update = _chunk()
    update.additional_information = serialize_additional_information({'text_token': [], 'text_token_start': 12})
    with pytest.raises(ValueError, match='max_stream_tokens'):
        scheduler.add_request(update)
    scheduler.finish_requests('stream', RequestStatus.FINISHED_ABORTED)
    assert 'waiting' in scheduler.schedule().num_scheduled_tokens


def _chunk(*, initial=False, request_id='stream'):
    info = {'text_token': [2, 3] if initial else [4, 5], 'text_token_start': 0 if initial else 2}
    if initial:
        info.update(speaker_id='eng', use_cfg=True, cfg_scale=2.5, temperature=0.7, top_k=80, max_stream_tokens=12)
    request = _request(request_id, info=info, prompt=None if initial else [0], resumable=True)
    request.max_tokens = request.sampling_params.max_tokens = 2
    return request
