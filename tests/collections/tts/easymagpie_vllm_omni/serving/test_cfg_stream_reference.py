# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed engine segments and synchronous CPU accounting oracle for P6.

Synthetic sampled decisions exercise the real pinned scheduler/cache allocator.
These traces do not establish retained model state or codec/audio equivalence.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
from easymagpie_vllm_omni.serving_stream import EasyMagpieInputStream
from test_cfg_concurrency import _batch_output, _mixed_scheduler
from test_cfg_scheduler import _request
from vllm.sampling_params import SamplingParams
from vllm.v1.request import RequestStatus
from vllm_omni.engine.serialization import deserialize_additional_information

_FIXTURE = Path(__file__).parent / 'fixtures' / 'cfg_stream_segments.json'


@pytest.mark.asyncio
async def test_fixed_engine_segments_match_frontend_without_coalescing():
    fixture = json.loads(_FIXTURE.read_text())
    prompt = copy.deepcopy(fixture['segments'][0]['prompt'])
    for key in ('text_token', 'text_token_start', 'max_stream_tokens', 'text_prefill_num', 'prefill_text_tokens'):
        prompt['additional_information'].pop(key)
    stream = EasyMagpieInputStream(
        prefill_prompt=prompt,
        sampling_params=SamplingParams(max_tokens=10, temperature=0, stop_token_ids=[1], detokenize=False),
        text_eos_id=fixture['text_eos_id'],
        max_new_tokens=fixture['max_stream_tokens'],
        text_prefill_num=fixture['text_prefill_num'],
        coalesce_queued_tokens=False,
        pace_timeout_s=0,
    )
    await stream.put_tokens([2, 3, 4])
    await stream.put_tokens([5, 6])
    await stream.finish()
    actual = [
        dict(prompt=segment.prompt, max_tokens=segment.sampling_params.max_tokens) async for segment in stream.inputs()
    ]
    assert actual == fixture['segments']
    assert sum(segment['max_tokens'] for segment in actual) == fixture['max_stream_tokens']


@pytest.mark.parametrize('queued', [False, True])
@pytest.mark.parametrize('early_stop', [False, True])
def test_synchronous_fixed_segments_repeat_accounting_reference(queued, early_stop):
    """Compare repeat runs against explicit positions; a forced acoustic EOS ends the stream."""
    trace = _reference(queued=queued, early_stop=early_stop)
    assert trace == _reference(queued=queued, early_stop=early_stop)
    expected_positions = [5] if early_stop else list(range(5, 15))
    assert [event['position'] for event in trace if event['event'] == 'confirmed'] == expected_positions
    destination = os.environ.get('P6_TRACE_DIR')
    if destination:
        Path(destination).mkdir(parents=True, exist_ok=True)
        name = f'sync_{"queued" if queued else "parked"}_{"forced_stop" if early_stop else "limits"}.json'
        (Path(destination) / name).write_text(json.dumps(trace, indent=2) + '\n')


def _reference(*, queued, early_stop):
    segments = json.loads(_FIXTURE.read_text())['segments']
    scheduler = _mixed_scheduler()
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    scheduler.add_request(_segment_request(segments[0]))
    pair = scheduler.cfg_pairs['stream']
    trace = []
    blocks = None
    ended = False
    for index, segment in enumerate(segments):
        if ended:
            # Acoustic EOS ended the utterance: later text is dropped, never resumed.
            if not queued:
                scheduler.add_request(_segment_request(segment))
            assert not scheduler.requests and not scheduler._cfg_streams
            continue
        if index and not queued:
            # Several empty polls must retain the original histories/reservation.
            parked_blocks = {
                rid: scheduler.kv_cache_manager.get_blocks(rid).get_block_ids() for rid in pair.physical_ids
            }
            for _ in range(2):
                empty = scheduler.schedule()
                assert not empty.num_scheduled_tokens
                assert all(r.status == RequestStatus.WAITING_FOR_STREAMING_REQ for r in scheduler.requests.values())
                assert all(
                    scheduler.kv_cache_manager.get_blocks(rid).get_block_ids() == parked_blocks[rid]
                    for rid in pair.physical_ids
                )
            scheduler.add_request(_segment_request(segment))
        for token in range(1 if early_stop and index == 0 else segment['max_tokens']):
            step = scheduler.schedule()
            assert set(step.num_scheduled_tokens) == set(pair.physical_ids)
            if blocks is None:
                blocks = {rid: scheduler.kv_cache_manager.get_blocks(rid).get_block_ids() for rid in pair.physical_ids}
            # Native MambaManager retires skipped state blocks to the null block.
            # Live entries must retain their original per-position allocation.
            current = {rid: scheduler.kv_cache_manager.get_blocks(rid).get_block_ids() for rid in pair.physical_ids}
            null = scheduler.kv_cache_manager.block_pool.null_block.block_id
            for rid in pair.physical_ids:
                assert current[rid][0] == blocks[rid][0]
                assert len(current[rid][1]) == len(blocks[rid][1])
                assert all(now in (old, null) for now, old in zip(current[rid][1], blocks[rid][1]))
                assert any(block != null for block in current[rid][1])
            assert set(current[pair.conditional_id][1]).intersection(current[pair.unconditional_id][1]) <= {null}
            for request in scheduler.requests.values():
                info = deserialize_additional_information(request.additional_information)
                expected_info = segment['prompt']['additional_information']
                assert info['text_token'] == expected_info['text_token']
                if expected_info['text_token']:
                    assert info['text_token_start'] == expected_info['text_token_start']
                assert request.max_tokens == segment['max_tokens']
                assert request.num_output_placeholders == 0
            if queued and index == 0 and token == 0:
                before = [_consumed(scheduler.requests[rid]) for rid in pair.physical_ids]
                for update in segments[1:]:
                    scheduler.add_request(_segment_request(update))
                scheduler.add_request(_request('stream', prompt=[0], info={}, resumable=False))
                assert [_consumed(scheduler.requests[rid]) for rid in pair.physical_ids] == before
                assert all(
                    scheduler.kv_cache_manager.get_blocks(rid).get_block_ids() == current[rid]
                    for rid in pair.physical_ids
                )
                stream = scheduler._cfg_streams['stream']
                assert stream.closed and stream.tail_submitted
                assert stream.text_tokens == [2, 3, 4, 5, 6, 9]
                assert stream.submitted_tokens == stream.max_tokens == 10
                assert all(len(r.streaming_queue) == 4 for r in scheduler.requests.values())
                trace.append(
                    dict(event='closed_with_pending_input', queued_updates=4, admitted_text=list(stream.text_tokens))
                )
            position = scheduler.requests['stream'].num_computed_tokens
            outputs = scheduler.update_from_output(
                step, _batch_output(step, stopped={'stream'} if early_stop and index == 0 else ())
            )
            public = outputs[0].outputs
            assert len(public) == 1 and public[0].request_id == 'stream'
            assert public[0].new_token_ids == [1 if early_stop and index == 0 else 0]
            boundary = token + 1 == (1 if early_stop and index == 0 else segment['max_tokens'])
            assert public[0].is_segment_finished == (boundary and bool(scheduler.requests))
            assert (public[0].finish_reason is not None) == boundary
            branches = [_consumed(scheduler.requests[rid]) for rid in pair.physical_ids] if scheduler.requests else []
            if branches:
                assert branches[0] == branches[1]
                assert branches[0]['prompt_boundary'] == 5
                assert all(r.async_tokens_to_discard == 0 for r in scheduler.requests.values())
            trace.append(
                dict(
                    event='confirmed',
                    segment=index,
                    position=position,
                    branches=branches,
                    finish_reason=public[0].finish_reason,
                    segment_finished=public[0].is_segment_finished,
                )
            )
            ended = not scheduler.requests
        # Only acoustic EOS, or the final sentinel queued behind the last segment, ends the stream here.
        assert ended == (early_stop or (queued and index + 1 == len(segments)))
        if not queued and not ended:
            assert all(r.status == RequestStatus.WAITING_FOR_STREAMING_REQ for r in scheduler.requests.values())
    if not queued:
        scheduler.add_request(_request('stream', prompt=[0], info={}, resumable=False))
    assert not scheduler.requests and not scheduler.cfg_pairs and not scheduler._cfg_streams
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    assert scheduler.schedule().finished_req_ids == set(pair.physical_ids)
    scheduler.add_request(_segment_request(segments[0]))
    assert len(scheduler.schedule().num_scheduled_tokens) == 2
    return trace


def _segment_request(segment):
    request = _request(
        'stream',
        prompt=list(segment['prompt']['prompt_token_ids']),
        info=copy.deepcopy(segment['prompt']['additional_information']),
        resumable=True,
    )
    request.max_tokens = request.sampling_params.max_tokens = segment['max_tokens']
    return request


def _consumed(request):
    info = deserialize_additional_information(request.additional_information)
    return dict(
        position=request.num_computed_tokens,
        prompt=list(request.prompt_token_ids),
        outputs=list(request.output_token_ids),
        limit=request.max_tokens,
        text=info['text_token'],
        text_start=info['text_token_start'],
        prompt_boundary=info['_stream_prompt_len'],
        status=request.status.name,
    )
