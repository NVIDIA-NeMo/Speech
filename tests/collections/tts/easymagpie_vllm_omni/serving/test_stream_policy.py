# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-0 streaming stop policy shared by the CFG and non-CFG schedulers."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
from easymagpie_vllm_omni.stream_policy import (
    StreamStop,
    end_stream_at_terminal_stop,
    is_acoustic_eos,
    is_late_stream_input,
)
from vllm import SamplingParams


def _stopped(last_token, *, resumable=True, queue=(), eos_token_id=None, ignore_eos=False):
    return SimpleNamespace(
        output_token_ids=[0, 0] if last_token is None else [0, last_token],
        sampling_params=SamplingParams(stop_token_ids=[7], ignore_eos=ignore_eos),
        eos_token_id=eos_token_id,
        resumable=resumable,
        streaming_queue=deque(queue),
    )


@pytest.mark.parametrize(
    ('last', 'eos_token_id', 'ignore_eos', 'expected'),
    [
        (7, None, False, True),
        (0, None, False, False),
        (5, 5, False, True),
        (5, 5, True, False),
        (None, None, False, False),
    ],
)
def test_acoustic_eos_matches_the_engine_stop_tokens(last, eos_token_id, ignore_eos, expected):
    request = _stopped(last, eos_token_id=eos_token_id, ignore_eos=ignore_eos)
    assert is_acoustic_eos(request) is expected


def test_acoustic_eos_requires_a_sampled_token():
    request = _stopped(7)
    request.output_token_ids = []
    assert not is_acoustic_eos(request)


@pytest.mark.parametrize('queue', [(), ('update',), ('update', None), (None,)])
def test_acoustic_eos_ends_an_utterance_stream_and_drops_queued_input(queue):
    request = _stopped(7, queue=queue)
    assert end_stream_at_terminal_stop(request, eos_ends_stream=True) is StreamStop.ACOUSTIC_EOS
    assert request.resumable is False
    assert not request.streaming_queue


@pytest.mark.parametrize('queue', [(), ('next reply',)])
def test_acoustic_eos_ends_only_the_reply_of_a_dialogue_session(queue):
    request = _stopped(7, queue=queue)
    assert end_stream_at_terminal_stop(request, eos_ends_stream=False) is None
    assert request.resumable is True
    assert list(request.streaming_queue) == list(queue)


@pytest.mark.parametrize('eos_ends_stream', [False, True])
def test_final_sentinel_ends_the_stream_at_a_segment_limit(eos_ends_stream):
    request = _stopped(0, queue=(None,))
    assert end_stream_at_terminal_stop(request, eos_ends_stream=eos_ends_stream) is StreamStop.FINAL_INPUT
    assert request.resumable is False


@pytest.mark.parametrize('queue', [(), ('update',), ('update', None)])
def test_segment_limit_keeps_the_stream_resumable(queue):
    request = _stopped(0, queue=queue)
    assert end_stream_at_terminal_stop(request, eos_ends_stream=True) is None
    assert request.resumable is True
    assert list(request.streaming_queue) == list(queue)


def test_policy_ignores_non_resumable_requests():
    request = _stopped(7, resumable=False, queue=('update',))
    assert end_stream_at_terminal_stop(request, eos_ends_stream=True) is None
    assert list(request.streaming_queue) == ['update']


@pytest.mark.parametrize(
    ('prompt', 'live', 'expected'),
    [([0], False, True), ([0], True, False), ([0, 0, 0], False, False), ([], False, False), ([5], False, False)],
)
def test_late_stream_input_is_a_placeholder_for_a_request_that_is_not_live(prompt, live, expected):
    request = SimpleNamespace(prompt_token_ids=prompt)
    assert is_late_stream_input(request, live=live) is expected
