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


def _stopped(
    last_token: int | None,
    *,
    resumable: bool = True,
    queue: tuple[str | None, ...] = (),
    eos_token_id: int | None = None,
    ignore_eos: bool = False,
) -> SimpleNamespace:
    """Build a stand-in for a stage-0 request that has just stopped.

    Args:
        last_token: Last sampled token, or None to use placeholder zero output tokens.
        resumable: Whether the request is a resumable streaming request.
        queue: Pending streaming updates; None marks the final input sentinel.
        eos_token_id: Model EOS token ID, or None when the model defines none.
        ignore_eos: Whether sampling ignores the model EOS token.

    Returns:
        A request namespace with output tokens, sampling parameters, and a streaming queue; stop token 7 is
        always configured.
    """
    return SimpleNamespace(
        output_token_ids=[0, 0] if last_token is None else [0, last_token],
        sampling_params=SamplingParams(stop_token_ids=[7], ignore_eos=ignore_eos),
        eos_token_id=eos_token_id,
        resumable=resumable,
        streaming_queue=deque(queue),
    )


@pytest.mark.parametrize(
    ("last", "eos_token_id", "ignore_eos", "expected"),
    [
        (7, None, False, True),
        (0, None, False, False),
        (5, 5, False, True),
        (5, 5, True, False),
        (None, None, False, False),
    ],
)
def test_acoustic_eos_matches_the_engine_stop_tokens(
    last: int | None,
    eos_token_id: int | None,
    ignore_eos: bool,
    expected: bool,
) -> None:
    """Verify acoustic EOS is detected from the configured stop tokens and the honored model EOS.

    Args:
        last: Last sampled token, or None for placeholder zero output tokens.
        eos_token_id: Model EOS token ID, or None.
        ignore_eos: Whether sampling ignores the model EOS token.
        expected: Whether the request is expected to stop at acoustic EOS.
    """
    request = _stopped(last, eos_token_id=eos_token_id, ignore_eos=ignore_eos)
    assert is_acoustic_eos(request) is expected


def test_acoustic_eos_requires_a_sampled_token() -> None:
    """Verify a request without sampled output tokens is not treated as acoustic EOS."""
    request = _stopped(7)
    request.output_token_ids = []
    assert not is_acoustic_eos(request)


@pytest.mark.parametrize("queue", [(), ("update",), ("update", None), (None,)])
def test_acoustic_eos_ends_an_utterance_stream_and_drops_queued_input(queue: tuple[str | None, ...]) -> None:
    """Verify acoustic EOS ends an utterance stream, clears resumability, and drops queued input.

    Args:
        queue: Streaming updates pending when the request stops.
    """
    request = _stopped(7, queue=queue)
    assert end_stream_at_terminal_stop(request, eos_ends_stream=True) is StreamStop.ACOUSTIC_EOS
    assert request.resumable is False
    assert not request.streaming_queue


@pytest.mark.parametrize("queue", [(), ("next reply",)])
def test_acoustic_eos_ends_only_the_reply_of_a_dialogue_session(queue: tuple[str | None, ...]) -> None:
    """Verify acoustic EOS in a dialogue session keeps the stream resumable with its queued input.

    Args:
        queue: Streaming updates pending when the request stops.
    """
    request = _stopped(7, queue=queue)
    assert end_stream_at_terminal_stop(request, eos_ends_stream=False) is None
    assert request.resumable is True
    assert list(request.streaming_queue) == list(queue)


@pytest.mark.parametrize("eos_ends_stream", [False, True])
def test_final_sentinel_ends_the_stream_at_a_segment_limit(eos_ends_stream: bool) -> None:
    """Verify a queued final sentinel ends the stream at a segment limit regardless of the EOS policy.

    Args:
        eos_ends_stream: Whether acoustic EOS ends the whole stream.
    """
    request = _stopped(0, queue=(None,))
    assert end_stream_at_terminal_stop(request, eos_ends_stream=eos_ends_stream) is StreamStop.FINAL_INPUT
    assert request.resumable is False


@pytest.mark.parametrize("queue", [(), ("update",), ("update", None)])
def test_segment_limit_keeps_the_stream_resumable(queue: tuple[str | None, ...]) -> None:
    """Verify a segment-limit stop without a leading final sentinel keeps the stream resumable.

    Args:
        queue: Streaming updates pending when the request stops.
    """
    request = _stopped(0, queue=queue)
    assert end_stream_at_terminal_stop(request, eos_ends_stream=True) is None
    assert request.resumable is True
    assert list(request.streaming_queue) == list(queue)


def test_policy_ignores_non_resumable_requests() -> None:
    """Verify the stop policy leaves non-resumable requests and their queues untouched."""
    request = _stopped(7, resumable=False, queue=("update",))
    assert end_stream_at_terminal_stop(request, eos_ends_stream=True) is None
    assert list(request.streaming_queue) == ["update"]


@pytest.mark.parametrize(
    ("prompt", "live", "expected"),
    [([0], False, True), ([0], True, False), ([0, 0, 0], False, False), ([], False, False), ([5], False, False)],
)
def test_late_stream_input_is_a_placeholder_for_a_request_that_is_not_live(
    prompt: list[int],
    live: bool,
    expected: bool,
) -> None:
    """Verify only a single placeholder prompt token for a request that is not live counts as late input.

    Args:
        prompt: Prompt token IDs of the incoming stream input.
        live: Whether the target request is still live.
        expected: Whether the input is expected to be classified as late.
    """
    request = SimpleNamespace(prompt_token_ids=prompt)
    assert is_late_stream_input(request, live=live) is expected
