# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-0 streaming stop policy shared by the EasyMagpie CFG and non-CFG schedulers.

A resumable request pauses at each text-segment limit and resumes with the next text update. Two stops end the
session instead:

* **Acoustic EOS of an utterance stream.** An utterance stream feeds the text of one utterance incrementally (the
  WebSocket input and every CFG stream). The model's end-of-speech decision is final, as in NeMo's
  ``EasyMagpieTTSInferenceModel.streaming_step`` and in complete-text requests. Queued text is dropped, and nothing
  is generated after EOS. A dialogue session instead ends one reply per EOS and resumes its retained state with the
  next reply.
* **The input iterator's final sentinel** at the head of the update queue.

For utterance streams this policy is also what makes async lookahead exact. The scheduler knows each segment limit
and never schedules past it. EOS is the only stop it cannot predict, and EOS is terminal. A speculative successor
step can therefore only target a finished request: its result is discarded, and its state is freed after the
worker-cleanup fence. Nothing a speculative step writes (Mamba, acoustic feedback or sampling state of that stream)
can reach a resumed segment.
"""
from __future__ import annotations

import enum
from typing import Any

__all__ = ["StreamStop", "end_stream_at_terminal_stop", "is_acoustic_eos", "is_late_stream_input"]


class StreamStop(enum.Enum):
    """Why a stopped resumable request ended its utterance instead of pausing."""

    ACOUSTIC_EOS = "acoustic_eos"
    FINAL_INPUT = "final_input"


def end_stream_at_terminal_stop(request: Any, *, eos_ends_stream: bool) -> StreamStop | None:
    """Make a stopped resumable request terminal when its stop ends the session.

    Call before vLLM's ``Scheduler._handle_stopped_request``. That handler then finishes the request, and the
    chunk-transfer adapter publishes a terminal codec payload instead of a segment boundary.

    Args:
        request: Stage-0 request whose last sampled token stopped it.
        eos_ends_stream: Whether the session is an utterance stream, which acoustic EOS ends. False for a dialogue
            session, which pauses after each reply.

    Returns:
        The reason the session ended, or None when the request pauses for its next input.
    """
    if not getattr(request, "resumable", False):
        return None
    queue = getattr(request, "streaming_queue", None)
    if eos_ends_stream and is_acoustic_eos(request):
        request.resumable = False
        if queue:
            queue.clear()
        return StreamStop.ACOUSTIC_EOS
    if queue and queue[0] is None:
        request.resumable = False
        return StreamStop.FINAL_INPUT
    return None


def is_acoustic_eos(request: Any) -> bool:
    """Return whether a request's last sampled token is one of its engine stop tokens.

    Stage 0 samples a two-way continue/stop token for every frame, and the stop token marks acoustic EOS. Inspect the
    token rather than the finish status: vLLM reports a length stop when EOS lands on a segment's last frame.

    Args:
        request: Stage-0 request.

    Returns:
        Whether the last sampled token stops the request as vLLM's ``check_stop`` does.
    """
    output_ids = getattr(request, "output_token_ids", None)
    if not output_ids:
        return False
    last = output_ids[-1]
    params = getattr(request, "sampling_params", None)
    if last in (getattr(params, "stop_token_ids", None) or ()):
        return True
    eos_token_id = getattr(request, "eos_token_id", None)
    return eos_token_id is not None and last == eos_token_id and not getattr(params, "ignore_eos", False)


def is_late_stream_input(request: Any, *, live: bool) -> bool:
    """Return whether an incoming request is leftover input for a stream that is no longer live.

    Text updates and the final input sentinel carry a single placeholder prompt token. A fresh synthesis request
    always carries its speaker and context prefill. The orchestrator turns an update for an unknown request into a
    fresh admission, so such leftovers must be dropped, not synthesized. They arise after an acoustic EOS ended the
    stream, or after an abort raced the input iterator. Raising from ``add_request`` would kill the engine core.

    Args:
        request: Incoming Stage-0 request.
        live: Whether the scheduler still owns a request with this identity.

    Returns:
        Whether the request should be dropped.
    """
    return not live and list(getattr(request, "prompt_token_ids", None) or ()) == [0]
