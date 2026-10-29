# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The streaming probe must detect failures and release its input iterator."""
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from probe_cfg_streaming import run_stream_probe

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.mark.asyncio
@pytest.mark.parametrize("abort", [False, True])
async def test_streaming_probe_requires_delayed_chunks_and_cleans_up(abort: bool) -> None:
    """Verify streaming probe requires delayed chunks and cleans up.

    Args:
        abort: Whether to cancel after the requested boundary.
    """
    engine = _Engine()
    result = await run_stream_probe(engine, _meta(), request_id="public", text="hello", abort_waiting=abort)
    assert result["aborted"] is abort
    assert result["codec_finished"] is not abort
    assert len(result["chunk_offsets"]) == (1 if abort else 4)
    assert engine.closed and engine.aborted


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["private", "missing_codec", "error", "no_tail"])
async def test_streaming_probe_rejects_bad_outputs_and_cleans_up(failure: str | None) -> None:
    """Verify streaming probe rejects bad outputs and cleans up.

    Args:
        failure: Failure mode injected by the fake engine.
    """
    engine = _Engine(failure)
    with pytest.raises(RuntimeError):
        await run_stream_probe(engine, _meta(), request_id="public", text="hello")
    assert engine.closed and engine.aborted


class _Engine:
    """Test double for Engine behavior."""

    def __init__(self, failure: str | None = None) -> None:
        """Initialize Engine state.

        Args:
            failure: Failure mode injected by the fake engine.
        """
        self.failure = failure
        self.aborted = []
        self.closed = False

    async def generate(
        self,
        inputs: AsyncIterator[Any],
        request_id: str,
    ) -> AsyncIterator[SimpleNamespace]:
        """Yield configured public outputs from the test engine.

        Args:
            inputs: Incremental engine input iterator.
            request_id: Public request identity.

        Yields:
            Incremental input or public engine output for the current request.
        """
        try:
            async for chunk in inputs:
                if self.failure == "no_tail" and chunk.prompt["additional_information"]["text_token"] == []:
                    yield SimpleNamespace(request_id=request_id, stage_id=1, finished=True)
                    return
                yield SimpleNamespace(
                    request_id="private" if self.failure == "private" else request_id,
                    stage_id=0,
                    finished=False,
                    error="failed" if self.failure == "error" else None,
                    outputs=[
                        SimpleNamespace(token_ids=[0] * chunk.sampling_params.max_tokens, finish_reason="length")
                    ],
                )
            if self.failure != "missing_codec":
                yield SimpleNamespace(request_id=request_id, stage_id=1, finished=True)
        finally:
            self.closed = True

    async def abort(self, request_id: str) -> None:
        """Record cancellation of the selected test request.

        Args:
            request_id: Public request identity.
        """
        self.aborted.append(request_id)


def _meta() -> SimpleNamespace:
    """Build model metadata for the incremental streaming probe test.

    Returns:
        Model metadata and a tokenizer for the streaming probe test.
    """
    return SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: list(range(8))),
        text_prefill_num=4,
        stop_token_id=1,
        text_eos_id=99,
        prompt_len=6,
        speaker_id="eng",
    )
