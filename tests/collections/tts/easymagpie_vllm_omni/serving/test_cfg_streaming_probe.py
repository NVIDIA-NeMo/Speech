# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The streaming probe must detect failures and release its input iterator."""

from types import SimpleNamespace

import pytest
from probe_cfg_streaming import run_stream_probe


@pytest.mark.asyncio
@pytest.mark.parametrize('abort', [False, True])
async def test_streaming_probe_requires_delayed_chunks_and_cleans_up(abort):
    engine = _Engine()
    result = await run_stream_probe(engine, _meta(), request_id='public', text='hello', abort_waiting=abort)
    assert result['aborted'] is abort
    assert result['codec_finished'] is not abort
    assert len(result['chunk_offsets']) == (1 if abort else 4)
    assert engine.closed and engine.aborted


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['private', 'missing_codec', 'error', 'no_tail'])
async def test_streaming_probe_rejects_bad_outputs_and_cleans_up(failure):
    engine = _Engine(failure)
    with pytest.raises(RuntimeError):
        await run_stream_probe(engine, _meta(), request_id='public', text='hello')
    assert engine.closed and engine.aborted


class _Engine:
    def __init__(self, failure=None):
        self.failure = failure
        self.aborted = []
        self.closed = False

    async def generate(self, inputs, request_id):
        try:
            async for chunk in inputs:
                if self.failure == 'no_tail' and chunk.prompt['additional_information']['text_token'] == []:
                    yield SimpleNamespace(request_id=request_id, stage_id=1, finished=True)
                    return
                yield SimpleNamespace(
                    request_id='private' if self.failure == 'private' else request_id,
                    stage_id=0,
                    finished=False,
                    error='failed' if self.failure == 'error' else None,
                    outputs=[
                        SimpleNamespace(token_ids=[0] * chunk.sampling_params.max_tokens, finish_reason='length')
                    ],
                )
            if self.failure != 'missing_codec':
                yield SimpleNamespace(request_id=request_id, stage_id=1, finished=True)
        finally:
            self.closed = True

    async def abort(self, request_id):
        self.aborted.append(request_id)


def _meta():
    return SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: list(range(8))),
        text_prefill_num=4,
        stop_token_id=1,
        text_eos_id=99,
        prompt_len=6,
        speaker_id='eng',
    )
