# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU paired-CFG incremental-text completion, waiting cancellation, and reuse probe."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from benchmark_model import _load_model_meta
from easymagpie_vllm_omni.serving_stream import EasyMagpieInputStream


async def run_stream_probe(omni, meta, *, request_id, text, timeout=300, abort_waiting=False, use_cfg=True):
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    params = SamplingParams(
        temperature=0,
        max_tokens=400,
        detokenize=False,
        ignore_eos=True,
        stop_token_ids=[meta.stop_token_id],
        output_kind=RequestOutputKind.DELTA,
    )
    tokens = list(meta.tokenizer.encode(text, add_special_tokens=False))
    first_size = max(4, meta.text_prefill_num)
    if len(tokens) <= first_size:
        raise ValueError('Probe text must span multiple chunks')
    stream = EasyMagpieInputStream(
        prefill_prompt={
            'prompt_token_ids': [0] * meta.prompt_len,
            'additional_information': dict(
                speaker_id=meta.speaker_id,
                context_text='[EN]',
                use_cfg=use_cfg,
                cfg_scale=2.5,
                temperature=0.7,
                top_k=80,
                max_stream_tokens=400,
            ),
        },
        sampling_params=params,
        text_eos_id=meta.text_eos_id,
        max_new_tokens=400,
        text_prefill_num=meta.text_prefill_num,
        coalesce_queued_tokens=False,
    )
    await stream.put_tokens(tokens[:first_size])
    boundaries = asyncio.Queue()
    count, stage0_frames, codec_finished, chunks = 0, 0, False, []
    tail_frames = 0
    tail_finish_reason = None

    async def inputs():
        async for chunk in stream.inputs():
            chunks.append(chunk.prompt['additional_information'].get('text_token_start'))
            print(
                json.dumps(
                    dict(
                        request_id=request_id,
                        event='input',
                        offset=chunks[-1],
                        max_tokens=chunk.sampling_params.max_tokens,
                    )
                ),
                flush=True,
            )
            yield chunk

    generator = omni.generate(inputs(), request_id=request_id)

    async def consume():
        nonlocal count, stage0_frames, codec_finished, tail_frames, tail_finish_reason
        async for output in generator:
            if output.request_id != request_id or getattr(output, 'error', None):
                raise RuntimeError(f'Unexpected public output: {output}')
            count += 1
            stream.observe_output(output)
            codec_finished |= output.stage_id == 1 and output.finished
            if output.stage_id == 0:
                for item in output.outputs or []:
                    stage0_frames += len(getattr(item, 'token_ids', None) or [])
                    if chunks and chunks[-1] is None:
                        tail_frames += len(getattr(item, 'token_ids', None) or [])
                    if item.finish_reason is not None:
                        if chunks and chunks[-1] is None:
                            tail_finish_reason = item.finish_reason
                        print(
                            json.dumps(
                                dict(
                                    request_id=request_id,
                                    event='segment',
                                    frames=stage0_frames,
                                    reason=item.finish_reason,
                                )
                            ),
                            flush=True,
                        )
                        boundaries.put_nowait(None)

    consumer = asyncio.create_task(consume())
    try:
        async with asyncio.timeout(timeout):
            boundary = asyncio.create_task(boundaries.get())
            try:
                done, _ = await asyncio.wait((consumer, boundary), return_when=asyncio.FIRST_COMPLETED)
                if consumer in done:
                    await consumer
                    raise RuntimeError('Stream ended before waiting for another text chunk')
                await boundary
            finally:
                boundary.cancel()
                await asyncio.gather(boundary, return_exceptions=True)
            parked_frames = stage0_frames
            await asyncio.sleep(0.5)
            if consumer.done():
                await consumer
                raise RuntimeError('Stream ended while waiting for text')
            if stage0_frames != parked_frames:
                raise RuntimeError('Stage 0 advanced while waiting for text')
            if abort_waiting:
                await omni.abort(request_id)
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)
            else:
                for start in range(first_size, len(tokens), 4):
                    await stream.put_tokens(tokens[start : start + 4])
                await stream.finish()
                await consumer
                if not codec_finished or len(chunks) < 3 or not tail_frames or tail_finish_reason is None:
                    raise RuntimeError(
                        'Stream ended without chunk delivery, acoustic tail, and completed codec output'
                    )
        return dict(
            request_id=request_id,
            aborted=abort_waiting,
            public_outputs=count,
            stage0_frames=stage0_frames,
            tail_frames=tail_frames,
            tail_finish_reason=tail_finish_reason,
            codec_finished=codec_finished,
            chunk_offsets=chunks,
        )
    finally:
        try:
            await omni.abort(request_id)
        finally:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            await generator.aclose()


async def main(args):
    from vllm.plugins import load_general_plugins
    from vllm_omni import AsyncOmni

    load_general_plugins()
    meta = _load_model_meta(args.model, args.speaker)
    omni = AsyncOmni(model=args.model, deploy_config=str(args.deploy_config), log_stats=False)
    results = []
    tasks = []
    try:
        for index, abort in enumerate((False, True, False)):
            result = await run_stream_probe(
                omni, meta, request_id=f'cfg-stream-{index}', text=args.text, timeout=args.timeout, abort_waiting=abort
            )
            results.append(result)
            print(json.dumps(result), flush=True)
        if args.mixed:
            from benchmark_model import build_prompt
            from probe_cfg_pairing import run_probe

            tasks = [
                asyncio.create_task(
                    run_stream_probe(
                        omni,
                        meta,
                        request_id=f'mixed-stream-{index}',
                        text=args.text,
                        timeout=args.timeout,
                        abort_waiting=index == 1,
                        use_cfg=index != 3,
                    )
                )
                for index in range(4)
            ]
            prompt = build_prompt(args.text, meta)
            prompt['additional_information'].update(use_cfg=False, temperature=0.7, top_k=80)
            complete = asyncio.create_task(run_probe(omni, prompt, request_id='mixed-complete', timeout=args.timeout))
            tasks.append(complete)
            mixed = await asyncio.gather(*tasks)
            results.extend(mixed[:-1])
            results.append(dict(request_id='mixed-complete', public_outputs=mixed[-1]))
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        omni.shutdown()
    report = dict(status='passed', model=args.model, speaker=args.speaker, results=results)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--speaker', default='eng')
    parser.add_argument('--text', default='This voice is conditioned from request-time reference audio.')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mixed', action='store_true', help='Also pressure-test streaming and complete-text requests')
    parser.add_argument(
        '--deploy-config', type=Path, default=Path(__file__).resolve().parents[1] / 'deploy' / 'easymagpie_cfg.yaml'
    )
    return parser.parse_args()


if __name__ == '__main__':
    asyncio.run(main(parse_args()))
