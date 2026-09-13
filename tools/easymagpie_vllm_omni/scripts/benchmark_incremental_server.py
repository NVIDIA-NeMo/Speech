#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark token-chunked EasyMagpieTTS through /v1/audio/speech/stream.

The input text is tokenized once in the parent process. Each worker sends exact
EasyMagpie token IDs in configurable chunk sizes while concurrently receiving
binary PCM frames from the WebSocket.

Examples:
    python scripts/benchmark_incremental_server.py \
        --model ./converted_model --text-file vctk_subset.txt -n 100 -c 1 8
    python scripts/benchmark_incremental_server.py \
        --model ./converted_model --text-file vctk_subset.txt -n 100 \
        -c 1 8 --tokens-per-chunk 1 3 5 10
    python scripts/benchmark_incremental_server.py \
        --model ./converted_model --text-file vctk_subset.txt -n 50 \
        --tokens-per-chunk 5 --send-delay-ms 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import benchmark_server as base
import numpy as np
import websockets
from easymagpie_vllm_omni.tokenizer import EasyMagpieTextTokenizer


@dataclass
class IncrementalRequestResult(base.RequestResult):
    num_text_tokens: int = 0
    num_input_chunks: int = 0
    input_send_s: float = 0.0


def _websocket_url(server_url: str) -> str:
    parts = urlsplit(server_url)
    if parts.scheme not in {"http", "https", "ws", "wss"}:
        raise ValueError(f"Unsupported server URL scheme: {parts.scheme!r}")
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
    path = f"{parts.path.rstrip('/')}/v1/audio/speech/stream"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


async def _do_request_async(task: dict) -> IncrementalRequestResult:
    uttid = task["uttid"]
    token_ids = task["token_ids"]
    tokens_per_chunk = int(task["tokens_per_chunk"])
    token_chunks = [
        token_ids[index : index + tokens_per_chunk] for index in range(0, len(token_ids), tokens_per_chunk)
    ]
    t0 = time.perf_counter()
    t_first: float | None = None
    input_send_s = 0.0
    num_samples = 0
    sr = int(task["sample_rate"])
    trailing_byte = b""
    chunk_arrivals: list[float] = []
    chunk_durations: list[float] = []
    output_parts: list[np.ndarray] = []

    try:
        async with asyncio.timeout(float(task["timeout"])):
            async with websockets.connect(
                _websocket_url(task["url"]),
                max_size=64 * 1024 * 1024,
                close_timeout=5,
            ) as websocket:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "session.config",
                            "voice": task["speaker_id"] or "eng",
                            "stream_audio": True,
                            "response_format": "pcm",
                            "max_new_tokens": task["max_new_tokens"],
                        }
                    )
                )

                async def send_tokens() -> None:
                    nonlocal input_send_s
                    for chunk in token_chunks:
                        await websocket.send(json.dumps({"type": "input.tokens", "tokens": chunk}))
                        if task["send_delay_s"] > 0:
                            await asyncio.sleep(task["send_delay_s"])
                    await websocket.send(json.dumps({"type": "input.done"}))
                    input_send_s = time.perf_counter() - t0

                sender = asyncio.create_task(send_tokens())
                try:
                    while True:
                        message = await websocket.recv()
                        now = time.perf_counter()
                        if isinstance(message, bytes):
                            data = trailing_byte + message
                            even_bytes = len(data) - (len(data) % 2)
                            trailing_byte = data[even_bytes:]
                            if even_bytes == 0:
                                continue
                            if t_first is None:
                                t_first = now
                            pcm = np.frombuffer(data[:even_bytes], dtype="<i2").astype(np.float32) / 32768.0
                            num_samples += pcm.size
                            chunk_arrivals.append(now - t0)
                            chunk_durations.append(pcm.size / sr if sr else 0.0)
                            if task["output_dir"]:
                                output_parts.append(pcm)
                            continue

                        event = json.loads(message)
                        event_type = event.get("type")
                        if event_type == "audio.start":
                            sr = int(event.get("sample_rate", sr))
                        elif event_type == "audio.done" and event.get("error"):
                            raise RuntimeError(f"Server reported generation failure: {event}")
                        elif event_type == "error":
                            raise RuntimeError(event.get("message", str(event)))
                        elif event_type == "session.done":
                            _ = await sender
                            break
                finally:
                    if not sender.done():
                        sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
    except Exception as exc:  # noqa: BLE001 - report any client/server failure
        return IncrementalRequestResult(
            uttid=uttid,
            elapsed_s=time.perf_counter() - t0,
            num_text_tokens=len(token_ids),
            num_input_chunks=len(token_chunks),
            input_send_s=input_send_s,
            error=repr(exc),
        )

    elapsed = time.perf_counter() - t0
    if num_samples == 0:
        return IncrementalRequestResult(
            uttid=uttid,
            sr=sr,
            elapsed_s=elapsed,
            num_text_tokens=len(token_ids),
            num_input_chunks=len(token_chunks),
            input_send_s=input_send_s,
            error="empty audio response",
        )
    if task["output_dir"]:
        base._save_wav(Path(task["output_dir"]) / f"{uttid}.wav", np.concatenate(output_parts), sr)
    return IncrementalRequestResult(
        uttid=uttid,
        num_samples=num_samples,
        sr=sr,
        elapsed_s=elapsed,
        ttfa_s=(t_first - t0) if t_first is not None else elapsed,
        chunk_arrivals=chunk_arrivals,
        chunk_durations=chunk_durations,
        num_text_tokens=len(token_ids),
        num_input_chunks=len(token_chunks),
        input_send_s=input_send_s,
    )


def _do_request(task: dict) -> IncrementalRequestResult:
    """Run one WebSocket request inside a benchmark worker process."""
    return asyncio.run(_do_request_async(task))


def _make_tasks(
    items: list[tuple[str, str, list[int]]],
    n: int,
    *,
    url: str,
    speaker_id: str | None,
    max_new_tokens: int,
    sample_rate: int,
    timeout: float,
    output_dir: str | None,
    tokens_per_chunk: int,
    send_delay_s: float,
) -> list[dict]:
    selected_items = random.sample(items, n) if n <= len(items) else random.choices(items, k=n)
    tasks = []
    for uttid, _, token_ids in selected_items:
        tasks.append(
            {
                "url": url,
                "uttid": uttid,
                "token_ids": token_ids,
                "speaker_id": speaker_id,
                "max_new_tokens": max_new_tokens,
                "sample_rate": sample_rate,
                "timeout": timeout,
                "output_dir": output_dir,
                "tokens_per_chunk": tokens_per_chunk,
                "send_delay_s": send_delay_s,
            }
        )
    return tasks


def _run_level(tasks: list[dict], concurrency: int) -> tuple[list[IncrementalRequestResult], float]:
    wall0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(_do_request, tasks))
    return results, time.perf_counter() - wall0


def _summarize(
    results: list[IncrementalRequestResult],
    wall_s: float,
    concurrency: int,
    tokens_per_chunk: int,
) -> dict:
    summary = base._summarize(results, wall_s, concurrency)
    ok = [result for result in results if result.error is None]
    input_send_ms = sorted(result.input_send_s * 1000.0 for result in ok)
    post_input_ms = sorted((result.elapsed_s - result.input_send_s) * 1000.0 for result in ok)
    total_tokens = sum(result.num_text_tokens for result in ok)
    total_input_chunks = sum(result.num_input_chunks for result in ok)
    summary.update(
        {
            "tokens_per_chunk": tokens_per_chunk,
            "total_text_tokens": total_tokens,
            "total_input_chunks": total_input_chunks,
            "text_tokens_per_s": total_tokens / wall_s if wall_s else 0.0,
            "input_chunks_per_s": total_input_chunks / wall_s if wall_s else 0.0,
            "input_send_mean_ms": base._mean(input_send_ms),
            "input_send_p95_ms": base._percentile(input_send_ms, 0.95),
            "post_input_mean_ms": base._mean(post_input_ms),
            "post_input_p95_ms": base._percentile(post_input_ms, 0.95),
        }
    )
    return summary


def _print_detailed(summary: dict) -> None:
    print(
        f"[tokens/chunk={summary['tokens_per_chunk']}, concurrency={summary['concurrency']}]  "
        f"{summary['ok']} ok / {summary['failed']} failed"
    )
    print(
        f"    req/s {summary['tput']:.2f}  |  audio RTF {summary['rtf']:.2f}x  "
        f"(audio {summary['audio_s']:.0f}s / wall {summary['wall_s']:.2f}s)"
    )
    print(f"    ttfa  mean {summary['ttfa_mean_ms']:.1f}ms  p95 {summary['ttfa_p95_ms']:.1f}ms")
    print(f"    lat   mean {summary['lat_mean_s']:.2f}s   p95 {summary['lat_p95_s']:.2f}s")
    print(
        f"    input sent  mean {summary['input_send_mean_ms']:.1f}ms  "
        f"p95 {summary['input_send_p95_ms']:.1f}ms  |  "
        f"after input.done mean {summary['post_input_mean_ms']:.1f}ms"
    )
    print(
        f"    input rate  {summary['text_tokens_per_s']:.1f} token/s  |  "
        f"{summary['input_chunks_per_s']:.1f} chunks/s"
    )
    print(
        f"    audio ITL mean {summary['itl_mean_ms']:.1f}ms  p95 {summary['itl_p95_ms']:.1f}ms  |  "
        f"underruns {summary['total_underruns']}/{summary['total_chunks']} "
        f"({summary['underrun_pct']:.2f}%)"
    )


def _print_summary(summary: dict) -> None:
    print(
        f"tokens/chunk={summary['tokens_per_chunk']}, concurrency={summary['concurrency']}: "
        f"req/s {summary['tput']:.2f}, ttfa {summary['ttfa_mean_ms']:.1f}ms, "
        f"lat {summary['lat_mean_s']:.2f}s, rtf {summary['rtf']:.2f}x, "
        f"input {summary['text_tokens_per_s']:.1f} token/s, "
        f"{summary['ok']} ok / {summary['failed']} failed"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark incremental EasyMagpieTTS WebSocket serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, help="Converted EasyMagpie model directory (for its tokenizer)")
    parser.add_argument("--text-file", required=True, help="Path to file with '<uttid>\\t<text>' per line")
    parser.add_argument("-n", "--num-requests", type=int, required=True, help="Requests per benchmark level")
    parser.add_argument("-c", "--concurrency", type=int, nargs="+", default=[4], help="Concurrency levels")
    parser.add_argument(
        "--tokens-per-chunk",
        type=int,
        nargs="+",
        default=[5],
        help="Token chunk sizes to benchmark (default: %(default)s)",
    )
    parser.add_argument("--send-delay-ms", type=float, default=0.0, help="Delay after every input chunk")
    parser.add_argument("--url", default="http://localhost:8091", help="Server base URL (default: %(default)s)")
    parser.add_argument("--speaker-id", default=None, help="Speaker id (default: server default)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--sample-rate", type=int, default=22050, help="Fallback raw PCM sample rate")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds")
    parser.add_argument("--no-warmup", action="store_true", help="Skip one warmup request per worker")
    parser.add_argument("--output-dir", default=None, help="If set, save each generated waveform")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_requests < 1:
        raise ValueError("--num-requests must be positive")
    if any(value < 1 for value in args.concurrency):
        raise ValueError("--concurrency values must be positive")
    if any(value < 1 for value in args.tokens_per_chunk):
        raise ValueError("--tokens-per-chunk values must be positive")
    if args.send_delay_ms < 0:
        raise ValueError("--send-delay-ms cannot be negative")

    text_items = base._load_items(args.text_file)
    if not text_items:
        raise ValueError(f"No usable lines found in {args.text_file}")
    tokenizer = EasyMagpieTextTokenizer.from_pretrained(args.model)
    items = [(uttid, text, tokenizer.encode(text, add_special_tokens=False)) for uttid, text in text_items]

    output_dir: str | None = None
    if args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        output_dir = args.output_dir

    print(
        f"Loaded {len(items)} utterances; {args.num_requests} req/level; "
        f"concurrency {args.concurrency}; tokens/chunk {args.tokens_per_chunk}; "
        f"send delay {args.send_delay_ms:.1f}ms; url {args.url}"
    )
    summaries = []
    for tokens_per_chunk in args.tokens_per_chunk:
        for concurrency in args.concurrency:
            common = {
                "url": args.url,
                "speaker_id": args.speaker_id,
                "max_new_tokens": args.max_new_tokens,
                "sample_rate": args.sample_rate,
                "timeout": args.timeout,
                "output_dir": output_dir,
                "tokens_per_chunk": tokens_per_chunk,
                "send_delay_s": args.send_delay_ms / 1000.0,
            }
            if not args.no_warmup:
                warmup_tasks = _make_tasks(items, concurrency, **{**common, "output_dir": None})
                _run_level(warmup_tasks, concurrency)

            tasks = _make_tasks(items, args.num_requests, **common)
            results, wall_s = _run_level(tasks, concurrency)
            summary = _summarize(results, wall_s, concurrency, tokens_per_chunk)
            summaries.append(summary)
            _print_detailed(summary)

    print("\n=== Summary ===")
    for summary in summaries:
        _print_summary(summary)


if __name__ == "__main__":
    main()
