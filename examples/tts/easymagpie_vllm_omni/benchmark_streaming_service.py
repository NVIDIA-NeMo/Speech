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
"""Benchmark the EasyMagpie TTS Triton server in streaming-text mode (gRPC).

Unlike ``benchmark_service.py`` (whole-text), each request feeds subword ids one
chunk at a time over a single gRPC stream (``stream_start`` ... ``stream_end``);
all audio rides back on the ``stream_start`` request's response. Multiple
concurrency levels can be benchmarked in sequence.

Usage:
    python benchmark_streaming_service.py --text-file vctk_subset.txt \
        --model-dir ./easymp_vllm_model -n 100 -c 8
    python benchmark_streaming_service.py --text-file vctk_subset.txt \
        --model-dir ./easymp_vllm_model -n 50 -c 1 4 8
"""

import argparse
import queue
import random
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer

SAMPLE_RATE = 22_050
MODEL_NAME = "easymp"


@dataclass
class RequestResult:
    uttid: str
    num_samples: int = 0
    elapsed_s: float = 0.0
    ttfa_s: float = 0.0
    reached_eos: bool = False
    keepup_ratios: list[float] = field(default_factory=list)
    audio: np.ndarray | None = None
    error: str | None = None


@dataclass
class BenchmarkStats:
    lock: threading.Lock = field(default_factory=threading.Lock)
    results: list[RequestResult] = field(default_factory=list)

    def add(self, result: RequestResult):
        with self.lock:
            self.results.append(result)


def _str_input(name: str, value: str):
    t = grpcclient.InferInput(name, [1, 1], "BYTES")
    t.set_data_from_numpy(np.array([[value]], dtype=object))
    return t


def _bool_input(name: str, value: bool):
    t = grpcclient.InferInput(name, [1, 1], "BOOL")
    t.set_data_from_numpy(np.array([[value]], dtype=bool))
    return t


def _token_input(tokens):
    t = grpcclient.InferInput("text_token", [1, len(tokens)], "INT32")
    t.set_data_from_numpy(np.array([tokens], dtype=np.int32))
    return t


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def synthesize_streaming(
    client: grpcclient.InferenceServerClient,
    result_q: queue.Queue,
    tokenizer,
    uttid: str,
    text: str,
    speaker: str,
    context_text: str,
    tokens_per_chunk: int,
    token_delay: float,
    chunk_timeout: float,
    save_audio: bool = False,
) -> RequestResult:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = [token_ids[i : i + tokens_per_chunk] for i in range(0, len(token_ids), tokens_per_chunk)] or [[]]
    num_requests = len(chunks)

    stream_id = uuid.uuid4().hex
    start_rid = f"{stream_id}-start"

    def _send(inputs, request_id):
        client.async_stream_infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=[grpcclient.InferRequestedOutput("audio")],
            request_id=request_id,
        )

    t0 = time.perf_counter()

    start_inputs = [
        _str_input("stream_id", stream_id),
        _bool_input("stream_start", True),
        _str_input("speaker", speaker),
        _str_input("context_text", context_text),
    ]
    if chunks[0]:
        start_inputs.append(_token_input(chunks[0]))
    if num_requests == 1:
        start_inputs.append(_bool_input("stream_end", True))
    _send(start_inputs, start_rid)

    for ci, chunk in enumerate(chunks[1:], start=1):
        if token_delay:
            time.sleep(token_delay)
        inputs = [_str_input("stream_id", stream_id), _token_input(chunk)]
        if ci == num_requests - 1:
            inputs.append(_bool_input("stream_end", True))
        _send(inputs, f"{stream_id}-c{ci}")

    num_samples = 0
    t_first: float | None = None
    t_prev: float | None = None
    keepup: list[float] = []
    audio_chunks: list[np.ndarray] = []
    reached_eos = False

    # All audio and the authoritative final ride on the stream_start response;
    # follow-up token requests complete with an empty final that Triton does not
    # surface to the client, so we key off start_rid only (matches the demo).
    while True:
        try:
            result, error = result_q.get(timeout=chunk_timeout)
        except queue.Empty:
            elapsed = time.perf_counter() - t0
            return RequestResult(uttid=uttid, elapsed_s=elapsed, ttfa_s=elapsed, error="no chunk within chunk_timeout")

        if error:
            elapsed = time.perf_counter() - t0
            return RequestResult(uttid=uttid, elapsed_s=elapsed, ttfa_s=elapsed, error=str(error))

        response = result.get_response()
        if response.id != start_rid:
            continue

        audio = result.as_numpy("audio")
        if audio is not None:
            audio = audio.squeeze()
            if audio.size > 0:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                else:
                    keepup.append((now - t_prev) / (audio.size / SAMPLE_RATE))
                t_prev = now
                num_samples += int(audio.size)
                if save_audio:
                    audio_chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))

        if bool(getattr(response.parameters.get("triton_final_response"), "bool_param", False)):
            reached_eos = True
            break

    # Drain any straggler follow-up responses so they don't bleed into the next
    # utterance on this persistent stream (cross-request leftovers are also
    # filtered out by the start_rid check above, since each request uses a fresh id).
    while True:
        try:
            result_q.get_nowait()
        except queue.Empty:
            break

    elapsed = time.perf_counter() - t0
    ttfa = (t_first - t0) if t_first is not None else elapsed
    return RequestResult(
        uttid=uttid,
        num_samples=num_samples,
        elapsed_s=elapsed,
        ttfa_s=ttfa,
        reached_eos=reached_eos,
        keepup_ratios=keepup,
        audio=(np.concatenate(audio_chunks) if save_audio and audio_chunks else None),
    )


def worker(
    worker_id: int,
    triton_url: str,
    tokenizer,
    items: list[tuple[str, str]],
    task_queue: list[int],
    queue_lock: threading.Lock,
    stats: BenchmarkStats,
    speaker: str,
    context_text: str,
    tokens_per_chunk: int,
    token_delay: float,
    chunk_timeout: float,
    output_dir: Path | None,
    verbose: bool,
):
    result_q: queue.Queue = queue.Queue()
    client = grpcclient.InferenceServerClient(url=triton_url)
    client.start_stream(callback=lambda result, error: result_q.put((result, error)))

    try:
        while True:
            with queue_lock:
                if not task_queue:
                    return
                task_idx = task_queue.pop()

            uttid, text = random.choice(items)
            result = synthesize_streaming(
                client,
                result_q,
                tokenizer,
                uttid,
                text,
                speaker,
                context_text,
                tokens_per_chunk,
                token_delay,
                chunk_timeout,
                save_audio=output_dir is not None,
            )
            stats.add(result)

            if result.error is not None:
                client.stop_stream()
                client.start_stream(callback=lambda result, error: result_q.put((result, error)))
                if verbose:
                    print(f"[worker {worker_id:02d}] req {task_idx} ({uttid}) FAILED — {result.error}")
            else:
                if output_dir is not None and result.audio is not None and result.audio.size > 0:
                    _save_wav(output_dir / f"{uttid}.wav", result.audio)
                if verbose:
                    print(
                        f"[worker {worker_id:02d}] req {task_idx} ({uttid}) — "
                        f"{result.num_samples / SAMPLE_RATE:.2f}s audio in {result.elapsed_s:.2f}s "
                        f"(TTFA {result.ttfa_s * 1000:.0f}ms)"
                    )
    finally:
        client.stop_stream()


def _load_items(text_file: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    with open(text_file) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Expected '<uttid>\\t<text>' per line, got: {line!r}")
            uttid, text = parts[0].strip(), parts[1].strip()
            if not uttid or not text:
                raise ValueError(f"Empty uttid or text in line: {line!r}")
            items.append((uttid, text))
    return items


def _run_workers(
    num_workers: int,
    triton_url: str,
    tokenizer,
    items: list[tuple[str, str]],
    num_tasks: int,
    speaker: str,
    context_text: str,
    tokens_per_chunk: int,
    token_delay: float,
    chunk_timeout: float,
    output_dir: Path | None,
    verbose: bool,
) -> tuple[BenchmarkStats, float]:
    task_queue = list(range(num_tasks))
    queue_lock = threading.Lock()
    stats = BenchmarkStats()

    threads = [
        threading.Thread(
            target=worker,
            args=(
                i,
                triton_url,
                tokenizer,
                items,
                task_queue,
                queue_lock,
                stats,
                speaker,
                context_text,
                tokens_per_chunk,
                token_delay,
                chunk_timeout,
                output_dir,
                verbose,
            ),
        )
        for i in range(num_workers)
    ]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return stats, time.perf_counter() - wall_start


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * pct))
    return sorted_vals[idx]


def _summarize(stats: BenchmarkStats, wall_s: float, concurrency: int) -> dict:
    eos = [r for r in stats.results if r.reached_eos]
    audio_s = sum(r.num_samples for r in eos) / SAMPLE_RATE
    ttfas_ms = sorted(r.ttfa_s * 1000 for r in eos)
    keepup = sorted(ratio for r in eos for ratio in r.keepup_ratios)
    return {
        "concurrency": concurrency,
        "failed": len(stats.results) - len(eos),
        "wall_s": wall_s,
        "audio_s": audio_s,
        "rtx": audio_s / wall_s if wall_s > 0 else 0.0,
        "tput": len(eos) / wall_s if wall_s > 0 else 0.0,
        "ttfa_mean_ms": (sum(ttfas_ms) / len(ttfas_ms)) if ttfas_ms else 0.0,
        "ttfa_p95_ms": _percentile(ttfas_ms, 0.95),
        "keepup_mean": (sum(keepup) / len(keepup)) if keepup else 0.0,
        "keepup_p95": _percentile(keepup, 0.95),
    }


def _print_summary(s: dict):
    print(
        f"[concurrency={s['concurrency']}] rtx = synt / wall = "
        f"{s['rtx']:.2f}x = {s['audio_s']:.2f} / {s['wall_s']:.0f}"
    )
    print(
        f"throughput={s['tput']:.2f} req/s; failed = {s['failed']}; "
        f"TTFA={s['ttfa_mean_ms']:.1f} / {s['ttfa_p95_ms']:.1f} (p95); "
        f"keepup={s['keepup_mean']:.3f} / {s['keepup_p95']:.3f} (p95)"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark EasyMagpie TTS Triton server (streaming-text mode)")
    parser.add_argument("--text-file", required=True, help="Path to file with '<uttid>\\t<text>' per line")
    parser.add_argument("--model-dir", required=True, help="Model dir with the tokenizer (text -> subword ids)")
    parser.add_argument("-n", "--num-requests", type=int, required=True, help="Requests per concurrency level")
    parser.add_argument("-c", "--concurrency", type=int, nargs="+", default=[4], help="Concurrency levels to test")
    parser.add_argument("--triton-url", default="localhost:8001", help="Triton gRPC endpoint (default localhost:8001)")
    parser.add_argument("--speaker", default="eng", help="Speaker id (default: eng)")
    parser.add_argument("--context-text", default="[EN]", help="Context text (default: [EN])")
    parser.add_argument("--tokens-per-chunk", type=int, default=1, help="Subword ids fed per stream chunk (default 1)")
    parser.add_argument(
        "--token-delay", type=float, default=0.0, help="Sleep between token chunks to mimic upstream LLM (default 0)"
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup phase (3 requests per worker)")
    parser.add_argument("--chunk-timeout", type=float, default=60, help="Per-chunk receive timeout, s (default: 60)")
    parser.add_argument(
        "--output-dir", default=None, help="If set, write each generated waveform to <output-dir>/<uttid>.wav"
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-request lines")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    items = _load_items(args.text_file)
    if not items:
        print(f"ERROR: no usable lines found in {args.text_file}")
        return

    print(f"Loading tokenizer from {args.model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    print(
        f"Loaded {len(items)} utterances; {args.num_requests} req/level; concurrency {args.concurrency} "
        f"({args.tokens_per_chunk} tok/chunk, {args.token_delay}s delay)"
    )

    summaries = []
    for concurrency in args.concurrency:
        if not args.no_warmup:
            _run_workers(
                concurrency,
                args.triton_url,
                tokenizer,
                items,
                concurrency * 3,
                args.speaker,
                args.context_text,
                args.tokens_per_chunk,
                args.token_delay,
                args.chunk_timeout,
                output_dir=None,
                verbose=False,
            )

        stats, wall_elapsed = _run_workers(
            concurrency,
            args.triton_url,
            tokenizer,
            items,
            args.num_requests,
            args.speaker,
            args.context_text,
            args.tokens_per_chunk,
            args.token_delay,
            args.chunk_timeout,
            output_dir=output_dir,
            verbose=args.verbose,
        )
        summary = _summarize(stats, wall_elapsed, concurrency)
        summaries.append(summary)
        _print_summary(summary)

    if len(summaries) > 1:
        print("\n=== Summary ===")
        for s in summaries:
            _print_summary(s)


if __name__ == "__main__":
    main()
