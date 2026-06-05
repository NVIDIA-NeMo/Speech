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
"""
Benchmark script for the EasyMagpie TTS Triton server (decoupled mode, gRPC).

Spawns N concurrent workers that send TTS requests in parallel against the
``easymp`` Triton model (see ``model_repository/easymp/config.pbtxt``).
Each line of the text file is parsed as ``<uttid>\\t<text>``.
Texts are randomly sampled for each request.

Usage:
    python benchmark_easymagpie_triton.py --text-file vctk_subset.txt --num-requests 100 --num-workers 8
    python benchmark_easymagpie_triton.py --text-file vctk_subset.txt --num-requests 50 \
        --output-dir out_wavs
"""

import argparse
import queue
import random
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient

SAMPLE_RATE = 22_050  # codec output_sample_rate (matches run_server_request.ipynb)
MODEL_NAME = "easymp"


@dataclass
class RequestResult:
    uttid: str
    num_samples: int
    duration_s: float
    ttfa_s: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkStats:
    lock: threading.Lock = field(default_factory=threading.Lock)
    results: list[RequestResult] = field(default_factory=list)

    def add(self, result: RequestResult):
        with self.lock:
            self.results.append(result)


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def synthesize(
    client: grpcclient.InferenceServerClient,
    result_q: queue.Queue,
    text: str,
    chunk_timeout: float,
):
    """Send one TTS request and collect streamed chunks.

    Returns ``(audio, ttfa_s, elapsed_s, error)``.
    """
    text_input = grpcclient.InferInput("text", [1, 1], "BYTES")
    text_input.set_data_from_numpy(np.array([[text]], dtype=object))

    t0 = time.perf_counter()
    t_first: float | None = None
    chunks: list[np.ndarray] = []

    client.async_stream_infer(
        model_name=MODEL_NAME,
        inputs=[text_input],
        outputs=[grpcclient.InferRequestedOutput("audio")],
    )

    while True:
        try:
            result, error = result_q.get(timeout=chunk_timeout)
        except queue.Empty:
            elapsed = time.perf_counter() - t0
            return None, elapsed, elapsed, "no chunk within chunk_timeout"

        if error:
            elapsed = time.perf_counter() - t0
            return None, elapsed, elapsed, str(error)

        audio = result.as_numpy("audio").squeeze()
        if audio.size > 0:
            if t_first is None:
                t_first = time.perf_counter()
            chunks.append(audio)

        response = result.get_response()
        final_param = response.parameters.get("triton_final_response")
        if final_param and getattr(final_param, "bool_param", False):
            break

    elapsed = time.perf_counter() - t0
    ttfa = (t_first - t0) if t_first else elapsed
    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    return audio, ttfa, elapsed, None


def worker(
    worker_id: int,
    triton_url: str,
    items: list[tuple[str, str]],
    task_queue: list[int],
    queue_lock: threading.Lock,
    stats: BenchmarkStats,
    chunk_timeout: float,
    output_dir: Path | None,
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
            audio, ttfa, elapsed, error = synthesize(client, result_q, text, chunk_timeout)

            if error is not None:
                # Reset the stream so late chunks don't bleed into the next
                # request.
                client.stop_stream()
                client.start_stream(callback=lambda result, error: result_q.put((result, error)))
                stats.add(
                    RequestResult(
                        uttid=uttid,
                        num_samples=0,
                        duration_s=elapsed,
                        ttfa_s=ttfa,
                        error=error,
                    )
                )
                print(f"[worker {worker_id:02d}] request {task_idx} ({uttid}) FAILED ({elapsed:.1f}s) — {error}")
                continue

            num_samples = len(audio)
            if output_dir is not None and num_samples > 0:
                _save_wav(output_dir / f"{uttid}.wav", audio)

            stats.add(
                RequestResult(
                    uttid=uttid,
                    num_samples=num_samples,
                    duration_s=elapsed,
                    ttfa_s=ttfa,
                )
            )
            print(
                f"[worker {worker_id:02d}] request {task_idx} ({uttid}) done — "
                f"{num_samples / SAMPLE_RATE:.2f}s audio in {elapsed:.2f}s "
                f"(TTFA: {ttfa:.3f}s)"
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
    items: list[tuple[str, str]],
    num_tasks: int,
    chunk_timeout: float,
    output_dir: Path | None,
) -> tuple[BenchmarkStats, float]:
    task_queue = list(range(num_tasks))
    queue_lock = threading.Lock()
    stats = BenchmarkStats()

    threads = [
        threading.Thread(
            target=worker,
            args=(i, triton_url, items, task_queue, queue_lock, stats, chunk_timeout, output_dir),
        )
        for i in range(num_workers)
    ]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return stats, time.perf_counter() - wall_start


def main():
    parser = argparse.ArgumentParser(description="Benchmark EasyMagpie TTS Triton server")
    parser.add_argument("--text-file", required=True, help="Path to file with '<uttid>\\t<text>' per line")
    parser.add_argument("--num-requests", type=int, required=True, help="Total number of requests to send")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of concurrent workers (default: 4)")
    parser.add_argument(
        "--triton-url", default="localhost:8001", help="Triton gRPC endpoint (default: localhost:8001)"
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup phase (3 requests per worker)")
    parser.add_argument(
        "--chunk-timeout", type=float, default=60, help="Per-chunk receive timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--output-dir", default=None, help="If set, write each generated waveform to <output-dir>/<uttid>.wav"
    )
    args = parser.parse_args()

    items = _load_items(args.text_file)
    if not items:
        print(f"ERROR: no usable lines found in {args.text_file}")
        return

    output_dir: Path | None = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(items)} utterances from {args.text_file}")
    print(f"Sending {args.num_requests} requests with {args.num_workers} workers to {args.triton_url}")
    if output_dir is not None:
        print(f"Writing WAVs to {output_dir.resolve()}")
    print("-" * 70)

    if not args.no_warmup:
        total_warmup = args.num_workers * 3
        print(f"Warmup: {total_warmup} requests (3 per worker) ...")
        _run_workers(
            args.num_workers,
            args.triton_url,
            items,
            total_warmup,
            args.chunk_timeout,
            output_dir=None,
        )
        print("Warmup complete.")
        print("-" * 70)

    stats, wall_elapsed = _run_workers(
        args.num_workers,
        args.triton_url,
        items,
        args.num_requests,
        args.chunk_timeout,
        output_dir,
    )

    successes = [r for r in stats.results if r.error is None]
    failures = [r for r in stats.results if r.error is not None]
    total_audio_seconds = sum(r.num_samples for r in successes) / SAMPLE_RATE

    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Total requests sent:      {args.num_requests}")
    print(f"  Successful:               {len(successes)}")
    print(f"  Failed:                   {len(failures)}")
    print(f"  Concurrent workers:       {args.num_workers}")
    print()
    print(f"  Wall-clock time:          {wall_elapsed:.2f} s")
    print(f"  Total audio synthesized:  {total_audio_seconds:.2f} s")
    print(f"  Real-time factor (RTF):   {total_audio_seconds / wall_elapsed:.2f}x")
    print(f"  Throughput:               {len(successes) / wall_elapsed:.2f} requests/s")

    if successes:
        ttfas_ms = sorted(r.ttfa_s * 1000 for r in successes)
        mean_ttfa = sum(ttfas_ms) / len(ttfas_ms)
        print()
        print("  Time to first audio (TTFA):")
        print(f"    mean:   {mean_ttfa:.1f} ms")
        print(f"    p95:    {ttfas_ms[int(len(ttfas_ms) * 0.95)]:.1f} ms")

    print("=" * 70)


if __name__ == "__main__":
    main()
