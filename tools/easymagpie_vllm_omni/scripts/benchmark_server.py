#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Benchmark EasyMagpieTTS served by ``scripts/run_server.sh``.

Sends requests from ``-c`` parallel processes against POST /v1/audio/speech and
reports throughput, time-to-first-audio (TTFA), and streaming playback metrics.

Usage:
    python benchmark_server.py --text-file vctk_subset.txt -n 100 -c 8
    python benchmark_server.py --text-file lines.txt -n 50 -c 1 4 8 --url http://localhost:8091
"""
from __future__ import annotations

import argparse
import base64
import io
import random
import time
import wave
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests


@dataclass
class RequestResult:
    uttid: str
    num_samples: int = 0
    sr: int = 22050
    elapsed_s: float = 0.0
    ttfa_s: float = 0.0
    error: str | None = None
    chunk_arrivals: list[float] = field(default_factory=list)
    chunk_durations: list[float] = field(default_factory=list)


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _do_request(task: dict) -> RequestResult:
    """One streaming TTS request; runs inside a worker process."""
    url = task["url"]
    uttid = task["uttid"]
    payload = {
        "input": task["text"],
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
        "max_new_tokens": task["max_new_tokens"],
    }
    if task.get("ref_audio") is not None:
        payload["ref_audio"] = task["ref_audio"]
    else:
        payload["voice"] = task["speaker_id"] or "eng"

    t0 = time.perf_counter()
    t_first: float | None = None
    chunks: list[np.ndarray] = []
    chunk_arrivals: list[float] = []
    chunk_durations: list[float] = []
    num_samples = 0
    sr = int(task["sample_rate"])
    trailing_byte = b""
    try:
        with requests.post(f"{url}/v1/audio/speech", json=payload, stream=True, timeout=task["timeout"]) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=None):
                if not chunk:
                    continue
                now = time.perf_counter()
                data = trailing_byte + chunk
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
                    chunks.append(pcm)
    except Exception as exc:  # noqa: BLE001 - report any client/server failure
        return RequestResult(uttid=uttid, elapsed_s=time.perf_counter() - t0, error=repr(exc))

    elapsed = time.perf_counter() - t0
    if num_samples == 0:
        return RequestResult(uttid=uttid, sr=sr, elapsed_s=elapsed, error="empty audio response")
    ttfa = (t_first - t0) if t_first is not None else elapsed
    if task["output_dir"] and num_samples > 0:
        _save_wav(Path(task["output_dir"]) / f"{uttid}.wav", np.concatenate(chunks), sr)
    return RequestResult(
        uttid=uttid,
        num_samples=num_samples,
        sr=sr,
        elapsed_s=elapsed,
        ttfa_s=ttfa,
        chunk_arrivals=chunk_arrivals,
        chunk_durations=chunk_durations,
    )


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


def _make_tasks(
    items,
    n,
    url,
    speaker_id,
    max_new_tokens,
    sample_rate,
    timeout,
    output_dir,
    reference_audio=None,
    randomize_reference_audio=False,
) -> list[dict]:
    selected_items = random.sample(items, n) if n <= len(items) else random.choices(items, k=n)
    tasks = []
    for uttid, text in selected_items:
        tasks.append(
            {
                "url": url,
                "uttid": uttid,
                "text": text,
                "speaker_id": speaker_id,
                "max_new_tokens": max_new_tokens,
                "sample_rate": sample_rate,
                "timeout": timeout,
                "output_dir": output_dir,
                "ref_audio": (
                    _reference_audio_data_url(reference_audio, randomize_reference_audio)
                    if reference_audio is not None
                    else None
                ),
            }
        )
    return tasks


def _reference_audio_data_url(path: str | Path, randomize: bool) -> str:
    """Encode a WAV reference, optionally perturbing two PCM16 samples."""
    path = Path(path).expanduser()
    if not randomize:
        audio = path.read_bytes()
    else:
        with wave.open(str(path), "rb") as wf:
            if wf.getcomptype() != "NONE" or wf.getsampwidth() != 2:
                raise ValueError("Reference randomization requires an uncompressed PCM16 WAV")
            params = wf.getparams()
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").copy()
        if samples.size == 0:
            raise ValueError("Reference audio is empty")
        for index in random.sample(range(samples.size), min(2, samples.size)):
            step = random.choice((-1, 1))
            value = int(samples[index])
            if value + step < np.iinfo(np.int16).min or value + step > np.iinfo(np.int16).max:
                step = -step
            samples[index] = value + step
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setparams(params)
            wf.writeframes(samples.tobytes())
        audio = buffer.getvalue()
    return f"data:audio/wav;base64,{base64.b64encode(audio).decode()}"


def _run_level(tasks: list[dict], concurrency: int, output_dir: str | None) -> tuple[list[RequestResult], float]:
    for t in tasks:
        t["output_dir"] = output_dir
    wall0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(_do_request, tasks))
    return results, time.perf_counter() - wall0


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * pct))
    return sorted_vals[idx]


def _mean(vals: list[float]) -> float:
    return (sum(vals) / len(vals)) if vals else 0.0


def _playback_metrics(arrivals: list[float], durations: list[float]) -> dict:
    n = len(arrivals)
    if n == 0:
        return {
            "chunks": 0,
            "underruns": 0,
            "deadline_misses": 0,
            "gaps": [],
            "chunk_rtfs": [],
            "headrooms": [],
            "transitions": [],
        }

    underruns = 0
    deadline_misses = 0
    gaps: list[float] = []
    chunk_rtfs: list[float] = []
    headrooms: list[float] = []
    transitions: list[tuple[float, float, bool, bool]] = []
    playback_end = arrivals[0] + durations[0]
    for i in range(1, n):
        gap = arrivals[i] - arrivals[i - 1]
        gaps.append(gap)
        previous_audio = durations[i - 1]
        headrooms.append(previous_audio - gap)
        if gap > 0:
            # The next chunk must arrive while the *previous* chunk is playing.
            # Using durations[i] hid startup underruns when the first logical
            # codec chunk was one frame and the second was a larger steady hop.
            chunk_rtfs.append(previous_audio / gap)
        missed_deadline = gap > previous_audio
        if missed_deadline:
            deadline_misses += 1
        cumulative_underrun = arrivals[i] > playback_end
        if cumulative_underrun:
            underruns += 1
            playback_end = arrivals[i]
        transitions.append((previous_audio, gap, missed_deadline, cumulative_underrun))
        playback_end += durations[i]
    return {
        "chunks": n,
        "underruns": underruns,
        "deadline_misses": deadline_misses,
        "gaps": gaps,
        "chunk_rtfs": chunk_rtfs,
        "headrooms": headrooms,
        "transitions": transitions,
    }


def _summarize(results: list[RequestResult], wall_s: float, concurrency: int) -> dict:
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    sr = ok[0].sr if ok else 22050
    audio_s = sum(r.num_samples for r in ok) / sr if sr else 0.0
    ttfa_ms = sorted(r.ttfa_s * 1000.0 for r in ok)
    lat_s = sorted(r.elapsed_s for r in ok)

    itl_ms: list[float] = []
    chunk_rtfs: list[float] = []
    headrooms_ms: list[float] = []
    total_chunks = 0
    total_underruns = 0
    total_deadline_misses = 0
    reqs_with_underrun = 0
    transition_buckets: dict[int, dict] = {}
    for r in ok:
        pm = _playback_metrics(r.chunk_arrivals, r.chunk_durations)
        total_chunks += pm["chunks"]
        total_underruns += pm["underruns"]
        total_deadline_misses += pm["deadline_misses"]
        if pm["underruns"] > 0:
            reqs_with_underrun += 1
        itl_ms.extend(g * 1000.0 for g in pm["gaps"])
        chunk_rtfs.extend(pm["chunk_rtfs"])
        headrooms_ms.extend(value * 1000.0 for value in pm["headrooms"])
        for previous_audio, gap, missed_deadline, cumulative_underrun in pm["transitions"]:
            duration_ms = int(round(previous_audio * 1000.0))
            bucket = transition_buckets.setdefault(
                duration_ms, {"count": 0, "deadline_misses": 0, "underruns": 0, "gaps_ms": []}
            )
            bucket["count"] += 1
            bucket["deadline_misses"] += int(missed_deadline)
            bucket["underruns"] += int(cumulative_underrun)
            bucket["gaps_ms"].append(gap * 1000.0)
    itl_ms.sort()
    headrooms_ms.sort()
    transition_summary = []
    for duration_ms, bucket in sorted(transition_buckets.items()):
        gaps_ms = sorted(bucket.pop("gaps_ms"))
        transition_summary.append(
            {
                "duration_ms": duration_ms,
                **bucket,
                "gap_p95_ms": _percentile(gaps_ms, 0.95),
                "gap_max_ms": gaps_ms[-1],
            }
        )

    return {
        "concurrency": concurrency,
        "ok": len(ok),
        "failed": len(failed),
        "wall_s": wall_s,
        "audio_s": audio_s,
        "tput": len(ok) / wall_s if wall_s > 0 else 0.0,
        "rtf": audio_s / wall_s if wall_s > 0 else 0.0,
        "ttfa_mean_ms": _mean(ttfa_ms),
        "ttfa_p95_ms": _percentile(ttfa_ms, 0.95),
        "lat_mean_s": _mean(lat_s),
        "lat_p95_s": _percentile(lat_s, 0.95),
        "itl_mean_ms": _mean(itl_ms),
        "itl_p95_ms": _percentile(itl_ms, 0.95),
        "total_chunks": total_chunks,
        "total_underruns": total_underruns,
        "total_deadline_misses": total_deadline_misses,
        "reqs_with_underrun": reqs_with_underrun,
        "underrun_pct": (100.0 * total_underruns / total_chunks) if total_chunks else 0.0,
        "playback_rtf_mean": _mean(chunk_rtfs),
        "playback_headroom_mean_ms": _mean(headrooms_ms),
        "playback_headroom_p05_ms": _percentile(headrooms_ms, 0.05),
        "playback_transitions": transition_summary,
    }


def _print_detailed(s: dict) -> None:
    print(f"[concurrency={s['concurrency']}]  {s['ok']} ok / {s['failed']} failed")
    print(f"    req/s {s['tput']:.2f}  |  rtf {s['rtf']:.2f}x  (audio {s['audio_s']:.0f}s / wall {s['wall_s']:.2f}s)")
    print(f"    ttfa  mean {s['ttfa_mean_ms']:.1f}ms  p95 {s['ttfa_p95_ms']:.1f}ms")
    print(f"    lat   mean {s['lat_mean_s']:.2f}s   p95 {s['lat_p95_s']:.2f}s")
    print(f"    itl   mean {s['itl_mean_ms']:.1f}ms  p95 {s['itl_p95_ms']:.1f}ms")
    print(
        f"    playback  underruns {s['total_underruns']}/{s['total_chunks']} chunks "
        f"({s['underrun_pct']:.2f}%) in {s['reqs_with_underrun']}/{s['ok']} reqs  |  "
        f"deadlines missed {s['total_deadline_misses']}  |  "
        f"realtime factor mean {s['playback_rtf_mean']:.2f}x (previous chunk play / fetch)  |  "
        f"headroom mean {s['playback_headroom_mean_ms']:.1f}ms p05 {s['playback_headroom_p05_ms']:.1f}ms"
    )
    for row in s["playback_transitions"]:
        print(
            f"        after {row['duration_ms']:4d}ms audio: {row['count']:4d} gaps, "
            f"{row['deadline_misses']:3d} deadlines / {row['underruns']:3d} underruns, "
            f"gap p95 {row['gap_p95_ms']:.1f}ms max {row['gap_max_ms']:.1f}ms"
        )


def _print_summary(s: dict) -> None:
    print(
        f"concurrency={s['concurrency']}:  req/s {s['tput']:.2f},  "
        f"ttfa {s['ttfa_mean_ms']:.1f}ms,  itl {s['itl_mean_ms']:.1f}ms,  "
        f"rtf {s['rtf']:.2f}x,  underrun {s['underrun_pct']:.1f}%,  "
        f"{s['ok']} ok / {s['failed']} failed"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the EasyMagpieTTS HTTP server")
    parser.add_argument("--text-file", required=True, help="Path to file with '<uttid>\\t<text>' per line")
    parser.add_argument("-n", "--num-requests", type=int, required=True, help="Requests per concurrency level")
    parser.add_argument("-c", "--concurrency", type=int, nargs="+", default=[4], help="Concurrency (process) levels")
    parser.add_argument("--url", default="http://localhost:8091", help="Server base URL (default: %(default)s)")
    conditioning = parser.add_mutually_exclusive_group()
    conditioning.add_argument("--speaker-id", default=None, help="Speaker id (default: server default)")
    conditioning.add_argument("--reference-audio", help="WAV reference for zero-shot synthesis")
    parser.add_argument(
        "--randomize-reference-audio",
        action="store_true",
        help="Perturb the reference for every request to bypass audio caches",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--sample-rate", type=int, default=22050, help="Raw PCM sample rate (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=300, help="Per-request timeout, s (default: 300)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup phase (concurrency requests)")
    parser.add_argument("--output-dir", default=None, help="If set, write each waveform to <output-dir>/<uttid>.wav")
    args = parser.parse_args()
    if args.randomize_reference_audio and args.reference_audio is None:
        parser.error("--randomize-reference-audio requires --reference-audio")

    items = _load_items(args.text_file)
    if not items:
        print(f"ERROR: no usable lines found in {args.text_file}")
        return

    output_dir: str | None = None
    if args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        output_dir = args.output_dir

    print(
        f"Loaded {len(items)} utterances; {args.num_requests} req/level; concurrency {args.concurrency}; url {args.url}"
    )

    summaries = []
    for concurrency in args.concurrency:
        if not args.no_warmup:
            warmup = _make_tasks(
                items,
                concurrency,
                args.url,
                args.speaker_id,
                args.max_new_tokens,
                args.sample_rate,
                args.timeout,
                None,
                reference_audio=args.reference_audio,
                randomize_reference_audio=args.randomize_reference_audio,
            )
            _run_level(warmup, concurrency, None)

        tasks = _make_tasks(
            items,
            args.num_requests,
            args.url,
            args.speaker_id,
            args.max_new_tokens,
            args.sample_rate,
            args.timeout,
            output_dir,
            reference_audio=args.reference_audio,
            randomize_reference_audio=args.randomize_reference_audio,
        )
        results, wall = _run_level(tasks, concurrency, output_dir)
        summary = _summarize(results, wall, concurrency)
        summaries.append(summary)
        _print_detailed(summary)

    print("\n=== Summary ===")
    for s in summaries:
        _print_summary(s)


if __name__ == "__main__":
    main()
