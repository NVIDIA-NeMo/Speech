#!/usr/bin/env python3
"""Run an exact materialized-SFT census over newline-aligned byte ranges."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from count_materialized_sft_tokens import _finalize, _merge, _new_stats
from count_materialized_sft_tokens_parallel import publish_atomic


@dataclass(frozen=True)
class Range:
    file_index: int
    range_index: int
    path: Path
    source_bytes: int
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--expected-files", type=int, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--ranges-per-shard", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=40)
    parser.add_argument("--output", required=True)
    parser.add_argument("--success-marker", default=None)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--range-script", default=None)
    parser.add_argument("--additional-special-token", action="append", default=[])
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def newline_boundaries(path: Path, pieces: int) -> list[int]:
    size = path.stat().st_size
    boundaries = [0]
    with path.open("rb") as stream:
        for part in range(1, pieces):
            stream.seek(size * part // pieces)
            stream.readline()
            boundary = stream.tell()
            if boundary <= boundaries[-1] or boundary >= size:
                raise ValueError(f"Unable to form {pieces} nonempty ranges for {path}")
            boundaries.append(boundary)
    boundaries.append(size)
    return boundaries


def resolve_ranges(args: argparse.Namespace) -> list[Range]:
    paths = [Path(raw) for raw in args.shard]
    if len(paths) != args.expected_files or len({str(p.resolve()) for p in paths}) != len(paths):
        raise ValueError("Shard closure is not the expected distinct file count")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("One or more shard paths do not exist")
    total_bytes = sum(path.stat().st_size for path in paths)
    if total_bytes != args.expected_bytes:
        raise ValueError(f"Expected bytes={args.expected_bytes}, observed {total_bytes}")
    ranges = []
    for file_index, path in enumerate(paths):
        size = path.stat().st_size
        boundaries = newline_boundaries(path, args.ranges_per_shard)
        for range_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            ranges.append(Range(file_index, range_index, path, size, start, end))
    if sum(item.end - item.start for item in ranges) != total_bytes:
        raise ValueError("Range byte closure does not equal source byte closure")
    return ranges


def command_for(args: argparse.Namespace, script: Path, item: Range, report: Path) -> list[str]:
    command = [
        sys.executable,
        str(script),
        "--tokenizer",
        args.tokenizer,
        "--path",
        str(item.path),
        "--byte-start",
        str(item.start),
        "--byte-end",
        str(item.end),
        "--range-index",
        str(item.range_index),
        "--output",
        str(report),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    for token in args.additional_special_token:
        command.extend(["--additional-special-token", token])
    return command


def run_ranges(args: argparse.Namespace, ranges: list[Range], script: Path, work: Path) -> list[dict]:
    pending = list(ranges)
    running: dict[subprocess.Popen, tuple[Range, Path, Path, object]] = {}
    reports: dict[tuple[int, int], dict] = {}
    try:
        while pending or running:
            while pending and len(running) < args.max_workers:
                item = pending.pop(0)
                stem = f"file-{item.file_index:02d}-range-{item.range_index:02d}"
                report = work / f"{stem}.json"
                log = work / f"{stem}.log"
                stream = log.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command_for(args, script, item, report),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running[process] = (item, report, log, stream)
                print(json.dumps({"event": "range_task_start", "file_index": item.file_index,
                                  "range_index": item.range_index, "path": str(item.path),
                                  "byte_start": item.start, "byte_end": item.end}, sort_keys=True), flush=True)
            completed = [process for process in running if process.poll() is not None]
            if not completed:
                time.sleep(0.5)
                continue
            for process in completed:
                item, report, log, stream = running.pop(process)
                stream.close()
                if process.returncode:
                    for other in running:
                        other.terminate()
                    for other, (_, _, _, other_stream) in running.items():
                        other.wait(timeout=30)
                        other_stream.close()
                    tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
                    raise RuntimeError(f"Range {item.file_index}:{item.range_index} failed: {tail}")
                data = json.loads(report.read_text())
                reports[(item.file_index, item.range_index)] = data
                print(json.dumps({"event": "range_task_done", "file_index": item.file_index,
                                  "range_index": item.range_index}, sort_keys=True), flush=True)
        return [reports[(item.file_index, item.range_index)] for item in ranges]
    finally:
        for _, _, _, stream in running.values():
            if not stream.closed:
                stream.close()


def merge_reports(args: argparse.Namespace, ranges: list[Range], reports: list[dict]) -> dict:
    aggregate = _new_stats()
    by_file = [_new_stats() for _ in range(args.expected_files)]
    summaries = []
    for item, report in zip(ranges, reports, strict=True):
        expected = {
            "tokenizer": args.tokenizer,
            "additional_special_tokens": args.additional_special_token,
            "chunk_tokenization_validated": True,
            "path": str(item.path),
            "source_bytes": item.source_bytes,
            "byte_start": item.start,
            "byte_end": item.end,
            "range_index": item.range_index,
        }
        for key, value in expected.items():
            if report.get(key) != value:
                raise ValueError(f"Range {item.file_index}:{item.range_index} {key} mismatch")
        stats = dict(report["stats"])
        stats["role_messages"] = Counter(stats["role_messages"])
        if stats["files"] != 0 or stats["bytes"] != item.end - item.start:
            raise ValueError(f"Range {item.file_index}:{item.range_index} accounting mismatch")
        _merge(aggregate, stats)
        _merge(by_file[item.file_index], stats)
        summaries.append({"file_index": item.file_index, "range_index": item.range_index,
                          "path": str(item.path), "byte_start": item.start, "byte_end": item.end,
                          "packed_rows": stats["packed_rows"], "total_tokens": stats["total_tokens"],
                          "assistant_tokens": stats["assistant_tokens"]})
    aggregate["files"] = args.expected_files
    if aggregate["bytes"] != args.expected_bytes:
        raise ValueError("Merged range bytes do not match accepted closure")
    component = _finalize(aggregate)
    return {
        "schema_version": 3,
        "metric": "exact identity-preformatted tokens; assistant tokens selected by top-level role",
        "tokenizer": args.tokenizer,
        "additional_special_tokens": args.additional_special_token,
        "chunk_tokenization_validated": True,
        "range_partitions": summaries,
        "components": {"saffron": component},
        "total": component,
    }


def main() -> None:
    args = parse_args()
    if args.ranges_per_shard < 2 or args.max_workers < 1:
        raise ValueError("ranges-per-shard must be >=2 and max-workers must be positive")
    ranges = resolve_ranges(args)
    script = Path(args.range_script) if args.range_script else Path(__file__).with_name(
        "count_materialized_sft_token_range.py"
    )
    work = Path(args.work_dir)
    if work.exists():
        raise FileExistsError(f"Work directory already exists: {work}")
    work.mkdir(parents=True)
    try:
        reports = run_ranges(args, ranges, script, work)
        merged = merge_reports(args, ranges, reports)
        output = Path(args.output)
        marker = Path(args.success_marker or f"{args.output}.SUCCESS")
        publish_atomic(merged, output=output, success_marker=marker)
        print(json.dumps({"event": "range_census_published", "output": str(output),
                          "success_marker": str(marker), "total_tokens": merged["total"]["total_tokens"],
                          "assistant_tokens": merged["total"]["assistant_tokens"]}, sort_keys=True), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
