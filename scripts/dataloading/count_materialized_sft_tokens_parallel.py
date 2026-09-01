"""Run exact materialized-SFT token censuses per shard and merge fail-closed."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ADDITIVE_FIELDS = (
    "files",
    "bytes",
    "packed_rows",
    "source_conversations",
    "messages",
    "total_tokens",
    "assistant_tokens",
    "system_chunks_with_multiple_wire_system_starts",
    "system_chunks_with_multiple_empty_wire_systems",
)


@dataclass(frozen=True)
class Shard:
    index: int
    path: Path
    bytes: int


@dataclass
class RunningShard:
    shard: Shard
    process: subprocess.Popen
    report: Path
    log: Path
    log_stream: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--expected-files", type=int, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--success-marker", default=None)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--counter-script", default=None)
    parser.add_argument("--additional-special-token", action="append", default=[])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    return parser.parse_args()


def resolve_shards(
    raw_paths: list[str], *, expected_files: int, expected_bytes: int
) -> list[Shard]:
    paths = [Path(raw) for raw in raw_paths]
    if len(paths) != expected_files:
        raise ValueError(f"Expected {expected_files} shard arguments, got {len(paths)}")
    normalized = [str(path.resolve()) for path in paths]
    if len(set(normalized)) != len(paths):
        raise ValueError("Shard arguments must be distinct after path resolution")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing shard files: {missing}")
    shards = [
        Shard(index=index, path=path, bytes=path.stat().st_size)
        for index, path in enumerate(paths)
    ]
    actual_bytes = sum(shard.bytes for shard in shards)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Expected aggregate bytes={expected_bytes}, observed {actual_bytes}"
        )
    return shards


def _empty_stats() -> dict:
    return {
        **{field: 0 for field in ADDITIVE_FIELDS},
        "role_messages": Counter(),
        "min_row_tokens": None,
        "max_row_tokens": 0,
    }


def _validate_shard_report(
    report: dict, *, shard: Shard, tokenizer: str, special_tokens: list[str]
) -> dict:
    if report.get("schema_version") != 1:
        raise ValueError(f"Shard {shard.index} has unsupported schema_version")
    if report.get("tokenizer") != tokenizer:
        raise ValueError(f"Shard {shard.index} tokenizer mismatch")
    if report.get("additional_special_tokens") != special_tokens:
        raise ValueError(f"Shard {shard.index} special-token mismatch")
    if report.get("chunk_tokenization_validated") is not True:
        raise ValueError(f"Shard {shard.index} disabled chunk-tokenization parity")
    component = report.get("components", {}).get("saffron")
    if not isinstance(component, dict) or component != report.get("total"):
        raise ValueError(f"Shard {shard.index} component/total report mismatch")
    if component.get("files") != 1 or component.get("bytes") != shard.bytes:
        raise ValueError(f"Shard {shard.index} file/byte accounting mismatch")
    for field in ADDITIVE_FIELDS:
        value = component.get(field)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Shard {shard.index} invalid {field}={value!r}")
    if component["assistant_tokens"] > component["total_tokens"]:
        raise ValueError(f"Shard {shard.index} assistant tokens exceed total tokens")
    return component


def merge_reports(
    *,
    shards: list[Shard],
    reports: list[dict],
    tokenizer: str,
    special_tokens: list[str],
    expected_bytes: int,
) -> dict:
    if len(reports) != len(shards):
        raise ValueError(f"Expected {len(shards)} reports, got {len(reports)}")
    aggregate = _empty_stats()
    shard_summaries = []
    for shard, report in zip(shards, reports, strict=True):
        stats = _validate_shard_report(
            report,
            shard=shard,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
        )
        for field in ADDITIVE_FIELDS:
            aggregate[field] += stats[field]
        aggregate["role_messages"].update(stats["role_messages"])
        row_min = stats["min_row_tokens"]
        if row_min is not None:
            aggregate["min_row_tokens"] = (
                row_min
                if aggregate["min_row_tokens"] is None
                else min(aggregate["min_row_tokens"], row_min)
            )
        aggregate["max_row_tokens"] = max(
            aggregate["max_row_tokens"], stats["max_row_tokens"]
        )
        shard_summaries.append(
            {
                "index": shard.index,
                "path": str(shard.path),
                "bytes": shard.bytes,
                "packed_rows": stats["packed_rows"],
                "total_tokens": stats["total_tokens"],
                "assistant_tokens": stats["assistant_tokens"],
            }
        )
    if aggregate["files"] != len(shards):
        raise ValueError("Merged report file count does not match shard closure")
    if aggregate["bytes"] != expected_bytes:
        raise ValueError("Merged report byte count does not match accepted closure")
    aggregate["role_messages"] = dict(sorted(aggregate["role_messages"].items()))
    aggregate["mean_row_tokens"] = (
        aggregate["total_tokens"] / aggregate["packed_rows"]
        if aggregate["packed_rows"]
        else 0.0
    )
    aggregate["assistant_token_fraction"] = (
        aggregate["assistant_tokens"] / aggregate["total_tokens"]
        if aggregate["total_tokens"]
        else 0.0
    )
    return {
        "schema_version": 2,
        "metric": (
            "exact identity-preformatted tokens; assistant tokens selected "
            "by top-level role"
        ),
        "tokenizer": tokenizer,
        "additional_special_tokens": special_tokens,
        "chunk_tokenization_validated": True,
        "parallel_shards": shard_summaries,
        "components": {"saffron": dict(aggregate)},
        "total": dict(aggregate),
    }


def _counter_command(
    *, args: argparse.Namespace, counter: Path, shard: Shard, report: Path
) -> list[str]:
    command = [
        sys.executable,
        str(counter),
        "--tokenizer",
        args.tokenizer,
        "--component",
        f"saffron={shard.path}",
        "--expected-files",
        "saffron=1",
        "--output",
        str(report),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    for token in args.additional_special_token:
        command.extend(["--additional-special-token", token])
    return command


def run_shards(
    *, args: argparse.Namespace, shards: list[Shard], counter: Path, work_dir: Path
) -> list[dict]:
    running: list[RunningShard] = []
    try:
        for shard in shards:
            report = work_dir / f"shard-{shard.index:02d}.json"
            log = work_dir / f"shard-{shard.index:02d}.log"
            log_stream = log.open("w", encoding="utf-8")
            command = _counter_command(
                args=args, counter=counter, shard=shard, report=report
            )
            print(
                json.dumps(
                    {
                        "event": "shard_task_start",
                        "index": shard.index,
                        "path": str(shard.path),
                        "bytes": shard.bytes,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            process = subprocess.Popen(
                command,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append(
                RunningShard(shard, process, report, log, log_stream)
            )

        pending = set(range(len(running)))
        while pending:
            for index in list(pending):
                item = running[index]
                status = item.process.poll()
                if status is None:
                    continue
                item.log_stream.close()
                pending.remove(index)
                if status != 0:
                    for other_index in pending:
                        running[other_index].process.terminate()
                    for other_index in pending:
                        running[other_index].process.wait(timeout=30)
                        running[other_index].log_stream.close()
                    tail = "\n".join(
                        item.log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                    )
                    raise RuntimeError(
                        f"Shard {item.shard.index} counter failed with {status}:\n{tail}"
                    )
                if not item.report.is_file():
                    raise FileNotFoundError(
                        f"Shard {item.shard.index} exited successfully without report"
                    )
                print(
                    json.dumps(
                        {"event": "shard_task_done", "index": item.shard.index},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if pending:
                time.sleep(args.poll_seconds)
        return [json.loads(item.report.read_text()) for item in running]
    finally:
        for item in running:
            if not item.log_stream.closed:
                item.log_stream.close()


def publish_atomic(report: dict, *, output: Path, success_marker: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or success_marker.exists():
        raise FileExistsError(
            f"Refusing to overwrite output/marker: {output}, {success_marker}"
        )
    nonce = f"{os.getpid()}.{uuid.uuid4().hex}"
    staged_output = output.with_name(f".{output.name}.tmp.{nonce}")
    staged_marker = success_marker.with_name(f".{success_marker.name}.tmp.{nonce}")
    try:
        staged_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with staged_output.open("rb") as stream:
            os.fsync(stream.fileno())
        staged_marker.write_text(
            json.dumps(
                {
                    "schema_version": report["schema_version"],
                    "files": report["total"]["files"],
                    "bytes": report["total"]["bytes"],
                    "total_tokens": report["total"]["total_tokens"],
                    "assistant_tokens": report["total"]["assistant_tokens"],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with staged_marker.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(staged_output, output)
        os.replace(staged_marker, success_marker)
    finally:
        staged_output.unlink(missing_ok=True)
        staged_marker.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.expected_files < 1 or args.expected_bytes < 1:
        raise ValueError("--expected-files and --expected-bytes must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    shards = resolve_shards(
        args.shard,
        expected_files=args.expected_files,
        expected_bytes=args.expected_bytes,
    )
    print(
        json.dumps(
            {
                "event": "parallel_closure_verified",
                "files": len(shards),
                "bytes": sum(shard.bytes for shard in shards),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    counter = (
        Path(args.counter_script)
        if args.counter_script
        else Path(__file__).with_name("count_materialized_sft_tokens.py")
    )
    if not counter.is_file():
        raise FileNotFoundError(f"Counter script not found: {counter}")
    work_dir = Path(args.work_dir)
    if work_dir.exists():
        raise FileExistsError(f"Work directory already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    try:
        reports = run_shards(
            args=args, shards=shards, counter=counter, work_dir=work_dir
        )
        merged = merge_reports(
            shards=shards,
            reports=reports,
            tokenizer=args.tokenizer,
            special_tokens=args.additional_special_token,
            expected_bytes=args.expected_bytes,
        )
        output = Path(args.output)
        marker = Path(args.success_marker or f"{args.output}.SUCCESS")
        publish_atomic(merged, output=output, success_marker=marker)
        print(
            json.dumps(
                {
                    "event": "parallel_census_published",
                    "output": str(output),
                    "success_marker": str(marker),
                    "total_tokens": merged["total"]["total_tokens"],
                    "assistant_tokens": merged["total"]["assistant_tokens"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
