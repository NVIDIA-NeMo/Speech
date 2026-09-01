#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Exhaustively validate JSON records addressed by a Lhotse index pack."""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import click
from lhotse.audio.source import resolve_s3_to_local_mirror
from lhotse.index_pack import IndexPack, IndexPackCollectionSpec
from lhotse.packed_lazy import read_packed_range
from lhotse.utils import is_valid_url

from nemo.collections.common.data.lhotse.indexed_adapters import read_exact_range


@dataclass(frozen=True)
class IndexPackRecordValidationSummary:
    records_checked: int
    jsonl_collections_checked: int
    skip_marker_records: int
    top_level_skip_marker_records: int
    custom_skip_marker_records: int
    errors: int
    errors_reported: int


class IndexPackRecordValidationError(ValueError):
    """Raised after a complete scan finds one or more unreadable JSON records."""

    def __init__(self, summary: IndexPackRecordValidationSummary):
        self.summary = summary
        super().__init__(
            "Index-pack JSON record validation failed: "
            f"errors={summary.errors} records_checked={summary.records_checked} "
            f"jsonl_collections_checked={summary.jsonl_collections_checked} "
            f"skip_marker_records={summary.skip_marker_records} "
            f"errors_reported={summary.errors_reported}"
        )


@dataclass(frozen=True)
class _CollectionDescriptor:
    key: bytes
    role: str
    kind: str


@dataclass(frozen=True)
class _PartitionValidationResult:
    summary: IndexPackRecordValidationSummary
    messages: tuple[str, ...]


def _collection_descriptors(
    pack: IndexPack,
    collection_specs: Iterable[IndexPackCollectionSpec] | None,
) -> list[_CollectionDescriptor]:
    if collection_specs is not None:
        return [_CollectionDescriptor(key=spec.key, role=spec.role, kind=spec.kind) for spec in collection_specs]

    # The pack format persists keys and kinds, but not the human-readable role.
    # IndexPack has no public collection enumeration API yet, so the pack-only
    # CLI uses its catalog. Workflow callers pass specs and retain role labels.
    return [
        _CollectionDescriptor(key=key, role="<not-stored-in-pack>", kind=row[3])
        for key, row in pack._collections.items()
    ]


def _read_record(pack: IndexPack, path: str, start: int, end: int) -> bytes:
    read_path = resolve_s3_to_local_mirror(path) if path.startswith("s3://") else path
    if is_valid_url(read_path):
        return read_exact_range(read_path, start, end)
    return read_packed_range(pack, read_path, start, end)


class _ShardRecordReader:
    """Reuse one remote object reader for every record in a JSONL shard."""

    def __init__(self, pack: IndexPack, path: str):
        self.pack = pack
        self.path = path
        self.read_path = resolve_s3_to_local_mirror(path) if path.startswith("s3://") else path
        self.source = None

    def __enter__(self):
        if self.read_path.startswith(("ais://", "s3://")):
            from lhotse.ais import AISRangeReader

            self.source = AISRangeReader(self.read_path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.source is not None:
            self.source.close()

    def read(self, path: str, start: int, end: int) -> bytes:
        if path != self.path:
            raise ValueError(f"Shard reader path changed from {self.path!r} to {path!r}")
        if self.source is None:
            return _read_record(self.pack, path, start, end)
        if start < 0 or end < start:
            raise ValueError(f"Invalid byte range [{start}, {end})")
        if end > self.source.size:
            raise EOFError(
                f"Short indexed read from {path}: requested [{start}, {end}), " f"source size is {self.source.size}"
            )
        self.source.seek(start)
        chunks = []
        remaining = end - start
        while remaining:
            chunk = self.source.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) != end - start:
            raise EOFError(
                f"Short indexed read from {path}: requested [{start}, {end}), " f"received {len(data)} bytes"
            )
        return data


def _scan_json_record_partition(
    pack: IndexPack,
    descriptors: list[_CollectionDescriptor],
    *,
    worker_index: int,
    num_workers: int,
    max_reported_errors: int,
) -> _PartitionValidationResult:
    records_checked = 0
    collections_checked = 0
    skip_marker_records = 0
    top_level_skip_marker_records = 0
    custom_skip_marker_records = 0
    errors = 0
    messages: list[str] = []
    assigned_records = [0] * num_workers

    for descriptor in descriptors:
        if descriptor.kind != "jsonl":
            continue
        collection = pack.collection(descriptor.key)
        collection_has_work = False
        global_base = 0
        for shard_index in range(collection.sequence_count):
            shard_length = collection.shard_length(shard_index)
            assigned_worker = min(
                range(num_workers),
                key=lambda index: (assigned_records[index], index),
            )
            assigned_records[assigned_worker] += shard_length
            if assigned_worker != worker_index:
                global_base += shard_length
                continue
            collection_has_work = True
            source_path = collection.path_for_shard(shard_index)
            with _ShardRecordReader(pack, source_path) as record_reader:
                for local_index in range(shard_length):
                    global_index = global_base + local_index
                    start = None
                    end = None
                    try:
                        location = collection.locate_in_shard(shard_index, local_index)
                        start, end = location.start, location.end
                        raw = record_reader.read(location.path, start, end)
                        decoded = json.loads(raw.decode("utf-8"))
                        if not isinstance(decoded, dict):
                            raise TypeError("Expected a JSON object record, got " f"{type(decoded).__name__}")
                        top_level_marker = bool(decoded.get("_skipme", False))
                        custom = decoded.get("custom")
                        custom_marker = isinstance(custom, dict) and bool(custom.get("_skipme", False))
                        top_level_skip_marker_records += int(top_level_marker)
                        custom_skip_marker_records += int(custom_marker)
                        skip_marker_records += int(top_level_marker or custom_marker)
                    except (
                        OSError,
                        EOFError,
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                        TypeError,
                        ValueError,
                    ) as ex:
                        errors += 1
                        if max_reported_errors == 0 or len(messages) < max_reported_errors:
                            messages.append(
                                "INVALID_IDXPACK_JSON_RECORD "
                                f"collection_key={descriptor.key.hex()} "
                                f"collection_role={descriptor.role!r} "
                                f"collection_kind={descriptor.kind!r} "
                                f"shard_index={shard_index} "
                                f"source_path={source_path!r} "
                                f"global_index={global_index} "
                                f"local_index={local_index} "
                                f"byte_range=[{start}, {end}) "
                                f"exception={type(ex).__name__}: {ex}"
                            )
                    records_checked += 1
            global_base += shard_length
        collections_checked += int(collection_has_work)

    return _PartitionValidationResult(
        summary=IndexPackRecordValidationSummary(
            records_checked=records_checked,
            jsonl_collections_checked=collections_checked,
            skip_marker_records=skip_marker_records,
            top_level_skip_marker_records=top_level_skip_marker_records,
            custom_skip_marker_records=custom_skip_marker_records,
            errors=errors,
            errors_reported=len(messages),
        ),
        messages=tuple(messages),
    )


def _scan_json_record_partition_from_path(
    index_pack: str,
    descriptors: list[_CollectionDescriptor],
    worker_index: int,
    num_workers: int,
    max_reported_errors: int,
) -> _PartitionValidationResult:
    with IndexPack(index_pack) as pack:
        return _scan_json_record_partition(
            pack,
            descriptors,
            worker_index=worker_index,
            num_workers=num_workers,
            max_reported_errors=max_reported_errors,
        )


def validate_idxpack_json_records(
    index_pack: str | Path | IndexPack,
    collection_specs: Iterable[IndexPackCollectionSpec] | None = None,
    *,
    max_reported_errors: int = 100,
    report: Callable[[str], None] = print,
    num_workers: int = 1,
) -> IndexPackRecordValidationSummary:
    """Read and JSON-decode every record in every JSONL pack collection.

    The scan never stops at the first malformed record. ``max_reported_errors``
    limits diagnostic volume only; zero reports every error. The returned or
    raised summary always contains the complete error count.
    """
    if max_reported_errors < 0:
        raise ValueError("max_reported_errors must be non-negative")
    if num_workers < 1:
        raise ValueError("num_workers must be at least one")

    owns_pack = not isinstance(index_pack, IndexPack)
    pack = IndexPack(index_pack) if owns_pack else index_pack
    try:
        descriptors = _collection_descriptors(pack, collection_specs)
    finally:
        if owns_pack:
            pack.close()

    if num_workers == 1:
        active_pack = IndexPack(index_pack) if owns_pack else index_pack
        try:
            results = [
                _scan_json_record_partition(
                    active_pack,
                    descriptors,
                    worker_index=0,
                    num_workers=1,
                    max_reported_errors=max_reported_errors,
                )
            ]
        finally:
            if owns_pack:
                active_pack.close()
    else:
        if not owns_pack:
            raise ValueError("num_workers > 1 requires an index-pack path, not an open IndexPack")
        pack_path = str(index_pack)
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            results = list(
                executor.map(
                    _scan_json_record_partition_from_path,
                    [pack_path] * num_workers,
                    [descriptors] * num_workers,
                    range(num_workers),
                    [num_workers] * num_workers,
                    [max_reported_errors] * num_workers,
                )
            )

    messages = [message for result in results for message in result.messages]
    if max_reported_errors:
        messages = messages[:max_reported_errors]
    for message in messages:
        report(message)
    summary = IndexPackRecordValidationSummary(
        records_checked=sum(result.summary.records_checked for result in results),
        jsonl_collections_checked=sum(descriptor.kind == "jsonl" for descriptor in descriptors),
        skip_marker_records=sum(result.summary.skip_marker_records for result in results),
        top_level_skip_marker_records=sum(result.summary.top_level_skip_marker_records for result in results),
        custom_skip_marker_records=sum(result.summary.custom_skip_marker_records for result in results),
        errors=sum(result.summary.errors for result in results),
        errors_reported=len(messages),
    )
    if summary.errors:
        raise IndexPackRecordValidationError(summary)
    return summary


@click.command(context_settings={"show_default": True})
@click.argument("index_pack", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--max-reported-errors",
    type=click.IntRange(min=0),
    default=100,
    help="Maximum individual diagnostics; zero reports every error while still scanning all records.",
)
@click.option(
    "--num-workers",
    type=click.IntRange(min=1),
    default=1,
    help="Process workers for disjoint JSONL shard validation.",
)
def main(index_pack: str, max_reported_errors: int, num_workers: int) -> None:
    """Read and JSON-decode every JSONL record referenced by INDEX_PACK."""
    try:
        summary = validate_idxpack_json_records(
            index_pack,
            max_reported_errors=max_reported_errors,
            report=lambda message: click.echo(message, err=True),
            num_workers=num_workers,
        )
    except IndexPackRecordValidationError as ex:
        raise click.ClickException(str(ex)) from ex
    click.echo(
        f"Validated idxpack JSON records: {index_pack} "
        f"records_checked={summary.records_checked} "
        f"jsonl_collections_checked={summary.jsonl_collections_checked} "
        f"skip_marker_records={summary.skip_marker_records} "
        f"top_level_skip_marker_records={summary.top_level_skip_marker_records} "
        f"custom_skip_marker_records={summary.custom_skip_marker_records} errors=0"
    )


if __name__ == "__main__":
    main()
