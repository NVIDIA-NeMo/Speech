#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Rebind an index pack to an equivalent relocated input configuration.

Index-pack collection keys include the declarative source specification. When
an immutable data tree moves, concrete segment paths and the source
specification may both change. A path-relocated pack therefore needs new
collection keys even though its offset payload remains valid.

This tool fails closed unless the target configuration describes exactly the
same ordered collections, storage kinds, offset modes, and concrete paths as
the input pack. It then changes only the collection keys and header layout
digest, validates every payload checksum, and publishes atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import click
from lhotse.index_pack import (
    _COLLECTION,
    _COLLECTION_FIXED_ARRAY,
    _COLLECTION_PATHS_ONLY,
    _HEADER,
    _HEADER_SIZE,
    _SEGMENT,
    _SEGMENT_FIXED_ARRAY,
    _SEGMENT_PATH_ONLY,
    _SEQUENCE,
    IndexPack,
    IndexPackCollectionSpec,
    _StringTableBuilder,
)
from omegaconf import DictConfig, ListConfig

from scripts.dataloading.build_indexes import (
    _NO_INDEX_TYPES,
    _TRANSFORM_TYPES,
    WDS_TAR_V2,
    _flatten_path_spec,
    _load_input_cfg,
    _resolve_input_cfg,
)
from scripts.dataloading.convert_indexes_to_idxpack import NativeTarOrdinalMapSpec, discover_pack_collections

from nemo.collections.common.data.lhotse.nemo_tar_routing import NEMO_TAR_ORDINAL_MAP_KIND


@dataclass(frozen=True)
class _ObservedCollection:
    key: bytes
    kind: str
    paths: tuple[str, ...]
    offsets_required: bool
    sequence_start: int
    segment_ids: tuple[int, ...]
    cumulative_ends: tuple[int, ...]
    total_records: int
    is_array: bool
    shard_lengths: tuple[int, ...]


@dataclass(frozen=True)
class _ObservedSegment:
    path: str
    offsets_position: int
    flags: int
    offsets_count: int
    source_size: int
    offsets_size: int
    checksum: int
    metadata_flags: int

    @property
    def offsets_required(self) -> bool:
        return not self.is_array and not bool(self.flags & _SEGMENT_PATH_ONLY)

    @property
    def is_array(self) -> bool:
        return bool(self.flags & _SEGMENT_FIXED_ARRAY)


def _read_exact(stream, size: int, offset: int) -> bytes:
    stream.seek(offset)
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"Truncated index-pack read at {offset}: wanted {size}, got {len(value)}")
    return value


def _decode_string(stream, position: int, length: int) -> str:
    return _read_exact(stream, length, position).decode("utf-8")


def _read_pack_layout(
    path: Path,
) -> tuple[
    list[_ObservedCollection],
    tuple,
    tuple[tuple[int, int], ...],
    tuple[_ObservedSegment, ...],
]:
    with path.open("rb") as stream:
        header_blob = _read_exact(stream, _HEADER_SIZE, 0)
        header = _HEADER.unpack_from(header_blob)
        (
            _magic,
            _version,
            _header_size,
            collection_offset,
            num_collections,
            sequence_offset,
            num_sequences,
            segment_offset,
            num_segments,
            _strings_offset,
            _strings_size,
            _offsets_offset,
            _offsets_size,
            _layout_hash,
        ) = header
        collection_rows = [
            _COLLECTION.unpack(_read_exact(stream, _COLLECTION.size, collection_offset + index * _COLLECTION.size))
            for index in range(num_collections)
        ]
        sequences = [
            _SEQUENCE.unpack(_read_exact(stream, _SEQUENCE.size, sequence_offset + index * _SEQUENCE.size))
            for index in range(num_sequences)
        ]
        segments = [
            _SEGMENT.unpack(_read_exact(stream, _SEGMENT.size, segment_offset + index * _SEGMENT.size))
            for index in range(num_segments)
        ]
        observed_segments = tuple(
            _ObservedSegment(
                path=_decode_string(stream, row[0], row[2]),
                offsets_position=row[1],
                flags=row[3],
                offsets_count=row[4],
                source_size=row[5],
                offsets_size=row[6],
                checksum=row[7],
                metadata_flags=row[8],
            )
            for row in segments
        )

        observed = []
        for row in collection_rows:
            key, sequence_start, sequence_count, total, kind_position, kind_length, flags = row
            paths = []
            segment_ids = []
            cumulative_ends = []
            shard_lengths = []
            previous_cumulative_end = 0
            for sequence_index in range(sequence_start, sequence_start + sequence_count):
                segment_id, cumulative_end = sequences[sequence_index]
                if segment_id >= len(observed_segments):
                    raise ValueError(f"Collection references invalid segment {segment_id}")
                paths.append(observed_segments[segment_id].path)
                segment_ids.append(segment_id)
                cumulative_ends.append(cumulative_end)
                shard_lengths.append(cumulative_end - previous_cumulative_end)
                previous_cumulative_end = cumulative_end
            is_array = bool(flags & _COLLECTION_FIXED_ARRAY)
            observed.append(
                _ObservedCollection(
                    key=key,
                    kind=_decode_string(stream, kind_position, kind_length),
                    paths=tuple(paths),
                    offsets_required=not is_array and not bool(flags & _COLLECTION_PATHS_ONLY),
                    sequence_start=sequence_start,
                    segment_ids=tuple(segment_ids),
                    cumulative_ends=tuple(cumulative_ends),
                    total_records=total,
                    is_array=is_array,
                    shard_lengths=tuple(shard_lengths),
                )
            )
    return observed, header, tuple(sequences), observed_segments


def _read_ordered_collections(path: Path) -> tuple[list[_ObservedCollection], tuple]:
    observed, header, _sequences, _segments = _read_pack_layout(path)
    return observed, header


def _target_is_array(spec) -> bool:
    return isinstance(spec, NativeTarOrdinalMapSpec)


def _target_sequence_count(spec) -> int:
    return spec.sequence_count if _target_is_array(spec) else len(spec.paths)


def _target_layout_hash(observed: Sequence[_ObservedCollection], target: Sequence) -> bytes:
    """Compute the Lhotse layout hash without requiring array build sidecars."""
    digest = hashlib.sha256()
    for actual, expected in zip(observed, target, strict=True):
        digest.update(expected.key)
        if actual.is_array:
            digest.update(b"\x02")
            digest.update(struct.pack("<Q", expected.sequence_count))
            for shard_length in actual.shard_lengths:
                digest.update(struct.pack("<Q", shard_length))
            continue
        digest.update(bytes((expected.offsets_required,)))
        digest.update(struct.pack("<Q", len(expected.paths)))
        for path in expected.paths:
            encoded = path.encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
    return digest.digest()


def _routes_for_source_topology(
    native_routes: Sequence[NativeTarOrdinalMapSpec], observed: Sequence[_ObservedCollection]
) -> list[NativeTarOrdinalMapSpec]:
    observed_routes = [
        collection for collection in observed if collection.is_array and collection.kind == NEMO_TAR_ORDINAL_MAP_KIND
    ]
    if not observed_routes:
        return []
    if len(observed_routes) != len(native_routes):
        raise ValueError(
            "Native-tar route collection count changed during rebinding: "
            f"pack={len(observed_routes)}, target={len(native_routes)}"
        )
    return list(native_routes)


def discover_rebind_pack_collections(
    entry,
    observed: Sequence[_ObservedCollection],
    *,
    data_blend_dir: str | Path | None = None,
) -> list:
    """Discover ordinary collections followed by build-free native route descriptors."""
    native_routes = []
    collections = discover_pack_collections(
        entry,
        data_blend_dir=data_blend_dir,
        native_tar_ordinal_maps=native_routes,
    )
    return [*collections, *_routes_for_source_topology(native_routes, observed)]


def validate_rebinding_contract(
    observed: Sequence[_ObservedCollection],
    target: Sequence,
) -> None:
    """Require exact positional equivalence before rebinding identities."""
    if len(observed) != len(target):
        raise ValueError("Collection count changed during relocation: " f"pack={len(observed)}, target={len(target)}")
    target_keys = [spec.key for spec in target]
    if len(set(target_keys)) != len(target_keys):
        raise ValueError("Target configuration contains duplicate collection keys")

    for index, (actual, expected) in enumerate(zip(observed, target)):
        if actual.kind != expected.kind:
            raise ValueError(f"Collection {index} storage kind changed: {actual.kind!r} != {expected.kind!r}")
        if actual.is_array != _target_is_array(expected):
            raise ValueError(
                f"Collection {index} fixed-array mode changed: {actual.is_array} != {_target_is_array(expected)}"
            )
        if actual.is_array:
            if len(actual.shard_lengths) != expected.sequence_count:
                raise ValueError(
                    f"Collection {index} array shard count changed: "
                    f"{len(actual.shard_lengths)} != {expected.sequence_count}"
                )
            continue
        if actual.offsets_required != expected.offsets_required:
            raise ValueError(
                f"Collection {index} offset mode changed: " f"{actual.offsets_required} != {expected.offsets_required}"
            )
        if actual.paths != expected.paths:
            mismatch = next(
                (position for position, pair in enumerate(zip(actual.paths, expected.paths)) if pair[0] != pair[1]),
                min(len(actual.paths), len(expected.paths)),
            )
            actual_path = actual.paths[mismatch] if mismatch < len(actual.paths) else "<missing>"
            expected_path = expected.paths[mismatch] if mismatch < len(expected.paths) else "<missing>"
            raise ValueError(
                f"Collection {index} ordered paths changed at position {mismatch}: "
                f"{actual_path!r} != {expected_path!r}"
            )


def validate_relocation_contract(
    observed: Sequence[_ObservedCollection],
    segments: Sequence[_ObservedSegment],
    target: Sequence,
) -> tuple[str, ...]:
    """Validate positional equivalence and return one target path per segment."""
    if len(observed) != len(target):
        raise ValueError("Collection count changed during relocation: " f"pack={len(observed)}, target={len(target)}")
    target_keys = [spec.key for spec in target]
    if len(set(target_keys)) != len(target_keys):
        raise ValueError("Target configuration contains duplicate collection keys")

    relocated: list[str | None] = ["" if segment.is_array else None for segment in segments]
    target_owners: dict[tuple[str, bool], int] = {}
    for collection_index, (actual, expected) in enumerate(zip(observed, target)):
        if actual.kind != expected.kind:
            raise ValueError(
                f"Collection {collection_index} storage kind changed: " f"{actual.kind!r} != {expected.kind!r}"
            )
        if actual.is_array != _target_is_array(expected):
            raise ValueError(
                f"Collection {collection_index} fixed-array mode changed: "
                f"{actual.is_array} != {_target_is_array(expected)}"
            )
        if actual.is_array:
            if len(actual.shard_lengths) != expected.sequence_count:
                raise ValueError(
                    f"Collection {collection_index} array shard count changed: "
                    f"{len(actual.shard_lengths)} != {expected.sequence_count}"
                )
            continue
        if actual.offsets_required != expected.offsets_required:
            raise ValueError(
                f"Collection {collection_index} offset mode changed: "
                f"{actual.offsets_required} != {expected.offsets_required}"
            )
        if len(actual.paths) != len(expected.paths):
            raise ValueError(
                f"Collection {collection_index} path count changed: " f"{len(actual.paths)} != {len(expected.paths)}"
            )

        for shard_index, (segment_id, target_path) in enumerate(zip(actual.segment_ids, expected.paths, strict=True)):
            prior_path = relocated[segment_id]
            if prior_path is not None and prior_path != target_path:
                raise ValueError(
                    "Shared source segment maps to inconsistent target paths: "
                    f"segment={segment_id}, collection={collection_index}, "
                    f"shard={shard_index}, {prior_path!r} != {target_path!r}"
                )
            relocated[segment_id] = target_path

            identity = (target_path, segments[segment_id].offsets_required)
            prior_owner = target_owners.get(identity)
            if prior_owner is not None and prior_owner != segment_id:
                raise ValueError(
                    "Distinct source segments collapse onto one target identity: "
                    f"segments={prior_owner},{segment_id}, path={target_path!r}, "
                    f"offsets_required={identity[1]}"
                )
            target_owners[identity] = segment_id

    missing = [index for index, path in enumerate(relocated) if path is None]
    if missing:
        raise ValueError(f"Pack has unreferenced source segments: {missing[:10]}")
    return tuple(path for path in relocated if path is not None)


def _map_path(path: str, prefix_map: Mapping[str, str]) -> str:
    matches = [
        source for source in prefix_map if path == source.rstrip("/") or path.startswith(source.rstrip("/") + "/")
    ]
    if not matches:
        raise ValueError(f"No relocation prefix matches source path {path!r}")
    source = max(matches, key=len).rstrip("/")
    suffix = path[len(source) :].lstrip("/")
    target = prefix_map[source].rstrip("/")
    return f"{target}/{suffix}" if suffix else target


def discover_relocated_pack_collections(
    entry,
    observed: Sequence[_ObservedCollection],
    *,
    path_prefix_map: Mapping[str, str],
    data_blend_dir: str | Path | None = None,
) -> list[IndexPackCollectionSpec]:
    """Discover target collections without remote WDS catalog reads.

    Non-WDS collections use normal declarative discovery. Each target WDS leaf
    is paired positionally with the next WDS collection in the sealed source
    pack; its concrete shard paths are produced by an explicit longest-prefix
    mapping and must remain beneath the target ``data_dir`` declaration.
    """
    if not path_prefix_map:
        raise ValueError("Offline relocation requires at least one path-prefix mapping")
    normalized_map = {str(source).rstrip("/"): str(target).rstrip("/") for source, target in path_prefix_map.items()}
    if any(not source or not target for source, target in normalized_map.items()):
        raise ValueError("Relocation path-prefix mappings must be non-empty")

    source_wds = [collection for collection in observed if collection.kind == WDS_TAR_V2]
    target: list = []
    native_routes: list[NativeTarOrdinalMapSpec] = []
    wds_index = 0

    def walk(node) -> None:
        nonlocal wds_index
        if isinstance(node, (list, tuple, ListConfig)):
            for item in node:
                walk(item)
            return
        if not isinstance(node, (dict, DictConfig)):
            return

        typ = node.get("type")
        if typ in _NO_INDEX_TYPES:
            return
        if typ is None:
            for value in node.values():
                walk(value)
            return
        if typ == "group":
            nested = _resolve_input_cfg(node.get("input_cfg"), data_blend_dir)
            if nested is not None:
                walk(nested)
            return
        if typ in _TRANSFORM_TYPES:
            nested = _resolve_input_cfg(node.get("input_cfg"), data_blend_dir)
            if nested is not None:
                walk(nested)
                return

        if typ != "share_gpt_webdataset":
            discover_pack_collections(
                node,
                target,
                data_blend_dir=data_blend_dir,
                native_tar_ordinal_maps=native_routes,
            )
            return

        version = int(node.get("wds_sample_index_version", 1))
        if version != 2:
            raise ValueError("Offline WDS relocation requires wds_sample_index_version: 2; " f"got {version}")
        if wds_index >= len(source_wds):
            raise ValueError("Target configuration has more WDS collections than source pack")
        data_dir = node.get("data_dir")
        if data_dir is None:
            raise ValueError("Target WDS v2 entry is missing data_dir")
        roots = tuple(str(path).rstrip("/") for path in _flatten_path_spec(data_dir))
        if not roots:
            raise ValueError("Target WDS v2 data_dir expands to no roots")

        source_collection = source_wds[wds_index]
        mapped_paths = tuple(_map_path(path, normalized_map) for path in source_collection.paths)
        for path in mapped_paths:
            if not any(path.startswith(root + "/") for root in roots):
                raise ValueError(f"Relocated WDS shard {path!r} is outside target data_dir roots {roots!r}")
        target.append(
            IndexPackCollectionSpec(
                role="wds_tar",
                kind=WDS_TAR_V2,
                source_spec=data_dir,
                paths=mapped_paths,
            )
        )
        wds_index += 1

    walk(entry)
    if wds_index != len(source_wds):
        raise ValueError(
            "Target configuration has fewer WDS collections than source pack: "
            f"target={wds_index}, source={len(source_wds)}"
        )
    return [*target, *_routes_for_source_topology(native_routes, observed)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_relocated_payloads(segments: Sequence[_ObservedSegment], relocated_paths: Sequence[str]) -> None:
    """Prove byte identity for every changed local source path."""
    for segment_id, (segment, target_path) in enumerate(zip(segments, relocated_paths, strict=True)):
        if segment.is_array:
            continue
        if segment.path == target_path:
            continue
        if "://" in segment.path or "://" in target_path:
            raise ValueError(
                "Cannot prove relocated remote payload identity for segment "
                f"{segment_id}: {segment.path!r} -> {target_path!r}. "
                "Provide an external transfer attestation and rerun with "
                "--trust-relocated-payloads."
            )
        source_path = Path(segment.path)
        target = Path(target_path)
        if not source_path.is_file() or not target.is_file():
            raise ValueError(
                "Cannot verify relocated payload because a source is missing: "
                f"segment={segment_id} source={source_path} target={target}"
            )
        source_size = source_path.stat().st_size
        target_size = target.stat().st_size
        if source_size != segment.source_size or target_size != segment.source_size:
            raise ValueError(
                "Relocated payload size mismatch for segment "
                f"{segment_id}: packed={segment.source_size} "
                f"source={source_size} target={target_size}"
            )
        if _sha256_file(source_path) != _sha256_file(target):
            raise ValueError(
                f"Relocated payload content mismatch for segment {segment_id}: " f"{source_path} != {target}"
            )


def rebind_idxpack_collections(
    source: str | Path,
    output: str | Path,
    target: Sequence,
    *,
    verify_payloads: bool = True,
) -> dict:
    """Publish ``source`` with collection identities derived from ``target``."""
    source = Path(source)
    output = Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Source and output index-pack paths must differ")
    if output.exists():
        raise FileExistsError(f"Output index pack already exists: {output}")

    observed, header = _read_ordered_collections(source)
    validate_rebinding_contract(observed, target)
    target_layout_hash = _target_layout_hash(observed, target)

    with IndexPack(source) as pack:
        if verify_payloads:
            for segment_id in range(pack.num_segments):
                pack.verify_segment(segment_id)

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.tmp.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as destination, source.open("rb") as origin:
            shutil.copyfileobj(origin, destination, length=8 * 1024 * 1024)
            collection_offset = header[3]
            for index, (actual, expected) in enumerate(zip(observed, target)):
                if actual.key == expected.key:
                    continue
                destination.seek(collection_offset + index * _COLLECTION.size)
                row = list(_COLLECTION.unpack(_read_exact(destination, _COLLECTION.size, destination.tell())))
                row[0] = expected.key
                destination.seek(collection_offset + index * _COLLECTION.size)
                destination.write(_COLLECTION.pack(*row))

            rebound_header = list(header)
            rebound_header[-1] = target_layout_hash
            destination.seek(0)
            destination.write(_HEADER.pack(*rebound_header))
            destination.flush()
            os.fsync(destination.fileno())

        os.chmod(temporary, source.stat().st_mode & 0o777)
        with IndexPack(temporary, expected_layout_hash=target_layout_hash) as pack:
            for spec in target:
                collection = pack.collection(spec.key)
                if collection.sequence_count != _target_sequence_count(spec):
                    raise ValueError("Rebound collection path count changed during publication")
                if _target_is_array(spec):
                    if not collection.is_array:
                        raise ValueError("Rebound fixed-array collection changed storage mode")
                else:
                    for shard_index, expected_path in enumerate(spec.paths):
                        if collection.path_for_shard(shard_index) != expected_path:
                            raise ValueError("Rebound collection path changed during publication")
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"Output index pack already exists: {output}") from error
        temporary.unlink()
        directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "source": str(source),
        "output": str(output),
        "collections": len(target),
        "keys_changed": sum(actual.key != expected.key for actual, expected in zip(observed, target)),
        "segments": IndexPack(output).num_segments,
        "layout_sha256": target_layout_hash.hex(),
        "output_size": output.stat().st_size,
        "output_sha256": _sha256_file(output),
        "payloads_verified": verify_payloads,
    }


def relocate_idxpack_collections(
    source: str | Path,
    output: str | Path,
    target: Sequence,
    *,
    trust_relocated_payloads: bool = False,
) -> dict:
    """Relocate catalog paths while preserving and verifying packed offsets."""
    source = Path(source)
    output = Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Source and output index-pack paths must differ")
    if output.exists():
        raise FileExistsError(f"Output index pack already exists: {output}")

    observed, header, sequences, segments = _read_pack_layout(source)
    relocated_paths = validate_relocation_contract(observed, segments, target)
    target_layout_hash = _target_layout_hash(observed, target)
    if not trust_relocated_payloads:
        _verify_relocated_payloads(segments, relocated_paths)

    strings = _StringTableBuilder()
    kind_positions = [strings.add(spec.kind) for spec in target]
    path_positions = [strings.add(path) for path in relocated_paths]
    string_blob = bytes(strings.data)

    collection_offset = _HEADER_SIZE
    sequence_offset = collection_offset + len(observed) * _COLLECTION.size
    segment_offset = sequence_offset + len(sequences) * _SEQUENCE.size
    strings_offset = segment_offset + len(segments) * _SEGMENT.size
    offsets_offset = strings_offset + len(string_blob)
    offsets_offset += (-offsets_offset) % 8
    offsets_size = sum(segment.offsets_size for segment in segments)
    if offsets_size != header[12]:
        raise ValueError(f"Pack offset payload size disagrees with its header: {offsets_size} != {header[12]}")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.tmp.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as destination, source.open("rb") as origin:
            destination.write(
                _HEADER.pack(
                    header[0],
                    header[1],
                    _HEADER_SIZE,
                    collection_offset,
                    len(observed),
                    sequence_offset,
                    len(sequences),
                    segment_offset,
                    len(segments),
                    strings_offset,
                    len(string_blob),
                    offsets_offset,
                    offsets_size,
                    target_layout_hash,
                )
            )
            destination.write(b"\0" * (_HEADER_SIZE - destination.tell()))

            for actual, expected, (kind_relative, kind_length) in zip(observed, target, kind_positions, strict=True):
                destination.write(
                    _COLLECTION.pack(
                        expected.key,
                        actual.sequence_start,
                        len(actual.segment_ids),
                        actual.total_records,
                        strings_offset + kind_relative,
                        kind_length,
                        (
                            _COLLECTION_FIXED_ARRAY
                            if actual.is_array
                            else 0 if expected.offsets_required else _COLLECTION_PATHS_ONLY
                        ),
                    )
                )
            for sequence in sequences:
                destination.write(_SEQUENCE.pack(*sequence))
            destination.write(b"\0" * (len(segments) * _SEGMENT.size))
            destination.write(string_blob)
            if destination.tell() < offsets_offset:
                destination.write(b"\0" * (offsets_offset - destination.tell()))

            relocated_segment_rows = []
            payload_cursor = offsets_offset
            for segment_id, (segment, (path_relative, path_length)) in enumerate(
                zip(segments, path_positions, strict=True)
            ):
                origin.seek(segment.offsets_position)
                remaining = segment.offsets_size
                checksum = 0
                while remaining:
                    chunk = origin.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(
                            f"Truncated source payload for segment {segment_id}: " f"{remaining} bytes remain"
                        )
                    destination.write(chunk)
                    checksum = zlib.crc32(chunk, checksum)
                    remaining -= len(chunk)
                checksum &= 0xFFFFFFFF
                if checksum != segment.checksum:
                    raise ValueError(
                        f"Source segment {segment_id} CRC mismatch: " f"{checksum:#010x} != {segment.checksum:#010x}"
                    )
                relocated_segment_rows.append(
                    (
                        strings_offset + path_relative,
                        payload_cursor,
                        path_length,
                        segment.flags,
                        segment.offsets_count,
                        segment.source_size,
                        segment.offsets_size,
                        checksum,
                        segment.metadata_flags,
                    )
                )
                payload_cursor += segment.offsets_size

            if destination.tell() != offsets_offset + offsets_size:
                raise AssertionError(
                    "Internal relocated pack size mismatch: "
                    f"{destination.tell()} != {offsets_offset + offsets_size}"
                )
            destination.seek(segment_offset)
            for row in relocated_segment_rows:
                destination.write(_SEGMENT.pack(*row))
            destination.flush()
            os.fsync(destination.fileno())

        os.chmod(temporary, source.stat().st_mode & 0o777)
        with IndexPack(temporary, expected_layout_hash=target_layout_hash) as pack:
            for segment_id in range(pack.num_segments):
                pack.verify_segment(segment_id)
            for spec in target:
                collection = pack.collection(spec.key)
                if collection.sequence_count != _target_sequence_count(spec):
                    raise ValueError("Relocated collection path count changed")
                if _target_is_array(spec):
                    if not collection.is_array:
                        raise ValueError("Relocated fixed-array collection changed storage mode")
                else:
                    for shard_index, expected_path in enumerate(spec.paths):
                        if collection.path_for_shard(shard_index) != expected_path:
                            raise ValueError("Relocated collection path changed")
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"Output index pack already exists: {output}") from error
        temporary.unlink()
        directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "source": str(source),
        "output": str(output),
        "collections": len(target),
        "keys_changed": sum(actual.key != expected.key for actual, expected in zip(observed, target)),
        "paths_changed": sum(
            actual != expected
            for actual, expected in zip((segment.path for segment in segments), relocated_paths, strict=True)
        ),
        "segments": len(segments),
        "layout_sha256": target_layout_hash.hex(),
        "output_size": output.stat().st_size,
        "output_sha256": _sha256_file(output),
        "payloads_verified": not trust_relocated_payloads,
        "relocation_trusted": trust_relocated_payloads,
    }


@click.command(context_settings={"show_default": True})
@click.argument("input_cfg", type=click.Path(exists=True, dir_okay=False))
@click.option("--source-pack", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(dir_okay=False))
@click.option("--data-blend-dir", type=click.Path(file_okay=False))
@click.option("--report", type=click.Path(dir_okay=False))
@click.option("--skip-payload-verification", is_flag=True)
@click.option(
    "--relocate-paths",
    is_flag=True,
    help="Allow a one-to-one positional path relocation while preserving packed offsets.",
)
@click.option(
    "--trust-relocated-payloads",
    is_flag=True,
    help="Acknowledge externally attested payload identity when local byte verification is impossible.",
)
@click.option(
    "--path-prefix-map",
    "path_prefix_maps",
    nargs=2,
    multiple=True,
    metavar="SOURCE_PREFIX TARGET_PREFIX",
    help="Offline WDS shard relocation mapping; repeat for multiple source roots.",
)
def main(
    input_cfg: str,
    source_pack: str,
    output: str,
    data_blend_dir: str | None,
    report: str | None,
    skip_payload_verification: bool,
    relocate_paths: bool,
    path_prefix_maps: tuple[tuple[str, str], ...],
    trust_relocated_payloads: bool,
) -> None:
    """Rebind SOURCE_PACK to the collection contract in INPUT_CFG."""
    try:
        config = _load_input_cfg(input_cfg, data_blend_dir)
        if relocate_paths:
            if skip_payload_verification:
                raise ValueError("--skip-payload-verification cannot be used with --relocate-paths")
            if path_prefix_maps:
                prefix_map: dict[str, str] = {}
                for source_prefix, target_prefix in path_prefix_maps:
                    prior = prefix_map.get(source_prefix)
                    if prior is not None and prior != target_prefix:
                        raise ValueError(f"Conflicting targets for source prefix {source_prefix!r}")
                    prefix_map[source_prefix] = target_prefix
                observed, _header, _sequences, _segments = _read_pack_layout(Path(source_pack))
                target = discover_relocated_pack_collections(
                    config,
                    observed,
                    path_prefix_map=prefix_map,
                    data_blend_dir=data_blend_dir,
                )
            else:
                observed, _header, _sequences, _segments = _read_pack_layout(Path(source_pack))
                target = discover_rebind_pack_collections(
                    config,
                    observed,
                    data_blend_dir=data_blend_dir,
                )
            result = relocate_idxpack_collections(
                source_pack,
                output,
                target,
                trust_relocated_payloads=trust_relocated_payloads,
            )
        else:
            if path_prefix_maps:
                raise ValueError("--path-prefix-map requires --relocate-paths")
            observed, _header, _sequences, _segments = _read_pack_layout(Path(source_pack))
            target = discover_rebind_pack_collections(
                config,
                observed,
                data_blend_dir=data_blend_dir,
            )
            if trust_relocated_payloads:
                raise ValueError("--trust-relocated-payloads requires --relocate-paths")
            result = rebind_idxpack_collections(
                source_pack,
                output,
                target,
                verify_payloads=not skip_payload_verification,
            )
        if report is not None:
            report_path = Path(report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, report_path)
        click.echo(json.dumps(result, sort_keys=True))
    except (OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
