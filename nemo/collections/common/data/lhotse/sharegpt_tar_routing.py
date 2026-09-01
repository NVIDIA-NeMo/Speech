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
"""Sealed ShareGPT JSONL-to-unordered-tar routing indexes.

This module implements the ``.sgroute`` v2 contract. Runtime code only opens
and validates a sealed route; route construction is an explicit CPU-side
operation over indexed JSONL manifests and ordered, offset-bearing tar files.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import mmap
import os
import struct
import tarfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Self

from nemo.collections.common.data.lhotse.indexed_adapters import (
    _parse_pax_headers,
    indexed_source_metadata,
    read_exact_range,
)

SGROUTE_MAGIC = b"SGROUTE2"
SGROUTE_VERSION = 2
SGROUTE_HEADER_SIZE = 288
SGROUTE_FLAGS = 0b11
SGROUTE_RECORD_SIZE = 8

_HEADER = struct.Struct("<8sIIIIQQQQQQII32s32s32s32s32s32s16s")
_U64 = struct.Struct("<Q")
_ROUTE = struct.Struct("<II")
_AUDIO_FIELDS = ("sound", "speech", "ori_sound")
_DEFAULT_AUDIO_PLACEHOLDERS = ("<sound>", "<speech>")
_UINT32_MAX = (1 << 32) - 1
_REMOTE_MANIFEST_BATCH_BYTES = 64 << 20
_REMOTE_TAR_METADATA_MAX_BYTES = 8 << 20


@dataclass(frozen=True)
class TarRoute:
    """One route from a manifest audio reference to an ordered tar member."""

    tar_shard_index: int
    tar_member_local_index: int


@dataclass(frozen=True)
class NativeTarMemberRange:
    """A proven native-tar index range containing one regular member."""

    local_index: int
    name: str
    start: int
    end: int
    data_end: int


@dataclass(frozen=True)
class ShareGptTarRouteHeader:
    """Decoded immutable metadata from a ``.sgroute`` v2 header."""

    manifest_row_count: int
    route_record_count: int
    tar_shard_count: int
    row_offsets_offset: int
    routes_offset: int
    file_size: int
    manifest_spec_path_digest: bytes
    manifest_content_digest: bytes
    manifest_source_identity_digest: bytes
    tar_catalog_digest: bytes
    audio_prefix_map_digest: bytes
    payload_sha256: bytes


class ShareGptTarRoutingIndex:
    """Memory-mapped, self-validating ``.sgroute`` v2 reader."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("rb")
        try:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self.header, self._row_offsets = _validate_layout(self.path, self._mmap)
        except BaseException:
            mapping = getattr(self, "_mmap", None)
            if mapping is not None:
                mapping.close()
            self._file.close()
            raise

    def __len__(self) -> int:
        return self.header.manifest_row_count

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Path]:
        """Serialize only the path; worker processes reopen and revalidate the mmap."""
        return {"path": self.path}

    def __setstate__(self, state: dict[str, Path]) -> None:
        self.__init__(state["path"])

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        mapping = getattr(self, "_mmap", None)
        if mapping is not None:
            mapping.close()
            self._mmap = None
        source = getattr(self, "_file", None)
        if source is not None:
            source.close()
            self._file = None

    def routes_for_row(self, row_index: int) -> tuple[TarRoute, ...]:
        """Return routes for one source JSONL row, preserving audio-path order."""
        if row_index < 0:
            row_index += len(self)
        if row_index < 0 or row_index >= len(self):
            raise IndexError(f"Manifest row index {row_index} is out of bounds for {len(self)} rows")
        begin = self._row_offsets[row_index]
        end = self._row_offsets[row_index + 1]
        routes = []
        for record_index in range(begin, end):
            offset = self.header.routes_offset + record_index * SGROUTE_RECORD_SIZE
            tar_shard_index, tar_member_local_index = _ROUTE.unpack_from(self._mmap, offset)
            routes.append(TarRoute(tar_shard_index, tar_member_local_index))
        return tuple(routes)


def canonical_audio_prefix_map_digest(prefix_map: Mapping[str, str] | None) -> bytes:
    """Return SHA-256 of the canonical sorted compact JSON prefix-map object."""
    normalized = {} if prefix_map is None else dict(prefix_map)
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in normalized.items()):
        raise TypeError("Audio prefix map keys and values must be strings")
    canonical = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).digest()


def ordered_manifest_spec_path_digest(
    manifest_paths: Sequence[str | Path], manifest_specs: Sequence[str] | None = None
) -> bytes:
    """Digest ordered configured specs together with their expanded manifest paths."""
    paths = [os.fspath(path) for path in manifest_paths]
    specs = paths if manifest_specs is None else list(manifest_specs)
    if len(specs) != len(paths):
        raise ValueError(f"Expected one manifest spec per path, got {len(specs)} specs for {len(paths)} paths")
    records = (_canonical_json_bytes({"path": path, "spec": spec}) for path, spec in zip(paths, specs, strict=True))
    return _length_prefixed_digest(records)


def ordered_manifest_content_digest(manifest_paths: Sequence[str | Path]) -> bytes:
    """Digest each complete manifest as one ordered, length-prefixed record."""
    digest = hashlib.sha256()
    for path_like in manifest_paths:
        path = os.fspath(path_like)
        before = indexed_source_metadata(path)
        size = before["size_bytes"]
        digest.update(_U64.pack(size))
        consumed = 0
        while consumed < size:
            end = min(size, consumed + _REMOTE_MANIFEST_BATCH_BYTES)
            digest.update(read_exact_range(path, consumed, end))
            consumed = end
        if consumed != size:
            raise RuntimeError(f"Manifest changed while digesting: {path}")
        if indexed_source_metadata(path) != before:
            raise RuntimeError(f"Manifest changed while digesting: {path}")
    return digest.digest()


def ordered_manifest_source_identity_digest(
    manifest_paths: Sequence[str | Path],
) -> bytes:
    """Digest ordered O(1)-read identities for runtime validation."""
    return _length_prefixed_digest(
        _canonical_json_bytes(_source_identity_record(os.fspath(path))) for path in manifest_paths
    )


def ordered_manifest_content_digest_from_mirrors(
    manifest_paths: Sequence[str | Path], mirror_paths: Sequence[str | Path]
) -> bytes:
    """Digest local mirror bytes for ordered manifests declared at other paths."""
    if len(manifest_paths) != len(mirror_paths):
        raise ValueError(
            "Expected one content mirror per manifest path, got "
            f"{len(mirror_paths)} mirrors for {len(manifest_paths)} paths"
        )
    digest = hashlib.sha256()
    for index, mirror_like in enumerate(mirror_paths):
        mirror = Path(mirror_like)
        size = mirror.stat().st_size
        digest.update(_U64.pack(size))
        consumed = 0
        with mirror.open("rb") as source:
            while chunk := source.read(_REMOTE_MANIFEST_BATCH_BYTES):
                digest.update(chunk)
                consumed += len(chunk)
        if consumed != size or mirror.stat().st_size != size:
            raise RuntimeError(f"Manifest content mirror changed while digesting at position {index}: {mirror}")
    return digest.digest()


def ordered_manifest_source_identity_digest_from_metadata(
    manifest_paths: Sequence[str | Path], metadata_records: Sequence[Mapping]
) -> bytes:
    """Digest sealed local-manifest identities without mounting their filesystem."""
    if len(manifest_paths) != len(metadata_records):
        raise ValueError(
            "Expected one metadata record per manifest path, got "
            f"{len(metadata_records)} records for {len(manifest_paths)} paths"
        )
    expected_keys = {
        "path",
        "size_bytes",
        "mtime_ns",
        "object_identity",
        "device",
        "inode",
    }
    records = []
    for index, (path_like, record) in enumerate(zip(manifest_paths, metadata_records, strict=True)):
        path = os.fspath(path_like)
        if _is_remote_path(path):
            raise ValueError(f"Offline local-manifest metadata requires a local path at position {index}: {path!r}")
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            actual = sorted(record) if isinstance(record, Mapping) else type(record).__name__
            raise ValueError(
                f"Invalid manifest metadata keys at position {index}: expected "
                f"{sorted(expected_keys)}, got {actual}"
            )
        if record["path"] != path:
            raise ValueError(
                f"Manifest metadata path mismatch at position {index}: " f"{record['path']!r} != {path!r}"
            )
        for field in ("size_bytes", "mtime_ns", "device", "inode"):
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid manifest metadata {field} at position {index}: {value!r}")
        if record["object_identity"] is not None:
            raise ValueError(f"Local manifest object_identity must be null at position {index}")
        records.append(dict(record))
    return _length_prefixed_digest(_canonical_json_bytes(record) for record in records)


def ordered_tar_catalog_digest(tar_paths: Sequence[str | Path]) -> bytes:
    """Digest ordered tar paths and mutation-sensitive source identities."""
    return _length_prefixed_digest(
        _canonical_json_bytes(_source_identity_record(os.fspath(path))) for path in tar_paths
    )


def ordered_tar_catalog_digest_from_metadata(
    tar_paths: Sequence[str | Path], metadata_records: Sequence[Mapping]
) -> bytes:
    """Digest sealed remote identities without contacting their object backend."""
    if len(tar_paths) != len(metadata_records):
        raise ValueError(
            "Expected one metadata record per tar path, got "
            f"{len(metadata_records)} records for {len(tar_paths)} paths"
        )
    expected_keys = {"path", "size_bytes", "mtime_ns", "object_identity"}
    records = []
    for index, (path_like, record) in enumerate(zip(tar_paths, metadata_records, strict=True)):
        path = os.fspath(path_like)
        if not _is_remote_path(path):
            raise ValueError(f"Offline tar metadata requires a remote path at position {index}: {path!r}")
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            actual = sorted(record) if isinstance(record, Mapping) else type(record).__name__
            raise ValueError(
                f"Invalid tar metadata keys at position {index}: expected " f"{sorted(expected_keys)}, got {actual}"
            )
        if record["path"] != path:
            raise ValueError(f"Tar metadata path mismatch at position {index}: " f"{record['path']!r} != {path!r}")
        size = record["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"Invalid tar metadata size at position {index}: {size!r}")
        if record["mtime_ns"] != 0:
            raise ValueError(f"Remote tar metadata mtime_ns must be zero at position {index}")
        identity = record["object_identity"]
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"Remote tar metadata has no stable object identity at position {index}")
        records.append(dict(record))
    return _length_prefixed_digest(_canonical_json_bytes(record) for record in records)


def extract_sharegpt_audio_paths(
    document: Mapping,
    *,
    audio_placeholders: Sequence[str] = _DEFAULT_AUDIO_PLACEHOLDERS,
) -> tuple[str, ...]:
    """Extract one ShareGPT row's audio paths in parser order.

    Aliases use the frozen precedence ``sound`` → ``speech`` → ``ori_sound``.
    The returned order is source JSON list order and is checked against the
    total number of configured placeholders in the conversation turns.
    """
    selected_field = next(
        (field for field in _AUDIO_FIELDS if document.get(field) not in (None, "", [])),
        None,
    )
    if selected_field is None:
        raise ValueError(f"ShareGPT row has no supported audio field; expected one of {_AUDIO_FIELDS}")
    value = document[selected_field]
    if isinstance(value, str):
        paths = (value,)
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        paths = tuple(value)
    else:
        raise ValueError(f"ShareGPT {selected_field!r} must be a string or flat list of strings")
    if not paths or any(not path for path in paths):
        raise ValueError(f"ShareGPT {selected_field!r} must contain one or more non-empty paths")

    conversations = document.get("conversations")
    if not isinstance(conversations, list):
        raise TypeError("ShareGPT conversations must be a list")
    placeholder_count = 0
    for turn in conversations:
        if not isinstance(turn, Mapping):
            raise TypeError("ShareGPT conversation turns must be mappings")
        speaker = turn.get("from")
        if not isinstance(speaker, str):
            raise TypeError("ShareGPT conversation turn roles must be strings")
        if speaker.lower() not in ("human", "user"):
            continue
        text = turn.get("value", "")
        if not isinstance(text, str):
            raise TypeError("ShareGPT conversation turn values must be strings")
        placeholder_count += _count_audio_placeholders(text, audio_placeholders)
    if len(paths) > 1 and placeholder_count > 1 and placeholder_count != len(paths):
        raise ValueError(
            f"ShareGPT audio-path/placeholder cardinality mismatch: {len(paths)} paths but "
            f"{placeholder_count} placeholders"
        )
    return paths


def write_sharegpt_tar_routing_index(
    path: str | Path,
    row_routes: Iterable[Iterable[TarRoute | tuple[int, int]]],
    *,
    tar_shard_count: int,
    manifest_spec_path_digest: bytes | str,
    manifest_content_digest: bytes | str,
    manifest_source_identity_digest: bytes | str,
    tar_catalog_digest: bytes | str,
    audio_prefix_map_digest: bytes | str,
) -> Path:
    """Write, fsync, reopen-validate, and atomically publish a sealed route."""
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite sealed ShareGPT tar route: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if tar_shard_count < 0 or tar_shard_count > _UINT32_MAX + 1:
        raise ValueError(f"tar_shard_count is outside the v2 range: {tar_shard_count}")

    row_offsets = [0]
    records: list[TarRoute] = []
    for row in row_routes:
        for route_like in row:
            route = route_like if isinstance(route_like, TarRoute) else TarRoute(*route_like)
            _validate_route(route, tar_shard_count)
            records.append(route)
        row_offsets.append(len(records))

    payload = bytearray()
    for offset in row_offsets:
        payload.extend(_U64.pack(offset))
    for route in records:
        payload.extend(_ROUTE.pack(route.tar_shard_index, route.tar_member_local_index))

    row_count = len(row_offsets) - 1
    route_count = len(records)
    routes_offset = SGROUTE_HEADER_SIZE + 8 * (row_count + 1)
    file_size = routes_offset + SGROUTE_RECORD_SIZE * route_count
    payload_sha256 = hashlib.sha256(payload).digest()
    digests = {
        "manifest_spec_path_digest": _coerce_digest(manifest_spec_path_digest, "manifest spec/path digest"),
        "manifest_content_digest": _coerce_digest(manifest_content_digest, "manifest content digest"),
        "manifest_source_identity_digest": _coerce_digest(
            manifest_source_identity_digest, "manifest source-identity digest"
        ),
        "tar_catalog_digest": _coerce_digest(tar_catalog_digest, "tar catalog digest"),
        "audio_prefix_map_digest": _coerce_digest(audio_prefix_map_digest, "audio prefix-map digest"),
    }
    header = _HEADER.pack(
        SGROUTE_MAGIC,
        SGROUTE_VERSION,
        SGROUTE_HEADER_SIZE,
        SGROUTE_FLAGS,
        0,
        row_count,
        route_count,
        tar_shard_count,
        SGROUTE_HEADER_SIZE,
        routes_offset,
        file_size,
        SGROUTE_RECORD_SIZE,
        0,
        digests["manifest_spec_path_digest"],
        digests["manifest_content_digest"],
        digests["manifest_source_identity_digest"],
        digests["tar_catalog_digest"],
        digests["audio_prefix_map_digest"],
        payload_sha256,
        bytes(16),
    )
    assert len(header) == SGROUTE_HEADER_SIZE

    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(header)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        validate_sharegpt_tar_routing_index(
            temporary,
            expected_manifest_row_count=row_count,
            expected_manifest_spec_path_digest=digests["manifest_spec_path_digest"],
            expected_manifest_content_digest=digests["manifest_content_digest"],
            expected_manifest_source_identity_digest=digests["manifest_source_identity_digest"],
            expected_tar_shard_count=tar_shard_count,
            expected_tar_catalog_digest=digests["tar_catalog_digest"],
            expected_audio_prefix_map_digest=digests["audio_prefix_map_digest"],
            offset_bearing_tar_collections=True,
        )
        _atomic_publish_no_replace(temporary, output)
        _fsync_directory(output.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def validate_sharegpt_tar_routing_index(
    path: str | Path,
    *,
    offset_bearing_tar_collections: bool,
    expected_manifest_row_count: int | None = None,
    expected_manifest_spec_path_digest: bytes | str | None = None,
    expected_manifest_content_digest: bytes | str | None = None,
    expected_manifest_source_identity_digest: bytes | str | None = None,
    expected_tar_shard_count: int | None = None,
    expected_tar_catalog_digest: bytes | str | None = None,
    expected_audio_prefix_map_digest: bytes | str | None = None,
) -> ShareGptTarRouteHeader:
    """Validate self-contained and caller-supplied open-time route invariants."""
    if not offset_bearing_tar_collections:
        raise ValueError("ShareGPT collection routing requires offset-bearing tar collections")
    with ShareGptTarRoutingIndex(path) as routing:
        header = routing.header
        _expect_equal("manifest row count", header.manifest_row_count, expected_manifest_row_count)
        _expect_equal("tar shard count", header.tar_shard_count, expected_tar_shard_count)
        _expect_digest(
            "manifest spec/path digest",
            header.manifest_spec_path_digest,
            expected_manifest_spec_path_digest,
        )
        _expect_digest(
            "manifest content digest",
            header.manifest_content_digest,
            expected_manifest_content_digest,
        )
        _expect_digest(
            "manifest source-identity digest",
            header.manifest_source_identity_digest,
            expected_manifest_source_identity_digest,
        )
        _expect_digest("tar catalog digest", header.tar_catalog_digest, expected_tar_catalog_digest)
        _expect_digest(
            "audio prefix-map digest",
            header.audio_prefix_map_digest,
            expected_audio_prefix_map_digest,
        )
        return header


def _validate_native_tar_ranges(
    *,
    tar_path: str,
    offsets: Sequence[int],
    members: Sequence[tuple[str, int, int]],
) -> tuple[NativeTarMemberRange, ...]:
    """Validate aligned, ordered native-tar ranges in linear time."""
    result = []
    for local_index, ((name, expected_start, data_end), start, end) in enumerate(
        zip(members, offsets[:-1], offsets[1:], strict=True)
    ):
        next_regular_start = members[local_index + 1][1] if local_index + 1 < len(members) else None
        if start != expected_start or (next_regular_start is not None and next_regular_start < end) or data_end > end:
            raise ValueError(
                f"Native tar index range [{start}, {end}) at local member {local_index} in {tar_path} "
                "does not represent exactly one regular member"
            )
        result.append(NativeTarMemberRange(local_index, name, start, end, data_end))
    return tuple(result)


def validate_native_tar_member_index(tar_path: str | Path, index_path: str | Path) -> tuple[NativeTarMemberRange, ...]:
    """Prove that every native tar-index range contains one regular member."""
    tar_path = os.fspath(tar_path)
    index_path = Path(index_path)
    source_metadata = indexed_source_metadata(tar_path)
    source_size = source_metadata["size_bytes"]
    index_bytes = index_path.read_bytes()
    if len(index_bytes) < 8 or len(index_bytes) % 8:
        raise ValueError(f"Native tar index must contain raw little-endian uint64 offsets: {index_path}")
    offsets = list(struct.unpack(f"<{len(index_bytes) // 8}Q", index_bytes))

    if _is_remote_path(tar_path):
        members = [_read_remote_native_tar_member(tar_path, start, end) for start, end in pairwise(offsets)]
    else:
        members = []
        with tarfile.open(tar_path, "r:") as archive:
            for member in archive:
                if member.isreg():
                    data_end = member.offset_data + _round_up_tar_block(member.size)
                    members.append((member.name, member.offset, data_end))
    if len(offsets) != len(members) + 1:
        raise ValueError(
            f"Native tar index must provide exactly one regular member range: {index_path} has "
            f"{max(0, len(offsets) - 1)} ranges for {len(members)} regular members"
        )
    if any(left >= right for left, right in pairwise(offsets)):
        raise ValueError("Native tar index must provide exactly one regular member per strictly increasing range")
    if offsets[-1] > source_size:
        raise ValueError(f"Native tar index sentinel {offsets[-1]} exceeds tar size {source_size}: {index_path}")

    result = _validate_native_tar_ranges(
        tar_path=tar_path,
        offsets=offsets,
        members=members,
    )
    if indexed_source_metadata(tar_path) != source_metadata:
        raise RuntimeError(f"Native tar source changed while validating its index: {tar_path}")
    return result


def build_sharegpt_tar_routing_index(
    output_path: str | Path,
    *,
    manifest_paths: Sequence[str | Path],
    tar_paths: Sequence[str | Path],
    manifest_specs: Sequence[str] | None = None,
    manifest_index_paths: Sequence[str | Path] | None = None,
    tar_index_paths: Sequence[str | Path] | None = None,
    audio_prefix_map: Mapping[str, str] | None = None,
    audio_placeholders: Sequence[str] = _DEFAULT_AUDIO_PLACEHOLDERS,
) -> Path:
    """Build one deterministic route from ordered indexed manifests/tars."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite sealed ShareGPT tar route: {output}")
    if not manifest_paths:
        raise ValueError("At least one manifest path is required")
    if not tar_paths:
        raise ValueError("At least one tar path is required")
    manifest_paths = list(manifest_paths)
    tar_paths = list(tar_paths)
    manifest_indexes = _resolve_companion_paths(manifest_paths, manifest_index_paths, ".idx", "manifest")
    tar_indexes = _resolve_companion_paths(tar_paths, tar_index_paths, ".idx", "tar")

    manifest_stats = [_stable_source_identity(path) for path in manifest_paths]
    manifest_spec_digest = ordered_manifest_spec_path_digest(manifest_paths, manifest_specs)
    manifest_content_digest = ordered_manifest_content_digest(manifest_paths)
    manifest_source_identity_digest = ordered_manifest_source_identity_digest(manifest_paths)
    tar_digest_before = ordered_tar_catalog_digest(tar_paths)
    exact_names, basenames = _scan_ordered_tar_collection(tar_paths, tar_indexes)

    row_routes = []
    for manifest_path, manifest_index in zip(manifest_paths, manifest_indexes, strict=True):
        for document in _iter_indexed_jsonl(manifest_path, manifest_index):
            paths = extract_sharegpt_audio_paths(document, audio_placeholders=audio_placeholders)
            row_routes.append([_resolve_audio_reference(path, exact_names, basenames) for path in paths])
    if manifest_stats != [_stable_source_identity(path) for path in manifest_paths]:
        raise RuntimeError("A source manifest changed while building the ShareGPT tar route")
    if tar_digest_before != ordered_tar_catalog_digest(tar_paths):
        raise RuntimeError("The ordered tar catalog changed while building the ShareGPT tar route")

    return write_sharegpt_tar_routing_index(
        output,
        row_routes,
        tar_shard_count=len(tar_paths),
        manifest_spec_path_digest=manifest_spec_digest,
        manifest_content_digest=manifest_content_digest,
        manifest_source_identity_digest=manifest_source_identity_digest,
        tar_catalog_digest=tar_digest_before,
        audio_prefix_map_digest=canonical_audio_prefix_map_digest(audio_prefix_map),
    )


def _validate_layout(path: Path, data: mmap.mmap) -> tuple[ShareGptTarRouteHeader, tuple[int, ...]]:
    actual_size = len(data)
    if actual_size < SGROUTE_HEADER_SIZE:
        raise ValueError(
            f"ShareGPT tar route file size {actual_size} is smaller than the "
            f"{SGROUTE_HEADER_SIZE}-byte header: {path}"
        )
    fields = _HEADER.unpack_from(data)
    (
        magic,
        version,
        header_size,
        flags,
        reserved0,
        row_count,
        route_count,
        tar_count,
        row_offsets_offset,
        routes_offset,
        file_size,
        route_record_size,
        reserved1,
        manifest_spec_digest,
        manifest_content_digest,
        manifest_source_identity_digest,
        tar_catalog_digest,
        prefix_map_digest,
        payload_digest,
        reserved_tail,
    ) = fields
    if magic != SGROUTE_MAGIC:
        raise ValueError(f"Invalid ShareGPT tar route magic {magic!r}: {path}")
    if version != SGROUTE_VERSION:
        raise ValueError(f"Unsupported ShareGPT tar route version {version}: {path}")
    if header_size != SGROUTE_HEADER_SIZE:
        raise ValueError(f"Invalid ShareGPT tar route header size {header_size}: {path}")
    if flags != SGROUTE_FLAGS:
        raise ValueError(
            f"Invalid ShareGPT tar route flags {flags:#x}; v2 requires exactly " f"{SGROUTE_FLAGS:#x}: {path}"
        )
    if reserved0 != 0 or reserved1 != 0 or reserved_tail != bytes(16):
        raise ValueError(f"ShareGPT tar route reserved header bytes must be zero: {path}")
    if route_record_size != SGROUTE_RECORD_SIZE:
        raise ValueError(f"Invalid ShareGPT tar route route-record size {route_record_size}: {path}")
    expected_routes_offset = SGROUTE_HEADER_SIZE + 8 * (row_count + 1)
    expected_file_size = expected_routes_offset + SGROUTE_RECORD_SIZE * route_count
    if row_offsets_offset != SGROUTE_HEADER_SIZE:
        raise ValueError(f"Invalid ShareGPT tar route row-offset array file offset {row_offsets_offset}: {path}")
    if routes_offset != expected_routes_offset:
        raise ValueError(
            f"Invalid ShareGPT tar route record-array file offset {routes_offset}; expected {expected_routes_offset}: {path}"
        )
    if file_size != expected_file_size or actual_size != file_size:
        raise ValueError(
            f"Invalid ShareGPT tar route file size: header={file_size}, expected={expected_file_size}, "
            f"actual={actual_size}: {path}"
        )
    if hashlib.sha256(data[SGROUTE_HEADER_SIZE:file_size]).digest() != payload_digest:
        raise ValueError(f"ShareGPT tar route payload SHA-256 mismatch: {path}")

    row_offsets = tuple(
        _U64.unpack_from(data, row_offsets_offset + index * _U64.size)[0] for index in range(row_count + 1)
    )
    if not row_offsets or row_offsets[0] != 0:
        raise ValueError(f"ShareGPT tar route row_offsets[0] must be zero: {path}")
    if row_offsets[-1] != route_count:
        raise ValueError(
            f"ShareGPT tar route terminal row offset {row_offsets[-1]} does not equal route count {route_count}: {path}"
        )
    if any(left > right for left, right in pairwise(row_offsets)):
        raise ValueError(f"ShareGPT tar route row offsets must be monotonic: {path}")
    for record_index in range(route_count):
        shard_index, _ = _ROUTE.unpack_from(data, routes_offset + record_index * SGROUTE_RECORD_SIZE)
        if shard_index >= tar_count:
            raise ValueError(
                f"ShareGPT tar route record {record_index} addresses shard {shard_index}, "
                f"but the catalog has {tar_count} shards: {path}"
            )
    header = ShareGptTarRouteHeader(
        row_count,
        route_count,
        tar_count,
        row_offsets_offset,
        routes_offset,
        file_size,
        manifest_spec_digest,
        manifest_content_digest,
        manifest_source_identity_digest,
        tar_catalog_digest,
        prefix_map_digest,
        payload_digest,
    )
    return header, row_offsets


def _scan_ordered_tar_collection(
    tar_paths: Sequence[str | Path], tar_index_paths: Sequence[str | Path]
) -> tuple[dict[str, TarRoute], dict[str, list[TarRoute]]]:
    exact_names: dict[str, TarRoute] = {}
    basenames: dict[str, list[TarRoute]] = {}
    for shard_index, (tar_path, index_path) in enumerate(zip(tar_paths, tar_index_paths, strict=True)):
        members = validate_native_tar_member_index(tar_path, index_path)
        for member in members:
            route = TarRoute(shard_index, member.local_index)
            if member.name in exact_names:
                raise ValueError(f"duplicate tar member name {member.name!r} in ordered tar collection")
            exact_names[member.name] = route
            basenames.setdefault(PurePosixPath(member.name).name, []).append(route)
    return exact_names, basenames


def _resolve_audio_reference(
    reference: str,
    exact_names: Mapping[str, TarRoute],
    basenames: Mapping[str, list[TarRoute]],
) -> TarRoute:
    exact = exact_names.get(reference)
    if exact is not None:
        return exact
    basename = PurePosixPath(reference).name
    candidates = basenames.get(basename, ())
    if not candidates:
        raise ValueError(f"missing audio member {reference!r} in ordered tar collection")
    if len(candidates) != 1:
        raise ValueError(f"ambiguous basename {basename!r} for audio reference {reference!r} in tar collection")
    return candidates[0]


def _iter_indexed_jsonl(path_like: str | Path, index_path_like: str | Path):
    path = os.fspath(path_like)
    offsets = _read_offsets(Path(index_path_like))
    source_metadata = indexed_source_metadata(path)
    source_size = source_metadata["size_bytes"]
    if not offsets or offsets[0] != 0 or offsets[-1] != source_size:
        raise ValueError(f"Indexed ShareGPT manifest must have zero origin and physical-size sentinel: {path}")
    if any(left >= right for left, right in pairwise(offsets)):
        raise ValueError(f"Indexed ShareGPT manifest offsets must define non-empty ordered rows: {path}")
    row_index = 0
    while row_index < len(offsets) - 1:
        batch_start = offsets[row_index]
        batch_end_index = row_index + 1
        while (
            batch_end_index < len(offsets) - 1
            and offsets[batch_end_index + 1] - batch_start <= _REMOTE_MANIFEST_BATCH_BYTES
        ):
            batch_end_index += 1
        batch_end = offsets[batch_end_index]
        if batch_end - batch_start > _REMOTE_MANIFEST_BATCH_BYTES:
            raise ValueError(
                f"Indexed ShareGPT row {row_index} in {path} exceeds the "
                f"{_REMOTE_MANIFEST_BATCH_BYTES}-byte bounded-read limit"
            )
        raw = read_exact_range(path, batch_start, batch_end)
        while row_index < batch_end_index:
            start = offsets[row_index]
            end = offsets[row_index + 1]
            payload = raw[start - batch_start : end - batch_start]
            try:
                document = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Malformed JSON at indexed row {row_index} in {path}: {error}") from error
            if not isinstance(document, dict):
                raise TypeError(f"ShareGPT indexed row {row_index} in {path} must be a JSON object")
            yield document
            row_index += 1
    if indexed_source_metadata(path) != source_metadata:
        raise RuntimeError(f"ShareGPT manifest changed while reading indexed rows: {path}")


def _read_offsets(path: Path) -> tuple[int, ...]:
    raw = path.read_bytes()
    if len(raw) < 8 or len(raw) % 8:
        raise ValueError(f"Offset index must contain raw little-endian uint64 values: {path}")
    return struct.unpack(f"<{len(raw) // 8}Q", raw)


def _resolve_companion_paths(
    source_paths: Sequence[str | Path],
    explicit_paths: Sequence[str | Path] | None,
    suffix: str,
    label: str,
) -> list[Path]:
    if explicit_paths is None:
        return [Path(f"{path}{suffix}") for path in source_paths]
    if len(explicit_paths) != len(source_paths):
        raise ValueError(
            f"Expected one {label} index per source, got {len(explicit_paths)} indexes for {len(source_paths)} sources"
        )
    return [Path(path) for path in explicit_paths]


def _stable_source_identity(path_like: str | Path) -> bytes:
    return _canonical_json_bytes(_source_identity_record(os.fspath(path_like)))


def _source_identity_record(path: str) -> dict:
    """Return stable local stat or remote object identity without payload reads."""
    metadata = indexed_source_metadata(path)
    if _is_remote_path(path):
        return metadata
    stat = Path(path).stat()
    if metadata["size_bytes"] != stat.st_size or metadata["mtime_ns"] != stat.st_mtime_ns:
        raise RuntimeError(f"Source changed while recording identity: {path}")
    return {
        **metadata,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _is_remote_path(path: str) -> bool:
    return path.startswith(("ais://", "s3://"))


def _read_remote_native_tar_member(path: str, start: int, end: int) -> tuple[str, int, int]:
    if start < 0 or end <= start:
        raise ValueError(f"Invalid native tar member range [{start}, {end}) in {path}")
    position = start
    pax_headers: dict[str, str] = {}
    long_name: str | None = None
    while position + tarfile.BLOCKSIZE <= end:
        header = read_exact_range(path, position, position + tarfile.BLOCKSIZE)
        if header == bytes(tarfile.BLOCKSIZE):
            break
        try:
            info = tarfile.TarInfo.frombuf(header, tarfile.ENCODING, "surrogateescape")
        except tarfile.TarError as error:
            raise ValueError(
                f"Invalid remote tar header at byte {position} in indexed range [{start}, {end}) of {path}: {error}"
            ) from error
        data_start = position + tarfile.BLOCKSIZE
        data_end = data_start + _round_up_tar_block(info.size)
        if data_end > end:
            raise ValueError(f"Remote tar member {info.name!r} exceeds indexed range [{start}, {end}) in {path}")
        if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME):
            if info.size > _REMOTE_TAR_METADATA_MAX_BYTES:
                raise ValueError(
                    f"Remote tar metadata member {info.name!r} is {info.size} bytes; "
                    f"maximum is {_REMOTE_TAR_METADATA_MAX_BYTES}"
                )
            data = read_exact_range(path, data_start, data_start + info.size)
            if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
                pax_headers.update(_parse_pax_headers(data))
            else:
                long_name = data.rstrip(b"\0\n").decode(tarfile.ENCODING, "surrogateescape")
            position = data_end
            continue
        if info.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
            name = pax_headers.get("path") or long_name or info.name
            return name, start, data_end
        position = data_end
    raise ValueError(f"Native tar index range [{start}, {end}) in {path} contains no regular member")


def _validate_route(route: TarRoute, tar_shard_count: int) -> None:
    if not isinstance(route.tar_shard_index, int) or not isinstance(route.tar_member_local_index, int):
        raise TypeError(f"Tar route indices must be integers, got {route!r}")
    if route.tar_shard_index < 0 or route.tar_shard_index >= tar_shard_count:
        raise ValueError(f"Tar route shard index {route.tar_shard_index} is outside [0, {tar_shard_count})")
    if route.tar_member_local_index < 0 or route.tar_member_local_index > _UINT32_MAX:
        raise ValueError(f"Tar route local member index is outside uint32 range: {route.tar_member_local_index}")


def _coerce_digest(value: bytes | str, label: str) -> bytes:
    if isinstance(value, str):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{label} must be 64 lower-case hexadecimal characters")
        value = bytes.fromhex(value)
    if not isinstance(value, bytes) or len(value) != 32:
        raise ValueError(f"{label} must contain exactly 32 bytes")
    return value


def _expect_equal(label: str, actual, expected) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"ShareGPT tar route {label} mismatch: expected {expected}, got {actual}")


def _expect_digest(label: str, actual: bytes, expected: bytes | str | None) -> None:
    if expected is not None:
        expected_bytes = _coerce_digest(expected, label)
        if actual != expected_bytes:
            raise ValueError(
                f"ShareGPT tar route {label} mismatch: expected {expected_bytes.hex()}, got {actual.hex()}"
            )


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _length_prefixed_digest(records: Iterable[bytes]) -> bytes:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_U64.pack(len(record)))
        digest.update(record)
    return digest.digest()


def _count_audio_placeholders(text: str, placeholders: Sequence[str]) -> int:
    count = 0
    remaining = text
    while True:
        matches = [
            (index, placeholder)
            for placeholder in placeholders
            if placeholder and (index := remaining.find(placeholder)) >= 0
        ]
        if not matches:
            return count
        index, placeholder = min(matches, key=lambda match: match[0])
        count += 1
        remaining = remaining[index + len(placeholder) :]


def _round_up_tar_block(size: int) -> int:
    return (size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE * tarfile.BLOCKSIZE


def _atomic_publish_no_replace(source: Path, destination: Path) -> None:
    def publish_by_hard_link() -> None:
        try:
            os.link(source, destination)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite sealed ShareGPT tar route: {destination}") from error
        source.unlink()

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        publish_by_hard_link()
        return

    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        == 0
    ):
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"Refusing to overwrite sealed ShareGPT tar route: {destination}")
    if error_number in {
        errno.ENOSYS,
        errno.EINVAL,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
    }:
        publish_by_hard_link()
        return
    raise OSError(
        error_number,
        f"Unable to atomically publish ShareGPT tar route {destination}",
        destination,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
