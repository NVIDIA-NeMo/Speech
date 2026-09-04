# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time routing between native NeMo manifests and paired audio tars."""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lhotse.indexing import read_index
from lhotse.serialization import decode_json_line

from nemo.collections.common.data.lhotse.indexed_adapters import (
    IndexedTarMemberReader,
    _open_data_path,
    _resolve_data_path,
    indexed_source_metadata,
)

NEMO_TAR_ORDINAL_MAP_ROLE = "native_tar_route"
NEMO_TAR_ORDINAL_MAP_KIND = "nemo_manifest_row_to_tar_ordinal_v1"
NEMO_TAR_MEMBER_NAME_NORMALIZATION = "nemo-audio-member-v1"
NEMO_TAR_SKIP_ORDINAL = (1 << 32) - 1

_OFFSET_PATTERN = re.compile(r'^(?P<stem>.+)(?P<sub>-sub\d+)(?P<ext>\.\w+)?$')
_MANIFEST_BATCH_BYTES = 64 << 20
_U32 = struct.Struct("<I")


@dataclass(frozen=True)
class NativeTarOrdinalMapBuildSummary:
    shard_rows: tuple[int, ...]
    records_checked: int
    skip_marker_records: int
    top_level_skip_marker_records: int
    custom_skip_marker_records: int
    input_snapshot: NativeTarOrdinalMapInputSnapshot | None = None


@dataclass(frozen=True)
class NativeTarOrdinalMapInputSnapshot:
    source_paths: tuple[tuple[str, str], ...]
    source_identities: tuple[tuple[object, object], ...]
    index_paths: tuple[tuple[str | Path, str | Path], ...]
    index_identities: tuple[tuple[object, object], ...]

    @classmethod
    def capture(
        cls,
        manifest_paths: tuple[str, ...],
        tar_paths: tuple[str, ...],
        manifest_index_paths: tuple[str | Path, ...],
        tar_index_paths: tuple[str | Path, ...],
    ) -> NativeTarOrdinalMapInputSnapshot:
        source_paths = tuple(zip(manifest_paths, tar_paths))
        index_paths = tuple(zip(manifest_index_paths, tar_index_paths))
        return cls(
            source_paths=source_paths,
            source_identities=tuple(
                (_source_identity(manifest_path), _source_identity(tar_path))
                for manifest_path, tar_path in source_paths
            ),
            index_paths=index_paths,
            index_identities=tuple(
                (_file_identity(manifest_index_path), _file_identity(tar_index_path))
                for manifest_index_path, tar_index_path in index_paths
            ),
        )

    def validate(self) -> None:
        current_sources = tuple(
            (_source_identity(manifest_path), _source_identity(tar_path))
            for manifest_path, tar_path in self.source_paths
        )
        if self.source_identities != current_sources:
            raise ValueError("A native NeMo source changed after native-tar route construction")
        current_indexes = tuple(
            (_file_identity(manifest_index_path), _file_identity(tar_index_path))
            for manifest_index_path, tar_index_path in self.index_paths
        )
        if self.index_identities != current_indexes:
            raise ValueError("A native NeMo index changed after native-tar route construction")


def manifest_entry_is_explicitly_skipped(data: Mapping) -> bool:
    """Return whether a manifest row carries a truthy canonical skip marker."""
    if bool(data.get("_skipme", False)):
        return True
    custom = data.get("custom")
    return isinstance(custom, Mapping) and bool(custom.get("_skipme", False))


def nemo_tar_audio_member_name(audio_filepath: str) -> str:
    """Normalize a NeMo tarred-manifest audio path to its actual tar member."""
    if not isinstance(audio_filepath, str) or not audio_filepath:
        raise ValueError(f"audio_filepath must be a non-empty string, got {audio_filepath!r}")
    match = _OFFSET_PATTERN.match(audio_filepath)
    if match is None:
        return audio_filepath
    return match.group("stem") + (match.group("ext") or "")


def nemo_tar_ordinal_map_source_spec(manifest_source_spec, tar_source_spec) -> dict:
    """Return the stable identity used for an embedded native-tar ordinal map."""
    return {
        "manifest": manifest_source_spec,
        "tar": tar_source_spec,
        "normalization": NEMO_TAR_MEMBER_NAME_NORMALIZATION,
    }


def nemo_tar_ordinal_map_collection_key(manifest_source_spec, tar_source_spec) -> bytes:
    """Return the idxpack collection key for a native-tar ordinal map."""
    from lhotse.index_pack import index_pack_collection_key

    return index_pack_collection_key(
        NEMO_TAR_ORDINAL_MAP_ROLE,
        NEMO_TAR_ORDINAL_MAP_KIND,
        nemo_tar_ordinal_map_source_spec(manifest_source_spec, tar_source_spec),
    )


def write_nemo_tar_ordinal_map_shard(
    output_path: str | Path,
    *,
    manifest_path: str,
    manifest_index_path: str | Path,
    tar_path: str,
    tar_index_path: str | Path,
    tar_sentinel_size_override: int | None = None,
) -> NativeTarOrdinalMapBuildSummary:
    """Write one raw uint32 manifest-row to tar-member permutation shard.

    This is a build-time intermediate consumed by ``IndexPackArraySpec``. It
    reads indexed manifest records and tar headers, but never audio payloads.
    """
    source_identity_before = (_source_identity(manifest_path), _source_identity(tar_path))
    index_identity_before = (_file_identity(manifest_index_path), _file_identity(tar_index_path))
    tar_reader = IndexedTarMemberReader(
        tar_path,
        idx_path=tar_index_path,
        auto_create_index=False,
        sentinel_size_override=tar_sentinel_size_override,
    )
    try:
        member_ordinals = tar_reader.member_name_index(reject_duplicates=True)
        row_count = 0
        skip_marker_records = 0
        top_level_skip_marker_records = 0
        custom_skip_marker_records = 0
        with Path(output_path).open("xb") as output:
            buffer = bytearray()
            for row_index, data in enumerate(_iter_indexed_manifest_rows(manifest_path, manifest_index_path)):
                top_level_marker = bool(data.get("_skipme", False))
                custom = data.get("custom")
                custom_marker = isinstance(custom, Mapping) and bool(custom.get("_skipme", False))
                if top_level_marker or custom_marker:
                    ordinal = NEMO_TAR_SKIP_ORDINAL
                    skip_marker_records += 1
                    top_level_skip_marker_records += int(top_level_marker)
                    custom_skip_marker_records += int(custom_marker)
                else:
                    try:
                        expected_name = nemo_tar_audio_member_name(data["audio_filepath"])
                    except KeyError as ex:
                        raise ValueError(
                            f"Native NeMo manifest row {row_index} in {manifest_path!r} " "is missing audio_filepath"
                        ) from ex
                    try:
                        ordinal = member_ordinals[expected_name]
                    except KeyError as ex:
                        raise ValueError(
                            f"Native NeMo manifest row {row_index} in {manifest_path!r} references "
                            f"missing tar member {expected_name!r} in {tar_path!r}"
                        ) from ex
                    if ordinal >= NEMO_TAR_SKIP_ORDINAL:
                        raise ValueError(
                            f"Native NeMo tar member ordinal {ordinal} in {tar_path!r} "
                            "cannot be represented by the uint32 routing format"
                        )
                buffer.extend(_U32.pack(ordinal))
                row_count += 1
                if len(buffer) >= 1024 * 1024:
                    output.write(buffer)
                    buffer.clear()
            if buffer:
                output.write(buffer)
        if source_identity_before != (_source_identity(manifest_path), _source_identity(tar_path)):
            raise ValueError(
                f"A native NeMo source changed while building the ordinal map for "
                f"manifest={manifest_path!r} tar={tar_path!r}"
            )
        if index_identity_before != (_file_identity(manifest_index_path), _file_identity(tar_index_path)):
            raise ValueError(
                f"A native NeMo index changed while building the ordinal map for "
                f"manifest={manifest_path!r} tar={tar_path!r}"
            )
        return NativeTarOrdinalMapBuildSummary(
            shard_rows=(row_count,),
            records_checked=row_count,
            skip_marker_records=skip_marker_records,
            top_level_skip_marker_records=top_level_skip_marker_records,
            custom_skip_marker_records=custom_skip_marker_records,
        )
    finally:
        tar_reader.close()


def write_nemo_tar_ordinal_map_shards(
    output_paths: tuple[str | Path, ...],
    *,
    manifest_paths: tuple[str, ...],
    manifest_index_paths: tuple[str | Path, ...],
    tar_paths: tuple[str, ...],
    tar_index_paths: tuple[str | Path, ...],
    tar_sentinel_size_overrides: tuple[int | None, ...],
) -> NativeTarOrdinalMapBuildSummary:
    """Write every shard of one routing collection from a stable source snapshot."""
    lengths = {
        "outputs": len(output_paths),
        "manifests": len(manifest_paths),
        "manifest indexes": len(manifest_index_paths),
        "tars": len(tar_paths),
        "tar indexes": len(tar_index_paths),
        "tar sentinel overrides": len(tar_sentinel_size_overrides),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Native NeMo ordinal-map shard counts differ: {lengths}")

    input_snapshot = NativeTarOrdinalMapInputSnapshot.capture(
        manifest_paths,
        tar_paths,
        manifest_index_paths,
        tar_index_paths,
    )
    shard_summaries = tuple(
        write_nemo_tar_ordinal_map_shard(
            output_path,
            manifest_path=manifest_path,
            manifest_index_path=manifest_index_path,
            tar_path=tar_path,
            tar_index_path=tar_index_path,
            tar_sentinel_size_override=sentinel_override,
        )
        for output_path, manifest_path, manifest_index_path, tar_path, tar_index_path, sentinel_override in zip(
            output_paths,
            manifest_paths,
            manifest_index_paths,
            tar_paths,
            tar_index_paths,
            tar_sentinel_size_overrides,
        )
    )
    input_snapshot.validate()
    return NativeTarOrdinalMapBuildSummary(
        shard_rows=tuple(summary.records_checked for summary in shard_summaries),
        records_checked=sum(summary.records_checked for summary in shard_summaries),
        skip_marker_records=sum(summary.skip_marker_records for summary in shard_summaries),
        top_level_skip_marker_records=sum(summary.top_level_skip_marker_records for summary in shard_summaries),
        custom_skip_marker_records=sum(summary.custom_skip_marker_records for summary in shard_summaries),
        input_snapshot=input_snapshot,
    )


def _iter_indexed_manifest_rows(path: str, index_path: str | Path):
    offsets = read_index(index_path)
    if len(offsets) < 1 or int(offsets[0]) != 0:
        raise ValueError(f"Native NeMo manifest index must begin at byte zero: {index_path}")
    if len(offsets) > 1 and (offsets[1:] <= offsets[:-1]).any():
        raise ValueError(f"Native NeMo manifest index must contain strictly increasing offsets: {index_path}")

    row_index = 0
    row_count = len(offsets) - 1
    with _open_data_path(path) as source:
        while row_index < row_count:
            batch_start = int(offsets[row_index])
            batch_end_index = row_index + 1
            while (
                batch_end_index < row_count
                and int(offsets[batch_end_index + 1]) - batch_start <= _MANIFEST_BATCH_BYTES
            ):
                batch_end_index += 1
            batch_end = int(offsets[batch_end_index])
            source.seek(batch_start)
            raw = source.read(batch_end - batch_start)
            if len(raw) != batch_end - batch_start:
                raise EOFError(
                    f"Short indexed manifest read from {path!r}: requested "
                    f"[{batch_start}, {batch_end}), received {len(raw)} bytes"
                )
            while row_index < batch_end_index:
                start = int(offsets[row_index]) - batch_start
                end = int(offsets[row_index + 1]) - batch_start
                try:
                    encoded = raw[start:end].decode("utf-8")
                    data = decode_json_line(encoded)
                except (UnicodeDecodeError, json.JSONDecodeError) as ex:
                    raise ValueError(f"Malformed JSON at indexed row {row_index} in {path!r}: {ex}") from ex
                if not isinstance(data, Mapping):
                    raise ValueError(
                        f"Native NeMo indexed row {row_index} in {path!r} must be a JSON object, "
                        f"got {type(data).__name__}"
                    )
                yield data
                row_index += 1


def _source_identity(path: str):
    resolved_path = _resolve_data_path(path)
    metadata = indexed_source_metadata(resolved_path)
    if metadata.get("object_identity") is None:
        stat = Path(resolved_path).stat()
        return {**metadata, "device": stat.st_dev, "inode": stat.st_ino}
    return metadata


def _file_identity(path: str | Path) -> tuple[int, int, int, int]:
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
