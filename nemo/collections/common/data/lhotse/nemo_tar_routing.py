# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time routing between native NeMo manifests and paired audio tars."""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
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
NEMO_TAR_SHARD_MAP_ROLE = "native_tar_shard_route"
NEMO_TAR_SHARD_MAP_KIND = "nemo_manifest_row_to_tar_shard_v1"
NEMO_TAR_MEMBER_NAME_NORMALIZATION = "nemo-audio-member-v1"
NEMO_TAR_SKIP_ORDINAL = (1 << 32) - 1

_OFFSET_PATTERN = re.compile(r"^(?P<stem>.+)(?P<sub>-sub\d+)(?P<ext>\.\w+)?$")
_TAR_SHARD_PATTERN = re.compile(r"audio[^/]*_(\d+)[^/]*\.tar$")
_MANIFEST_BATCH_BYTES = 64 << 20
_U32 = struct.Struct("<I")


@dataclass(frozen=True)
class NativeTarOrdinalMapBuildSummary:
    shard_rows: tuple[int, ...]
    records_checked: int
    skip_marker_records: int
    top_level_skip_marker_records: int
    custom_skip_marker_records: int
    input_snapshot: NativeTarOrdinalMapInputSnapshot | NativeTarShardMapInputSnapshot | None = None


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


@dataclass(frozen=True)
class NativeTarShardMapInputSnapshot:
    """Manifest state that a shard-only route depends on."""

    manifest_path: str
    manifest_identity: object
    manifest_index_path: str | Path
    manifest_index_identity: tuple[int, int, int, int]

    @classmethod
    def capture(cls, manifest_path: str, manifest_index_path: str | Path) -> NativeTarShardMapInputSnapshot:
        return cls(
            manifest_path=manifest_path,
            manifest_identity=_source_identity(manifest_path),
            manifest_index_path=manifest_index_path,
            manifest_index_identity=_file_identity(manifest_index_path),
        )

    def validate(self) -> None:
        if _source_identity(self.manifest_path) != self.manifest_identity:
            raise ValueError("A native NeMo manifest changed after shard-route construction")
        if _file_identity(self.manifest_index_path) != self.manifest_index_identity:
            raise ValueError("A native NeMo manifest index changed after shard-route construction")


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


def nemo_tar_shard_map_source_spec(manifest_source_spec, tar_source_spec) -> dict:
    """Return the stable identity for aggregate-manifest tar-shard routing."""
    return {
        "manifest": manifest_source_spec,
        "tar": tar_source_spec,
        "routing": "manifest-shard-id-v1",
    }


def nemo_tar_shard_map_collection_key(manifest_source_spec, tar_source_spec) -> bytes:
    """Return the idxpack key for an aggregate manifest's tar-shard map."""
    from lhotse.index_pack import index_pack_collection_key

    return index_pack_collection_key(
        NEMO_TAR_SHARD_MAP_ROLE,
        NEMO_TAR_SHARD_MAP_KIND,
        nemo_tar_shard_map_source_spec(manifest_source_spec, tar_source_spec),
    )


def _load_native_tar_member_indexes(
    tar_paths: tuple[str, ...],
    tar_index_paths: tuple[str | Path, ...],
    tar_sentinel_size_overrides: tuple[int | None, ...],
) -> list[dict[str, int]]:
    """Load each tar's unique member-name-to-ordinal lookup and close its reader."""
    member_indexes = []
    for tar_path, tar_index_path, sentinel_override in zip(tar_paths, tar_index_paths, tar_sentinel_size_overrides):
        reader = IndexedTarMemberReader(
            tar_path,
            idx_path=tar_index_path,
            auto_create_index=False,
            sentinel_size_override=sentinel_override,
        )
        try:
            member_indexes.append(reader.member_name_index(reject_duplicates=True))
        finally:
            reader.close()
    return member_indexes


def _native_tar_shard_positions(tar_paths: tuple[str, ...]) -> dict[int, int]:
    """Map numeric tar shard identities to positions in the expanded path list."""
    positions = {}
    for position, tar_path in enumerate(tar_paths):
        match = _TAR_SHARD_PATTERN.search(tar_path)
        if match is None:
            raise ValueError(
                "Cannot determine aggregate native NeMo shard_id from tar path "
                f"{tar_path!r}; expected a numbered audio_*.tar name"
            )
        shard_id = int(match.group(1))
        if shard_id in positions:
            raise ValueError(
                "Aggregate native NeMo tar paths contain duplicate shard_id "
                f"{shard_id}: {tar_paths[positions[shard_id]]!r} and {tar_path!r}"
            )
        positions[shard_id] = position
    return positions


def _resolve_aggregate_manifest_shard(
    data: Mapping,
    *,
    row_index: int,
    manifest_path: str,
    shard_positions: Mapping[int, int],
) -> tuple[int, bool, bool]:
    """Resolve a manifest shard identity to its tar-collection position."""
    top_level_marker = bool(data.get("_skipme", False))
    custom = data.get("custom")
    custom_marker = isinstance(custom, Mapping) and bool(custom.get("_skipme", False))
    if top_level_marker or custom_marker:
        return NEMO_TAR_SKIP_ORDINAL, top_level_marker, custom_marker

    shard_id = data.get("shard_id")
    if isinstance(shard_id, bool) or not isinstance(shard_id, int):
        raise ValueError(
            f"Aggregate native NeMo manifest row {row_index} in {manifest_path!r} "
            f"requires an integer shard_id, got {shard_id!r}"
        )
    try:
        return shard_positions[shard_id], False, False
    except KeyError as ex:
        raise ValueError(
            f"Aggregate native NeMo manifest row {row_index} in {manifest_path!r} "
            f"references shard_id={shard_id}, which has no matching tar path; "
            f"available shard IDs: {sorted(shard_positions)}"
        ) from ex


def _resolve_aggregate_manifest_route(
    data: Mapping,
    *,
    row_index: int,
    manifest_path: str,
    tar_paths: tuple[str, ...],
    shard_positions: Mapping[int, int],
    member_indexes: list[dict[str, int]],
) -> tuple[int, int, bool, bool]:
    """Resolve one aggregate-manifest row to tar shard and member ordinals.

    The booleans preserve which supported skip marker caused a sentinel route.
    """
    tar_shard, top_level_marker, custom_marker = _resolve_aggregate_manifest_shard(
        data,
        row_index=row_index,
        manifest_path=manifest_path,
        shard_positions=shard_positions,
    )
    if tar_shard == NEMO_TAR_SKIP_ORDINAL:
        return (
            NEMO_TAR_SKIP_ORDINAL,
            NEMO_TAR_SKIP_ORDINAL,
            top_level_marker,
            custom_marker,
        )

    try:
        expected_name = nemo_tar_audio_member_name(data["audio_filepath"])
    except KeyError as ex:
        raise ValueError(
            f"Aggregate native NeMo manifest row {row_index} in {manifest_path!r} " "is missing audio_filepath"
        ) from ex
    try:
        member_ordinal = member_indexes[tar_shard][expected_name]
    except KeyError as ex:
        raise ValueError(
            f"Aggregate native NeMo manifest row {row_index} in {manifest_path!r} "
            f"references missing tar member {expected_name!r} in {tar_paths[tar_shard]!r}"
        ) from ex
    if member_ordinal >= NEMO_TAR_SKIP_ORDINAL:
        raise ValueError(
            f"Native NeMo tar member ordinal {member_ordinal} in {tar_paths[tar_shard]!r} "
            "cannot be represented by the uint32 routing format"
        )
    return tar_shard, member_ordinal, False, False


def write_nemo_tar_aggregate_ordinal_maps(
    ordinal_output_path: str | Path,
    shard_output_path: str | Path,
    *,
    manifest_path: str,
    manifest_index_path: str | Path,
    tar_paths: tuple[str, ...],
    tar_index_paths: tuple[str | Path, ...],
    tar_sentinel_size_overrides: tuple[int | None, ...],
) -> NativeTarOrdinalMapBuildSummary:
    """Route one aggregate manifest to many tar shards via each row's ``shard_id``."""
    if not tar_paths:
        raise ValueError("Aggregate native NeMo routing requires at least one tar shard")
    if len({len(tar_paths), len(tar_index_paths), len(tar_sentinel_size_overrides)}) != 1:
        raise ValueError(
            "Aggregate native NeMo tar/index/sentinel counts differ: "
            f"tars={len(tar_paths)} indexes={len(tar_index_paths)} "
            f"sentinels={len(tar_sentinel_size_overrides)}"
        )
    snapshot = NativeTarOrdinalMapInputSnapshot.capture(
        (manifest_path,) * len(tar_paths),
        tar_paths,
        (manifest_index_path,) * len(tar_paths),
        tar_index_paths,
    )
    shard_positions = _native_tar_shard_positions(tar_paths)
    member_indexes = _load_native_tar_member_indexes(tar_paths, tar_index_paths, tar_sentinel_size_overrides)

    row_count = 0
    skip_marker_records = 0
    top_level_skip_marker_records = 0
    custom_skip_marker_records = 0
    with (
        Path(ordinal_output_path).open("xb") as ordinal_output,
        Path(shard_output_path).open("xb") as shard_output,
    ):
        ordinal_buffer = bytearray()
        shard_buffer = bytearray()
        for row_index, data in enumerate(_iter_indexed_manifest_rows(manifest_path, manifest_index_path)):
            tar_shard, member_ordinal, top_level_marker, custom_marker = _resolve_aggregate_manifest_route(
                data,
                row_index=row_index,
                manifest_path=manifest_path,
                tar_paths=tar_paths,
                shard_positions=shard_positions,
                member_indexes=member_indexes,
            )
            if top_level_marker or custom_marker:
                skip_marker_records += 1
                top_level_skip_marker_records += int(top_level_marker)
                custom_skip_marker_records += int(custom_marker)
            ordinal_buffer.extend(_U32.pack(member_ordinal))
            shard_buffer.extend(_U32.pack(tar_shard))
            row_count += 1
            if len(ordinal_buffer) >= 1024 * 1024:
                ordinal_output.write(ordinal_buffer)
                shard_output.write(shard_buffer)
                ordinal_buffer.clear()
                shard_buffer.clear()
        if ordinal_buffer:
            ordinal_output.write(ordinal_buffer)
            shard_output.write(shard_buffer)
    snapshot.validate()
    return NativeTarOrdinalMapBuildSummary(
        shard_rows=(row_count,),
        records_checked=row_count,
        skip_marker_records=skip_marker_records,
        top_level_skip_marker_records=top_level_skip_marker_records,
        custom_skip_marker_records=custom_skip_marker_records,
        input_snapshot=snapshot,
    )


def write_nemo_tar_aggregate_shard_map(
    output_path: str | Path,
    *,
    manifest_path: str,
    manifest_index_path: str | Path,
    tar_paths: tuple[str, ...],
) -> NativeTarOrdinalMapBuildSummary:
    """Write row-to-tar-position routing without reading tar member indexes."""
    if not tar_paths:
        raise ValueError("Aggregate native NeMo routing requires at least one tar shard")
    snapshot = NativeTarShardMapInputSnapshot.capture(manifest_path, manifest_index_path)
    shard_positions = _native_tar_shard_positions(tar_paths)
    row_count = 0
    skip_marker_records = 0
    top_level_skip_marker_records = 0
    custom_skip_marker_records = 0
    with Path(output_path).open("xb") as output:
        buffer = bytearray()
        for row_index, data in enumerate(_iter_indexed_manifest_rows(manifest_path, manifest_index_path)):
            tar_shard, top_level_marker, custom_marker = _resolve_aggregate_manifest_shard(
                data,
                row_index=row_index,
                manifest_path=manifest_path,
                shard_positions=shard_positions,
            )
            if top_level_marker or custom_marker:
                skip_marker_records += 1
                top_level_skip_marker_records += int(top_level_marker)
                custom_skip_marker_records += int(custom_marker)
            buffer.extend(_U32.pack(tar_shard))
            row_count += 1
            if len(buffer) >= 1024 * 1024:
                output.write(buffer)
                buffer.clear()
        if buffer:
            output.write(buffer)
    snapshot.validate()
    return NativeTarOrdinalMapBuildSummary(
        shard_rows=(row_count,),
        records_checked=row_count,
        skip_marker_records=skip_marker_records,
        top_level_skip_marker_records=top_level_skip_marker_records,
        custom_skip_marker_records=custom_skip_marker_records,
        input_snapshot=snapshot,
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
    source_identity_before = (
        _source_identity(manifest_path),
        _source_identity(tar_path),
    )
    index_identity_before = (
        _file_identity(manifest_index_path),
        _file_identity(tar_index_path),
    )
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
        if source_identity_before != (
            _source_identity(manifest_path),
            _source_identity(tar_path),
        ):
            raise ValueError(
                f"A native NeMo source changed while building the ordinal map for "
                f"manifest={manifest_path!r} tar={tar_path!r}"
            )
        if index_identity_before != (
            _file_identity(manifest_index_path),
            _file_identity(tar_index_path),
        ):
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
    num_workers: int = 1,
) -> NativeTarOrdinalMapBuildSummary:
    """Write every shard of one routing collection from a stable source snapshot.

    Multiple workers use spawn and write only their preassigned output paths.
    Results are restored to input order so scheduling cannot affect pack layout.
    """
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
    if num_workers < 1:
        raise ValueError(f"Native NeMo ordinal-map workers must be positive, got {num_workers}")

    input_snapshot = NativeTarOrdinalMapInputSnapshot.capture(
        manifest_paths,
        tar_paths,
        manifest_index_paths,
        tar_index_paths,
    )
    tasks = tuple(
        (
            output_path,
            manifest_path,
            manifest_index_path,
            tar_path,
            tar_index_path,
            sentinel_override,
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
    if num_workers == 1 or len(tasks) <= 1:
        shard_summaries = tuple(_write_nemo_tar_ordinal_map_shard_task(task) for task in tasks)
    else:
        # Each task owns a distinct, preassigned output path. Results are placed
        # back into input order so process scheduling cannot affect the array
        # shard order or the resulting idxpack layout hash.
        ordered_summaries: list[NativeTarOrdinalMapBuildSummary | None] = [None] * len(tasks)
        with ProcessPoolExecutor(
            max_workers=min(num_workers, len(tasks)),
            mp_context=get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(_write_nemo_tar_ordinal_map_shard_task, task): shard_index
                for shard_index, task in enumerate(tasks)
            }
            try:
                for future in as_completed(futures):
                    ordered_summaries[futures[future]] = future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        assert all(summary is not None for summary in ordered_summaries)
        shard_summaries = tuple(summary for summary in ordered_summaries if summary is not None)
    input_snapshot.validate()
    return NativeTarOrdinalMapBuildSummary(
        shard_rows=tuple(summary.records_checked for summary in shard_summaries),
        records_checked=sum(summary.records_checked for summary in shard_summaries),
        skip_marker_records=sum(summary.skip_marker_records for summary in shard_summaries),
        top_level_skip_marker_records=sum(summary.top_level_skip_marker_records for summary in shard_summaries),
        custom_skip_marker_records=sum(summary.custom_skip_marker_records for summary in shard_summaries),
        input_snapshot=input_snapshot,
    )


def _write_nemo_tar_ordinal_map_shard_task(
    task: tuple[str | Path, str, str | Path, str, str | Path, int | None],
) -> NativeTarOrdinalMapBuildSummary:
    """Process-pool entry point for one independently addressable shard."""
    (
        output_path,
        manifest_path,
        manifest_index_path,
        tar_path,
        tar_index_path,
        sentinel_override,
    ) = task
    return write_nemo_tar_ordinal_map_shard(
        output_path,
        manifest_path=manifest_path,
        manifest_index_path=manifest_index_path,
        tar_path=tar_path,
        tar_index_path=tar_index_path,
        tar_sentinel_size_override=sentinel_override,
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
