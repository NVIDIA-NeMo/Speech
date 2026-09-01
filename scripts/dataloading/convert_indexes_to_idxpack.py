#!/usr/bin/env python
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

"""
Convert an existing NeMo/Lhotse input_cfg and its ``.idx`` sidecars to one
dataset-level ``.idxpack``.

The input YAML may contain nested groups and transform wrappers. This initial
integration packs the formats consumed by the indexed runtime: native NeMo
manifests/tars, Nemotron text JSONL/tars, and ShareGPT/multimodal-conversation
JSONL manifests. No source manifest or tar is rescanned: the command consumes
existing ``.idx`` files, normally from ``--indexes-root``. Native tar sidecars keep their
headerless uint64 layout. The converter determines compatibility from
the data itself: a sidecar is current exactly when its sentinel equals the
physical local or remote source size. Stale sidecars fail before packing;
rebuild them with build_indexes.py ``--force``.

Example::

    python scripts/dataloading/convert_indexes_to_idxpack.py \
        --indexes-root /data/indexes \
        --output /data/index-packs/speech.idxpack \
        /data/configs/speech.yaml
"""

from __future__ import annotations

import logging
import os
import re
import struct
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Optional

import click
from lhotse.index_pack import IndexPack, IndexPackCollectionSpec, write_index_pack
from lhotse.indexing import index_file_path
from omegaconf import DictConfig, ListConfig
from scripts.dataloading._sharegpt_route_cli import ensure_sharegpt_route
from scripts.dataloading._sharegpt_route_config import discover_sharegpt_route_specs
from scripts.dataloading.build_indexes import (
    _NO_INDEX_TYPES,
    _TRANSFORM_TYPES,
    JSONL,
    NEMO_TAR,
    WDS_TAR_V2,
    _discover_share_gpt_webdataset,
    _expand_jsonl,
    _expand_tars,
    _flatten_path_spec,
    _load_input_cfg,
    _resolve_input_cfg,
)
from scripts.dataloading.validate_idxpack_records import validate_idxpack_json_records

from nemo.collections.common.data.lhotse.indexed_adapters import validate_wds_v2_tar_index, wds_v2_index_path


def _add_collection(
    collections: list[IndexPackCollectionSpec],
    *,
    role: str,
    kind: str,
    source_spec,
    paths,
) -> None:
    paths = tuple(map(str, paths))
    if not paths:
        return
    candidate = IndexPackCollectionSpec(
        role=role,
        kind=kind,
        source_spec=source_spec,
        paths=paths,
    )
    for existing in collections:
        if existing.key != candidate.key:
            continue
        if existing.paths != candidate.paths:
            raise ValueError(
                f"Collection-key collision for role={role!r}, kind={kind!r}, "
                f"source_spec={source_spec!r}"
            )
        return
    collections.append(candidate)


_REBUILD_TAR_INDEXES_HINT = (
    "Rebuild native tar indexes with: python scripts/dataloading/build_indexes.py "
    "--force [--indexes-root INDEXES_ROOT] INPUT_CFG."
)
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_IN_BAND_NEMO_TAR_INDEX_MAGIC = b"NEMOTAR\0"
_MAX_VERIFIED_ZERO_PADDING_GROWTH = 64 * 1024 * 1024

# The source .idx format stays intentionally unversioned. Sentinel/source-size
# equality is the semantic compatibility check; the output .idxpack already
# has its own magic and version.


def _is_remote_path(path) -> bool:
    return bool(_URL_RE.match(str(path)))


def _resolve_local_sidecar(path: str, indexes_root) -> Path:
    idx_path = index_file_path(path, indexes_root)
    if _is_remote_path(idx_path):
        raise ValueError(
            "Index-pack conversion requires local .idx sidecars; "
            f"resolved {path} to remote sidecar {idx_path}."
        )
    return Path(idx_path)


def _source_size(path: str) -> int:
    if not _is_remote_path(path):
        try:
            return Path(path).stat().st_size
        except FileNotFoundError as ex:
            raise FileNotFoundError(f"Indexed source not found: {path}") from ex

    if str(path).startswith("s3://"):
        from lhotse.audio.source import resolve_s3_to_local_mirror

        mirrored_path = resolve_s3_to_local_mirror(str(path))
        if mirrored_path != str(path):
            try:
                return Path(mirrored_path).stat().st_size
            except FileNotFoundError:
                pass

    try:
        from lhotse.ais import AISRangeReader

        with AISRangeReader(str(path)) as source:
            return int(source.size)
    except Exception as ex:
        raise ValueError(
            f"Could not determine the current size of remote tar source {path} "
            f"from object metadata ({ex}). Strict conversion cannot safely use "
            f"its sidecar. {_REBUILD_TAR_INDEXES_HINT}"
        ) from ex


def _read_raw_tar_sentinel(idx_path: Path) -> tuple[int, os.stat_result]:
    try:
        index_stat = idx_path.stat()
    except FileNotFoundError as ex:
        raise FileNotFoundError(f"Missing .idx sidecar: {idx_path}") from ex

    if index_stat.st_size < 8 or index_stat.st_size % 8:
        raise ValueError(
            f"Invalid native tar index {idx_path}: size must be a positive "
            f"multiple of 8 bytes, got {index_stat.st_size}. "
            f"{_REBUILD_TAR_INDEXES_HINT}"
        )

    with idx_path.open("rb") as stream:
        first_word = stream.read(8)
        stream.seek(-8, os.SEEK_END)
        (sentinel,) = struct.unpack("<Q", stream.read(8))

    if first_word == _IN_BAND_NEMO_TAR_INDEX_MAGIC:
        raise ValueError(
            f"Native tar index {idx_path} uses the incompatible experimental "
            f"in-band NEMOTAR header. {_REBUILD_TAR_INDEXES_HINT}"
        )
    return sentinel, index_stat


def _repair_local_native_tar_sidecar(path: str, repair_root) -> Path:
    if _is_remote_path(path):
        raise ValueError(f"Refusing to repair non-local native tar source: {path}")
    from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index as create_nemo_tar_index

    repair_idx = _resolve_local_sidecar(path, repair_root)
    repair_idx.parent.mkdir(parents=True, exist_ok=True)
    source_size = Path(path).stat().st_size
    create_nemo_tar_index(path, repair_idx)
    sentinel, index_stat = _read_raw_tar_sentinel(repair_idx)
    if sentinel != source_size:
        raise ValueError(
            f"Private native-tar repair {repair_idx} has sentinel {sentinel}, "
            f"but source {path} is {source_size} bytes."
        )
    if Path(path).stat().st_mtime_ns > index_stat.st_mtime_ns:
        raise ValueError(
            f"Source {path} changed while rebuilding private sidecar {repair_idx}."
        )
    logging.info(
        "Rebuilt stale local native-tar sidecar privately: source=%s bytes=%d sidecar=%s",
        path,
        source_size,
        repair_idx,
    )
    return repair_idx


def _verify_trailing_zero_padding(path: str, start: int, source_size: int) -> None:
    growth = source_size - start
    if growth <= 0 or growth > _MAX_VERIFIED_ZERO_PADDING_GROWTH:
        raise ValueError(
            f"Cannot accept native tar growth for {path}: {growth} bytes is outside "
            f"the verified zero-padding bound (1..{_MAX_VERIFIED_ZERO_PADDING_GROWTH})."
        )
    if _is_remote_path(path):
        from lhotse.ais import AISRangeReader

        stream = AISRangeReader(path)
    else:
        stream = Path(path).open("rb")
    try:
        stream.seek(start)
        remaining = growth
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(
                    f"Short read while verifying trailing padding for {path}: "
                    f"{remaining} bytes remain"
                )
            if any(chunk):
                raise ValueError(
                    f"Native tar {path} contains non-zero data after stale sentinel "
                    f"{start}; rebuild its index. {_REBUILD_TAR_INDEXES_HINT}"
                )
            remaining -= len(chunk)
    finally:
        stream.close()


def _validate_native_tar_sidecar(
    path: str,
    indexes_root,
    *,
    accept_trailing_zero_padding: bool = False,
    repair_stale_local_sidecars_root=None,
) -> tuple[Path, int | None]:
    shared_idx_path = _resolve_local_sidecar(path, indexes_root)
    repair_idx_path = (
        _resolve_local_sidecar(path, repair_stale_local_sidecars_root)
        if repair_stale_local_sidecars_root is not None and not _is_remote_path(path)
        else None
    )
    idx_path = (
        repair_idx_path
        if repair_idx_path is not None and repair_idx_path.exists()
        else shared_idx_path
    )
    sentinel, index_stat = _read_raw_tar_sentinel(idx_path)

    source_size = _source_size(path)
    source_size_override = None
    if sentinel != source_size:
        accepted_padding = False
        if accept_trailing_zero_padding and sentinel < source_size:
            try:
                _verify_trailing_zero_padding(path, sentinel, source_size)
            except ValueError:
                if _is_remote_path(path) or repair_stale_local_sidecars_root is None:
                    raise
            else:
                source_size_override = source_size
                accepted_padding = True
                logging.info(
                    "Accepted verified trailing zero padding for %s: sentinel=%d source_size=%d growth=%d",
                    path,
                    sentinel,
                    source_size,
                    source_size - sentinel,
                )
        if not accepted_padding:
            if (
                not _is_remote_path(path)
                and repair_stale_local_sidecars_root is not None
            ):
                idx_path = _repair_local_native_tar_sidecar(
                    path, repair_stale_local_sidecars_root
                )
                sentinel, index_stat = _read_raw_tar_sentinel(idx_path)
            else:
                raise ValueError(
                    f"Native tar index {idx_path} has sentinel {sentinel}, but source "
                    f"{path} is {source_size} bytes. {_REBUILD_TAR_INDEXES_HINT}"
                )

    if not _is_remote_path(path):
        source_stat = Path(path).stat()
        if (
            source_stat.st_mtime_ns > index_stat.st_mtime_ns
            and source_size_override is None
        ):
            if repair_stale_local_sidecars_root is not None:
                idx_path = _repair_local_native_tar_sidecar(
                    path, repair_stale_local_sidecars_root
                )
            else:
                raise ValueError(
                    f"Source {path} is newer than native tar index {idx_path}. "
                    f"{_REBUILD_TAR_INDEXES_HINT}"
                )
    return idx_path, source_size_override


def _preflight_native_tar_sidecars(
    collections,
    indexes_root,
    *,
    accept_trailing_zero_padding: bool = False,
    repair_stale_local_sidecars_root=None,
) -> tuple[dict[str, int], dict[str, Path]]:
    validated = set()
    source_size_overrides = {}
    index_path_overrides = {}
    for collection in collections:
        if collection.kind != NEMO_TAR or not collection.offsets_required:
            continue
        for path in collection.paths:
            path = str(path)
            if path in validated:
                continue
            _idx_path, override = _validate_native_tar_sidecar(
                path,
                indexes_root,
                accept_trailing_zero_padding=accept_trailing_zero_padding,
                repair_stale_local_sidecars_root=repair_stale_local_sidecars_root,
            )
            if override is not None:
                source_size_overrides[path] = override
            shared_idx_path = _resolve_local_sidecar(path, indexes_root)
            if _idx_path != shared_idx_path:
                index_path_overrides[path] = _idx_path
            validated.add(path)
    return source_size_overrides, index_path_overrides


def _preflight_wds_v2_sidecars(collections, indexes_root) -> dict[str, Path]:
    index_path_overrides = {}
    validated = set()
    for collection in collections:
        if collection.kind != WDS_TAR_V2 or not collection.offsets_required:
            continue
        for path in collection.paths:
            path = str(path)
            if path in validated:
                continue
            idx_path = wds_v2_index_path(path, indexes_root)
            validate_wds_v2_tar_index(path, idx_path=idx_path)
            index_path_overrides[path] = idx_path
            validated.add(path)
    return index_path_overrides


def _path_only_source_sizes(collections) -> dict[str, int]:
    """Capture live sizes so path-only pack segments remain reusable safely."""
    return {
        str(path): _source_size(str(path))
        for collection in collections
        if not collection.offsets_required
        for path in collection.paths
    }


def _discover_paths_collections(
    raw_paths,
    collections: list[IndexPackCollectionSpec],
) -> None:
    jsonls = []
    tars = []
    for raw in _flatten_path_spec(raw_paths):
        for expanded in _expand_jsonl(raw):
            path = Path(expanded)
            if path.is_dir():
                tars.extend(map(str, sorted(path.rglob("*.tar"))))
            elif path.suffix == ".tar":
                tars.append(str(path))
            else:
                jsonls.append(str(path))
    if jsonls and tars:
        raise ValueError(
            "Packed Nemotron text paths must be homogeneous. Split mixed "
            "JSONL/tar paths into separate dataset entries."
        )
    _add_collection(
        collections,
        role="paths",
        kind=NEMO_TAR if tars else JSONL,
        source_spec=raw_paths,
        paths=jsonls or tars,
    )


def _require_scalar_spec(value, field: str) -> None:
    if not isinstance(value, (str, Path)):
        raise ValueError(
            f"Packed {field} must be a string/Path (brace expansion is "
            "supported); list forms are not supported."
        )


def _is_nonempty_flat_path_list(value) -> bool:
    return (
        isinstance(value, (list, tuple, ListConfig))
        and bool(value)
        and all(isinstance(item, (str, Path)) for item in value)
    )


def _require_scalar_or_flat_path_list(value, field: str) -> None:
    if isinstance(value, (str, Path)):
        return
    if _is_nonempty_flat_path_list(value):
        return
    raise ValueError(
        f"Packed native NeMo {field} must be a string/Path or a non-empty flat "
        "list of strings/Paths; nested and weighted list forms are not supported."
    )


def _shard_number(path: str) -> int | None:
    matches = re.findall(r"\d+", Path(path).stem)
    return int(matches[-1]) if matches else None


def _validate_native_pair(manifests: list[str], tars: list[str]) -> None:
    if len(manifests) != len(tars):
        raise ValueError(
            "Packed native NeMo data requires one manifest per tar shard: "
            f"manifests={len(manifests)}, tars={len(tars)}"
        )
    if len(manifests) < 2:
        return
    manifest_ids = [_shard_number(path) for path in manifests]
    tar_ids = [_shard_number(path) for path in tars]
    if None in manifest_ids or None in tar_ids:
        raise ValueError(
            "Cannot verify native NeMo manifest/tar shard identity from file "
            "names; use numbered shard names."
        )
    if manifest_ids != tar_ids:
        raise ValueError(
            "Native NeMo manifest/tar shards are not positionally aligned: "
            f"manifest ids={manifest_ids}, tar ids={tar_ids}"
        )


def _expand_flat_native_pairs(manifest_specs, tar_specs) -> tuple[list[str], list[str]]:
    if len(manifest_specs) != len(tar_specs):
        raise ValueError(
            "Packed native NeMo flat lists require one tar path spec per "
            f"manifest path spec: manifests={len(manifest_specs)}, tars={len(tar_specs)}"
        )

    manifests: list[str] = []
    tars: list[str] = []
    for position, (manifest_spec, tar_spec) in enumerate(
        zip(manifest_specs, tar_specs)
    ):
        pair_manifests = _expand_jsonl(manifest_spec)
        pair_tars = _expand_tars(tar_spec)
        if len(pair_manifests) != len(pair_tars):
            raise ValueError(
                "Packed native NeMo flat lists require each positional manifest/tar "
                f"pair to expand to the same number of shards; position={position}, "
                f"manifests={len(pair_manifests)}, tars={len(pair_tars)}"
            )
        manifests.extend(pair_manifests)
        tars.extend(pair_tars)
    return manifests, tars


def discover_pack_collections(
    entry,
    collections: Optional[list[IndexPackCollectionSpec]] = None,
    *,
    data_blend_dir: str | Path | None = None,
) -> list[IndexPackCollectionSpec]:
    """Discover ordered, runtime-addressable collections in one input_cfg."""
    if collections is None:
        collections = []
    if isinstance(entry, (list, ListConfig)):
        for item in entry:
            discover_pack_collections(item, collections, data_blend_dir=data_blend_dir)
        return collections
    if not isinstance(entry, (dict, DictConfig)):
        return collections

    typ = entry.get("type")
    if typ in _NO_INDEX_TYPES:
        return collections

    if typ is None:
        for value in entry.values():
            discover_pack_collections(value, collections, data_blend_dir=data_blend_dir)
        return collections

    if typ == "group":
        sub = _resolve_input_cfg(entry.get("input_cfg"), data_blend_dir)
        if sub is not None:
            discover_pack_collections(sub, collections, data_blend_dir=data_blend_dir)
        return collections

    if typ in _TRANSFORM_TYPES:
        sub = _resolve_input_cfg(entry.get("input_cfg"), data_blend_dir)
        if sub is not None:
            discover_pack_collections(sub, collections, data_blend_dir=data_blend_dir)
            return collections
        if entry.get("manifest_filepath") is None:
            return collections

    supported = {
        "nemo",
        "nemo_tarred",
        "multimodal_conversation",
        "nemotron_text_converation",
        "materialized_sft_messages",
        "share_gpt",
        "share_gpt_webdataset",
        *_TRANSFORM_TYPES,
    }
    if typ not in supported:
        raise NotImplementedError(
            f"idxpack conversion does not support dataset type {typ!r}."
        )

    if typ == "share_gpt_webdataset":
        version = int(entry.get("wds_sample_index_version", 1))
        if version != 2:
            raise NotImplementedError(
                "Packed share_gpt_webdataset requires wds_sample_index_version: 2; " f"got {version}."
            )
        jobs = []
        data_dir = entry.get("data_dir")
        if data_dir is None:
            raise ValueError("Packed WDS v2 requires share_gpt_webdataset.data_dir")
        _discover_share_gpt_webdataset(
            data_dir,
            jobs,
            None,
            index_version=version,
        )
        _add_collection(
            collections,
            role="wds_tar",
            kind=WDS_TAR_V2,
            source_spec=data_dir,
            paths=[job.path for job in jobs if job.kind == WDS_TAR_V2],
        )
        return collections
    if (
        typ
        in {
            "nemo",
            "nemo_tarred",
            "multimodal_conversation",
            "share_gpt",
            *_TRANSFORM_TYPES,
        }
        and entry.get("manifest_filepath") is not None
    ):
        raw = entry.get("manifest_filepath")
        collection_mode = (
            typ == "share_gpt" and entry.get("tar_lookup_mode") == "collection"
        )
        if collection_mode:
            route = entry.get("tar_routing_filepath")
            legacy_route = entry.get("tar_routing_index")
            if route and legacy_route and str(route) != str(legacy_route):
                raise ValueError("tar_routing_filepath and tar_routing_index disagree")
            route = route or legacy_route
            if not isinstance(route, (str, Path)) or not str(route).endswith(
                ".sgroute"
            ):
                raise ValueError(
                    "Packed ShareGPT collection mode requires tar_routing_filepath " "with the .sgroute suffix."
                )
            _require_scalar_or_flat_path_list(raw, "manifest_filepath")
            raw_tars = entry.get("tarred_audio_filepaths")
            if raw_tars is None:
                raise ValueError(
                    "Packed ShareGPT collection mode requires tarred_audio_filepaths."
                )
            _require_scalar_or_flat_path_list(raw_tars, "tarred_audio_filepaths")
            manifests = _expand_jsonl(raw)
            tars = _expand_tars(raw_tars)
            _add_collection(
                collections,
                role="manifest",
                kind=JSONL,
                source_spec=raw,
                paths=manifests,
            )
            _add_collection(
                collections,
                role="tar_collection",
                kind=NEMO_TAR,
                source_spec=raw_tars,
                paths=tars,
            )
            return collections
        if typ in {"multimodal_conversation", "share_gpt"}:
            _require_scalar_spec(raw, "manifest_filepath")
        else:
            _require_scalar_or_flat_path_list(raw, "manifest_filepath")
        raw_tars = entry.get("tarred_audio_filepaths")
        if raw_tars is None:
            manifests = _expand_jsonl(raw)
            tars = None
        else:
            if typ == "share_gpt":
                raise NotImplementedError(
                    "Packed ShareGPT supports JSONL manifests with direct/remote "
                    "audio paths, not paired audio tar files."
                )
            if typ == "multimodal_conversation":
                raise NotImplementedError(
                    "Packed multimodal_conversation supports JSONL manifests with "
                    "direct/remote audio paths, not paired audio tar files."
                )
            _require_scalar_or_flat_path_list(raw_tars, "tarred_audio_filepaths")
            manifest_is_flat_list = _is_nonempty_flat_path_list(raw)
            tar_is_flat_list = _is_nonempty_flat_path_list(raw_tars)
            if manifest_is_flat_list != tar_is_flat_list:
                raise ValueError(
                    "Packed native NeMo manifest_filepath and tarred_audio_filepaths "
                    "must both use scalar path specs or both use non-empty flat lists."
                )
            if manifest_is_flat_list:
                manifests, tars = _expand_flat_native_pairs(raw, raw_tars)
            else:
                manifests = _expand_jsonl(raw)
                tars = _expand_tars(raw_tars)
                _validate_native_pair(manifests, tars)
        _add_collection(
            collections,
            role="manifest",
            kind=JSONL,
            source_spec=raw,
            paths=manifests,
        )
        if tars is not None:
            _add_collection(
                collections,
                role="tar",
                kind=NEMO_TAR,
                source_spec=raw_tars,
                paths=tars,
            )

    if typ in {"nemotron_text_converation", "materialized_sft_messages"}:
        _discover_paths_collections(entry.get("paths"), collections)

    return collections


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_validated_index_pack(
    output,
    collections,
    *,
    indexes_root,
    overwrite: bool,
    source_size_overrides,
    index_path_overrides,
    record_validation_workers: int,
):
    """Build privately, validate every JSON record, then publish atomically."""
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Index pack already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name(
        f".{output.name}.record-validation.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        write_index_pack(
            staged,
            collections,
            indexes_root=indexes_root,
            source_size_overrides=source_size_overrides,
            index_path_overrides=index_path_overrides,
        )
        summary = validate_idxpack_json_records(
            staged,
            collections,
            report=logging.error,
            num_workers=record_validation_workers,
        )
        if overwrite:
            os.replace(staged, output)
        else:
            try:
                os.link(staged, output)
            except FileExistsError as ex:
                raise FileExistsError(f"Index pack already exists: {output}") from ex
            staged.unlink()
        _fsync_directory(output.parent)
        return summary
    finally:
        if staged.exists():
            staged.unlink()


@click.command(context_settings={"show_default": True})
@click.argument("input_cfg", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(dir_okay=False))
@click.option(
    "--indexes-root",
    default=None,
    help=(
        "Root of the existing mirrored .idx sidecars. Omit for sidecars next "
        "to sources. Stale native tar indexes must be rebuilt with "
        "build_indexes.py --force before conversion."
    ),
)
@click.option(
    "--data-blend-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Resolve ${data_blend_dir} in nested input_cfg references.",
)
@click.option(
    "--overwrite", is_flag=True, help="Atomically replace an existing output pack."
)
@click.option(
    "--native-tar-paths-only",
    is_flag=True,
    help=(
        "Store native NeMo tar shard names without copying tar-member offsets. "
        "Use for AIS URL-backed audio; manifest offsets remain fully packed."
    ),
)
@click.option(
    "--accept-trailing-zero-tar-padding",
    is_flag=True,
    help=(
        "Accept a stale native-tar sentinel only when the source grew by at most "
        "64 MiB and every appended byte is zero; rewrite only the packed sentinel."
    ),
)
@click.option(
    "--repair-stale-local-native-tar-sidecars-root",
    type=click.Path(file_okay=False),
    default=None,
    help=(
        "Privately rebuild only stale local native-tar sidecars under this "
        "mirror root instead of mutating --indexes-root. Remote sources are "
        "never repaired by this option."
    ),
)
@click.option(
    "--record-validation-workers",
    type=click.IntRange(min=1),
    default=1,
    help="Process workers for disjoint exhaustive JSONL record validation.",
)
@click.option(
    "--dry-run", is_flag=True, help="Print discovered collections without writing."
)
def main(
    input_cfg: str,
    output: str,
    indexes_root: Optional[str],
    data_blend_dir: Optional[str],
    overwrite: bool,
    native_tar_paths_only: bool,
    accept_trailing_zero_tar_padding: bool,
    repair_stale_local_native_tar_sidecars_root: Optional[str],
    record_validation_workers: int,
    dry_run: bool,
) -> None:
    """Convert one INPUT_CFG dataset and its existing sidecars to one idxpack.

    Native tar indexes are validated against local or remote source metadata.
    A stale sentinel must be rebuilt with build_indexes.py --force.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = _load_input_cfg(input_cfg, data_blend_dir)
    collections = discover_pack_collections(config, data_blend_dir=data_blend_dir)
    route_specs = discover_sharegpt_route_specs(config, data_blend_dir=data_blend_dir)
    if native_tar_paths_only:
        if any(collection.role == "tar_collection" for collection in collections):
            raise click.ClickException(
                "ShareGPT collection mode requires offset-bearing tar_collection indexes; "
                "--native-tar-paths-only is not allowed."
            )

        collections = [
            (
                replace(collection, offsets_required=False)
                if collection.kind == NEMO_TAR and collection.role == "tar"
                else collection
            )
            for collection in collections
        ]
    num_paths = sum(len(collection.paths) for collection in collections)
    click.echo(
        f"Discovered {len(collections)} collections with {num_paths} ordered paths."
    )
    if dry_run:
        for collection in collections:
            click.echo(
                f"  role={collection.role} kind={collection.kind} "
                f"paths={len(collection.paths)} offsets={collection.offsets_required} key={collection.key.hex()}"
            )
        return
    try:
        source_size_overrides, index_path_overrides = _preflight_native_tar_sidecars(
            collections,
            indexes_root,
            accept_trailing_zero_padding=accept_trailing_zero_tar_padding,
            repair_stale_local_sidecars_root=repair_stale_local_native_tar_sidecars_root,
        )
        source_size_overrides.update(_path_only_source_sizes(collections))
        wds_index_paths = _preflight_wds_v2_sidecars(collections, indexes_root)
        index_path_overrides.update(wds_index_paths)
        for route_spec in route_specs:
            route_path = Path(route_spec.route_path)
            if not route_path.is_absolute():
                route_path = Path(output).parent / route_path
            ensure_sharegpt_route(
                route_path,
                manifest_paths=route_spec.manifest_paths,
                tar_paths=route_spec.tar_paths,
                manifest_specs=route_spec.manifest_specs,
                indexes_root=indexes_root,
                audio_prefix_map=route_spec.audio_prefix_map,
                audio_placeholders=route_spec.audio_placeholders,
                build_if_missing=True,
            )
        summary = _write_validated_index_pack(
            output,
            collections,
            indexes_root=indexes_root,
            overwrite=overwrite,
            source_size_overrides=source_size_overrides,
            index_path_overrides=index_path_overrides,
            record_validation_workers=record_validation_workers,
        )
    except (OSError, ValueError) as ex:
        raise click.ClickException(str(ex)) from ex
    with IndexPack(output) as pack:
        click.echo(
            f"Wrote {output}: collections={pack.num_collections} "
            f"segments={pack.num_segments} layout={pack.layout_hash.hex()} "
            f"json_records_validated={summary.records_checked} "
            f"skip_marker_records={summary.skip_marker_records}"
        )


if __name__ == "__main__":
    main()
