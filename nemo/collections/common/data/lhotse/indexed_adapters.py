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
import hashlib
import json
import os
import re
import struct
import tarfile
import uuid
from collections import OrderedDict
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import NamedTuple
from urllib.parse import urlsplit

import numpy as np

try:
    from lhotse.audio.source import resolve_s3_to_local_mirror as _resolve_s3_to_local_mirror
except ImportError:
    _resolve_s3_to_local_mirror = None

# Tar block size + the all-zeros block that marks end-of-archive in tar.
_TAR_BLOCK_SIZE = 512
_TAR_ZERO_BLOCK = b"\0" * _TAR_BLOCK_SIZE

# Recognized URL schemes whose authority ("host" component) is part of the
# logical path (e.g. the bucket name). Stripping just the scheme keeps the
# bucket+key in the relative path used to mirror under indexes_root.
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

WDS_V2_INDEX_VERSION = 2
WDS_V2_INDEX_FORMAT = "nemo-wds-sample-index"
WDS_V2_SAMPLE_KEY_ALGORITHM = "posix-parent-plus-basename-prefix-before-first-dot-v1"
_WDS_V2_INDEX_SUFFIX = ".wds-v2.idx"
_WDS_V2_METADATA_SUFFIX = ".meta.json"
_SUPPORTED_REMOTE_RANGE_SCHEMES = frozenset({"ais", "s3"})


class TarSampleMember(NamedTuple):
    """One named non-JSON payload from a variable-member WebDataset sample."""

    name: str
    data: bytes


class TarSampleBundle(NamedTuple):
    """One v2 WebDataset sample with exactly one JSON object and named payloads."""

    sample_key: str
    json_data: dict
    audio_members: tuple[TarSampleMember, ...]
    source_path: str | None = None
    source_range_bytes: int | None = None


def wds_sample_key(member_name: str) -> str:
    """Return the v2 logical sample key for one regular tar member name."""
    if not isinstance(member_name, str) or not member_name:
        raise ValueError(f"WebDataset member name must be a non-empty string, got {member_name!r}")
    if member_name.startswith("/"):
        raise ValueError(f"WebDataset member name must not be absolute: {member_name!r}")
    raw_parts = member_name.split("/")
    if any(part in (".", "..") for part in raw_parts):
        raise ValueError(f"WebDataset member name has a forbidden path component: {member_name!r}")
    path = PurePosixPath(member_name)
    if path.is_absolute():
        raise ValueError(f"WebDataset member name must not be absolute: {member_name!r}")
    basename = path.name
    if not basename:
        raise ValueError(f"WebDataset member name has an empty basename: {member_name!r}")
    first_dot = basename.find(".")
    if first_dot < 0 or first_dot == len(basename) - 1:
        raise ValueError(f"WebDataset member basename is without an extension: {member_name!r}")
    if first_dot == 0:
        raise ValueError(f"WebDataset member basename has an empty prefix: {member_name!r}")
    prefix = basename[:first_dot]
    parent = path.parent
    return prefix if str(parent) == "." else str(parent / prefix)


def wds_v2_index_path(data_path: str | Path, indexes_root: str | Path | None = None) -> Path:
    """Return the local mirrored ``.wds-v2.idx`` path for ``data_path``."""
    from lhotse.indexing import index_file_path

    legacy_path = str(index_file_path(str(data_path), indexes_root))
    if _is_remote_path(legacy_path):
        raise ValueError(
            "WDS v2 sidecars must be local; pass indexes_root or an explicit idx_path "
            f"for remote source {data_path!s}."
        )
    if not legacy_path.endswith(".idx"):
        raise ValueError(f"Unexpected Lhotse index path without .idx suffix: {legacy_path}")
    return Path(legacy_path[: -len(".idx")] + _WDS_V2_INDEX_SUFFIX)


def wds_v2_metadata_path(idx_path: str | Path) -> Path:
    """Return the canonical metadata companion path for a WDS v2 offset sidecar."""
    idx_path = Path(idx_path)
    if not str(idx_path).endswith(_WDS_V2_INDEX_SUFFIX):
        raise ValueError(f"WDS v2 index path must end with {_WDS_V2_INDEX_SUFFIX!r}: {idx_path}")
    return Path(str(idx_path) + _WDS_V2_METADATA_SUFFIX)


def read_exact_range(path: str | Path, start: int, end: int) -> bytes:
    """Read the exact half-open byte range from a local file or AIStore-backed URL."""
    if start < 0 or end < start:
        raise ValueError(f"Invalid byte range [{start}, {end})")
    size = end - start
    if size == 0:
        return b""
    source_path_str = os.fspath(path)
    read_path_str = _resolve_data_path(source_path_str)
    if not _is_remote_path(read_path_str):
        fd = os.open(read_path_str, os.O_RDONLY)
        try:
            chunks = []
            position = start
            while position < end:
                chunk = os.pread(fd, end - position, position)
                if not chunk:
                    break
                chunks.append(chunk)
                position += len(chunk)
            data = b"".join(chunks)
        finally:
            os.close(fd)
    else:
        scheme = urlsplit(read_path_str).scheme.lower()
        if scheme not in _SUPPORTED_REMOTE_RANGE_SCHEMES:
            raise ValueError(
                f"Unsupported remote range-read scheme {scheme!r} for {source_path_str!r}; "
                f"supported schemes are {sorted(_SUPPORTED_REMOTE_RANGE_SCHEMES)}"
            )
        from lhotse.ais import AISRangeReader

        with AISRangeReader(read_path_str) as source:
            if end > source.size:
                raise EOFError(
                    f"Short indexed read from {source_path_str}: requested [{start}, {end}), source size is {source.size}"
                )
            source.seek(start)
            chunks = []
            remaining = size
            while remaining:
                chunk = source.read(remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
    if len(data) != size:
        raise EOFError(
            f"Short indexed read from {source_path_str}: requested [{start}, {end}), received {len(data)} bytes"
        )
    return data


def create_wds_v2_tar_index(
    tar_path: str | Path,
    idx_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Build and atomically publish an immutable variable-member WDS v2 index pair."""
    tar_path_str = os.fspath(tar_path)
    idx_path = wds_v2_index_path(tar_path_str) if idx_path is None else Path(idx_path)
    metadata_path = wds_v2_metadata_path(idx_path) if metadata_path is None else Path(metadata_path)
    _require_wds_v2_sidecar_paths(idx_path, metadata_path)
    if idx_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"WDS v2 sidecar already exists for {tar_path_str}: idx={idx_path}, metadata={metadata_path}"
        )

    before = indexed_source_metadata(tar_path_str)
    offsets, regular_member_count, source_size = _scan_wds_v2_tar(tar_path_str)
    after = indexed_source_metadata(tar_path_str)
    if after != before:
        raise ValueError(f"WebDataset tar changed while indexing: {tar_path_str}")
    source = after
    if source["size_bytes"] != source_size:
        raise ValueError(
            f"WebDataset tar size changed while indexing {tar_path_str}: "
            f"scanned={source_size}, current={source['size_bytes']}"
        )

    raw_offsets = b"".join(struct.pack("<Q", offset) for offset in (*offsets, source_size))
    metadata = {
        "format": WDS_V2_INDEX_FORMAT,
        "version": WDS_V2_INDEX_VERSION,
        "sample_key_algorithm": WDS_V2_SAMPLE_KEY_ALGORITHM,
        "source": source,
        "regular_member_count": regular_member_count,
        "sample_count": len(offsets),
        "offsets_sha256": hashlib.sha256(raw_offsets).hexdigest(),
    }
    raw_metadata = _canonical_json_bytes(metadata)

    idx_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    idx_stem = idx_path.name[: -len(_WDS_V2_INDEX_SUFFIX)]
    tmp_idx = idx_path.with_name(f".{idx_stem}.tmp.{token}{_WDS_V2_INDEX_SUFFIX}")
    tmp_metadata = wds_v2_metadata_path(tmp_idx)
    published_idx = False
    try:
        _write_fsynced(tmp_idx, raw_offsets)
        _write_fsynced(tmp_metadata, raw_metadata)
        validate_wds_v2_tar_index(tar_path_str, tmp_idx, tmp_metadata)
        try:
            os.link(tmp_idx, idx_path)
            published_idx = True
            os.link(tmp_metadata, metadata_path)
        except OSError as ex:
            if published_idx:
                idx_path.unlink(missing_ok=True)
            if not isinstance(ex, FileExistsError):
                raise
            raise FileExistsError(
                f"WDS v2 sidecar already exists for {tar_path_str}: idx={idx_path}, metadata={metadata_path}"
            ) from ex
        _fsync_directory(idx_path.parent)
        if metadata_path.parent != idx_path.parent:
            _fsync_directory(metadata_path.parent)
    finally:
        tmp_idx.unlink(missing_ok=True)
        tmp_metadata.unlink(missing_ok=True)
    return idx_path, metadata_path


def validate_wds_v2_tar_index(
    tar_path: str | Path,
    idx_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Validate one WDS v2 offset/metadata pair against its configured source."""
    tar_path_str = os.fspath(tar_path)
    idx_path = wds_v2_index_path(tar_path_str) if idx_path is None else Path(idx_path)
    metadata_path = wds_v2_metadata_path(idx_path) if metadata_path is None else Path(metadata_path)
    _require_wds_v2_sidecar_paths(idx_path, metadata_path)
    try:
        raw_offsets = idx_path.read_bytes()
    except FileNotFoundError as ex:
        raise FileNotFoundError(f"Missing WDS v2 offset sidecar: {idx_path}") from ex
    try:
        raw_metadata = metadata_path.read_bytes()
    except FileNotFoundError as ex:
        raise FileNotFoundError(f"Missing WDS v2 metadata sidecar: {metadata_path}") from ex
    try:
        metadata = json.loads(raw_metadata)
    except (UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise ValueError(f"Malformed WDS v2 metadata JSON in {metadata_path}: {ex}") from ex
    if raw_metadata != _canonical_json_bytes(metadata):
        raise ValueError(f"WDS v2 metadata is not canonical sorted compact JSON: {metadata_path}")

    _validate_wds_v2_metadata_shape(metadata, metadata_path)
    if metadata["source"]["path"] != tar_path_str:
        raise ValueError(
            f"WDS v2 metadata source path mismatch: configured={tar_path_str!r}, "
            f"metadata={metadata['source']['path']!r}"
        )
    actual_sha256 = hashlib.sha256(raw_offsets).hexdigest()
    if metadata["offsets_sha256"] != actual_sha256:
        raise ValueError(
            f"WDS v2 offsets_sha256 mismatch for {idx_path}: "
            f"metadata={metadata['offsets_sha256']}, actual={actual_sha256}"
        )
    if len(raw_offsets) < 8 or len(raw_offsets) % 8:
        raise ValueError(
            f"Invalid WDS v2 offset sidecar size for {idx_path}: expected a positive multiple of 8, "
            f"got {len(raw_offsets)}"
        )
    offsets = np.frombuffer(raw_offsets, dtype="<u8").copy()
    if offsets.shape[0] != metadata["sample_count"] + 1:
        raise ValueError(
            f"WDS v2 sample count mismatch for {idx_path}: offsets={offsets.shape[0] - 1}, "
            f"metadata={metadata['sample_count']}"
        )
    if offsets.shape[0] > 1 and np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError(f"WDS v2 offsets must be strictly increasing in {idx_path}")

    current_source = indexed_source_metadata(tar_path_str)
    expected_source = metadata["source"]
    if _is_remote_path(tar_path_str) and expected_source["object_identity"] is None:
        raise ValueError(
            f"WDS v2 metadata for remote source {tar_path_str} is missing a stable "
            "object_identity; rebuild the sidecar against a backend exposing a "
            "version, ETag, or checksum."
        )
    if current_source["size_bytes"] != expected_source["size_bytes"]:
        raise ValueError(
            f"WDS v2 source size mismatch for {tar_path_str}: "
            f"metadata={expected_source['size_bytes']}, current={current_source['size_bytes']}"
        )
    if int(offsets[-1]) != current_source["size_bytes"]:
        raise ValueError(
            f"WDS v2 sentinel mismatch for {tar_path_str}: index={int(offsets[-1])}, "
            f"source={current_source['size_bytes']}"
        )
    if current_source["mtime_ns"] != expected_source["mtime_ns"]:
        raise ValueError(
            f"WDS v2 source mtime_ns mismatch for {tar_path_str}: "
            f"metadata={expected_source['mtime_ns']}, current={current_source['mtime_ns']}"
        )
    if expected_source["object_identity"] is not None and (
        current_source["object_identity"] != expected_source["object_identity"]
    ):
        raise ValueError(
            f"WDS v2 source object_identity mismatch for {tar_path_str}: "
            f"metadata={expected_source['object_identity']!r}, current={current_source['object_identity']!r}"
        )
    return offsets, metadata


def _is_remote_path(path) -> bool:
    """True if *path* is a URL/URI (s3://, ais://, http(s)://, gs://, …)."""
    return bool(_URL_RE.match(str(path)))


def _resolve_data_path(path: str) -> str:
    """Resolve an S3 identity to its configured local mirror when available."""
    read_path = str(path)
    if read_path.startswith("s3://") and _resolve_s3_to_local_mirror is not None:
        read_path = _resolve_s3_to_local_mirror(read_path)
    return read_path


def _open_data_path(path: str):
    """
    Return a seekable file-like for *path*, suitable for the indexed
    tar readers' ``self._fh`` slot.

    Local paths get a regular ``open(path, "rb")``. URL/URI paths return an
    :class:`lhotse.ais.AISRangeReader` (imported from lhotse to keep the
    seekable-AIS wrapper as a single source of truth shared with
    :func:`lhotse.indexing._open_for_indexed_read`). Other URL schemes
    (``http://``, ``gs://``, …) currently fall through to ``AISRangeReader``
    as well — the aistore SDK is the only seekable remote backend lhotse
    exposes today; if a future backend gains a seekable wrapper, dispatch
    here.
    """
    read_path = _resolve_data_path(path)
    if _is_remote_path(read_path):
        from lhotse.ais import AISRangeReader

        return AISRangeReader(read_path)
    return open(read_path, "rb")


def _load_index(data_path: str, idx_path: str | None = None):
    """
    Load an offset index for *data_path*, layering NeMo-specific validation
    on top of :func:`lhotse.indexing.read_index`.

    Returns ``(offsets, num_samples)`` where ``offsets`` always has
    ``num_samples + 1`` entries — the last one being the data file size
    (appended if absent in the on-disk index, for legacy ``.idx`` files
    written before the sentinel convention was added).

    Validates that all sample offsets fall within the data file.

    For remote ``data_path`` URIs (``s3://`` / ``ais://`` / ``http(s)://`` /
    ``gs://``) ``os.path.getsize`` is not callable; we trust the size
    sentinel that ``create_tar_index`` / ``create_jsonl_index`` recorded as
    the last offset in the on-disk index. The same indexes are emitted for
    local and remote sources, so the on-disk format is identical — only the
    file-size cross-check is skipped.
    """
    from lhotse.indexing import read_index

    if idx_path is None:
        idx_path = data_path + ".idx"
    offsets = read_index(idx_path)
    if _URL_RE.match(str(data_path)):
        if offsets.shape[0] < 1:
            raise ValueError(
                f"Index for remote source {data_path} is empty; expected at "
                f"least a size sentinel. Rebuild via build_indexes.py."
            )
        data_size = int(offsets[-1])
        num_samples = offsets.shape[0] - 1
    else:
        data_size = os.path.getsize(data_path)
        if offsets[-1] == data_size:
            num_samples = offsets.shape[0] - 1
        else:
            num_samples = offsets.shape[0]
            offsets = np.append(offsets, np.uint64(data_size))
    if num_samples > 0:
        max_offset = int(offsets[:num_samples].max())
        if max_offset >= data_size:
            raise ValueError(
                f"Index for {data_path} contains offset {max_offset} "
                f"beyond file size {data_size}. "
                f"The .idx file may have been created by an incompatible tool "
                f"or for a different file."
            )
    return offsets, num_samples


def _resolve_idx(idx: int, length: int) -> int:
    if idx < 0:
        idx += length
    if idx < 0 or idx >= length:
        raise IndexError("Index out of bounds")
    return idx


class TarSample(NamedTuple):
    """A single sample extracted from a WebDataset tar archive."""

    json_data: dict
    audio_bytes: bytes
    audio_name: str


def _split_json_audio_pair(name_a, bytes_a, name_b, bytes_b) -> TarSample:
    """Classify two tar members into a ``TarSample`` regardless of order."""
    is_json_a = name_a.endswith(".json")
    is_json_b = name_b.endswith(".json")
    if is_json_a == is_json_b:
        raise ValueError(f"Expected exactly one .json member in tar sample pair, got: {name_a}, {name_b}")
    if is_json_a:
        json_name, json_bytes = name_a, bytes_a
        audio_name, audio_bytes = name_b, bytes_b
    else:
        json_name, json_bytes = name_b, bytes_b
        audio_name, audio_bytes = name_a, bytes_a
    json_key = PurePosixPath(json_name).with_suffix("").as_posix()
    audio_key = PurePosixPath(audio_name).with_suffix("").as_posix()
    if json_key != audio_key:
        raise ValueError("WebDataset tar pair has different sample keys: " f"json={json_name!r} audio={audio_name!r}.")
    return TarSample(json.loads(json_bytes), audio_bytes, audio_name)


class IndexedTarSampleReader:
    """
    Random access to WebDataset tar samples (``N.json`` + ``N.<audio>``) via an index file.
    Index format is the same little-endian ``uint64`` offsets as
    :class:`lhotse.indexing.IndexedJsonlReader`, optionally followed by a
    sentinel equal to the tar file size.
    """

    def __init__(self, tar_path: str | Path, idx_path: str | Path | None = None):
        self.data_path = str(tar_path)
        self.offsets, self._len = _load_index(self.data_path, str(idx_path) if idx_path else None)
        self._data_size = int(self.offsets[-1])
        self._validate_index()

    def _validate_index(self):
        """Tar-specific validation: check that indexed offsets point to valid tar headers."""
        if self._len == 0:
            return
        # Validate first offset is a valid tar header.
        self._check_offset_is_tar_header(int(self.offsets[0]), label="first")
        # Strip trailing sentinels: some tools store the offset of the
        # end-of-archive zero-block marker as a sentinel instead of the
        # file size (which _load_index already handles).
        while self._len > 0:
            last = int(self.offsets[self._len - 1])
            with _open_data_path(self.data_path) as f:
                f.seek(last)
                buf = f.read(_TAR_BLOCK_SIZE)
            if len(buf) < _TAR_BLOCK_SIZE or buf == _TAR_ZERO_BLOCK:
                self._len -= 1
            else:
                break

    def _check_offset_is_tar_header(self, offset: int, label: str = ""):
        with _open_data_path(self.data_path) as f:
            f.seek(offset)
            buf = f.read(_TAR_BLOCK_SIZE)
        if len(buf) < _TAR_BLOCK_SIZE:
            raise ValueError(
                f"Tar index for {self.data_path}: {label} offset {offset} "
                f"is too close to EOF (file size {self._data_size})."
            )
        if buf == _TAR_ZERO_BLOCK:
            raise ValueError(
                f"Tar index for {self.data_path}: {label} offset {offset} "
                f"points to a zero block (end-of-archive marker), not a tar header. "
                f"The .idx file may have been created by an incompatible tool "
                f"or for a different file."
            )
        try:
            tarfile.TarInfo.frombuf(buf, tarfile.ENCODING, "surrogateescape")
        except tarfile.TarError as e:
            raise ValueError(
                f"Tar index for {self.data_path}: {label} offset {offset} "
                f"does not point to a valid tar header: {e}. "
                f"The .idx file may have been created by an incompatible tool "
                f"(e.g. has a binary header or stores per-member offsets) "
                f"or for a different file."
            ) from e

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        idx = _resolve_idx(idx, self._len)
        offset = int(self.offsets[idx])
        with _open_data_path(self.data_path) as f:
            f.seek(offset)
            try:
                name_a, bytes_a = _read_tar_member(f)
            except (EOFError, tarfile.TarError) as e:
                raise type(e)(
                    f"{e} — reading first member of sample {idx}/{self._len} "
                    f"at offset {offset} in {self.data_path} "
                    f"(file size {self._data_size})"
                ) from e
            try:
                name_b, bytes_b = _read_tar_member(f)
            except (EOFError, tarfile.TarError) as e:
                raise type(e)(
                    f"{e} — reading second member of sample {idx}/{self._len} "
                    f"(first member was '{name_a}', {len(bytes_a)} bytes) "
                    f"at offset {offset} in {self.data_path} "
                    f"(file size {self._data_size})"
                ) from e
        return _split_json_audio_pair(name_a, bytes_a, name_b, bytes_b)


class IndexedTarSampleBundleReader:
    """Random access to variable-member WebDataset samples through a validated v2 index."""

    def __init__(
        self,
        tar_path: str | Path,
        idx_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ):
        self.data_path = os.fspath(tar_path)
        self.offsets, self.metadata = validate_wds_v2_tar_index(
            self.data_path,
            idx_path=idx_path,
            metadata_path=metadata_path,
        )
        self._len = int(self.metadata["sample_count"])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> TarSampleBundle:
        idx = _resolve_idx(idx, self._len)
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        raw = read_exact_range(self.data_path, start, end)
        try:
            return _decode_wds_v2_sample_range(raw, self.data_path, idx, start, end)
        except (
            EOFError,
            tarfile.TarError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
        ) as ex:
            raise ValueError(
                f"Invalid WDS v2 sample {idx}/{self._len} in {self.data_path} " f"at byte range [{start}, {end}): {ex}"
            ) from ex


class PackedTarSampleBundleReader:
    """Random access to variable-member WebDataset samples through an idxpack collection."""

    def __init__(self, collection, max_open_files: int = 32):
        if collection.kind != "wds_tar_v2":
            raise ValueError(f"Expected a wds_tar_v2 collection, got {collection.kind!r}")
        if not collection.offsets_required:
            raise ValueError("A wds_tar_v2 collection must contain sample offsets")
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.collection = collection
        self.max_open_files = max_open_files

    def __len__(self) -> int:
        return len(self.collection)

    def __getitem__(self, idx: int) -> TarSampleBundle:
        sample, _ = self.read_with_location(idx)
        return sample

    def read_with_location(self, idx: int):
        """Read one sample together with its resolved source byte range."""
        normalized_idx = _resolve_idx(idx, len(self))
        location = self.collection.locate(normalized_idx)
        return self._read_location(location, normalized_idx), location

    def read_shard(self, shard_index: int, local_index: int) -> TarSampleBundle:
        """Read one sample by its source shard and shard-local position."""
        location = self.collection.locate_in_shard(shard_index, local_index)
        return self._read_location(location, (shard_index, local_index))

    def path_for_shard(self, shard_index: int) -> str:
        """Return the concrete tar path for one logical source shard."""
        return self.collection.path_for_shard(shard_index)

    def shard_length(self, shard_index: int) -> int:
        """Return the number of indexed samples in one source shard."""
        return self.collection.shard_length(shard_index)

    def _read_location(self, location, idx) -> TarSampleBundle:
        if _is_remote_path(location.path):
            raw = read_exact_range(location.path, location.start, location.end)
        else:
            from lhotse.packed_lazy import read_packed_range

            raw = read_packed_range(
                self.collection.pack,
                location.path,
                location.start,
                location.end,
                max_open_files=self.max_open_files,
            )
        try:
            return _decode_wds_v2_sample_range(raw, location.path, idx, location.start, location.end)
        except (
            EOFError,
            tarfile.TarError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
        ) as ex:
            raise ValueError(
                f"Invalid packed WDS v2 sample {idx}/{len(self)} in {location.path} "
                f"at byte range [{location.start}, {location.end}): {ex}"
            ) from ex


class IndexedTarMemberReader:
    """
    Random access to a NeMo-style tar archive that stores **one regular member
    per sample** (e.g. ``<cut_id>.flac`` per line of an external NeMo manifest).

    Uses the same ``.idx`` format as :class:`lhotse.indexing.IndexedJsonlReader`
    and :class:`IndexedTarSampleReader`: little-endian uint64 byte offsets, with
    a sentinel equal to the tar file size at the end. Each entry points at
    one tar header, and the corresponding payload starts ``512`` bytes later.

    Two access patterns:

    * Positional: ``reader[idx]`` returns ``(member_name, payload_bytes)``.
    * Name-keyed: ``reader.get(name)`` returns just the payload bytes. The
      name → position map is built lazily on first use by walking the tar
      headers (no payload reads), then cached for subsequent calls.
    """

    def __init__(
        self,
        tar_path: str | Path,
        idx_path: str | Path | None = None,
        auto_create_index: bool = True,
    ):
        self.data_path = str(tar_path)
        resolved_idx = str(idx_path) if idx_path else self.data_path + ".idx"
        if auto_create_index and not os.path.exists(resolved_idx):
            create_tar_index(self.data_path, resolved_idx)
        self.offsets, self._len = _load_index(self.data_path, resolved_idx)
        self._fh = None
        self._name_to_idx: dict[str, int] | None = None

    def _ensure_open(self):
        if self._fh is None:
            self._fh = _open_data_path(self.data_path)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_fh"] = None  # file handles are not picklable
        return s

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[str, bytes]:
        idx = _resolve_idx(idx, self._len)
        offset = int(self.offsets[idx])
        self._ensure_open()
        self._fh.seek(offset)
        try:
            name, data = _read_tar_member(self._fh)
        except (EOFError, tarfile.TarError) as e:
            raise type(e)(f"{e} — reading sample {idx}/{self._len} at offset {offset} " f"in {self.data_path}") from e
        return name, data

    def _build_name_index(self) -> dict[str, int]:
        """Walk the tar headers once to build a name → sample-index map.

        Reads only the 512-byte tar headers (no payloads), so this is
        relatively cheap even on remote storage. Done lazily on first
        :meth:`get` call.

        ``tar.add`` writes a PAX extended header (``@PaxHeader``) before any
        member with a long path or extended attributes. We skip those and
        record the *regular* file's name at each indexed offset.
        """
        name_to_idx: dict[str, int] = {}
        self._ensure_open()
        for i in range(self._len):
            self._fh.seek(int(self.offsets[i]))
            while True:
                header = self._fh.read(_TAR_BLOCK_SIZE)
                if len(header) < _TAR_BLOCK_SIZE or header == _TAR_ZERO_BLOCK:
                    break
                info = tarfile.TarInfo.frombuf(header, tarfile.ENCODING, "surrogateescape")
                if info.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    name_to_idx[info.name] = i
                    break
                # Skip non-regular member (PAX/GNU long-name) data + padding.
                size_blocks = -(-info.size // _TAR_BLOCK_SIZE) * _TAR_BLOCK_SIZE
                self._fh.seek(size_blocks, 1)
        return name_to_idx

    def get(self, name: str) -> bytes:
        """Return the payload bytes of the tar member named ``name``."""
        if self._name_to_idx is None:
            self._name_to_idx = self._build_name_index()
        try:
            idx = self._name_to_idx[name]
        except KeyError as e:
            raise KeyError(
                f"Tar {self.data_path} has no member named '{name}'. "
                f"The .idx may be stale or the manifest is referencing a "
                f"different tar."
            ) from e
        _, data = self[idx]
        return data

    def __contains__(self, name: str) -> bool:
        if self._name_to_idx is None:
            self._name_to_idx = self._build_name_index()
        return name in self._name_to_idx


def _read_tar_member(f):
    """Read the next regular-file tar member, skipping non-regular entries
    (PAX headers, GNU long-name headers, directory entries, etc.).

    We read tar headers manually instead of using ``tarfile.open()`` because
    the stdlib ``tarfile`` module does not support random-access seeks into the
    middle of an archive — it always reads sequentially from the start.
    By parsing individual headers via ``TarInfo.frombuf`` we can seek to an
    arbitrary byte offset and read just the members we need in O(1).
    """
    pax_headers: dict[str, str] = {}
    long_name: str | None = None
    while True:
        header_buf = f.read(_TAR_BLOCK_SIZE)
        if len(header_buf) < _TAR_BLOCK_SIZE or header_buf == _TAR_ZERO_BLOCK:
            raise EOFError("End of tar archive or unexpected EOF")
        info = tarfile.TarInfo.frombuf(header_buf, tarfile.ENCODING, "surrogateescape")
        data = f.read(info.size)
        if len(data) < info.size:
            raise EOFError("Unexpected end of tar file while reading data")
        remainder = info.size % _TAR_BLOCK_SIZE
        if remainder:
            f.seek(_TAR_BLOCK_SIZE - remainder, 1)
        if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
            pax_headers.update(_parse_pax_headers(data))
            continue
        if info.type == tarfile.GNUTYPE_LONGNAME:
            long_name = data.rstrip(b"\0\n").decode(tarfile.ENCODING, "surrogateescape")
            continue
        if info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
            continue
        name = pax_headers.get("path") or long_name or info.name
        return name, data


def _parse_pax_headers(data: bytes) -> dict[str, str]:
    """Parse POSIX.1-2001 PAX length-prefixed key/value records."""
    headers = {}
    position = 0
    while position < len(data):
        space = data.find(b" ", position)
        if space < 0:
            raise tarfile.ReadError("Malformed PAX header: missing record length separator")
        try:
            length = int(data[position:space])
        except ValueError as ex:
            raise tarfile.ReadError("Malformed PAX header record length") from ex
        end = position + length
        if length <= 0 or end > len(data):
            raise tarfile.ReadError("Malformed PAX header record bounds")
        record = data[space + 1 : end]
        if record.endswith(b"\n"):
            record = record[:-1]
        key, separator, value = record.partition(b"=")
        if not separator:
            raise tarfile.ReadError("Malformed PAX header key/value record")
        headers[key.decode("utf-8", "surrogateescape")] = value.decode("utf-8", "surrogateescape")
        position = end
    return headers


class PackedTarMemberReader:
    """Random access to native tar members through an idxpack collection."""

    def __init__(self, collection, max_open_files: int = 32):
        if collection.kind != "nemo_tar":
            raise ValueError(f"Expected a nemo_tar collection, got {collection.kind!r}")
        self.collection = collection
        self.max_open_files = max_open_files
        self._shard_name_indexes: OrderedDict[int, dict[str, int]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.collection)

    def __getitem__(self, idx: int) -> tuple[str, bytes]:
        item, _ = self.read_with_location(idx)
        return item

    def read_with_location(self, idx: int):
        """Read one member together with its resolved source byte range."""
        location = self.collection.locate(_resolve_idx(idx, len(self)))
        return self._read_location(location, idx), location

    def read_shard(self, shard_index: int, local_index: int) -> tuple[str, bytes]:
        """Read one member by its paired manifest shard/local position."""
        item, _ = self.read_shard_with_location(shard_index, local_index)
        return item

    def read_shard_with_location(self, shard_index: int, local_index: int):
        """Read one member and return the exact packed source byte range."""
        location = self.collection.locate_in_shard(shard_index, local_index)
        return self._read_location(location, (shard_index, local_index)), location

    def _member_name(self, location) -> str:
        """Read only tar headers at one packed offset and return the regular member name."""
        from lhotse.packed_lazy import read_packed_range

        def read_range(start: int, end: int) -> bytes:
            if _is_remote_path(location.path):
                return read_exact_range(location.path, start, end)
            return read_packed_range(
                self.collection.pack,
                location.path,
                start,
                end,
                max_open_files=self.max_open_files,
            )

        position = location.start
        pax_headers: dict[str, str] = {}
        long_name: str | None = None
        while position + _TAR_BLOCK_SIZE <= location.end:
            header = read_range(position, position + _TAR_BLOCK_SIZE)
            if header == _TAR_ZERO_BLOCK:
                break
            try:
                info = tarfile.TarInfo.frombuf(header, tarfile.ENCODING, "surrogateescape")
            except tarfile.TarError as ex:
                raise type(ex)(f"{ex} — reading packed tar header at {position} in {location.path}") from ex
            data_position = position + _TAR_BLOCK_SIZE
            if info.type in (
                tarfile.XHDTYPE,
                tarfile.XGLTYPE,
                tarfile.GNUTYPE_LONGNAME,
            ):
                data = read_range(data_position, data_position + info.size)
                if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
                    pax_headers.update(_parse_pax_headers(data))
                else:
                    long_name = data.rstrip(b"\0\n").decode(tarfile.ENCODING, "surrogateescape")
            elif info.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
                return pax_headers.get("path") or long_name or info.name
            position = data_position + (-(-info.size // _TAR_BLOCK_SIZE) * _TAR_BLOCK_SIZE)
        raise EOFError(
            f"No regular tar member in packed range [{location.start}, {location.end}) " f"in {location.path}"
        )

    def _name_index_for_shard(self, shard_index: int) -> dict[str, int]:
        try:
            index = self._shard_name_indexes.pop(shard_index)
        except KeyError:
            index = {}
            for local_index in range(self.collection.shard_length(shard_index)):
                location = self.collection.locate_in_shard(shard_index, local_index)
                name = self._member_name(location)
                if name in index:
                    raise ValueError(
                        f"Duplicate tar member name {name!r} in {location.path}; "
                        "name-keyed packed access is ambiguous"
                    )
                index[name] = local_index
        self._shard_name_indexes[shard_index] = index
        while len(self._shard_name_indexes) > self.max_open_files:
            self._shard_name_indexes.popitem(last=False)
        return index

    def get_shard(self, shard_index: int, name: str) -> tuple[str, bytes]:
        """Read a member by name from one shard, supporting filtered manifests."""
        index = self._name_index_for_shard(shard_index)
        try:
            local_index = index[name]
        except KeyError as ex:
            path = self.collection.path_for_shard(shard_index)
            raise KeyError(f"Tar {path} has no member named {name!r}.") from ex
        return self.read_shard(shard_index, local_index)

    def _read_location(self, location, idx) -> tuple[str, bytes]:
        if _is_remote_path(location.path):
            raw = read_exact_range(location.path, location.start, location.end)
        else:
            from lhotse.packed_lazy import read_packed_range

            raw = read_packed_range(
                self.collection.pack,
                location.path,
                location.start,
                location.end,
                max_open_files=self.max_open_files,
            )
        try:
            return _read_tar_member(BytesIO(raw))
        except (EOFError, tarfile.TarError) as ex:
            raise type(ex)(
                f"{ex} — reading packed tar sample {idx}/{len(self)} "
                f"at [{location.start}, {location.end}) in {location.path}"
            ) from ex


class _CountingReader:
    """
    Minimal file-like wrapper that delegates everything to an inner stream
    while counting the total number of bytes read. Used by
    :func:`create_tar_index` to compute a tar file's size without calling
    ``tell()`` — necessary because non-seekable remote streams (AIStore's
    ``ObjectFileReader``, smart_open's S3 reader without seek support, …)
    raise ``io.UnsupportedOperation`` on ``tell()`` even when sequential
    reads succeed.
    """

    def __init__(self, fileobj):
        self._f = fileobj
        self.bytes_read = 0

    def read(self, n=-1):
        data = self._f.read(n)
        self.bytes_read += len(data)
        return data

    def readable(self):
        return True

    def seekable(self):
        # tarfile's ``r|`` (stream) mode falls back to read+discard when
        # the fileobj is not seekable, which is exactly what we want.
        return False


_TAR_INDEX_PROBE_REGULAR_MEMBERS = 32
_TAR_INDEX_STREAMING_MIN_REGULAR_MEMBERS = 8
_TAR_INDEX_STREAMING_MAX_AVERAGE_SPAN_BYTES = 1024 * 1024


def _select_local_tar_index_mode(fileobj) -> str:
    """
    Select the cheaper stdlib tar traversal for a seekable local source.

    Seeking wins when member payloads are large because it avoids reading
    them. For dense archives, however, ``tarfile``'s seekable mode performs
    one tiny header read after each payload seek; on network filesystems those
    random reads can cost much more than a sequential pass. Probe at most the
    first 32 regular members and use their physical archive span as a cheap
    density estimate. The caller always starts the selected full scan from
    the original stream position.
    """
    start = fileobj.tell()
    first_offset = None
    last_end = None
    regular_members = 0
    try:
        with tarfile.open(fileobj=fileobj, mode="r:") as archive:
            for member in archive:
                if not member.isreg():
                    continue
                if first_offset is None:
                    first_offset = member.offset
                padded_size = (member.size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE * tarfile.BLOCKSIZE
                last_end = member.offset_data + padded_size
                regular_members += 1
                if regular_members >= _TAR_INDEX_PROBE_REGULAR_MEMBERS:
                    break
    finally:
        fileobj.seek(start)

    if regular_members < _TAR_INDEX_STREAMING_MIN_REGULAR_MEMBERS:
        return "r:"
    average_span = (last_end - first_offset) / regular_members
    if average_span <= _TAR_INDEX_STREAMING_MAX_AVERAGE_SPAN_BYTES:
        return "r|"
    return "r:"


def create_tar_index(tar_path, idx_path):
    """
    Creates a raw binary index file for a WebDataset tar archive.
    Stores the byte offset of the first member of each sample (grouped by basename),
    followed by a sentinel equal to the tar file size. On-disk format matches
    :func:`lhotse.indexing.create_jsonl_index` and the other readers in this
    module: a sequence of little-endian uint64 byte offsets.

    Reads ``tar_path`` through the indexed-source opener for local files and
    ``s3://`` / ``ais://`` / ``http(s)://`` URIs. When
    ``LHOTSE_S3_LOCAL_MIRROR_ROOTS`` maps an S3 identity to a local file, the
    mirror is read without changing the logical source path; otherwise the
    remote range reader is used. Local and mirrored files adaptively choose
    streaming iteration (``r|``) for dense/small-member archives and seekable
    iteration (``r:``) for sparse/large-payload archives. Unresolved remote
    URLs retain streaming iteration, and the sentinel records the total bytes
    read through ``_CountingReader``.

    Written atomically: data is staged in a per-process temp file next to
    ``idx_path`` and then ``os.replace()``-d into place, so concurrent writers
    can't observe a half-written ``.idx``.
    """

    def scan_offsets(tar):
        offsets = []
        prev_stem = None
        for member in tar:
            if not member.isreg():
                continue
            stem = Path(member.name).stem
            if stem != prev_stem:
                offsets.append(member.offset)
                prev_stem = stem
        return offsets

    def scan_stream(fileobj):
        counter = _CountingReader(fileobj)
        with tarfile.open(fileobj=counter, mode="r|") as tar:
            offsets = scan_offsets(tar)
        # tarfile stops at the end-of-archive marker; consume trailing
        # record padding so the sentinel matches the physical object size.
        while counter.read(1024 * 1024):
            pass
        return offsets, counter.bytes_read

    read_path = _resolve_data_path(str(tar_path))
    scheme = urlsplit(read_path).scheme.lower() if _is_remote_path(read_path) else ""
    if scheme and scheme not in _SUPPORTED_REMOTE_RANGE_SCHEMES:
        from lhotse.serialization import open_best

        with open_best(read_path, "rb") as f:
            offsets, file_size = scan_stream(f)
    else:
        with _open_data_path(read_path) as f:
            if _is_remote_path(read_path):
                offsets, file_size = scan_stream(f)
            else:
                mode = _select_local_tar_index_mode(f)
                with tarfile.open(fileobj=f, mode=mode) as tar:
                    offsets = scan_offsets(tar)
                # Seek to the physical end rather than using tarfile's logical
                # end-of-archive position, preserving trailing record padding in
                # the sentinel without reading it.
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
    tmp_path = f"{idx_path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f_out:
        buf = bytearray()
        for off in offsets:
            buf.extend(struct.pack("<Q", off))
        buf.extend(struct.pack("<Q", file_size))
        f_out.write(buf)
    os.replace(tmp_path, idx_path)


def _scan_wds_v2_tar(tar_path: str) -> tuple[list[int], int, int]:
    offsets: list[int] = []
    seen_member_names: set[str] = set()
    closed_keys: set[str] = set()
    current_key: str | None = None
    json_count = 0
    payload_count = 0
    regular_member_count = 0

    def finish_sample() -> None:
        if current_key is None:
            return
        if json_count != 1:
            raise ValueError(
                f"WebDataset sample {current_key!r} in {tar_path} must contain exactly one .json member; "
                f"found {json_count}"
            )
        if payload_count == 0:
            raise ValueError(
                f"WebDataset sample {current_key!r} in {tar_path} must contain at least one non-JSON payload"
            )

    with _open_data_path(tar_path) as source:
        counter = _CountingReader(source)
        with tarfile.open(fileobj=counter, mode="r|") as archive:
            for member in archive:
                if not member.isreg():
                    continue
                key = wds_sample_key(member.name)
                if member.name in seen_member_names:
                    raise ValueError(f"WebDataset tar {tar_path} has duplicate regular member name {member.name!r}")
                seen_member_names.add(member.name)
                if key != current_key:
                    finish_sample()
                    if key in closed_keys:
                        raise ValueError(f"WebDataset tar {tar_path} has non-contiguous reuse of sample key {key!r}")
                    if current_key is not None:
                        closed_keys.add(current_key)
                    current_key = key
                    json_count = 0
                    payload_count = 0
                    offsets.append(member.offset)

                regular_member_count += 1
                if member.name.endswith(".json"):
                    json_count += 1
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"Could not read JSON member {member.name!r} in {tar_path}")
                    raw_json = extracted.read()
                    try:
                        decoded = json.loads(raw_json)
                    except (UnicodeDecodeError, json.JSONDecodeError) as ex:
                        raise ValueError(
                            f"WebDataset sample {key!r} has malformed JSON member {member.name!r}: {ex}"
                        ) from ex
                    if not isinstance(decoded, dict):
                        raise ValueError(
                            f"WebDataset sample {key!r} JSON member {member.name!r} must decode to a JSON object"
                        )
                else:
                    payload_count += 1
            finish_sample()
        while counter.read(1024 * 1024):
            pass
        source_size = counter.bytes_read
    return offsets, regular_member_count, source_size


def _decode_wds_v2_sample_range(raw: bytes, source_path: str, idx: int, start: int, end: int) -> TarSampleBundle:
    members = list(_iter_regular_tar_members(raw))
    if not members:
        raise ValueError("sample range contains no regular tar members")
    names: set[str] = set()
    sample_key: str | None = None
    json_items: list[tuple[str, bytes]] = []
    payloads: list[TarSampleMember] = []
    for name, data in members:
        if name in names:
            raise ValueError(f"duplicate regular member name {name!r}")
        names.add(name)
        key = wds_sample_key(name)
        if sample_key is None:
            sample_key = key
        elif key != sample_key:
            raise ValueError(
                f"sample range [{start}, {end}) mixes indexed key {sample_key!r} with member {name!r} key {key!r}"
            )
        if name.endswith(".json"):
            json_items.append((name, data))
        else:
            payloads.append(TarSampleMember(name, data))
    if len(json_items) != 1:
        raise ValueError(f"sample key {sample_key!r} must contain exactly one .json member; found {len(json_items)}")
    if not payloads:
        raise ValueError(f"sample key {sample_key!r} must contain at least one non-JSON payload")
    json_name, raw_json = json_items[0]
    try:
        json_data = json.loads(raw_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise ValueError(f"malformed JSON member {json_name!r}: {ex}") from ex
    if not isinstance(json_data, dict):
        raise ValueError(f"JSON member {json_name!r} must decode to a JSON object")  # noqa: TRY004
    assert sample_key is not None
    return TarSampleBundle(
        sample_key=sample_key,
        json_data=json_data,
        audio_members=tuple(payloads),
        source_path=source_path,
        source_range_bytes=end - start,
    )


def _iter_regular_tar_members(raw: bytes):
    position = 0
    pax_headers: dict[str, str] = {}
    long_name: str | None = None
    while position < len(raw):
        remaining = raw[position:]
        if len(remaining) < _TAR_BLOCK_SIZE:
            if any(remaining):
                raise EOFError(f"Truncated tar header at relative byte offset {position}")
            return
        header = remaining[:_TAR_BLOCK_SIZE]
        if header == _TAR_ZERO_BLOCK:
            if any(remaining):
                raise tarfile.ReadError(f"Non-zero data follows tar end marker at relative byte offset {position}")
            return
        info = tarfile.TarInfo.frombuf(header, tarfile.ENCODING, "surrogateescape")
        data_start = position + _TAR_BLOCK_SIZE
        data_end = data_start + info.size
        padded_end = data_start + (-(-info.size // _TAR_BLOCK_SIZE) * _TAR_BLOCK_SIZE)
        if data_end > len(raw) or padded_end > len(raw):
            raise EOFError(
                f"Truncated tar member {info.name!r} at relative byte offset {position}: "
                f"needs {padded_end} bytes, range has {len(raw)}"
            )
        data = raw[data_start:data_end]
        position = padded_end
        if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
            pax_headers.update(_parse_pax_headers(data))
            continue
        if info.type == tarfile.GNUTYPE_LONGNAME:
            long_name = data.rstrip(b"\0\n").decode(tarfile.ENCODING, "surrogateescape")
            continue
        if info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
            continue
        name = pax_headers.get("path") or long_name or info.name
        yield name, data
        pax_headers = {}
        long_name = None


def _local_source_identity(path: str):
    if _is_remote_path(path):
        return None
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def indexed_source_metadata(path: str | Path) -> dict:
    """Return source size and a mutation-sensitive local/remote identity.

    Remote ``ais://`` and ``s3://`` sources fail closed unless AIStore exposes
    a stable version, ETag, checksum, or an equivalent identity supplied by
    the range-reader backend. Size alone is not an identity because an object
    may be replaced in place with another object of the same size.
    """
    path = os.fspath(path)
    if not _is_remote_path(path):
        stat = Path(path).stat()
        return {
            "path": path,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "object_identity": None,
        }
    scheme = urlsplit(path).scheme.lower()
    if scheme not in _SUPPORTED_REMOTE_RANGE_SCHEMES:
        raise ValueError(
            f"Unsupported remote range-read scheme {scheme!r} for {path!r}; "
            f"supported schemes are {sorted(_SUPPORTED_REMOTE_RANGE_SCHEMES)}"
        )
    from lhotse.ais import AISRangeReader

    with AISRangeReader(path) as source:
        size = int(source.size)
        object_identity = _remote_object_identity(source)
    if not object_identity:
        raise ValueError(
            f"Remote indexed source {path!r} has no stable object identity; "
            "size-only validation cannot detect same-size replacement. Require "
            "an AIStore/S3 version, ETag, checksum, or a sealed catalog backend "
            "that exposes one of those identities."
        )
    return {
        "path": path,
        "size_bytes": size,
        "mtime_ns": 0,
        "object_identity": object_identity,
    }


def _wds_v2_source_metadata(path: str) -> dict:
    """Compatibility alias for callers/tests written against the v2 helper."""
    return indexed_source_metadata(path)


def _remote_object_identity(source) -> str | None:
    direct = getattr(source, "object_identity", None)
    if direct:
        return str(direct)

    obj = getattr(source, "_obj", None)
    if obj is None:
        return None
    attributes = None
    head_v2 = getattr(obj, "head_v2", None)
    if head_v2 is not None:
        try:
            attributes = head_v2("size,version,checksum,etag")
        except Exception:  # noqa: BLE001 - V2 metadata is optional; cached V1 properties remain authoritative.
            attributes = None
    if attributes is None:
        attributes = getattr(obj, "props_cached", None)

    version = getattr(attributes, "obj_version", "") if attributes is not None else ""
    if version:
        return f"version:{version}"
    etag = getattr(attributes, "etag", "") if attributes is not None else ""
    if etag:
        return f"etag:{etag}"
    checksum = getattr(attributes, "checksum_value", "") if attributes is not None else ""
    if checksum:
        checksum_type = getattr(attributes, "checksum_type", "") or "backend"
        return f"checksum:{checksum_type}:{checksum}"
    return None


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _validate_wds_v2_metadata_shape(metadata: dict, metadata_path: Path) -> None:
    expected_keys = {
        "format",
        "version",
        "sample_key_algorithm",
        "source",
        "regular_member_count",
        "sample_count",
        "offsets_sha256",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise ValueError(
            f"Invalid WDS v2 metadata keys in {metadata_path}: expected {sorted(expected_keys)}, "
            f"got {sorted(metadata) if isinstance(metadata, dict) else type(metadata).__name__}"
        )
    if metadata["format"] != WDS_V2_INDEX_FORMAT or metadata["version"] != WDS_V2_INDEX_VERSION:
        raise ValueError(
            f"Unsupported WDS v2 metadata format/version in {metadata_path}: "
            f"format={metadata['format']!r}, version={metadata['version']!r}"
        )
    if metadata["sample_key_algorithm"] != WDS_V2_SAMPLE_KEY_ALGORITHM:
        raise ValueError(
            f"Unsupported WDS v2 sample-key algorithm in {metadata_path}: " f"{metadata['sample_key_algorithm']!r}"
        )
    source = metadata["source"]
    source_keys = {"path", "size_bytes", "mtime_ns", "object_identity"}
    if not isinstance(source, dict) or set(source) != source_keys:
        raise ValueError(f"Invalid WDS v2 source metadata in {metadata_path}: {source!r}")
    if not isinstance(source["path"], str) or not source["path"]:
        raise ValueError(f"Invalid WDS v2 source path in {metadata_path}: {source['path']!r}")
    for field in ("size_bytes", "mtime_ns"):
        if type(source[field]) is not int or source[field] < 0:
            raise ValueError(f"Invalid WDS v2 source {field} in {metadata_path}: {source[field]!r}")
    if source["object_identity"] is not None and (
        not isinstance(source["object_identity"], str) or not source["object_identity"]
    ):
        raise ValueError(f"Invalid WDS v2 source object_identity in {metadata_path}: {source['object_identity']!r}")
    for field in ("regular_member_count", "sample_count"):
        if type(metadata[field]) is not int or metadata[field] < 0:
            raise ValueError(f"Invalid WDS v2 {field} in {metadata_path}: {metadata[field]!r}")
    if (
        not isinstance(metadata["offsets_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", metadata["offsets_sha256"]) is None
    ):
        raise ValueError(f"Invalid WDS v2 offsets_sha256 in {metadata_path}: {metadata['offsets_sha256']!r}")


def _require_wds_v2_sidecar_paths(idx_path: Path, metadata_path: Path) -> None:
    if not str(idx_path).endswith(_WDS_V2_INDEX_SUFFIX):
        raise ValueError(f"WDS v2 index path must end with {_WDS_V2_INDEX_SUFFIX!r}: {idx_path}")
    expected_metadata = wds_v2_metadata_path(idx_path)
    if metadata_path != expected_metadata:
        raise ValueError(
            f"WDS v2 metadata path must be the canonical companion {expected_metadata}, got {metadata_path}"
        )


def _write_fsynced(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
