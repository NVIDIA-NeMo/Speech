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
import json
import os
import re
import struct
import tarfile
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np


# Tar block size + the all-zeros block that marks end-of-archive in tar.
_TAR_BLOCK_SIZE = 512
_TAR_ZERO_BLOCK = b'\0' * _TAR_BLOCK_SIZE

# Recognized URL schemes whose authority ("host" component) is part of the
# logical path (e.g. the bucket name). Stripping just the scheme keeps the
# bucket+key in the relative path used to mirror under indexes_root.
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _is_remote_path(path) -> bool:
    """True if *path* is a URL/URI (s3://, ais://, http(s)://, gs://, …)."""
    return bool(_URL_RE.match(str(path)))


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
    if _is_remote_path(path):
        from lhotse.ais import AISRangeReader

        return AISRangeReader(str(path))
    return open(path, "rb")


def _load_index(data_path: str, idx_path: Optional[str] = None):
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
        idx_path = data_path + '.idx'
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
    if name_a.endswith('.json'):
        return TarSample(json.loads(bytes_a), bytes_b, name_b)
    if name_b.endswith('.json'):
        return TarSample(json.loads(bytes_b), bytes_a, name_a)
    raise ValueError(f"Expected one .json member in tar sample pair, got: {name_a}, {name_b}")


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
        if info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
            continue
        return info.name, data


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


def create_tar_index(tar_path, idx_path):
    """
    Creates a raw binary index file for a WebDataset tar archive.
    Stores the byte offset of the first member of each sample (grouped by basename),
    followed by a sentinel equal to the tar file size. On-disk format matches
    :func:`lhotse.indexing.create_jsonl_index` and the other readers in this
    module: a sequence of little-endian uint64 byte offsets.

    Reads ``tar_path`` via ``lhotse.serialization.open_best`` so the function
    works for local files as well as ``s3://`` / ``ais://`` / ``http(s)://``
    URIs. The tar is opened in streaming mode (``r|``) — remote backends are
    not seekable — and the sentinel records the total bytes read through a
    ``_CountingReader`` wrapper rather than ``os.path.getsize`` /
    ``f.tell()``, both of which fail on non-seekable URI streams.

    Written atomically: data is staged in a per-process temp file next to
    ``idx_path`` and then ``os.replace()``-d into place, so concurrent writers
    can't observe a half-written ``.idx``.
    """
    from lhotse.serialization import open_best

    offsets = []
    prev_stem = None
    with open_best(tar_path, "rb") as f:
        counter = _CountingReader(f)
        with tarfile.open(fileobj=counter, mode='r|') as tar:
            for member in tar:
                if not member.isreg():
                    continue
                stem = Path(member.name).stem
                if stem != prev_stem:
                    offsets.append(member.offset)
                    prev_stem = stem
        file_size = counter.bytes_read
    tmp_path = f"{idx_path}.tmp.{os.getpid()}"
    with open(tmp_path, 'wb') as f_out:
        buf = bytearray()
        for off in offsets:
            buf.extend(struct.pack('<Q', off))
        buf.extend(struct.pack('<Q', file_size))
        f_out.write(buf)
    os.replace(tmp_path, idx_path)
