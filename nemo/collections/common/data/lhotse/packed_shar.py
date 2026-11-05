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
Index-pack (``.idxpack``) support for uncompressed Lhotse Shar directories.

A Shar directory has ``cuts.NNNNNN.jsonl`` manifest shards and, for every extra field (``recording``,
``features``, custom arrays, ...), ``<field>.NNNNNN.tar`` shards that hold one (data, metadata) member pair per cut.
The conventional ``LazyIndexedSharIterator`` opens one ``.idx`` sidecar per shard and per field at construction
(3 x 2000 files for a 2000-shard directory with two fields). An index pack stores all of these offset arrays in one
memory-mapped file, so construction opens only the pack's catalog, and the source files are read with ``pread``
through a bounded descriptor LRU.

Pack layout for one Shar directory ``shar_path`` (see :func:`shar_pack_collection_specs`):

* ``shar_cuts``   (kind ``jsonl``)   one path per ``cuts.*.jsonl`` shard; one record per cut.
* ``shar_field``  (kind ``wds_tar``) one collection per field; one path per ``<field>.*.tar`` shard; one record per
  (data, metadata) member pair, i.e. one per cut.
* ``shar_fields`` (path-only collection) the ordered field names, so the reader finds the fields without listing
  the directory.

Collection identities are derived from the ``shar_path`` string exactly as written in the YAML entry.
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Optional

from lhotse.dataset.dataloading import PartitionedIndexedIterator, resolve_seed
from lhotse.index_pack import IndexPack, IndexPackCollectionSpec, index_pack_collection_key, open_index_pack
from lhotse.lazy import attach_graph_origin, normalize_graph_token
from lhotse.packed_lazy import read_packed_range
from lhotse.serialization import decode_json_line, deserialize_item
from lhotse.shar.readers.indexed import LazyIndexedSharIterator
from lhotse.shar.utils import fill_shar_placeholder, fill_shar_placeholder_lazy

SHAR_CUTS_ROLE = "shar_cuts"
SHAR_CUTS_KIND = "jsonl"
SHAR_FIELD_ROLE = "shar_field"
SHAR_FIELD_KIND = "wds_tar"
SHAR_FIELDS_ROLE = "shar_fields"
SHAR_FIELDS_KIND = "shar_fields"

_TAR_BLOCK = 512


def _ceil_block(size: int) -> int:
    return (size + _TAR_BLOCK - 1) // _TAR_BLOCK * _TAR_BLOCK


def shar_cuts_collection_key(shar_path) -> bytes:
    return index_pack_collection_key(SHAR_CUTS_ROLE, SHAR_CUTS_KIND, {"shar_path": str(shar_path)})


def shar_fields_collection_key(shar_path) -> bytes:
    return index_pack_collection_key(SHAR_FIELDS_ROLE, SHAR_FIELDS_KIND, {"shar_path": str(shar_path)})


def shar_field_collection_key(shar_path, field: str) -> bytes:
    return index_pack_collection_key(SHAR_FIELD_ROLE, SHAR_FIELD_KIND, {"shar_path": str(shar_path), "field": field})


def discover_shar_shards(shar_dir) -> tuple[list[str], dict[str, list[str]]]:
    """
    List the uncompressed shards of one Shar directory in the order Lhotse's directory discovery uses
    (sorted by path). Returns ``(cuts_shards, {field: tar_shards})``. Compressed manifests (``.jsonl.gz``) and
    non-tar field shards are rejected: they are not randomly addressable.
    """
    from lhotse.shar.readers.lazy import _discover_fields

    fields, streams = _discover_fields(Path(shar_dir))
    cuts = [str(p) for p in streams["cuts"]]
    if not cuts:
        raise ValueError(f"No cuts.*.jsonl shards in {shar_dir}")
    bad = [p for p in cuts if not p.endswith(".jsonl")]
    if bad:
        raise ValueError(f"Packed Shar requires uncompressed cuts shards (.jsonl); got {bad[0]}")
    out = {}
    for field in sorted(fields):
        shards = [str(p) for p in streams[field]]
        bad = [p for p in shards if not p.endswith(".tar")]
        if bad:
            raise ValueError(f"Packed Shar requires uncompressed tar field shards; field {field!r} has {bad[0]}")
        if len(shards) != len(cuts):
            raise ValueError(
                f"Shar field {field!r} has {len(shards)} shards but there are {len(cuts)} cuts shards in {shar_dir}"
            )
        out[field] = shards
    return cuts, out


def shar_pack_collection_specs(shar_path, cuts_shards, field_shards) -> list[IndexPackCollectionSpec]:
    """Collection specs (to be passed to ``write_index_pack``) of one Shar directory."""
    specs = [
        IndexPackCollectionSpec(
            role=SHAR_CUTS_ROLE,
            kind=SHAR_CUTS_KIND,
            source_spec={"shar_path": str(shar_path)},
            paths=tuple(cuts_shards),
        ),
        IndexPackCollectionSpec(
            role=SHAR_FIELDS_ROLE,
            kind=SHAR_FIELDS_KIND,
            source_spec={"shar_path": str(shar_path)},
            paths=tuple(sorted(field_shards)),
            offsets_required=False,
        ),
    ]
    for field in sorted(field_shards):
        specs.append(
            IndexPackCollectionSpec(
                role=SHAR_FIELD_ROLE,
                kind=SHAR_FIELD_KIND,
                source_spec={"shar_path": str(shar_path), "field": field},
                paths=tuple(field_shards[field]),
            )
        )
    return specs


def _parse_member(raw: bytes, offset: int, source: str):
    header = raw[offset : offset + _TAR_BLOCK]
    if len(header) < _TAR_BLOCK:
        raise RuntimeError(f"Unexpected EOF reading tar header at offset {offset} of the packed range from {source}")
    info = tarfile.TarInfo.frombuf(header, tarfile.ENCODING, "surrogateescape")
    path = Path(info.name)
    if path.suffix in (".nodata", ".nometa"):
        return None, path, info
    start = offset + _TAR_BLOCK
    data = raw[start : start + info.size]
    if len(data) != info.size:
        raise EOFError(f"Truncated tar member {info.name!r} in the packed range from {source}")
    return data, path, info


def decode_shar_pair(raw: bytes, source: str = "<tar>"):
    """Decode one Shar (data, metadata) member pair from its exact byte range, as ``IndexedTarReader.__getitem__``."""
    data, data_path, info = _parse_member(raw, 0, source)
    next_offset = _TAR_BLOCK + _ceil_block(info.size)
    meta_bytes, _, _ = _parse_member(raw, next_offset, source)
    if meta_bytes is None:
        return None, data_path
    meta = deserialize_item(decode_json_line(meta_bytes.decode("utf-8")))
    fill_shar_placeholder(manifest=meta, data=data, tarpath=data_path)
    return meta, data_path


class LazyPackedSharIterator(LazyIndexedSharIterator):
    """
    :class:`~lhotse.shar.readers.indexed.LazyIndexedSharIterator` whose offsets come from an ``.idxpack``.

    The iteration order, worker partitioning, shuffling and ``state_dict`` are inherited unchanged, so a packed and a
    sidecar-based reader over the same directory yield the same cuts in the same order and resume identically. Only
    ``__init__``, random access and pickling differ.

    :param index_pack: an :class:`IndexPack` or a path to the ``.idxpack`` file.
    :param shar_path: the Shar directory string the pack was built with (it determines the collection keys).
    :param fields: optional subset of the packed fields to attach to cuts (default: all of them).
    :param lazy: attach Shar lazy pointers instead of inlining the payload (as in ``LazyIndexedSharIterator``).
    :param max_open_files: descriptor LRU bound shared by all readers of this pack in a process.
    """

    def __init__(
        self,
        index_pack,
        shar_path,
        *,
        fields: Optional[list[str]] = None,
        shuffle: bool = False,
        seed=42,
        split_for_dataloading: bool = False,
        lazy: bool = False,
        max_open_files: int = 32,
    ) -> None:
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.index_pack = index_pack if isinstance(index_pack, IndexPack) else open_index_pack(index_pack)
        self.shar_path = str(shar_path)
        self.max_open_files = max_open_files
        self.in_dir = None
        self._cuts = self.index_pack.collection(shar_cuts_collection_key(self.shar_path))
        fields_collection = self.index_pack.collection(shar_fields_collection_key(self.shar_path))
        packed_fields = [fields_collection.path_for_shard(i) for i in range(fields_collection.sequence_count)]
        if fields is not None:
            missing = sorted(set(fields) - set(packed_fields))
            if missing:
                raise ValueError(f"Fields {missing} are not packed for {self.shar_path}; packed: {packed_fields}")
            packed_fields = [f for f in packed_fields if f in set(fields)]
        self._field_order = packed_fields
        self.fields = set(packed_fields)
        self._field_collections = {
            f: self.index_pack.collection(shar_field_collection_key(self.shar_path, f)) for f in packed_fields
        }
        for f, col in self._field_collections.items():
            if col.sequence_count != self._cuts.sequence_count or len(col) != len(self._cuts):
                raise ValueError(
                    f"Packed Shar field {f!r} of {self.shar_path} is inconsistent with its cuts: "
                    f"{col.sequence_count} shards / {len(col)} records vs "
                    f"{self._cuts.sequence_count} / {len(self._cuts)}"
                )
        self.num_shards = self._cuts.sequence_count
        self._index_streams = None
        self._lazy = lazy
        self.shuffle = shuffle
        self.seed = seed
        self.split_for_dataloading = split_for_dataloading
        self.epoch = 0
        self._iter_state = PartitionedIndexedIterator(
            shuffle=self.shuffle, seed=resolve_seed(self.seed) if isinstance(self.seed, int) else 0
        )
        self._total_len = len(self._cuts)

    def __len__(self) -> int:
        return self._total_len

    def _read(self, collection, location) -> bytes:
        return read_packed_range(
            self.index_pack, location.path, location.start, location.end, max_open_files=self.max_open_files
        )

    def __getitem__(self, idx):
        idx = normalize_graph_token(idx)
        item_epoch = self.epoch
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise TypeError(
                    "LazyPackedSharIterator expects graph restore tokens shaped like (global_index, shar_epoch)."
                )
            idx, item_epoch = idx
        if idx < 0:
            idx += self._total_len
        if idx < 0 or idx >= self._total_len:
            raise IndexError(f"index {idx} out of range for LazyPackedSharIterator with {self._total_len} cuts")

        loc = self._cuts.locate(idx)
        cut = deserialize_item(decode_json_line(self._read(self._cuts, loc).decode("utf-8")))
        shard_idx, pos = loc.shard_index, loc.local_index
        for field in self._field_order:
            floc = self._field_collections[field].locate_in_shard(shard_idx, pos)
            if self._lazy:
                fill_shar_placeholder_lazy(
                    cut, field=field, tar_path=floc.path, offset=floc.start, end_offset=floc.end
                )
                continue
            maybe_manifest, data_path = decode_shar_pair(self._read(None, floc), floc.path)
            if maybe_manifest is not None:
                assert (
                    str(data_path.parent / data_path.stem) == cut.id
                ), f"Mismatched IDs: cut ID is '{cut.id}' but found data with name '{data_path}' for field {field}"
                setattr(cut, field, maybe_manifest)

        cut.shard_origin = Path(loc.path)
        cut.shar_epoch = item_epoch
        attach_graph_origin(cut, (idx, item_epoch))
        return cut

    # The parent pickles its reader caches; this class has none, and the pack and collections pickle themselves.
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__ = state
