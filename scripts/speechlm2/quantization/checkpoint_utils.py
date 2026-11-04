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

"""Safetensors and config helpers shared by the SpeechLM quantization scripts.

Tensors are read and copied at the byte level, so a module copied from the source checkpoint (the
perception weights, the MTP draft head, the CTC timestamp head) stays bit-identical to it.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
# os.link errors that mean "this filesystem will not link these files", e.g. across filesystems.
LINK_REFUSED = {errno.EXDEV, errno.EPERM, errno.EMLINK, errno.EOPNOTSUPP}


def check_output_path(output: Path, *inputs: Path) -> None:
    """Refuse an output path that is an input, contains one, or lies inside one.

    Run before anything deletes or writes ``output``: replacing an output that holds an input would delete it.
    """
    output = Path(output).resolve()
    for path in (Path(p).resolve() for p in inputs):
        if output == path or output in path.parents or path in output.parents:
            raise ValueError(f"output {output} overlaps input {path}; choose a separate directory")


def link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link ``source`` to ``destination``, or copy it where the filesystem refuses the link."""
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in LINK_REFUSED:
            raise
        shutil.copy2(source, destination)


def read_json(path: Path) -> dict:
    """Load a JSON file."""
    with Path(path).open() as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    """Write ``data`` as indented JSON with a trailing newline."""
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def read_header(path: Path) -> tuple[dict, int]:
    """Return a safetensors file's header and the byte offset where its tensor data starts."""
    with Path(path).open("rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(length)), 8 + length


def weight_files(checkpoint: Path) -> list[str]:
    """Return the weight files of a checkpoint: those its index lists, or its single ``model.safetensors``."""
    checkpoint = Path(checkpoint)
    if (checkpoint / INDEX_FILE).exists():
        return sorted(set(read_json(checkpoint / INDEX_FILE)["weight_map"].values()))
    if (checkpoint / SINGLE_FILE).exists():
        return [SINGLE_FILE]
    raise FileNotFoundError(f"{checkpoint} has neither {INDEX_FILE} nor {SINGLE_FILE}")


def read_weight_map(checkpoint: Path) -> dict[str, str]:
    """Return a checkpoint's tensor-to-file map: its index, or the header of its single ``model.safetensors``."""
    checkpoint = Path(checkpoint)
    if (checkpoint / INDEX_FILE).exists():
        return dict(read_json(checkpoint / INDEX_FILE)["weight_map"])
    return {name: SINGLE_FILE for name in read_header(checkpoint / SINGLE_FILE)[0] if name != "__metadata__"}


@dataclass(frozen=True)
class TensorEntry:
    """Where one tensor's bytes live: ``path`` holds them between the absolute offsets ``start`` and ``end``."""

    name: str
    dtype: str
    shape: list[int]
    path: Path
    start: int
    end: int


def find_tensors(checkpoint: Path, prefixes: tuple[str, ...]) -> list[TensorEntry]:
    """Return every tensor of ``checkpoint`` whose name starts with one of ``prefixes``, sorted by name."""
    found = []
    for file_name in weight_files(checkpoint):
        path = Path(checkpoint) / file_name
        header, base = read_header(path)
        for name, entry in header.items():
            if name != "__metadata__" and name.startswith(prefixes):
                start, end = entry["data_offsets"]
                found.append(TensorEntry(name, entry["dtype"], entry["shape"], path, base + start, base + end))
    return sorted(found, key=lambda entry: entry.name)


def file_metadata(entries: Iterable[TensorEntry], prefix: str) -> dict[str, str]:
    """Return the ``__metadata__`` entries starting with ``prefix`` from the files that hold ``entries``."""
    found: dict[str, str] = {}
    for path in sorted({entry.path for entry in entries}):
        metadata = read_header(path)[0].get("__metadata__") or {}
        for key, value in metadata.items():
            if key.startswith(prefix):
                if found.get(key, value) != value:
                    raise ValueError(f"source files disagree on metadata entry {key!r}")
                found[key] = value
    return found


def copy_tensors(entries: list[TensorEntry], out: Path, metadata: dict[str, str] | None = None) -> int:
    """Write ``entries`` byte for byte into a new safetensors file, verify it by SHA-256, and return its data size."""
    header: dict = {"__metadata__": {"format": "pt", **(metadata or {})}}
    offset = 0
    for entry in entries:
        size = entry.end - entry.start
        header[entry.name] = {"dtype": entry.dtype, "shape": entry.shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)

    digests = {}
    with Path(out).open("wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for entry in entries:
            data = _read_bytes(entry)
            f.write(data)
            digests[entry.name] = hashlib.sha256(data).hexdigest()

    written = {entry.name: entry for entry in find_tensors_in_file(out)}
    if sorted(written) != sorted(digests):
        raise RuntimeError(f"{out} does not hold the tensors that were copied into it")
    for name, digest in digests.items():
        if hashlib.sha256(_read_bytes(written[name])).hexdigest() != digest:
            raise RuntimeError(f"{out}: bytes of {name} differ from the source")
    return offset


def find_tensors_in_file(path: Path) -> list[TensorEntry]:
    """Return every tensor of one safetensors file."""
    header, base = read_header(path)
    return [
        TensorEntry(
            name,
            entry["dtype"],
            entry["shape"],
            Path(path),
            base + entry["data_offsets"][0],
            base + entry["data_offsets"][1],
        )
        for name, entry in header.items()
        if name != "__metadata__"
    ]


def _read_bytes(entry: TensorEntry) -> bytes:
    with entry.path.open("rb") as f:
        f.seek(entry.start)
        data = f.read(entry.end - entry.start)
    if len(data) != entry.end - entry.start:
        raise RuntimeError(f"short read for {entry.name} in {entry.path}")
    return data
