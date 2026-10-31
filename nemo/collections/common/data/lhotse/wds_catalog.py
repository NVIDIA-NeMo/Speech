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
"""Bounded WebDataset shard discovery shared by runtime and index tooling."""

from __future__ import annotations

import json
import posixpath
import re
from fnmatch import fnmatchcase
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit

import yaml
from braceexpand import braceexpand

_REMOTE_SCHEMES = frozenset({"ais", "s3"})
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_MAX_CATALOG_BYTES = 64 << 20
_MAX_CATALOG_SHARDS = 1_000_000
_GENERATED_CATALOG_SUFFIX = ".wds-catalog.json"


def discover_webdataset_shards(data_dir: str, *, require_catalog: bool = False) -> list[str]:
    """Return WebDataset tar paths in explicit catalog order.

    ``data_dir`` may be a local/remote dataset root or a direct generated
    ``*.wds-catalog.json`` path. Remote roots are never listed or recursively
    scanned: only ``wids-meta.json``, ``.nv-meta/split.yaml``, and
    ``wds-catalog.json`` are probed by exact object name. The legacy recursive
    fallback remains local-only for callers that do not require a catalog.
    """
    data_dir = str(data_dir)
    if _is_remote(data_dir):
        catalog_path, catalog_kind, raw = _read_remote_catalog(data_dir)
        base = _catalog_base(data_dir, catalog_path, catalog_kind)
        return _parse_catalog(raw, catalog_path, catalog_kind, base)

    root = Path(data_dir)
    if root.name.endswith(_GENERATED_CATALOG_SUFFIX):
        raw = _read_local_catalog(root)
        return _parse_catalog(raw, str(root), "generated", str(root.parent))

    candidates = (
        (root / "wids-meta.json", "wids"),
        (root / ".nv-meta" / "split.yaml", "split"),
        (root / "wds-catalog.json", "generated"),
    )
    for path, kind in candidates:
        if path.is_file():
            return _parse_catalog(_read_local_catalog(path), str(path), kind, str(root))

    if require_catalog:
        raise _missing_catalog_error(data_dir)

    shards = sorted(str(path) for path in root.rglob("*.tar")) if root.is_dir() else []
    if not shards:
        raise FileNotFoundError(
            f"No wids-meta.json and no .tar files found under {root}; " "no bounded shard catalog was found."
        )
    return shards


def _is_remote(path: str) -> bool:
    if not _URL_RE.match(path):
        return False
    parsed = urlsplit(path)
    if parsed.scheme.lower() not in _REMOTE_SCHEMES:
        raise ValueError(
            f"Unsupported remote WDS scheme {parsed.scheme!r} for {path!r}; "
            f"supported schemes are {sorted(_REMOTE_SCHEMES)}"
        )
    if not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"Remote WDS path must have an authority and no query/fragment: {path!r}")
    return True


def _read_remote_catalog(data_dir: str) -> tuple[str, str, bytes]:
    if data_dir.endswith(_GENERATED_CATALOG_SUFFIX):
        return data_dir, "generated", _read_remote_bytes(data_dir)

    candidates = (
        (_join_remote(data_dir, "wids-meta.json"), "wids"),
        (_join_remote(data_dir, ".nv-meta/split.yaml"), "split"),
        (_join_remote(data_dir, "wds-catalog.json"), "generated"),
    )
    for path, kind in candidates:
        try:
            return path, kind, _read_remote_bytes(path)
        except Exception as error:
            if not _is_missing_remote_object(error):
                raise RuntimeError(f"Could not read remote WDS catalog candidate {path}: {error}") from error
    raise _missing_catalog_error(data_dir)


def _read_remote_bytes(path: str) -> bytes:
    from lhotse.ais import AISRangeReader

    with AISRangeReader(path) as source:
        size = int(source.size)
        if size > _MAX_CATALOG_BYTES:
            raise ValueError(f"Remote WDS catalog {path} is {size} bytes; maximum is {_MAX_CATALOG_BYTES}")
        source.seek(0)
        data = source.read(size)
    if len(data) != size:
        raise EOFError(f"Short remote WDS catalog read for {path}: expected {size}, received {len(data)}")
    return data


def _read_local_catalog(path: Path) -> bytes:
    size = path.stat().st_size
    if size > _MAX_CATALOG_BYTES:
        raise ValueError(f"WDS catalog {path} is {size} bytes; maximum is {_MAX_CATALOG_BYTES}")
    return path.read_bytes()


def _is_missing_remote_object(error: Exception) -> bool:
    if isinstance(error, FileNotFoundError):
        return True
    return error.__class__.__name__ in {
        "ErrBckNotFound",
        "ErrObjNotFound",
        "ErrRemoteBckNotFound",
    }


def _catalog_base(data_dir: str, catalog_path: str, catalog_kind: str) -> str:
    if catalog_kind == "generated" and data_dir.endswith(_GENERATED_CATALOG_SUFFIX):
        parsed = urlsplit(catalog_path)
        parent = posixpath.dirname(parsed.path)
        return urlunsplit(SplitResult(parsed.scheme, parsed.netloc, parent, "", ""))
    return data_dir.rstrip("/")


def _parse_catalog(raw: bytes, catalog_path: str, kind: str, base: str) -> list[str]:
    try:
        if kind in {"wids", "generated"}:
            catalog = json.loads(raw)
        else:
            catalog = yaml.safe_load(raw) or {}
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"Malformed WDS catalog {catalog_path}: {error}") from error

    excludes: list[str] = []
    if kind == "wids":
        shardlist = catalog.get("shardlist") if isinstance(catalog, dict) else None
        specs = _extract_urls(shardlist, catalog_path, "WIDS shardlist")
    elif kind == "split":
        split_parts = catalog.get("split_parts") if isinstance(catalog, dict) else None
        train = split_parts.get("train") if isinstance(split_parts, dict) else None
        specs = _extract_strings(train, catalog_path, "train shard list")
        raw_excludes = catalog.get("exclude", [])
        excludes = _extract_strings(raw_excludes, catalog_path, "exclude list", allow_empty=True)
    elif kind == "generated":
        if not isinstance(catalog, dict) or catalog.get("format") != "nemo-wds-shard-catalog":
            raise ValueError(f"Invalid generated WDS catalog format in {catalog_path}")
        if catalog.get("version") != 1:
            raise ValueError(f"Unsupported generated WDS catalog version in {catalog_path}")
        specs = _extract_urls(catalog.get("shards"), catalog_path, "generated shard list")
    else:  # pragma: no cover - internal invariant
        raise AssertionError(kind)

    expanded_excludes = []
    for spec in excludes:
        expanded_excludes.extend(_bounded_expand(spec, catalog_path, len(expanded_excludes)))

    shards: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        for expanded in _bounded_expand(spec, catalog_path, len(shards)):
            if any(fnmatchcase(expanded, pattern) for pattern in expanded_excludes):
                continue
            resolved = _resolve_catalog_path(base, expanded, catalog_path)
            if resolved in seen:
                raise ValueError(f"Duplicate WDS shard {resolved!r} in {catalog_path}")
            seen.add(resolved)
            shards.append(resolved)
    if not shards:
        raise ValueError(f"Bounded shard catalog {catalog_path} resolved to zero tar paths")
    return shards


def _extract_urls(value, catalog_path: str, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Invalid or empty {label} in {catalog_path}")
    urls = []
    for position, item in enumerate(value):
        url = item.get("url") if isinstance(item, dict) else None
        if not isinstance(url, str) or not url:
            raise ValueError(f"Invalid WDS shard URL at position {position} in {catalog_path}")
        urls.append(url)
    return urls


def _extract_strings(value, catalog_path: str, label: str, *, allow_empty: bool = False) -> list[str]:
    if value is None and allow_empty:
        return []
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"Invalid or empty {label} in {catalog_path}")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Invalid {label} in {catalog_path}")
    return list(value)


def _bounded_expand(spec: str, catalog_path: str, already_expanded: int) -> list[str]:
    normalized = spec
    for opener in ("(", "[", "<", "_OP_"):
        normalized = normalized.replace(opener, "{")
    for closer in (")", "]", ">", "_CL_"):
        normalized = normalized.replace(closer, "}")
    expanded = []
    for value in braceexpand(normalized, escape=False):
        if already_expanded + len(expanded) >= _MAX_CATALOG_SHARDS:
            raise ValueError(f"WDS catalog {catalog_path} exceeds the {_MAX_CATALOG_SHARDS}-shard expansion bound")
        expanded.append(value)
    return expanded


def _resolve_catalog_path(base: str, spec: str, catalog_path: str) -> str:
    if _URL_RE.match(spec):
        _is_remote(spec)
        return spec
    if _is_remote(base):
        return _join_remote(base, spec, catalog_path=catalog_path)
    path = Path(spec)
    if path.is_absolute():
        return str(path)
    if any(part == ".." for part in path.parts):
        raise ValueError(f"WDS catalog path traversal is forbidden in {catalog_path}: {spec!r}")
    return str(Path(base) / path)


def _join_remote(base: str, relative: str, *, catalog_path: str | None = None) -> str:
    parsed = urlsplit(base)
    if _URL_RE.match(relative):
        _is_remote(relative)
        return relative
    if relative.startswith("/"):
        raise ValueError(f"Remote WDS catalog path must be relative: {relative!r}")

    depth = len([part for part in parsed.path.split("/") if part])
    for part in relative.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            depth -= 1
            if depth < 0:
                origin = catalog_path or base
                raise ValueError(f"Remote WDS catalog path traversal escapes its authority in {origin}: {relative!r}")
        else:
            depth += 1
    path = posixpath.normpath(posixpath.join(parsed.path or "/", relative))
    if not path.startswith("/"):
        path = f"/{path}"
    return urlunsplit(SplitResult(parsed.scheme, parsed.netloc, path, "", ""))


def _missing_catalog_error(data_dir: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"No bounded shard catalog found under {data_dir}; expected wids-meta.json, "
        ".nv-meta/split.yaml, or wds-catalog.json"
    )
