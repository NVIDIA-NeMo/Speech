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

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import SplitResult, urlsplit, urlunsplit


@dataclass(frozen=True)
class _PrefixEntry:
    source: str
    destination: str
    destination_url: SplitResult | None


class AudioPathPrefixMap:
    """Safely map immutable manifest audio paths to cluster storage roots.

    Relative paths and already URL-addressed paths are returned unchanged.
    When at least one prefix is configured, every absolute POSIX source path
    must match a unique longest component prefix.
    """

    def __init__(self, mapping: Mapping[str, str] | None) -> None:
        raw_mapping = {} if mapping is None else dict(mapping)
        entries: list[_PrefixEntry] = []
        normalized_sources: dict[str, str] = {}
        canonical_mapping: dict[str, str] = {}

        for raw_source, raw_destination in raw_mapping.items():
            if not isinstance(raw_source, str) or not isinstance(raw_destination, str):
                raise TypeError("audio_path_prefix_map keys and values must be strings")
            source = _normalize_absolute_posix(raw_source, field="source prefix")
            previous = normalized_sources.get(source)
            if previous is not None:
                raise ValueError(
                    f"Audio path prefix keys {previous!r} and {raw_source!r} normalize to the same source prefix "
                    f"{source!r}."
                )
            normalized_sources[source] = raw_source

            destination, destination_url = _normalize_destination(raw_destination)
            canonical_mapping[source] = destination
            entries.append(
                _PrefixEntry(
                    source=source,
                    destination=destination,
                    destination_url=destination_url,
                )
            )

        self._entries = tuple(
            sorted(
                entries,
                key=lambda entry: (
                    len(PurePosixPath(entry.source).parts),
                    entry.source,
                ),
                reverse=True,
            )
        )
        self.mapping = dict(sorted(canonical_mapping.items()))
        self.canonical_json = json.dumps(self.mapping, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        self.digest = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def resolve(self, path: str) -> str:
        """Resolve one source path while preserving relative and URL inputs."""

        if not isinstance(path, str):
            raise TypeError(f"Audio path must be a string, got {type(path).__name__}")
        if _is_url(path):
            return path

        _reject_lexical_traversal(path, field="audio path")
        if not path.startswith("/"):
            return path

        source_path = _normalize_absolute_posix(path, field="audio path")
        if not self._entries:
            return path

        matching = [entry for entry in self._entries if _is_component_prefix(entry.source, source_path)]
        if not matching:
            raise ValueError(
                f"Absolute audio path {path!r} does not match any configured source prefix in "
                "audio_path_prefix_map."
            )
        entry = matching[0]
        source_root = PurePosixPath(entry.source)
        relative = PurePosixPath(source_path).relative_to(source_root)
        if entry.destination_url is not None:
            return _join_url_root(entry.destination_url, relative)
        return _join_local_root(entry.destination, relative)


def _is_url(value: str) -> bool:
    parsed = urlsplit(value)
    return bool(parsed.scheme and "://" in value)


def _reject_lexical_traversal(value: str, *, field: str) -> None:
    path_part = urlsplit(value).path if _is_url(value) else value
    if any(part in {".", ".."} for part in path_part.split("/")):
        raise ValueError(f"Audio {field} {value!r} must not contain '.' or '..' path components.")


def _normalize_absolute_posix(value: str, *, field: str) -> str:
    _reject_lexical_traversal(value, field=field)
    path = PurePosixPath(value)
    if not path.is_absolute() or value.startswith("//"):
        raise ValueError(f"Audio {field} {value!r} must be an absolute POSIX path.")
    return str(path)


def _normalize_destination(value: str) -> tuple[str, SplitResult | None]:
    _reject_lexical_traversal(value, field="destination root")
    parsed = urlsplit(value)
    if parsed.scheme and "://" in value:
        if not parsed.netloc:
            raise ValueError(f"Audio destination root {value!r} must include a URL authority/bucket.")
        if parsed.query or parsed.fragment:
            raise ValueError(f"Audio destination root {value!r} must not contain a URL query or fragment.")
        normalized_path = str(PurePosixPath(parsed.path or "/"))
        normalized = urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))
        return normalized, urlsplit(normalized)
    try:
        return _normalize_absolute_posix(value, field="destination root"), None
    except ValueError as error:
        raise ValueError(f"Audio destination root {value!r} must be an absolute POSIX path or URL.") from error


def _is_component_prefix(prefix: str, path: str) -> bool:
    if prefix == "/":
        return path.startswith("/")
    return path == prefix or path.startswith(f"{prefix}/")


def _join_local_root(destination: str, relative: PurePosixPath) -> str:
    root = PurePosixPath(destination)
    resolved = root / relative
    if resolved != root and not _is_component_prefix(str(root), str(resolved)):
        raise ValueError(f"Resolved audio path {str(resolved)!r} escapes destination root {destination!r}.")
    return str(resolved)


def _join_url_root(destination: SplitResult, relative: PurePosixPath) -> str:
    root = PurePosixPath(destination.path or "/")
    resolved = root / relative
    if resolved != root and not _is_component_prefix(str(root), str(resolved)):
        raise ValueError(f"Resolved URL path {str(resolved)!r} escapes destination root {str(root)!r}.")
    return urlunsplit((destination.scheme, destination.netloc, str(resolved), "", ""))
