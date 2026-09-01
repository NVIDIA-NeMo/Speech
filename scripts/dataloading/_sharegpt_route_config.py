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

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, ListConfig
from scripts.dataloading.build_indexes import (
    _TRANSFORM_TYPES,
    _expand_jsonl_with_specs,
    _expand_tars,
    _resolve_input_cfg,
)


@dataclass(frozen=True)
class ShareGptRouteSpec:
    route_path: str
    manifest_paths: tuple[str, ...]
    tar_paths: tuple[str, ...]
    manifest_specs: tuple[str, ...]
    audio_prefix_map: dict[str, str]
    audio_placeholders: tuple[str, ...]


def discover_sharegpt_route_specs(
    entry,
    specs=None,
    *,
    data_blend_dir: str | Path | None = None,
) -> list[ShareGptRouteSpec]:
    """Discover collection-mode route declarations through groups/transforms."""
    if specs is None:
        specs = []
    if isinstance(entry, (list, ListConfig)):
        for item in entry:
            discover_sharegpt_route_specs(item, specs, data_blend_dir=data_blend_dir)
        return specs
    if not isinstance(entry, (dict, DictConfig)):
        return specs

    typ = entry.get("type")
    if typ is None:
        for value in entry.values():
            discover_sharegpt_route_specs(value, specs, data_blend_dir=data_blend_dir)
        return specs
    if typ == "group" or typ in _TRANSFORM_TYPES:
        nested = _resolve_input_cfg(entry.get("input_cfg"), data_blend_dir)
        if nested is not None:
            discover_sharegpt_route_specs(nested, specs, data_blend_dir=data_blend_dir)
        return specs
    if typ != "share_gpt" or entry.get("tar_lookup_mode") != "collection":
        return specs

    route_path = entry.get("tar_routing_filepath")
    legacy_route_path = entry.get("tar_routing_index")
    if route_path and legacy_route_path and str(route_path) != str(legacy_route_path):
        raise ValueError("tar_routing_filepath and tar_routing_index disagree")
    route_path = route_path or legacy_route_path
    if not isinstance(route_path, (str, Path)) or not str(route_path).endswith(
        ".sgroute"
    ):
        raise ValueError(
            "ShareGPT collection mode requires tar_routing_filepath with the .sgroute suffix"
        )
    manifest_paths, manifest_specs = _expand_jsonl_with_specs(
        entry.get("manifest_filepath")
    )
    tar_paths = _expand_tars(entry.get("tarred_audio_filepaths"))
    specs.append(
        ShareGptRouteSpec(
            route_path=str(route_path),
            manifest_paths=tuple(manifest_paths),
            tar_paths=tuple(tar_paths),
            manifest_specs=tuple(manifest_specs),
            audio_prefix_map=dict(entry.get("audio_path_prefix_map") or {}),
            audio_placeholders=tuple(entry.get("audio_placeholders") or ()),
        )
    )
    return specs
