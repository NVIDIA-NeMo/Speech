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

from pathlib import Path

from lhotse.indexing import index_file_path
from nemo.collections.common.data.lhotse.sharegpt_tar_routing import (
    build_sharegpt_tar_routing_index,
    canonical_audio_prefix_map_digest,
    ordered_manifest_content_digest,
    ordered_manifest_source_identity_digest,
    ordered_manifest_spec_path_digest,
    ordered_tar_catalog_digest,
    validate_sharegpt_tar_routing_index,
)


def ensure_sharegpt_route(
    output_path,
    *,
    manifest_paths,
    tar_paths,
    manifest_specs,
    indexes_root,
    audio_prefix_map,
    audio_placeholders=(),
    build_if_missing: bool,
) -> Path:
    """Build a missing sealed route or validate an existing route against sources."""
    output_path = Path(output_path)
    manifest_paths = tuple(map(str, manifest_paths))
    tar_paths = tuple(map(str, tar_paths))
    manifest_specs = tuple(map(str, manifest_specs))
    manifest_index_paths = tuple(
        Path(index_file_path(path, indexes_root)) for path in manifest_paths
    )
    tar_index_paths = tuple(
        Path(index_file_path(path, indexes_root)) for path in tar_paths
    )
    prefix_map = dict(audio_prefix_map or {})

    if output_path.exists():
        validate_sharegpt_tar_routing_index(
            output_path,
            offset_bearing_tar_collections=True,
            expected_manifest_spec_path_digest=ordered_manifest_spec_path_digest(
                manifest_paths, manifest_specs
            ),
            expected_manifest_content_digest=ordered_manifest_content_digest(
                manifest_paths
            ),
            expected_manifest_source_identity_digest=ordered_manifest_source_identity_digest(
                manifest_paths
            ),
            expected_tar_shard_count=len(tar_paths),
            expected_tar_catalog_digest=ordered_tar_catalog_digest(tar_paths),
            expected_audio_prefix_map_digest=canonical_audio_prefix_map_digest(
                prefix_map
            ),
        )
        return output_path
    if not build_if_missing:
        raise FileNotFoundError(f"Missing sealed ShareGPT tar route: {output_path}")

    kwargs = {
        "manifest_paths": manifest_paths,
        "tar_paths": tar_paths,
        "manifest_specs": manifest_specs,
        "manifest_index_paths": manifest_index_paths,
        "tar_index_paths": tar_index_paths,
        "audio_prefix_map": prefix_map,
    }
    if audio_placeholders:
        kwargs["audio_placeholders"] = tuple(audio_placeholders)
    return build_sharegpt_tar_routing_index(output_path, **kwargs)
