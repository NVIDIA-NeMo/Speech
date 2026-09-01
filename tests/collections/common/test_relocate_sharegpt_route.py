# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import pytest

from nemo.collections.common.data.lhotse.sharegpt_tar_routing import (
    ShareGptTarRoutingIndex,
    canonical_audio_prefix_map_digest,
    ordered_manifest_content_digest,
    ordered_manifest_content_digest_from_mirrors,
    ordered_manifest_source_identity_digest,
    ordered_manifest_source_identity_digest_from_metadata,
    ordered_manifest_spec_path_digest,
    ordered_tar_catalog_digest,
    ordered_tar_catalog_digest_from_metadata,
    write_sharegpt_tar_routing_index,
)
from scripts.dataloading._sharegpt_route_config import ShareGptRouteSpec
from scripts.dataloading.relocate_sharegpt_route import relocate_sharegpt_route


def _spec(manifest: Path, tar: Path) -> ShareGptRouteSpec:
    return ShareGptRouteSpec(
        route_path="route.sgroute",
        manifest_paths=(str(manifest),),
        tar_paths=(str(tar),),
        manifest_specs=(str(manifest),),
        audio_prefix_map={"/old": "/new"},
        audio_placeholders=("<audio>",),
    )


def _write_source_route(path: Path, spec: ShareGptRouteSpec) -> None:
    write_sharegpt_tar_routing_index(
        path,
        [[(0, 3)], [(0, 7)]],
        tar_shard_count=1,
        manifest_spec_path_digest=ordered_manifest_spec_path_digest(
            spec.manifest_paths, spec.manifest_specs
        ),
        manifest_content_digest=ordered_manifest_content_digest(spec.manifest_paths),
        manifest_source_identity_digest=ordered_manifest_source_identity_digest(
            spec.manifest_paths
        ),
        tar_catalog_digest=ordered_tar_catalog_digest(spec.tar_paths),
        audio_prefix_map_digest=canonical_audio_prefix_map_digest({}),
    )


def test_relocate_preserves_payload_and_rebinds_digests(tmp_path: Path) -> None:
    old_manifest = tmp_path / "old.jsonl"
    new_manifest = tmp_path / "new.jsonl"
    old_tar = tmp_path / "old.tar"
    new_tar = tmp_path / "new.tar"
    old_manifest.write_bytes(b'{"sample": 1}\n{"sample": 2}\n')
    new_manifest.write_bytes(old_manifest.read_bytes())
    old_tar.write_bytes(b"old tar identity")
    new_tar.write_bytes(old_tar.read_bytes())
    old_spec = _spec(old_manifest, old_tar)
    new_spec = _spec(new_manifest, new_tar)
    source = tmp_path / "source.sgroute"
    output = tmp_path / "output.sgroute"
    _write_source_route(source, old_spec)

    relocate_sharegpt_route(
        source,
        output,
        spec=new_spec,
        source_tar_paths=old_spec.tar_paths,
    )

    with ShareGptTarRoutingIndex(output) as route:
        assert route.routes_for_row(0)[0].tar_member_local_index == 3
        assert route.routes_for_row(1)[0].tar_member_local_index == 7
        assert route.header.manifest_spec_path_digest == ordered_manifest_spec_path_digest(
            new_spec.manifest_paths, new_spec.manifest_specs
        )
        assert route.header.tar_catalog_digest == ordered_tar_catalog_digest(
            new_spec.tar_paths
        )
        assert route.header.audio_prefix_map_digest == canonical_audio_prefix_map_digest(
            new_spec.audio_prefix_map
        )


def test_relocate_rejects_changed_tar_payload_with_same_size(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    old_tar = tmp_path / "old.tar"
    new_tar = tmp_path / "new.tar"
    manifest.write_text("row\n")
    old_tar.write_bytes(b"sealed payload")
    new_tar.write_bytes(b"mutated paylod")
    old_spec = _spec(manifest, old_tar)
    new_spec = _spec(manifest, new_tar)
    source = tmp_path / "source.sgroute"
    _write_source_route(source, old_spec)

    with pytest.raises(ValueError, match="payload content mismatch"):
        relocate_sharegpt_route(
            source,
            tmp_path / "output.sgroute",
            spec=new_spec,
            source_tar_paths=old_spec.tar_paths,
        )


def test_relocate_rejects_changed_manifest_content(tmp_path: Path) -> None:
    old_manifest = tmp_path / "old.jsonl"
    new_manifest = tmp_path / "new.jsonl"
    old_tar = tmp_path / "old.tar"
    new_tar = tmp_path / "new.tar"
    old_manifest.write_text("old\n")
    new_manifest.write_text("changed\n")
    old_tar.write_bytes(b"tar")
    new_tar.write_bytes(b"tar")
    source = tmp_path / "source.sgroute"
    _write_source_route(source, _spec(old_manifest, old_tar))

    with pytest.raises(ValueError, match="Manifest content changed"):
        relocate_sharegpt_route(
            source,
            tmp_path / "output.sgroute",
            spec=_spec(new_manifest, new_tar),
        )


def test_relocate_uses_ordered_offline_remote_metadata(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    old_tar = tmp_path / "source.tar"
    manifest.write_text("row\n")
    old_tar.write_bytes(b"tar")
    source = tmp_path / "source.sgroute"
    _write_source_route(source, _spec(manifest, old_tar))
    remote_path = "s3://fixture-bucket/payload/shard.tar"
    remote_spec = ShareGptRouteSpec(
        route_path="route.sgroute",
        manifest_paths=(str(manifest),),
        tar_paths=(remote_path,),
        manifest_specs=(str(manifest),),
        audio_prefix_map={"/source": "s3://fixture-bucket/payload"},
        audio_placeholders=("<audio>",),
    )
    records = [
        {
            "path": remote_path,
            "size_bytes": 3,
            "mtime_ns": 0,
            "object_identity": "etag:0123456789abcdef",
        }
    ]
    output = tmp_path / "output.sgroute"

    relocate_sharegpt_route(
        source,
        output,
        spec=remote_spec,
        tar_metadata=records,
        trust_relocated_tar_payloads=True,
    )

    with ShareGptTarRoutingIndex(output) as route:
        assert route.header.tar_catalog_digest == ordered_tar_catalog_digest_from_metadata(
            remote_spec.tar_paths, records
        )
        assert route.routes_for_row(0)[0].tar_member_local_index == 3
        assert route.routes_for_row(1)[0].tar_member_local_index == 7


@pytest.mark.parametrize(
    "record,match",
    [
        (
            {
                "path": "s3://fixture-bucket/payload/other.tar",
                "size_bytes": 3,
                "mtime_ns": 0,
                "object_identity": "etag:0123456789abcdef",
            },
            "path mismatch",
        ),
        (
            {
                "path": "s3://fixture-bucket/payload/shard.tar",
                "size_bytes": 3,
                "mtime_ns": 0,
                "object_identity": "",
            },
            "no stable object identity",
        ),
    ],
)
def test_offline_remote_metadata_fails_closed(record: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ordered_tar_catalog_digest_from_metadata(
            ("s3://fixture-bucket/payload/shard.tar",), [record]
        )


def test_relocate_uses_offline_manifest_mirror_and_metadata(tmp_path: Path) -> None:
    source_manifest = tmp_path / "source.jsonl"
    mirror = tmp_path / "mirror.jsonl"
    source_tar = tmp_path / "source.tar"
    source_manifest.write_text("row\n")
    mirror.write_bytes(source_manifest.read_bytes())
    source_tar.write_bytes(b"tar")
    source = tmp_path / "source.sgroute"
    _write_source_route(source, _spec(source_manifest, source_tar))
    target_manifest = "/cluster/manifests/manifest.jsonl"
    target_spec = ShareGptRouteSpec(
        route_path="route.sgroute",
        manifest_paths=(target_manifest,),
        tar_paths=(str(source_tar),),
        manifest_specs=(target_manifest,),
        audio_prefix_map={"/source": "/cluster/payload"},
        audio_placeholders=("<audio>",),
    )
    records = [
        {
            "path": target_manifest,
            "size_bytes": mirror.stat().st_size,
            "mtime_ns": 123,
            "object_identity": None,
            "device": 456,
            "inode": 789,
        }
    ]
    output = tmp_path / "output.sgroute"

    relocate_sharegpt_route(
        source,
        output,
        spec=target_spec,
        manifest_mirrors=(mirror,),
        manifest_metadata=records,
    )

    with ShareGptTarRoutingIndex(output) as route:
        assert route.header.manifest_content_digest == (
            ordered_manifest_content_digest_from_mirrors(
                target_spec.manifest_paths, (mirror,)
            )
        )
        assert route.header.manifest_source_identity_digest == (
            ordered_manifest_source_identity_digest_from_metadata(
                target_spec.manifest_paths, records
            )
        )


def test_relocate_rejects_incomplete_offline_manifest_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    tar = tmp_path / "source.tar"
    manifest.write_text("row\n")
    tar.write_bytes(b"tar")
    source = tmp_path / "source.sgroute"
    spec = _spec(manifest, tar)
    _write_source_route(source, spec)

    with pytest.raises(ValueError, match="requires metadata"):
        relocate_sharegpt_route(
            source,
            tmp_path / "output.sgroute",
            spec=spec,
            manifest_mirrors=(manifest,),
        )


def test_relocate_uses_digest_only_offline_manifest_identity(tmp_path: Path) -> None:
    source_manifest = tmp_path / "source.jsonl"
    source_tar = tmp_path / "source.tar"
    source_manifest.write_text("row\n")
    source_tar.write_bytes(b"tar")
    source = tmp_path / "source.sgroute"
    old_spec = _spec(source_manifest, source_tar)
    _write_source_route(source, old_spec)
    target_manifest = "/cluster/manifests/manifest.jsonl"
    target_spec = ShareGptRouteSpec(
        route_path="route.sgroute",
        manifest_paths=(target_manifest,),
        tar_paths=(str(source_tar),),
        manifest_specs=(target_manifest,),
        audio_prefix_map={"/source": "/cluster/payload"},
        audio_placeholders=("<audio>",),
    )
    records = [
        {
            "path": target_manifest,
            "size_bytes": source_manifest.stat().st_size,
            "mtime_ns": 123,
            "object_identity": None,
            "device": 456,
            "inode": 789,
        }
    ]
    digest = ordered_manifest_content_digest(old_spec.manifest_paths)
    output = tmp_path / "output.sgroute"

    relocate_sharegpt_route(
        source,
        output,
        spec=target_spec,
        manifest_metadata=records,
        manifest_content_digest=digest.hex(),
    )

    with ShareGptTarRoutingIndex(output) as route:
        assert route.header.manifest_content_digest == digest
        assert route.header.manifest_source_identity_digest == (
            ordered_manifest_source_identity_digest_from_metadata(
                target_spec.manifest_paths, records
            )
        )
