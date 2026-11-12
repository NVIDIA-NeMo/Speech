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

import struct
from pathlib import Path

import pytest
from lhotse.index_pack import IndexPack, IndexPackArraySpec, IndexPackCollectionSpec, write_index_pack
from lhotse.indexing import create_jsonl_index
from scripts.dataloading.convert_indexes_to_idxpack import NativeTarOrdinalMapSpec
from scripts.dataloading.rebind_idxpack_collections import (
    _read_pack_layout,
    discover_rebind_pack_collections,
    discover_relocated_pack_collections,
    rebind_idxpack_collections,
    relocate_idxpack_collections,
)

from nemo.collections.common.data.lhotse.nemo_tar_routing import nemo_tar_ordinal_map_source_spec


def _spec(path: Path, declaration: str) -> IndexPackCollectionSpec:
    return IndexPackCollectionSpec(
        role="records",
        kind="json-lines",
        source_spec=declaration,
        paths=(str(path),),
        offsets_required=False,
    )


def test_rebinds_collection_key_and_layout_without_changing_payload(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text('{"id": 1}\n')
    source_spec = _spec(records, "/old/location/records.jsonl")
    target_spec = _spec(records, "/new/location/records.jsonl")
    source = tmp_path / "source.idxpack"
    output = tmp_path / "output.idxpack"
    write_index_pack(source, [source_spec])

    result = rebind_idxpack_collections(source, output, [target_spec])

    assert result["keys_changed"] == 1
    with IndexPack(output) as pack:
        assert pack.collection(target_spec.key).path_for_shard(0) == str(records)
        with pytest.raises(KeyError):
            pack.collection(source_spec.key)


@pytest.mark.parametrize("relocate", [False, True])
def test_rebind_and_relocation_preserve_native_tar_route_array(tmp_path: Path, relocate: bool) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_manifest = source_root / "manifest.jsonl"
    source_tar = source_root / "audio.tar"
    source_manifest.write_text("a\nb\nc\nd\n")
    source_tar.write_text("A\nB\nC\nD\n")
    create_jsonl_index(source_manifest)
    create_jsonl_index(source_tar)
    route_values = (2, 0, 3, 1)
    route_path = tmp_path / "route.u32"
    route_path.write_bytes(struct.pack("<IIII", *route_values))

    source_manifest_spec = IndexPackCollectionSpec(
        role="manifest",
        kind="jsonl",
        source_spec="old-manifest-declaration",
        paths=(str(source_manifest),),
    )
    source_tar_spec = IndexPackCollectionSpec(
        role="tar",
        kind="nemo_tar",
        source_spec="old-tar-declaration",
        paths=(str(source_tar),),
    )
    source_route = NativeTarOrdinalMapSpec(
        manifest_source_spec=source_manifest_spec.source_spec,
        tar_source_spec=source_tar_spec.source_spec,
        manifest_paths=source_manifest_spec.paths,
        tar_paths=source_tar_spec.paths,
    )
    source_route_array = IndexPackArraySpec(
        role=source_route.role,
        kind=source_route.kind,
        source_spec=nemo_tar_ordinal_map_source_spec(
            source_route.manifest_source_spec,
            source_route.tar_source_spec,
        ),
        shard_paths=(route_path,),
    )
    source = tmp_path / "source.idxpack"
    write_index_pack(source, [source_manifest_spec, source_tar_spec, source_route_array])

    if relocate:
        target_root = tmp_path / "target"
        target_root.mkdir()
        target_manifest = target_root / source_manifest.name
        target_tar = target_root / source_tar.name
        target_manifest.write_bytes(source_manifest.read_bytes())
        target_tar.write_bytes(source_tar.read_bytes())
    else:
        target_manifest = source_manifest
        target_tar = source_tar
    target_manifest_spec = IndexPackCollectionSpec(
        role="manifest",
        kind="jsonl",
        source_spec="new-manifest-declaration",
        paths=(str(target_manifest),),
    )
    target_tar_spec = IndexPackCollectionSpec(
        role="tar",
        kind="nemo_tar",
        source_spec="new-tar-declaration",
        paths=(str(target_tar),),
    )
    target_route = NativeTarOrdinalMapSpec(
        manifest_source_spec=target_manifest_spec.source_spec,
        tar_source_spec=target_tar_spec.source_spec,
        manifest_paths=target_manifest_spec.paths,
        tar_paths=target_tar_spec.paths,
    )
    target = [target_manifest_spec, target_tar_spec, target_route]
    output = tmp_path / "output.idxpack"

    if relocate:
        result = relocate_idxpack_collections(source, output, target)
    else:
        result = rebind_idxpack_collections(source, output, target)

    assert result["keys_changed"] == 3
    with IndexPack(output) as pack:
        route = pack.collection(target_route.key)
        assert route.is_array
        assert [route.value(index) for index in range(len(route_values))] == list(route_values)
        with pytest.raises(KeyError):
            pack.collection(source_route.key)


@pytest.mark.parametrize("relocate", [False, True])
def test_v2_native_tar_rebind_discovery_preserves_route_free_topology(tmp_path: Path, relocate: bool) -> None:
    manifest = tmp_path / "manifest.jsonl"
    tar_path = tmp_path / "audio.tar"
    manifest.write_text('{"audio_filepath": "sample.wav"}\n')
    tar_path.write_text("sample\n")
    create_jsonl_index(manifest)
    create_jsonl_index(tar_path)
    config = {
        "type": "nemo_tarred",
        "manifest_filepath": str(manifest),
        "tarred_audio_filepaths": str(tar_path),
    }
    source_specs = [
        IndexPackCollectionSpec(
            role="manifest",
            kind="jsonl",
            source_spec=str(manifest),
            paths=(str(manifest),),
        ),
        IndexPackCollectionSpec(
            role="tar",
            kind="nemo_tar",
            source_spec=str(tar_path),
            paths=(str(tar_path),),
        ),
    ]
    source = tmp_path / "source.idxpack"
    write_index_pack(source, source_specs)
    observed, _header, _sequences, _segments = _read_pack_layout(source)

    target = discover_rebind_pack_collections(config, observed)

    assert len(target) == 2
    output = tmp_path / "output.idxpack"
    if relocate:
        relocate_idxpack_collections(source, output, target)
    else:
        rebind_idxpack_collections(source, output, target)
    with IndexPack(output) as pack:
        assert pack.version == 2
        assert pack.num_collections == 2


def test_rejects_ordered_path_mismatch(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    other = tmp_path / "other.jsonl"
    records.write_text('{"id": 1}\n')
    other.write_text('{"id": 1}\n')
    source_spec = _spec(records, "/old/location/records.jsonl")
    target_spec = _spec(other, "/new/location/records.jsonl")
    source = tmp_path / "source.idxpack"
    write_index_pack(source, [source_spec])

    with pytest.raises(ValueError, match="ordered paths changed"):
        rebind_idxpack_collections(source, tmp_path / "output.idxpack", [target_spec])


def test_relocates_paths_and_preserves_offset_payloads(tmp_path: Path) -> None:
    source_records = tmp_path / "source" / "records.jsonl"
    target_records = tmp_path / "target" / "records.jsonl"
    source_records.parent.mkdir()
    target_records.parent.mkdir()
    payload = '{"id": 1}\n{"id": 2}\n'
    source_records.write_text(payload)
    target_records.write_text(payload)
    create_jsonl_index(source_records)
    source_spec = IndexPackCollectionSpec(
        role="records",
        kind="json-lines",
        source_spec="source/records.jsonl",
        paths=(str(source_records),),
    )
    target_spec = IndexPackCollectionSpec(
        role="records",
        kind="json-lines",
        source_spec="target/records.jsonl",
        paths=(str(target_records),),
    )
    source = tmp_path / "source.idxpack"
    output = tmp_path / "output.idxpack"
    write_index_pack(source, [source_spec])

    result = relocate_idxpack_collections(source, output, [target_spec])

    assert result["keys_changed"] == 1
    assert result["paths_changed"] == 1
    assert result["payloads_verified"] is True
    with IndexPack(output) as pack:
        collection = pack.collection(target_spec.key)
        assert collection.path_for_shard(0) == str(target_records)
        assert len(collection) == 2
        location = collection.locate(0)
        assert target_records.read_bytes()[location.start : location.end] == b'{"id": 1}\n'
        with pytest.raises(KeyError):
            pack.collection(source_spec.key)


def test_relocation_rejects_changed_payload_with_same_size(tmp_path: Path) -> None:
    source_records = tmp_path / "source.jsonl"
    target_records = tmp_path / "target.jsonl"
    source_records.write_text('{"id": 1}\n')
    target_records.write_text('{"id": 2}\n')
    create_jsonl_index(source_records)
    source_spec = IndexPackCollectionSpec(
        role="records",
        kind="json-lines",
        source_spec="source",
        paths=(str(source_records),),
    )
    target_spec = IndexPackCollectionSpec(
        role="records",
        kind="json-lines",
        source_spec="target",
        paths=(str(target_records),),
    )
    source = tmp_path / "source.idxpack"
    write_index_pack(source, [source_spec])

    with pytest.raises(ValueError, match="payload content mismatch"):
        relocate_idxpack_collections(
            source,
            tmp_path / "output.idxpack",
            [target_spec],
        )


def test_relocation_preserves_shared_segment_identity(tmp_path: Path) -> None:
    source_records = tmp_path / "source.jsonl"
    target_records = tmp_path / "target.jsonl"
    source_records.write_text('{"id": 1}\n')
    target_records.write_text('{"id": 1}\n')
    create_jsonl_index(source_records)
    source_specs = [
        IndexPackCollectionSpec(
            role=role,
            kind="json-lines",
            source_spec={"location": "source", "role": role},
            paths=(str(source_records),),
        )
        for role in ("records-a", "records-b")
    ]
    target_specs = [
        IndexPackCollectionSpec(
            role=role,
            kind="json-lines",
            source_spec={"location": "target", "role": role},
            paths=(str(target_records),),
        )
        for role in ("records-a", "records-b")
    ]
    source = tmp_path / "source.idxpack"
    output = tmp_path / "output.idxpack"
    write_index_pack(source, source_specs)

    result = relocate_idxpack_collections(source, output, target_specs)

    assert result["segments"] == 1
    with IndexPack(output) as pack:
        locations = [pack.collection(spec.key).locate(0) for spec in target_specs]
        assert locations[0].segment_id == locations[1].segment_id
        assert locations[0].path == locations[1].path == str(target_records)


def test_relocation_rejects_inconsistent_shared_mapping(tmp_path: Path) -> None:
    source_records = tmp_path / "source.jsonl"
    source_records.write_text('{"id": 1}\n')
    create_jsonl_index(source_records)
    source_specs = [
        IndexPackCollectionSpec(
            role=role,
            kind="json-lines",
            source_spec=role,
            paths=(str(source_records),),
        )
        for role in ("records-a", "records-b")
    ]
    target_specs = [
        IndexPackCollectionSpec(
            role=role,
            kind="json-lines",
            source_spec=role,
            paths=(str(tmp_path / f"target-{role}.jsonl"),),
        )
        for role in ("records-a", "records-b")
    ]
    source = tmp_path / "source.idxpack"
    write_index_pack(source, source_specs)

    with pytest.raises(ValueError, match="inconsistent target paths"):
        relocate_idxpack_collections(source, tmp_path / "output.idxpack", target_specs)


def test_relocation_rejects_target_path_collision(tmp_path: Path) -> None:
    source_paths = []
    for index in range(2):
        path = tmp_path / f"source-{index}.jsonl"
        path.write_text(f'{{"id": {index}}}\n')
        create_jsonl_index(path)
        source_paths.append(str(path))
    source_spec = IndexPackCollectionSpec(
        role="records",
        kind="json-lines",
        source_spec="source",
        paths=tuple(source_paths),
    )
    target_path = str(tmp_path / "target.jsonl")
    target_spec = IndexPackCollectionSpec(
        role="records",
        kind="json-lines",
        source_spec="target",
        paths=(target_path, target_path),
    )
    source = tmp_path / "source.idxpack"
    write_index_pack(source, [source_spec])

    with pytest.raises(ValueError, match="collapse onto one target identity"):
        relocate_idxpack_collections(source, tmp_path / "output.idxpack", [target_spec])


def test_offline_wds_discovery_uses_explicit_prefix_map(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_tar = source_root / "shard-0.tar"
    source_tar.write_text("row\n")
    create_jsonl_index(source_tar)
    source_spec = IndexPackCollectionSpec(
        role="wds_tar",
        kind="wds_tar_v2",
        source_spec=str(source_root),
        paths=(str(source_tar),),
    )
    source = tmp_path / "source.idxpack"
    write_index_pack(source, [source_spec])
    observed, _header, _sequences, _segments = _read_pack_layout(source)
    target_root = "s3://fixture-bucket/payload/dataset"
    config = [
        {
            "type": "share_gpt_webdataset",
            "data_dir": target_root,
            "wds_sample_index_version": 2,
        }
    ]

    target = discover_relocated_pack_collections(
        config,
        observed,
        path_prefix_map={str(source_root): target_root},
    )

    assert target == [
        IndexPackCollectionSpec(
            role="wds_tar",
            kind="wds_tar_v2",
            source_spec=target_root,
            paths=(f"{target_root}/shard-0.tar",),
        )
    ]


def test_offline_wds_discovery_rejects_unmapped_source(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_tar = source_root / "shard-0.tar"
    source_tar.write_text("row\n")
    create_jsonl_index(source_tar)
    source_spec = IndexPackCollectionSpec(
        role="wds_tar",
        kind="wds_tar_v2",
        source_spec=str(source_root),
        paths=(str(source_tar),),
    )
    source = tmp_path / "source.idxpack"
    write_index_pack(source, [source_spec])
    observed, _header, _sequences, _segments = _read_pack_layout(source)

    with pytest.raises(ValueError, match="No relocation prefix matches"):
        discover_relocated_pack_collections(
            [
                {
                    "type": "share_gpt_webdataset",
                    "data_dir": "s3://fixture-bucket/payload/dataset",
                    "wds_sample_index_version": 2,
                }
            ],
            observed,
            path_prefix_map={str(tmp_path / "different"): "s3://fixture-bucket/payload"},
        )
