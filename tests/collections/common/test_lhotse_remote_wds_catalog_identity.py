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

import io
import json
import struct
import tarfile
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from click.testing import CliRunner
from lhotse.indexing import index_file_path
from scripts.dataloading import build_indexes, convert_indexes_to_idxpack

from nemo.collections.common.data.lhotse.indexed_adapters import create_wds_v2_tar_index, validate_wds_v2_tar_index
from nemo.collections.common.data.lhotse.sharegpt_tar_routing import (
    ShareGptTarRoutingIndex,
    TarRoute,
    build_sharegpt_tar_routing_index,
    ordered_manifest_content_digest,
    ordered_tar_catalog_digest,
)
from nemo.collections.common.data.lhotse.wds_catalog import discover_webdataset_shards


def _tar_bytes(sample_id: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as archive:
        for name, payload in (
            (f"{sample_id}.json", json.dumps({"id": sample_id}).encode()),
            (f"{sample_id}.wav", b"audio"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _native_tar_bytes() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as archive:
        payload = b"audio"
        info = tarfile.TarInfo("audio.wav")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _write_remote_sidecars(indexes_root: Path, manifest: str, tar: str, manifest_bytes: bytes, tar_bytes: bytes):
    manifest_idx = Path(index_file_path(manifest, indexes_root))
    tar_idx = Path(index_file_path(tar, indexes_root))
    manifest_idx.parent.mkdir(parents=True, exist_ok=True)
    tar_idx.parent.mkdir(parents=True, exist_ok=True)
    manifest_idx.write_bytes(struct.pack("<2Q", 0, len(manifest_bytes)))
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
        offsets = [member.offset for member in archive if member.isreg()]
    tar_idx.write_bytes(struct.pack(f"<{len(offsets) + 1}Q", *offsets, len(tar_bytes)))
    return manifest_idx, tar_idx


class _FakeAISRangeReader:
    objects: ClassVar[dict[str, bytes]] = {}
    identities: ClassVar[dict[str, str | None]] = {}
    opened: ClassVar[list[str]] = []

    def __init__(self, url: str):
        self.url = url
        self.position = 0
        type(self).opened.append(url)
        if url not in type(self).objects:
            raise FileNotFoundError(url)

    @property
    def size(self) -> int:
        return len(type(self).objects[self.url])

    @property
    def object_identity(self) -> str | None:
        return type(self).identities.get(self.url)

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.size + offset
        else:
            raise ValueError(whence)
        return self.position

    def read(self, size: int = -1) -> bytes:
        payload = type(self).objects[self.url]
        end = len(payload) if size < 0 else min(len(payload), self.position + size)
        data = payload[self.position : end]
        self.position = end
        return data

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def fake_ais(monkeypatch):
    import lhotse.ais

    _FakeAISRangeReader.objects = {}
    _FakeAISRangeReader.identities = {}
    _FakeAISRangeReader.opened = []
    monkeypatch.setattr(lhotse.ais, "AISRangeReader", _FakeAISRangeReader)
    return _FakeAISRangeReader


def test_remote_wds_discovery_reads_only_explicit_bounded_catalog(fake_ais):
    root = "s3://bucket/dataset"
    fake_ais.objects[f"{root}/wids-meta.json"] = json.dumps(
        {
            "shardlist": [
                {"url": "shards/part-{0..1}.tar"},
                {"url": "ais://other-bucket/final.tar"},
            ]
        }
    ).encode()

    assert discover_webdataset_shards(root, require_catalog=True) == [
        "s3://bucket/dataset/shards/part-0.tar",
        "s3://bucket/dataset/shards/part-1.tar",
        "ais://other-bucket/final.tar",
    ]
    assert fake_ais.opened == [f"{root}/wids-meta.json"]


def test_remote_wds_discovery_supports_direct_generated_ordered_catalog(fake_ais):
    catalog = "ais://bucket/control/leaf.wds-catalog.json"
    fake_ais.objects[catalog] = json.dumps(
        {
            "format": "nemo-wds-shard-catalog",
            "version": 1,
            "shards": [
                {"url": "../payload/b.tar"},
                {"url": "../payload/a.tar"},
            ],
        }
    ).encode()

    assert discover_webdataset_shards(catalog, require_catalog=True) == [
        "ais://bucket/payload/b.tar",
        "ais://bucket/payload/a.tar",
    ]
    assert fake_ais.opened == [catalog]


def test_remote_wds_discovery_fails_closed_without_catalog(fake_ais):
    with pytest.raises(FileNotFoundError, match="bounded shard catalog"):
        discover_webdataset_shards("s3://bucket/no-catalog", require_catalog=False)

    assert fake_ais.opened == [
        "s3://bucket/no-catalog/wids-meta.json",
        "s3://bucket/no-catalog/.nv-meta/split.yaml",
        "s3://bucket/no-catalog/wds-catalog.json",
    ]


def test_remote_wds_v2_build_uses_range_reader_and_rejects_same_size_replacement(tmp_path, fake_ais):
    remote = "s3://bucket/shard.tar"
    original = _tar_bytes("0")
    replacement = _tar_bytes("1")
    assert len(original) == len(replacement)
    fake_ais.objects[remote] = original
    fake_ais.identities[remote] = "etag:old"
    idx_path = tmp_path / "shard.wds-v2.idx"

    _, metadata_path = create_wds_v2_tar_index(remote, idx_path=idx_path)
    _, metadata = validate_wds_v2_tar_index(remote, idx_path=idx_path)
    assert metadata["source"]["object_identity"] == "etag:old"

    metadata["source"]["object_identity"] = None
    metadata_path.write_text(json.dumps(metadata, separators=(",", ":"), sort_keys=True))
    with pytest.raises(ValueError, match="missing a stable object_identity"):
        validate_wds_v2_tar_index(remote, idx_path=idx_path)
    metadata["source"]["object_identity"] = "etag:old"
    metadata_path.write_text(json.dumps(metadata, separators=(",", ":"), sort_keys=True))

    fake_ais.objects[remote] = replacement
    fake_ais.identities[remote] = "etag:new"
    with pytest.raises(ValueError, match="object_identity mismatch"):
        validate_wds_v2_tar_index(remote, idx_path=idx_path)


def test_remote_wds_v2_fails_closed_without_stable_identity(tmp_path, fake_ais):
    remote = "ais://bucket/no-identity.tar"
    fake_ais.objects[remote] = _tar_bytes("0")

    with pytest.raises(ValueError, match="stable object identity"):
        create_wds_v2_tar_index(remote, idx_path=tmp_path / "missing.wds-v2.idx")


def test_remote_catalog_flows_through_builder_and_converter_discovery(fake_ais):
    root = "s3://bucket/dataset"
    fake_ais.objects[f"{root}/wids-meta.json"] = json.dumps(
        {"shardlist": [{"url": "shards/part-0.tar"}, {"url": "shards/part-1.tar"}]}
    ).encode()
    entry = {
        "type": "share_gpt_webdataset",
        "data_dir": root,
        "wds_sample_index_version": 2,
    }

    jobs = []
    build_indexes.discover(entry, jobs, indexes_root="/tmp/indexes")
    collections = convert_indexes_to_idxpack.discover_pack_collections(entry)

    expected = (
        "s3://bucket/dataset/shards/part-0.tar",
        "s3://bucket/dataset/shards/part-1.tar",
    )
    assert tuple(job.path for job in jobs) == expected
    assert collections[0].paths == expected


def test_remote_catalog_rejects_traversal_outside_uri_authority(fake_ais):
    root = "s3://bucket/dataset"
    fake_ais.objects[f"{root}/wids-meta.json"] = json.dumps({"shardlist": [{"url": "../../../escape.tar"}]}).encode()

    with pytest.raises(ValueError, match="traversal"):
        discover_webdataset_shards(root, require_catalog=True)


def test_remote_sharegpt_route_build_uses_bounded_sources_and_local_sidecars(tmp_path, fake_ais):
    manifest = "s3://bucket/rows.jsonl"
    tar = "ais://bucket/audio.tar"
    manifest_bytes = (
        json.dumps(
            {
                "sound": "audio.wav",
                "conversations": [
                    {"from": "human", "value": "<sound>"},
                    {"from": "gpt", "value": "ok"},
                ],
            }
        ).encode()
        + b"\n"
    )
    tar_bytes = _native_tar_bytes()
    fake_ais.objects.update({manifest: manifest_bytes, tar: tar_bytes})
    fake_ais.identities.update({manifest: "etag:manifest-v1", tar: "etag:tar-v1"})
    manifest_idx, tar_idx = _write_remote_sidecars(tmp_path / "indexes", manifest, tar, manifest_bytes, tar_bytes)
    route_path = tmp_path / "remote.sgroute"

    build_sharegpt_tar_routing_index(
        route_path,
        manifest_paths=[manifest],
        tar_paths=[tar],
        manifest_index_paths=[manifest_idx],
        tar_index_paths=[tar_idx],
    )

    with ShareGptTarRoutingIndex(route_path) as routing:
        assert routing.routes_for_row(0) == (TarRoute(0, 0),)
        assert routing.header.manifest_content_digest == ordered_manifest_content_digest([manifest])
        assert routing.header.tar_catalog_digest == ordered_tar_catalog_digest([tar])

    fake_ais.identities[tar] = "etag:tar-v2"
    assert ordered_tar_catalog_digest([tar]) != routing.header.tar_catalog_digest


def test_converter_builds_remote_sharegpt_route_from_local_sidecars(tmp_path, fake_ais):
    manifest = "s3://bucket/converter-rows.jsonl"
    tar = "s3://bucket/converter-audio.tar"
    manifest_bytes = (
        json.dumps(
            {
                "sound": "audio.wav",
                "conversations": [{"from": "human", "value": "<sound>"}],
            }
        ).encode()
        + b"\n"
    )
    tar_bytes = _native_tar_bytes()
    fake_ais.objects.update({manifest: manifest_bytes, tar: tar_bytes})
    fake_ais.identities.update({manifest: "etag:manifest", tar: "etag:tar"})
    indexes_root = tmp_path / "indexes"
    _write_remote_sidecars(indexes_root, manifest, tar, manifest_bytes, tar_bytes)
    route_path = tmp_path / "converter.sgroute"
    config_path = tmp_path / "remote-sharegpt.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "type": "share_gpt",
                "manifest_filepath": manifest,
                "tarred_audio_filepaths": tar,
                "tar_lookup_mode": "collection",
                "tar_routing_filepath": str(route_path),
            }
        )
    )

    result = CliRunner().invoke(
        convert_indexes_to_idxpack.main,
        [
            "--indexes-root",
            str(indexes_root),
            "--output",
            str(tmp_path / "remote.idxpack"),
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert route_path.is_file()
