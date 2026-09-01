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

import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from lhotse.index_pack import IndexPack, IndexPackCollectionSpec, write_index_pack
from lhotse.indexing import create_tar_index as create_legacy_wds_tar_index
from nemo.collections.common.data.lhotse.indexed_adapters import (
    IndexedTarSampleBundleReader,
    IndexedTarSampleReader,
    PackedTarSampleBundleReader,
    create_wds_v2_tar_index,
    read_exact_range,
    validate_wds_v2_tar_index,
    wds_sample_key,
    wds_v2_index_path,
    wds_v2_metadata_path,
)


def _json(sample_id: str, **extra) -> bytes:
    return json.dumps({"id": sample_id, **extra}).encode("utf-8")


def _write_tar(
    path: Path, members: list[tuple[str, bytes | None, bytes | None]]
) -> Path:
    """Write ``(name, payload, type)`` entries; ``payload=None`` means no data."""
    with tarfile.open(path, "w:") as archive:
        for name, payload, member_type in members:
            info = tarfile.TarInfo(name)
            if member_type is not None:
                info.type = member_type
            if payload is not None:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            else:
                archive.addfile(info)
    return path


def _regular(name: str, payload: bytes) -> tuple[str, bytes, None]:
    return name, payload, None


def test_wds_v2_conventional_pair_offsets_match_legacy_byte_for_byte(tmp_path):
    tar_path = _write_tar(
        tmp_path / "pairs.tar",
        [
            _regular("0.json", _json("zero")),
            _regular("0.wav", b"audio-zero"),
            _regular("1.wav", b"audio-one"),
            _regular("1.json", _json("one")),
        ],
    )
    legacy_idx = tmp_path / "legacy.idx"
    create_legacy_wds_tar_index(tar_path, output_path=legacy_idx)

    idx_path, metadata_path = create_wds_v2_tar_index(tar_path)

    assert idx_path.read_bytes() == legacy_idx.read_bytes()
    assert idx_path == Path(f"{tar_path}.wds-v2.idx")
    assert metadata_path == Path(f"{tar_path}.wds-v2.idx.meta.json")
    metadata_bytes = metadata_path.read_bytes()
    assert metadata_bytes == json.dumps(
        json.loads(metadata_bytes),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    offsets, metadata = validate_wds_v2_tar_index(tar_path)
    assert len(offsets) == 3
    assert metadata["sample_count"] == 2
    assert metadata["regular_member_count"] == 4

    # The legacy reader and API remain available and unchanged.
    legacy_sample = IndexedTarSampleReader(tar_path, legacy_idx)[0]
    assert legacy_sample.json_data["id"] == "zero"
    assert legacy_sample.audio_name == "0.wav"
    assert legacy_sample.audio_bytes == b"audio-zero"


@pytest.mark.parametrize(
    "members,expected_audio_names",
    [
        (
            [
                _regular("dir.v1/1.json", _json("sample-1")),
                _regular("dir.v1/1.flac", b"first"),
                _regular("dir.v1/1.1.flac", b"second"),
            ],
            ("dir.v1/1.flac", "dir.v1/1.1.flac"),
        ),
        (
            [
                _regular("dir.v1/1.1.flac", b"second"),
                _regular("dir.v1/1.json", _json("sample-1")),
                _regular("dir.v1/1.flac", b"first"),
            ],
            ("dir.v1/1.1.flac", "dir.v1/1.flac"),
        ),
    ],
)
def test_wds_v2_variable_members_preserve_physical_audio_order(
    tmp_path, members, expected_audio_names
):
    tar_path = _write_tar(tmp_path / "variable.tar", members)
    create_wds_v2_tar_index(tar_path)

    sample = IndexedTarSampleBundleReader(tar_path)[0]

    assert sample.sample_key == "dir.v1/1"
    assert sample.json_data["id"] == "sample-1"
    assert tuple(member.name for member in sample.audio_members) == expected_audio_names
    assert {member.name: member.data for member in sample.audio_members} == {
        "dir.v1/1.flac": b"first",
        "dir.v1/1.1.flac": b"second",
    }


def test_wds_v2_ignores_nonregular_members(tmp_path):
    tar_path = _write_tar(
        tmp_path / "nonregular.tar",
        [
            ("ignored-dir", None, tarfile.DIRTYPE),
            ("ignored-link", None, tarfile.SYMTYPE),
            _regular("0.json", _json("zero")),
            _regular("0.wav", b"zero"),
            ("between", None, tarfile.DIRTYPE),
            _regular("1.wav", b"one"),
            _regular("1.json", _json("one")),
        ],
    )

    create_wds_v2_tar_index(tar_path)
    reader = IndexedTarSampleBundleReader(tar_path)

    assert len(reader) == 2
    assert [reader[idx].sample_key for idx in range(2)] == ["0", "1"]
    assert [reader[idx].json_data["id"] for idx in range(2)] == ["zero", "one"]


def test_wds_v2_supports_extended_tar_names(tmp_path):
    prefix = "nested/" + "long-segment-" * 10 + "sample"
    tar_path = _write_tar(
        tmp_path / "long-names.tar",
        [
            _regular(f"{prefix}.json", _json("long")),
            _regular(f"{prefix}.flac", b"long-audio"),
        ],
    )

    create_wds_v2_tar_index(tar_path)
    sample = IndexedTarSampleBundleReader(tar_path)[0]

    assert sample.sample_key == prefix
    assert sample.json_data["id"] == "long"
    assert sample.audio_members[0].name == f"{prefix}.flac"


def test_packed_wds_v2_reader_supports_global_and_shard_local_access(tmp_path):
    tar_paths = (
        _write_tar(
            tmp_path / "first.tar",
            [_regular("0.json", _json("zero")), _regular("0.wav", b"zero")],
        ),
        _write_tar(
            tmp_path / "second.tar",
            [
                _regular("1.wav", b"one"),
                _regular("1.json", _json("one")),
                _regular("2.json", _json("two")),
                _regular("2.flac", b"two-a"),
                _regular("2.1.flac", b"two-b"),
            ],
        ),
    )
    index_paths = {str(path): create_wds_v2_tar_index(path)[0] for path in tar_paths}
    spec = IndexPackCollectionSpec(
        role="wds_tar",
        kind="wds_tar_v2",
        source_spec=[str(path) for path in tar_paths],
        paths=tuple(str(path) for path in tar_paths),
    )
    pack_path = tmp_path / "wds.idxpack"
    write_index_pack(pack_path, [spec], index_path_overrides=index_paths)

    with IndexPack(pack_path) as pack:
        reader = PackedTarSampleBundleReader(
            pack.collection(spec.key), max_open_files=1
        )

        assert len(reader) == 3
        assert reader.path_for_shard(1) == str(tar_paths[1])
        assert reader.shard_length(0) == 1
        assert reader.shard_length(1) == 2
        assert reader[0].json_data["id"] == "zero"
        assert reader.read_shard(1, 0).json_data["id"] == "one"
        sample, location = reader.read_with_location(-1)
        assert sample.sample_key == "2"
        assert [member.data for member in sample.audio_members] == [b"two-a", b"two-b"]
        assert location.path == str(tar_paths[1])
        assert location.shard_index == 1
        assert location.local_index == 1


@pytest.mark.parametrize(
    "kind,offsets_required,error",
    [
        ("wds_tar", True, "Expected a wds_tar_v2 collection"),
        ("wds_tar_v2", False, "must contain sample offsets"),
    ],
)
def test_packed_wds_v2_reader_rejects_wrong_collection_contract(
    kind, offsets_required, error
):
    collection = type(
        "Collection",
        (),
        {"kind": kind, "offsets_required": offsets_required},
    )()

    with pytest.raises(ValueError, match=error):
        PackedTarSampleBundleReader(collection)


def test_packed_wds_v2_reader_uses_remote_exact_range(tmp_path, monkeypatch):
    tar_path = _write_tar(
        tmp_path / "remote-source.tar",
        [_regular("0.json", _json("remote")), _regular("0.wav", b"remote-audio")],
    )
    create_wds_v2_tar_index(tar_path)
    offsets, _ = validate_wds_v2_tar_index(tar_path)
    tar_bytes = tar_path.read_bytes()

    class FakeAISRangeReader:
        def __init__(self, url):
            assert url == "s3://bucket/remote-source.tar"
            self.position = 0

        @property
        def size(self):
            return len(tar_bytes)

        def seek(self, offset):
            self.position = offset

        def read(self, size):
            data = tar_bytes[self.position : self.position + size]
            self.position += len(data)
            return data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    import lhotse.ais

    monkeypatch.setattr(lhotse.ais, "AISRangeReader", FakeAISRangeReader)
    location = SimpleNamespace(
        path="s3://bucket/remote-source.tar",
        start=int(offsets[0]),
        end=int(offsets[1]),
    )

    class RemoteCollection:
        kind = "wds_tar_v2"
        offsets_required = True

        def __len__(self):
            return 1

        def locate(self, idx):
            assert idx == 0
            return location

    sample = PackedTarSampleBundleReader(RemoteCollection())[0]

    assert sample.json_data["id"] == "remote"
    assert sample.audio_members[0].data == b"remote-audio"


def test_wds_v2_reader_rejects_a_range_that_mixes_sample_keys(tmp_path):
    tar_path = _write_tar(
        tmp_path / "mixed-range.tar",
        [
            _regular("0.json", _json("zero")),
            _regular("0.wav", b"zero"),
            _regular("1.json", _json("one")),
            _regular("1.wav", b"one"),
        ],
    )
    idx_path, metadata_path = create_wds_v2_tar_index(tar_path)
    raw_offsets = (0).to_bytes(8, "little") + tar_path.stat().st_size.to_bytes(
        8, "little"
    )
    idx_path.write_bytes(raw_offsets)
    metadata = json.loads(metadata_path.read_bytes())
    metadata["sample_count"] = 1
    metadata["offsets_sha256"] = hashlib.sha256(raw_offsets).hexdigest()
    metadata_path.write_bytes(
        json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    )

    with pytest.raises(ValueError, match="mixes indexed key"):
        IndexedTarSampleBundleReader(tar_path)[0]


@pytest.mark.parametrize(
    "members,error",
    [
        (
            [
                _regular("0.json", _json("zero")),
                _regular("0.wav", b"zero"),
                _regular("1.json", _json("one")),
                _regular("1.wav", b"one"),
                _regular("0.1.flac", b"late"),
            ],
            "non-contiguous reuse",
        ),
        (
            [
                _regular("0.json", _json("zero")),
                _regular("0.wav", b"zero"),
                _regular("0.wav", b"duplicate"),
            ],
            "duplicate regular member name",
        ),
        ([_regular("0.wav", b"zero")], "exactly one .json"),
        (
            [
                _regular("0.json", _json("zero")),
                _regular("0.meta.json", _json("zero-meta")),
                _regular("0.wav", b"zero"),
            ],
            "exactly one .json",
        ),
        ([_regular("0.json", _json("zero"))], "at least one non-JSON payload"),
        ([_regular("0.json", b"{"), _regular("0.wav", b"zero")], "malformed JSON"),
        ([_regular("0.json", b"[]"), _regular("0.wav", b"zero")], "JSON object"),
    ],
)
def test_wds_v2_builder_rejects_malformed_samples(tmp_path, members, error):
    tar_path = _write_tar(tmp_path / "bad.tar", members)

    with pytest.raises(ValueError, match=error):
        create_wds_v2_tar_index(tar_path)

    assert not Path(f"{tar_path}.wds-v2.idx").exists()
    assert not Path(f"{tar_path}.wds-v2.idx.meta.json").exists()


@pytest.mark.parametrize(
    "name,error",
    [
        ("/absolute/0.json", "absolute"),
        ("../0.json", "forbidden path component"),
        ("dir/./0.json", "forbidden path component"),
        ("0", "without an extension"),
        (".json", "empty prefix"),
    ],
)
def test_wds_sample_key_rejects_unsafe_or_malformed_names(name, error):
    with pytest.raises(ValueError, match=error):
        wds_sample_key(name)


def test_wds_v2_validator_rejects_stale_source_and_corrupt_offsets(tmp_path):
    tar_path = _write_tar(
        tmp_path / "stale.tar",
        [_regular("0.json", _json("zero")), _regular("0.wav", b"zero")],
    )
    idx_path, _ = create_wds_v2_tar_index(tar_path)

    stat = tar_path.stat()
    os.utime(tar_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    with pytest.raises(ValueError, match="mtime_ns"):
        validate_wds_v2_tar_index(tar_path)

    os.utime(tar_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    idx_path.write_bytes(
        idx_path.read_bytes()[:-8] + (tar_path.stat().st_size - 1).to_bytes(8, "little")
    )
    with pytest.raises(ValueError, match="offsets_sha256"):
        validate_wds_v2_tar_index(tar_path)


def test_wds_v2_path_helpers_support_index_mirrors(tmp_path):
    source = tmp_path / "source" / "data.tar"
    mirror = tmp_path / "indexes"
    expected = mirror / source.relative_to(source.anchor)
    expected = Path(f"{expected}.wds-v2.idx")

    assert wds_v2_index_path(source, mirror) == expected
    assert wds_v2_metadata_path(expected) == Path(f"{expected}.meta.json")


def test_read_exact_range_local_and_ais(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"0123456789")
    assert read_exact_range(source, 2, 7) == b"23456"

    class FakeAISRangeReader:
        def __init__(self, url):
            assert url == "s3://bucket/source.bin"
            self.payload = b"abcdefghij"
            self.position = 0

        @property
        def size(self):
            return len(self.payload)

        def seek(self, offset):
            self.position = offset

        def read(self, size):
            data = self.payload[self.position : self.position + size]
            self.position += len(data)
            return data

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    import lhotse.ais

    monkeypatch.setattr(lhotse.ais, "AISRangeReader", FakeAISRangeReader)
    assert read_exact_range("s3://bucket/source.bin", 3, 8) == b"defgh"

    with pytest.raises(ValueError, match="Unsupported remote range-read scheme"):
        read_exact_range("https://example.com/source.bin", 0, 1)


def test_wds_v2_builder_does_not_overwrite_sealed_sidecars(tmp_path):
    tar_path = _write_tar(
        tmp_path / "sealed.tar",
        [_regular("0.json", _json("zero")), _regular("0.wav", b"zero")],
    )
    create_wds_v2_tar_index(tar_path)

    with pytest.raises(FileExistsError, match="already exists"):
        create_wds_v2_tar_index(tar_path)
