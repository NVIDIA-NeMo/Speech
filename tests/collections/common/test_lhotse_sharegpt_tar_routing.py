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
import pickle
import struct
import tarfile
from pathlib import Path

import pytest
from lhotse.indexing import create_jsonl_index

from nemo.collections.common.data.lhotse.sharegpt_tar_routing import (
    SGROUTE_FLAGS,
    SGROUTE_HEADER_SIZE,
    SGROUTE_MAGIC,
    SGROUTE_RECORD_SIZE,
    SGROUTE_VERSION,
    ShareGptTarRoutingIndex,
    TarRoute,
    _atomic_publish_no_replace,
    _validate_native_tar_ranges,
    build_sharegpt_tar_routing_index,
    canonical_audio_prefix_map_digest,
    extract_sharegpt_audio_paths,
    ordered_manifest_content_digest,
    ordered_manifest_source_identity_digest,
    ordered_manifest_spec_path_digest,
    ordered_tar_catalog_digest,
    validate_native_tar_member_index,
    validate_sharegpt_tar_routing_index,
    write_sharegpt_tar_routing_index,
)


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    create_jsonl_index(path)
    return path


def _write_tar(path: Path, members: list[tuple[str, bytes]], *, directory: str | None = None) -> Path:
    with tarfile.open(path, "w:") as archive:
        if directory is not None:
            info = tarfile.TarInfo(directory)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    with tarfile.open(path, "r:") as archive:
        offsets = [member.offset for member in archive if member.isreg()]
    offsets.append(path.stat().st_size)
    Path(f"{path}.idx").write_bytes(struct.pack(f"<{len(offsets)}Q", *offsets))
    return path


def _conversation(placeholders: str) -> list[dict[str, str]]:
    return [
        {"from": "human", "value": placeholders},
        {"from": "gpt", "value": "answer"},
    ]


def _digests():
    return {
        "manifest_spec_path_digest": bytes(range(32)),
        "manifest_content_digest": bytes(range(32, 64)),
        "manifest_source_identity_digest": bytes(range(64, 96)),
        "tar_catalog_digest": bytes(range(96, 128)),
        "audio_prefix_map_digest": bytes(range(128, 160)),
    }


def _refresh_payload_digest(raw: bytearray) -> None:
    raw[240:272] = hashlib.sha256(raw[SGROUTE_HEADER_SIZE:]).digest()


def test_sgroute_writer_has_exact_header_layout_and_is_deterministic(tmp_path):
    row_routes = [[TarRoute(2, 5)], [], [TarRoute(0, 1), TarRoute(1, 3)]]
    first = tmp_path / "first.sgroute"
    second = tmp_path / "second.sgroute"

    write_sharegpt_tar_routing_index(first, row_routes, tar_shard_count=3, **_digests())
    write_sharegpt_tar_routing_index(second, row_routes, tar_shard_count=3, **_digests())

    raw = first.read_bytes()
    assert raw == second.read_bytes()
    assert raw[:8] == SGROUTE_MAGIC == b"SGROUTE2"
    assert struct.unpack_from("<I", raw, 8)[0] == SGROUTE_VERSION == 2
    assert struct.unpack_from("<I", raw, 12)[0] == SGROUTE_HEADER_SIZE == 288
    assert struct.unpack_from("<I", raw, 16)[0] == SGROUTE_FLAGS == 3
    assert struct.unpack_from("<I", raw, 20)[0] == 0
    assert struct.unpack_from("<Q", raw, 24)[0] == 3
    assert struct.unpack_from("<Q", raw, 32)[0] == 3
    assert struct.unpack_from("<Q", raw, 40)[0] == 3
    assert struct.unpack_from("<Q", raw, 48)[0] == SGROUTE_HEADER_SIZE
    assert struct.unpack_from("<Q", raw, 56)[0] == SGROUTE_HEADER_SIZE + 8 * 4
    assert struct.unpack_from("<Q", raw, 64)[0] == len(raw)
    assert struct.unpack_from("<I", raw, 72)[0] == SGROUTE_RECORD_SIZE == 8
    assert struct.unpack_from("<I", raw, 76)[0] == 0
    assert raw[80:112] == _digests()["manifest_spec_path_digest"]
    assert raw[112:144] == _digests()["manifest_content_digest"]
    assert raw[144:176] == _digests()["manifest_source_identity_digest"]
    assert raw[176:208] == _digests()["tar_catalog_digest"]
    assert raw[208:240] == _digests()["audio_prefix_map_digest"]
    assert raw[240:272] == hashlib.sha256(raw[SGROUTE_HEADER_SIZE:]).digest()
    assert raw[272:288] == bytes(16)
    assert struct.unpack_from("<4Q", raw, SGROUTE_HEADER_SIZE) == (0, 1, 1, 3)
    assert struct.unpack_from("<6I", raw, SGROUTE_HEADER_SIZE + 32) == (
        2,
        5,
        0,
        1,
        1,
        3,
    )

    with ShareGptTarRoutingIndex(first) as routes:
        assert len(routes) == 3
        assert routes.routes_for_row(0) == (TarRoute(2, 5),)
        assert routes.routes_for_row(1) == ()
        assert routes.routes_for_row(2) == (TarRoute(0, 1), TarRoute(1, 3))


def test_sgroute_reader_reopens_after_worker_pickling(tmp_path):
    path = tmp_path / "routes.sgroute"
    write_sharegpt_tar_routing_index(path, [[TarRoute(0, 2)], []], tar_shard_count=1, **_digests())

    original = ShareGptTarRoutingIndex(path)
    restored = pickle.loads(pickle.dumps(original))
    original.close()
    try:
        assert len(restored) == 2
        assert restored.routes_for_row(0) == (TarRoute(0, 2),)
        assert restored.routes_for_row(1) == ()
    finally:
        restored.close()


def test_sgroute_validator_checks_external_digests_counts_and_offset_bearing_requirement(
    tmp_path,
):
    path = tmp_path / "routes.sgroute"
    write_sharegpt_tar_routing_index(path, [[TarRoute(0, 7)]], tar_shard_count=1, **_digests())

    header = validate_sharegpt_tar_routing_index(
        path,
        expected_manifest_row_count=1,
        expected_manifest_spec_path_digest=_digests()["manifest_spec_path_digest"],
        expected_manifest_content_digest=_digests()["manifest_content_digest"],
        expected_manifest_source_identity_digest=_digests()["manifest_source_identity_digest"],
        expected_tar_shard_count=1,
        expected_tar_catalog_digest=_digests()["tar_catalog_digest"],
        expected_audio_prefix_map_digest=_digests()["audio_prefix_map_digest"],
        offset_bearing_tar_collections=True,
    )
    assert header.route_record_count == 1

    with pytest.raises(ValueError, match="offset-bearing"):
        validate_sharegpt_tar_routing_index(path, offset_bearing_tar_collections=False)
    with pytest.raises(ValueError, match="manifest row count"):
        validate_sharegpt_tar_routing_index(path, expected_manifest_row_count=2, offset_bearing_tar_collections=True)
    with pytest.raises(ValueError, match="manifest content digest"):
        validate_sharegpt_tar_routing_index(
            path,
            expected_manifest_content_digest=bytes(32),
            offset_bearing_tar_collections=True,
        )


@pytest.mark.parametrize(
    "offset,value,error",
    [
        (0, b"BADMAGIC", "magic"),
        (8, struct.pack("<I", 3), "version"),
        (12, struct.pack("<I", 128), "header size"),
        (16, struct.pack("<I", 7), "flags"),
        (20, struct.pack("<I", 1), "reserved"),
        (72, struct.pack("<I", 16), "route-record size"),
        (272, b"x" + bytes(15), "reserved"),
    ],
)
def test_sgroute_reader_rejects_invalid_header_constants(tmp_path, offset, value, error):
    path = tmp_path / "routes.sgroute"
    write_sharegpt_tar_routing_index(path, [[TarRoute(0, 0)]], tar_shard_count=1, **_digests())
    raw = bytearray(path.read_bytes())
    raw[offset : offset + len(value)] = value
    path.write_bytes(raw)

    with pytest.raises(ValueError, match=error):
        ShareGptTarRoutingIndex(path)


def test_sgroute_reader_rejects_payload_corruption_and_extent_mismatch(tmp_path):
    corrupt = tmp_path / "corrupt.sgroute"
    write_sharegpt_tar_routing_index(corrupt, [[TarRoute(0, 0)]], tar_shard_count=1, **_digests())
    raw = bytearray(corrupt.read_bytes())
    raw[-1] ^= 1
    corrupt.write_bytes(raw)
    with pytest.raises(ValueError, match="payload SHA-256"):
        ShareGptTarRoutingIndex(corrupt)

    truncated = tmp_path / "truncated.sgroute"
    write_sharegpt_tar_routing_index(truncated, [[TarRoute(0, 0)]], tar_shard_count=1, **_digests())
    truncated.write_bytes(truncated.read_bytes()[:-1])
    with pytest.raises(ValueError, match="file size"):
        ShareGptTarRoutingIndex(truncated)


def test_sgroute_reader_rejects_invalid_csr_and_route_shard_after_payload_rehash(
    tmp_path,
):
    terminal = tmp_path / "terminal.sgroute"
    write_sharegpt_tar_routing_index(terminal, [[TarRoute(0, 0)]], tar_shard_count=1, **_digests())
    raw = bytearray(terminal.read_bytes())
    struct.pack_into("<Q", raw, SGROUTE_HEADER_SIZE + 8, 0)
    _refresh_payload_digest(raw)
    terminal.write_bytes(raw)
    with pytest.raises(ValueError, match="terminal row offset"):
        ShareGptTarRoutingIndex(terminal)

    monotonic = tmp_path / "monotonic.sgroute"
    write_sharegpt_tar_routing_index(
        monotonic,
        [[TarRoute(0, 0)], [], []],
        tar_shard_count=1,
        **_digests(),
    )
    raw = bytearray(monotonic.read_bytes())
    struct.pack_into("<Q", raw, SGROUTE_HEADER_SIZE + 2 * 8, 0)
    _refresh_payload_digest(raw)
    monotonic.write_bytes(raw)
    with pytest.raises(ValueError, match="monotonic"):
        ShareGptTarRoutingIndex(monotonic)

    shard = tmp_path / "shard.sgroute"
    write_sharegpt_tar_routing_index(shard, [[TarRoute(0, 0)]], tar_shard_count=1, **_digests())
    raw = bytearray(shard.read_bytes())
    routes_offset = struct.unpack_from("<Q", raw, 56)[0]
    struct.pack_into("<I", raw, routes_offset, 1)
    _refresh_payload_digest(raw)
    shard.write_bytes(raw)
    with pytest.raises(ValueError, match="catalog has 1 shards"):
        ShareGptTarRoutingIndex(shard)


def test_sgroute_writer_never_overwrites_a_sealed_route(tmp_path):
    path = tmp_path / "sealed.sgroute"
    write_sharegpt_tar_routing_index(path, [[TarRoute(0, 0)]], tar_shard_count=1, **_digests())
    original = path.read_bytes()

    with pytest.raises(FileExistsError, match="sealed"):
        write_sharegpt_tar_routing_index(path, [[]], tar_shard_count=1, **_digests())

    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_atomic_publish_falls_back_when_filesystem_rejects_rename_noreplace(tmp_path, monkeypatch):
    class UnsupportedRenameNoReplace:
        argtypes = None
        restype = None

        def __call__(self, *_args):
            import ctypes
            import errno

            ctypes.set_errno(errno.EINVAL)
            return -1

    class UnsupportedLibc:
        renameat2 = UnsupportedRenameNoReplace()

    monkeypatch.setattr(
        "nemo.collections.common.data.lhotse.sharegpt_tar_routing.ctypes.CDLL",
        lambda *_args, **_kwargs: UnsupportedLibc(),
    )
    source = tmp_path / ".route.tmp"
    destination = tmp_path / "route.sgroute"
    source.write_bytes(b"sealed")

    _atomic_publish_no_replace(source, destination)

    assert not source.exists()
    assert destination.read_bytes() == b"sealed"


def test_builder_preserves_manifest_and_audio_path_order_with_exact_then_basename_resolution(
    tmp_path,
):
    tar_a = _write_tar(
        tmp_path / "a.tar",
        [("nested/exact.flac", b"a"), ("other/exact.flac", b"b")],
        directory="ignored/",
    )
    tar_b = _write_tar(tmp_path / "b.tar", [("unique.wav", b"c"), ("z.wav", b"d")])
    manifest_a = _write_manifest(
        tmp_path / "a.jsonl",
        [
            {"sound": "nested/exact.flac", "conversations": _conversation("<sound>")},
            {
                "speech": "/foreign/unique.wav",
                "conversations": _conversation("<speech>"),
            },
        ],
    )
    manifest_b = _write_manifest(
        tmp_path / "b.jsonl",
        [
            {
                "sound": ["z.wav", "other/exact.flac"],
                "conversations": _conversation("<sound> then <sound>"),
            }
        ],
    )
    output = tmp_path / "ordered.sgroute"

    build_sharegpt_tar_routing_index(
        output,
        manifest_paths=[manifest_b, manifest_a],
        manifest_specs=["declared-b", "declared-a"],
        tar_paths=[tar_a, tar_b],
        audio_prefix_map={},
    )

    with ShareGptTarRoutingIndex(output) as routes:
        assert [routes.routes_for_row(row) for row in range(len(routes))] == [
            (TarRoute(1, 1), TarRoute(0, 1)),
            (TarRoute(0, 0),),
            (TarRoute(1, 0),),
        ]
        assert routes.header.manifest_spec_path_digest == ordered_manifest_spec_path_digest(
            [manifest_b, manifest_a], ["declared-b", "declared-a"]
        )
        assert routes.header.manifest_content_digest == ordered_manifest_content_digest([manifest_b, manifest_a])
        assert routes.header.manifest_source_identity_digest == ordered_manifest_source_identity_digest(
            [manifest_b, manifest_a]
        )
        assert routes.header.tar_catalog_digest == ordered_tar_catalog_digest([tar_a, tar_b])
        assert routes.header.audio_prefix_map_digest == canonical_audio_prefix_map_digest({})


@pytest.mark.parametrize(
    "reference,error",
    [
        ("/foreign/shared.flac", "ambiguous basename"),
        ("missing.flac", "missing audio member"),
    ],
)
def test_builder_rejects_ambiguous_or_missing_basename_resolution(tmp_path, reference, error):
    tar_a = _write_tar(tmp_path / "a.tar", [("left/shared.flac", b"a")])
    tar_b = _write_tar(tmp_path / "b.tar", [("right/shared.flac", b"b")])
    manifest = _write_manifest(
        tmp_path / "data.jsonl",
        [{"sound": reference, "conversations": _conversation("<sound>")}],
    )
    output = tmp_path / "bad.sgroute"

    with pytest.raises(ValueError, match=error):
        build_sharegpt_tar_routing_index(output, manifest_paths=[manifest], tar_paths=[tar_a, tar_b])

    assert not output.exists()


def test_builder_rejects_duplicate_original_tar_member_names(tmp_path):
    tar_a = _write_tar(tmp_path / "a.tar", [("same.flac", b"a")])
    tar_b = _write_tar(tmp_path / "b.tar", [("same.flac", b"b")])
    manifest = _write_manifest(
        tmp_path / "data.jsonl",
        [{"sound": "same.flac", "conversations": _conversation("<sound>")}],
    )

    with pytest.raises(ValueError, match="duplicate tar member name"):
        build_sharegpt_tar_routing_index(
            tmp_path / "duplicate.sgroute",
            manifest_paths=[manifest],
            tar_paths=[tar_a, tar_b],
        )


def test_audio_path_extraction_uses_alias_precedence_and_checks_cardinality():
    document = {
        "sound": ["first.wav", "second.wav"],
        "speech": "ignored.wav",
        "ori_sound": "also-ignored.wav",
        "conversations": _conversation("<sound> <speech>"),
    }
    assert extract_sharegpt_audio_paths(document) == ("first.wav", "second.wav")

    document["conversations"] = _conversation("<sound> <speech> <sound>")
    with pytest.raises(ValueError, match="cardinality"):
        extract_sharegpt_audio_paths(document)

    with pytest.raises(ValueError, match="flat list of strings"):
        extract_sharegpt_audio_paths({"sound": [["nested.wav"]], "conversations": _conversation("<sound>")})


def test_audio_path_extraction_preserves_existing_sharegpt_cardinality_rules():
    assert extract_sharegpt_audio_paths(
        {
            "sound": "one.wav",
            "conversations": _conversation("<sound> then <speech>"),
        }
    ) == ("one.wav",)
    assert extract_sharegpt_audio_paths(
        {
            "sound": ["one.wav", "two.wav"],
            "conversations": _conversation("<sound>"),
        }
    ) == ("one.wav", "two.wav")
    assert extract_sharegpt_audio_paths(
        {
            "sound": "",
            "speech": "fallback.wav",
            "conversations": [
                {"from": "human", "value": "<speech>"},
                {"from": "gpt", "value": "<sound> is literal assistant text"},
            ],
        }
    ) == ("fallback.wav",)


def test_native_tar_index_validation_proves_one_regular_member_per_range(tmp_path):
    tar_path = _write_tar(
        tmp_path / "audio.tar",
        [("first.flac", b"first"), ("second.flac", b"second")],
        directory="ignored/",
    )
    idx_path = Path(f"{tar_path}.idx")

    members = validate_native_tar_member_index(tar_path, idx_path)
    assert [(member.local_index, member.name) for member in members] == [
        (0, "first.flac"),
        (1, "second.flac"),
    ]
    assert all(member.start < member.data_end <= member.end for member in members)

    # Existing indexes may end at logical tar EOF rather than physical size;
    # that is valid only when the last range still encloses exactly one member.
    logical_eof = members[-1].data_end
    offsets = [members[0].start, members[1].start, logical_eof]
    idx_path.write_bytes(struct.pack(f"<{len(offsets)}Q", *offsets))
    assert len(validate_native_tar_member_index(tar_path, idx_path)) == 2

    # One range now contains two regular members, so it cannot be addressed by
    # a (shard, local-member) route record.
    idx_path.write_bytes(struct.pack("<2Q", members[0].start, tar_path.stat().st_size))
    with pytest.raises(ValueError, match="exactly one regular member"):
        validate_native_tar_member_index(tar_path, idx_path)


def test_native_tar_range_validation_scales_linearly():
    class CountingInt(int):
        comparisons = 0

        def _compare(self, operation, other):
            type(self).comparisons += 1
            return operation(self, other)

        def __lt__(self, other):
            return self._compare(int.__lt__, other)

        def __gt__(self, other):
            return self._compare(int.__gt__, other)

        def __ne__(self, other):
            return self._compare(int.__ne__, other)

    count = 4096
    offsets = [CountingInt(index * 1024) for index in range(count + 1)]
    members = [(f"{index}.flac", offsets[index], CountingInt(offsets[index] + 512)) for index in range(count)]

    validated = _validate_native_tar_ranges(
        tar_path="synthetic.tar",
        offsets=offsets,
        members=members,
    )

    assert len(validated) == count
    assert CountingInt.comparisons <= 4 * count
