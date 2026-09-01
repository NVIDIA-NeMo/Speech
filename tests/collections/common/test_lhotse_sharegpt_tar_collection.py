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
import pickle
import struct
import tarfile
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from lhotse.index_pack import IndexPackCollectionSpec, write_index_pack
from lhotse.indexing import create_jsonl_index

from nemo.collections.common.data.lhotse import indexed_adapters, sharegpt_tar_routing
from nemo.collections.common.data.lhotse.indexed_adapters import PackedTarMemberReader, create_tar_index
from nemo.collections.common.data.lhotse.sharegpt_tar_routing import build_sharegpt_tar_routing_index
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, NeMoMultimodalConversationShareGPTJsonlAdapter


def _wav(duration: float) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * round(duration * 16000))
    return stream.getvalue()


def _write_tar(path: Path, members: list[tuple[str, bytes]]) -> Path:
    with tarfile.open(path, "w:") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    create_tar_index(path, Path(f"{path}.idx"))
    return path


def _write_manifest(path: Path) -> Path:
    rows = [
        {
            "id": "first",
            "sound": "first.wav",
            "conversations": [
                {"from": "human", "value": "Listen <sound>"},
                {"from": "gpt", "value": "done"},
            ],
        },
        {
            "id": "second",
            "sound": "/original/second.wav",
            "conversations": [
                {"from": "human", "value": "Listen <sound>"},
                {"from": "gpt", "value": "done"},
            ],
        },
    ]
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    create_jsonl_index(path)
    return path


def _sources(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "manifest.jsonl")
    tars = [
        _write_tar(
            tmp_path / "a.tar",
            [("unrelated.wav", _wav(0.5)), ("second.wav", _wav(1.5))],
        ),
        _write_tar(tmp_path / "b.tar", [("first.wav", _wav(1.0))]),
    ]
    route = tmp_path / "data.sgroute"
    build_sharegpt_tar_routing_index(route, manifest_paths=[manifest], tar_paths=tars)
    return manifest, tars, route


def _assert_conversations(adapter):
    first, second = list(adapter)
    first_audio = [turn for turn in first.turns if isinstance(turn, AudioTurn)]
    second_audio = [turn for turn in second.turns if isinstance(turn, AudioTurn)]
    assert [turn.cut.duration for turn in first_audio] == [1.0]
    assert [turn.cut.duration for turn in second_audio] == [1.5]
    assert [turn.cut.custom["_source_codec"] for turn in first_audio + second_audio] == ["wav", "wav"]
    assert all(turn.cut.custom["_source_range_bytes"] > 0 for turn in first_audio + second_audio)
    assert len({turn.cut.custom["_source_read_key"] for turn in first_audio + second_audio}) == 2


def test_sharegpt_loose_index_tar_collection_routes_manifest_rows(tmp_path):
    manifest, tars, route = _sources(tmp_path)
    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=str(manifest),
        tarred_audio_filepaths=[str(path) for path in tars],
        tar_lookup_mode="collection",
        tar_routing_filepath=route,
        audio_locator_tag="[audio]",
        audio_placeholders=["<sound>"],
        indexed=True,
    )

    _assert_conversations(adapter)


def test_sharegpt_packed_tar_collection_routes_without_paths_only_fallback(tmp_path):
    manifest, tars, route = _sources(tmp_path)
    raw_tars = [str(path) for path in tars]
    manifest_spec = str(manifest)
    manifest_collection = IndexPackCollectionSpec(
        role="manifest",
        kind="jsonl",
        source_spec=manifest_spec,
        paths=(manifest_spec,),
    )
    tar_collection = IndexPackCollectionSpec(
        role="tar_collection",
        kind="nemo_tar",
        source_spec=raw_tars,
        paths=tuple(raw_tars),
    )
    pack_path = tmp_path / "collection.idxpack"
    write_index_pack(pack_path, [manifest_collection, tar_collection])

    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=manifest_spec,
        tarred_audio_filepaths=raw_tars,
        tar_lookup_mode="collection",
        tar_routing_filepath=route,
        audio_locator_tag="[audio]",
        audio_placeholders=["<sound>"],
        indexed=True,
        index_pack=pack_path,
    )

    restored = pickle.loads(pickle.dumps(adapter))
    _assert_conversations(restored)


def test_sharegpt_packed_tar_collection_resolves_relative_route_beside_pack(tmp_path):
    manifest, tars, route = _sources(tmp_path)
    raw_tars = [str(path) for path in tars]
    manifest_spec = str(manifest)
    pack_path = tmp_path / "collection.idxpack"
    write_index_pack(
        pack_path,
        [
            IndexPackCollectionSpec(
                role="manifest",
                kind="jsonl",
                source_spec=manifest_spec,
                paths=(manifest_spec,),
            ),
            IndexPackCollectionSpec(
                role="tar_collection",
                kind="nemo_tar",
                source_spec=raw_tars,
                paths=tuple(raw_tars),
            ),
        ],
    )

    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=manifest_spec,
        tarred_audio_filepaths=raw_tars,
        tar_lookup_mode="collection",
        tar_routing_filepath=route.name,
        audio_locator_tag="[audio]",
        audio_placeholders=["<sound>"],
        indexed=True,
        index_pack=pack_path,
    )

    assert adapter.tar_routing_filepath == route
    _assert_conversations(adapter)


def test_sharegpt_collection_rejects_stale_prefix_map_digest(tmp_path):
    manifest, tars, route = _sources(tmp_path)

    with pytest.raises(ValueError, match="audio prefix-map digest"):
        NeMoMultimodalConversationShareGPTJsonlAdapter(
            manifest_filepath=str(manifest),
            tarred_audio_filepaths=[str(path) for path in tars],
            tar_lookup_mode="collection",
            tar_routing_filepath=route,
            audio_path_prefix_map={"/source": "/mirror"},
            audio_locator_tag="[audio]",
            audio_placeholders=["<sound>"],
            indexed=True,
        )


def test_sharegpt_collection_rejects_paths_only_tar_pack(tmp_path):
    manifest, tars, route = _sources(tmp_path)
    raw_tars = [str(path) for path in tars]
    manifest_spec = str(manifest)
    manifest_collection = IndexPackCollectionSpec(
        role="manifest", kind="jsonl", source_spec=manifest_spec, paths=(manifest_spec,)
    )
    paths_only = IndexPackCollectionSpec(
        role="tar_collection",
        kind="nemo_tar",
        source_spec=raw_tars,
        paths=tuple(raw_tars),
        offsets_required=False,
    )
    pack_path = tmp_path / "paths-only.idxpack"
    write_index_pack(pack_path, [manifest_collection, paths_only])

    with pytest.raises(ValueError, match="offset-bearing"):
        NeMoMultimodalConversationShareGPTJsonlAdapter(
            manifest_filepath=manifest_spec,
            tarred_audio_filepaths=raw_tars,
            tar_lookup_mode="collection",
            tar_routing_filepath=route,
            audio_locator_tag="[audio]",
            indexed=True,
            index_pack=pack_path,
        )


def test_sharegpt_packed_collection_validates_every_manifest_shard(tmp_path):
    manifests = []
    for index, audio_name in enumerate(("first.wav", "second.wav")):
        manifest = tmp_path / f"manifest-{index}.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "id": f"row-{index}",
                    "sound": audio_name,
                    "conversations": [{"from": "human", "value": "Listen <sound>"}],
                }
            )
            + "\n"
        )
        create_jsonl_index(manifest)
        manifests.append(manifest)
    tars = [
        _write_tar(
            tmp_path / "audio.tar",
            [("first.wav", _wav(1.0)), ("second.wav", _wav(2.0))],
        )
    ]
    route = tmp_path / "multi.sgroute"
    build_sharegpt_tar_routing_index(
        route,
        manifest_paths=manifests,
        manifest_specs=[str(path) for path in manifests],
        tar_paths=tars,
    )
    manifest_spec = [str(path) for path in manifests]
    tar_spec = [str(path) for path in tars]
    pack_path = tmp_path / "multi.idxpack"
    write_index_pack(
        pack_path,
        [
            IndexPackCollectionSpec(
                role="manifest",
                kind="jsonl",
                source_spec=manifest_spec,
                paths=tuple(manifest_spec),
            ),
            IndexPackCollectionSpec(
                role="tar_collection",
                kind="nemo_tar",
                source_spec=tar_spec,
                paths=tuple(tar_spec),
            ),
        ],
    )

    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=manifest_spec,
        tarred_audio_filepaths=tar_spec,
        tar_lookup_mode="collection",
        tar_routing_filepath=route,
        audio_locator_tag="[audio]",
        audio_placeholders=["<sound>"],
        indexed=True,
        index_pack=pack_path,
    )

    conversations = list(adapter)
    assert [item.id for item in conversations] == ["row-0", "row-1"]
    assert [[turn.cut.duration for turn in item.turns if isinstance(turn, AudioTurn)] for item in conversations] == [
        [1.0],
        [2.0],
    ]


def test_packed_tar_collection_uses_bounded_remote_ranges(tmp_path, monkeypatch):
    tar_path = _write_tar(tmp_path / "remote-source.tar", [("audio.wav", _wav(1.0))])
    raw = tar_path.read_bytes()
    offsets = struct.unpack(
        f"<{Path(f'{tar_path}.idx').stat().st_size // 8}Q",
        Path(f"{tar_path}.idx").read_bytes(),
    )
    location = SimpleNamespace(path="s3://bucket/audio.tar", start=offsets[0], end=offsets[1])

    class Collection:
        kind = "nemo_tar"
        pack = object()

        def __len__(self):
            return 1

        def locate_in_shard(self, shard_index, local_index):
            assert (shard_index, local_index) == (0, 0)
            return location

        def shard_length(self, shard_index):
            assert shard_index == 0
            return 1

        def path_for_shard(self, shard_index):
            assert shard_index == 0
            return location.path

    calls = []

    def read_remote(path, start, end):
        calls.append((path, start, end))
        return raw[start:end]

    monkeypatch.setattr(indexed_adapters, "read_exact_range", read_remote)
    reader = PackedTarMemberReader(Collection())

    name, payload = reader.get_shard(0, "audio.wav")

    assert name == "audio.wav"
    assert payload == _wav(1.0)
    assert calls
    assert all(call[0] == "s3://bucket/audio.tar" for call in calls)


def test_sharegpt_collection_runtime_startup_does_not_read_manifest_payload(tmp_path, monkeypatch):
    manifest, tars, route = _sources(tmp_path)
    calls = []

    def reject_payload_read(path, start, end):
        calls.append((path, start, end))
        raise AssertionError("runtime route startup must not read source payload bytes")

    monkeypatch.setattr(sharegpt_tar_routing, "read_exact_range", reject_payload_read)
    adapter = NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=str(manifest),
        tarred_audio_filepaths=[str(path) for path in tars],
        tar_lookup_mode="collection",
        tar_routing_filepath=route,
        audio_locator_tag="[audio]",
        audio_placeholders=["<sound>"],
        indexed=True,
    )

    assert len(adapter) == 2
    assert calls == []


def test_sharegpt_collection_rejects_stale_manifest_content(tmp_path):
    manifest, tars, route = _sources(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["conversations"][1]["value"] = "changed after route sealing"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    Path(f"{manifest}.idx").unlink()
    create_jsonl_index(manifest)

    with pytest.raises(ValueError, match="manifest source-identity digest"):
        NeMoMultimodalConversationShareGPTJsonlAdapter(
            manifest_filepath=str(manifest),
            tarred_audio_filepaths=[str(path) for path in tars],
            tar_lookup_mode="collection",
            tar_routing_filepath=route,
            audio_locator_tag="[audio]",
            audio_placeholders=["<sound>"],
            indexed=True,
        )


def test_sharegpt_collection_rejects_reordered_tar_catalog(tmp_path):
    manifest, tars, route = _sources(tmp_path)

    with pytest.raises(ValueError, match="tar catalog digest"):
        NeMoMultimodalConversationShareGPTJsonlAdapter(
            manifest_filepath=str(manifest),
            tarred_audio_filepaths=[str(path) for path in reversed(tars)],
            tar_lookup_mode="collection",
            tar_routing_filepath=route,
            audio_locator_tag="[audio]",
            audio_placeholders=["<sound>"],
            indexed=True,
        )
