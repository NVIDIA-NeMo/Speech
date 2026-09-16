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
import tarfile
import wave
from pathlib import Path

import pytest
import yaml
from lhotse.index_pack import IndexPackCollectionSpec, write_index_pack

from nemo.collections.common.data.lhotse import text_adapters
from nemo.collections.common.data.lhotse.indexed_adapters import (
    TarSampleBundle,
    TarSampleMember,
    create_wds_v2_tar_index,
)
from nemo.collections.common.data.lhotse.text_adapters import (
    AudioTurn,
    NeMoMultimodalConversationShareGPTWebdatasetAdapter,
)


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
    return path


def _json(**extra) -> bytes:
    return json.dumps(
        {
            "id": "variable-member-sample",
            "speech": ["/original/7.wav", "/another/root/7.1.wav"],
            "conversations": [
                {"from": "human", "value": "First <speech> then <speech>"},
                {"from": "gpt", "value": "done"},
            ],
            **extra,
        }
    ).encode()


@pytest.mark.parametrize("json_first", [True, False])
def test_sharegpt_wds_v2_resolves_multiple_audio_members_in_json_path_order(tmp_path, json_first):
    data_dir = tmp_path / "wds"
    data_dir.mkdir()
    members = [("7.json", _json()), ("7.1.wav", _wav(1.5)), ("7.wav", _wav(1.0))]
    if not json_first:
        members = [members[1], members[2], members[0]]
    tar_path = _write_tar(data_dir / "shard.tar", members)
    (data_dir / "wids-meta.json").write_text(json.dumps({"shardlist": [{"url": "shard.tar", "nsamples": 1}]}))
    create_wds_v2_tar_index(tar_path)

    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(data_dir),
        audio_locator_tag="[audio]",
        audio_placeholders=["<speech>"],
        indexed=True,
        wds_sample_index_version=2,
    )

    (conversation,) = list(adapter)
    audio_turns = [turn for turn in conversation.turns if isinstance(turn, AudioTurn)]
    assert [turn.cut.duration for turn in audio_turns] == [1.0, 1.5]
    assert [turn.cut.load_audio().shape for turn in audio_turns] == [
        (1, 16000),
        (1, 24000),
    ]
    assert [turn.cut.custom["_source_codec"] for turn in audio_turns] == ["wav", "wav"]
    assert len({turn.cut.custom["_source_read_key"] for turn in audio_turns}) == 1
    assert all(turn.cut.custom["_source_range_bytes"] > 0 for turn in audio_turns)


def test_sharegpt_wds_v2_packed_reader_uses_catalog_without_data_dir_scan(tmp_path):
    tar_path = _write_tar(
        tmp_path / "packed.tar",
        [("7.json", _json()), ("7.wav", _wav(1.0)), ("7.1.wav", _wav(1.5))],
    )
    idx_path, _ = create_wds_v2_tar_index(tar_path)
    logical_data_dir = "/logical/source/that/does/not/exist"
    spec = IndexPackCollectionSpec(
        role="wds_tar",
        kind="wds_tar_v2",
        source_spec=logical_data_dir,
        paths=(str(tar_path),),
    )
    pack_path = tmp_path / "wds-v2.idxpack"
    write_index_pack(pack_path, [spec], index_path_overrides={str(tar_path): idx_path})

    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=logical_data_dir,
        audio_locator_tag="[audio]",
        audio_placeholders=["<speech>"],
        indexed=True,
        wds_sample_index_version=2,
        index_pack=pack_path,
    )

    assert len(adapter) == 1
    conversation = adapter[0]
    assert [turn.cut.duration for turn in conversation.turns if isinstance(turn, AudioTurn)] == [1.0, 1.5]


def test_sharegpt_wds_v2_reads_nv_split_catalog_in_declared_order(tmp_path):
    data_dir = tmp_path / "wds"
    shard_dir = data_dir / "shards"
    shard_dir.mkdir(parents=True)
    tar_paths = []
    for idx in range(2):
        tar_path = _write_tar(
            shard_dir / f"shard-{idx}.tar",
            [
                (f"{idx}.json", _json(id=f"sample-{idx}", speech=f"{idx}.wav")),
                (f"{idx}.wav", _wav(1.0)),
            ],
        )
        create_wds_v2_tar_index(tar_path)
        tar_paths.append(tar_path)
    unrelated = _write_tar(data_dir / "ignore-me.tar", [("9.json", _json()), ("9.wav", _wav(1.0))])
    (data_dir / ".nv-meta").mkdir()
    (data_dir / ".nv-meta" / "split.yaml").write_text(
        yaml.safe_dump({"split_parts": {"train": ["shards/shard-{0..1}.tar"]}})
    )

    adapter = NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(data_dir),
        audio_locator_tag="[audio]",
        audio_placeholders=["<speech>"],
        indexed=True,
        wds_sample_index_version=2,
    )

    assert adapter._shard_paths == [str(path) for path in tar_paths]
    assert str(unrelated) not in adapter._shard_paths
    assert [conversation.id for conversation in adapter] == ["sample-0", "sample-1"]


def test_sharegpt_wds_v2_requires_bounded_shard_catalog(tmp_path):
    data_dir = tmp_path / "wds"
    data_dir.mkdir()
    _write_tar(data_dir / "shard.tar", [("0.json", _json()), ("0.wav", _wav(1.0))])

    with pytest.raises(FileNotFoundError, match="bounded shard catalog"):
        NeMoMultimodalConversationShareGPTWebdatasetAdapter(
            data_dir=str(data_dir),
            audio_locator_tag="[audio]",
            indexed=True,
            wds_sample_index_version=2,
        )


def test_sharegpt_wds_v2_single_audio_accepts_legacy_source_path():
    adapter = object.__new__(NeMoMultimodalConversationShareGPTWebdatasetAdapter)
    adapter.audio_locator_tag = "[audio]"
    adapter.audio_placeholders = ["<sound>", "<speech>"]
    adapter.token_equivalent_duration = None
    bundle = TarSampleBundle(
        "0",
        json.loads(
            _json(
                speech="/original/audio_0.tar/renamed-before-packing.wav",
                conversations=[
                    {"from": "human", "value": "Listen <speech>"},
                    {"from": "gpt", "value": "done"},
                ],
            )
        ),
        (TarSampleMember("0.wav", _wav(1.0)),),
    )

    conversation = adapter._yield_from_sample_bundle(bundle)

    assert [turn.cut.duration for turn in conversation.turns if isinstance(turn, AudioTurn)] == [1.0]


@pytest.mark.parametrize(
    "bundle,error",
    [
        (
            TarSampleBundle(
                "7",
                json.loads(_json(speech="missing.wav")),
                (
                    TarSampleMember("7.wav", _wav(1.0)),
                    TarSampleMember("7.flac", _wav(1.0)),
                ),
            ),
            "missing audio member",
        ),
        (
            TarSampleBundle(
                "7",
                json.loads(_json(speech="same.wav")),
                (
                    TarSampleMember("left/same.wav", _wav(1.0)),
                    TarSampleMember("right/same.wav", _wav(1.0)),
                ),
            ),
            "ambiguous audio member basename",
        ),
    ],
)
def test_sharegpt_wds_v2_fails_on_missing_or_ambiguous_audio_member(tmp_path, bundle, error):
    adapter = object.__new__(NeMoMultimodalConversationShareGPTWebdatasetAdapter)
    adapter.audio_locator_tag = "[audio]"
    adapter.audio_placeholders = ["<sound>", "<speech>"]
    adapter.token_equivalent_duration = None

    with pytest.raises(ValueError, match=error):
        adapter._yield_from_sample_bundle(bundle)


def test_sharegpt_wds_v2_decodes_repeated_member_once(monkeypatch):
    adapter = object.__new__(NeMoMultimodalConversationShareGPTWebdatasetAdapter)
    adapter.audio_locator_tag = "[audio]"
    adapter.audio_placeholders = ["<speech>"]
    adapter.token_equivalent_duration = None
    bundle = TarSampleBundle(
        "7",
        {
            "id": "repeated",
            "speech": "7.wav",
            "conversations": [
                {"from": "human", "value": "First <speech> then <speech>"},
                {"from": "gpt", "value": "done"},
            ],
        },
        (TarSampleMember("7.wav", _wav(1.0)),),
    )
    original = text_adapters.Recording.from_bytes
    calls = 0

    def counted_from_bytes(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(text_adapters.Recording, "from_bytes", staticmethod(counted_from_bytes))

    conversation = adapter._yield_from_sample_bundle(bundle)

    assert len([turn for turn in conversation.turns if isinstance(turn, AudioTurn)]) == 2
    assert calls == 1
