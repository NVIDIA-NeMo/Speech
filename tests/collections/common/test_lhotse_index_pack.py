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

import pytest
import yaml
from click.testing import CliRunner
from lhotse.index_pack import IndexPack, IndexPackCollectionSpec, index_pack_collection_key, write_index_pack
from lhotse.indexing import create_jsonl_index
from omegaconf import OmegaConf
from scripts.dataloading import convert_indexes_to_idxpack as converter
from scripts.dataloading.convert_indexes_to_idxpack import main

from nemo.collections.common.data.lhotse import nemo_adapters, text_adapters
from nemo.collections.common.data.lhotse.cutset import read_nemo_manifest
from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index as create_nemo_tar_index
from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator, LazyNeMoTarredIterator


def _make_native_tar_dataset(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": "sample.wav"}) + "\n")
    create_jsonl_index(manifest)

    tar_path = tmp_path / "audio.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = b"audio"
        info = tarfile.TarInfo("sample.wav")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    idx_path = tmp_path / "audio.tar.idx"
    create_nemo_tar_index(tar_path, idx_path)
    input_cfg = tmp_path / "dataset.yaml"
    input_cfg.write_text(
        yaml.safe_dump(
            {
                "type": "nemo_tarred",
                "manifest_filepath": str(manifest),
                "tarred_audio_filepaths": str(tar_path),
            }
        )
    )
    return tar_path, idx_path, input_cfg


def test_converter_accepts_current_headerless_native_tar_sidecar(tmp_path):
    tar_path, idx_path, input_cfg = _make_native_tar_dataset(tmp_path)

    output = tmp_path / "dataset.idxpack"
    result = CliRunner().invoke(main, ["--output", str(output), str(input_cfg)])

    assert result.exit_code == 0, result.output
    assert idx_path.read_bytes()[:8] == struct.pack("<Q", 0)
    with IndexPack(output) as pack:
        key = index_pack_collection_key("tar", "nemo_tar", str(tar_path))
        collection = pack.collection(key)
        assert collection.locate(0).end == tar_path.stat().st_size


def test_converter_rejects_native_tar_sentinel_mismatch(tmp_path):
    tar_path, idx_path, input_cfg = _make_native_tar_dataset(tmp_path)
    with idx_path.open("r+b") as stream:
        stream.seek(-8, io.SEEK_END)
        stream.write(struct.pack("<Q", tar_path.stat().st_size - 512))

    result = CliRunner().invoke(
        main,
        ["--output", str(tmp_path / "dataset.idxpack"), str(input_cfg)],
    )

    assert result.exit_code != 0
    assert "sentinel" in result.output
    assert "build_indexes.py --force" in result.output


def test_converter_validates_remote_native_tar_sentinel(tmp_path, monkeypatch):
    tar_path, local_idx, _ = _make_native_tar_dataset(tmp_path)
    remote_path = "ais://bucket/audio.tar"
    remote_idx = converter._resolve_local_sidecar(remote_path, tmp_path)
    remote_idx.parent.mkdir(parents=True, exist_ok=True)
    remote_idx.write_bytes(local_idx.read_bytes())
    monkeypatch.setattr(converter, "_source_size", lambda path: tar_path.stat().st_size + 1)

    with pytest.raises(ValueError, match="sentinel"):
        converter._validate_native_tar_sidecar(remote_path, tmp_path)


def test_converter_rejects_experimental_in_band_header(tmp_path):
    _, idx_path, _ = _make_native_tar_dataset(tmp_path)
    idx_path.write_bytes(b"NEMOTAR\0" + idx_path.read_bytes())

    with pytest.raises(ValueError, match="in-band NEMOTAR header"):
        converter._read_raw_tar_sentinel(idx_path)


def test_convert_input_cfg_sidecars_to_one_index_pack(tmp_path):
    manifests = []
    source_spec = str(tmp_path / "manifest__OP_0..1_CL_.jsonl")
    for shard in range(2):
        path = tmp_path / f"manifest_{shard}.jsonl"
        with path.open("w") as f:
            for item in range(shard + 1):
                f.write(json.dumps({"id": f"{shard}-{item}"}) + "\n")
        create_jsonl_index(path)
        manifests.append(str(path))

    input_cfg = tmp_path / "dataset.yaml"
    input_cfg.write_text(
        yaml.safe_dump(
            [
                {
                    "type": "nemo",
                    "manifest_filepath": source_spec,
                }
            ]
        )
    )
    output = tmp_path / "dataset.idxpack"
    result = CliRunner().invoke(
        main,
        [
            "--output",
            str(output),
            str(input_cfg),
        ],
    )
    assert result.exit_code == 0, result.output

    with IndexPack(output) as pack:
        key = index_pack_collection_key("manifest", "jsonl", source_spec)
        collection = pack.collection(key)
        assert len(collection) == 3
        assert pack.num_segments == 2


def test_flat_native_lists_use_aggregate_pack_and_preserve_positional_pairs(tmp_path, monkeypatch):
    manifests = []
    declared_tar_paths = []
    expected_texts = []
    for position, (manifest_id, tar_id) in enumerate(((3076, 2), (4100, 0), (5200, 1))):
        manifest = tmp_path / f"manifest_{manifest_id}.jsonl"
        member = f"sample-{position}.wav"
        text = f"text-{position}"
        manifest.write_text(
            json.dumps(
                {
                    "audio_filepath": member,
                    "duration": 1.0,
                    "text": text,
                    "lang": "en",
                }
            )
            + "\n"
        )
        create_jsonl_index(manifest)
        manifests.append(str(manifest))
        declared_tar_paths.append(f"ais://bucket/audio_{tar_id}.tar")
        expected_texts.append(text)

    input_cfg = tmp_path / "flat-lists.yaml"
    input_cfg.write_text(
        yaml.safe_dump(
            {
                "type": "nemo_tarred",
                "manifest_filepath": manifests,
                "tarred_audio_filepaths": declared_tar_paths,
            }
        )
    )
    output = tmp_path / "flat-lists.idxpack"
    result = CliRunner().invoke(
        main,
        [
            "--output",
            str(output),
            "--native-tar-paths-only",
            str(input_cfg),
        ],
    )
    assert result.exit_code == 0, result.output

    with IndexPack(output) as pack:
        manifest_collection = pack.collection(index_pack_collection_key("manifest", "jsonl", manifests))
        tar_collection = pack.collection(index_pack_collection_key("tar", "nemo_tar", declared_tar_paths))
        assert manifest_collection.sequence_count == 3
        assert tar_collection.sequence_count == 3
        assert [tar_collection.path_for_shard(idx) for idx in range(3)] == declared_tar_paths

    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    config = OmegaConf.create(
        {
            "manifest_filepath": manifests,
            "tarred_audio_filepaths": declared_tar_paths,
            "indexed": True,
            "index_pack": str(output),
            "force_finite": True,
        }
    )
    cuts, is_tarred = read_nemo_manifest(config)
    cuts = list(cuts)
    assert is_tarred
    assert [cut.supervisions[0].text for cut in cuts] == expected_texts
    assert [cut.recording.sources[0].source for cut in cuts] == [
        f"{tar_path}/sample-{position}.wav" for position, tar_path in enumerate(declared_tar_paths)
    ]


def test_converter_rejects_mixed_scalar_and_flat_native_pairs(tmp_path):
    input_cfg = tmp_path / "mixed-path-forms.yaml"
    input_cfg.write_text(
        yaml.safe_dump(
            {
                "type": "nemo_tarred",
                "manifest_filepath": [str(tmp_path / "manifest.jsonl")],
                "tarred_audio_filepaths": str(tmp_path / "audio.tar"),
            }
        )
    )

    result = CliRunner().invoke(
        main,
        ["--dry-run", "--output", str(tmp_path / "mixed.idxpack"), str(input_cfg)],
    )
    assert result.exit_code != 0
    assert "must both use scalar path specs or both use non-empty flat lists" in str(result.exception)


def test_converter_rejects_nested_native_manifest_lists(tmp_path):
    input_cfg = tmp_path / "nested-lists.yaml"
    input_cfg.write_text(
        yaml.safe_dump(
            {
                "type": "nemo",
                "manifest_filepath": [[str(tmp_path / "manifest.jsonl"), 0.5]],
            }
        )
    )

    result = CliRunner().invoke(
        main,
        ["--dry-run", "--output", str(tmp_path / "nested.idxpack"), str(input_cfg)],
    )
    assert result.exit_code != 0
    assert "nested and weighted list forms are not supported" in str(result.exception)


def test_lazy_nemo_iterator_uses_pack_without_expanding_shards(tmp_path, monkeypatch):
    manifests = []
    source_spec = str(tmp_path / "manifest__OP_0..1_CL_.jsonl")
    expected_texts = []
    for shard in range(2):
        path = tmp_path / f"manifest_{shard}.jsonl"
        with path.open("w") as f:
            for item in range(shard + 1):
                text = f"text-{shard}-{item}"
                expected_texts.append(text)
                f.write(
                    json.dumps(
                        {
                            "audio_filepath": f"audio-{shard}-{item}.wav",
                            "duration": 1.0,
                            "text": text,
                            "lang": "en",
                        }
                    )
                    + "\n"
                )
        create_jsonl_index(path)
        manifests.append(str(path))

    spec = IndexPackCollectionSpec(
        role="manifest",
        kind="jsonl",
        source_spec=source_spec,
        paths=tuple(manifests),
    )
    output = tmp_path / "dataset.idxpack"
    write_index_pack(output, [spec])

    def fail_expand(*args, **kwargs):
        raise AssertionError("packed construction must not expand shard paths")

    monkeypatch.setattr(nemo_adapters, "expand_sharded_filepaths", fail_expand)
    iterator = LazyNeMoIterator(
        source_spec,
        metadata_only=True,
        indexed=True,
        index_pack=output,
    )
    assert len(iterator) == len(expected_texts)
    assert [cut.supervisions[0].text for cut in iterator] == expected_texts


def test_lazy_nemo_tarred_iterator_uses_pack_without_expanding_shards(tmp_path, monkeypatch):
    manifest_paths = []
    tar_paths = []
    manifest_spec = str(tmp_path / "manifest__OP_0..1_CL_.jsonl")
    tar_spec = str(tmp_path / "audio__OP_0..1_CL_.tar")
    expected_texts = []
    for shard in range(2):
        manifest = tmp_path / f"manifest_{shard}.jsonl"
        tar_path = tmp_path / f"audio_{shard}.tar"
        members = []
        with manifest.open("w") as f:
            for item in range(shard + 1):
                member = f"sample-{shard}-{item}.wav"
                text_value = f"tarred-{shard}-{item}"
                members.append(member)
                expected_texts.append(text_value)
                f.write(
                    json.dumps(
                        {
                            "audio_filepath": member,
                            "duration": 1.0,
                            "text": text_value,
                            "lang": "en",
                        }
                    )
                    + "\n"
                )
        with tarfile.open(tar_path, "w") as archive:
            for member in members:
                payload = b"not-read-in-ais-mode"
                info = tarfile.TarInfo(member)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        create_jsonl_index(manifest)
        manifest_paths.append(str(manifest))
        tar_paths.append(str(tar_path))

    input_cfg = tmp_path / "tarred.yaml"
    input_cfg.write_text(
        yaml.safe_dump(
            {
                "type": "nemo",
                "manifest_filepath": manifest_spec,
                "tarred_audio_filepaths": tar_spec,
            }
        )
    )
    output = tmp_path / "tarred.idxpack"
    result = CliRunner().invoke(
        main,
        [
            "--output",
            str(output),
            "--native-tar-paths-only",
            str(input_cfg),
        ],
    )
    assert result.exit_code == 0, result.output

    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")

    def fail_expand(*args, **kwargs):
        raise AssertionError("packed tarred construction must not expand shard paths")

    monkeypatch.setattr(nemo_adapters, "expand_sharded_filepaths", fail_expand)
    iterator = LazyNeMoTarredIterator(
        manifest_spec,
        tar_spec,
        indexed=True,
        index_pack=output,
    )
    cuts = list(iterator)
    assert [cut.supervisions[0].text for cut in cuts] == expected_texts
    assert [cut.recording.sources[0].source for cut in cuts] == [
        f"{tar_path}/{member}"
        for tar_path, shard in zip(tar_paths, range(2))
        for member in [f"sample-{shard}-{item}.wav" for item in range(shard + 1)]
    ]


def test_nemotron_text_jsonl_uses_one_pack(tmp_path, monkeypatch):
    def sample(sample_id):
        return {
            "id": sample_id,
            "conversation": [
                {"sender": "User", "fragments": ["question"]},
                {"sender": "Assistant", "fragments": [sample_id]},
            ],
        }

    manifest = tmp_path / "text.jsonl"
    manifest.write_text(json.dumps(sample("jsonl-sample-0")) + "\n" + json.dumps(sample("jsonl-sample-1")) + "\n")
    create_jsonl_index(manifest)
    paths = str(manifest)
    input_cfg = tmp_path / "text.yaml"
    input_cfg.write_text(
        yaml.safe_dump(
            {
                "type": "nemotron_text_converation",
                "paths": paths,
            }
        )
    )
    output = tmp_path / "text.idxpack"
    result = CliRunner().invoke(
        main,
        [
            "--output",
            str(output),
            str(input_cfg),
        ],
    )
    assert result.exit_code == 0, result.output

    def fail_expand(*args, **kwargs):
        raise AssertionError("packed text construction must not expand paths")

    monkeypatch.setattr(text_adapters, "expand_sharded_filepaths", fail_expand)
    iterator = text_adapters.NemotronTextConversationAdapter(
        paths,
        indexed=True,
        index_pack=output,
    )
    conversations = list(iterator)
    assert [item.id for item in conversations] == [
        "jsonl-sample-0",
        "jsonl-sample-1",
    ]


def test_converter_rejects_mixed_nemotron_text_paths(tmp_path):
    input_cfg = tmp_path / "mixed.yaml"
    input_cfg.write_text(
        yaml.safe_dump(
            {
                "type": "nemotron_text_converation",
                "paths": [str(tmp_path / "text.jsonl"), str(tmp_path / "text.tar")],
            }
        )
    )

    result = CliRunner().invoke(main, ["--dry-run", "--output", str(tmp_path / "mixed.idxpack"), str(input_cfg)])
    assert result.exit_code != 0
    assert "must be homogeneous" in str(result.exception)


def test_packed_nemotron_tar_reports_the_source_tar(tmp_path):
    tar_path = tmp_path / "text.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = json.dumps({"conversation": []}).encode()
        info = tarfile.TarInfo("not-json.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    create_nemo_tar_index(tar_path, str(tar_path) + ".idx")

    input_cfg = tmp_path / "text.yaml"
    input_cfg.write_text(
        yaml.safe_dump(
            {
                "type": "nemotron_text_converation",
                "paths": str(tar_path),
            }
        )
    )
    pack_path = tmp_path / "text.idxpack"
    result = CliRunner().invoke(main, ["--output", str(pack_path), str(input_cfg)])
    assert result.exit_code == 0, result.output

    iterator = text_adapters.NemotronTextConversationAdapter(
        str(tar_path),
        indexed=True,
        index_pack=pack_path,
    )
    with pytest.raises(RuntimeError, match=str(tar_path)):
        iterator[0]


def test_converter_rejects_misaligned_native_shards(tmp_path):
    input_cfg = tmp_path / "misaligned.yaml"
    input_cfg.write_text(
        yaml.safe_dump(
            {
                "type": "nemo_tarred",
                "manifest_filepath": str(tmp_path / "manifest__OP_0..1_CL_.jsonl"),
                "tarred_audio_filepaths": str(tmp_path / "audio__OP_1..2_CL_.tar"),
            }
        )
    )

    result = CliRunner().invoke(
        main,
        [
            "--dry-run",
            "--output",
            str(tmp_path / "misaligned.idxpack"),
            str(input_cfg),
        ],
    )
    assert result.exit_code != 0
    assert "not positionally aligned" in str(result.exception)


def test_local_packed_native_tar_validates_lengths_per_shard(tmp_path, monkeypatch):
    manifest_paths = []
    tar_paths = []
    for shard, (manifest_count, tar_count) in enumerate(((1, 2), (2, 1))):
        manifest = tmp_path / f"manifest_{shard}.jsonl"
        manifest.write_text(
            "".join(
                json.dumps(
                    {
                        "audio_filepath": f"sample-{shard}-{idx}.wav",
                        "duration": 1.0,
                        "text": "text",
                    }
                )
                + "\n"
                for idx in range(manifest_count)
            )
        )
        tar_path = tmp_path / f"audio_{shard}.tar"
        with tarfile.open(tar_path, "w") as archive:
            for idx in range(tar_count):
                payload = b"audio"
                info = tarfile.TarInfo(f"sample-{shard}-{idx}.wav")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        create_jsonl_index(manifest)
        create_nemo_tar_index(tar_path, str(tar_path) + ".idx")
        manifest_paths.append(str(manifest))
        tar_paths.append(str(tar_path))

    manifest_spec = str(tmp_path / "manifest__OP_0..1_CL_.jsonl")
    tar_spec = str(tmp_path / "audio__OP_0..1_CL_.tar")
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(
        pack_path,
        [
            IndexPackCollectionSpec(
                role="manifest",
                kind="jsonl",
                source_spec=manifest_spec,
                paths=tuple(manifest_paths),
            ),
            IndexPackCollectionSpec(
                role="tar",
                kind="nemo_tar",
                source_spec=tar_spec,
                paths=tuple(tar_paths),
            ),
        ],
    )
    monkeypatch.delenv("USE_AIS_GET_BATCH", raising=False)

    with pytest.raises(ValueError, match="length mismatch in shard 0"):
        LazyNeMoTarredIterator(
            manifest_spec,
            tar_spec,
            indexed=True,
            index_pack=pack_path,
        )


def test_share_gpt_jsonl_uses_pack(tmp_path, monkeypatch):
    manifest = tmp_path / "sharegpt.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample",
                "conversations": [
                    {"from": "human", "value": "question"},
                    {"from": "gpt", "value": "answer"},
                ],
            }
        )
        + "\n"
    )
    create_jsonl_index(manifest)
    input_cfg = tmp_path / "sharegpt.yaml"
    input_cfg.write_text(
        yaml.safe_dump(
            {
                "type": "share_gpt",
                "manifest_filepath": str(manifest),
            }
        )
    )
    pack_path = tmp_path / "sharegpt.idxpack"
    result = CliRunner().invoke(main, ["--output", str(pack_path), str(input_cfg)])
    assert result.exit_code == 0, result.output

    def fail_expand(*args, **kwargs):
        raise AssertionError("packed ShareGPT construction must not expand paths")

    monkeypatch.setattr(text_adapters, "expand_sharded_filepaths", fail_expand)
    iterator = text_adapters.NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=str(manifest),
        audio_locator_tag="<audio>",
        audio_placeholders=[],
        indexed=True,
        index_pack=pack_path,
    )
    conversation = iterator[0]
    assert conversation.id == "sample"
    assert [turn.value for turn in conversation.turns] == ["question", "answer"]
