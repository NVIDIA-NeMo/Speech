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
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from lhotse.index_pack import IndexPack, index_pack_collection_key
from omegaconf import OmegaConf
from scripts.dataloading import build_indexes, convert_indexes_to_idxpack, validate_dataloader

from nemo.collections.common.data.lhotse.indexed_adapters import create_wds_v2_tar_index, wds_v2_metadata_path
from nemo.collections.speechlm2.data.salm_dataset import SALMDataset


def _write_wds_tar(path: Path) -> Path:
    with tarfile.open(path, "w:") as archive:
        for name, payload in (
            ("0.json", json.dumps({"id": "zero"}).encode()),
            ("0.wav", b"audio-zero"),
            ("1.wav", b"audio-one"),
            ("1.json", json.dumps({"id": "one"}).encode()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _write_wds_config(tmp_path: Path, tar_path: Path) -> Path:
    (tmp_path / "wids-meta.json").write_text(json.dumps({"shardlist": [{"url": tar_path.name}]}))
    config_path = tmp_path / "wds.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "type": "share_gpt_webdataset",
                "data_dir": str(tmp_path),
                "wds_sample_index_version": 2,
            }
        )
    )
    return config_path


def test_build_indexes_accepts_repeatable_kind_filter(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n")
    config_path = tmp_path / "input.yaml"
    config_path.write_text(yaml.safe_dump({"type": "nemo", "manifest_filepath": str(manifest)}))

    result = CliRunner().invoke(
        build_indexes.main,
        ["--dry-run", "--kind", "jsonl", "--kind", "wds_tar_v2", str(config_path)],
    )

    assert result.exit_code == 0, result.output


def _write_nested_data_blend(tmp_path: Path) -> tuple[Path, Path, Path]:
    blend_dir = tmp_path / "blend"
    blend_dir.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n")
    inner = blend_dir / "inner.yaml"
    inner.write_text(
        yaml.safe_dump(
            {
                "type": "share_gpt",
                "manifest_filepath": str(manifest),
                "audio_locator_tag": "[audio]",
                "indexed": True,
            }
        )
    )
    outer = tmp_path / "outer.yaml"
    outer.write_text(
        yaml.safe_dump(
            [
                {
                    "type": "group",
                    "input_cfg": "${data_blend_dir}/inner.yaml",
                    "weight": 1.0,
                }
            ]
        )
    )
    return blend_dir, outer, manifest


def test_build_indexes_resolves_nested_data_blend_dir(tmp_path):
    blend_dir, outer, manifest = _write_nested_data_blend(tmp_path)

    result = CliRunner().invoke(
        build_indexes.main,
        ["--workers", "1", "--data-blend-dir", str(blend_dir), str(outer)],
    )

    assert result.exit_code == 0, result.output
    assert Path(f"{manifest}.idx").is_file()


def test_build_indexes_resolves_nested_input_cfg_relative_to_declaring_file(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n")
    wrapper_dir = tmp_path / "wrapper"
    wrapper_dir.mkdir()
    inner = wrapper_dir / "inner.yaml"
    inner.write_text(
        yaml.safe_dump(
            {
                "type": "share_gpt",
                "manifest_filepath": str(manifest),
                "audio_locator_tag": "[audio]",
                "indexed": True,
            }
        )
    )
    outer = wrapper_dir / "outer.yaml"
    outer.write_text(yaml.safe_dump([{"type": "group", "input_cfg": "inner.yaml", "weight": 1.0}]))

    result = CliRunner().invoke(
        build_indexes.main,
        ["--workers", "1", "--data-blend-dir", str(tmp_path), str(outer)],
    )

    assert result.exit_code == 0, result.output
    assert Path(f"{manifest}.idx").is_file()


def test_converter_resolves_nested_data_blend_dir(tmp_path):
    blend_dir, outer, _ = _write_nested_data_blend(tmp_path)

    result = CliRunner().invoke(
        convert_indexes_to_idxpack.main,
        [
            "--dry-run",
            "--data-blend-dir",
            str(blend_dir),
            "--output",
            str(tmp_path / "out.idxpack"),
            str(outer),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Discovered 1 collections with 1 ordered paths." in result.output


def test_build_indexes_discovers_versioned_wds_sidecar(tmp_path):
    tar_path = _write_wds_tar(tmp_path / "shard.tar")
    config = OmegaConf.load(_write_wds_config(tmp_path, tar_path))

    jobs = []
    build_indexes.discover(config, jobs)

    assert [(job.path, job.kind) for job in jobs] == [(str(tar_path), build_indexes.WDS_TAR_V2)]
    assert jobs[0].idx_path() == Path(f"{tar_path}.wds-v2.idx")


def test_build_indexes_discovers_split_catalog_without_recursive_scan(tmp_path):
    (tmp_path / "wds").mkdir()
    (tmp_path / "unrelated").mkdir()
    cataloged = [_write_wds_tar(tmp_path / "wds" / f"shard-{idx}.tar") for idx in range(2)]
    unrelated = _write_wds_tar(tmp_path / "unrelated" / "ignore-me.tar")
    (tmp_path / ".nv-meta").mkdir()
    (tmp_path / ".nv-meta" / "split.yaml").write_text(
        yaml.safe_dump(
            {
                "split_parts": {"train": ["wds/shard-{0..1}.tar"]},
                "exclude": [],
            }
        )
    )
    config = OmegaConf.create(
        {
            "type": "share_gpt_webdataset",
            "data_dir": str(tmp_path),
            "wds_sample_index_version": 2,
        }
    )

    jobs = []
    build_indexes.discover(config, jobs)

    assert [(job.path, job.kind) for job in jobs] == [(str(path), build_indexes.WDS_TAR_V2) for path in cataloged]
    assert all(job.path != str(unrelated) for job in jobs)


def test_build_indexes_wds_v2_requires_bounded_shard_catalog(tmp_path):
    _write_wds_tar(tmp_path / "shard.tar")
    config = OmegaConf.create(
        {
            "type": "share_gpt_webdataset",
            "data_dir": str(tmp_path),
            "wds_sample_index_version": 2,
        }
    )

    with pytest.raises(FileNotFoundError, match="bounded shard catalog"):
        build_indexes.discover(config, [])


def test_build_indexes_revalidates_existing_wds_v2_sidecar(tmp_path):
    tar_path = _write_wds_tar(tmp_path / "shard.tar")
    create_wds_v2_tar_index(tar_path)
    job = build_indexes.IndexJob(str(tar_path), build_indexes.WDS_TAR_V2)
    assert build_indexes._is_indexed(job)

    metadata_path = wds_v2_metadata_path(job.idx_path())
    metadata = json.loads(metadata_path.read_text())
    metadata["offsets_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))

    assert not build_indexes._is_indexed(job)


def test_converter_packs_validated_wds_v2_collection(tmp_path):
    tar_path = _write_wds_tar(tmp_path / "shard.tar")
    create_wds_v2_tar_index(tar_path)
    config_path = _write_wds_config(tmp_path, tar_path)
    output = tmp_path / "wds.idxpack"

    result = CliRunner().invoke(
        convert_indexes_to_idxpack.main,
        ["--output", str(output), str(config_path)],
    )

    assert result.exit_code == 0, result.output
    with IndexPack(output) as pack:
        key = index_pack_collection_key("wds_tar", "wds_tar_v2", str(tmp_path))
        collection = pack.collection(key)
        assert collection.kind == "wds_tar_v2"
        assert collection.locate(1).end == tar_path.stat().st_size


def test_converter_rejects_tampered_wds_v2_metadata(tmp_path):
    tar_path = _write_wds_tar(tmp_path / "shard.tar")
    idx_path, metadata_path = create_wds_v2_tar_index(tar_path)
    metadata = json.loads(metadata_path.read_text())
    metadata["sample_count"] += 1
    metadata_path.write_text(json.dumps(metadata, separators=(",", ":"), sort_keys=True))
    config_path = _write_wds_config(tmp_path, tar_path)

    result = CliRunner().invoke(
        convert_indexes_to_idxpack.main,
        ["--output", str(tmp_path / "bad.idxpack"), str(config_path)],
    )

    assert result.exit_code != 0
    assert str(idx_path) in result.output
    assert "sample count mismatch" in result.output.lower()


def test_converter_discovers_sharegpt_tar_collection():
    config = OmegaConf.create(
        {
            "type": "share_gpt",
            "manifest_filepath": "rows.jsonl",
            "tarred_audio_filepaths": "audio-{000..001}.tar",
            "tar_lookup_mode": "collection",
            "tar_routing_index": "rows.sgroute",
        }
    )

    collections = convert_indexes_to_idxpack.discover_pack_collections(config)

    assert [(item.role, item.kind) for item in collections] == [
        ("manifest", "jsonl"),
        ("tar_collection", "nemo_tar"),
    ]
    assert all(item.offsets_required for item in collections)


def test_converter_collection_mode_rejects_native_tar_paths_only(tmp_path):
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text("{}\n")
    tar_path = tmp_path / "audio.tar"
    config_path = tmp_path / "sharegpt.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "type": "share_gpt",
                "manifest_filepath": str(manifest),
                "tarred_audio_filepaths": str(tar_path),
                "tar_lookup_mode": "collection",
                "tar_routing_index": "rows.sgroute",
            }
        )
    )

    result = CliRunner().invoke(
        convert_indexes_to_idxpack.main,
        [
            "--dry-run",
            "--native-tar-paths-only",
            "--output",
            str(tmp_path / "x.idxpack"),
            str(config_path),
        ],
    )

    assert result.exit_code != 0
    assert "collection mode" in result.output.lower()
    assert "--native-tar-paths-only" in result.output


class _Tokenizer:
    pad = 0
    unk_id = 0


def test_full_mode_builds_production_salm_dataset():
    config = OmegaConf.create(
        {
            "model": {"use_nemo_automodel": True, "packed_encoder_sequences": True},
            "data": {"train_ds": {"batch_tokens": 1024}},
        }
    )

    dataset = validate_dataloader._build_validation_dataset(config, _Tokenizer(), mode="full")

    assert isinstance(dataset, SALMDataset)
    assert dataset.strict_audio_loading is True


def test_full_mode_batch_validation_and_id_only_summary():
    class Conversation:
        def __init__(self, id_):
            self.id = id_

    batch = {
        "packed_audio_samples": object(),
        "audio_cu_seqlens": object(),
        "audio_lens": object(),
        "input_ids": object(),
        "loss_mask": object(),
        "conversations": [Conversation("a"), Conversation("b")],
    }

    validate_dataloader._validate_full_batch(batch, step=0)
    assert validate_dataloader._extract_cuts(batch) == (
        ['semantic:"a"', 'semantic:"b"'],
        0,
    )
    assert validate_dataloader._extract_semantic_cut_ids(batch, 2) == ["a", "b"]

    with pytest.raises(Exception, match="audio payload"):
        validate_dataloader._validate_full_batch(
            {
                "audio_lens": object(),
                "input_ids": object(),
                "loss_mask": object(),
                "conversations": [Conversation("a")],
            },
            step=1,
        )


def test_validate_dataloader_extracts_content_free_source_labels():
    batch = {
        "cut_ids": [["a", "b"]],
        "source_groups": [["group-a", "group-b"]],
        "source_ids": [["1", "47"]],
    }
    assert validate_dataloader._extract_source_labels(batch, 2) == (
        ["group-a", "group-b"],
        ["1", "47"],
    )


def test_build_indexes_collection_mode_requires_tar_sources():
    config = OmegaConf.create(
        {
            "type": "share_gpt",
            "manifest_filepath": "rows.jsonl",
            "tar_lookup_mode": "collection",
            "tar_routing_filepath": "rows.sgroute",
        }
    )

    with pytest.raises(ValueError, match="tarred_audio_filepaths"):
        build_indexes.discover(config, [])


def test_converter_wds_v2_requires_data_dir():
    config = OmegaConf.create(
        {
            "type": "share_gpt_webdataset",
            "wds_sample_index_version": 2,
        }
    )

    with pytest.raises(ValueError, match="data_dir"):
        convert_indexes_to_idxpack.discover_pack_collections(config)
