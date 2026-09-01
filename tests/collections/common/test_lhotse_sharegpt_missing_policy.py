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
from lhotse.audio import AudioLoadingError
from lhotse.indexing import create_jsonl_index
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import cutset as cutset_module
from nemo.collections.common.data.lhotse.indexed_adapters import (
    TarSampleBundle,
    TarSampleMember,
    create_tar_index,
    create_wds_v2_tar_index,
)
from nemo.collections.common.data.lhotse.text_adapters import (
    NeMoMultimodalConversationShareGPTJsonlAdapter,
    NeMoMultimodalConversationShareGPTWebdatasetAdapter,
)


def _wav(duration: float = 0.02) -> bytes:
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


def _sharegpt_row(index: int, audio_name: str | None = None) -> dict:
    audio_name = audio_name or f"{index}.wav"
    return {
        "id": f"sample-{index}",
        "sound": audio_name,
        "conversations": [
            {"from": "human", "value": "Listen <sound>"},
            {"from": "gpt", "value": f"answer-{index}"},
        ],
    }


def _write_paired_manifest(path: Path, num_rows: int = 3) -> Path:
    path.write_text("".join(json.dumps(_sharegpt_row(index)) + "\n" for index in range(num_rows)))
    create_jsonl_index(path)
    return path


def _paired_adapter(
    manifest_path: Path,
    tar_path: Path,
    *,
    indexed: bool,
    skip_missing_manifest_entries: bool,
    fault_tolerant_audio_loading: bool,
):
    return NeMoMultimodalConversationShareGPTJsonlAdapter(
        manifest_filepath=str(manifest_path),
        tarred_audio_filepaths=str(tar_path),
        audio_locator_tag="[audio]",
        audio_placeholders=["<sound>"],
        indexed=indexed,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["missing", "decode"])
@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_paired_sharegpt_indexed_expected_audio_failure_is_resumable(
    tmp_path, failure, skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    manifest_path = _write_paired_manifest(tmp_path / "manifest.jsonl")
    payloads = [_wav(), b"not-an-audio-file", _wav()]
    members = [
        (f"{index}.wav", payload)
        for index, payload in enumerate(payloads)
        if not (failure == "missing" and index == 1)
    ]
    tar_path = _write_tar(tmp_path / "audio.tar", members)
    create_tar_index(tar_path, f"{tar_path}.idx")

    def build():
        return _paired_adapter(
            manifest_path,
            tar_path,
            indexed=True,
            skip_missing_manifest_entries=skip_missing_manifest_entries,
            fault_tolerant_audio_loading=fault_tolerant_audio_loading,
        )

    expected_error = KeyError if failure == "missing" else AudioLoadingError
    if not fault_tolerant_audio_loading:
        with pytest.raises(expected_error):
            list(build())
        return

    permissive = build()
    assert len(permissive) == 3
    assert [item.id for item in permissive] == ["sample-0", "sample-2"]
    with pytest.raises(IndexError, match="not decodable"):
        permissive[1]

    uninterrupted = build()
    source = iter(uninterrupted)
    assert next(source).id == "sample-0"
    state = uninterrupted.state_dict()
    expected_remainder = [item.id for item in source]

    resumed = build()
    resumed.load_state_dict(state)
    assert [item.id for item in resumed] == expected_remainder == ["sample-2"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "failure,expected_ids",
    [("missing", ["sample-0", "sample-1"]), ("decode", ["sample-0"])],
)
@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_paired_sharegpt_streaming_expected_audio_failure_obeys_policy(
    tmp_path, failure, expected_ids, skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    manifest_path = _write_paired_manifest(tmp_path / "manifest.jsonl")
    if failure == "missing":
        members = [("0.wav", _wav()), ("1.wav", _wav())]
    else:
        members = [("0.wav", _wav()), ("1.wav", b"not-audio"), ("2.wav", _wav())]
    tar_path = _write_tar(tmp_path / "audio.tar", members)

    adapter = _paired_adapter(
        manifest_path,
        tar_path,
        indexed=False,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )
    if not fault_tolerant_audio_loading:
        with pytest.raises(RuntimeError, match="Failed to load paired ShareGPT tar"):
            list(adapter)
    else:
        assert [item.id for item in adapter] == expected_ids


@pytest.mark.unit
@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_paired_sharegpt_streaming_extra_member_obeys_only_missing_manifest_policy(
    tmp_path, skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    manifest_path = _write_paired_manifest(tmp_path / "manifest.jsonl", num_rows=1)
    tar_path = _write_tar(
        tmp_path / "audio.tar",
        [("extra.wav", _wav()), ("0.wav", _wav())],
    )

    adapter = _paired_adapter(
        manifest_path,
        tar_path,
        indexed=False,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )
    if skip_missing_manifest_entries:
        assert [item.id for item in adapter] == ["sample-0"]
    else:
        with pytest.raises(ValueError, match="no corresponding JSONL entry"):
            list(adapter)


@pytest.mark.unit
@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_paired_sharegpt_streaming_surplus_tar_member_obeys_only_missing_manifest_policy(
    tmp_path, skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    manifest_path = _write_paired_manifest(tmp_path / "manifest.jsonl", num_rows=1)
    tar_path = _write_tar(
        tmp_path / "audio.tar",
        [("0.wav", _wav()), ("extra.wav", _wav())],
    )
    adapter = _paired_adapter(
        manifest_path,
        tar_path,
        indexed=False,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )

    if skip_missing_manifest_entries:
        assert [item.id for item in adapter] == ["sample-0"]
    else:
        with pytest.raises(ValueError, match="no corresponding JSONL entry"):
            list(adapter)


def _write_wds(root: Path, *, version: int) -> Path:
    root.mkdir()
    members = []
    for index in range(3):
        members.append(
            (
                f"{index}.json",
                json.dumps(_sharegpt_row(index)).encode("utf-8"),
            )
        )
        members.append((f"{index}.wav", b"not-an-audio-file" if index == 1 else _wav()))
    tar_path = _write_tar(root / "shard.tar", members)
    (root / "wids-meta.json").write_text(json.dumps({"shardlist": [{"url": "shard.tar", "nsamples": 3}]}))
    if version == 1:
        create_tar_index(tar_path, f"{tar_path}.idx")
    else:
        create_wds_v2_tar_index(tar_path)
    return root


def _wds_adapter(
    data_dir: Path,
    *,
    version: int,
    indexed: bool,
    skip_missing_manifest_entries: bool,
    fault_tolerant_audio_loading: bool,
):
    return NeMoMultimodalConversationShareGPTWebdatasetAdapter(
        data_dir=str(data_dir),
        audio_locator_tag="[audio]",
        audio_placeholders=["<sound>"],
        indexed=indexed,
        wds_sample_index_version=version,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "version,indexed",
    [(1, False), (1, True), (2, True)],
)
@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_sharegpt_wds_audio_decode_failure_obeys_policy_and_indexed_resume(
    tmp_path, version, indexed, skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    data_dir = _write_wds(tmp_path / "wds", version=version)

    if not fault_tolerant_audio_loading:
        with pytest.raises(AudioLoadingError, match="Failed to decode ShareGPT audio payload"):
            list(
                _wds_adapter(
                    data_dir,
                    version=version,
                    indexed=indexed,
                    skip_missing_manifest_entries=skip_missing_manifest_entries,
                    fault_tolerant_audio_loading=False,
                )
            )
        return

    def build():
        return _wds_adapter(
            data_dir,
            version=version,
            indexed=indexed,
            skip_missing_manifest_entries=skip_missing_manifest_entries,
            fault_tolerant_audio_loading=fault_tolerant_audio_loading,
        )

    assert [item.id for item in build()] == ["sample-0", "sample-2"]
    if not indexed:
        return

    direct = build()
    assert len(direct) == 3
    with pytest.raises(IndexError, match="not decodable"):
        direct[1]

    uninterrupted = build()
    source = iter(uninterrupted)
    assert next(source).id == "sample-0"
    state = uninterrupted.state_dict()
    expected_remainder = [item.id for item in source]

    resumed = build()
    resumed.load_state_dict(state)
    assert [item.id for item in resumed] == expected_remainder == ["sample-2"]


@pytest.mark.unit
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_sharegpt_wds_v1_rejects_cross_sample_member_pairing(
    tmp_path,
    indexed,
    skip_missing_manifest_entries,
    fault_tolerant_audio_loading,
):
    data_dir = tmp_path / "wds"
    data_dir.mkdir()
    tar_path = _write_tar(
        data_dir / "shard.tar",
        [
            ("0.json", json.dumps(_sharegpt_row(0)).encode("utf-8")),
            ("1.wav", _wav()),
            ("1.json", json.dumps(_sharegpt_row(1)).encode("utf-8")),
            ("2.wav", _wav()),
        ],
    )
    (data_dir / "wids-meta.json").write_text(json.dumps({"shardlist": [{"url": "shard.tar", "nsamples": 2}]}))
    if indexed:
        create_tar_index(tar_path, f"{tar_path}.idx")
    adapter = _wds_adapter(
        data_dir,
        version=1,
        indexed=indexed,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )

    with pytest.raises(ValueError, match="different sample keys"):
        list(adapter)


@pytest.mark.unit
def test_sharegpt_wds_missing_member_is_skippable_but_ambiguity_is_structural():
    missing = TarSampleBundle(
        "missing",
        _sharegpt_row(0, audio_name="missing.wav"),
        (
            TarSampleMember("left.wav", _wav()),
            TarSampleMember("right.wav", _wav()),
        ),
    )
    ambiguous = TarSampleBundle(
        "ambiguous",
        _sharegpt_row(0, audio_name="same.wav"),
        (
            TarSampleMember("left/same.wav", _wav()),
            TarSampleMember("right/same.wav", _wav()),
        ),
    )

    permissive = object.__new__(NeMoMultimodalConversationShareGPTWebdatasetAdapter)
    permissive.audio_locator_tag = "[audio]"
    permissive.audio_placeholders = ["<sound>"]
    permissive.token_equivalent_duration = None
    permissive.wds_sample_index_version = 2
    permissive.skip_missing_manifest_entries = True

    assert permissive._build_wds_sample(missing, sample_idx=0) is None
    with pytest.raises(ValueError, match="ambiguous audio member basename"):
        permissive._build_wds_sample(ambiguous, sample_idx=1)


@pytest.mark.unit
def test_sharegpt_wds_parser_receives_loader_wide_missing_entry_policy(monkeypatch):
    captured = {}

    def fake_adapter(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        cutset_module,
        "NeMoMultimodalConversationShareGPTWebdatasetAdapter",
        fake_adapter,
    )
    config = OmegaConf.create(
        {
            "data_dir": "/unused",
            "audio_locator_tag": "[audio]",
            "shuffle": False,
            "shard_seed": 0,
            "force_finite": True,
            "skip_missing_manifest_entries": True,
        }
    )

    cuts, is_tarred = cutset_module.read_share_gpt_webdataset_as_conversation(config)

    assert list(cuts) == []
    assert is_tarred is True
    assert captured["skip_missing_manifest_entries"] is True
