# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Indexed (loose ``.idx``) nemo_tarred with manifest rows that can't be looked up positionally in the tar."""
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from click.testing import CliRunner
from lhotse.indexing import create_jsonl_index
from lhotse.indexing import create_tar_index as create_lhotse_tar_index
from scripts.dataloading import build_indexes

from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index
from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator
from nemo.collections.common.data.lhotse.nemo_tar_routing import nemo_tar_route_sidecar_paths

SR = 16000
MEMBER_SAMPLES = 1600


@pytest.mark.unit
@pytest.mark.parametrize("layout", ["subn", "aggregate"])
def test_indexed_nemo_tarred_route_loads_correct_audio_for_every_row(tmp_path, layout):
    data = _write_dataset(tmp_path, layout)
    _build_indexes(tmp_path, data)

    cuts = list(_iterator(data))

    expected = [row for row in data["rows"] if not row.get("_skipme")]
    assert [cut.supervisions[0].text for cut in cuts] == [row["text"] for row in expected]
    for cut, row in zip(cuts, expected):
        np.testing.assert_array_equal(cut.load_audio()[0], _member_audio(row["member"]))


@pytest.mark.unit
@pytest.mark.parametrize("layout", ["subn", "aggregate"])
def test_nemo_tarred_subn_rows_get_unique_cut_ids_identical_in_streaming_and_indexed_mode(tmp_path, layout):
    data = _write_dataset(tmp_path, layout)
    _build_indexes(tmp_path, data)

    indexed = {cut.supervisions[0].text: cut.id for cut in _iterator(data)}
    streaming = {
        cut.supervisions[0].text: cut.id
        for cut in LazyNeMoTarredIterator(
            manifest_path=data["manifest_spec"],
            tar_paths=data["tar_spec"],
            fault_tolerant_audio_loading=False,
            # The aggregate fixture's tars contain members that its manifest doesn't reference.
            skip_missing_manifest_entries=True,
        )
    }

    assert streaming == indexed
    assert len(set(indexed.values())) == len(indexed)
    # ``-subN`` rows are keyed by their own audio_filepath; other rows keep the member name as their ID.
    assert sorted(indexed.values()) == sorted(r["audio_filepath"] for r in data["rows"] if not r.get("_skipme"))


@pytest.mark.unit
@pytest.mark.parametrize("layout", ["subn", "aggregate"])
def test_indexed_nemo_tarred_route_resumes_exactly_with_stateful_dataloader(tmp_path, layout):
    data = _write_dataset(tmp_path, layout)
    _build_indexes(tmp_path, data)

    uninterrupted = list(_dataloader(data))
    assert sorted(text for batch in uninterrupted for text, _ in batch) == sorted(
        row["text"] for row in data["rows"] if not row.get("_skipme")
    )

    first = _dataloader(data)
    it = iter(first)
    consumed = [next(it), next(it)]
    state = first.state_dict()
    resumed = _dataloader(data)
    resumed.load_state_dict(state)

    assert consumed + list(resumed) == uninterrupted


@pytest.mark.unit
def test_indexed_nemo_tarred_aggregate_manifest_routes_rows_to_tars_in_ais_get_batch_mode(tmp_path, monkeypatch):
    data = _write_dataset(tmp_path, "aggregate")
    _build_indexes(tmp_path, data)
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")

    cuts = list(_iterator(data))

    expected = [row for row in data["rows"] if not row.get("_skipme")]
    assert [cut.tar_origin for cut in cuts] == [data["tars"][row["shard_id"]] for row in expected]
    assert [cut.recording.sources[0].source for cut in cuts] == [
        f"{data['tars'][row['shard_id']]}/{row['member']}.wav" for row in expected
    ]


@pytest.mark.unit
def test_indexed_nemo_tarred_aggregate_manifest_requires_route_sidecars(tmp_path):
    data = _write_dataset(tmp_path, "aggregate")
    for path in [*data["manifests"], *data["tars"]]:
        if path.endswith(".tar"):
            create_tar_index(path, f"{path}.idx")
        else:
            create_jsonl_index(path)

    with pytest.raises(ValueError, match=r"one manifest .* paired with 3 tars.*build_indexes\.py"):
        _iterator(data)


@pytest.mark.unit
def test_indexed_nemo_tarred_subn_rows_without_route_fail_at_construction(tmp_path):
    data = _write_dataset(tmp_path, "subn")
    for path in [*data["manifests"], *data["tars"]]:
        if path.endswith(".tar"):
            create_tar_index(path, f"{path}.idx")
        else:
            create_jsonl_index(path)

    with pytest.raises(ValueError, match=r"no row-to-member route sidecar.*'-subN' rows.*build_indexes\.py"):
        _iterator(data)


@pytest.mark.unit
def test_build_indexes_writes_routes_only_when_needed_and_fixes_lhotse_paired_tar_index(tmp_path):
    data = _write_dataset(tmp_path, "subn")
    # Start shard 1 from the wrong (Shar paired) layout, as written by ``lhotse index tar``.
    create_lhotse_tar_index(data["tars"][1])

    _build_indexes(tmp_path, data)

    jobs = [build_indexes.IndexJob(m, build_indexes.NEMO_TAR_ROUTE, tar_paths=(t,)) for m, t in data["pairs"]]
    assert all(build_indexes._is_indexed(job) for job in jobs)
    # Shard 0 repeats members (route needed); shard 1 maps rows 1:1 to members (positional, no route).
    assert nemo_tar_route_sidecar_paths(data["manifests"][0])[0].is_file()
    assert not nemo_tar_route_sidecar_paths(data["manifests"][1])[0].exists()
    assert len(list(_iterator(data))) == len(data["rows"])

    # A rebuilt tar index makes the route stale, so the next run rebuilds the route too.
    create_tar_index(data["tars"][0], f"{data['tars'][0]}.idx")
    assert not build_indexes._is_indexed(jobs[0])
    assert jobs[0] in build_indexes._select_todo([jobs[0]])


def _write_dataset(tmp_path: Path, layout: str) -> dict:
    """Write a tiny nemo_tarred dataset whose manifest rows can't be looked up positionally in its tars.

    ``subn``: two shards, each manifest paired with one tar; shard 0 has repeated ``-subN`` rows
    (5 rows for 3 tar members) and shard 1 is 1:1. ``aggregate``: one manifest paired with three tars,
    rows interleaved across tars by ``shard_id``, with a ``-subN`` row and a ``_skipme`` row.
    """
    if layout == "subn":
        tar_members = {0: ["a", "b", "c"], 1: ["d", "e"]}
        manifest_rows = {
            0: [("a", ""), ("a", "-sub1"), ("b", ""), ("c", ""), ("c", "-sub1")],
            1: [("d", ""), ("e", "")],
        }
    else:
        tar_members = {0: ["a", "b"], 1: ["c", "d"], 2: ["e", "f"]}
        manifest_rows = {None: [("e", ""), ("a", ""), ("c", ""), ("a", "-sub1"), ("skip", ""), ("f", ""), ("d", "")]}
    member_shard = {member: shard for shard, members in tar_members.items() for member in members}
    tars = []
    for shard, members in tar_members.items():
        tar_path = tmp_path / f"audio_{shard}.tar"
        with tarfile.open(tar_path, "w") as archive:
            for member in members:
                payload = _wav_bytes(_member_audio(member))
                info = tarfile.TarInfo(f"{member}.wav")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        tars.append(str(tar_path))
    manifests, all_rows = [], []
    for shard, rows in manifest_rows.items():
        manifest_path = tmp_path / ("manifest.json" if shard is None else f"manifest_{shard}.json")
        with manifest_path.open("w") as f:
            for member, sub in rows:
                row = {
                    "audio_filepath": f"{member}{sub}.wav",
                    "duration": MEMBER_SAMPLES / SR,
                    "text": f"row{len(all_rows)}",
                    "shard_id": member_shard.get(member, 0) if shard is None else shard,
                    "sampling_rate": SR,
                    "member": member,
                }
                if member == "skip":
                    row["_skipme"] = True
                f.write(json.dumps(row) + "\n")
                all_rows.append(row)
        manifests.append(str(manifest_path))
    if layout == "subn":
        pairs = list(zip(manifests, tars))
        manifest_spec = str(tmp_path / "manifest__OP_0..1_CL_.json")
        tar_spec = str(tmp_path / "audio__OP_0..1_CL_.tar")
    else:
        pairs = []
        manifest_spec = manifests[0]
        tar_spec = str(tmp_path / "audio__OP_0..2_CL_.tar")
    return {
        "rows": all_rows,
        "manifests": manifests,
        "tars": tars,
        "pairs": pairs,
        "manifest_spec": manifest_spec,
        "tar_spec": tar_spec,
    }


def _build_indexes(tmp_path: Path, data: dict) -> None:
    cfg_path = tmp_path / "input_cfg.yaml"
    entry = {
        "type": "nemo_tarred",
        "manifest_filepath": data["manifest_spec"],
        "tarred_audio_filepaths": data["tar_spec"],
    }
    cfg_path.write_text(json.dumps([entry]))
    result = CliRunner().invoke(build_indexes.main, [str(cfg_path), "--executor", "thread"])
    assert result.exit_code == 0, result.output


def _iterator(data: dict) -> LazyNeMoTarredIterator:
    return LazyNeMoTarredIterator(
        manifest_path=data["manifest_spec"],
        tar_paths=data["tar_spec"],
        indexed=True,
        fault_tolerant_audio_loading=False,
    )


def _dataloader(data: dict):
    config = {
        "use_lhotse": True,
        "input_cfg": [
            {
                "type": "nemo_tarred",
                "manifest_filepath": data["manifest_spec"],
                "tarred_audio_filepaths": data["tar_spec"],
            }
        ],
        "indexed": True,
        "force_finite": True,
        "use_stateful_dataloader": True,
        "shuffle": True,
        "seed": 0,
        "shard_seed": 0,
        "batch_size": 2,
        "num_workers": 0,
    }
    return get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=_TextAndAudio())


class _TextAndAudio(torch.utils.data.Dataset):
    def __getitem__(self, cuts):
        return [(cut.supervisions[0].text, float(cut.load_audio().sum())) for cut in cuts]


def _member_audio(member: str) -> np.ndarray:
    """Distinct int16-representable audio per member (decoded as float32 by soundfile)."""
    seed = sum(map(ord, member))
    samples = ((seed * 37 + np.arange(MEMBER_SAMPLES)) % 2000 - 1000).astype(np.int16)
    return samples.astype(np.float32) / 32768


def _wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, SR, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
