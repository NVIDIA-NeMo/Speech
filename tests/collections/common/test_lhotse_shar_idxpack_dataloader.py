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
"""
Full dataloader plumbing for packed ``lhotse_shar`` sources mixed with ``nemo_tarred`` sources, packed and loose.

The blend is built through ``get_lhotse_dataloader_from_config`` with indexed, stateful, multi-worker loading, and is
compared against a run of the same configuration that reads loose sidecars only.
"""
import importlib.util
import io
import json
import os
import tarfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from click.testing import CliRunner
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.array import Array
from lhotse.dataset.collation import collate_audio
from lhotse.shar.writers import SharWriter
from omegaconf import OmegaConf
from scripts.dataloading.build_indexes import main as build_indexes_main
from scripts.dataloading.convert_indexes_to_idxpack import main as convert_main

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.cutset import read_dataset_config

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("lhotse.shar.index_pack") is None,
    reason="requires a Lhotse version with Shar index-pack support",
)

SR = 16000
N_SHAR = {"a": 13, "b": 9, "c": 7, "d": 11, "e": 5, "f": 10}  # cuts per Shar set
SHARD_SIZE = {"a": 4, "b": 9, "c": 3, "d": 5, "e": 5, "f": 4}  # "b" and "e" are a single shard
N_TARRED = {"t1": 12, "t2": 8, "t3": 10}  # samples per tarred set, 4 per tar shard


class IdsAndAudioDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet):
        audio, audio_lens = collate_audio(cuts)
        return {"ids": [c.id for c in cuts], "audio_sum": audio.double().sum(dim=1), "audio_lens": audio_lens}


def _wav_bytes(samples, fmt="WAV", subtype="PCM_16"):
    buf = io.BytesIO()
    sf.write(buf, samples, SR, format=fmt, subtype=subtype)
    return buf.getvalue()


def _audio(rng, i):
    return (rng.randn(SR // 2 + 400 * i) * 0.1).astype(np.float32)  # 0.5 s and up


def _make_shar(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.RandomState(sum(map(ord, name)))
    cuts = []
    for i in range(N_SHAR[name]):
        cid = f"{name}-{i}"
        rec = Recording.from_bytes(_wav_bytes(_audio(rng, i)), cid)
        cut = MonoCut(
            id=cid,
            start=0.0,
            duration=rec.duration,
            channel=0,
            recording=rec,
            supervisions=[SupervisionSegment(cid, cid, 0.0, rec.duration, text=cid)],
        )
        buf = io.BytesIO()
        np.save(buf, rng.randn(3 + i, 4).astype(np.float16))
        cut.feats = Array("memory_npy", "", buf.getvalue(), [3 + i, 4])
        cuts.append(cut)
    with SharWriter(
        str(d),
        fields={"recording": "flac", "feats": "numpy"},
        shard_size=SHARD_SIZE[name],
        warn_unused_fields=False,
        compress_jsonl=False,
        create_index=True,
    ) as writer:
        for cut in cuts:
            writer.write(cut)
    return d


def _make_tarred(root: Path, name: str) -> dict:
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.RandomState(sum(map(ord, name)))
    n = N_TARRED[name]
    num_shards = (n + 3) // 4
    for shard in range(num_shards):
        rows = []
        with tarfile.open(d / f"audio_{shard}.tar", "w") as tar:
            for i in range(shard * 4, min(n, shard * 4 + 4)):
                samples = _audio(rng, i)
                payload = _wav_bytes(samples)
                info = tarfile.TarInfo(f"{name}-{i}.wav")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
                rows.append(
                    {
                        "audio_filepath": f"{name}-{i}.wav",
                        "duration": len(samples) / SR,
                        "text": f"{name}-{i}",
                        "shard_id": shard,
                        "sampling_rate": SR,
                    }
                )
        (d / f"manifest_{shard}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {
        "type": "nemo_tarred",
        "manifest_filepath": str(d / f"manifest__OP_0..{num_shards - 1}_CL_.jsonl"),
        "tarred_audio_filepaths": str(d / f"audio__OP_0..{num_shards - 1}_CL_.tar"),
    }


def _mapping(shar_dir: Path, fields) -> dict:
    out = {"cuts": sorted(str(p) for p in shar_dir.glob("cuts.*.jsonl"))}
    for field in fields:
        out[field] = sorted(str(p) for p in shar_dir.glob(f"{field}.*.tar"))
    return out


class Blend:
    """Everything the tests need: data sources, their YAML groups, and packs built by the documented workflow."""

    def __init__(self, root: Path):
        self.root = root
        self.shar = {name: _make_shar(root, name) for name in N_SHAR}
        self.tarred = {name: _make_tarred(root, name) for name in N_TARRED}
        # Packed Shar group: a directory, a weighted list, an unweighted list and a per-field mapping, nested.
        self.shar_entries = [
            {"type": "lhotse_shar", "shar_path": str(self.shar["a"]), "weight": 2.0},
            {
                "type": "group",
                "weight": 3.0,
                "input_cfg": [
                    {
                        "type": "lhotse_shar",
                        "shar_path": [[str(self.shar["b"]), 1.0], [str(self.shar["c"]), 2.0]],
                        "weight": 1.0,
                    },
                    {"type": "lhotse_shar", "shar_path": [str(self.shar["d"]), str(self.shar["e"])], "weight": 2.0},
                ],
            },
            {
                "type": "lhotse_shar",
                "shar_path": _mapping(self.shar["f"], ["recording", "feats"]),
                "weight": 1.0,
            },
        ]
        # Packed tarred group (own pack) and a loose-sidecar tarred source.
        self.packed_tarred_entries = [
            {**self.tarred["t1"], "weight": 1.0},
            {**self.tarred["t2"], "weight": 1.0},
        ]
        self.loose_tarred_entry = {**self.tarred["t3"], "weight": 1.0}
        self.packs = root / "packs"
        self.shar_yaml = self._write_yaml("shar", self.shar_entries)
        self.tarred_yaml = self._write_yaml("tarred", self.packed_tarred_entries)
        self.loose_yaml = self._write_yaml("loose", [self.loose_tarred_entry])

        runner = CliRunner()
        for cfg in (self.shar_yaml, self.tarred_yaml, self.loose_yaml):
            result = runner.invoke(build_indexes_main, [str(cfg)])
            assert result.exit_code == 0, result.output
        for name, cfg in (("shar", self.shar_yaml), ("tarred", self.tarred_yaml)):
            result = runner.invoke(convert_main, ["--output", str(self.packs / f"{name}.idxpack"), str(cfg)])
            assert result.exit_code == 0, (result.output, result.exception)

    def _write_yaml(self, name, entries) -> Path:
        path = self.root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(entries))
        return path

    @property
    def total(self) -> int:
        return sum(N_SHAR.values()) + sum(N_TARRED.values())

    def input_cfg(self, packed: bool) -> list:
        def pack(name):
            return {"index_pack": f"{name}.idxpack"} if packed else {}

        return [
            {"type": "group", "weight": 5.0, "input_cfg": deepcopy(self.shar_entries), **pack("shar")},
            {"type": "group", "weight": 2.0, "input_cfg": deepcopy(self.packed_tarred_entries), **pack("tarred")},
            deepcopy(self.loose_tarred_entry) | {"weight": 1.0},
        ]

    def config(self, packed: bool, **overrides) -> OmegaConf:
        cfg = {
            "indexed": True,
            "use_stateful_dataloader": True,
            "force_map_dataset": False,
            "force_finite": True,
            "shuffle": True,
            "seed": 7,
            "shard_seed": 7,
            "sample_rate": SR,
            "num_workers": 3,
            "use_bucketing": True,
            "concurrent_bucketing": False,
            "num_buckets": 3,
            "bucket_buffer_size": 40,
            "shuffle_buffer_size": 20,
            "batch_duration": 4.0,
            "quadratic_duration": 15.0,
            "drop_last": False,
            "input_cfg": self.input_cfg(packed),
        }
        if packed:
            cfg.update(index_pack_root=str(self.packs), index_pack_max_open_files=3)
        cfg.update(overrides)
        return OmegaConf.create(cfg)

    def loader(self, packed: bool, **overrides):
        return get_lhotse_dataloader_from_config(
            self.config(packed, **overrides), global_rank=0, world_size=1, dataset=IdsAndAudioDataset()
        )


@pytest.fixture(scope="module")
def blend(tmp_path_factory) -> Blend:
    return Blend(tmp_path_factory.mktemp("shar_idxpack_blend"))


def _stop(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


def _drain(loader):
    batches = list(loader)
    _stop(loader)
    return batches


def _ids(batches):
    return [list(b["ids"]) for b in batches]


def _flat(batches):
    return [i for b in batches for i in b["ids"]]


@pytest.mark.parametrize("use_bucketing", [False, True])
def test_packed_blend_matches_sidecar_run(blend, use_bucketing):
    extra = {"use_bucketing": use_bucketing}
    expected = _drain(blend.loader(packed=False, **extra))
    actual = _drain(blend.loader(packed=True, **extra))
    assert _ids(actual) == _ids(expected)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a["audio_sum"], e["audio_sum"])
        torch.testing.assert_close(a["audio_lens"], e["audio_lens"])
    # The finite epoch covers every cut of every source exactly once, across all workers.
    ids = _flat(actual)
    assert len(ids) == blend.total
    assert len(set(ids)) == blend.total
    assert {i.rsplit("-", 1)[0] for i in ids} == set(N_SHAR) | set(N_TARRED)


def test_packed_blend_does_not_open_shar_sidecars(blend, tmp_path):
    """A pack must stand on its own: hide the Shar sidecars while the blend is read."""
    hidden = []
    for d in blend.shar.values():
        for idx in d.glob("*.idx"):
            target = tmp_path / f"{d.name}-{idx.name}"
            idx.rename(target)
            hidden.append((idx, target))
    try:
        ids = _flat(_drain(blend.loader(packed=True)))
    finally:
        for idx, target in hidden:
            target.rename(idx)
    assert len(set(ids)) == blend.total


@pytest.mark.parametrize("num_workers", [0, 2])
def test_state_dict_resume_is_exact(blend, num_workers):
    torchdata = pytest.importorskip("torchdata.stateful_dataloader")
    full = _drain(blend.loader(packed=True, num_workers=num_workers))
    assert len(full) > 6

    partial = blend.loader(packed=True, num_workers=num_workers)
    iterator = iter(partial)
    prefix = [next(iterator) for _ in range(4)]
    state = deepcopy(partial.state_dict())
    _stop(partial)

    resumed = blend.loader(packed=True, num_workers=num_workers)
    resumed.load_state_dict(state)
    suffix = _drain(resumed)
    assert _ids(prefix + suffix) == _ids(full)


def test_resume_across_packed_and_sidecar_runs(blend):
    """The iterator-graph state does not depend on whether offsets come from a pack or from sidecars."""
    pytest.importorskip("torchdata.stateful_dataloader")
    full = _drain(blend.loader(packed=False, num_workers=2))
    for first, second in [(True, False), (False, True)]:
        partial = blend.loader(packed=first, num_workers=2)
        iterator = iter(partial)
        prefix = [next(iterator) for _ in range(3)]
        state = deepcopy(partial.state_dict())
        _stop(partial)
        resumed = blend.loader(packed=second, num_workers=2)
        resumed.load_state_dict(state)
        assert _ids(prefix + _drain(resumed)) == _ids(full)


def _open_shar_fds(blend) -> int:
    roots = tuple(str(d) for d in blend.shar.values())
    count = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith(roots) and target.endswith((".tar", ".jsonl")):
            count += 1
    return count


@pytest.mark.parametrize("max_open_files", [1, 2, 4])
def test_open_file_descriptors_of_packed_shar_are_bounded(blend, max_open_files):
    """The Shar shard files held open through the pack stay within ``index_pack_max_open_files``."""
    shar_only = [
        {"type": "group", "weight": 1.0, "input_cfg": deepcopy(blend.shar_entries), "index_pack": "shar.idxpack"}
    ]
    peak, batches = 0, 0
    loader = blend.loader(
        packed=True,
        num_workers=0,
        use_bucketing=False,
        input_cfg=shar_only,
        index_pack_max_open_files=max_open_files,
    )
    for _ in loader:
        peak = max(peak, _open_shar_fds(blend))
        batches += 1
    assert batches > 5

    sidecar_only = [{**shar_only[0]}]
    del sidecar_only[0]["index_pack"]
    sidecar_peak = 0
    for _ in blend.loader(packed=False, num_workers=0, use_bucketing=False, input_cfg=sidecar_only):
        sidecar_peak = max(sidecar_peak, _open_shar_fds(blend))
    assert 0 < peak <= max_open_files, peak
    assert sidecar_peak > peak, (peak, sidecar_peak)


def test_input_cfg_yaml_references_as_documented(blend):
    """Groups that point at the YAML files given to ``build_indexes.py`` / the converter, as in the docs."""
    input_cfg = [
        {"type": "group", "weight": 5.0, "input_cfg": str(blend.shar_yaml), "index_pack": "shar.idxpack"},
        {"type": "group", "weight": 2.0, "input_cfg": str(blend.tarred_yaml), "index_pack": "tarred.idxpack"},
        {"type": "group", "weight": 1.0, "input_cfg": str(blend.loose_yaml)},
    ]
    actual = _drain(blend.loader(packed=True, input_cfg=input_cfg))
    expected = _drain(blend.loader(packed=False))
    assert sorted(_flat(actual)) == sorted(_flat(expected))
    assert len(set(_flat(actual))) == blend.total


@pytest.mark.parametrize("world_size", [2, 3])
def test_ranks_partition_the_blend(blend, world_size):
    per_rank = []
    for rank in range(world_size):
        loader = get_lhotse_dataloader_from_config(
            blend.config(packed=True, num_workers=2),
            global_rank=rank,
            world_size=world_size,
            dataset=IdsAndAudioDataset(),
        )
        per_rank.append(_flat(_drain(loader)))
    everything = [i for ids in per_rank for i in ids]
    assert len(everything) == blend.total
    assert len(set(everything)) == blend.total


def test_packed_blend_with_a_second_epoch(blend):
    """An infinite (non-finite) stream draws from every source; packs stay readable across many epochs."""
    loader = blend.loader(packed=True, force_finite=False, num_workers=2, use_bucketing=False)
    ids = []
    for batch in loader:
        ids.extend(batch["ids"])
        if len(ids) > 3 * blend.total:
            break
    _stop(loader)
    assert len(set(ids)) == blend.total
    assert {i.rsplit("-", 1)[0] for i in ids} == set(N_SHAR) | set(N_TARRED)


def test_converter_refuses_shar_mixed_with_tarred(blend, tmp_path):
    cfg = tmp_path / "mixed.yaml"
    cfg.write_text(yaml.safe_dump(blend.shar_entries[:1] + blend.packed_tarred_entries[:1]))
    out = tmp_path / "mixed.idxpack"
    result = CliRunner().invoke(convert_main, ["--output", str(out), str(cfg)])
    assert result.exit_code != 0
    assert "separate input_cfg" in str(result.exception)
    assert not out.exists()


@pytest.mark.parametrize("trailing_slash", [False, True])
def test_converter_and_runtime_agree_on_source_spelling(blend, tmp_path, trailing_slash):
    """Brace-expanded mapping values and directory spellings resolve to the same pack source in both places."""
    a = blend.shar["a"]
    n = len(list(a.glob("cuts.*.jsonl")))
    last = f"{n - 1:06d}"
    mapping = {
        "cuts": f"{a}/cuts.{{000000..{last}}}.jsonl",
        "recording": f"{a}/recording.{{000000..{last}}}.tar",
    }
    entries = [
        {"type": "lhotse_shar", "shar_path": mapping},
        {"type": "lhotse_shar", "shar_path": str(a) + ("/" if trailing_slash else "")},
    ]
    cfg = tmp_path / "spelling.yaml"
    cfg.write_text(yaml.safe_dump(entries))
    pack = tmp_path / "spelling.idxpack"
    result = CliRunner().invoke(convert_main, ["--output", str(pack), str(cfg)])
    assert result.exit_code == 0, (result.output, result.exception)
    for entry in entries:
        runtime = {
            "indexed": True,
            "force_finite": True,
            "shard_seed": 1,
            "input_cfg": [{**entry, "index_pack": str(pack)}],
        }
        packed = read_dataset_config(OmegaConf.create(runtime))[0]
        runtime["input_cfg"] = [entry]
        loose = read_dataset_config(OmegaConf.create(runtime))[0]
        assert [c.id for c in packed] == [c.id for c in loose]
        assert len(list(packed)) == N_SHAR["a"]


def test_gzip_cuts_shar_is_packed_and_read(tmp_path):
    pytest.importorskip("indexed_gzip")
    source = _make_shar(tmp_path, "d")
    gz = tmp_path / "gz"
    gz.mkdir()
    with SharWriter(
        str(gz),
        fields={"recording": "flac", "feats": "numpy"},
        shard_size=4,
        warn_unused_fields=False,
        compress_jsonl=True,
        create_index=True,
    ) as writer:
        for cut in CutSet.from_shar(in_dir=source):
            cut.custom.pop("shard_origin", None)
            cut.custom.pop("shar_epoch", None)
            writer.write(cut)
    assert list(gz.glob("cuts.*.jsonl.gz"))
    entry = {"type": "lhotse_shar", "shar_path": str(gz)}
    cfg = tmp_path / "gz.yaml"
    cfg.write_text(yaml.safe_dump([entry]))
    pack = tmp_path / "gz.idxpack"
    result = CliRunner().invoke(convert_main, ["--output", str(pack), str(cfg)])
    assert result.exit_code == 0, (result.output, result.exception)
    runtime = {
        "indexed": True,
        "force_finite": True,
        "shard_seed": 1,
        "input_cfg": [{**entry, "index_pack": str(pack)}],
    }
    packed = read_dataset_config(OmegaConf.create(runtime))[0]
    runtime["input_cfg"] = [entry]
    loose = read_dataset_config(OmegaConf.create(runtime))[0]
    assert [c.id for c in packed] == [c.id for c in loose]
    assert len(list(packed)) == N_SHAR["d"]
