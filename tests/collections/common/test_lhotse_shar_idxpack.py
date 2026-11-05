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
"""Index packs for ``lhotse_shar`` entries (directory, list of directories, per-field mapping)."""
import importlib.util

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.array import Array
from lhotse.shar.writers import SharWriter
from omegaconf import OmegaConf
from scripts.dataloading.convert_indexes_to_idxpack import main

from nemo.collections.common.data.lhotse.cutset import read_dataset_config

requires_shar_index_pack = pytest.mark.skipif(
    importlib.util.find_spec("lhotse.shar.index_pack") is None,
    reason="requires a Lhotse version with Shar index-pack support",
)


def _make_shar(root, name, n=11, shard_size=4, seed=0):
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.RandomState(seed)
    cuts = []
    for i in range(n):
        cid = f"{name}-{i}"
        samples = (rng.randn(1600 + 100 * i) * 0.1).astype(np.float32)
        rec = Recording.from_bytes(_wav(samples), cid)
        c = MonoCut(
            id=cid,
            start=0.0,
            duration=rec.duration,
            channel=0,
            recording=rec,
            supervisions=[SupervisionSegment(id=cid, recording_id=cid, start=0.0, duration=rec.duration, text=cid)],
        )
        c.feats = Array(
            storage_type="memory_npy", storage_path="", storage_key=_npy(rng.randn(3 + i, 4)), shape=[3 + i, 4]
        )
        cuts.append(c)
    with SharWriter(
        str(d),
        fields={"recording": "flac", "feats": "numpy"},
        shard_size=shard_size,
        warn_unused_fields=False,
        compress_jsonl=False,
        create_index=True,
    ) as w:
        for c in cuts:
            w.write(c)
    return d


def _npy(a):
    import io

    b = io.BytesIO()
    np.save(b, a.astype(np.float16))
    return b.getvalue()


def _wav(x):
    import io

    import soundfile as sf

    b = io.BytesIO()
    sf.write(b, x, 16000, format="WAV", subtype="PCM_16")
    return b.getvalue()


def _convert(tmp_path, entries, name="pack"):
    cfg = tmp_path / f"{name}.yaml"
    cfg.write_text(yaml.safe_dump(entries))
    pack = tmp_path / "packs" / f"{name}.idxpack"
    return pack, CliRunner().invoke(main, ["--output", str(pack), str(cfg)])


def _read(tmp_path, entry, **loader_kwargs):
    cfg = {"indexed": True, "force_finite": True, "shuffle": False, "shard_seed": 5, "input_cfg": [entry]}
    cfg.update(loader_kwargs)
    return read_dataset_config(OmegaConf.create(cfg))[0]


def _payload(cut):
    return cut.id, cut.recording.sources[0].source, cut.feats.storage_key, cut.supervisions[0].text


def _drop_sidecars(*dirs):
    for d in dirs:
        for idx in d.glob("*.idx"):
            idx.unlink()


@pytest.fixture
def shar_dirs(tmp_path):
    return _make_shar(tmp_path, "a"), _make_shar(tmp_path, "b", n=7, seed=1)


@requires_shar_index_pack
def test_single_directory(tmp_path, shar_dirs):
    a, _ = shar_dirs
    entry = {"type": "lhotse_shar", "shar_path": str(a)}
    pack, result = _convert(tmp_path, [entry])
    assert result.exit_code == 0, result.output
    expected = [_payload(c) for c in _read(tmp_path, entry)]
    _drop_sidecars(a)
    packed = _read(tmp_path, {**entry, "index_pack": pack.name}, index_pack_root=str(pack.parent))
    assert [_payload(c) for c in packed] == expected
    assert len(expected) == 11


@requires_shar_index_pack
@pytest.mark.parametrize("weighted", [False, True])
def test_list_of_directories_share_one_pack(tmp_path, shar_dirs, weighted):
    a, b = shar_dirs
    shar_path = [[str(a), 1.0], [str(b), 2.0]] if weighted else [str(a), str(b)]
    entry = {"type": "lhotse_shar", "shar_path": shar_path}
    pack, result = _convert(tmp_path, [entry])
    assert result.exit_code == 0, result.output
    expected = sorted(_payload(c) for c in _read(tmp_path, entry))
    _drop_sidecars(a, b)
    packed = _read(tmp_path, {**entry, "index_pack": str(pack)})
    assert sorted(_payload(c) for c in packed) == expected
    assert len(expected) == 18


@requires_shar_index_pack
def test_per_field_mapping(tmp_path, shar_dirs):
    a, _ = shar_dirs
    entry = {
        "type": "lhotse_shar",
        "shar_path": {
            "cuts": [str(p) for p in sorted(a.glob("cuts.*.jsonl"))],
            "recording": [str(p) for p in sorted(a.glob("recording.*.tar"))],
        },
    }
    pack, result = _convert(tmp_path, [entry])
    assert result.exit_code == 0, result.output
    expected = [(c.id, c.recording.sources[0].source) for c in _read(tmp_path, entry)]
    _drop_sidecars(a)
    packed = _read(tmp_path, {**entry, "index_pack": str(pack)})
    assert [(c.id, c.recording.sources[0].source) for c in packed] == expected
    assert all(c.feats.storage_type != "memory_npy" for c in packed)


def test_metadata_only_ignores_the_pack(tmp_path, shar_dirs):
    # Also a regression test: the cuts glob must skip the .idx sidecars that sit next to the shards.
    a, _ = shar_dirs
    entry = {"type": "lhotse_shar", "shar_path": str(a), "index_pack": str(tmp_path / "missing.idxpack")}
    cuts = _read(tmp_path, entry, metadata_only=True)
    assert sorted(c.id for c in cuts) == sorted(f"a-{i}" for i in range(11))


@requires_shar_index_pack
def test_requires_indexed_and_existing_pack(tmp_path, shar_dirs):
    a, _ = shar_dirs
    entry = {"type": "lhotse_shar", "shar_path": str(a), "index_pack": "x.idxpack"}
    with pytest.raises(ValueError, match="requires indexed"):
        read_dataset_config(OmegaConf.create({"force_finite": True, "input_cfg": [entry]}))
    with pytest.raises(FileNotFoundError):
        _read(tmp_path, entry, index_pack_root=str(tmp_path))


@requires_shar_index_pack
def test_converter_rejects_shar_mixed_with_other_types(tmp_path, shar_dirs):
    a, _ = shar_dirs
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"audio_filepath": "x.wav", "duration": 1.0, "text": "x"}\n')
    entries = [
        {"type": "lhotse_shar", "shar_path": str(a)},
        {"type": "nemo", "manifest_filepath": str(manifest)},
    ]
    _, result = _convert(tmp_path, entries)
    assert result.exit_code != 0
    assert "separate input_cfg" in str(result.exception)
