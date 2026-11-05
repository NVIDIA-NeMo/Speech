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
"""Index packs for uncompressed Lhotse Shar directories (``lhotse_shar``)."""
import itertools
import pickle

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.array import Array
from lhotse.shar.readers.indexed import LazyIndexedSharIterator
from lhotse.shar.writers import SharWriter
from omegaconf import OmegaConf
from scripts.dataloading.convert_indexes_to_idxpack import main

from nemo.collections.common.data.lhotse.cutset import read_dataset_config
from nemo.collections.common.data.lhotse.packed_shar import LazyPackedSharIterator


def _make_shar(root, name, n=11, shard_size=4, seed=0, compression="jsonl"):
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


def _convert(tmp_path, shar_dir, name="pack"):
    cfg = tmp_path / f"{name}.yaml"
    cfg.write_text(yaml.safe_dump([{"type": "lhotse_shar", "shar_path": str(shar_dir)}]))
    pack = tmp_path / f"{name}.idxpack"
    result = CliRunner().invoke(main, ["--output", str(pack), str(cfg)])
    assert result.exit_code == 0, result.output
    return pack


def _payload(cut):
    return (
        cut.id,
        cut.recording.sources[0].source,
        cut.feats.storage_key,
        cut.supervisions[0].text,
        str(cut.shard_origin),
    )


@pytest.fixture
def shar(tmp_path):
    d = _make_shar(tmp_path, "ds")
    assert len(list(d.glob("cuts.*.jsonl"))) == 3
    return d


@pytest.mark.parametrize("shuffle", [False, True])
def test_packed_matches_sidecar_reader(tmp_path, shar, shuffle):
    pack = _convert(tmp_path, shar)
    ref = LazyIndexedSharIterator(in_dir=str(shar), shuffle=shuffle, seed=7)
    got = LazyPackedSharIterator(pack, str(shar), shuffle=shuffle, seed=7)
    assert len(ref) == len(got) == 11
    a, b = list(ref), list(got)
    assert [_payload(x) for x in a] == [_payload(x) for x in b]
    assert a[3].to_dict() == b[3].to_dict()
    if shuffle:
        assert [c.id for c in b] != sorted(c.id for c in b) or len(b) == 1
    # second epoch is reshuffled identically
    assert [c.id for c in ref] == [c.id for c in got]


def test_random_access_and_negative_index(tmp_path, shar):
    pack = _convert(tmp_path, shar)
    ref = LazyIndexedSharIterator(in_dir=str(shar))
    got = LazyPackedSharIterator(pack, str(shar))
    for i in (0, 3, 4, 10, -1):
        assert _payload(ref[i]) == _payload(got[i])
    with pytest.raises(IndexError):
        got[11]


def test_state_dict_resume_and_pickle(tmp_path, shar):
    pack = _convert(tmp_path, shar)
    full = [c.id for c in LazyPackedSharIterator(pack, str(shar), shuffle=True, seed=3)]
    it = LazyPackedSharIterator(pack, str(shar), shuffle=True, seed=3)
    head = [c.id for c in itertools.islice(iter(it), 5)]
    sd = it.state_dict()
    resumed = pickle.loads(pickle.dumps(LazyPackedSharIterator(pack, str(shar), shuffle=True, seed=3)))
    resumed.load_state_dict(sd)
    assert head + [c.id for c in resumed] == full
    # The state is interchangeable with the sidecar reader's.
    ref = LazyIndexedSharIterator(in_dir=str(shar), shuffle=True, seed=3)
    ref.load_state_dict(sd)
    assert head + [c.id for c in ref] == full


def test_field_subset_and_unknown_field(tmp_path, shar):
    pack = _convert(tmp_path, shar)
    cut = next(iter(LazyPackedSharIterator(pack, str(shar), fields=["recording"])))
    assert cut.recording.sources[0].type == "memory" and cut.feats.storage_type != "memory_npy"
    with pytest.raises(ValueError, match="not packed"):
        LazyPackedSharIterator(pack, str(shar), fields=["nope"])


def test_pack_does_not_open_sidecars(tmp_path, shar, monkeypatch):
    pack = _convert(tmp_path, shar)
    for idx in shar.glob("*.idx"):
        idx.unlink()
    got = LazyPackedSharIterator(pack, str(shar))
    assert len(list(got)) == 11


def test_config_parser_index_pack(tmp_path, shar):
    pack = _convert(tmp_path, shar)
    cfg = OmegaConf.create(
        {
            "indexed": True,
            "force_finite": True,
            "shuffle": False,
            "shard_seed": 5,
            "index_pack_root": str(tmp_path),
            "input_cfg": [{"type": "lhotse_shar", "shar_path": str(shar), "index_pack": pack.name}],
        }
    )
    cuts, is_tarred = read_dataset_config(cfg)
    assert is_tarred
    assert isinstance(cuts.data, LazyPackedSharIterator)
    ids = [c.id for c in cuts]
    assert sorted(ids) == sorted(c.id for c in LazyIndexedSharIterator(in_dir=str(shar)))


def test_config_requires_indexed_and_existing_pack(tmp_path, shar):
    base = {
        "force_finite": True,
        "input_cfg": [{"type": "lhotse_shar", "shar_path": str(shar), "index_pack": "x.idxpack"}],
    }
    with pytest.raises(ValueError, match="requires indexed"):
        read_dataset_config(OmegaConf.create(base))
    with pytest.raises(FileNotFoundError):
        read_dataset_config(OmegaConf.create({**base, "indexed": True, "index_pack_root": str(tmp_path)}))


def test_converter_rejects_compressed_and_list_shar_path(tmp_path, shar):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump([{"type": "lhotse_shar", "shar_path": [str(shar)]}]))
    result = CliRunner().invoke(main, ["--output", str(tmp_path / "bad.idxpack"), str(cfg)])
    assert result.exit_code != 0


def test_converter_rejects_nonzero_trailing_bytes(tmp_path, shar):
    tar = sorted(shar.glob("feats.*.tar"))[0]
    with open(tar, "ab") as f:
        f.write(b"\x01" * 1024)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump([{"type": "lhotse_shar", "shar_path": str(shar)}]))
    result = CliRunner().invoke(main, ["--output", str(tmp_path / "c.idxpack"), str(cfg)])
    assert result.exit_code != 0
