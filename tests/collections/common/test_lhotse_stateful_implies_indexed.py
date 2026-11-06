# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""``use_stateful_dataloader: true`` with ``indexed`` unset reads every source through its index.

Before, an unset ``indexed`` let each source type decide: a ``lhotse`` ``cuts_path`` used its ``.idx`` if one
happened to exist, while ``nemo`` manifests streamed. A stateful blend could then end up partly indexed, which
can't be checkpointed with bucketing and resumes by O(N) replay instead of O(1).
"""
import json
import logging

import pytest
from lhotse import CutSet
from lhotse.indexing import create_jsonl_index, create_tar_index, index_file_path
from lhotse.testing.dummies import DummyManifest
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.dataloader import (
    get_lhotse_dataloader_from_config,
    get_lhotse_sampler_from_config,
    make_structured_with_schema_warnings,
)
from nemo.collections.common.data.lhotse.indexed_adapters import IndexedTarMemberReader, MissingIndexError
from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index as create_nemo_tar_index


class _Identity:
    def __getitem__(self, cuts):
        return cuts


@pytest.fixture
def lhotse_path(tmp_path):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=20, with_data=False)
    for c in cuts:
        c.duration = 1.0
    path = tmp_path / "cuts.jsonl"
    cuts.to_file(path)
    return path


@pytest.fixture
def nemo_path(tmp_path):
    path = tmp_path / "nemo_manifest.json"
    with path.open("w") as f:
        for i in range(20):
            print(json.dumps({"audio_filepath": f"{i}.wav", "duration": 1.0, "text": "x"}), file=f)
    return path


def _cfg(sources, **overrides):
    return {
        "use_lhotse": True,
        "seed": 0,
        "shard_seed": 0,
        "shuffle": True,
        "num_workers": 0,
        "use_stateful_dataloader": True,
        "force_map_dataset": True,
        "batch_size": 2,
        "input_cfg": [{"weight": 1.0, **src} for src in sources],
        **overrides,
    }


def _sampler_cuts(cfg):
    config = make_structured_with_schema_warnings(OmegaConf.create(cfg))
    sampler, _ = get_lhotse_sampler_from_config(config, global_rank=0, world_size=1)
    (cuts,) = sampler.cuts
    return cuts


def _nemo(path, **kw):
    return {"type": "nemo", "manifest_filepath": str(path), **kw}


def _lhotse(path, **kw):
    return {"type": "lhotse", "cuts_path": str(path), **kw}


@pytest.mark.unit
def test_stateful_unset_indexed_reads_every_source_through_its_index(lhotse_path, nemo_path):
    create_jsonl_index(lhotse_path)
    create_jsonl_index(nemo_path)
    # Before: the nemo source streamed (an unset 'indexed' is falsy for it) while the lhotse one was indexed.
    assert _sampler_cuts(_cfg([_nemo(nemo_path)])).is_indexed
    assert _sampler_cuts(_cfg([_nemo(nemo_path), _lhotse(lhotse_path)])).is_indexed


@pytest.mark.unit
def test_stateful_unset_indexed_top_level_manifest_filepath(nemo_path):
    create_jsonl_index(nemo_path)
    cfg = _cfg([])
    del cfg["input_cfg"]
    cfg["manifest_filepath"] = str(nemo_path)
    assert _sampler_cuts(cfg).is_indexed


@pytest.mark.unit
def test_stateful_unset_indexed_missing_index_raises_at_startup(lhotse_path, nemo_path):
    create_jsonl_index(lhotse_path)
    with pytest.raises(MissingIndexError) as exc:
        _sampler_cuts(_cfg([_nemo(nemo_path), _lhotse(lhotse_path)]))
    msg = str(exc.value)
    assert str(nemo_path) in msg
    assert "scripts/dataloading/build_indexes.py" in msg
    assert "indexed: false" in msg
    # The index isn't silently built next to the data.
    assert not nemo_path.with_name(nemo_path.name + ".idx").exists()


@pytest.fixture
def shar_dir(tmp_path):
    import numpy as np
    import soundfile as sf
    from lhotse import Recording

    cuts = []
    for i in range(4):
        wav = tmp_path / f"{i}.wav"
        sf.write(wav, np.zeros(1600, dtype=np.float32), 16000)
        cuts.append(Recording.from_file(wav).to_cut())
    out = tmp_path / "shar"
    out.mkdir()
    CutSet(cuts).to_shar(out, fields={"recording": "wav"}, shard_size=2, compress_jsonl=False)
    for p in out.glob("*.idx"):  # start without indexes, whatever the Shar writer does
        p.unlink()
    return out


@pytest.fixture
def nemo_tarred_paths(tmp_path):
    import io
    import tarfile

    import numpy as np
    import soundfile as sf

    tar_path, manifest_path = tmp_path / "audio_0.tar", tmp_path / "manifest_0.json"
    with tarfile.open(tar_path, "w") as tar, manifest_path.open("w") as mf:
        for i in range(4):
            buf = io.BytesIO()
            sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
            info = tarfile.TarInfo(f"{i}.wav")
            info.size = buf.tell()
            buf.seek(0)
            tar.addfile(info, buf)
            print(json.dumps({"audio_filepath": f"{i}.wav", "duration": 0.1, "text": "x", "shard_id": 0}), file=mf)
    return manifest_path, tar_path


def _all_sources(paths):
    lhotse_path, nemo_path, shar, (tarred_manifest, tar) = paths
    return {
        "nemo": (_nemo(nemo_path), [nemo_path]),
        "lhotse": (_lhotse(lhotse_path), [lhotse_path]),
        "lhotse_shar": ({"type": "lhotse_shar", "shar_path": str(shar)}, sorted(shar.glob("*.*[lr]"))),
        "nemo_tarred": (
            {"type": "nemo_tarred", "manifest_filepath": str(tarred_manifest), "tarred_audio_filepaths": str(tar)},
            [tarred_manifest, tar],
        ),
    }


@pytest.mark.unit
@pytest.mark.parametrize("source", ["nemo", "lhotse", "lhotse_shar", "nemo_tarred"])
@pytest.mark.parametrize("stateful,indexed", [(True, None), (True, True), (False, True)])
def test_indexed_read_requires_prebuilt_indexes(
    lhotse_path, nemo_path, shar_dir, nemo_tarred_paths, source, stateful, indexed
):
    # Implied (stateful + unset) and explicit 'indexed: true' alike: a missing index is an error at startup,
    # and no index is built next to the data.
    src, data_files = _all_sources((lhotse_path, nemo_path, shar_dir, nemo_tarred_paths))[source]
    data_files = [f for f in data_files if f.suffix in (".jsonl", ".json", ".tar")]
    cfg = _cfg([src], use_stateful_dataloader=stateful, indexed=indexed)
    with pytest.raises(MissingIndexError, match="build_indexes.py") as exc:
        _sampler_cuts(cfg)
    assert str(data_files[0]) in str(exc.value)
    assert not any(f.with_name(f.name + ".idx").exists() for f in data_files)

    # With the indexes built ahead of time, the source is read through them.
    for f in data_files:
        if f.suffix == ".tar" and source == "nemo_tarred":
            create_nemo_tar_index(f, f.with_name(f.name + ".idx"))
        elif f.suffix == ".tar":
            create_tar_index(f)
        else:
            create_jsonl_index(f)
    assert _sampler_cuts(cfg).is_indexed


@pytest.mark.unit
def test_missing_index_error_respects_indexes_root(nemo_path, tmp_path):
    idx_root = tmp_path / "idx_root"
    create_jsonl_index(nemo_path)  # next to the data, but the config points at indexes_root
    cfg = _cfg([_nemo(nemo_path)], indexes_root=str(idx_root))
    with pytest.raises(MissingIndexError, match=str(idx_root)):
        _sampler_cuts(cfg)
    create_jsonl_index(nemo_path, output_path=index_file_path(nemo_path, str(idx_root)))
    assert _sampler_cuts(cfg).is_indexed


@pytest.mark.unit
def test_offline_tools_can_still_build_tar_indexes(nemo_tarred_paths):
    _, tar = nemo_tarred_paths
    with pytest.raises(MissingIndexError):
        IndexedTarMemberReader(tar)
    assert len(IndexedTarMemberReader(tar, auto_create_index=True)) == 4  # explicit opt-in


@pytest.mark.unit
def test_stateful_unset_indexed_streaming_only_type_raises(tmp_path):
    txt = tmp_path / "text.txt"
    txt.write_text("hello\nworld\n")
    src = {"type": "txt", "paths": str(txt), "language": "en"}
    with pytest.raises(ValueError, match="indexed: false"):
        _sampler_cuts(_cfg([src], force_map_dataset=False, force_iterable_dataset=True))
    # An explicit opt-out streams it.
    assert not _sampler_cuts(
        _cfg([{**src, "indexed": False}], force_map_dataset=False, force_iterable_dataset=True)
    ).is_indexed


@pytest.mark.unit
@pytest.mark.parametrize("where", ["top", "entry"])
def test_stateful_explicit_indexed_false_streams(lhotse_path, nemo_path, where):
    sources = [_nemo(nemo_path), _lhotse(lhotse_path)]
    if where == "top":
        cfg = _cfg(sources, indexed=False)
    else:
        cfg = _cfg([{**s, "indexed": False} for s in sources])
    assert not _sampler_cuts(cfg).is_indexed
    assert not nemo_path.with_name(nemo_path.name + ".idx").exists()
    assert not lhotse_path.with_name(lhotse_path.name + ".idx").exists()


@pytest.mark.unit
def test_non_stateful_unset_indexed_keeps_auto_detection(lhotse_path, nemo_path):
    create_jsonl_index(lhotse_path)
    create_jsonl_index(nemo_path)
    assert _sampler_cuts(_cfg([_lhotse(lhotse_path)], use_stateful_dataloader=False)).is_indexed
    assert not _sampler_cuts(_cfg([_nemo(nemo_path)], use_stateful_dataloader=False)).is_indexed
    # No index: streams, and no index is created.
    other = lhotse_path.with_name("other.jsonl")
    other.write_text(lhotse_path.read_text())
    assert not _sampler_cuts(_cfg([_lhotse(other)], use_stateful_dataloader=False)).is_indexed
    assert not other.with_name("other.jsonl.idx").exists()


def _partly_indexed_warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and "only partly indexed" in r.getMessage()]


@pytest.mark.unit
def test_non_stateful_partly_indexed_blend_warns_once(lhotse_path, nemo_path, caplog):
    create_jsonl_index(lhotse_path)
    cfg = _cfg([_nemo(nemo_path), _lhotse(lhotse_path)], use_stateful_dataloader=False)
    with caplog.at_level(logging.WARNING):
        get_lhotse_dataloader_from_config(OmegaConf.create(cfg), global_rank=0, world_size=1, dataset=_Identity())
    assert len(_partly_indexed_warnings(caplog)) == 1


@pytest.mark.unit
@pytest.mark.parametrize("indexed", [True, False])
def test_non_stateful_explicit_indexed_blend_does_not_warn(lhotse_path, nemo_path, caplog, indexed):
    create_jsonl_index(lhotse_path)
    create_jsonl_index(nemo_path)
    cfg = _cfg([_nemo(nemo_path), _lhotse(lhotse_path)], use_stateful_dataloader=False, indexed=indexed)
    with caplog.at_level(logging.WARNING):
        get_lhotse_dataloader_from_config(OmegaConf.create(cfg), global_rank=0, world_size=1, dataset=_Identity())
    assert _partly_indexed_warnings(caplog) == []


@pytest.mark.unit
def test_stateful_unset_indexed_resolves_in_nested_input_cfg(lhotse_path, nemo_path, tmp_path):
    create_jsonl_index(lhotse_path)
    create_jsonl_index(nemo_path)
    group = {
        "type": "group",
        "input_cfg": [
            {"weight": 0.5, **_nemo(nemo_path)},
            {"weight": 0.5, **_lhotse(lhotse_path, indexed=None)},  # explicit null == unset
        ],
    }
    assert _sampler_cuts(_cfg([group])).is_indexed

    # Missing index in a nested entry fails too.
    other = tmp_path / "other.json"
    other.write_text(nemo_path.read_text())
    group["input_cfg"].append({"weight": 0.5, **_nemo(other)})
    with pytest.raises(MissingIndexError, match="other.json"):
        _sampler_cuts(_cfg([group]))

    # A nested opt-out is respected (no bucketing here, so a partly indexed blend is allowed).
    group["input_cfg"][-1]["indexed"] = False
    cuts = _sampler_cuts(_cfg([group]))
    assert not cuts.is_indexed
    assert not other.with_name("other.json.idx").exists()


def _multi_cfg(sub_a, sub_b, **overrides):
    return {
        "use_lhotse": True,
        "multi_config": True,
        "sampler_fusion": "round_robin",
        "seed": 0,
        "shard_seed": 0,
        "shuffle": True,
        "num_workers": 0,
        "use_stateful_dataloader": True,
        "a": {"force_map_dataset": True, "batch_size": 2, **sub_a},
        "b": {"force_map_dataset": True, "batch_size": 2, **sub_b},
        **overrides,
    }


def _multi_sampler_cuts(cfg):
    dl = get_lhotse_dataloader_from_config(OmegaConf.create(cfg), global_rank=0, world_size=1, dataset=_Identity())
    return [s.cuts[0] for s in dl.sampler.samplers]


@pytest.mark.unit
def test_stateful_unset_indexed_resolves_in_multi_config(lhotse_path, nemo_path):
    create_jsonl_index(lhotse_path)
    create_jsonl_index(nemo_path)
    a, b = _multi_sampler_cuts(_multi_cfg({"manifest_filepath": str(nemo_path)}, {"cuts_path": str(lhotse_path)}))
    assert a.is_indexed and b.is_indexed

    # A sub-config's explicit opt-out is kept (it used to be overwritten by the unset top-level value).
    a, b = _multi_sampler_cuts(
        _multi_cfg({"manifest_filepath": str(nemo_path), "indexed": False}, {"cuts_path": str(lhotse_path)})
    )
    assert not a.is_indexed and b.is_indexed

    # A top-level value still applies to every sub-config.
    a, b = _multi_sampler_cuts(
        _multi_cfg({"manifest_filepath": str(nemo_path)}, {"cuts_path": str(lhotse_path)}, indexed=False)
    )
    assert not a.is_indexed and not b.is_indexed


@pytest.mark.unit
def test_stateful_unset_indexed_multi_config_missing_index_raises(lhotse_path, nemo_path):
    create_jsonl_index(lhotse_path)
    with pytest.raises(MissingIndexError, match="build_indexes.py"):
        _multi_sampler_cuts(_multi_cfg({"manifest_filepath": str(nemo_path)}, {"cuts_path": str(lhotse_path)}))
