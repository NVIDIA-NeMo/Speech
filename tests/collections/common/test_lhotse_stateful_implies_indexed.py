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

import pytest
import torch
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


@pytest.mark.unit
@pytest.mark.parametrize("supported", [True, False])
def test_from_shar_gets_auto_create_index_false_when_lhotse_supports_it(monkeypatch, supported):
    # Remote Shar directories can't be listed for the NeMo-side pre-check, so Lhotse's own
    # 'auto_create_index=False' switch is passed whenever the installed version has it.
    from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config

    calls = []
    dummy = DummyManifest(CutSet, begin_id=0, end_id=2)

    if supported:

        def fake_from_shar(fields=None, in_dir=None, *, indexed=None, auto_create_index=True, **kwargs):
            calls.append({"indexed": indexed, "auto_create_index": auto_create_index})
            return dummy

    else:

        def fake_from_shar(fields=None, in_dir=None, *, indexed=None, **kwargs):
            assert "auto_create_index" not in kwargs
            calls.append({"indexed": indexed})
            return dummy

    monkeypatch.setattr(CutSet, "from_shar", staticmethod(fake_from_shar))
    cfg = OmegaConf.create({"shar_path": "ais://bucket/shar", "indexed": True, "force_finite": True})
    read_cutset_from_config(cfg)
    expected = {"indexed": True, "auto_create_index": False} if supported else {"indexed": True}
    assert calls == [expected]


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


class _LoadAudio:
    """Loads and collates the batch's audio, like the ASR datasets do."""

    def __getitem__(self, cuts):
        from lhotse.dataset import AudioSamples

        audio, audio_lens = AudioSamples(fault_tolerant=False)(cuts)
        return {"ids": [c.id for c in cuts], "audio": audio, "audio_lens": audio_lens}


@pytest.fixture
def audio_sources(tmp_path):
    """Four short WAVs of distinct lengths, described by a NeMo manifest and by a Lhotse cuts manifest."""
    import numpy as np
    import soundfile as sf
    from lhotse import Recording

    nemo_manifest, cuts = tmp_path / "audio_nemo.json", []
    with nemo_manifest.open("w") as f:
        for i in range(4):
            wav = tmp_path / f"utt{i}.wav"
            sf.write(wav, np.full(1600 * (i + 1), 0.01 * (i + 1), dtype=np.float32), 16000)
            print(json.dumps({"audio_filepath": str(wav), "duration": 0.1 * (i + 1), "text": f"utt {i}"}), file=f)
            cuts.append(Recording.from_file(wav, recording_id=f"lhotse-utt{i}").to_cut())
    lhotse_manifest = tmp_path / "audio_cuts.jsonl"
    CutSet(cuts).to_file(lhotse_manifest)
    return {"nemo": nemo_manifest, "lhotse": lhotse_manifest}


def _build_audio_loader(cfg):
    return get_lhotse_dataloader_from_config(OmegaConf.create(cfg), global_rank=0, world_size=1, dataset=_LoadAudio())


def _assert_collated(batch):
    assert len(batch["ids"]) == 2
    assert batch["audio"].shape[0] == 2
    assert batch["audio_lens"].tolist() == [int((batch["audio"][i] != 0).sum()) for i in range(2)]


@pytest.mark.unit
@pytest.mark.parametrize(
    "sources,overrides,expect_indexed",
    [
        (["nemo"], {}, True),  # stateful + unset -> indexed
        (["lhotse"], {}, True),
        (["nemo", "lhotse"], {}, True),  # blend, both indexed
        (["nemo"], {"indexed": True}, True),  # explicit
        (["nemo", "lhotse"], {"indexed": False}, False),  # explicit opt-out streams, no index needed
    ],
)
def test_stateful_dataloader_end_to_end(audio_sources, sources, overrides, expect_indexed):
    entries = [_nemo(audio_sources["nemo"]) if s == "nemo" else _lhotse(audio_sources["lhotse"]) for s in sources]
    cfg = _cfg(entries, **overrides)
    data_files = [audio_sources[s] for s in sources]

    if expect_indexed:
        # Startup rejects missing indexes without creating them ...
        with pytest.raises(MissingIndexError):
            _build_audio_loader(cfg)
        assert not any(f.with_name(f.name + ".idx").exists() for f in data_files)
        # ... and loads once they're built ahead of time.
        for f in data_files:
            create_jsonl_index(f)
    assert _sampler_cuts(cfg).is_indexed == expect_indexed
    if not expect_indexed:
        assert not any(f.with_name(f.name + ".idx").exists() for f in data_files)

    # Consume collated batches through the real (stateful) dataloader, then check checkpoint/restore.
    dl = _build_audio_loader(cfg)
    it = iter(dl)
    _assert_collated(next(it))
    state = dl.state_dict()
    second = next(it)
    _assert_collated(second)

    restored = _build_audio_loader(cfg)
    restored.load_state_dict(state)
    second_restored = next(iter(restored))
    assert second_restored["ids"] == second["ids"]
    assert torch.equal(second_restored["audio"], second["audio"])


@pytest.mark.unit
def test_non_stateful_unset_indexed_end_to_end_auto_detects(audio_sources):
    # Without the stateful dataloader: a Lhotse manifest with a sidecar is indexed, a NeMo manifest streams.
    create_jsonl_index(audio_sources["lhotse"])
    for source, expect_indexed in (("lhotse", True), ("nemo", False)):
        entry = _lhotse(audio_sources["lhotse"]) if source == "lhotse" else _nemo(audio_sources["nemo"])
        cfg = _cfg([entry], use_stateful_dataloader=False)
        assert _sampler_cuts(cfg).is_indexed == expect_indexed
        _assert_collated(next(iter(_build_audio_loader(cfg))))
    assert not audio_sources["nemo"].with_name(audio_sources["nemo"].name + ".idx").exists()


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
def test_standalone_tar_reader_keeps_auto_building_by_default(nemo_tarred_paths):
    # The standalone reader's default is unchanged: it builds a missing index. Only the dataloader adapters
    # pass auto_create_index=False.
    _, tar = nemo_tarred_paths
    idx = tar.with_name(tar.name + ".idx")
    reader = IndexedTarMemberReader(tar)
    assert idx.exists()
    assert len(reader) == 4
    name, payload = reader[0]
    assert name == "0.wav" and payload[:4] == b"RIFF"


@pytest.mark.unit
@pytest.mark.parametrize("adjacent", [True, False])
def test_tar_reader_auto_create_index_false_requires_prebuilt_index(nemo_tarred_paths, tmp_path, adjacent):
    _, tar = nemo_tarred_paths
    idx = tar.with_name(tar.name + ".idx") if adjacent else tmp_path / "elsewhere" / "audio_0.tar.idx"
    idx.parent.mkdir(exist_ok=True)
    with pytest.raises(MissingIndexError):
        IndexedTarMemberReader(tar, idx_path=idx, auto_create_index=False)
    assert not idx.exists()
    create_nemo_tar_index(tar, idx)
    reader = IndexedTarMemberReader(tar, idx_path=idx, auto_create_index=False)
    assert [reader[i][0] for i in range(len(reader))] == [f"{i}.wav" for i in range(4)]


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
def test_stateful_unset_indexed_resolution_does_not_mutate_callers_config(tmp_path, lhotse_path):
    from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config

    txt = tmp_path / "text.txt"
    txt.write_text("hello\nworld\n")
    cfg = OmegaConf.create(
        {"use_stateful_dataloader": True, "input_cfg": [{"type": "txt", "paths": str(txt), "language": "en"}]}
    )
    for _ in range(2):  # a failed read must not turn the implied value into an explicit one
        with pytest.raises(ValueError, match="only supports streaming"):
            read_cutset_from_config(cfg)
        assert cfg.get("indexed") is None

    create_jsonl_index(lhotse_path)
    cfg = OmegaConf.create({"use_stateful_dataloader": True, "input_cfg": [_lhotse(lhotse_path)]})
    for _ in range(2):
        cuts, _ = read_cutset_from_config(cfg)
        assert cuts.is_indexed
        assert cfg.get("indexed") is None


def _conversation_cuts_path(tmp_path, name, begin):
    from lhotse.testing.dummies import dummy_cut, dummy_supervision

    path = tmp_path / f"{name}.jsonl"
    CutSet(
        [dummy_cut(i, supervisions=[dummy_supervision(i, text=f"text {i}")]) for i in range(begin, begin + 2)]
    ).to_file(path)
    return path


def _as_conversation(*sources):
    return {
        "type": "lhotse_as_conversation",
        "audio_locator_tag": "[audio]",
        "token_equivalent_duration": 0.08,
        "input_cfg": list(sources),
    }


@pytest.mark.unit
def test_stateful_converter_respects_nested_opt_out(tmp_path):
    # The converter reads its nested input_cfg recursively; a nested 'indexed: false' must stream instead of
    # making the enclosing converter fail as if it were a streaming-only source.
    path = _conversation_cuts_path(tmp_path, "a", 0)
    cuts = _sampler_cuts(_cfg([_as_conversation(_lhotse(path, indexed=False))]))
    assert not cuts.is_indexed
    assert sorted(c.id for c in cuts) == ["dummy-mono-cut-0000", "dummy-mono-cut-0001"]

    # Without an opt-out, the nested source is indexed (and needs its index).
    with pytest.raises(MissingIndexError):
        _sampler_cuts(_cfg([_as_conversation(_lhotse(path))]))
    create_jsonl_index(path)
    assert _sampler_cuts(_cfg([_as_conversation(_lhotse(path))])).is_indexed


@pytest.mark.unit
def test_stateful_converter_nested_mixed_blend(tmp_path):
    indexed_path = _conversation_cuts_path(tmp_path, "a", 0)
    streamed_path = _conversation_cuts_path(tmp_path, "b", 10)
    create_jsonl_index(indexed_path)
    conv = _as_conversation(
        {"weight": 0.5, **_lhotse(indexed_path)}, {"weight": 0.5, **_lhotse(streamed_path, indexed=False)}
    )
    # Allowed without bucketing ...
    assert not _sampler_cuts(_cfg([conv])).is_indexed
    # ... and rejected by the blend check when bucketing can't checkpoint it.
    bucketing = dict(
        use_bucketing=True,
        num_buckets=2,
        bucket_duration_bins=[1.5],
        batch_duration=3.0,
        bucket_buffer_size=20,
        concurrent_bucketing=False,
    )
    with pytest.raises(ValueError, match="can't checkpoint this data blend"):
        _sampler_cuts(_cfg([conv], **bucketing))


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


def _build_and_collect_partly_indexed_warnings(monkeypatch, cfg) -> list[str]:
    # Spy on NeMo's logger directly rather than using caplog: whether NeMo's logger propagates to the root
    # logger depends on what other tests ran earlier in the same session.
    from nemo.collections.common.data.lhotse import dataloader as dataloader_module

    messages = []
    original = dataloader_module.logging.warning

    def spy(msg, *args, **kwargs):
        messages.append(str(msg))
        return original(msg, *args, **kwargs)

    monkeypatch.setattr(dataloader_module.logging, "warning", spy)
    get_lhotse_dataloader_from_config(OmegaConf.create(cfg), global_rank=0, world_size=1, dataset=_Identity())
    return [m for m in messages if "only partly indexed" in m]


@pytest.mark.unit
def test_non_stateful_partly_indexed_blend_warns_once(lhotse_path, nemo_path, monkeypatch):
    create_jsonl_index(lhotse_path)
    cfg = _cfg([_nemo(nemo_path), _lhotse(lhotse_path)], use_stateful_dataloader=False)
    assert len(_build_and_collect_partly_indexed_warnings(monkeypatch, cfg)) == 1


@pytest.mark.unit
@pytest.mark.parametrize("indexed", [True, False])
def test_non_stateful_explicit_indexed_blend_does_not_warn(lhotse_path, nemo_path, monkeypatch, indexed):
    create_jsonl_index(lhotse_path)
    create_jsonl_index(nemo_path)
    cfg = _cfg([_nemo(nemo_path), _lhotse(lhotse_path)], use_stateful_dataloader=False, indexed=indexed)
    assert _build_and_collect_partly_indexed_warnings(monkeypatch, cfg) == []


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
