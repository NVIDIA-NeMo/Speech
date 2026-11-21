# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compressed Shar coverage through NeMo's index builder and dataloader."""

import bz2
import io
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lhotse import CutSet, fastcopy
from lhotse.indexing import create_shar_index, index_exists
from lhotse.shar.writers import SharWriter
from lhotse.testing.dummies import DummyManifest, dummy_in_memory_features
from omegaconf import OmegaConf
from packaging.version import parse as parse_version
from scripts.dataloading import build_indexes

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config


class _Identity(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet) -> CutSet:
        return cuts


@pytest.fixture
def gzip_shar(tmp_path):
    pytest.importorskip("indexed_gzip")
    root = tmp_path / "shar"
    return root, _write_shar(root, create_index=False)


def _fields(root: Path) -> dict[str, list[str]]:
    return {
        "cuts": [str(path) for path in sorted(root.glob("cuts.*.jsonl.gz"))],
        "recording": [str(path) for path in sorted(root.glob("recording.*.tar"))],
    }


@pytest.mark.parametrize("declaration", ["directory", "fields"])
def test_gzip_shar_index_build_and_dataloader(gzip_shar, tmp_path, declaration, monkeypatch):
    root, expected_ids = gzip_shar
    mirror = tmp_path / "mirror"
    shar_path = str(root) if declaration == "directory" else _fields(root)
    entry = {"type": "lhotse_shar", "shar_path": shar_path, "indexed": True}
    jobs = []
    build_indexes.discover(entry, jobs, str(mirror))
    assert len([job for job in jobs if job.kind == build_indexes.JSONL]) == 3
    assert len([job for job in jobs if job.kind == build_indexes.WDS_TAR]) == 3
    for job in jobs:
        build_indexes._build_one(job)
        assert build_indexes._is_indexed(job)
    for path in _fields(root)["cuts"]:
        assert not Path(f"{path}.idx").exists()
        assert index_exists(path, index_path=build_indexes.IndexJob(path, build_indexes.JSONL, str(mirror)).idx_path())
    assert len(list(mirror.rglob("*.gzidx"))) == 3

    def reject_autobuild(*args, **kwargs):
        raise AssertionError("Runtime must use the prebuilt mirror sidecars")

    monkeypatch.setattr("lhotse.shar.readers.indexed.create_jsonl_index", reject_autobuild)
    monkeypatch.setattr("lhotse.shar.readers.indexed.create_tar_index", reject_autobuild)

    config = OmegaConf.create(
        {
            "shar_path": shar_path,
            "indexed": True,
            "indexes_root": str(mirror),
            "force_finite": True,
            "shuffle": False,
            "shard_seed": 0,
            "sample_rate": 16000,
            "batch_size": 2,
            "num_workers": 0,
            "drop_last": False,
        }
    )
    cuts, is_tarred = read_cutset_from_config(config)
    assert is_tarred
    assert cuts.is_indexed
    assert sorted(cut.id for cut in cuts) == sorted(expected_ids)
    loader = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())
    batch = next(iter(loader))
    assert len(batch) == 2
    assert isinstance(batch[0].load_audio(), np.ndarray)

    gzip_job = next(job for job in jobs if job.kind == build_indexes.JSONL)
    seek_index = Path(str(gzip_job.idx_path())[:-4] + ".gzidx")
    seek_index.unlink()
    assert not build_indexes._is_indexed(gzip_job)
    build_indexes._build_one(gzip_job, force=True)
    assert seek_index.is_file()
    assert build_indexes._is_indexed(gzip_job)


def test_gzip_shar_metadata_only_excludes_sidecars(gzip_shar):
    root, expected_ids = gzip_shar
    create_shar_index(root)
    assert list(root.glob("cuts.*.jsonl.gz.idx"))
    assert list(root.glob("cuts.*.jsonl.gz.gzidx"))
    config = OmegaConf.create(
        {"shar_path": str(root), "indexed": True, "metadata_only": True, "force_finite": True, "shard_seed": 0}
    )
    cuts, is_tarred = read_cutset_from_config(config)
    assert is_tarred
    assert cuts.is_indexed
    assert sorted(cut.id for cut in cuts) == sorted(expected_ids)


def test_streaming_shar_metadata_only_preserves_bzip2(tmp_path):
    expected = list(DummyManifest(CutSet, begin_id=0, end_id=2))
    path = tmp_path / "cuts.000000.jsonl.bz2"
    content = b"".join(json.dumps(cut.to_dict()).encode() + b"\n" for cut in expected)
    path.write_bytes(bz2.compress(content))
    Path(f"{path}.idx").write_bytes(b"ignored sidecar")
    Path(f"{path}.gzidx").write_bytes(b"ignored sidecar")
    config = OmegaConf.create(
        {"shar_path": str(tmp_path), "indexed": False, "metadata_only": True, "force_finite": True, "shard_seed": 0}
    )
    cuts, _ = read_cutset_from_config(config)
    assert [cut.id for cut in cuts] == [cut.id for cut in expected]


def test_gzip_shar_exact_restore_through_nemo_config(gzip_shar):
    root, expected_ids = gzip_shar
    create_shar_index(root)
    config = OmegaConf.create({"shar_path": str(root), "indexed": True, "force_finite": True, "shard_seed": 23})
    full, _ = read_cutset_from_config(config)
    expected_order = [cut.id for cut in full]
    assert sorted(expected_order) == sorted(expected_ids)

    interrupted, _ = read_cutset_from_config(config)
    stream = iter(interrupted)
    consumed = [next(stream).id for _ in range(5)]
    state = interrupted.state_dict()
    restored, _ = read_cutset_from_config(config)
    restored.load_state_dict(state)
    assert consumed + [cut.id for cut in restored] == expected_order


@pytest.mark.parametrize("num_workers", [0, 2])
def test_automatically_indexed_gzip_shar_stateful_dataloader_resume(tmp_path, num_workers):
    pytest.importorskip("torchdata.stateful_dataloader")
    pytest.importorskip("indexed_gzip")
    indexed_root = tmp_path / "auto-indexed"
    expected_ids = _write_shar(indexed_root, create_index=True)
    assert all(index_exists(path) for path in indexed_root.glob("cuts.*.jsonl.gz"))
    config = OmegaConf.create(
        {
            "shar_path": str(indexed_root),
            "indexed": True,
            "use_stateful_dataloader": True,
            "force_map_dataset": False,
            "force_finite": True,
            "shuffle": True,
            "shard_seed": 23,
            "seed": 23,
            "sample_rate": 16000,
            "batch_size": 2,
            "num_workers": num_workers,
            "drop_last": False,
        }
    )

    def make_loader():
        return get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())

    def batch_ids(batches):
        return [[cut.id for cut in batch] for batch in batches]

    loaders = []
    try:
        full = make_loader()
        loaders.append(full)
        expected = batch_ids(full)
        assert sorted(cut_id for batch in expected for cut_id in batch) == sorted(expected_ids)
        partial = make_loader()
        loaders.append(partial)
        iterator = iter(partial)
        first = next(iterator)
        assert isinstance(first[0].load_audio(), np.ndarray)
        prefix = batch_ids([first])
        state = deepcopy(partial.state_dict())
        resumed = make_loader()
        loaders.append(resumed)
        resumed.load_state_dict(state)
        assert prefix + batch_ids(resumed) == expected
    finally:
        for loader in loaders:
            iterator = getattr(loader, "_iterator", None)
            if num_workers and iterator is not None:
                iterator._shutdown_workers()


def _write_shar(root: Path, *, create_index: bool) -> list[str]:
    root.mkdir()
    cuts = DummyManifest(CutSet, begin_id=0, end_id=8, with_data=True)
    for cut in cuts:
        cut.features = None
        cut.custom = None
        cut.supervisions[0].custom = None
    with SharWriter(
        root, fields={"recording": "wav"}, shard_size=3, compress_jsonl=True, create_index=create_index
    ) as writer:
        for cut in cuts:
            writer.write(cut)
    return [cut.id for cut in cuts]


def _write_field_shar(root, *, compress, audio_format, array_format, include_cuts=True, long_ids=False):
    root.mkdir(exist_ok=True)
    original = list(DummyManifest(CutSet, begin_id=0, end_id=4, with_data=True))
    fields = {
        "recording": audio_format,
        "features": array_format,
        "custom_embedding": array_format,
        "custom_features": array_format,
        "custom_indexes": "numpy",
        "custom_recording": audio_format,
        "label": "jsonl",
    }
    for cut in original:
        if long_ids:
            cut.id = f"nested/{cut.id}-" + "音声" * 70
        cut.features = dummy_in_memory_features(0)
        cut.label = {"id": cut.id, "languages": ["en", "ja"]}
    original[-1].features = None
    for field in fields.keys() - {"recording", "features"}:
        original[-1].custom.pop(field)
    with SharWriter(
        root, fields=fields, shard_size=3, compress_jsonl=compress, create_index=False, include_cuts=include_cuts
    ) as writer:
        for cut in original:
            writer.write(cut)
    return writer.output_paths


@pytest.mark.parametrize("compress", [False, True])
def test_indexed_shar_long_unicode_cut_ids(tmp_path, compress):
    pytest.importorskip("indexed_gzip")
    root = tmp_path / "shar"
    _write_field_shar(root, compress=compress, audio_format="flac", array_format="lilcom", long_ids=True)
    create_shar_index(root)
    expected = {cut.id: _shar_payloads(cut) for cut in CutSet.from_shar(in_dir=root, indexed=False)}
    cuts, _ = read_cutset_from_config(
        OmegaConf.create({"shar_path": str(root), "indexed": True, "force_finite": True, "shard_seed": 0})
    )
    actual = {cut.id: _shar_payloads(cut) for cut in cuts}
    assert actual.keys() == expected.keys()
    for cut_id in actual:
        _assert_shar_payloads(actual[cut_id], expected[cut_id])


@pytest.mark.parametrize("compress", [False, True])
def test_indexed_shar_fields_added_without_rewriting_cuts(tmp_path, compress):
    pytest.importorskip("indexed_gzip")
    root = tmp_path / "shar"
    root.mkdir()
    with SharWriter(root, fields={}, shard_size=3, compress_jsonl=compress) as writer:
        for cut in DummyManifest(CutSet, begin_id=0, end_id=4):
            writer.write(fastcopy(cut, recording=None, features=None, custom=None))
    _write_field_shar(root, compress=compress, audio_format="flac", array_format="lilcom", include_cuts=False)
    create_shar_index(root)
    expected = {cut.id: _shar_payloads(cut) for cut in CutSet.from_shar(in_dir=root, indexed=False)}
    config = OmegaConf.create({"shar_path": str(root), "indexed": True, "force_finite": True, "shard_seed": 0})
    actual, _ = read_cutset_from_config(config)
    result = {cut.id: _shar_payloads(cut) for cut in actual}
    assert result.keys() == expected.keys()
    for cut_id in result:
        _assert_shar_payloads(result[cut_id], expected[cut_id])


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("array_format", ["numpy", "lilcom"])
@pytest.mark.parametrize("audio_format", ["wav", "flac", "mp3", "opus", "original"])
def test_shar_all_field_formats_through_nemo(tmp_path, compress, array_format, audio_format):
    pytest.importorskip("indexed_gzip")
    if array_format == "lilcom":
        pytest.importorskip("lilcom")
    root = tmp_path / "shar"
    paths = _write_field_shar(root, compress=compress, audio_format=audio_format, array_format=array_format)
    mirror = tmp_path / "mirror"
    jobs = []
    build_indexes.discover({"type": "lhotse_shar", "shar_path": paths}, jobs, str(mirror))
    assert len(jobs) == 16
    for job in jobs:
        build_indexes._build_one(job)
        assert build_indexes._is_indexed(job)
    assert bool(list(mirror.rglob("*.gzidx"))) == compress
    # Directory and explicit field declarations must produce identical payloads.
    reference, _ = read_cutset_from_config(
        OmegaConf.create({"shar_path": str(root), "indexed": False, "force_finite": True, "shard_seed": 0})
    )
    expected = {cut.id: _shar_payloads(cut) for cut in reference}
    for declaration in (str(root), paths):
        config = OmegaConf.create(
            {
                "shar_path": declaration,
                "indexed": True,
                "indexes_root": str(mirror),
                "force_finite": True,
                "shard_seed": 0,
                "shuffle": False,
                "sample_rate": 16000,
                "batch_size": 2,
                "num_workers": 0,
                "drop_last": False,
            }
        )
        cuts, _ = read_cutset_from_config(config)
        for cut in cuts:
            _assert_shar_payloads(_shar_payloads(cut), expected[cut.id])
        loader = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())
        actual = {cut.id: _shar_payloads(cut) for batch in loader for cut in batch}
        assert actual.keys() == expected.keys()
        for cut_id in expected:
            # The dataloader's existing resampling transform drops precomputed features.
            loader_expected = {field: value for field, value in expected[cut_id].items() if field != "features"}
            _assert_shar_payloads(actual[cut_id], loader_expected)


def _shar_payloads(cut):
    result = {"recording": cut.resample(16000).load_audio()}
    if cut.has_features:
        result["features"] = cut.load_features()
    for field in ("custom_embedding", "custom_features", "custom_indexes"):
        if cut.has_custom(field):
            result[field] = cut.load_custom(field)
    if cut.has_custom("custom_recording"):
        result["custom_recording"] = cut.custom_recording.resample(16000).load_audio()
    if cut.has_custom("label"):
        result["label"] = cut.label
    return result


def _assert_shar_payloads(actual, expected):
    assert actual.keys() == expected.keys()
    for field in expected:
        if field == "label":
            assert actual[field] == expected[field]
        else:
            np.testing.assert_allclose(actual[field], expected[field], atol=1e-6)


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("seekable", [False, True])
def test_shar_ais_index_builder_and_loading(tmp_path, monkeypatch, compress, seekable):
    """Exercise real AISRangeReader and AIStoreIOBackend with fake SDK transport."""
    pytest.importorskip("indexed_gzip")
    from lhotse.serialization import AIStoreIOBackend, BuiltinIOBackend, CompositeIOBackend, GzipIOBackend

    root = tmp_path / "shar"
    paths = _write_field_shar(root, compress=compress, audio_format="flac", array_format="lilcom")
    reference = CutSet.from_shar(in_dir=root, indexed=False)
    expected = {cut.id: _shar_payloads(cut) for cut in reference}
    objects = {}
    requests = []

    class ReadStream(io.BufferedIOBase):
        def __init__(self, payload):
            self.payload = io.BytesIO(payload)

        def read(self, size=-1):
            return self.payload.read(size)

        def readable(self):
            return True

    read_stream = io.BytesIO if seekable else ReadStream

    class Object:
        def __init__(self, url):
            self.url = url

        @property
        def props(self):
            return SimpleNamespace(size=len(objects[self.url]))

        def get_reader(self, byte_range=None):
            payload = objects[self.url]
            requests.append((self.url, byte_range))
            if byte_range is not None:
                start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
                assert 0 <= start <= end < len(payload)
                payload = payload[start : end + 1]
            return SimpleNamespace(as_file=lambda: read_stream(payload), read_all=lambda: payload)

    client = SimpleNamespace(get_object_from_url=Object)
    monkeypatch.setattr("lhotse.serialization.get_aistore_client", lambda: (client, parse_version("1.10.0")))
    monkeypatch.setattr(
        "lhotse.serialization.CURRENT_IO_BACKEND",
        CompositeIOBackend([GzipIOBackend(), AIStoreIOBackend(), BuiltinIOBackend()]),
    )
    remote_fields = {}
    for field, shards in paths.items():
        remote_fields[field] = []
        for path in shards:
            url = f"ais://bucket/shar/{Path(path).name}"
            objects[url] = Path(path).read_bytes()
            remote_fields[field].append(url)
    mirror = tmp_path / "mirror"
    jobs = []
    build_indexes.discover({"type": "lhotse_shar", "shar_path": remote_fields}, jobs, str(mirror))
    assert len(jobs) == 16
    for job in jobs:
        build_indexes._build_one(job)
        assert build_indexes._is_indexed(job)
    requests.clear()
    config = OmegaConf.create(
        {
            "shar_path": remote_fields,
            "indexed": True,
            "indexes_root": str(mirror),
            "force_finite": True,
            "shard_seed": 0,
        }
    )
    actual, _ = read_cutset_from_config(config)
    result = {cut.id: _shar_payloads(cut) for cut in actual}
    assert result.keys() == expected.keys()
    for cut_id in result:
        _assert_shar_payloads(result[cut_id], expected[cut_id])
    assert any(url.endswith(".tar") and range_ for url, range_ in requests)
    assert any(url.endswith(".jsonl.gz") and range_ for url, range_ in requests) == compress
    assert bool(list(mirror.rglob("*.gzidx"))) == compress
    assert not any(url.endswith((".idx", ".gzidx")) for url in objects)
