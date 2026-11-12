# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests: every NeMo indexed adapter must produce disjoint slices
across (DP rank x DataLoader worker) shards.

The bug this guards against: each adapter's ``_iter_indexed`` previously
iterated ``range(0, total_len)`` with no call to ``get_worker_partition()``,
so under multi-rank training every rank yielded the same items
(see ``sweeps/0909/debugging-duplication.md``). All 7 buggy adapters now
delegate position+topology to ``PartitionedIndexedIterator``; this file
asserts that contract at the adapter level so the next refactor can't quietly
regress it.

Each test simulates the env-var setup ``worker_init_fn`` would perform in a
DataLoader worker subprocess, builds the adapter with ``indexed=True``, walks
every (rank in range(world_size)) instance, and asserts:

* per-rank slices are pairwise disjoint;
* union over all ranks equals the full manifest (each example seen exactly
  once across the world).
"""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from lhotse import CutSet
from lhotse.audio import AudioLoadingError
from lhotse.dataset import AudioSamples
from lhotse.dataset.dataloading import LHOTSE_USE_WORKER_PARTITION
from lhotse.indexing import create_jsonl_index, read_index
from lhotse.serialization import load_jsonl, save_to_jsonl
from lhotse.shar.lazy_pointer import decode_pointer
from lhotse.testing.dummies import DummyManifest

from nemo.collections.common.data.lhotse import indexed_adapters, nemo_adapters, text_adapters

_PARTITION_ENV_KEYS = ("RANK", "WORLD_SIZE", LHOTSE_USE_WORKER_PARTITION)


@contextmanager
def _env_partition(rank: int, world_size: int):
    """Mimic the worker-subprocess env that ``worker_init_fn`` sets."""
    saved = {k: os.environ.get(k) for k in _PARTITION_ENV_KEYS}
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ[LHOTSE_USE_WORKER_PARTITION] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _collect_disjoint_per_rank(build_iter_for_rank, world_size: int) -> tuple[list, set]:
    """Run an adapter across every rank in ``range(world_size)`` and return
    ``(per_rank_id_lists, union_of_all_ids)``. Asserts pairwise disjointness."""
    per_rank: list[list] = []
    union: set = set()
    for rank in range(world_size):
        with _env_partition(rank=rank, world_size=world_size):
            ids = list(build_iter_for_rank())
        # Disjointness against every prior rank.
        for prev in per_rank:
            assert set(prev).isdisjoint(ids), (
                f"rank {rank} slice overlaps prior rank: " f"{sorted(set(prev) & set(ids))}"
            )
        per_rank.append(ids)
        union.update(ids)
    return per_rank, union


# ---------------------------------------------------------------------------
# Fixture: 20 single-channel cuts saved as one NeMo manifest + one tar file.
# Used by the LazyNeMoTarredIterator + parquet tests.
# ---------------------------------------------------------------------------

N_CUTS = 20


@pytest.fixture
def tmp_audio_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("audio")


@pytest.fixture
def nemo_tarred_manifest(tmp_audio_root) -> tuple[Path, Path]:
    """20-utterance NeMo tarred manifest (single shard) as
    (manifest_filepath, tarred_audio_filepath)."""
    from lhotse.serialization import SequentialJsonlWriter
    from lhotse.shar.writers import TarWriter

    cuts = DummyManifest(CutSet, begin_id=0, end_id=N_CUTS, with_data=True).save_audios(
        tmp_audio_root, progress_bar=False
    )
    root = tmp_audio_root / "tarred"
    root.mkdir(exist_ok=True)
    with (
        TarWriter(f"{root}/audios_0.tar", shard_size=None) as tar_writer,
        SequentialJsonlWriter(root / "manifest_0.jsonl") as mft_writer,
    ):
        for idx, cut in enumerate(cuts):
            src = cut.recording.sources[0].source
            name = Path(src).name
            with open(src, "rb") as f:
                tar_writer.write(name, BytesIO(f.read()))
            mft_writer.write(
                {
                    "audio_filepath": name,
                    "text": "irrelevant",
                    "duration": cut.duration,
                    "sampling_rate": cut.sampling_rate,
                    "lang": "en",
                    "shard_id": 0,
                    "cut_id": cut.id,
                }
            )
    manifest_path = Path(mft_writer.path)
    tar_path = root / "audios_0.tar"
    create_jsonl_index(manifest_path)
    indexed_adapters.create_tar_index(tar_path, Path(f"{tar_path}.idx"))
    return manifest_path, tar_path


# ---------------------------------------------------------------------------
# 1. LazyNeMoTarredIterator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", [1, 2, 4, 5])
def test_lazy_nemo_tarred_iterator_indexed_partition(nemo_tarred_manifest, world_size):
    manifest_path, tar_path = nemo_tarred_manifest

    def build():
        it = nemo_adapters.LazyNeMoTarredIterator(
            manifest_path=str(manifest_path),
            tar_paths=str(tar_path),
            indexed=True,
        )
        return [cut.id for cut in it]

    per_rank, union = _collect_disjoint_per_rank(build, world_size)
    assert len(union) == N_CUTS, f"missing {N_CUTS - len(union)} items at world_size={world_size}"
    # All items get covered at least once (each exactly once due to disjointness).
    assert sum(len(r) for r in per_rank) == N_CUTS


@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_lazy_nemo_tarred_audio_decode_policy_is_independent_of_missing_manifest_policy(
    nemo_tarred_manifest, monkeypatch, skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    manifest_path, tar_path = nemo_tarred_manifest
    monkeypatch.setenv("USE_AIS_GET_BATCH", "false")

    def fail_info(*args, **kwargs):
        raise RuntimeError("synthetic soundfile.info failure")

    monkeypatch.setattr(nemo_adapters.soundfile, "info", fail_info)

    adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=False,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )
    if fault_tolerant_audio_loading:
        assert list(adapter) == []
    else:
        with pytest.raises(RuntimeError, match=r"Failed to decode .*NeMo tarred audio member"):
            next(iter(adapter))


def test_lazy_nemo_tarred_audio_decode_is_fault_tolerant_by_default(nemo_tarred_manifest, monkeypatch):
    manifest_path, tar_path = nemo_tarred_manifest
    monkeypatch.setenv("USE_AIS_GET_BATCH", "false")
    monkeypatch.setattr(
        nemo_adapters.soundfile, "info", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("corrupt"))
    )
    adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=False,
    )
    assert list(adapter) == []


@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_lazy_nemo_tarred_missing_manifest_entry_obeys_only_skip_policy(
    nemo_tarred_manifest, monkeypatch, skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    manifest_path, tar_path = nemo_tarred_manifest
    monkeypatch.setenv("USE_AIS_GET_BATCH", "false")
    save_to_jsonl(list(load_jsonl(manifest_path))[1:], manifest_path)
    adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=False,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )
    if skip_missing_manifest_entries:
        assert len(list(adapter)) == N_CUTS - 1
    else:
        with pytest.raises(RuntimeError, match="Cannot locate JSON entry"):
            next(iter(adapter))


def test_lazy_nemo_tarred_indexed_defers_audio_until_selected(nemo_tarred_manifest, monkeypatch):
    manifest_path, tar_path = nemo_tarred_manifest
    monkeypatch.setenv("USE_AIS_GET_BATCH", "false")

    eager_cut = next(
        iter(
            nemo_adapters.LazyNeMoTarredIterator(
                manifest_path=str(manifest_path), tar_paths=str(tar_path), indexed=False
            )
        )
    )
    eager_audio = eager_cut.load_audio()

    def fail_info(*args, **kwargs):
        raise AssertionError("indexed candidate construction must not inspect audio payloads")

    monkeypatch.setattr(nemo_adapters.soundfile, "info", fail_info)
    monkeypatch.setattr(
        indexed_adapters.IndexedTarMemberReader,
        "_member_header",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("indexed candidate construction must not read tar headers")
        ),
    )
    adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=True,
        skip_missing_manifest_entries=False,
    )
    cut = next(iter(adapter))

    source = cut.recording.sources[0]
    assert source.type == "shar_ptr"
    assert "&n=" in source.source
    pointer_path, pointer_start, pointer_end = decode_pointer(source.source)
    tar_offsets = read_index(f"{tar_path}.idx")
    assert pointer_path == str(tar_path)
    assert (pointer_start, pointer_end) == (tar_offsets[0], tar_offsets[1])

    monkeypatch.setattr(
        "lhotse.serialization.TarAsDirBackend.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("indexed audio loading must not search the tar member table")
        ),
    )
    audio = cut.load_audio()
    assert audio.shape[-1] == cut.num_samples
    assert cut.sampling_rate == eager_cut.sampling_rate
    assert cut.duration == eager_cut.duration
    np.testing.assert_array_equal(audio, eager_audio)


def test_lazy_nemo_tarred_indexed_resolves_filtered_manifest_member_at_audio_load(nemo_tarred_manifest, monkeypatch):
    manifest_path, tar_path = nemo_tarred_manifest
    monkeypatch.setenv("USE_AIS_GET_BATCH", "false")
    rows = list(load_jsonl(manifest_path))[1:]
    save_to_jsonl(rows, manifest_path)
    create_jsonl_index(manifest_path)

    adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=True,
    )
    source = adapter[0].recording.sources[0]
    pointer_path, pointer_start, pointer_end = decode_pointer(source.source)
    tar_offsets = read_index(f"{tar_path}.idx")

    assert source.type == "shar_ptr"
    assert "&n=" in source.source
    assert pointer_path == str(tar_path)
    assert (pointer_start, pointer_end) == (tar_offsets[0], tar_offsets[1])
    assert source.load_audio().shape[-1] == adapter[0].num_samples


@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_lazy_nemo_tarred_indexed_missing_audio_obeys_only_audio_policy(
    nemo_tarred_manifest,
    monkeypatch,
    skip_missing_manifest_entries,
    fault_tolerant_audio_loading,
):
    manifest_path, tar_path = nemo_tarred_manifest
    monkeypatch.setenv("USE_AIS_GET_BATCH", "false")
    rows = list(load_jsonl(manifest_path))
    rows[0]["audio_filepath"] = "missing.wav"
    save_to_jsonl(rows, manifest_path)
    create_jsonl_index(manifest_path)

    adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=True,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )
    cuts = CutSet.from_cuts([adapter[0], adapter[1]])
    loader = AudioSamples(fault_tolerant=fault_tolerant_audio_loading)

    if fault_tolerant_audio_loading:
        _, _, surviving = loader(cuts)
        assert [cut.custom["cut_id"] for cut in surviving] == [rows[1]["cut_id"]]
    else:
        with pytest.raises(AudioLoadingError, match="no member named 'missing.wav'"):
            loader(cuts)


def test_lazy_nemo_tarred_indexed_requires_trusted_sampling_rate_without_audio_io(nemo_tarred_manifest, monkeypatch):
    manifest_path, tar_path = nemo_tarred_manifest
    rows = list(load_jsonl(manifest_path))
    rows[0].pop("sampling_rate")
    save_to_jsonl(rows, manifest_path)
    create_jsonl_index(manifest_path)

    monkeypatch.setattr(
        nemo_adapters.soundfile,
        "info",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not inspect audio metadata")),
    )
    adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=True,
    )
    with pytest.raises(ValueError, match="trusted source sampling-rate metadata"):
        adapter[0]

    fallback_adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=True,
        input_sampling_rate=16000,
    )
    assert fallback_adapter[0].sampling_rate == 16000

    rows[0]["sample_rate"] = 8000
    save_to_jsonl(rows, manifest_path)
    create_jsonl_index(manifest_path)
    legacy_adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=True,
    )
    assert legacy_adapter[0].sampling_rate == 8000


def test_lazy_nemo_tarred_indexed_uses_nonstandard_manifest_sampling_rate_for_resampling(tmp_path, monkeypatch):
    sampling_rate = 8000
    audio_name = "eight-khz.wav"
    payload = BytesIO()
    nemo_adapters.soundfile.write(payload, np.zeros(sampling_rate, dtype=np.float32), sampling_rate, format="WAV")
    tar_path = tmp_path / "audio_0.tar"
    with tarfile.open(tar_path, "w:") as archive:
        info = tarfile.TarInfo(audio_name)
        info.size = len(payload.getvalue())
        archive.addfile(info, BytesIO(payload.getvalue()))
    indexed_adapters.create_tar_index(tar_path, Path(f"{tar_path}.idx"))
    manifest_path = tmp_path / "manifest.jsonl"
    save_to_jsonl(
        [
            {
                "audio_filepath": audio_name,
                "duration": 1.0,
                "sampling_rate": sampling_rate,
                "text": "eight kilohertz",
            }
        ],
        manifest_path,
    )
    create_jsonl_index(manifest_path)

    monkeypatch.setattr(
        nemo_adapters.soundfile,
        "info",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not inspect audio metadata")),
    )
    adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=True,
    )
    cut = adapter[0]
    assert cut.sampling_rate == sampling_rate
    resampled = cut.resample(16000)
    assert resampled.sampling_rate == 16000
    assert resampled.load_audio().shape[-1] == 16000


@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_lazy_nemo_tarred_missing_duration_is_always_a_manifest_error(
    nemo_tarred_manifest, monkeypatch, skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    manifest_path, tar_path = nemo_tarred_manifest
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    rows = list(load_jsonl(manifest_path))
    rows[0].pop("duration", None)
    save_to_jsonl(rows, manifest_path)

    adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=False,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )
    with pytest.raises(ValueError, match="missing duration"):
        next(iter(adapter))


@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_lazy_nemo_tarred_tar_read_error_obeys_only_audio_policy(
    nemo_tarred_manifest, monkeypatch, skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    manifest_path, tar_path = nemo_tarred_manifest
    monkeypatch.setenv("USE_AIS_GET_BATCH", "false")

    def fail_sequential(*args, **kwargs):
        raise tarfile.ReadError("synthetic tar failure")

    monkeypatch.setattr(nemo_adapters.LazyNeMoTarredIterator, "_iter_sequential", fail_sequential)

    adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=False,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )
    if fault_tolerant_audio_loading:
        assert list(adapter) == []
    else:
        with pytest.raises(RuntimeError, match="Failed to read NeMo tar archive"):
            next(iter(adapter))


@pytest.mark.parametrize("use_ais_get_batch", [False, True])
def test_lazy_nemo_tarred_indexed_skipme_is_canonical_filter(nemo_tarred_manifest, monkeypatch, use_ais_get_batch):
    manifest_path, tar_path = nemo_tarred_manifest
    monkeypatch.setenv("USE_AIS_GET_BATCH", str(use_ais_get_batch).lower())
    rows = list(load_jsonl(manifest_path))
    rows[0]["_skipme"] = "low char. rate"
    rows[0]["audio_filepath"] = "intentionally-missing.wav"
    rows[1]["custom"] = {"_skipme": 1}
    rows[2]["_skipme"] = ""
    rows[3]["custom"] = {"_skipme": 0}
    save_to_jsonl(rows, manifest_path)
    create_jsonl_index(manifest_path)

    for skip_missing_manifest_entries in (False, True):
        adapter = nemo_adapters.LazyNeMoTarredIterator(
            manifest_path=str(manifest_path),
            tar_paths=str(tar_path),
            indexed=True,
            skip_missing_manifest_entries=skip_missing_manifest_entries,
        )
        cuts = list(adapter)
        yielded_ids = {cut.custom["cut_id"] for cut in cuts}
        assert len(adapter) == N_CUTS
        assert len(cuts) == N_CUTS - 2
        assert rows[0]["cut_id"] not in yielded_ids
        assert rows[1]["cut_id"] not in yielded_ids
        assert rows[2]["cut_id"] in yielded_ids
        assert rows[3]["cut_id"] in yielded_ids
        with pytest.raises(IndexError, match="not decodable"):
            adapter[0]
        assert adapter[2].custom["cut_id"] == rows[2]["cut_id"]


def test_lazy_nemo_tarred_indexed_resume_is_stable_across_skipme(nemo_tarred_manifest, monkeypatch):
    manifest_path, tar_path = nemo_tarred_manifest
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    rows = list(load_jsonl(manifest_path))
    rows[1]["_skipme"] = True
    rows[2]["custom"] = {"_skipme": "filtered"}
    save_to_jsonl(rows, manifest_path)
    create_jsonl_index(manifest_path)

    def build():
        return nemo_adapters.LazyNeMoTarredIterator(
            manifest_path=str(manifest_path),
            tar_paths=str(tar_path),
            indexed=True,
            skip_missing_manifest_entries=False,
        )

    uninterrupted = build()
    source = iter(uninterrupted)
    first = next(source)
    state_after_first_physical_row = uninterrupted.state_dict()
    expected_remainder = [cut.id for cut in source]

    resumed = build()
    resumed.load_state_dict(state_after_first_physical_row)
    actual_remainder = [cut.id for cut in resumed]

    assert first.id not in actual_remainder
    assert actual_remainder == expected_remainder
    assert len(actual_remainder) == N_CUTS - 3
    assert len(resumed) == N_CUTS


def test_lazy_nemo_tarred_indexed_malformed_json_is_fatal(nemo_tarred_manifest, monkeypatch):
    manifest_path, tar_path = nemo_tarred_manifest
    monkeypatch.setenv("USE_AIS_GET_BATCH", "true")
    valid_rows = manifest_path.read_text().splitlines()
    manifest_path.write_text("{not-json}\n" + "\n".join(valid_rows[1:]) + "\n")
    create_jsonl_index(manifest_path)

    adapter = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=str(manifest_path),
        tar_paths=str(tar_path),
        indexed=True,
        skip_missing_manifest_entries=False,
    )
    with pytest.raises(json.JSONDecodeError):
        next(iter(adapter))


@pytest.fixture
def nemo_tarred_duplicate_bucket_manifest(tmp_audio_root) -> tuple[list[Path], list[Path]]:
    """Two bucket dirs that both contain manifest_0.jsonl/audios_0.tar.

    Indexed LazyNeMoTarredIterator used to key both paths by numeric shard id 0,
    silently overwriting the first bucket. The expected dataset size is 2*N_CUTS.
    """
    from lhotse.serialization import SequentialJsonlWriter
    from lhotse.shar.writers import TarWriter

    root = tmp_audio_root / "tarred_duplicate_buckets"
    root.mkdir(exist_ok=True)
    manifest_paths: list[Path] = []
    tar_paths: list[Path] = []
    for bucket_idx in range(2):
        cuts = DummyManifest(
            CutSet,
            begin_id=bucket_idx * N_CUTS,
            end_id=(bucket_idx + 1) * N_CUTS,
            with_data=True,
        ).save_audios(tmp_audio_root / f"bucket_audio_{bucket_idx}", progress_bar=False)
        bucket = root / f"bucket_{bucket_idx}"
        bucket.mkdir(exist_ok=True)
        manifest_path = bucket / "manifest_0.jsonl"
        tar_path = bucket / "audios_0.tar"
        with (
            TarWriter(str(tar_path), shard_size=None) as tar_writer,
            SequentialJsonlWriter(manifest_path) as mft_writer,
        ):
            for cut in cuts:
                src = cut.recording.sources[0].source
                name = Path(src).name
                with open(src, "rb") as f:
                    tar_writer.write(name, BytesIO(f.read()))
                mft_writer.write(
                    {
                        "audio_filepath": name,
                        "text": "irrelevant",
                        "duration": cut.duration,
                        "sampling_rate": cut.sampling_rate,
                        "lang": "en",
                        "shard_id": 0,
                        "cut_id": cut.id,
                    }
                )
        manifest_paths.append(manifest_path)
        tar_paths.append(tar_path)
    return manifest_paths, tar_paths


@pytest.mark.parametrize("world_size", [1, 2, 4, 5])
def test_lazy_nemo_tarred_iterator_indexed_preserves_duplicate_bucket_shard_ids(
    nemo_tarred_duplicate_bucket_manifest, world_size
):
    manifest_paths, tar_paths = nemo_tarred_duplicate_bucket_manifest

    def build():
        it = nemo_adapters.LazyNeMoTarredIterator(
            manifest_path=[str(path) for path in manifest_paths],
            tar_paths=[str(path) for path in tar_paths],
            indexed=True,
        )
        assert len(it) == 2 * N_CUTS
        assert len(it.shard_id_to_tar_path) == 2
        return [cut.custom["cut_id"] for cut in it]

    per_rank, union = _collect_disjoint_per_rank(build, world_size)
    assert len(union) == 2 * N_CUTS, f"missing {2 * N_CUTS - len(union)} items at world_size={world_size}"
    assert sum(len(r) for r in per_rank) == 2 * N_CUTS


def test_lazy_nemo_tarred_iterator_streaming_preserves_duplicate_bucket_shard_ids(
    nemo_tarred_duplicate_bucket_manifest,
):
    manifest_paths, tar_paths = nemo_tarred_duplicate_bucket_manifest
    it = nemo_adapters.LazyNeMoTarredIterator(
        manifest_path=[str(path) for path in manifest_paths],
        tar_paths=[str(path) for path in tar_paths],
        indexed=False,
    )

    ids = [cut.custom["cut_id"] for cut in it]
    assert len(ids) == 2 * N_CUTS
    assert len(set(ids)) == 2 * N_CUTS


# ---------------------------------------------------------------------------
# 2. LazyParquetIterator
# ---------------------------------------------------------------------------


@pytest.fixture
def parquet_manifest(tmp_audio_root) -> Path:
    """20-row parquet file: id + audio_bytes + text."""
    pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    import pandas as pd

    cuts = DummyManifest(CutSet, begin_id=0, end_id=N_CUTS, with_data=True).save_audios(
        tmp_audio_root / "parquet_audio", progress_bar=False
    )
    rows = []
    for cut in cuts:
        with open(cut.recording.sources[0].source, "rb") as f:
            rows.append(
                {
                    "id": cut.id,
                    "audio": {"bytes": f.read()},
                    "text": "irrelevant",
                    "duration": cut.duration,
                    "lang": "en",
                }
            )
    df = pd.DataFrame(rows)
    p = tmp_audio_root / "data.parquet"
    df.to_parquet(p, engine="pyarrow", row_group_size=7)  # > 1 row group exercise
    return p


@pytest.mark.parametrize("world_size", [1, 2, 4, 5])
def test_lazy_parquet_iterator_indexed_partition(parquet_manifest, world_size):
    pytest.importorskip("pyarrow")

    def build():
        it = nemo_adapters.LazyParquetIterator(path=str(parquet_manifest), indexed=True)
        return [cut.id for cut in it]

    per_rank, union = _collect_disjoint_per_rank(build, world_size)
    assert len(union) == N_CUTS


# ---------------------------------------------------------------------------
# 3. LhotseTextJsonlAdapter
# ---------------------------------------------------------------------------


@pytest.fixture
def text_jsonl(tmp_path) -> Path:
    p = tmp_path / "text.jsonl"
    with open(p, "w") as f:
        for i in range(N_CUTS):
            f.write(json.dumps({"id": f"t-{i:04d}", "text": f"line {i}"}) + "\n")
    return p


@pytest.mark.parametrize("world_size", [1, 2, 4, 5])
def test_lhotse_text_jsonl_adapter_indexed_partition(text_jsonl, world_size):
    def build():
        it = text_adapters.LhotseTextJsonlAdapter(paths=str(text_jsonl), language="en", indexed=True)
        return [ex.text for ex in it]

    per_rank, union = _collect_disjoint_per_rank(build, world_size)
    assert len(union) == N_CUTS


# ---------------------------------------------------------------------------
# 4. NeMoSFTJsonlAdapter
# ---------------------------------------------------------------------------


@pytest.fixture
def sft_jsonl(tmp_path) -> Path:
    """Minimal NeMo-SFT-chat JSONL — adapter wraps each line, doesn't parse."""
    p = tmp_path / "sft.jsonl"
    with open(p, "w") as f:
        for i in range(N_CUTS):
            f.write(json.dumps({"id": f"sft-{i:04d}", "marker": i}) + "\n")
    return p


@pytest.mark.parametrize("world_size", [1, 2, 4, 5])
def test_nemo_sft_jsonl_adapter_indexed_partition(sft_jsonl, world_size):
    def build():
        it = text_adapters.NeMoSFTJsonlAdapter(paths=str(sft_jsonl), language="en", indexed=True)
        # NeMoSFTExample stores the raw dict in .data; key by "id".
        return [ex.data["id"] for ex in it]

    per_rank, union = _collect_disjoint_per_rank(build, world_size)
    assert len(union) == N_CUTS


# ---------------------------------------------------------------------------
# 5. NeMoMultimodalConversationJsonlAdapter — non-tarred path
# ---------------------------------------------------------------------------


@pytest.fixture
def mm_conversation_jsonl(tmp_audio_root) -> Path:
    """20-line JSONL where each line is a 2-turn user/assistant conversation
    referring to a local audio file."""
    cuts = DummyManifest(CutSet, begin_id=0, end_id=N_CUTS, with_data=True).save_audios(
        tmp_audio_root / "mm_audio", progress_bar=False
    )
    p = tmp_audio_root / "mm_conversations.jsonl"
    with open(p, "w") as f:
        for i, cut in enumerate(cuts):
            audio_filepath = cut.recording.sources[0].source
            f.write(
                json.dumps(
                    {
                        "id": f"mm-{i:04d}",
                        "conversations": [
                            {
                                "type": "audio",
                                "from": "User",
                                "value": audio_filepath,
                                "duration": cut.duration,
                                "offset": 0.0,
                            },
                            {
                                "type": "text",
                                "from": "Assistant",
                                "value": f"answer {i}",
                            },
                        ],
                    }
                )
                + "\n"
            )
    return p


@pytest.mark.parametrize("world_size", [1, 2, 4, 5])
def test_nemo_multimodal_conversation_jsonl_adapter_indexed_partition(mm_conversation_jsonl, world_size):
    def build():
        it = text_adapters.NeMoMultimodalConversationJsonlAdapter(
            manifest_filepath=[str(mm_conversation_jsonl)],
            audio_locator_tag="<audio>",
            token_equivalent_duration=0.08,
            indexed=True,
        )
        return [convo.id for convo in it]

    per_rank, union = _collect_disjoint_per_rank(build, world_size)
    assert len(union) == N_CUTS


@pytest.mark.parametrize("skip_missing_manifest_entries", [False, True])
@pytest.mark.parametrize("fault_tolerant_audio_loading", [False, True])
def test_nemo_multimodal_missing_local_audio_obeys_only_audio_policy(
    mm_conversation_jsonl, skip_missing_manifest_entries, fault_tolerant_audio_loading
):
    rows = list(load_jsonl(mm_conversation_jsonl))
    rows[0]["conversations"][0]["value"] = str(mm_conversation_jsonl.parent / "missing.wav")
    save_to_jsonl(rows, mm_conversation_jsonl)

    adapter = text_adapters.NeMoMultimodalConversationJsonlAdapter(
        manifest_filepath=[str(mm_conversation_jsonl)],
        audio_locator_tag="<audio>",
        indexed=False,
        skip_missing_manifest_entries=skip_missing_manifest_entries,
        fault_tolerant_audio_loading=fault_tolerant_audio_loading,
    )
    if fault_tolerant_audio_loading:
        assert len(list(adapter)) == N_CUTS - 1
    else:
        with pytest.raises(RuntimeError, match="Failed to load multimodal conversation"):
            next(iter(adapter))


def test_nemo_multimodal_indexed_skipme_is_canonical_filter(mm_conversation_jsonl):
    rows = list(load_jsonl(mm_conversation_jsonl))
    rows[0]["_skipme"] = "filtered"
    rows[1]["custom"] = {"_skipme": 1}
    rows[2]["_skipme"] = ""
    rows[3]["custom"] = {"_skipme": 0}
    save_to_jsonl(rows, mm_conversation_jsonl)

    for skip_missing_manifest_entries in (False, True):
        adapter = text_adapters.NeMoMultimodalConversationJsonlAdapter(
            manifest_filepath=[str(mm_conversation_jsonl)],
            audio_locator_tag="<audio>",
            indexed=True,
            skip_missing_manifest_entries=skip_missing_manifest_entries,
        )
        conversations = list(adapter)
        yielded_ids = {conversation.id for conversation in conversations}
        assert len(adapter) == N_CUTS
        assert len(conversations) == N_CUTS - 2
        assert rows[0]["id"] not in yielded_ids
        assert rows[1]["id"] not in yielded_ids
        assert rows[2]["id"] in yielded_ids
        assert rows[3]["id"] in yielded_ids
        with pytest.raises(IndexError):
            adapter[0]


# ---------------------------------------------------------------------------
# 6. NeMoMultimodalConversationShareGPTJsonlAdapter — non-tarred path
# ---------------------------------------------------------------------------


@pytest.fixture
def sharegpt_conversation_jsonl(tmp_audio_root) -> Path:
    """ShareGPT-format JSONL with a single user audio + assistant turn each.

    Schema note: the audio path lives in the ``sound`` field (see
    ``_transform_sharegpt`` in nemo.collections.common.data.lhotse.text_adapters),
    not in ``audio_filepath`` — the adapter intentionally treats ShareGPT
    distinctly from NeMo manifests."""
    cuts = DummyManifest(CutSet, begin_id=0, end_id=N_CUTS, with_data=True).save_audios(
        tmp_audio_root / "sharegpt_audio", progress_bar=False
    )
    p = tmp_audio_root / "sharegpt.jsonl"
    with open(p, "w") as f:
        for i, cut in enumerate(cuts):
            audio_filepath = cut.recording.sources[0].source
            f.write(
                json.dumps(
                    {
                        "id": f"sgpt-{i:04d}",
                        "conversations": [
                            {"from": "User", "value": f"<audio>describe {i}"},
                            {"from": "Assistant", "value": f"this is example {i}"},
                        ],
                        "sound": audio_filepath,
                        "duration": cut.duration,
                    }
                )
                + "\n"
            )
    return p


@pytest.mark.parametrize("world_size", [1, 2, 4, 5])
def test_sharegpt_jsonl_adapter_indexed_partition(sharegpt_conversation_jsonl, world_size):
    def build():
        it = text_adapters.NeMoMultimodalConversationShareGPTJsonlAdapter(
            manifest_filepath=[str(sharegpt_conversation_jsonl)],
            audio_locator_tag="<audio>",
            audio_placeholders=["<audio>"],
            token_equivalent_duration=0.08,
            indexed=True,
        )
        return [convo.id for convo in it]

    per_rank, union = _collect_disjoint_per_rank(build, world_size)
    assert len(union) == N_CUTS


def test_sharegpt_jsonl_adapter_approved_exclusions_are_logical_and_resumable(
    sharegpt_conversation_jsonl,
):
    excluded_lines = [2, 5, 20]
    line_digest = hashlib.sha256((json.dumps(excluded_lines, separators=(",", ":")) + "\n").encode()).hexdigest()

    def build(lines=excluded_lines, digest=line_digest):
        return text_adapters.NeMoMultimodalConversationShareGPTJsonlAdapter(
            manifest_filepath=[str(sharegpt_conversation_jsonl)],
            audio_locator_tag="<audio>",
            audio_placeholders=["<audio>"],
            token_equivalent_duration=0.08,
            indexed=True,
            excluded_manifest_lines=lines,
            excluded_manifest_lines_sha256=digest,
            approved_exclusion_audit_sha256="a" * 64,
        )

    expected = [f"sgpt-{idx:04d}" for idx in range(N_CUTS) if idx + 1 not in excluded_lines]
    adapter = build()
    assert len(adapter) == N_CUTS - len(excluded_lines)
    assert adapter[0].id == "sgpt-0000"
    assert adapter[1].id == "sgpt-0002"
    assert adapter[-1].id == "sgpt-0018"

    stream = iter(adapter)
    prefix = [next(stream).id for _ in range(7)]
    state = adapter.state_dict()
    restored = build()
    restored.load_state_dict(state)
    assert prefix + [item.id for item in restored] == expected

    changed = build(lines=[2, 6, 20], digest=hashlib.sha256(b"[2,6,20]\n").hexdigest())
    with pytest.raises(ValueError, match="exclusion set changed across resume"):
        changed.load_state_dict(state)


def test_sharegpt_jsonl_adapter_approved_exclusions_validate_provenance(
    sharegpt_conversation_jsonl,
):
    common = {
        "manifest_filepath": [str(sharegpt_conversation_jsonl)],
        "audio_locator_tag": "<audio>",
        "audio_placeholders": ["<audio>"],
        "token_equivalent_duration": 0.08,
        "indexed": True,
        "excluded_manifest_lines": [2],
    }
    with pytest.raises(ValueError, match="approved_exclusion_audit_sha256"):
        text_adapters.NeMoMultimodalConversationShareGPTJsonlAdapter(**common)
    with pytest.raises(ValueError, match="does not match"):
        text_adapters.NeMoMultimodalConversationShareGPTJsonlAdapter(
            **common,
            excluded_manifest_lines_sha256="0" * 64,
            approved_exclusion_audit_sha256="a" * 64,
        )


# ---------------------------------------------------------------------------
# 7. NeMoMultimodalConversationShareGPTWebdatasetAdapter
# ---------------------------------------------------------------------------


@pytest.fixture
def sharegpt_webdataset_tar(tmp_audio_root) -> Path:
    """20-sample ShareGPT WebDataset tar: each example is a (.json, .wav) pair
    with matching stem. The adapter pairs alternating members. We also build
    the ``.idx`` sidecar that IndexedTarSampleReader requires (it does not
    auto-create indexes, unlike the JSONL reader)."""
    from lhotse.indexing import create_tar_index

    cuts = DummyManifest(CutSet, begin_id=0, end_id=N_CUTS, with_data=True).save_audios(
        tmp_audio_root / "wds_audio", progress_bar=False
    )
    p = tmp_audio_root / "shard_0.tar"
    with tarfile.open(p, "w") as tar:
        for i, cut in enumerate(cuts):
            stem = f"swds-{i:04d}"
            audio_path = cut.recording.sources[0].source
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            payload = json.dumps(
                {
                    "id": stem,
                    "conversations": [
                        {"from": "User", "value": f"<audio>q{i}"},
                        {"from": "Assistant", "value": f"a{i}"},
                    ],
                }
            ).encode()
            for ext, data in ((".json", payload), (".wav", audio_bytes)):
                info = tarfile.TarInfo(stem + ext)
                info.size = len(data)
                tar.addfile(info, BytesIO(data))
    create_tar_index(str(p), output_path=str(p) + ".idx")
    return p


@pytest.mark.parametrize("world_size", [1, 2, 4, 5])
def test_sharegpt_webdataset_adapter_indexed_partition(sharegpt_webdataset_tar, world_size):
    def build():
        it = text_adapters.NeMoMultimodalConversationShareGPTWebdatasetAdapter(
            data_dir=str(sharegpt_webdataset_tar.parent),
            audio_locator_tag="<audio>",
            audio_placeholders=["<audio>"],
            token_equivalent_duration=0.08,
            indexed=True,
        )
        return [convo.id for convo in it]

    per_rank, union = _collect_disjoint_per_rank(build, world_size)
    assert len(union) == N_CUTS
