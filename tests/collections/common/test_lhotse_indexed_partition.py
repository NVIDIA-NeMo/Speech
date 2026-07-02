# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import tarfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import pytest
from lhotse import CutSet
from lhotse.dataset.dataloading import LHOTSE_USE_WORKER_PARTITION
from lhotse.testing.dummies import DummyManifest

from nemo.collections.common.data.lhotse import nemo_adapters, text_adapters

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
                    "lang": "en",
                    "shard_id": 0,
                    "cut_id": cut.id,
                }
            )
    return Path(mft_writer.path), root / "audios_0.tar"


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
