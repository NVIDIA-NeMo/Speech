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
"""An integer ``shard_seed`` on a non-indexed iterable pipeline makes every dataloading worker and rank yield the
identical stream. Dataloader construction must reject that combination when there is more than one data shard,
while indexed sources (which partition by ``(rank, worker_id)``) keep accepting an integer seed."""
from io import BytesIO
from itertools import islice
from pathlib import Path

import pytest
import torch
from lhotse import CutSet
from lhotse.dataset.dataloading import LHOTSE_PROCESS_SEED, LHOTSE_USE_WORKER_PARTITION
from lhotse.indexing import create_jsonl_index
from lhotse.serialization import SequentialJsonlWriter
from lhotse.shar.writers import TarWriter
from lhotse.testing.dummies import DummyManifest

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config


class _Identity(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet) -> CutSet:
        return cuts


def _write_cuts(path: Path, begin_id: int) -> Path:
    cuts = DummyManifest(CutSet, begin_id=begin_id, end_id=begin_id + 8, with_data=False)
    cuts.to_file(path)
    create_jsonl_index(path)
    return path


@pytest.fixture
def cuts_path(tmp_path: Path) -> Path:
    return _write_cuts(tmp_path / "cuts.jsonl", begin_id=0)


@pytest.fixture
def other_cuts_path(tmp_path: Path) -> Path:
    return _write_cuts(tmp_path / "cuts_other.jsonl", begin_id=100)


def _write_nemo_tarred(root: Path, begin_id: int, cuts_per_shard: int) -> tuple[str, str]:
    """Write a non-indexed NeMo tarred dataset with four shards of ``cuts_per_shard`` cuts each."""
    cuts = DummyManifest(CutSet, begin_id=begin_id, end_id=begin_id + 4 * cuts_per_shard, with_data=True)
    cuts = cuts.save_audios(root / "audio", progress_bar=False)
    for shard_id in range(4):
        with (
            TarWriter(str(root / f"audio_{shard_id}.tar"), shard_size=None) as tar_writer,
            SequentialJsonlWriter(root / f"manifest_{shard_id}.jsonl") as manifest_writer,
        ):
            for cut in cuts.subset(first=cuts_per_shard * (shard_id + 1)).subset(last=cuts_per_shard):
                src = cut.recording.sources[0].source
                with open(src, "rb") as f:
                    tar_writer.write(Path(src).name, BytesIO(f.read()))
                manifest_writer.write(
                    {
                        "audio_filepath": Path(src).name,
                        "text": "irrelevant",
                        "duration": cut.duration,
                        "shard_id": shard_id,
                    }
                )
    return str(root / "manifest__OP_0..3_CL_.jsonl"), str(root / "audio__OP_0..3_CL_.tar")


@pytest.fixture
def nemo_tarred_four_shards(tmp_path: Path) -> tuple[str, str]:
    """Non-indexed NeMo tarred dataset: four shards with two cuts each."""
    return _write_nemo_tarred(tmp_path / "nemo_tar", begin_id=0, cuts_per_shard=2)


@pytest.fixture
def other_nemo_tarred_four_shards(tmp_path: Path) -> tuple[str, str]:
    """Like ``nemo_tarred_four_shards`` with different cut IDs."""
    return _write_nemo_tarred(tmp_path / "nemo_tar_other", begin_id=100, cuts_per_shard=2)


@pytest.fixture
def nemo_tarred_four_long_shards(tmp_path: Path) -> tuple[str, str]:
    """Non-indexed NeMo tarred dataset: four shards with eight cuts each (for slice_length tests)."""
    return _write_nemo_tarred(tmp_path / "nemo_tar_long", begin_id=0, cuts_per_shard=8)


@pytest.fixture(autouse=True)
def _restore_worker_partition_env(monkeypatch):
    """Loaders built with ``num_workers=0`` run lhotse's worker_init_fn in this process, which sets env vars."""
    for key in ("RANK", "WORLD_SIZE", LHOTSE_USE_WORKER_PARTITION, LHOTSE_PROCESS_SEED):
        monkeypatch.delenv(key, raising=False)


def _tarred_source(nemo_tarred_four_shards, **overrides) -> dict:
    manifest_filepath, tarred_audio_filepaths = nemo_tarred_four_shards
    return {
        "type": "nemo_tarred",
        "manifest_filepath": manifest_filepath,
        "tarred_audio_filepaths": tarred_audio_filepaths,
        **overrides,
    }


def _config(**overrides) -> dict:
    return {
        "batch_size": 2,
        "seed": 0,
        "shard_seed": 1234,
        "shuffle": True,
        "force_iterable_dataset": True,
        "force_finite": True,
        "num_workers": 2,
        **overrides,
    }


def _build(config: dict, world_size: int = 1, global_rank: int = 0, tokenizer=None):
    return get_lhotse_dataloader_from_config(
        config, global_rank=global_rank, world_size=world_size, dataset=_Identity(), tokenizer=tokenizer
    )


@pytest.mark.unit
@pytest.mark.parametrize("num_workers,world_size", [(2, 1), (0, 2), (1, 2), (4, 8)])
def test_int_shard_seed_non_indexed_iterable_multi_shard_raises(cuts_path, num_workers, world_size):
    config = _config(cuts_path=str(cuts_path), indexed=False, num_workers=num_workers)
    with pytest.raises(ValueError, match="shard_seed=1234"):
        _build(config, world_size=world_size)


@pytest.mark.unit
def test_int_shard_seed_partially_indexed_blend_raises(cuts_path, other_cuts_path):
    config = _config(
        input_cfg=[
            {"type": "lhotse", "cuts_path": str(cuts_path), "weight": 1.0, "indexed": True},
            {"type": "lhotse", "cuts_path": str(other_cuts_path), "weight": 1.0, "indexed": False},
        ],
    )
    with pytest.raises(ValueError, match="shard_seed=1234"):
        _build(config)


@pytest.mark.unit
@pytest.mark.parametrize("num_workers,world_size", [(0, 1), (1, 1)])
def test_int_shard_seed_non_indexed_iterable_single_shard_ok(cuts_path, num_workers, world_size):
    config = _config(cuts_path=str(cuts_path), indexed=False, num_workers=num_workers)
    _build(config, world_size=world_size)


@pytest.mark.unit
@pytest.mark.parametrize("shard_seed", ["randomized", "trng"])
def test_string_shard_seed_non_indexed_iterable_multi_shard_ok(cuts_path, shard_seed):
    config = _config(cuts_path=str(cuts_path), indexed=False, shard_seed=shard_seed)
    _build(config, world_size=2)


@pytest.mark.unit
def test_int_shard_seed_indexed_iterable_multi_shard_ok(cuts_path):
    config = _config(cuts_path=str(cuts_path), indexed=True)
    _build(config, world_size=2)


@pytest.mark.unit
def test_int_shard_seed_indexed_blend_multi_shard_ok(cuts_path, other_cuts_path):
    config = _config(
        indexed=True,
        input_cfg=[
            {"type": "lhotse", "cuts_path": str(cuts_path), "weight": 1.0},
            {"type": "lhotse", "cuts_path": str(other_cuts_path), "weight": 1.0},
        ],
    )
    _build(config, world_size=2)


@pytest.mark.unit
def test_int_shard_seed_non_indexed_map_style_multi_shard_ok(cuts_path):
    config = _config(cuts_path=str(cuts_path), indexed=False, force_iterable_dataset=False, force_map_dataset=True)
    _build(config, world_size=2)


_RANDOMIZED_SOURCE = {"shard_seed": "randomized", "shuffle": True}


def _per_worker_ids(dloader, num_batches: int = 8) -> list[list[str]]:
    """Batch cut ids per worker; with 2 workers the DataLoader alternates between them batch by batch."""
    batches = [[c.id for c in batch] for batch in islice(dloader, num_batches)]
    return [batches[0::2], batches[1::2]]


@pytest.mark.unit
def test_int_loader_seed_with_randomized_source_override_ok(nemo_tarred_four_shards):
    config = _config(
        shuffle=False,
        input_cfg=[_tarred_source(nemo_tarred_four_shards, **_RANDOMIZED_SOURCE)],
    )
    first = _per_worker_ids(_build(config))
    second = _per_worker_ids(_build(config))
    assert all(len(batch) == 2 for worker in first for batch in worker)
    # Fixed base seeding: rebuilding the loader replays the same per-worker sequences ...
    assert first == second
    # ... and the two workers do not replay the same stream.
    assert first[0] != first[1]


@pytest.mark.unit
def test_int_loader_seed_inherited_by_tarred_source_raises(nemo_tarred_four_shards):
    config = _config(shuffle=False, input_cfg=[_tarred_source(nemo_tarred_four_shards, shuffle=True)])
    with pytest.raises(ValueError, match="shard_seed=1234"):
        _build(config)


@pytest.mark.unit
@pytest.mark.parametrize("override_seed", [True, False])
def test_source_seed_override_through_nested_group(nemo_tarred_four_shards, override_seed):
    leaf = _tarred_source(nemo_tarred_four_shards, **(_RANDOMIZED_SOURCE if override_seed else {"shuffle": True}))
    config = _config(shuffle=False, input_cfg=[{"type": "group", "input_cfg": [leaf]}])
    if override_seed:
        _build(config)
    else:
        with pytest.raises(ValueError, match="shard_seed=1234"):
            _build(config)


@pytest.mark.unit
@pytest.mark.parametrize("override_seed", [True, False])
def test_source_seed_override_through_multi_config(nemo_tarred_four_shards, override_seed):
    leaf = _tarred_source(nemo_tarred_four_shards, **(_RANDOMIZED_SOURCE if override_seed else {"shuffle": True}))
    config = {
        "multi_config": True,
        "sampler_fusion": "round_robin",
        "seed": 0,
        "shard_seed": 1234,
        "shuffle": False,
        "num_workers": 2,
        "force_iterable_dataset": True,
        "a": {"input_cfg": [leaf], "batch_size": 2},
        "b": {"input_cfg": [dict(leaf)], "batch_size": 2},
    }
    if override_seed:
        _build(config)
    else:
        with pytest.raises(ValueError, match="shard_seed=1234"):
            _build(config)


@pytest.mark.unit
def test_randomized_source_does_not_exempt_other_blend_members(nemo_tarred_four_shards, cuts_path):
    config = _config(
        shuffle=False,
        input_cfg=[
            {**_tarred_source(nemo_tarred_four_shards, **_RANDOMIZED_SOURCE), "weight": 1.0},
            {"type": "lhotse", "cuts_path": str(cuts_path), "weight": 1.0, "indexed": False},
        ],
    )
    with pytest.raises(ValueError, match="shard_seed=1234"):
        _build(config)


@pytest.mark.unit
@pytest.mark.parametrize("loader_shuffle", [False, True])
def test_int_source_seed_override_under_randomized_loader(nemo_tarred_four_shards, loader_shuffle):
    config = _config(
        shard_seed="randomized",
        shuffle=loader_shuffle,
        input_cfg=[_tarred_source(nemo_tarred_four_shards, shard_seed=1234, shuffle=True)],
    )
    if loader_shuffle:
        # The sampler's shuffle buffer is seeded per worker, so the workers' streams differ.
        _build(config)
    else:
        with pytest.raises(ValueError, match=r"LazyNeMoTarredIterator\(seed=1234\)"):
            _build(config)


def _two_member_blend(active, disabled, disabled_weight: float) -> list[dict]:
    return [
        {**active, "weight": 1.0},
        {**disabled, "weight": disabled_weight},
    ]


@pytest.mark.unit
def test_zero_weight_fixed_seed_member_is_ignored(nemo_tarred_four_shards, other_nemo_tarred_four_shards):
    active = _tarred_source(nemo_tarred_four_shards, **_RANDOMIZED_SOURCE)
    disabled = _tarred_source(other_nemo_tarred_four_shards, shard_seed=1234, shuffle=True)
    config = _config(
        shuffle=False, force_finite=False, num_workers=0, input_cfg=_two_member_blend(active, disabled, 0.0)
    )
    # num_workers=0 on two ranks: two data shards without spawning worker processes.
    batches = list(islice(_build(config, world_size=2), 6))
    assert all(len(batch) == 2 for batch in batches)
    seen = {c.id for batch in batches for c in batch}
    # The active member has IDs 0..7, the disabled one 100..107.
    assert seen and all(int(i.split("-")[-1].split(".")[0]) < 100 for i in seen), seen


@pytest.mark.unit
def test_positive_weight_fixed_seed_member_raises(nemo_tarred_four_shards, other_nemo_tarred_four_shards):
    active = _tarred_source(nemo_tarred_four_shards, **_RANDOMIZED_SOURCE)
    disabled = _tarred_source(other_nemo_tarred_four_shards, shard_seed=1234, shuffle=True)
    config = _config(shuffle=False, force_finite=False, input_cfg=_two_member_blend(active, disabled, 0.5))
    with pytest.raises(ValueError, match=r"LazyNeMoTarredIterator\(seed=1234\)"):
        _build(config)


@pytest.mark.unit
@pytest.mark.parametrize("disabled_weight,raises", [(0.0, False), (0.5, True)])
def test_zero_weight_nested_group_is_ignored(
    nemo_tarred_four_shards, other_nemo_tarred_four_shards, disabled_weight, raises
):
    active = _tarred_source(nemo_tarred_four_shards, **_RANDOMIZED_SOURCE)
    disabled_group = {
        "type": "group",
        "input_cfg": [_tarred_source(other_nemo_tarred_four_shards, shard_seed=1234, shuffle=True)],
    }
    config = _config(
        shuffle=False, force_finite=False, input_cfg=_two_member_blend(active, disabled_group, disabled_weight)
    )
    if raises:
        with pytest.raises(ValueError, match="shard_seed=1234"):
            _build(config)
    else:
        _build(config)


def _packed_config(source: dict, **overrides) -> dict:
    return _config(
        shard_seed="randomized",
        shuffle=True,
        batch_size=None,
        batch_tokens=2,
        use_multimodal_sampling=True,
        pretokenize=False,
        use_packed_sequence_sampling=True,
        audio_token_estimator={
            "preprocessor": {"n_fft": 16000, "hop_length": 16000, "stft_pad_amount": 8000},
            "subsampling": [],
        },
        input_cfg=[source],
        **overrides,
    )


@pytest.mark.unit
def test_packed_sampler_does_not_count_as_randomizing(nemo_tarred_four_shards):
    # PackedSequenceDynamicCutSampler ignores ``shuffle``; the fixed-seed source order reaches every worker unchanged.
    config = _packed_config(_tarred_source(nemo_tarred_four_shards, shard_seed=1234, shuffle=True))
    with pytest.raises(ValueError, match=r"LazyNeMoTarredIterator\(seed=1234\)"):
        _build(config, tokenizer=object())


@pytest.mark.unit
def test_packed_sampler_accepts_randomized_source(nemo_tarred_four_shards):
    config = _packed_config(_tarred_source(nemo_tarred_four_shards, **_RANDOMIZED_SOURCE))
    _build(config, tokenizer=object())


@pytest.mark.unit
@pytest.mark.parametrize(
    "bucketing", [{"use_bucketing": True, "bucket_duration_bins": [1.5]}, {"bucket_duration_bins": [1.5]}]
)
def test_packed_bucketing_sampler_shuffle_counts_as_randomizing(nemo_tarred_four_shards, bucketing):
    # Explicit and auto-enabled bucketing select a shuffle-capable bucketing sampler.
    config = _packed_config(
        _tarred_source(nemo_tarred_four_shards, shard_seed=1234, shuffle=True), num_buckets=2, **bucketing
    )
    _build(config, tokenizer=object())


def _slice_source(nemo_tarred_four_long_shards, **overrides) -> dict:
    return _tarred_source(nemo_tarred_four_long_shards, shard_seed="randomized", shuffle=False, **overrides)


@pytest.mark.unit
@pytest.mark.parametrize("use_ais_get_batch", [False, True])
def test_randomized_slice_selection_counts_as_randomizing(
    nemo_tarred_four_long_shards, monkeypatch, use_ais_get_batch
):
    # Both the sequential and the AIS GetBatch readers draw per-shard slice offsets from the resolved source seed.
    monkeypatch.setenv("USE_AIS_GET_BATCH", str(use_ais_get_batch))
    config = _config(shuffle=False, input_cfg=[_slice_source(nemo_tarred_four_long_shards, slice_length=2)])
    if use_ais_get_batch:
        # Constructing is enough here: the AIS branch needs a GetBatch endpoint to iterate.
        _build(config)
        return
    first = _per_worker_ids(_build(config))
    assert all(len(batch) == 2 for worker in first for batch in worker)
    assert first == _per_worker_ids(_build(config))  # repeatable across rebuilds
    assert first[0] != first[1]  # but the two workers select different slices


@pytest.mark.unit
def test_slice_length_with_integer_seed_raises(nemo_tarred_four_long_shards):
    source = _tarred_source(nemo_tarred_four_long_shards, shuffle=False, slice_length=2)
    config = _config(shuffle=False, input_cfg=[source])
    with pytest.raises(ValueError, match="shard_seed=1234"):
        _build(config)


@pytest.mark.unit
def test_no_slicing_no_shuffle_randomized_source_seed_raises(nemo_tarred_four_long_shards):
    # A string source seed without shuffling or slicing doesn't randomize anything; the integer loader seed decides.
    config = _config(shuffle=False, input_cfg=[_slice_source(nemo_tarred_four_long_shards)])
    with pytest.raises(ValueError, match="shard_seed=1234"):
        _build(config)
