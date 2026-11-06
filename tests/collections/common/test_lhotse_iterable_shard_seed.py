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
from pathlib import Path

import pytest
import torch
from lhotse import CutSet
from lhotse.indexing import create_jsonl_index
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


def _build(config: dict, world_size: int = 1):
    return get_lhotse_dataloader_from_config(config, global_rank=0, world_size=world_size, dataset=_Identity())


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
