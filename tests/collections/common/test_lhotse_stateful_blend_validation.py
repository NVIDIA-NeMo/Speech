# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Weighted blends that Lhotse can't iterate or checkpoint must be rejected when the dataloader is built.

(a) Stateful loader + bucketing + a blend of indexed and streamed sources: before the fix, training started
    and then crashed at the first checkpoint save inside ``DynamicBucketingSampler.state_dict()``.
(b) Iterable blend of indexed sources split into several shards + ``shard_seed='randomized'``: before the fix,
    Lhotse raised from inside the dataloader iterator, naming its own arguments instead of NeMo's config keys.
"""
import json
from itertools import islice

import pytest
from lhotse import CutSet
from lhotse.indexing import create_jsonl_index
from lhotse.testing.dummies import DummyManifest
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.dataloader import (
    get_lhotse_dataloader_from_config,
    get_lhotse_sampler_from_config,
    make_structured_with_schema_warnings,
)


class _Identity:
    def __getitem__(self, cuts):
        return cuts


@pytest.fixture
def lhotse_cuts_paths(tmp_path):
    """Two Lhotse JSONL manifests: ``a`` has an .idx sidecar, ``b`` doesn't (yet)."""
    paths = {}
    for name, begin in (("a", 0), ("b", 100)):
        cuts = DummyManifest(CutSet, begin_id=begin, end_id=begin + 20, with_data=False)
        for c in cuts:
            c.duration = 1.0
        path = tmp_path / f"cuts_{name}.jsonl"
        cuts.to_file(path)
        paths[name] = path
    create_jsonl_index(paths["a"])
    return paths


@pytest.fixture
def nemo_manifest_path(tmp_path):
    path = tmp_path / "nemo_manifest.json"
    with path.open("w") as f:
        for i in range(20):
            print(json.dumps({"audio_filepath": f"{i}.wav", "duration": 1.0, "text": "x"}), file=f)
    return path


def _blend(paths, **overrides):
    return {
        "use_lhotse": True,
        "seed": 0,
        "shard_seed": 0,
        "shuffle": True,
        "num_workers": 0,
        "use_stateful_dataloader": True,
        "force_map_dataset": True,
        "use_bucketing": True,
        "num_buckets": 2,
        "bucket_duration_bins": [1.5],
        "batch_duration": 3.0,
        "bucket_buffer_size": 20,
        "shuffle_buffer_size": 20,
        "concurrent_bucketing": False,
        "input_cfg": [
            {"type": "group", "input_cfg": [{"weight": 0.5, **src} for src in paths]},
        ],
        **overrides,
    }


def _lhotse_sources(paths):
    return [{"type": "lhotse", "cuts_path": str(p)} for p in (paths["a"], paths["b"])]


def _build_loader(cfg, world_size=1):
    return get_lhotse_dataloader_from_config(
        OmegaConf.create(cfg), global_rank=0, world_size=world_size, dataset=_Identity()
    )


def _build_sampler(cfg, world_size=1):
    config = make_structured_with_schema_warnings(OmegaConf.create(cfg))
    return get_lhotse_sampler_from_config(config, global_rank=0, world_size=world_size)


@pytest.mark.unit
def test_bucketing_blend_of_indexed_and_streamed_lhotse_sources_raises(lhotse_cuts_paths):
    # 'indexed' is unset, so cuts_a.jsonl (has .idx) is indexed and cuts_b.jsonl is streamed.
    with pytest.raises(ValueError, match="indexed: true"):
        _build_loader(_blend(_lhotse_sources(lhotse_cuts_paths)))


@pytest.mark.unit
def test_bucketing_blend_of_nemo_and_indexed_lhotse_sources_raises(lhotse_cuts_paths, nemo_manifest_path):
    sources = [
        {"type": "nemo", "manifest_filepath": str(nemo_manifest_path)},
        {"type": "lhotse", "cuts_path": str(lhotse_cuts_paths["a"])},
    ]
    with pytest.raises(ValueError, match="use_bucketing: false"):
        _build_loader(_blend(sources))


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"indexed": False},
        {"use_bucketing": False, "bucket_duration_bins": None, "batch_size": 2},
        {"use_stateful_dataloader": False},
    ],
)
def test_bucketing_blend_mixed_indexing_passing_configs(lhotse_cuts_paths, overrides):
    _build_loader(_blend(_lhotse_sources(lhotse_cuts_paths), **overrides))


@pytest.mark.unit
@pytest.mark.parametrize("indexed", [True, False])
def test_bucketing_blend_with_uniform_indexing_checkpoints(lhotse_cuts_paths, indexed):
    create_jsonl_index(lhotse_cuts_paths["b"])
    sampler, _ = _build_sampler(_blend(_lhotse_sources(lhotse_cuts_paths), indexed=indexed))
    batches = list(islice(iter(sampler), 3))
    assert len(batches) == 3
    sampler.state_dict()  # used to be where the mixed case failed


def _iterable_indexed_blend(paths, **overrides):
    create_jsonl_index(paths["b"])
    cfg = _blend(
        _lhotse_sources(paths),
        indexed=True,
        force_map_dataset=False,
        force_iterable_dataset=True,
        use_bucketing=False,
        bucket_duration_bins=None,
        batch_size=2,
    )
    cfg.update(overrides)
    return cfg


@pytest.mark.unit
@pytest.mark.parametrize("num_workers,world_size", [(2, 1), (0, 2), (1, 2)])
def test_indexed_iterable_blend_multi_shard_randomized_shard_seed_raises(lhotse_cuts_paths, num_workers, world_size):
    cfg = _iterable_indexed_blend(lhotse_cuts_paths, shard_seed="randomized", num_workers=num_workers)
    with pytest.raises(ValueError, match="Set an integer shard_seed"):
        _build_loader(cfg, world_size=world_size)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides,world_size",
    [
        ({"shard_seed": 1234, "num_workers": 2}, 2),  # integer seed: disjoint slices, no duplication
        ({"shard_seed": "trng", "num_workers": 2}, 1),  # Lhotse accepts 'trng'
        ({"shard_seed": "randomized", "num_workers": 0}, 1),  # single shard
    ],
)
def test_indexed_iterable_blend_passing_shard_seeds(lhotse_cuts_paths, overrides, world_size):
    _build_sampler(_iterable_indexed_blend(lhotse_cuts_paths, **overrides), world_size=world_size)


@pytest.mark.unit
def test_streaming_iterable_blend_randomized_shard_seed_is_allowed(lhotse_cuts_paths):
    # Lhotse only restricts 'randomized' for blends of indexed sources.
    cfg = _iterable_indexed_blend(lhotse_cuts_paths, shard_seed="randomized", num_workers=2, indexed=False)
    _build_sampler(cfg, world_size=1)
