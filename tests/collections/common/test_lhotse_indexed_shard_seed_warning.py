# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Indexed NeMo manifest sources replace shard_seed 'trng'/'randomized' with a fixed seed of 0.

The replacement must stay (every rank and worker has to shuffle identically, and resumed runs depend on
the order), but it must not be silent: warn once per process.
"""
import json

import pytest
from lhotse.indexing import create_jsonl_index

from nemo.collections.common.data.lhotse import nemo_adapters
from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator


@pytest.fixture
def manifest_paths(tmp_path):
    paths = []
    for shard in range(3):
        path = tmp_path / f"manifest_{shard}.json"
        with path.open("w") as f:
            for i in range(4):
                entry = {"audio_filepath": f"{shard}_{i}.wav", "duration": 1.0, "text": "x"}
                print(json.dumps(entry), file=f)
        create_jsonl_index(path)
        paths.append(str(path))
    return paths


@pytest.fixture
def warnings_seen(monkeypatch):
    seen = []
    monkeypatch.setattr(nemo_adapters, "_INDEXED_SHARD_SEED_WARNED", False, raising=False)
    monkeypatch.setattr(nemo_adapters.logging, "warning", lambda msg, *a, **kw: seen.append(msg))
    return seen


def _shard_seed_warnings(seen):
    return [m for m in seen if "is ignored for indexed NeMo manifest sources" in m]


@pytest.mark.unit
@pytest.mark.parametrize("shard_seed", ["trng", "randomized"])
def test_indexed_nemo_source_warns_once_about_replaced_shard_seed(manifest_paths, warnings_seen, shard_seed):
    for _ in range(3):  # e.g. several blend entries in one process
        it = LazyNeMoIterator(manifest_paths, indexed=True, shuffle_shards=True, shard_seed=shard_seed)
        assert it.source.seed == 0  # the fallback value is unchanged
    msgs = _shard_seed_warnings(warnings_seen)
    assert len(msgs) == 1
    assert f"shard_seed={shard_seed!r}" in msgs[0]
    assert "fixed seed (0)" in msgs[0]
    assert "integer shard_seed" in msgs[0]


@pytest.mark.unit
def test_indexed_nemo_source_integer_shard_seed_does_not_warn(manifest_paths, warnings_seen):
    it = LazyNeMoIterator(manifest_paths, indexed=True, shuffle_shards=True, shard_seed=1234)
    assert it.source.seed == 1234
    assert _shard_seed_warnings(warnings_seen) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "shuffle_shards,num_paths",
    [(False, 3), (True, 1)],  # the shard shuffle seed isn't used, so nothing is ignored
)
def test_indexed_nemo_source_unused_shard_seed_does_not_warn(manifest_paths, warnings_seen, shuffle_shards, num_paths):
    LazyNeMoIterator(manifest_paths[:num_paths], indexed=True, shuffle_shards=shuffle_shards, shard_seed="trng")
    assert _shard_seed_warnings(warnings_seen) == []


@pytest.mark.unit
def test_streaming_nemo_source_does_not_warn(manifest_paths, warnings_seen):
    LazyNeMoIterator(manifest_paths, indexed=False, shuffle_shards=True, shard_seed="trng")
    assert _shard_seed_warnings(warnings_seen) == []
