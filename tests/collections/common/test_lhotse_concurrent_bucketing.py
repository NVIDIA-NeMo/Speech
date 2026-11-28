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
"""``concurrent_bucketing`` resolution: it must be off for the resumable (stateful) dataloader."""

import random
import time

import pytest
import torch
from lhotse import CutSet
from lhotse.testing.dummies import DummyManifest
from omegaconf import OmegaConf

import nemo.collections.common.data.lhotse.dataloader as lhotse_dataloader

# Import the module only once (the tests monkeypatch attributes on it), and alias the public names used here.
LhotseDataLoadingConfig = lhotse_dataloader.LhotseDataLoadingConfig
get_lhotse_dataloader_from_config = lhotse_dataloader.get_lhotse_dataloader_from_config


class _CutIds(torch.utils.data.Dataset):
    def __getitem__(self, cuts):
        # A little consumer-side work so the background bucketing thread (if any) races with batch selection.
        time.sleep(0.003)
        return tuple(c.id for c in cuts)


@pytest.fixture(scope="module")
def cutset_path(tmp_path_factory):
    """48 cuts with varied durations so they spread across buckets."""
    cuts = DummyManifest(CutSet, begin_id=0, end_id=48, with_data=False)
    for i, c in enumerate(cuts):
        c.duration = 1.0 + (i * 7 % 13) / 4
    path = tmp_path_factory.mktemp("cb") / "cuts.jsonl.gz"
    cuts.to_file(path)
    return path


@pytest.fixture
def sampler_concurrent_flags(monkeypatch):
    """Records the ``concurrent`` argument of every DynamicBucketingSampler that NeMo builds."""
    flags = []

    class _SpySampler(lhotse_dataloader.DynamicBucketingSampler):
        def __init__(self, *args, **kwargs):
            flags.append(kwargs["concurrent"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(lhotse_dataloader, "DynamicBucketingSampler", _SpySampler)
    return flags


@pytest.fixture
def slow_source(monkeypatch):
    """Slows the cut source with a random per-cut jitter so a background bucketing thread would race."""
    original_read = lhotse_dataloader.read_cutset_from_config
    jitter = random.Random(1234)  # only drives sleep durations, never the data order

    def _slow(cut):
        time.sleep(jitter.random() * 0.004)
        return cut

    def _read_slowly(config):
        cuts, is_tarred = original_read(config)
        return cuts.map(_slow), is_tarred

    monkeypatch.setattr(lhotse_dataloader, "read_cutset_from_config", _read_slowly)


def _bucketing_options(**overrides):
    options = dict(
        shuffle=True,
        num_workers=0,
        use_bucketing=True,
        num_buckets=4,
        batch_duration=8.0,
        bucket_buffer_size=24,
        shuffle_buffer_size=10,
        seed=0,
        shard_seed=0,
        force_map_dataset=True,
        drop_last=False,
    )
    options.update(overrides)
    return options


def _build(cutset_path, **overrides):
    config = OmegaConf.create({"cuts_path": str(cutset_path), **_bucketing_options(**overrides)})
    return get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=_CutIds())


@pytest.mark.unit
def test_concurrent_bucketing_schema_default_is_auto():
    assert OmegaConf.structured(LhotseDataLoadingConfig).concurrent_bucketing is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "stateful,concurrent_bucketing,expected",
    [
        (False, None, True),  # unchanged non-stateful default
        (True, None, False),  # auto-disabled for exact resume
        (False, True, True),
        (False, False, False),
        (True, False, False),
    ],
)
def test_concurrent_bucketing_resolution(
    cutset_path, sampler_concurrent_flags, stateful, concurrent_bucketing, expected
):
    if stateful:
        pytest.importorskip("torchdata.stateful_dataloader")
    overrides = {"use_stateful_dataloader": stateful}
    if concurrent_bucketing is not None:
        overrides["concurrent_bucketing"] = concurrent_bucketing
    _build(cutset_path, **overrides)
    assert sampler_concurrent_flags == [expected]


@pytest.mark.unit
def test_concurrent_bucketing_explicitly_enabled_with_stateful_dataloader_raises(cutset_path):
    with pytest.raises(ValueError, match="concurrent_bucketing=True is incompatible with use_stateful_dataloader"):
        _build(cutset_path, use_stateful_dataloader=True, concurrent_bucketing=True)


@pytest.mark.unit
def test_concurrent_bucketing_ignored_without_bucketing(cutset_path):
    pytest.importorskip("torchdata.stateful_dataloader")
    # The flag only affects the bucketing sampler, so an explicit true is harmless without bucketing.
    dl = _build(cutset_path, use_stateful_dataloader=True, concurrent_bucketing=True, use_bucketing=False)
    assert len(list(dl)) > 0


def _multi_config(cutset_path, top_level, sub_config):
    source = {"cuts_path": str(cutset_path), **_bucketing_options(**sub_config)}
    for key in ("seed", "shard_seed", "shuffle", "num_workers"):
        source.pop(key)
    return OmegaConf.create(
        {"seed": 0, "shard_seed": 0, "shuffle": True, "num_workers": 0, "multi_config": True, **top_level, "a": source}
    )


@pytest.mark.unit
@pytest.mark.parametrize("stateful,expected", [(True, False), (False, True)])
def test_concurrent_bucketing_multi_config_follows_shared_stateful_flag(
    cutset_path, sampler_concurrent_flags, stateful, expected
):
    if stateful:
        pytest.importorskip("torchdata.stateful_dataloader")
    # use_stateful_dataloader is a shared top-level option that overwrites sub-configs.
    config = _multi_config(cutset_path, {"use_stateful_dataloader": stateful}, {})
    get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=_CutIds())
    assert sampler_concurrent_flags == [expected]


@pytest.mark.unit
def test_concurrent_bucketing_multi_config_explicit_true_with_stateful_raises(cutset_path):
    config = _multi_config(cutset_path, {"use_stateful_dataloader": True}, {"concurrent_bucketing": True})
    with pytest.raises(ValueError, match="concurrent_bucketing=True is incompatible"):
        get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=_CutIds())


@pytest.mark.unit
def test_stateful_bucketing_batch_order_is_deterministic(cutset_path, slow_source, sampler_concurrent_flags):
    """
    Regression test: with the old default (concurrent_bucketing=True), the background producer thread filled
    the buckets while batches were being selected, so the batch order depended on thread timing and the
    resumable dataloader could not reproduce the interrupted run.

    The source is slowed down with a random per-cut jitter so the race is observable on a few dozen cuts.
    Before the fix, most repetitions differed (but a single comparison can match by chance), so we also
    assert the mechanism: the sampler is built with ``concurrent=False``.
    """
    pytest.importorskip("torchdata.stateful_dataloader")
    runs = [list(_build(cutset_path, use_stateful_dataloader=True)) for _ in range(3)]

    assert sampler_concurrent_flags == [False, False, False]
    assert sum(len(b) for b in runs[0]) == 48
    assert runs[1] == runs[0]
    assert runs[2] == runs[0]


@pytest.mark.unit
def test_stateful_bucketing_resume_matches_uninterrupted_run(cutset_path, slow_source):
    """Save the loader state mid-epoch, restore it into a fresh loader and compare the remaining batches."""
    pytest.importorskip("torchdata.stateful_dataloader")
    expected = list(_build(cutset_path, use_stateful_dataloader=True))
    split = len(expected) // 2
    assert 0 < split < len(expected)

    interrupted = _build(cutset_path, use_stateful_dataloader=True)
    it = iter(interrupted)
    consumed = [next(it) for _ in range(split)]
    state = interrupted.state_dict()
    del it, interrupted

    resumed = _build(cutset_path, use_stateful_dataloader=True)
    resumed.load_state_dict(state)
    remaining = list(resumed)

    assert consumed == expected[:split]
    assert remaining == expected[split:]
