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
"""A resumed map-style Lhotse ``StatefulDataLoader`` must match the uninterrupted run across epoch boundaries.

These tests drive the dataloader the way Lightning does, without Lightning: at the start of each epoch the sampler's
``set_epoch`` is called and the loader is iterated for ``limit`` batches (``limit_train_batches``). On resume,
Lightning loads the dataloader state, creates the iterator, and then calls ``set_epoch`` for the resumed epoch.
"""

import os
import pickle
import subprocess
import sys
from copy import deepcopy

import pytest
import torch
from lhotse import CutSet
from lhotse.testing.dummies import DummyManifest

from nemo.collections.common.data.lhotse.dataloader import (
    _build_dataloader,
    _PerRankStatefulDataLoader,
    get_lhotse_dataloader_from_config,
)

pytest.importorskip("torchdata.stateful_dataloader")

LIMIT, SPLIT, TOTAL = 4, 6, 12


@pytest.fixture(scope="module")
def cuts_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("epoch_boundary") / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=30).to_jsonl(path)
    return path


@pytest.mark.unit
def test_map_stateful_dataloader_resume_matches_across_epoch_boundary(cuts_path):
    # The map-style sampler lives in the main process for any num_workers, and torchdata restores it the same way
    # (index sampler, iter(), sampler iterator). num_workers=0 keeps the test fast: spawning workers that import
    # NeMo takes most of a minute.
    expected, state = [], None
    dloader = _make_dataloader(cuts_path)
    for epoch in range(TOTAL // LIMIT):
        dloader.sampler.set_epoch(epoch)
        for _, batch in zip(range(LIMIT), dloader):
            expected.append(batch)
            if len(expected) == SPLIT:
                state = deepcopy(dloader.state_dict())

    actual = []
    dloader = _make_dataloader(cuts_path)
    dloader.load_state_dict(state)
    iterator = iter(dloader)
    epoch = SPLIT // LIMIT
    dloader.sampler.set_epoch(epoch)
    actual.extend(batch for _, batch in zip(range(LIMIT - SPLIT % LIMIT), iterator))
    for epoch in range(epoch + 1, TOTAL // LIMIT):
        dloader.sampler.set_epoch(epoch)
        actual.extend(batch for _, batch in zip(range(LIMIT), dloader))

    assert actual == expected[SPLIT:]


@pytest.mark.unit
def test_pending_restore_is_cleared_only_for_restored_loaders(cuts_path, monkeypatch):
    dloader = _make_dataloader(cuts_path)
    calls = []
    monkeypatch.setattr(dloader.sampler, "allow_iter_to_reset_state", lambda: calls.append(1))
    next(iter(dloader))
    assert calls == []  # a fresh loader keeps the sampler untouched
    state = deepcopy(dloader.state_dict())

    dloader.load_state_dict(state)
    next(iter(dloader))
    assert calls == [1]


@pytest.mark.unit
def test_stateful_dataloader_is_picklable(cuts_path):
    # Spawn-based launchers pickle models that hold their dataloaders.
    dloader = _make_dataloader(cuts_path)
    restored = pickle.loads(pickle.dumps(dloader))
    assert next(iter(restored)) == next(iter(_make_dataloader(cuts_path)))

    minimal = _build_dataloader(True, dataset=torch.utils.data.TensorDataset(torch.arange(4)), batch_size=2)
    assert torch.equal(next(iter(pickle.loads(pickle.dumps(minimal))))[0], torch.tensor([0, 1]))

    per_rank = _build_dataloader(
        True, dp_rank=0, dp_world_size=2, dataset=torch.utils.data.TensorDataset(torch.arange(4)), batch_size=2
    )
    assert isinstance(per_rank, _PerRankStatefulDataLoader)
    assert torch.equal(next(iter(pickle.loads(pickle.dumps(per_rank))))[0], torch.tensor([0, 1]))


@pytest.mark.unit
def test_dataloader_import_does_not_import_torchdata_subclass():
    module = "nemo.collections.common.data.lhotse._stateful_dataloader"
    code = f"import sys, nemo.collections.common.data.lhotse.dataloader; assert {module!r} not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": ":".join(sys.path)})


class _CutIds(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet) -> list[str]:
        return [cut.id for cut in cuts]


def _make_dataloader(cuts_path):
    config = {
        "cuts_path": str(cuts_path),
        "batch_size": 3,
        "shuffle": True,
        "seed": 0,
        "shard_seed": 0,
        "num_workers": 0,
        "force_map_dataset": True,
        "use_stateful_dataloader": True,
    }
    return get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=_CutIds())
