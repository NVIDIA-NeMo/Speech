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

from copy import deepcopy

import pytest
import torch
from lhotse import CutSet
from lhotse.dataset import DynamicCutSampler
from lhotse.indexing import create_jsonl_index
from lhotse.testing.dummies import DummyManifest

from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

pytest.importorskip("torchdata.stateful_dataloader")

# Newer Lhotse cannot checkpoint a streaming sampler; retain the old fixture on
# versions that still support replay, so the same tests cover the current pin.
INDEXED_ONLY_CHECKPOINTS = hasattr(DynamicCutSampler, "_consume_pending_restore")

LIMIT, SPLIT, TOTAL = 4, 6, 12


@pytest.fixture(scope="module")
def cuts_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("epoch_boundary") / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=30).to_jsonl(path)
    create_jsonl_index(path)
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


class _CutIds(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet) -> list[str]:
        return [cut.id for cut in cuts]


def _make_dataloader(cuts_path):
    config = {
        "cuts_path": str(cuts_path),
        "indexed": INDEXED_ONLY_CHECKPOINTS,
        "batch_size": 3,
        "shuffle": True,
        "seed": 0,
        "shard_seed": 0,
        "num_workers": 0,
        "force_map_dataset": True,
        "use_stateful_dataloader": True,
    }
    return get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=_CutIds())
