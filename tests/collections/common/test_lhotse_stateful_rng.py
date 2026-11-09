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

import pickle
import random
from collections import defaultdict

import pytest
import torch
from lhotse import CutSet
from lhotse.dataset import SimpleCutSampler
from lhotse.testing.dummies import DummyManifest
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.stateful_rng import StatefulIterableDatasetWrapper, StatefulRandomMixin

pytestmark = pytest.mark.unit


class _RandomDrawDataset(StatefulRandomMixin, torch.utils.data.Dataset):
    """Returns one random draw per cut, like a random crop offset."""

    def __getitem__(self, cuts: CutSet) -> list[tuple[str, float]]:
        return [(cut.id, self.rng.random()) for cut in cuts]


class _StatelessDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet) -> list[str]:
        return [cut.id for cut in cuts]


def test_rng_state_dict_round_trip():
    dataset = _RandomDrawDataset()
    dataset.rng.random()
    state = dataset.state_dict()
    expected = [dataset.rng.random() for _ in range(5)]

    restored = _RandomDrawDataset()
    restored.load_state_dict(pickle.loads(pickle.dumps(state)))

    assert [restored.rng.random() for _ in range(5)] == expected


def test_rng_state_dict_uses_plain_types():
    state = _RandomDrawDataset().state_dict()["rng_state"]
    assert isinstance(state["version"], int)
    assert isinstance(state["internal_state"], list)
    assert all(isinstance(x, int) for x in state["internal_state"])


def test_rng_is_seeded_from_torch_initial_seed():
    torch.manual_seed(1)
    first = _RandomDrawDataset().rng.random()
    torch.manual_seed(1)
    same = _RandomDrawDataset().rng.random()
    torch.manual_seed(2)
    other = _RandomDrawDataset().rng.random()

    assert first == same
    assert first != other


def test_rng_ignores_global_random_state():
    torch.manual_seed(1)
    random.seed(1)
    first = _RandomDrawDataset().rng.random()
    torch.manual_seed(1)
    random.seed(2)
    assert _RandomDrawDataset().rng.random() == first


def test_iterable_wrapper_checkpoints_dataset_state():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    wrapper = StatefulIterableDatasetWrapper(_RandomDrawDataset(), SimpleCutSampler(cuts, max_cuts=2))
    it = iter(wrapper)
    next(it)
    state = wrapper.state_dict()
    expected = list(it)

    restored = StatefulIterableDatasetWrapper(_RandomDrawDataset(), SimpleCutSampler(cuts, max_cuts=2))
    restored.load_state_dict(state)

    assert "dataset_state" in state
    assert list(iter(restored)) == expected


def test_iterable_wrapper_accepts_checkpoint_without_dataset_state():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=4)
    wrapper = StatefulIterableDatasetWrapper(_RandomDrawDataset(), SimpleCutSampler(cuts, max_cuts=2))
    state = wrapper.state_dict()
    del state["dataset_state"]

    wrapper.load_state_dict(state)

    assert len(list(iter(wrapper))) == 2


def test_iterable_wrapper_raises_on_dataset_state_for_stateless_dataset():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=4)
    state = StatefulIterableDatasetWrapper(_RandomDrawDataset(), SimpleCutSampler(cuts, max_cuts=2)).state_dict()
    wrapper = StatefulIterableDatasetWrapper(_StatelessDataset(), SimpleCutSampler(cuts, max_cuts=2))

    with pytest.raises(RuntimeError, match="does not implement load_state_dict"):
        wrapper.load_state_dict(state)


@pytest.fixture(scope="module")
def cuts_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("stateful_rng") / "cuts.jsonl.gz"
    DummyManifest(CutSet, begin_id=0, end_id=20).to_file(path)
    return path


def _make_dataloader(cuts_path, mode: str):
    config = OmegaConf.create(
        {
            "cuts_path": str(cuts_path),
            "batch_size": 2,
            "shuffle": True,
            "use_bucketing": False,
            "num_workers": 0,
            "seed": 0,
            "shard_seed": 0,
            "use_stateful_dataloader": True,
            "force_map_dataset": mode == "map",
            "force_iterable_dataset": mode == "iterable",
        }
    )
    return get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=_RandomDrawDataset())


def _take(cuts_path, mode: str, num_batches: int, state=None):
    dloader = _make_dataloader(cuts_path, mode)
    if state is not None:
        dloader.load_state_dict(state)
    it = iter(dloader)
    batches = [next(it) for _ in range(num_batches)]
    return batches, dloader.state_dict()


@pytest.mark.parametrize("mode", ["map", "iterable"])
def test_resumed_dataloader_reproduces_random_draws(cuts_path, mode):
    uninterrupted, _ = _take(cuts_path, mode, num_batches=8)
    _, state = _take(cuts_path, mode, num_batches=3)
    resumed, _ = _take(cuts_path, mode, num_batches=5, state=state)

    assert resumed == uninterrupted[3:]


@pytest.mark.parametrize("mode", ["map", "iterable"])
def test_random_draws_differ_across_epochs(cuts_path, mode):
    dloader = _make_dataloader(cuts_path, mode)
    draws = defaultdict(list)
    for epoch in range(2):
        for batch in dloader:
            for cut_id, value in batch:
                draws[cut_id].append(value)

    assert len(draws) == 20
    assert all(len(values) == 2 and values[0] != values[1] for values in draws.values())
