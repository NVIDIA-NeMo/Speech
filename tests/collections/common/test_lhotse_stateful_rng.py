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

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config, stateful_rng
from nemo.collections.common.data.lhotse.stateful_rng import StatefulIterableDatasetWrapper, StatefulRandomMixin

pytestmark = pytest.mark.unit


class _RandomDrawDataset(StatefulRandomMixin, torch.utils.data.Dataset):
    """Returns one random draw per cut (like a random crop offset), with the id of the producing worker."""

    def __getitem__(self, cuts: CutSet) -> list[tuple[str, int, float]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = -1 if worker_info is None else worker_info.id
        return [(cut.id, worker_id, self.rng.random()) for cut in cuts]


class _StatelessDataset(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet) -> list[str]:
        return [cut.id for cut in cuts]


def _enabled() -> _RandomDrawDataset:
    dataset = _RandomDrawDataset()
    dataset.enable_checkpointed_rng()
    return dataset


def test_rng_is_the_global_module_unless_enabled():
    dataset = _RandomDrawDataset()

    assert not dataset.checkpointed_rng_enabled
    assert dataset.rng is random
    assert dataset.state_dict() == {"rng_state": None}
    dataset.load_state_dict({"rng_state": None})  # a no-op
    assert dataset.rng is random


def test_disabled_rng_rejects_saved_generator_state():
    state = _enabled().state_dict()

    with pytest.raises(RuntimeError, match="not enabled"):
        _RandomDrawDataset().load_state_dict(state)


def test_rng_state_dict_round_trip():
    dataset = _enabled()
    dataset.rng.random()
    state = dataset.state_dict()
    expected = [dataset.rng.random() for _ in range(5)]

    restored = _enabled()
    restored.load_state_dict(pickle.loads(pickle.dumps(state)))

    assert [restored.rng.random() for _ in range(5)] == expected


def test_rng_state_round_trip_keeps_cached_gaussian():
    dataset = _enabled()
    dataset.rng.gauss(0.0, 1.0)  # an odd number of gauss() calls leaves the second value cached
    state = dataset.state_dict()
    expected = [dataset.rng.gauss(0.0, 1.0) for _ in range(3)]

    restored = _enabled()
    restored.load_state_dict(state)

    assert state["rng_state"]["gauss_next"] is not None
    assert [restored.rng.gauss(0.0, 1.0) for _ in range(3)] == expected


def test_load_state_dict_restores_in_place():
    dataset = _enabled()
    holder = dataset.rng  # e.g. a G2P module holding a reference to the dataset generator
    state = dataset.state_dict()
    expected = [dataset.rng.random() for _ in range(3)]

    dataset.load_state_dict(state)

    assert dataset.rng is holder
    assert [holder.random() for _ in range(3)] == expected


def test_rng_state_dict_uses_plain_types():
    state = _enabled().state_dict()["rng_state"]
    assert isinstance(state["version"], int)
    assert isinstance(state["internal_state"], list)
    assert all(isinstance(x, int) for x in state["internal_state"])


def test_rng_is_seeded_from_torch_initial_seed():
    torch.manual_seed(1)
    first = _enabled().rng.random()
    torch.manual_seed(1)
    same = _enabled().rng.random()
    torch.manual_seed(2)
    other = _enabled().rng.random()

    assert first == same
    assert first != other


def test_rng_ignores_global_random_state():
    torch.manual_seed(1)
    random.seed(1)
    first = _enabled().rng.random()
    torch.manual_seed(1)
    random.seed(2)
    assert _enabled().rng.random() == first


def test_rng_inherited_from_another_process_is_reseeded(monkeypatch):
    dataset = _enabled()
    parent_rng = dataset.rng
    monkeypatch.setattr(stateful_rng.os, "getpid", lambda: -1)

    assert dataset.rng is not parent_rng


def test_iterable_wrapper_checkpoints_dataset_state():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    wrapper = StatefulIterableDatasetWrapper(_enabled(), SimpleCutSampler(cuts, max_cuts=2))
    it = iter(wrapper)
    next(it)
    state = wrapper.state_dict()
    expected = list(it)

    restored = StatefulIterableDatasetWrapper(_enabled(), SimpleCutSampler(cuts, max_cuts=2))
    restored.load_state_dict(state)

    assert "dataset_state" in state
    assert list(iter(restored)) == expected


def test_iterable_wrapper_accepts_checkpoint_without_dataset_state():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=4)
    wrapper = StatefulIterableDatasetWrapper(_enabled(), SimpleCutSampler(cuts, max_cuts=2))
    state = wrapper.state_dict()
    del state["dataset_state"]

    wrapper.load_state_dict(state)

    assert len(list(iter(wrapper))) == 2


def test_iterable_wrapper_raises_on_dataset_state_for_stateless_dataset():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=4)
    state = StatefulIterableDatasetWrapper(_enabled(), SimpleCutSampler(cuts, max_cuts=2)).state_dict()
    wrapper = StatefulIterableDatasetWrapper(_StatelessDataset(), SimpleCutSampler(cuts, max_cuts=2))

    with pytest.raises(RuntimeError, match="does not implement load_state_dict"):
        wrapper.load_state_dict(state)


@pytest.fixture(scope="module")
def cuts_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("stateful_rng") / "cuts.jsonl.gz"
    DummyManifest(CutSet, begin_id=0, end_id=20).to_file(path)
    return path


def _make_dataloader(cuts_path, mode: str, num_workers: int = 0, use_stateful_dataloader: bool = True):
    config = OmegaConf.create(
        {
            "cuts_path": str(cuts_path),
            "batch_size": 2,
            "shuffle": True,
            "use_bucketing": False,
            "num_workers": num_workers,
            "seed": 0,
            "shard_seed": 0,
            "use_stateful_dataloader": use_stateful_dataloader,
            "force_map_dataset": mode == "map",
            "force_iterable_dataset": mode == "iterable",
        }
    )
    # Seed first so the generator initialized below doesn't depend on the ambient torch seed.
    torch.manual_seed(config.seed)
    dataset = _RandomDrawDataset()
    # Initialize the generator in the parent process, as a state_dict() or a diagnostic read before the loader
    # starts would. Dataloader workers must still each get their own stream.
    dataset.enable_checkpointed_rng()
    dataset.state_dict()
    return get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=dataset)


def _take(cuts_path, mode: str, num_batches: int, state=None, num_workers: int = 0):
    dloader = _make_dataloader(cuts_path, mode, num_workers)
    if state is not None:
        dloader.load_state_dict(state)
    it = iter(dloader)
    batches = [next(it) for _ in range(num_batches)]
    return batches, dloader.state_dict()


def test_builder_enables_checkpointed_rng_only_for_stateful_loader(cuts_path):
    assert _make_dataloader(cuts_path, "map").dataset.checkpointed_rng_enabled
    plain = _RandomDrawDataset()
    config = OmegaConf.create({"cuts_path": str(cuts_path), "batch_size": 2, "force_map_dataset": True})
    get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=plain)
    assert not plain.checkpointed_rng_enabled


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("mode", ["map", "iterable"])
def test_resumed_dataloader_reproduces_random_draws(cuts_path, mode, num_workers):
    uninterrupted, _ = _take(cuts_path, mode, num_batches=8, num_workers=num_workers)
    _, state = _take(cuts_path, mode, num_batches=3, num_workers=num_workers)
    resumed, _ = _take(cuts_path, mode, num_batches=5, state=state, num_workers=num_workers)

    assert resumed == uninterrupted[3:]


@pytest.mark.parametrize("mode", ["map", "iterable"])
def test_dataloader_workers_draw_independent_streams(cuts_path, mode):
    batches, _ = _take(cuts_path, mode, num_batches=4, num_workers=2)
    first_draws = {}
    for batch in batches:
        for _, worker_id, value in batch:
            first_draws.setdefault(worker_id, value)

    assert set(first_draws) == {0, 1}
    assert first_draws[0] != first_draws[1]


@pytest.mark.parametrize("mode", ["map", "iterable"])
def test_random_draws_differ_across_epochs(cuts_path, mode):
    dloader = _make_dataloader(cuts_path, mode)
    draws = defaultdict(list)
    for epoch in range(2):
        for batch in dloader:
            for cut_id, _, value in batch:
                draws[cut_id].append(value)

    assert len(draws) == 20
    assert all(len(values) == 2 and values[0] != values[1] for values in draws.values())
