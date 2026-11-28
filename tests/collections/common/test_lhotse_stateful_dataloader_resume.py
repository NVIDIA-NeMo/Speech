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
"""Resume tests for ``use_stateful_dataloader: true`` through ``get_lhotse_dataloader_from_config``.

Unlike ``test_lhotse_per_rank_stateful_loader.py``, these tests use the real torchdata
``StatefulDataLoader`` (no stub): consume some batches, save the state, restore it into a
fresh loader, and compare the remaining batches with the uninterrupted run.
"""

from copy import deepcopy

import pytest
import torch
from lhotse import CutSet
from lhotse.indexing import create_jsonl_index
from lhotse.testing.dummies import dummy_cut
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.dataloader import (
    _PerRankStatefulDataLoader,
    get_lhotse_dataloader_from_config,
)

pytest.importorskip("torchdata.stateful_dataloader")

NUM_CUTS = 12
BATCH_SIZE = 2
NUM_BATCHES = NUM_CUTS // BATCH_SIZE
CONSUMED = 2


class _CutIds(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet) -> list[str]:
        return [cut.id for cut in cuts]


@pytest.fixture(scope="module")
def cuts_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("stateful_resume") / "cuts.jsonl"
    CutSet.from_cuts(dummy_cut(idx, duration=1.0) for idx in range(NUM_CUTS)).to_file(path)
    create_jsonl_index(path)
    return path


def _make_loader(cuts_path, num_workers: int, iterable: bool):
    config = OmegaConf.create(
        {
            "cuts_path": str(cuts_path),
            "batch_size": BATCH_SIZE,
            "num_workers": num_workers,
            "shuffle": True,
            "seed": 0,
            "shard_seed": 0,
            "drop_last": True,
            "use_stateful_dataloader": True,
            "indexed": iterable,
            "force_iterable_dataset": iterable,
            "force_map_dataset": not iterable,
            "force_finite": True,
        }
    )
    dl = get_lhotse_dataloader_from_config(config, global_rank=0, world_size=1, dataset=_CutIds())
    assert isinstance(dl, _PerRankStatefulDataLoader)
    return dl


def _shutdown(dl):
    iterator = getattr(dl._inner, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


@pytest.fixture(scope="module")
def runs(cuts_path):
    """Cache the uninterrupted batches and the state saved after ``CONSUMED`` batches per
    ``(num_workers, iterable)``, so worker processes are started as rarely as possible."""
    cache = {}

    def get(num_workers: int, iterable: bool):
        key = (num_workers, iterable)
        if key not in cache:
            full = _make_loader(cuts_path, num_workers, iterable)
            expected = list(full)
            _shutdown(full)

            partial = _make_loader(cuts_path, num_workers, iterable)
            it = iter(partial)
            prefix = [next(it) for _ in range(CONSUMED)]
            state = deepcopy(partial.state_dict())
            _shutdown(partial)
            cache[key] = expected, prefix, state
        expected, prefix, state = cache[key]
        return expected, prefix, deepcopy(state)

    return get


def _maybe_legacy(state: dict, legacy_bare_state: bool) -> dict:
    if not legacy_bare_state:
        return state
    # Single-GPU checkpoints written before the wrapper hold the bare StatefulDataLoader state;
    # take it from a real inner loader rather than writing the payload by hand.
    bare = state["train_dataloader_per_rank"][0]["state"]
    assert "train_dataloader_per_rank" not in bare
    return bare


@pytest.mark.unit
@pytest.mark.parametrize("iterable", [False, True], ids=["map", "indexed_iterable"])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("legacy_bare_state", [False, True], ids=["wrapped", "legacy_bare"])
def test_stateful_resume_matches_uninterrupted_run(cuts_path, runs, num_workers, iterable, legacy_bare_state):
    expected, prefix, state = runs(num_workers, iterable)
    assert len(expected) == NUM_BATCHES
    assert all(len(batch) == BATCH_SIZE for batch in expected)
    assert prefix == expected[:CONSUMED]

    resumed = _make_loader(cuts_path, num_workers, iterable)
    resumed.load_state_dict(_maybe_legacy(state, legacy_bare_state))
    suffix = list(resumed)
    _shutdown(resumed)

    assert suffix == expected[CONSUMED:]


@pytest.mark.unit
@pytest.mark.parametrize("iterable", [False, True], ids=["map", "indexed_iterable"])
@pytest.mark.parametrize("saved_workers,resumed_workers", [(2, 0), (0, 2)])
@pytest.mark.parametrize("legacy_bare_state", [False, True], ids=["wrapped", "legacy_bare"])
def test_stateful_resume_rejects_num_workers_change(
    cuts_path, runs, saved_workers, resumed_workers, iterable, legacy_bare_state
):
    _, _, state = runs(saved_workers, iterable)

    resumed = _make_loader(cuts_path, resumed_workers, iterable)
    with pytest.raises(
        RuntimeError, match=rf"saved with num_workers={saved_workers}.*current run has num_workers={resumed_workers}"
    ):
        resumed.load_state_dict(_maybe_legacy(state, legacy_bare_state))
    # The error is raised in load_state_dict, before torchdata receives the state or iteration starts.
    assert resumed._inner.next_iter_state is None
    assert resumed._inner._iterator is None


@pytest.mark.unit
def test_is_dataloader_recognises_stateful_wrapper_only_for_loaders(cuts_path):
    import numpy as np

    from nemo.collections.common.data.utils import is_dataloader

    assert is_dataloader(_make_loader(cuts_path, num_workers=0, iterable=False))
    assert is_dataloader(torch.utils.data.DataLoader(list(range(4))))
    for not_a_loader in ["a.wav", ["a.wav", "b.wav"], np.zeros(4), torch.zeros(4), [torch.zeros(4)]]:
        assert not is_dataloader(not_a_loader)


@pytest.mark.unit
def test_is_dataloader_without_lhotse_returns_false_for_non_loaders(monkeypatch):
    """With Lhotse not installed, non-loaders return False instead of raising ModuleNotFoundError."""
    import sys

    import numpy as np

    from nemo.collections.common.data.utils import is_dataloader

    # Block Lhotse (``sys.modules[name] = None`` makes both ``import`` and ``find_spec`` fail) and drop
    # the cached NeMo Lhotse modules so that an unguarded import would actually be attempted.
    for name in list(sys.modules):
        if name == "lhotse" or name.startswith("lhotse.") or name.startswith("nemo.collections.common.data.lhotse"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "lhotse", None)
    with pytest.raises(ImportError):
        import lhotse  # noqa: F401

    assert is_dataloader(torch.utils.data.DataLoader(list(range(4))))
    for not_a_loader in ["a.wav", None, ["a.wav"], np.zeros(4), torch.zeros(4)]:
        assert not is_dataloader(not_a_loader)
