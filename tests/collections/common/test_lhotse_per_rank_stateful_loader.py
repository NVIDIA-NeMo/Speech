# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for ``_PerRankStatefulDataLoader``.

The wrapper exists because Lightning's ``FitLoop`` saves
``CombinedLoader._state_dicts()`` (which captures rank-0's
``StatefulDataLoader.state_dict()`` only) and replays it on every rank on
resume — broadcasting rank-0's iterator state to every rank and corrupting
per-shard partitioning. The wrapper intercepts that pipeline so the saved
payload is a per-rank list and the load picks the right entry.

These tests intentionally do not spin up torch.distributed; the
all_gather path is the trivial 1-rank fallback. The
:func:`test_load_picks_correct_rank_entry` test simulates the multi-rank
case by handing the wrapper an externally-built per-rank state dict and
asserting the right inner state lands on the inner loader (proxied by a
stub that records what ``load_state_dict`` was called with).
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from nemo.collections.common.data.lhotse.dataloader import _build_dataloader, _PerRankStatefulDataLoader


class _StubStatefulDataLoader:
    """Stand-in for ``torchdata.stateful_dataloader.StatefulDataLoader``.

    The wrapper's tests only need ``state_dict()`` and ``load_state_dict()``
    to be observable; they don't care about iteration. We install this stub
    as the ``StatefulDataLoader`` import inside the wrapper module so the
    test runs without ``torchdata`` and stays focused on the gather/scatter
    logic we own.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self._state: dict = {"position": 0, "shard_id": None}
        self.load_calls: list[dict] = []

    def state_dict(self) -> dict:
        return dict(self._state)

    def load_state_dict(self, state_dict: dict) -> None:
        # record the call so tests can assert what was applied.
        self.load_calls.append(state_dict)
        self._state.update(state_dict)


@pytest.fixture(autouse=True)
def _patch_stateful_loader(monkeypatch):
    """Make ``from torchdata.stateful_dataloader import StatefulDataLoader``
    inside the wrapper resolve to our stub."""
    fake_module = types.ModuleType("torchdata.stateful_dataloader")
    fake_module.StatefulDataLoader = _StubStatefulDataLoader
    fake_pkg = types.ModuleType("torchdata")
    fake_pkg.stateful_dataloader = fake_module
    monkeypatch.setitem(sys.modules, "torchdata", fake_pkg)
    monkeypatch.setitem(sys.modules, "torchdata.stateful_dataloader", fake_module)


def _new_wrapper(dp_rank: int, dp_world_size: int, dp_group=None) -> _PerRankStatefulDataLoader:
    return _PerRankStatefulDataLoader(
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        dp_group=dp_group,
        # the stub ignores constructor kwargs, but we pass something
        # representative so the call signature mirrors real usage.
        dataset=object(),
        num_workers=4,
    )


def test_build_dataloader_forwards_dp_group():
    dp_group = object()

    dl = _build_dataloader(
        use_stateful_dataloader=True,
        dp_rank=1,
        dp_world_size=2,
        dp_group=dp_group,
        dataset=object(),
        batch_size=None,
        num_workers=0,
    )

    assert isinstance(dl, _PerRankStatefulDataLoader)
    assert dl._dp_group is dp_group


def test_state_dict_all_gather_uses_dp_group(monkeypatch):
    dp_group = object()
    dl = _new_wrapper(dp_rank=1, dp_world_size=2, dp_group=dp_group)
    dl._inner._state = {"position": 43, "shard_id": 1}
    calls = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_gather_object(per_rank, tagged, group=None):
        calls.append(group)
        per_rank[0] = {"dp_rank": 0, "dp_world_size": 2, "state": {"position": 42, "shard_id": 0}}
        per_rank[1] = tagged

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    sd = dl.state_dict()

    assert calls == [dp_group]
    assert sd["train_dataloader_per_rank"][1]["state"] == {"position": 43, "shard_id": 1}


def test_state_dict_single_rank_wraps_with_per_rank_list():
    dl = _new_wrapper(dp_rank=0, dp_world_size=1)
    dl._inner._state = {"position": 42, "shard_id": 0}

    sd = dl.state_dict()

    assert list(sd.keys()) == ["train_dataloader_per_rank"]
    per_rank = sd["train_dataloader_per_rank"]
    assert isinstance(per_rank, list) and len(per_rank) == 1
    assert per_rank[0] == {
        "dp_rank": 0,
        "dp_world_size": 1,
        "state": {"position": 42, "shard_id": 0},
    }


def test_load_state_dict_single_rank_unwraps_and_applies():
    dl = _new_wrapper(dp_rank=0, dp_world_size=1)

    dl.load_state_dict(
        {
            "train_dataloader_per_rank": [
                {"dp_rank": 0, "dp_world_size": 1, "state": {"position": 99, "shard_id": 0}},
            ]
        }
    )

    assert dl._inner.load_calls == [{"position": 99, "shard_id": 0}]


def test_load_picks_correct_rank_entry():
    """Hand a 32-rank per_rank list to a wrapper bound to rank 28; assert
    the inner loader receives rank 28's entry only.

    This is the regression for the 2026-05-14 LOAD-side bug: Lightning's
    FitLoop replays the saved state on every rank, and historically rank 28
    ended up applying rank 0's worker-0 state because the broadcast came
    from FitLoop AFTER our DataModule's per-rank load. With the wrapper,
    even the FitLoop's broadcast goes through the right per-rank scatter.
    """
    world = 32
    rank = 28
    per_rank = [
        {
            "dp_rank": r,
            "dp_world_size": world,
            "state": {"position": 100 + r, "shard_id": r * 4},
        }
        for r in range(world)
    ]

    dl = _new_wrapper(dp_rank=rank, dp_world_size=world)
    dl.load_state_dict({"train_dataloader_per_rank": per_rank})

    assert len(dl._inner.load_calls) == 1
    applied = dl._inner.load_calls[0]
    assert applied == {"position": 100 + rank, "shard_id": rank * 4}, (
        "Wrapper must consume per_rank[self._dp_rank] — bug would manifest "
        "as applying per_rank[0] (rank-0 broadcast collapse)."
    )


def test_load_rejects_world_size_mismatch():
    dl = _new_wrapper(dp_rank=0, dp_world_size=32)
    with pytest.raises(RuntimeError, match="dp_world_size"):
        dl.load_state_dict(
            {
                "train_dataloader_per_rank": [
                    {"dp_rank": 0, "dp_world_size": 4, "state": {}},
                    {"dp_rank": 1, "dp_world_size": 4, "state": {}},
                    {"dp_rank": 2, "dp_world_size": 4, "state": {}},
                    {"dp_rank": 3, "dp_world_size": 4, "state": {}},
                ]
            }
        )


def test_load_rejects_tag_mismatch():
    dl = _new_wrapper(dp_rank=0, dp_world_size=2)
    with pytest.raises(RuntimeError, match=r"tagged \(dp_rank=1"):
        dl.load_state_dict(
            {
                "train_dataloader_per_rank": [
                    # the entry at index 0 claims to be rank 1 — must reject.
                    {"dp_rank": 1, "dp_world_size": 2, "state": {}},
                    {"dp_rank": 1, "dp_world_size": 2, "state": {}},
                ]
            }
        )


def test_load_rejects_bare_inner_state():
    """Strict wire format: a state dict without the
    ``train_dataloader_per_rank`` top-level key is rejected. This guards
    against the legacy code path (``DataModule.load_state_dict`` calling
    ``dl.load_state_dict(entry["state"])`` with the raw inner state) and
    against Lightning's FitLoop broadcasting rank-0's
    ``StatefulDataLoader.state_dict()`` — both would otherwise look like
    valid bare inner state and produce wrong, silently-corrupt resumes.
    """
    dl = _new_wrapper(dp_rank=0, dp_world_size=1)

    with pytest.raises(RuntimeError, match="train_dataloader_per_rank"):
        dl.load_state_dict({"position": 7, "shard_id": 0})

    # an inner-shaped state (with ``_snapshot._worker_snapshots`` etc.) —
    # what Lightning's FitLoop used to feed back — must be rejected too.
    with pytest.raises(RuntimeError, match="train_dataloader_per_rank"):
        dl.load_state_dict(
            {
                "_iterator_finished": False,
                "_snapshot": {"_worker_snapshots": {"worker_0": {}}},
                "_steps_since_snapshot": 0,
            }
        )


def test_empty_state_is_a_noop():
    dl = _new_wrapper(dp_rank=0, dp_world_size=1)
    dl.load_state_dict({})
    assert dl._inner.load_calls == []
