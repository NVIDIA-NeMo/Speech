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
"""Tests for nemo/collections/common/data/lhotse/broadcasting.py.

Fake-mesh tests run on CPU without a real distributed group — they
exercise the noop short-circuits and the rank-coordinate logic. The
gloo-based multiprocess tests verify the broadcast contract end-to-end
on a 2-rank CPU group.
"""
from __future__ import annotations

import os
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo.collections.common.data.lhotse import broadcasting
from nemo.collections.common.data.lhotse.broadcasting import BroadcastingDataLoader, broadcast_batch, is_dp_source_rank

# ---------------------------------------------------------------------------
# Fake-mesh CPU-only tests (no distributed required).
# ---------------------------------------------------------------------------


class _FakeAxis:
    def __init__(self, size: int, local_rank: int, group=None, ranks=None):
        self._size = size
        self._local_rank = local_rank
        self._group = group
        self.mesh = torch.tensor(range(size) if ranks is None else ranks)

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._local_rank

    def get_group(self):
        return self._group


class _FakeMesh:
    """Minimal DeviceMesh stand-in covering ``mesh_dim_names`` + ``__getitem__``."""

    def __init__(self, sizes: dict[str, int], coords: dict[str, int], groups=None, ranks=None):
        assert sizes.keys() == coords.keys()
        self.mesh_dim_names = tuple(sizes.keys())
        self._sizes = sizes
        self._coords = coords
        self._groups = groups or {}
        self._ranks = ranks or {}

    def __getitem__(self, name):
        if isinstance(name, tuple):
            raise NotImplementedError("multi-axis slicing not needed for fake-mesh tests")
        return _FakeAxis(
            self._sizes[name],
            self._coords[name],
            group=self._groups.get(name),
            ranks=self._ranks.get(name),
        )


def test_is_dp_source_rank_none_mesh():
    assert is_dp_source_rank(None) is True


def test_is_dp_source_rank_all_axes_size_one():
    mesh = _FakeMesh({"cp": 1, "tp": 1}, {"cp": 0, "tp": 0})
    assert is_dp_source_rank(mesh) is True


def test_is_dp_source_rank_no_relevant_axes():
    mesh = _FakeMesh({"dp": 2}, {"dp": 1})
    assert is_dp_source_rank(mesh) is True


@pytest.mark.parametrize(
    "coords, expected",
    [
        ({"cp": 0, "tp": 0}, True),
        ({"cp": 1, "tp": 0}, False),
        ({"cp": 0, "tp": 1}, False),
        ({"cp": 1, "tp": 1}, False),
    ],
)
def test_is_dp_source_rank_cp_tp_grid(coords, expected):
    mesh = _FakeMesh({"cp": 2, "tp": 2}, coords)
    assert is_dp_source_rank(mesh) is expected


def test_is_dp_source_rank_only_cp_axis():
    mesh = _FakeMesh({"cp": 4}, {"cp": 3})
    assert is_dp_source_rank(mesh) is False


def test_broadcast_batch_noop_returns_input():
    payload = {"x": torch.arange(4)}
    out = broadcast_batch(payload, None)
    assert out is payload


def test_broadcast_batch_noop_when_axes_size_one():
    mesh = _FakeMesh({"cp": 1, "tp": 1}, {"cp": 0, "tp": 0})
    payload = "anything"
    assert broadcast_batch(payload, mesh) is payload


def test_group_resolution_does_not_cache_by_object_id(monkeypatch):
    group_a = object()
    group_b = object()
    mesh_a = _FakeMesh(
        {"cp": 2},
        {"cp": 0},
        groups={"cp": group_a},
        ranks={"cp": [2, 3]},
    )
    mesh_b = _FakeMesh(
        {"cp": 2},
        {"cp": 0},
        groups={"cp": group_b},
        ranks={"cp": [6, 7]},
    )
    monkeypatch.setattr(broadcasting, "id", lambda _: 1, raising=False)

    assert broadcasting._resolve_group_and_source(mesh_a, ("cp",)) == (group_a, 2)
    assert broadcasting._resolve_group_and_source(mesh_b, ("cp",)) == (group_b, 6)


def test_broadcasting_dataloader_noop_iterates_source():
    real = [{"i": i} for i in range(4)]
    loader = BroadcastingDataLoader(source=real, device_mesh=None)
    assert list(loader) == real


def test_broadcasting_dataloader_noop_with_no_source_is_empty():
    loader = BroadcastingDataLoader(source=None, device_mesh=None)
    assert list(loader) == []


def test_broadcasting_dataloader_noop_state_dict_passthrough():
    class _Stateful:
        def state_dict(self):
            return {"cursor": 5}

        def load_state_dict(self, sd):
            self._restored = sd

        def __iter__(self):
            return iter([])

    src = _Stateful()
    loader = BroadcastingDataLoader(source=src, device_mesh=None)
    assert loader.state_dict() == {"cursor": 5}
    loader.load_state_dict({"cursor": 10})
    assert src._restored == {"cursor": 10}


def test_broadcasting_dataloader_state_dict_empty_when_source_lacks_method():
    loader = BroadcastingDataLoader(source=[1, 2, 3], device_mesh=None)
    assert loader.state_dict() == {}
    loader.load_state_dict({"anything": 1})  # must not raise


def test_broadcasting_dataloader_passes_through_len_when_available():
    loader = BroadcastingDataLoader(source=[1, 2, 3, 4, 5], device_mesh=None)
    assert len(loader) == 5


def test_broadcasting_dataloader_len_raises_when_source_has_no_len():
    class _NoLen:
        def __iter__(self):
            return iter([])

    loader = BroadcastingDataLoader(source=_NoLen(), device_mesh=None)
    with pytest.raises(TypeError):
        len(loader)


# ---------------------------------------------------------------------------
# Distributed (gloo) end-to-end tests for the broadcast contract.
# ---------------------------------------------------------------------------


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _build_cp_mesh(world_size: int):
    return torch.distributed.device_mesh.init_device_mesh(
        device_type="cpu",
        mesh_shape=(world_size,),
        mesh_dim_names=("cp",),
    )


def _broadcast_batch_worker(rank: int, world_size: int, port: int, queue: mp.Queue) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = _build_cp_mesh(world_size)
        if is_dp_source_rank(mesh):
            payload: Any = {"tensor": torch.arange(8), "name": "hello"}
        else:
            payload = None
        result = broadcast_batch(payload, mesh)
        if isinstance(result, dict):
            queue.put(("ok", result["tensor"].tolist(), result["name"]))
        else:
            queue.put(("ok", None, None))
    except Exception as e:
        queue.put(("err", repr(e), None))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _broadcasting_loader_worker(rank: int, world_size: int, port: int, queue: mp.Queue) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = _build_cp_mesh(world_size)
        source = [{"i": i} for i in range(3)] if is_dp_source_rank(mesh) else None
        loader = BroadcastingDataLoader(source=source, device_mesh=mesh)
        received = [batch["i"] for batch in loader]
        queue.put(("ok", received))
    except Exception as e:
        queue.put(("err", repr(e)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class _RaisesAfterOneBatch:
    def __iter__(self):
        yield {"i": 0}
        raise ValueError("source iterator exploded")


class _UnpicklableBatch:
    def __reduce__(self):
        raise TypeError("batch is intentionally unpicklable")


def _source_iterator_error_worker(rank: int, world_size: int, port: int, queue: mp.Queue) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = _build_cp_mesh(world_size)
        source = _RaisesAfterOneBatch() if is_dp_source_rank(mesh) else None
        loader = BroadcastingDataLoader(source=source, device_mesh=mesh)
        received = []
        try:
            received.extend(batch["i"] for batch in loader)
        except Exception as error:
            queue.put(("err", rank, received, type(error).__name__, str(error)))
        else:
            queue.put(("unexpected-success", rank, received, None, None))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _serialization_error_worker(rank: int, world_size: int, port: int, queue: mp.Queue) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = _build_cp_mesh(world_size)
        source = [_UnpicklableBatch()] if is_dp_source_rank(mesh) else None
        loader = BroadcastingDataLoader(source=source, device_mesh=mesh)
        try:
            list(loader)
        except Exception as error:
            queue.put(("err", rank, type(error).__name__, str(error)))
        else:
            queue.put(("unexpected-success", rank, None, None))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _early_break_worker(rank: int, world_size: int, port: int, queue: mp.Queue) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = _build_cp_mesh(world_size)
        source = [{"i": i} for i in range(3)] if is_dp_source_rank(mesh) else None
        loader = BroadcastingDataLoader(source=source, device_mesh=mesh)
        received = []
        for batch in loader:
            received.append(batch["i"])
            break
        queue.put(("ok", rank, received))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_workers(target, world_size: int, timeout: int = 120) -> list:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _get_free_port()
    procs = [ctx.Process(target=target, args=(rank, world_size, port, queue)) for rank in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=timeout)
    results = []
    while not queue.empty():
        results.append(queue.get())
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    return results


def test_broadcast_batch_dispatches_payload_across_ranks():
    results = _spawn_workers(_broadcast_batch_worker, world_size=2)
    assert len(results) == 2, results
    for status, tensor_list, name in results:
        assert status == "ok", results
        assert tensor_list == list(range(8))
        assert name == "hello"


def test_broadcasting_dataloader_iterates_in_lockstep_across_ranks():
    results = _spawn_workers(_broadcasting_loader_worker, world_size=2)
    assert len(results) == 2, results
    for status, received in results:
        assert status == "ok", results
        assert received == [0, 1, 2]


def test_broadcasting_dataloader_propagates_source_iterator_error():
    results = _spawn_workers(_source_iterator_error_worker, world_size=2, timeout=15)
    assert len(results) == 2, results
    results_by_rank = {result[1]: result for result in results}
    assert results_by_rank[0][:4] == ("err", 0, [0], "ValueError")
    assert "source iterator exploded" in results_by_rank[0][4]
    assert results_by_rank[1][:4] == ("err", 1, [0], "RuntimeError")
    assert "source iterator failed" in results_by_rank[1][4]
    assert "source iterator exploded" in results_by_rank[1][4]


def test_broadcasting_dataloader_propagates_batch_serialization_error():
    results = _spawn_workers(_serialization_error_worker, world_size=2, timeout=15)
    assert len(results) == 2, results
    results_by_rank = {result[1]: result for result in results}
    assert results_by_rank[0][:3] == ("err", 0, "TypeError")
    assert "batch is intentionally unpicklable" in results_by_rank[0][3]
    assert results_by_rank[1][:3] == ("err", 1, "RuntimeError")
    assert "batch serialization failed" in results_by_rank[1][3]
    assert "batch is intentionally unpicklable" in results_by_rank[1][3]


def test_broadcasting_dataloader_allows_synchronized_early_break():
    results = _spawn_workers(_early_break_worker, world_size=2, timeout=15)
    assert len(results) == 2, results
    for status, _, received in results:
        assert status == "ok", results
        assert received == [0]
