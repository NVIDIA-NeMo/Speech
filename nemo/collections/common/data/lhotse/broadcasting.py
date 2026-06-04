# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""CP/TP-aware data loading.

Under context-parallel (CP) and tensor-parallel (TP) training, all ranks in
the same ``(cp, tp)`` sub-mesh of a DP slot must process the **same** global
batch each step — CP shards the sequence dimension and TP shards the
feature dimension, so a divergent global batch breaks the per-rank shape
contract that CP/TP collectives assume.

The fix: construct the dataloader on a single DP-source rank per slot and
broadcast each batch over NCCL to the other ranks in the ``(cp, tp)``
sub-mesh, eliminating the entire class of nondeterminism bug regardless of
source (Lhotse ``concurrent_bucketing``, ``shard_seed: randomized``, worker
scheduling jitter, etc.).

:class:`BroadcastingDataLoader` is the single-class API:

    # In the datamodule:
    return BroadcastingDataLoader(
        source=real_loader if is_dp_source_rank(mesh) else None,
        device_mesh=mesh,
    )

The wrapper hides the broadcast bookkeeping. ``state_dict`` /
``load_state_dict`` are delegated to the source loader on the source rank,
so checkpoint/resume works transparently with ``DataLoader``,
``torchdata.StatefulDataLoader``, or any other source object that
implements those methods.

Iteration termination is handled with two broadcasts per step: a
continue/stop boolean followed by the batch. This works regardless of
whether the source loader exposes ``__len__`` (Lhotse training loaders
typically don't).
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_dp_source_rank(
    device_mesh,
    axes: tuple[str, ...] = ("cp", "tp"),
) -> bool:
    """True iff this rank is the data-parallel source for its DP slot.

    A DP source rank has coordinate 0 along every named axis (e.g. ``cp_rank == 0``
    and ``tp_rank == 0``). Pass the real dataloader to
    :class:`BroadcastingDataLoader` only on DP source ranks; pass ``None``
    on the others.

    Returns True unconditionally when ``device_mesh`` is None or every named
    axis present in the mesh has size 1, so callers can short-circuit setup
    logic on single-rank-per-DP-slot runs without a separate code path.
    """
    if _is_noop(device_mesh, axes):
        return True
    present = _present_axes(device_mesh, axes)
    return all(device_mesh[ax].get_local_rank() == 0 for ax in present)


def broadcast_batch(
    batch: Any,
    device_mesh,
    axes: tuple[str, ...] = ("cp", "tp"),
) -> Any:
    """Broadcast ``batch`` from the DP source rank to all ranks in the
    sub-mesh covering ``axes``. Returns the source's batch on every rank.

    Low-level primitive used internally by :class:`BroadcastingDataLoader`.
    Most callers should use the class wrapper rather than calling this
    directly.

    No-op (returns ``batch`` unchanged) when ``device_mesh`` is None, every
    present named axis has size 1, or distributed isn't initialized.
    """
    if _is_noop(device_mesh, axes):
        return batch
    if not (dist.is_available() and dist.is_initialized()):
        return batch
    resolved = _resolve_group_and_source(device_mesh, axes)
    if resolved is None:
        return batch
    group, src = resolved
    obj_list = [batch]
    dist.broadcast_object_list(obj_list, src=src, group=group, device=_broadcast_device(group))
    return obj_list[0]


class BroadcastingDataLoader:
    """Thin wrapper around (real DataLoader | None) that broadcasts each
    batch from the DP source rank to non-source ranks in the ``(cp, tp)``
    sub-mesh.

    Pass ``source=real_loader`` on the DP source rank (``cp_rank == 0`` and
    ``tp_rank == 0``); pass ``source=None`` on every other rank. Iteration
    issues two broadcasts per step on every rank: a continue/stop boolean
    followed by the batch. After the source loader is exhausted, the
    continue broadcast is False and iteration ends in lockstep on all
    ranks regardless of whether the source exposes ``__len__``.

    ``state_dict`` / ``load_state_dict`` are delegated to the source on the
    source rank (no-ops on non-source ranks), so checkpoint/resume keeps
    working transparently with ``torch.utils.data.DataLoader``,
    ``torchdata.StatefulDataLoader``, or any other source that implements
    those methods.

    No-op when ``device_mesh`` is None or every named axis present has
    size 1 — iteration delegates to the source loader unchanged.
    """

    def __init__(
        self,
        source: Iterable | None,
        device_mesh,
        axes: tuple[str, ...] = ("cp", "tp"),
    ):
        self._source = source
        self._mesh = device_mesh
        self._axes = axes
        if not _is_noop(device_mesh, axes):
            self._is_source = is_dp_source_rank(device_mesh, axes)
            if self._is_source and source is None:
                raise ValueError("BroadcastingDataLoader on a DP source rank requires a non-None source")

    def __iter__(self) -> Iterator[Any]:
        if _is_noop(self._mesh, self._axes):
            if self._source is None:
                return
            yield from self._source
            return
        if self._is_source:
            for batch in self._source:
                broadcast_batch(True, self._mesh, self._axes)
                broadcast_batch(batch, self._mesh, self._axes)
                yield batch
            broadcast_batch(False, self._mesh, self._axes)
        else:
            while True:
                keep_iterating = broadcast_batch(None, self._mesh, self._axes)
                if not keep_iterating:
                    return
                batch = broadcast_batch(None, self._mesh, self._axes)
                yield batch

    def __len__(self) -> int:
        # Pass-through when the source defines __len__; raise TypeError
        # otherwise (matching Lhotse's typical behavior, which Lightning
        # already handles by treating the loader as iterable-style).
        if self._source is not None:
            return len(self._source)
        raise TypeError("BroadcastingDataLoader on non-source rank has no defined length")

    def state_dict(self) -> dict:
        if self._source is not None and hasattr(self._source, "state_dict"):
            return self._source.state_dict()
        return {}

    def load_state_dict(self, state_dict) -> None:
        if self._source is not None and hasattr(self._source, "load_state_dict"):
            self._source.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


# Cache: (id(device_mesh), tuple_of_axes) -> (process_group, source_global_rank).
# Sub-mesh creation calls ``_flatten`` which materializes a process group;
# we don't want to repeat that per training step.
_GROUP_CACHE: dict[tuple[int, tuple[str, ...]], tuple[Any, int]] = {}


def _present_axes(device_mesh, axes: Sequence[str]) -> tuple[str, ...]:
    if device_mesh is None:
        return ()
    names = device_mesh.mesh_dim_names or ()
    return tuple(ax for ax in axes if ax in names)


def _is_noop(device_mesh, axes: Sequence[str]) -> bool:
    if device_mesh is None:
        return True
    present = _present_axes(device_mesh, axes)
    if not present:
        return True
    return all(device_mesh[ax].size() == 1 for ax in present)


def _broadcast_device(group) -> torch.device:
    backend = dist.get_backend(group)
    if backend == "nccl" and torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _resolve_group_and_source(device_mesh, axes: Sequence[str]):
    if _is_noop(device_mesh, axes):
        return None
    present = _present_axes(device_mesh, axes)
    cache_key = (id(device_mesh), present)
    cached = _GROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if len(present) == 1:
        sub = device_mesh[present[0]]
    else:
        sub = device_mesh[present]._flatten(mesh_dim_name="_".join(present))

    group = sub.get_group()
    source_global_rank = int(sub.mesh.flatten()[0].item())
    _GROUP_CACHE[cache_key] = (group, source_global_rank)
    return group, source_global_rank
