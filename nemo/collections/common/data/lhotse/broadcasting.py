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

Each iteration broadcasts one framed message containing a data batch, a stop
signal, or details about a source-side error. This works regardless of
whether the source loader exposes ``__len__`` (Lhotse training loaders
typically don't) and prevents receivers from waiting indefinitely when the
source loader fails.
"""
from __future__ import annotations

import pickle
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
    packet = (_PACKET_DATA, batch) if dist.get_rank() == src else None
    return _packet_payload(_broadcast_packet(packet, group, src))


class BroadcastingDataLoader:
    """Thin wrapper around (real DataLoader | None) that broadcasts each
    batch from the DP source rank to non-source ranks in the ``(cp, tp)``
    sub-mesh.

    Pass ``source=real_loader`` on the DP source rank (``cp_rank == 0`` and
    ``tp_rank == 0``); pass ``source=None`` on every other rank. Iteration
    issues one framed broadcast per step on every rank. The frame contains a
    data batch, clean-exhaustion signal, or source-side error. This lets all
    ranks finish or fail in lockstep regardless of whether the source exposes
    ``__len__``.

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
        self._group_and_source = None
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
        if not (dist.is_available() and dist.is_initialized()):
            if self._source is None:
                return
            yield from self._source
            return

        if self._group_and_source is None:
            self._group_and_source = _resolve_group_and_source(self._mesh, self._axes)
        if self._group_and_source is None:
            if self._source is None:
                return
            yield from self._source
            return

        group, src = self._group_and_source
        if self._is_source:
            source_iterator = iter(self._source)
            while True:
                try:
                    batch = next(source_iterator)
                except StopIteration:
                    _broadcast_packet((_PACKET_STOP, None), group, src)
                    return
                except Exception as error:
                    _broadcast_packet(_error_packet(error, "source iterator failed"), group, src)
                    raise

                _broadcast_packet((_PACKET_DATA, batch), group, src)
                yield batch
        else:
            while True:
                packet = _broadcast_packet(None, group, src)
                if packet[0] == _PACKET_STOP:
                    return
                yield _packet_payload(packet)

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


_PACKET_DATA = "data"
_PACKET_STOP = "stop"
_PACKET_ERROR = "error"


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

    if len(present) == 1:
        sub = device_mesh[present[0]]
    else:
        sub = device_mesh[present]._flatten(mesh_dim_name="_".join(present))

    group = sub.get_group()
    source_global_rank = int(sub.mesh.flatten()[0].item())
    return group, source_global_rank


def _broadcast_packet(packet, group, src: int):
    """Broadcast one pre-serialized protocol packet over ``group``.

    Serializing before the first tensor collective lets the source replace an
    unpicklable data packet with a small error packet while receivers are still
    waiting for the packet size. Failures after a collective begins must be
    handled by the process-group timeout and distributed job teardown.
    """
    is_source = dist.get_rank() == src
    source_error = None
    if is_source:
        try:
            serialized = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as error:
            source_error = error
            packet = _error_packet(error, "batch serialization failed")
            serialized = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        serialized = b""

    device = _broadcast_device(group)
    serialized_size = torch.tensor([len(serialized)], dtype=torch.long, device=device)
    dist.broadcast(serialized_size, src=src, group=group)

    if is_source:
        serialized_buffer = bytearray(serialized)
        serialized_tensor = torch.frombuffer(serialized_buffer, dtype=torch.uint8).to(device)
    else:
        serialized_tensor = torch.empty(int(serialized_size.item()), dtype=torch.uint8, device=device)
    dist.broadcast(serialized_tensor, src=src, group=group)

    if source_error is not None:
        raise source_error
    if not is_source:
        packet = pickle.loads(serialized_tensor.cpu().numpy().tobytes())
    _validate_packet(packet)
    return packet


def _packet_payload(packet):
    kind, payload = packet
    if kind == _PACKET_ERROR:
        raise RuntimeError(f"BroadcastingDataLoader source error: {payload}")
    if kind != _PACKET_DATA:
        raise RuntimeError(f"Expected a data packet, received {kind!r}")
    return payload


def _error_packet(error: Exception, context: str):
    try:
        error_text = f"{type(error).__module__}.{type(error).__qualname__}: {error}"
    except Exception:
        error_text = f"{type(error).__module__}.{type(error).__qualname__}"
    return _PACKET_ERROR, f"{context}: {error_text}"


def _validate_packet(packet) -> None:
    if not isinstance(packet, tuple) or len(packet) != 2:
        raise RuntimeError(f"Received an invalid broadcast packet: {type(packet).__name__}")
    if packet[0] not in (_PACKET_DATA, _PACKET_STOP, _PACKET_ERROR):
        raise RuntimeError(f"Received an unknown broadcast packet kind: {packet[0]!r}")
