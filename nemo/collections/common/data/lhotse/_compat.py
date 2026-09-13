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
# pylint: disable=unused-import
"""Compatibility shims for optional Lhotse indexed/resumable dataloading APIs.

This module lets NeMo import with released Lhotse versions that do not expose
those APIs yet, while delegating to the real implementations when a resumable
Lhotse checkout is available.
"""
import os
from collections.abc import Generator, Iterable
from typing import Any

import torch
from torch import distributed as dist

__all__ = [
    "GraphOriginDict",
    "IteratorNode",
    "LazyIndexedManifestIterator",
    "PartitionedIndexedIterator",
    "attach_graph_origin",
    "normalize_graph_token",
]

try:
    from lhotse.dataset import dataloading as _lhotse_dataloading

    PartitionedIndexedIterator = _lhotse_dataloading.PartitionedIndexedIterator
except (ImportError, AttributeError):
    LHOTSE_USE_WORKER_PARTITION = "LHOTSE_USE_WORKER_PARTITION"

    def _get_world_size() -> int:
        if "WORLD_SIZE" in os.environ:
            return int(os.environ["WORLD_SIZE"])
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    def _get_rank() -> int:
        if "RANK" in os.environ:
            return int(os.environ["RANK"])
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _get_worker_partition() -> tuple[int, int]:
        if os.environ.get(LHOTSE_USE_WORKER_PARTITION) != "1":
            return 0, 1
        rank = _get_rank()
        world_size = _get_world_size()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id = worker_info.id
            num_workers = max(worker_info.num_workers, 1)
        return rank * num_workers + worker_id, world_size * num_workers

    class PartitionedIndexedIterator:
        def __init__(self, shuffle: bool = False, seed: int = 0) -> None:
            self._shuffle = shuffle
            self._seed = seed
            self._position = 0
            self._shard_id: int | None = None
            self._num_shards: int | None = None
            self._restored = False
            self._range = None
            self._pending_range_state = None

        @property
        def position(self) -> int:
            return self._position

        def iterate(self, total_len: int) -> Generator[int, None, None]:
            shard_id, num_shards = _get_worker_partition()

            if self._restored:
                self._restored = False
                if self._num_shards is not None and (self._shard_id != shard_id or self._num_shards != num_shards):
                    raise ValueError(
                        f"PartitionedIndexedIterator topology mismatch on resume: "
                        f"saved (shard_id={self._shard_id}, num_shards={self._num_shards}), "
                        f"current (shard_id={shard_id}, num_shards={num_shards})."
                    )
                start = self._position
            else:
                start = 0
                self._position = 0

            self._shard_id = shard_id
            self._num_shards = num_shards

            if self._shuffle:
                from lhotse.indexing import LazyShuffledRange

                self._range = LazyShuffledRange(total_len, seed=self._seed, shard_id=shard_id, num_shards=num_shards)
                if self._pending_range_state is not None:
                    self._range.load_state_dict(self._pending_range_state)
                    self._pending_range_state = None
                shard_len = len(self._range)
            else:
                self._range = None
                shard_len = (total_len - shard_id + num_shards - 1) // num_shards if total_len > shard_id else 0

            for i in range(start, shard_len):
                self._position = i + 1
                yield self._range[i] if self._range is not None else shard_id + i * num_shards

        def state_dict(self) -> dict:
            sd = {
                "position": self._position,
                "shard_id": self._shard_id,
                "num_shards": self._num_shards,
            }
            if self._range is not None:
                sd["range"] = self._range.state_dict()
            elif self._pending_range_state is not None:
                sd["range"] = self._pending_range_state
            return sd

        def load_state_dict(self, sd: dict) -> None:
            self._position = sd.get("position", 0)
            self._shard_id = sd.get("shard_id")
            self._num_shards = sd.get("num_shards")
            if self._shuffle:
                self._pending_range_state = sd.get("range")
                self._range = None
            self._restored = True


try:
    from lhotse import lazy as _lhotse_lazy

    GraphOriginDict = _lhotse_lazy.GraphOriginDict
    IteratorNode = _lhotse_lazy.IteratorNode
    LazyIndexedManifestIterator = _lhotse_lazy.LazyIndexedManifestIterator
    attach_graph_origin = _lhotse_lazy.attach_graph_origin
    normalize_graph_token = _lhotse_lazy.normalize_graph_token
except (ImportError, AttributeError):

    class IteratorNode(Iterable):
        is_checkpointable = False
        is_indexed = False
        has_constant_time_access = False

        def state_dict(self) -> dict:
            raise NotImplementedError(f"{type(self).__name__} is not checkpointable.")

        def load_state_dict(self, sd: dict) -> None:
            raise NotImplementedError(f"{type(self).__name__} is not checkpointable.")

        def iter_children(self):
            if hasattr(self, "source"):
                yield getattr(self, "source")
            if hasattr(self, "sources"):
                yield from getattr(self, "sources")

    class GraphOriginDict(dict):
        __slots__ = ("_graph_origin",)

    def normalize_graph_token(token: Any) -> Any:
        if isinstance(token, list):
            return tuple(normalize_graph_token(part) for part in token)
        if isinstance(token, tuple):
            return tuple(normalize_graph_token(part) for part in token)
        return token

    def attach_graph_origin(item: Any, token: Any) -> Any:
        try:
            object.__setattr__(item, "_graph_origin", token)
        except Exception:
            try:
                setattr(item, "_graph_origin", token)
            except Exception:
                # Immutable extension objects may not accept ad-hoc metadata.
                return item
        return item

    class LazyIndexedManifestIterator(IteratorNode):
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "LazyIndexedManifestIterator requires a Lhotse version with indexed/resumable dataloading support."
            )
