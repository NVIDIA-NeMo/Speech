# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import torch
from torch import Tensor


class CastCache:
    """Per-slot storage for a Cache-Aware `cache_last_channel` / `cache_last_time` tensor,
    held at a chosen float precision.

    The cache is stored as the underlying [L, B, T, D] tensor cast to `storage_dtype`; the
    slot dimension is dim 1. Three float precisions are supported:

    - `torch.float32`: an identity cast, i.e. full precision (no compression).
    - `torch.float16` / `torch.bfloat16`: 2x compression versus fp32.

    `source_dtype` is the dtype `gather_slots` returns (the precision the encoder consumes).
    It defaults to the seed tensor's dtype but can be set explicitly so the cache round-trips
    at the model's compute precision (e.g. bf16) instead of the fp32 seed, which makes a bf16
    cast cache an effectively lossless, half-memory no-op round trip.
    """

    def __init__(self, tensor: Tensor, storage_dtype: torch.dtype, source_dtype: torch.dtype | None = None):
        """
        Args:
            tensor (Tensor): initial cache tensor of shape [L, B, T, D].
            storage_dtype (torch.dtype): dtype used for backing storage (e.g. torch.bfloat16).
            source_dtype (torch.dtype | None): dtype returned by `gather_slots`. Defaults to
                `tensor.dtype`.
        """
        self._source_dtype = source_dtype if source_dtype is not None else tensor.dtype
        self._data = tensor.to(storage_dtype)
        self._gather_buf: Tensor | None = None

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        source_dtype: torch.dtype,
        storage_dtype: torch.dtype,
        device: torch.device,
    ) -> "CastCache":
        """Construct a zero-valued cast cache without materialising the full-precision tensor."""
        obj = cls.__new__(cls)
        obj._source_dtype = source_dtype
        obj._data = torch.zeros(shape, dtype=storage_dtype, device=device)
        obj._gather_buf = None
        return obj

    def _storage_gather_buf(self, shape: torch.Size) -> Tensor:
        # Reusable storage-dtype scratch for index_select, sized to the current gather.
        if self._gather_buf is None or self._gather_buf.shape != shape:
            self._gather_buf = torch.empty(shape, dtype=self._data.dtype, device=self._data.device)
        return self._gather_buf

    @property
    def device(self) -> torch.device:
        """Device on which the underlying storage lives."""
        return self._data.device

    def reset_slots(self, slot_ids: Tensor) -> None:
        """
        Zero out the given slots along the slot dimension (dim 1), in place.
        Args:
            slot_ids (Tensor): 1-D long tensor of slot indices to zero.
        """
        self._data.index_fill_(1, slot_ids, 0)

    def update_slots(self, dst_slot_ids: Tensor, src: Tensor, src_slot_ids: Tensor) -> None:
        """
        Copy `src[:, src_slot_ids]` into `self[:, dst_slot_ids]` along the slot dimension, in place.
        Args:
            dst_slot_ids (Tensor): 1-D long tensor of destination slot indices.
            src (Tensor): raw float tensor of shape [L, B, T, D] supplying new values.
            src_slot_ids (Tensor): 1-D long tensor of source slot indices into `src`.
        """
        src_slice = src.index_select(1, src_slot_ids).to(self._data.dtype)
        self._data.index_copy_(1, dst_slot_ids, src_slice)

    def gather_slots(self, slot_ids: list[int], out: Tensor | None = None) -> Tensor:
        """
        Return the float tensor for the requested slots, in the source dtype.
        Args:
            slot_ids (list[int]): slot indices to gather.
            out (Tensor | None): optional preallocated buffer of shape [L, len(slot_ids), T, D]
                in the source dtype. When provided with a matching shape it is written in place
                to avoid a per-step allocation on the streaming hot path. Callers MUST use the
                returned tensor rather than assuming `out` was written.
        Returns:
            Tensor of shape [L, len(slot_ids), T, D] in the source dtype.
        """
        if out is None:
            return self._data[:, slot_ids, :, :].to(self._source_dtype)
        idx = torch.as_tensor(slot_ids, device=self._data.device, dtype=torch.long)
        buf = self._storage_gather_buf(out.shape)
        torch.index_select(self._data, 1, idx, out=buf)
        out.copy_(buf)  # casts storage_dtype -> source_dtype in place
        return out

    def storage_nbytes(self) -> int:
        """Total bytes occupied by the underlying storage tensor (excludes Python overhead)."""
        return self._data.element_size() * self._data.numel()
