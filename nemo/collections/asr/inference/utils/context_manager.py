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


from queue import Queue
from typing import Any

import torch
from torch import Tensor


class CacheAwareContext:
    """
    Stores the cache state for the Cache-Aware models.
    """

    def __init__(
        self,
        cache_last_channel: Tensor | None = None,
        cache_last_time: Tensor | None = None,
        cache_last_channel_len: Tensor | None = None,
    ):
        """
        Args:
            cache_last_channel (Tensor | None): Last channel of the cache.
            cache_last_time (Tensor | None): Last time of the cache.
            cache_last_channel_len (Tensor | None): Last channel length of the cache.
        """
        self.cache_last_channel = cache_last_channel
        self.cache_last_time = cache_last_time
        self.cache_last_channel_len = cache_last_channel_len


class CacheAwareContextManager:
    """
    Manager class to manipulate the cached states for the Cache-Aware models.
    """

    def __init__(
        self,
        cache_aware_model: Any,
        num_slots: int,
        use_cache: bool = True,
    ):
        """
        Initialize the CacheAwareContextManager.
        Args:
            cache_aware_model (Any): Cache-Aware model object. It should have the get_initial_cache_state method.
            num_slots (int): Number of slots to use for the cache. It should be greater than or equal to the batch size.
            use_cache (bool): Whether to use the cache. Default is True. If False, the cache is disabled.
        """
        self.cache_aware_model = cache_aware_model
        # Cache aware model should have the following methods:
        if not hasattr(self.cache_aware_model, "get_initial_cache_state"):
            raise ValueError("Cache aware model should have the get_initial_cache_state method")

        self.num_slots = num_slots
        self.cache_disabled = not use_cache
        self.cache_last_channel = None
        self.cache_last_time = None
        self.cache_last_channel_len = None
        self.reset()

    def reset(self) -> None:
        """Resets the context manager"""
        if self.cache_disabled:
            return

        self.streamidx2slotidx = {}
        self.slotidx2streamidx = {}
        self.free_slots = Queue(self.num_slots)
        for i in range(self.num_slots):
            self.free_slots.put(i)
        (
            self.cache_last_channel,  # [17, B, 70, 512]
            self.cache_last_time,  # [17, B, 512, 8]
            self.cache_last_channel_len,  # B
        ) = self.cache_aware_model.get_initial_cache_state(self.num_slots)
        self.device = self.cache_last_channel.device

    def _reset_slots(self, slot_ids: list[int]) -> None:
        """
        Resets the slots for the given slot_ids
        Args:
            slot_ids: list of slot indices to reset
        """
        if self.cache_disabled:
            return

        slot_ids_tensor = torch.tensor(slot_ids, device=self.device, dtype=torch.long)
        self.cache_last_channel.index_fill_(1, slot_ids_tensor, 0.0)
        self.cache_last_time.index_fill_(1, slot_ids_tensor, 0.0)
        self.cache_last_channel_len.index_fill_(0, slot_ids_tensor, 0)

        # free the slot, so that it can be used by other streams
        # remove the stream from the mappings
        for slot_id in slot_ids:
            self.free_slots.put(slot_id)
            stream_id = self.slotidx2streamidx[slot_id]
            del self.slotidx2streamidx[slot_id]
            del self.streamidx2slotidx[stream_id]

    def update_cache(self, stream_ids: list[int], new_context: CacheAwareContext, mapping: dict) -> None:
        """
        Updates the cache for the given stream_ids with the new_context
        Args:
            stream_ids (list[int]): list of stream ids
            new_context (CacheAwareContext): new context to update corresponding to the stream_ids
            mapping (dict): mapping between the old and new slots
        """
        if self.cache_disabled:
            return

        slot_ids_list = [self.streamidx2slotidx[sid] for sid in stream_ids]
        slot_ids = torch.tensor(slot_ids_list, device=self.device, dtype=torch.long)
        tgt_slot_ids = torch.tensor(
            [mapping[sid] for sid in slot_ids_list],
            device=self.device,
            dtype=torch.long,
        )

        # In-place copy along batch/slot dimension
        self.cache_last_channel.index_copy_(1, slot_ids, new_context.cache_last_channel.index_select(1, tgt_slot_ids))
        self.cache_last_time.index_copy_(1, slot_ids, new_context.cache_last_time.index_select(1, tgt_slot_ids))
        self.cache_last_channel_len.index_copy_(
            0, slot_ids, new_context.cache_last_channel_len.index_select(0, tgt_slot_ids)
        )

    def reset_slots(self, stream_ids: list[int], eos_flags: list[bool]) -> None:
        """
        Resets the slots for the finished streams
        Args:
            stream_ids (list[int]): list of stream ids
            eos_flags (list[bool]): list of eos flags indicating whether the stream has finished
        """
        if self.cache_disabled:
            return

        if len(stream_ids) != len(eos_flags):
            raise ValueError("stream_ids and eos_flags must have the same length")

        if len(stream_ids) == 0:
            return

        # reset the slots for finished streams
        self._reset_slots([self.streamidx2slotidx[sid] for sid, eos in zip(stream_ids, eos_flags) if eos])

    def get_context(self, stream_ids: list[int]) -> tuple[CacheAwareContext, dict]:
        """
        Retrieves the context from the cache for the given stream_ids
        Args:
            stream_ids (list[int]): list of stream ids
        Returns:
            context (CacheAwareContext): context for the given stream_ids
            mapping (dict): mapping between the cache and retrieved context
        """

        if len(stream_ids) == 0 or self.cache_disabled:
            # Create a dummy context with None values
            return CacheAwareContext(), {}

        # if the stream_id is new, we need to assign a slot to it
        for stream_id in stream_ids:
            if stream_id not in self.streamidx2slotidx:
                if self.free_slots.empty():
                    raise RuntimeError("No free slots available")
                slot_idx = self.free_slots.get()
                self.streamidx2slotidx[stream_id] = slot_idx
                self.slotidx2streamidx[slot_idx] = stream_id

        # get the cache for the particular stream_ids
        slot_ids = [self.streamidx2slotidx[stream_id] for stream_id in stream_ids]
        cache_last_channel = self.cache_last_channel[:, slot_ids, :, :]
        cache_last_time = self.cache_last_time[:, slot_ids, :, :]
        cache_last_channel_len = self.cache_last_channel_len[slot_ids]

        # create a context object
        context = CacheAwareContext(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

        # mapping between cache and context
        mapping = dict(zip(slot_ids, range(len(slot_ids))))
        return context, mapping
