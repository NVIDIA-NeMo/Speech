# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import torch
from nemo.utils import logging

from nemo.collections.speechlm2.inference.model_wrappers.decode_state import (
    InferenceStepResult,
    StreamingDecodeState,
)


class S2SContextManager:

    def __init__(
        self,
        s2s_model,
        num_slots: int,
        max_len: int,
    ):
        self.s2s_model = s2s_model
        self.num_slots = num_slots
        
        self.max_len = max_len
        self.device = getattr(self.s2s_model, "device", torch.device("cpu"))
        self.dtype = getattr(self.s2s_model, "dtype", torch.float32)

        self.reset()

    def reset(self) -> None:
        """Reset all bookkeeping for a new streaming session."""
        self.streamidx2slotidx: dict[int, int] = {}
        self.slotidx2streamidx: dict[int, int] = {}
        self.free_slots = Queue(self.num_slots)
        for i in range(self.num_slots):
            self.free_slots.put(i)
        self.slot_contexts: list[StreamingDecodeState | None] = [None] * self.num_slots

    def _create_context(self) -> StreamingDecodeState:
        """Allocate a fresh context backed by the realtime inference model."""
        if not hasattr(self.s2s_model, "create_decode_state"):
            raise RuntimeError("s2s_model must provide create_decode_state(max_len)")
        return self.s2s_model.create_decode_state(self.max_len)

    def _ensure_slot(self, stream_id: int) -> int:
        if stream_id not in self.streamidx2slotidx:
            if self.free_slots.empty():
                # Emergency cleanup: force-release all slots for a fresh start
                # This handles cases where previous streams didn't end properly
                # (e.g., exceptions, client disconnects, missing is_last=True)
                logging.warning(f"No free slots available - forcing cleanup of all {self.num_slots} slots")
                orphaned_streams = list(self.slotidx2streamidx.values())
                if orphaned_streams:
                    logging.warning(f"Orphaned streams being cleaned up: {orphaned_streams}")
                for slot_idx in range(self.num_slots):
                    self.reset_slot(slot_idx)
            slot_idx = self.free_slots.get()
            # Ensure the slot is completely clean before assigning to new stream
            if self.slot_contexts[slot_idx] is not None:
                logging.warning(f"Slot {slot_idx} was not properly cleaned. Forcing cleanup.")
                self.slot_contexts[slot_idx] = None
            self.streamidx2slotidx[stream_id] = slot_idx
            self.slotidx2streamidx[slot_idx] = stream_id
        return self.streamidx2slotidx[stream_id]

    def reset_slot(self, slot_idx: int) -> None:
        """Release a slot back to the pool."""
        if slot_idx < 0 or slot_idx >= self.num_slots:
            return
        # Set to None to break reference and allow garbage collection
        self.slot_contexts[slot_idx] = None
        stream_id = self.slotidx2streamidx.get(slot_idx)
        if stream_id is not None:
            del self.slotidx2streamidx[slot_idx]
            del self.streamidx2slotidx[stream_id]
        self.free_slots.put(slot_idx)

    def update_context(
        self,
        stream_ids: list[int],
        step_result: InferenceStepResult,
        num_frames: int,
    ) -> None:
        """Advance frame counter and set subword mask after an inference step.

        All cache and tensor mutations (dynamic_cache, past_key_values, code,
        perception_cache, codec_cache, gen_text, gen_asr_text, etc.) are
        already applied in-place on the ``StreamingDecodeState`` by
        ``infer_one_step``.  This method only bumps ``frame_idx`` and marks
        the subword mask for the newly generated frames.
        """
        if len(stream_ids) == 0:
            return
        if len(stream_ids) != 1:
            raise NotImplementedError("update_context currently supports batch_size == 1")

        stream_id = stream_ids[0]
        slot_idx = self.streamidx2slotidx.get(stream_id)
        if slot_idx is None:
            raise RuntimeError(f"Stream {stream_id} is not registered in the context manager")

        context = self.slot_contexts[slot_idx]
        if context is None:
            context = self._create_context()
            self.slot_contexts[slot_idx] = context

        start_idx = context.frame_idx
        end_idx = start_idx + num_frames
        if end_idx > context.gen_text.shape[1]:
            raise RuntimeError(
                "Context maximum length exceeded. Consider increasing `streaming.max_len` in the configuration."
            )

        context.frame_idx = end_idx

        if context.subword_mask is not None:
            context.subword_mask[:, start_idx:end_idx] = True

    def reset_slots(self, stream_ids: list[int], eos_flags: list[bool]) -> None:
        """Release contexts for streams that signalled end-of-stream."""
        if len(stream_ids) != len(eos_flags):
            raise ValueError("stream_ids and eos_flags must have the same length")
        for stream_id, eos_flag in zip(stream_ids, eos_flags):
            if eos_flag and stream_id in self.streamidx2slotidx:
                self.reset_slot(self.streamidx2slotidx[stream_id])

    def get_context(self, stream_ids: list[int]) -> tuple[StreamingDecodeState, dict[int, int]]:
        """Return the cached context associated with the provided stream ids."""
        if len(stream_ids) == 0:
            return self._create_context(), {}
        if len(stream_ids) != 1:
            raise NotImplementedError("get_context currently supports batch_size == 1")

        stream_id = stream_ids[0]
        slot_idx = self._ensure_slot(stream_id)

        if self.slot_contexts[slot_idx] is None:
            self.slot_contexts[slot_idx] = self._create_context()

        return self.slot_contexts[slot_idx], {slot_idx: 0}
