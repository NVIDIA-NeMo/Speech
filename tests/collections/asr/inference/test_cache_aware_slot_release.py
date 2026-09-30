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

"""CPU unit tests for cache-aware slot free on delete_state / session reset (issue #16309).

No checkpoint or HF download is required. Slot pools are exercised with num_slots=2.
"""

from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.inference.pipelines.cache_aware_ctc_pipeline import CacheAwareCTCPipeline
from nemo.collections.asr.inference.pipelines.cache_aware_rnnt_pipeline import CacheAwareRNNTPipeline
from nemo.collections.asr.inference.streaming.buffering.audio_bufferer import AudioBufferer
from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.utils.constants import LOG_MEL_ZERO
from nemo.collections.asr.inference.utils.context_manager import CacheAwareContextManager

PIPELINES = [
    pytest.param(CacheAwareRNNTPipeline, id="cache_aware_rnnt"),
    pytest.param(CacheAwareCTCPipeline, id="cache_aware_ctc"),
]

NUM_SLOTS = 2


class _StubCacheAwareModel:
    """Minimal stand-in for get_initial_cache_state used by CacheAwareContextManager."""

    def get_initial_cache_state(self, num_slots: int):
        # Shapes match the comments in CacheAwareContextManager.reset (batch on dim 1 / 0).
        cache_last_channel = torch.zeros(1, num_slots, 2, 4)
        cache_last_time = torch.zeros(1, num_slots, 4, 2)
        cache_last_channel_len = torch.zeros(num_slots, dtype=torch.long)
        return cache_last_channel, cache_last_time, cache_last_channel_len


def _make_bufferer(num_slots: int = NUM_SLOTS) -> BatchedCacheFeatureBufferer:
    """Build a BatchedCacheFeatureBufferer without constructing a real preprocessor."""
    bufferer = BatchedCacheFeatureBufferer.__new__(BatchedCacheFeatureBufferer)
    bufferer.num_slots = num_slots
    bufferer.sample_rate = 16000
    bufferer.buffer_size_in_secs = 0.16
    bufferer.chunk_size_in_secs = 0.16
    bufferer.device = torch.device("cpu")
    bufferer.ZERO_LEVEL_SPEC_DB_VAL = LOG_MEL_ZERO
    bufferer.n_feat = 4
    bufferer.feature_buffer_len = 2
    bufferer.feature_chunk_len = 2
    bufferer.audio_bufferers = [AudioBufferer(16000, 0.16) for _ in range(num_slots)]
    bufferer.feature_buffer = torch.full(
        [num_slots, bufferer.n_feat, bufferer.feature_buffer_len],
        LOG_MEL_ZERO,
        dtype=torch.float32,
    )
    bufferer.streamidx2slotidx = {}
    bufferer.slotidx2streamidx = {}
    bufferer.available_slots = Queue(num_slots)
    for i in range(num_slots):
        bufferer.available_slots.put(i)
    return bufferer


def _allocate_bufferer_slot(bufferer: BatchedCacheFeatureBufferer, stream_id: int) -> int:
    if bufferer.available_slots.empty():
        raise RuntimeError("No free slots available")
    slot_idx = bufferer.available_slots.get()
    bufferer.streamidx2slotidx[stream_id] = slot_idx
    bufferer.slotidx2streamidx[slot_idx] = stream_id
    return slot_idx


def _make_context_manager(num_slots: int = NUM_SLOTS) -> CacheAwareContextManager:
    return CacheAwareContextManager(_StubCacheAwareModel(), num_slots=num_slots, use_cache=True)


def _allocate_context_slot(context_manager: CacheAwareContextManager, stream_id: int) -> None:
    # Exercise the real allocation path used during get_context.
    context_manager.get_context([stream_id])


def _make_pipeline(pipeline_cls, num_slots: int = NUM_SLOTS):
    """Pipeline shell with only the members delete_state / reset_session / close_session need."""
    pipeline = pipeline_cls.__new__(pipeline_cls)
    BasePipeline.__init__(pipeline)
    pipeline.bufferer = _make_bufferer(num_slots)
    pipeline.context_manager = _make_context_manager(num_slots)
    # RNNT close_session checks decoding_computer before calling super().
    pipeline.decoding_computer = None
    return pipeline


def _take_slots(pipeline, stream_ids):
    for stream_id in stream_ids:
        _allocate_bufferer_slot(pipeline.bufferer, stream_id)
        _allocate_context_slot(pipeline.context_manager, stream_id)
        pipeline._state_pool[stream_id] = SimpleNamespace(stream_id=stream_id)


def _free_counts(pipeline):
    return pipeline.context_manager.free_slots.qsize(), pipeline.bufferer.available_slots.qsize()


@pytest.mark.unit
def test_bufferer_free_stream_returns_slot_and_is_idempotent():
    bufferer = _make_bufferer()
    _allocate_bufferer_slot(bufferer, 0)
    _allocate_bufferer_slot(bufferer, 1)
    assert bufferer.available_slots.qsize() == 0

    bufferer.free_stream(0)
    assert bufferer.available_slots.qsize() == 1
    assert 0 not in bufferer.streamidx2slotidx

    bufferer.free_stream(0)  # already freed via is_last-equivalent path
    assert bufferer.available_slots.qsize() == 1

    _allocate_bufferer_slot(bufferer, 2)
    assert bufferer.available_slots.qsize() == 0


@pytest.mark.unit
def test_bufferer_reset_restores_all_slots_after_leak():
    bufferer = _make_bufferer()
    for stream_id in range(NUM_SLOTS):
        _allocate_bufferer_slot(bufferer, stream_id)
    assert bufferer.available_slots.qsize() == 0

    bufferer.reset()
    assert bufferer.available_slots.qsize() == NUM_SLOTS
    assert bufferer.streamidx2slotidx == {}
    assert bufferer.slotidx2streamidx == {}
    _allocate_bufferer_slot(bufferer, 99)
    assert bufferer.available_slots.qsize() == NUM_SLOTS - 1


@pytest.mark.unit
def test_context_manager_free_stream_returns_slot_and_is_idempotent():
    context_manager = _make_context_manager()
    _allocate_context_slot(context_manager, 0)
    _allocate_context_slot(context_manager, 1)
    assert context_manager.free_slots.qsize() == 0

    context_manager.free_stream(0)
    assert context_manager.free_slots.qsize() == 1
    assert 0 not in context_manager.streamidx2slotidx

    context_manager.free_stream(0)  # double free must not grow the queue past num_slots
    assert context_manager.free_slots.qsize() == 1

    _allocate_context_slot(context_manager, 2)
    assert context_manager.free_slots.qsize() == 0


@pytest.mark.unit
def test_context_manager_reset_restores_all_slots_after_leak():
    context_manager = _make_context_manager()
    for stream_id in range(NUM_SLOTS):
        _allocate_context_slot(context_manager, stream_id)
    assert context_manager.free_slots.qsize() == 0

    context_manager.reset()
    assert context_manager.free_slots.qsize() == NUM_SLOTS
    assert context_manager.streamidx2slotidx == {}
    _allocate_context_slot(context_manager, 99)
    assert context_manager.free_slots.qsize() == NUM_SLOTS - 1


@pytest.mark.unit
def test_context_manager_is_last_path_still_frees_via_reset_slots():
    """The intentional is_last free path (reset_slots with eos=True) must keep working."""
    context_manager = _make_context_manager()
    _allocate_context_slot(context_manager, 7)
    assert context_manager.free_slots.qsize() == NUM_SLOTS - 1

    context_manager.reset_slots([7], [True])
    assert context_manager.free_slots.qsize() == NUM_SLOTS
    assert 7 not in context_manager.streamidx2slotidx

    # Idempotent free_stream after is_last must not over-fill the queue.
    context_manager.free_stream(7)
    assert context_manager.free_slots.qsize() == NUM_SLOTS


@pytest.mark.unit
@pytest.mark.parametrize("pipeline_cls", PIPELINES)
def test_delete_state_frees_slots_for_streams_ended_without_is_last(pipeline_cls):
    """Many short streams ending via delete_state must recover free slot counts (issue repro shape)."""
    pipeline = _make_pipeline(pipeline_cls)
    assert _free_counts(pipeline) == (NUM_SLOTS, NUM_SLOTS)

    # Exhaust the pool with streams that never send is_last, then free via delete_state.
    for stream_id in range(NUM_SLOTS):
        _take_slots(pipeline, [stream_id])
    assert _free_counts(pipeline) == (0, 0)

    with pytest.raises(RuntimeError, match="No free slots available"):
        _allocate_bufferer_slot(pipeline.bufferer, NUM_SLOTS)

    for stream_id in range(NUM_SLOTS):
        pipeline.delete_state(stream_id)

    assert pipeline._state_pool == {}
    assert _free_counts(pipeline) == (NUM_SLOTS, NUM_SLOTS)

    # Further streams must allocate again (the hard fail at num_slots+1 is gone).
    for stream_id in range(NUM_SLOTS, NUM_SLOTS * 3):
        _take_slots(pipeline, [stream_id])
        pipeline.delete_state(stream_id)
        assert _free_counts(pipeline) == (NUM_SLOTS, NUM_SLOTS)


@pytest.mark.unit
@pytest.mark.parametrize("pipeline_cls", PIPELINES)
def test_delete_state_after_is_last_free_is_idempotent(pipeline_cls):
    """Normal pipeline.run path: is_last frees first, then delete_state must not double-free."""
    pipeline = _make_pipeline(pipeline_cls)
    _take_slots(pipeline, [0])

    # Simulate the is_last free that update() / reset_slots() perform today.
    pipeline.bufferer.free_stream(0)
    pipeline.context_manager.free_stream(0)
    assert _free_counts(pipeline) == (NUM_SLOTS, NUM_SLOTS)

    pipeline.delete_state(0)
    assert pipeline._state_pool == {}
    assert _free_counts(pipeline) == (NUM_SLOTS, NUM_SLOTS)


@pytest.mark.unit
@pytest.mark.parametrize("pipeline_cls", PIPELINES)
def test_close_session_and_open_session_restore_leaked_slots(pipeline_cls):
    """close_session / open_session must restore capacity even when delete_state was never called."""
    pipeline = _make_pipeline(pipeline_cls)
    _take_slots(pipeline, list(range(NUM_SLOTS)))
    assert _free_counts(pipeline) == (0, 0)

    pipeline.close_session()
    assert pipeline._state_pool == {}
    assert _free_counts(pipeline) == (NUM_SLOTS, NUM_SLOTS)

    _take_slots(pipeline, list(range(NUM_SLOTS)))
    assert _free_counts(pipeline) == (0, 0)

    pipeline.open_session()
    assert pipeline._state_pool == {}
    assert _free_counts(pipeline) == (NUM_SLOTS, NUM_SLOTS)


@pytest.mark.unit
@pytest.mark.parametrize("pipeline_cls", PIPELINES)
def test_without_delete_state_override_slots_remain_leaked(pipeline_cls):
    """Control: BasePipeline.delete_state alone does not return slots (documents the bug)."""
    pipeline = _make_pipeline(pipeline_cls)
    _take_slots(pipeline, [0])
    BasePipeline.delete_state(pipeline, 0)
    assert 0 not in pipeline._state_pool
    assert _free_counts(pipeline) == (NUM_SLOTS - 1, NUM_SLOTS - 1)
