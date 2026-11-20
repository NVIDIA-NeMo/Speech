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

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.inference.pipelines.cache_aware_ctc_pipeline import CacheAwareCTCPipeline
from nemo.collections.asr.inference.pipelines.cache_aware_rnnt_pipeline import CacheAwareRNNTPipeline
from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.state.cache_aware_rnnt_state import CacheAwareRNNTStreamingState
from nemo.collections.asr.inference.utils.context_manager import CacheAwareContextManager
from nemo.collections.asr.inference.utils.per_stream_biasing import release_auto_managed_stream_biasing
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import BiasingRequestItemConfig

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


_PREPROCESSOR_CFG = OmegaConf.create(
    {
        "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
        "sample_rate": 16000,
        "window_size": 0.025,
        "window_stride": 0.01,
        "features": 8,
        "n_fft": 512,
        "log": True,
        "dither": 0.0,
        "pad_to": 0,
    }
)


def _make_bufferer(num_slots: int = NUM_SLOTS) -> BatchedCacheFeatureBufferer:
    """Build the real BatchedCacheFeatureBufferer on CPU with a small mel preprocessor."""
    return BatchedCacheFeatureBufferer(
        num_slots=num_slots,
        sample_rate=16000,
        buffer_size_in_secs=0.16,
        chunk_size_in_secs=0.16,
        preprocessor_cfg=_PREPROCESSOR_CFG,
        device=torch.device("cpu"),
    )


def _allocate_bufferer_slot(bufferer: BatchedCacheFeatureBufferer, stream_id: int) -> int:
    if bufferer.available_slots.empty():
        raise RuntimeError("No free slots available")
    slot_idx = bufferer.available_slots.get()
    bufferer.streamidx2slotidx[stream_id] = slot_idx
    bufferer.slotidx2streamidx[slot_idx] = stream_id
    return slot_idx


def _fill_bufferer(bufferer: BatchedCacheFeatureBufferer) -> None:
    """Take every slot through the real update() path with non-zero audio."""
    bufferer.update(
        [
            Frame(samples=torch.full((bufferer.chunk_size,), float(stream_id + 1)), stream_id=stream_id, is_first=True)
            for stream_id in range(bufferer.num_slots)
        ]
    )


def _assert_bufferer_clean(bufferer: BatchedCacheFeatureBufferer) -> None:
    assert bufferer.available_slots.qsize() == bufferer.num_slots
    assert bufferer.streamidx2slotidx == {}
    assert bufferer.slotidx2streamidx == {}
    # compare with the configured fill value, not 0, so log-mel fill semantics stay intact
    assert torch.all(bufferer.feature_buffer == bufferer.ZERO_LEVEL_SPEC_DB_VAL)
    for audio_bufferer in bufferer.audio_bufferers:
        assert torch.count_nonzero(audio_bufferer.sample_buffer) == 0
        assert audio_bufferer.left_padding == audio_bufferer.buffer_size


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
    """reset() restores capacity and buffer contents after streams leaked their slots through update()."""
    bufferer = _make_bufferer()
    _fill_bufferer(bufferer)
    bufferer.feature_buffer.fill_(1.0)
    assert bufferer.available_slots.qsize() == 0

    bufferer.reset()

    _assert_bufferer_clean(bufferer)
    bufferer.update([Frame(samples=torch.zeros(bufferer.chunk_size), stream_id=99, is_first=True)])
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
def test_context_manager_reset_zeros_in_place_without_realloc():
    """reset() must zero existing cache tensors and keep the same tensor storage."""
    context_manager = _make_context_manager()
    channel = context_manager.cache_last_channel
    time = context_manager.cache_last_time
    channel_len = context_manager.cache_last_channel_len
    channel_ptr = channel.data_ptr()
    time_ptr = time.data_ptr()
    len_ptr = channel_len.data_ptr()

    # Mutate so we can assert zeroing happened (initial state is already zeros).
    channel.fill_(1.0)
    time.fill_(2.0)
    channel_len.fill_(3)
    _allocate_context_slot(context_manager, 0)
    assert context_manager.free_slots.qsize() == NUM_SLOTS - 1
    assert not torch.all(channel == 0)

    context_manager.reset()

    assert context_manager.free_slots.qsize() == NUM_SLOTS
    assert context_manager.streamidx2slotidx == {}
    assert context_manager.slotidx2streamidx == {}
    assert torch.all(context_manager.cache_last_channel == 0)
    assert torch.all(context_manager.cache_last_time == 0)
    assert torch.all(context_manager.cache_last_channel_len == 0)
    # Same tensor objects / storage: no realloc via get_initial_cache_state.
    assert context_manager.cache_last_channel is channel
    assert context_manager.cache_last_time is time
    assert context_manager.cache_last_channel_len is channel_len
    assert context_manager.cache_last_channel.data_ptr() == channel_ptr
    assert context_manager.cache_last_time.data_ptr() == time_ptr
    assert context_manager.cache_last_channel_len.data_ptr() == len_ptr


@pytest.mark.unit
def test_teardown_outside_inference_mode_after_inference_mode_setup():
    """Resources built and dirtied inside inference mode can be reset and freed outside it."""
    with torch.inference_mode():
        bufferer = _make_bufferer()
        _fill_bufferer(bufferer)
        context_manager = _make_context_manager()
        _allocate_context_slot(context_manager, 0)
        _allocate_context_slot(context_manager, 1)
        context_manager.cache_last_channel[:, 0].fill_(1.0)
        context_manager.cache_last_channel[:, 1].fill_(2.0)
        context_manager.cache_last_time[:, 1].fill_(2.0)
        context_manager.cache_last_channel_len[1] = 2
    # make sure the test really exercises inference tensors
    assert bufferer.feature_buffer.is_inference()
    assert bufferer.audio_bufferers[0].sample_buffer.is_inference()
    assert context_manager.cache_last_channel.is_inference()
    channel = context_manager.cache_last_channel
    channel_ptr = channel.data_ptr()

    context_manager.free_stream(0)
    assert torch.all(channel[:, 0] == 0)
    # the live neighbor keeps its cache
    assert torch.all(channel[:, 1] == 2.0)
    assert torch.all(context_manager.cache_last_time[:, 1] == 2.0)
    assert context_manager.cache_last_channel_len[1] == 2
    assert context_manager.free_slots.qsize() == 1
    context_manager.free_stream(0)
    assert context_manager.free_slots.qsize() == 1

    context_manager.reset()
    assert context_manager.free_slots.qsize() == NUM_SLOTS
    assert context_manager.streamidx2slotidx == {}
    assert context_manager.cache_last_channel is channel
    assert context_manager.cache_last_channel.data_ptr() == channel_ptr
    assert torch.all(context_manager.cache_last_channel == 0)
    assert torch.all(context_manager.cache_last_time == 0)
    assert torch.all(context_manager.cache_last_channel_len == 0)

    feature_buffer = bufferer.feature_buffer
    bufferer.reset()
    _assert_bufferer_clean(bufferer)
    assert bufferer.feature_buffer is feature_buffer


@pytest.mark.unit
def test_context_manager_with_cache_disabled_ignores_free_and_reset():
    context_manager = CacheAwareContextManager(_StubCacheAwareModel(), num_slots=NUM_SLOTS, use_cache=False)
    context_manager.free_stream(0)
    context_manager.reset()
    assert context_manager.cache_last_channel is None
    assert context_manager.cache_last_time is None
    assert context_manager.cache_last_channel_len is None


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
@pytest.mark.parametrize("auto_manage", [True, False])
@pytest.mark.parametrize("released_first", [False, True])
def test_delete_state_releases_only_owned_biasing(auto_manage, released_first):
    """RNNT delete_state drops an auto-managed biasing model once, while its state still exists."""
    pipeline = _make_pipeline(CacheAwareRNNTPipeline)
    _take_slots(pipeline, [0])
    cfg = BiasingRequestItemConfig(multi_model_id=7, auto_manage_multi_model=auto_manage)
    state = CacheAwareRNNTStreamingState()
    state.set_options(ASRRequestOptions(biasing_cfg=cfg))
    pipeline._state_pool[0] = state
    removed = []

    def remove_model(model_id):
        assert pipeline.get_state(0) is state
        removed.append(model_id)

    multi_model = SimpleNamespace(remove_model=remove_model)
    pipeline.decoding_computer = SimpleNamespace(per_stream_biasing_enabled=True, biasing_multi_model=multi_model)
    if released_first:
        # graceful is_last teardown already released the model and both slots
        release_auto_managed_stream_biasing(state, multi_model)
        pipeline.bufferer.free_stream(0)
        pipeline.context_manager.free_stream(0)

    pipeline.delete_state(0)
    pipeline.delete_state(0)

    assert removed == ([7] if auto_manage else [])
    assert cfg.multi_model_id == (None if auto_manage else 7)
    assert pipeline.get_state(0) is None
    assert _free_counts(pipeline) == (NUM_SLOTS, NUM_SLOTS)


@pytest.mark.unit
@pytest.mark.parametrize("pipeline_cls", PIPELINES)
def test_close_session_and_open_session_restore_all_slots(pipeline_cls):
    """close_session / open_session return every slot, including streams that never called delete_state."""
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
def test_base_delete_state_only_drops_state_and_keeps_slots(pipeline_cls):
    """BasePipeline.delete_state only drops state; the cache-aware override is what frees slots."""
    pipeline = _make_pipeline(pipeline_cls)
    _take_slots(pipeline, [0])
    BasePipeline.delete_state(pipeline, 0)
    assert 0 not in pipeline._state_pool
    assert _free_counts(pipeline) == (NUM_SLOTS - 1, NUM_SLOTS - 1)
