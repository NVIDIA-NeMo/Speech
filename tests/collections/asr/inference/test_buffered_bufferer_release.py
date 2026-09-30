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

from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.inference.pipelines.buffered_ctc_pipeline import BufferedCTCPipeline
from nemo.collections.asr.inference.pipelines.buffered_rnnt_pipeline import BufferedRNNTPipeline
from nemo.collections.asr.inference.pipelines.buffered_salm_pipeline import BufferedSALMPipeline
from nemo.collections.asr.inference.pipelines.cache_aware_ctc_pipeline import CacheAwareCTCPipeline
from nemo.collections.asr.inference.pipelines.cache_aware_rnnt_pipeline import CacheAwareRNNTPipeline
from nemo.collections.asr.inference.streaming.buffering.incremental_audio_bufferer import (
    BatchedIncrementalAudioBufferer,
)
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.state.cache_aware_ctc_state import CacheAwareCTCStreamingState
from nemo.collections.asr.inference.streaming.state.cache_aware_rnnt_state import CacheAwareRNNTStreamingState
from nemo.collections.asr.inference.streaming.state.ctc_state import CTCStreamingState
from nemo.collections.asr.inference.streaming.state.rnnt_state import RNNTStreamingState
from nemo.collections.asr.inference.streaming.state.salm_state import SALMStreamingState
from nemo.collections.asr.inference.utils.enums import RequestType

SAMPLE_RATE = 16000
CHUNK_SIZE_IN_SECS = 0.16
BUFFER_SIZE_IN_SECS = 0.64
CHUNK = int(SAMPLE_RATE * CHUNK_SIZE_IN_SECS)
PREPROCESSOR_CFG = OmegaConf.create({"features": 80, "window_stride": 0.01, "log": True})
FEATURE_BUFFER_LEN = int(BUFFER_SIZE_IN_SECS / PREPROCESSOR_CFG.window_stride)
SALMPipeline = BufferedSALMPipeline.__wrapped__  # the class behind @experimental

FRAME_PIPELINES = [
    pytest.param(BufferedRNNTPipeline, RNNTStreamingState, RequestType.FRAME, id="rnnt-frame"),
    pytest.param(BufferedCTCPipeline, CTCStreamingState, RequestType.FRAME, id="ctc-frame"),
    pytest.param(SALMPipeline, SALMStreamingState, RequestType.FRAME, id="salm-frame"),
]
PIPELINES = FRAME_PIPELINES + [
    pytest.param(BufferedRNNTPipeline, RNNTStreamingState, RequestType.FEATURE_BUFFER, id="rnnt-feature_buffer"),
    pytest.param(BufferedCTCPipeline, CTCStreamingState, RequestType.FEATURE_BUFFER, id="ctc-feature_buffer"),
]


@pytest.mark.unit
@pytest.mark.parametrize("pipeline_cls, state_cls, request_type", PIPELINES)
def test_delete_state_drops_bufferer_of_stream_ended_without_is_last(pipeline_cls, state_cls, request_type):
    """A stream dropped with delete_state() before an is_last request must not keep its bufferer."""
    pipeline = _make_pipeline(pipeline_cls, state_cls, request_type)
    for stream_id in range(3):
        pipeline.transcribe_step([_request(request_type, stream_id, is_first=True)])
    assert set(_bufferers(pipeline)) == {0, 1, 2}

    for stream_id in range(3):
        pipeline.delete_state(stream_id)
    pipeline.delete_state(99)  # an unknown stream id is ignored

    assert _bufferers(pipeline) == {}
    assert pipeline._state_pool == {}


@pytest.mark.unit
@pytest.mark.parametrize("pipeline_cls, state_cls, request_type", PIPELINES)
def test_delete_state_drops_only_the_deleted_streams_bufferer(pipeline_cls, state_cls, request_type):
    """delete_state() is a no-op after an is_last request and leaves the other streams' bufferers as they are."""
    pipeline = _make_pipeline(pipeline_cls, state_cls, request_type)
    control = _make_pipeline(pipeline_cls, state_cls, request_type)
    for p in (pipeline, control):
        for stream_id in range(3):
            p.transcribe_step([_request(request_type, stream_id, is_first=True, value=0.1 * (stream_id + 1))])
        p.transcribe_step([_request(request_type, 0, is_last=True)])  # stream 0 ends normally
    assert set(_bufferers(pipeline)) == {1, 2}

    pipeline.delete_state(0)  # its is_last request has already removed its bufferer
    pipeline.delete_state(1)  # stream 1 is dropped without an is_last request
    assert list(_bufferers(pipeline)) == [2]
    assert list(pipeline._state_pool) == [2]

    # stream 2 goes on with the buffer it had, as on a pipeline where no stream was deleted
    for p in (pipeline, control):
        p.transcribe_step([_request(request_type, 2, value=0.5)])
    buffer, padding = pipeline.last_seen[2]
    control_buffer, control_padding = control.last_seen[2]
    assert torch.equal(buffer, control_buffer)
    assert padding == control_padding


@pytest.mark.unit
@pytest.mark.parametrize("pipeline_cls, state_cls, request_type", PIPELINES)
def test_close_and_open_session_drop_all_bufferers(pipeline_cls, state_cls, request_type):
    """close_session() and open_session() drop the bufferers of streams that are still open."""
    pipeline = _make_pipeline(pipeline_cls, state_cls, request_type)
    pipeline.transcribe_step([_request(request_type, stream_id, is_first=True) for stream_id in range(3)])
    pipeline.delete_state(0)

    pipeline.close_session()
    assert _bufferers(pipeline) == {}
    assert pipeline._state_pool == {}

    pipeline.transcribe_step([_request(request_type, 7, is_first=True)])
    pipeline.open_session()
    assert _bufferers(pipeline) == {}


@pytest.mark.unit
@pytest.mark.parametrize("pipeline_cls, state_cls, request_type", FRAME_PIPELINES)
@pytest.mark.parametrize("end_stream", ["delete_state", "close_session"])
def test_reused_stream_id_starts_from_an_empty_buffer(pipeline_cls, state_cls, request_type, end_stream):
    """A new stream on the id of a stream ended without an is_last frame sees the audio buffer that a stream on a
    fresh pipeline sees, not the old stream's audio. (Feature-buffer requests carry their whole buffer, so only
    frames are buffered across requests.)"""
    fresh = _make_pipeline(pipeline_cls, state_cls, request_type)
    fresh.transcribe_step([_request(request_type, 5, is_first=True, value=0.3)])

    pipeline = _make_pipeline(pipeline_cls, state_cls, request_type)
    pipeline.transcribe_step([_request(request_type, 5, is_first=True, value=0.7)])
    pipeline.transcribe_step([_request(request_type, 5, value=0.9)])
    if end_stream == "delete_state":
        pipeline.delete_state(5)
    else:
        pipeline.close_session()
        pipeline.open_session()
    pipeline.transcribe_step([_request(request_type, 5, is_first=True, value=0.3)])

    buffer, padding = pipeline.last_seen[5]
    fresh_buffer, fresh_padding = fresh.last_seen[5]
    assert torch.equal(buffer, fresh_buffer)
    assert padding == fresh_padding


@pytest.mark.unit
@pytest.mark.parametrize(
    "pipeline_cls, state_cls",
    [
        pytest.param(CacheAwareRNNTPipeline, CacheAwareRNNTStreamingState, id="cache_aware_rnnt"),
        pytest.param(CacheAwareCTCPipeline, CacheAwareCTCStreamingState, id="cache_aware_ctc"),
    ],
)
def test_base_pipeline_leaves_other_bufferers_alone(pipeline_cls, state_cls):
    """BasePipeline.delete_state() and reset_session() release only the buffered pipelines' bufferers. Nothing is
    called on any other bufferer, such as the cache-aware pipelines' BatchedCacheFeatureBufferer, which its own
    pipeline manages. The BasePipeline methods are called directly, as a pipeline's own override does via super()."""
    pipeline = pipeline_cls.__new__(pipeline_cls)
    pipeline.bufferer = Mock()  # records any call made on it
    BasePipeline.__init__(pipeline)
    pipeline._state_pool[0] = _new_state(state_cls)

    BasePipeline.delete_state(pipeline, 0)
    BasePipeline.delete_state(pipeline, 99)
    BasePipeline.reset_session(pipeline)

    assert pipeline.bufferer.mock_calls == []
    assert pipeline._state_pool == {}


def _make_pipeline(pipeline_cls, state_cls, request_type):
    """A buffered pipeline whose model step is replaced by `_bufferer_step`; `transcribe_step`, `delete_state` and
    the sessions are the real ones, with the real per-stream bufferer the pipeline would build."""
    pipeline = pipeline_cls.__new__(pipeline_cls)
    pipeline.decoding_computer = None  # no decoder, so no per-stream biasing models
    pipeline.nmt_enabled = False
    pipeline.request_type = request_type
    pipeline.sample_rate = SAMPLE_RATE
    pipeline.buffer_size_in_secs = BUFFER_SIZE_IN_SECS
    pipeline.preprocessor_config = PREPROCESSOR_CFG
    pipeline.device = torch.device("cpu")
    if pipeline_cls is SALMPipeline:
        pipeline.audio_bufferer = BatchedIncrementalAudioBufferer(
            SAMPLE_RATE, BUFFER_SIZE_IN_SECS, CHUNK_SIZE_IN_SECS, BUFFER_SIZE_IN_SECS / 2
        )
    else:
        pipeline.init_bufferer_for_buffered_streaming()
    pipeline.get_sep = lambda: " "
    pipeline.create_state = lambda options: _new_state(state_cls)
    pipeline.transcribe_step_for_frames = lambda frames: _bufferer_step(pipeline, frames)
    pipeline.transcribe_step_for_feature_buffers = lambda fbuffers: _bufferer_step(pipeline, fbuffers)
    pipeline.last_seen = {}
    BasePipeline.__init__(pipeline)
    return pipeline


def _bufferer_step(pipeline, requests):
    """The bufferer call each buffered pipeline's transcribe step starts with. Keeps, per stream, the buffer and
    padding the model would get."""
    if isinstance(pipeline, SALMPipeline):
        buffers, paddings = pipeline.audio_bufferer.update(requests)
    elif isinstance(requests[0], Frame):
        buffers, paddings = pipeline.bufferer.update(requests)
    else:
        buffers = pipeline.bufferer.update(requests)
        paddings = [None] * len(buffers)
    for request, buffer, padding in zip(requests, buffers, paddings):
        pipeline.last_seen[request.stream_id] = (buffer, padding)


def _bufferers(pipeline):
    """The per-stream bufferers the pipeline holds, by stream id."""
    if isinstance(pipeline, SALMPipeline):
        return pipeline.audio_bufferer.bufferers
    return pipeline.bufferer.bufferers


def _new_state(state_cls):
    state = state_cls()
    state.set_options(ASRRequestOptions(asr_output_granularity="segment"))
    return state


def _request(request_type, stream_id, is_first=False, is_last=False, value=0.1):
    if request_type is RequestType.FEATURE_BUFFER:
        features = torch.full((PREPROCESSOR_CFG.features, FEATURE_BUFFER_LEN), value)
        return FeatureBuffer(features=features, stream_id=stream_id, is_first=is_first, is_last=is_last)
    samples = value * torch.linspace(-1.0, 1.0, CHUNK)
    return Frame(samples=samples, stream_id=stream_id, is_first=is_first, is_last=is_last, length=CHUNK)
