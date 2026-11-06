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

from types import SimpleNamespace

import pytest

from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.inference.pipelines.buffered_rnnt_pipeline import BufferedRNNTPipeline
from nemo.collections.asr.inference.pipelines.cache_aware_rnnt_pipeline import CacheAwareRNNTPipeline
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.state.cache_aware_rnnt_state import CacheAwareRNNTStreamingState
from nemo.collections.asr.inference.streaming.state.rnnt_state import RNNTStreamingState
from nemo.collections.asr.inference.utils.per_stream_biasing import (
    build_multi_biasing_ids_np,
    release_auto_managed_stream_biasing,
)
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import (
    BiasingRequestItemConfig,
    GPUBiasingMultiModel,
)
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig

PIPELINES = [
    pytest.param(CacheAwareRNNTPipeline, CacheAwareRNNTStreamingState, id="cache_aware_rnnt"),
    pytest.param(BufferedRNNTPipeline, RNNTStreamingState, id="buffered_rnnt"),
]


class _LetterTokenizer:
    """Maps a..z to token ids 1..26, enough to build a boosting tree without a checkpoint."""

    vocab_size = 27

    def text_to_ids(self, text):
        return [ord(char) - ord("a") + 1 for char in text]


def _make_pipeline(pipeline_cls):
    """A pipeline with only the members that stream bookkeeping uses, and a real biasing multi-model."""
    pipeline = pipeline_cls.__new__(pipeline_cls)
    BasePipeline.__init__(pipeline)
    biasing_multi_model = GPUBiasingMultiModel(vocab_size=_LetterTokenizer.vocab_size, use_triton=False)
    pipeline.decoding_computer = SimpleNamespace(
        biasing_multi_model=biasing_multi_model, per_stream_biasing_enabled=True
    )
    return pipeline, biasing_multi_model


def _add_stream(pipeline, state_cls, stream_id, phrases=None, auto_manage=True):
    state = state_cls()
    biasing_cfg = None
    if phrases is not None:
        biasing_cfg = BiasingRequestItemConfig(
            boosting_model_cfg=BoostingTreeModelConfig(key_phrases_list=phrases, use_triton=False),
            auto_manage_multi_model=auto_manage,
        )
    state.set_options(ASRRequestOptions(biasing_cfg=biasing_cfg))
    pipeline._state_pool[stream_id] = state
    return state


def _register(pipeline, states):
    """Register the streams' biasing models with the helper the cache-aware pipeline calls on each step.

    The buffered pipeline makes the same `add_to_multi_model` call inline; either way the model is added
    and its id is recorded in the stream's `multi_model_id`, which is what `delete_state` releases.
    """
    build_multi_biasing_ids_np(states, pipeline.decoding_computer.biasing_multi_model, _LetterTokenizer())


def _active_model_ids(biasing_multi_model):
    return {i for i in range(biasing_multi_model.num_models) if biasing_multi_model.model2active[i].item()}


@pytest.mark.unit
@pytest.mark.parametrize("pipeline_cls, state_cls", PIPELINES)
def test_delete_state_releases_biasing_model_of_stream_ended_without_is_last(pipeline_cls, state_cls):
    """A stream dropped with delete_state() before an is_last chunk must not keep its model in the decoder."""
    pipeline, biasing_multi_model = _make_pipeline(pipeline_cls)
    states = [_add_stream(pipeline, state_cls, stream_id, phrases=["nemo", "speech"]) for stream_id in range(3)]
    _register(pipeline, states)
    assert len(_active_model_ids(biasing_multi_model)) == 3

    for stream_id in range(3):
        pipeline.delete_state(stream_id)

    assert pipeline._state_pool == {}
    assert _active_model_ids(biasing_multi_model) == set()
    assert biasing_multi_model.num_states_total == 0
    assert all(state.options.biasing_cfg.multi_model_id is None for state in states)

    # a following stream reuses a released id instead of growing the model table
    _register(pipeline, [_add_stream(pipeline, state_cls, 3, phrases=["nemo"])])
    assert biasing_multi_model.num_models == 3


@pytest.mark.unit
@pytest.mark.parametrize("pipeline_cls, state_cls", PIPELINES)
def test_delete_state_releases_only_the_deleted_streams_model(pipeline_cls, state_cls):
    """delete_state() is a no-op after the is_last release and never releases another request's model."""
    pipeline, biasing_multi_model = _make_pipeline(pipeline_cls)
    finished = _add_stream(pipeline, state_cls, 0, phrases=["nemo"])
    dropped = _add_stream(pipeline, state_cls, 1, phrases=["speech"])
    user_managed = _add_stream(pipeline, state_cls, 2, phrases=["asr"], auto_manage=False)
    _add_stream(pipeline, state_cls, 3)  # no biasing request
    _register(pipeline, [finished, dropped])
    user_managed.options.biasing_cfg.add_to_multi_model(_LetterTokenizer(), biasing_multi_model)
    finished_id = finished.options.biasing_cfg.multi_model_id
    dropped_id = dropped.options.biasing_cfg.multi_model_id
    user_managed_id = user_managed.options.biasing_cfg.multi_model_id

    # stream 0 ends normally: its is_last step releases its model, and a new stream takes the freed id
    release_auto_managed_stream_biasing(finished, biasing_multi_model)
    newcomer = _add_stream(pipeline, state_cls, 4, phrases=["nemo", "asr"])
    _register(pipeline, [newcomer])
    newcomer_id = newcomer.options.biasing_cfg.multi_model_id
    assert newcomer_id == finished_id

    pipeline.delete_state(0)  # must not release the id again, now that it belongs to stream 4
    pipeline.delete_state(3)
    pipeline.delete_state(2)
    pipeline.delete_state(1)  # stream 1 is dropped without an is_last chunk

    assert dropped_id not in {newcomer_id, user_managed_id}
    assert _active_model_ids(biasing_multi_model) == {newcomer_id, user_managed_id}
    assert dropped.options.biasing_cfg.multi_model_id is None
    assert user_managed.options.biasing_cfg.multi_model_id == user_managed_id
    assert list(pipeline._state_pool) == [4]
