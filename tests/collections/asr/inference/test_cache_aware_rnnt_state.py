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

import pytest
import torch

from nemo.collections.asr.inference.streaming.decoders.greedy.greedy_rnnt_decoder import RNNTGreedyDecoder
from nemo.collections.asr.inference.streaming.state.cache_aware_rnnt_state import CacheAwareRNNTStreamingState
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

FRAMES_PER_CHUNK = 7


def _chunk_hypothesis(step: int) -> Hypothesis:
    """A chunk's decoded continuation: two tokens with global frame timestamps."""
    first_frame = step * FRAMES_PER_CHUNK
    return Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([step % 5, (step + 1) % 5]),
        timestamp=torch.tensor([first_frame + 1, first_frame + 4]),
        dec_state=None,
    )


def _run_stream(steps: int, trim: bool) -> tuple[list[tuple[dict, list[int]]], CacheAwareRNNTStreamingState]:
    decoder = RNNTGreedyDecoder(vocabulary=["a", "b", "c", "d", "e"])
    state = CacheAwareRNNTStreamingState()
    outputs = []
    for step in range(steps):
        continuation = _chunk_hypothesis(step)
        hyp = state.get_previous_hypothesis()
        if hyp is None:
            hyp = continuation
        else:
            hyp.merge_(continuation)
        state.set_previous_hypothesis(hyp)
        output, labels, new_offset = decoder(
            global_timestamps=hyp.timestamp, tokens=hyp.y_sequence, length=FRAMES_PER_CHUNK, offset=state.offset
        )
        state.set_offset(new_offset)
        outputs.append((output, labels))
        if trim:
            state.trim_consumed_hypothesis_()
    return outputs, state


@pytest.mark.unit
def test_trim_consumed_hypothesis_keeps_decoder_output():
    untrimmed, _ = _run_stream(steps=50, trim=False)
    trimmed, _ = _run_stream(steps=50, trim=True)
    assert trimmed == untrimmed


@pytest.mark.unit
def test_trim_consumed_hypothesis_keeps_carried_hypothesis_bounded():
    _, untrimmed_state = _run_stream(steps=50, trim=False)
    _, trimmed_state = _run_stream(steps=50, trim=True)
    assert len(untrimmed_state.previous_hypothesis.y_sequence) == 100
    assert len(trimmed_state.previous_hypothesis.y_sequence) == 0
    assert trimmed_state.offset == 0


@pytest.mark.unit
def test_trim_consumed_hypothesis_without_hypothesis_is_noop():
    state = CacheAwareRNNTStreamingState()
    state.set_offset(3)
    state.trim_consumed_hypothesis_()
    assert state.previous_hypothesis is None and state.offset == 3
