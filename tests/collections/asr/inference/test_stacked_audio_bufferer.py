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

import random

import pytest
import torch

from nemo.collections.asr.inference.pipelines.buffered_ctc_pipeline import BufferedCTCPipeline
from nemo.collections.asr.inference.streaming.buffering.audio_bufferer import BatchedAudioBufferer
from nemo.collections.asr.inference.streaming.buffering.stacked_audio_bufferer import StackedAudioBufferer
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.utils.enums import PaddingMode

# 4 samples per second and 2 s buffers: the buffer holds 8 samples and frames of 3 samples are used
SAMPLE_RATE = 4
BUFFER_SECS = 2.0


def _frame(stream_id, values, is_last=False):
    return Frame(samples=torch.tensor(values, dtype=torch.float32), stream_id=stream_id, is_last=is_last)


@pytest.mark.unit
def test_buffers_and_left_paddings_of_one_stream_match_hand_computed_values():
    bufferer = StackedAudioBufferer(SAMPLE_RATE, BUFFER_SECS, device="cpu")
    expected = [
        ([0, 0, 0, 0, 0, 1, 2, 3], 5),
        ([0, 0, 1, 2, 3, 4, 5, 6], 2),
        ([2, 3, 4, 5, 6, 7, 8, 9], 0),
    ]
    for step, (buffer, left_padding) in enumerate(expected):
        values = [3 * step + 1, 3 * step + 2, 3 * step + 3]
        buffers, left_paddings = bufferer.update([_frame(0, values)])
        assert buffers.tolist() == [buffer]
        assert left_paddings.tolist() == [left_padding]


@pytest.mark.unit
def test_a_stream_that_takes_over_a_released_slot_starts_from_a_zero_buffer():
    bufferer = StackedAudioBufferer(SAMPLE_RATE, BUFFER_SECS, device="cpu", num_slots=1)
    bufferer.update([_frame(0, [1, 2, 3], is_last=True)])

    buffers, left_paddings = bufferer.update([_frame(1, [7, 8, 9])])

    assert buffers.tolist() == [[0, 0, 0, 0, 0, 7, 8, 9]]
    assert left_paddings.tolist() == [5]


@pytest.mark.unit
def test_buffers_of_streams_that_start_and_end_at_different_steps_match_the_per_stream_bufferer():
    rng = random.Random(0)
    # (first step, number of frames) per stream; more streams than the initial slots, so slots are reused and grown
    schedule = {0: (0, 4), 1: (0, 2), 2: (1, 5), 3: (3, 3), 4: (3, 1), 5: (4, 4)}
    reference = BatchedAudioBufferer(SAMPLE_RATE, BUFFER_SECS)
    stacked = StackedAudioBufferer(SAMPLE_RATE, BUFFER_SECS, device="cpu", num_slots=2)

    for step in range(9):
        frames = []
        for stream_id, (start, count) in schedule.items():
            if start <= step < start + count:
                values = [rng.random() for _ in range(3)]
                frames.append(_frame(stream_id, values, is_last=step == start + count - 1))
        if not frames:
            continue
        ref_buffers, ref_left_paddings = reference.update(frames)
        buffers, left_paddings = stacked.update(frames)
        assert torch.equal(buffers, torch.stack(ref_buffers))
        assert left_paddings.tolist() == ref_left_paddings


@pytest.mark.unit
def test_batch_with_frames_of_different_sizes_or_repeated_streams_is_rejected():
    bufferer = StackedAudioBufferer(SAMPLE_RATE, BUFFER_SECS, device="cpu")
    with pytest.raises(ValueError, match="same size"):
        bufferer.update([_frame(0, [1, 2, 3]), _frame(1, [1, 2])])
    with pytest.raises(ValueError, match="one frame per stream"):
        bufferer.update([_frame(0, [1, 2, 3]), _frame(0, [4, 5, 6])])


@pytest.mark.unit
def test_buffered_ctc_rolls_each_buffer_by_its_own_left_padding_and_shortens_its_length():
    seen = {}

    def preprocess(buffers, buffer_lens, expected_feature_buffer_len):
        seen["buffers"], seen["buffer_lens"] = buffers, buffer_lens
        return torch.zeros(2, 1, 1), torch.ones(2, dtype=torch.int64)

    class _Model:
        def get_logprobs(self, processed_signal, processed_signal_length):
            return torch.zeros(2, 1, 1)

    pipeline = object.__new__(BufferedCTCPipeline)
    pipeline.right_padding = True
    pipeline.padding_mode = PaddingMode.RIGHT
    pipeline.preprocess = preprocess
    pipeline.asr_model = _Model()
    pipeline.device = torch.device("cpu")
    pipeline.expected_feature_buffer_len = 1
    pipeline.sample_rate = SAMPLE_RATE
    pipeline.model_stride_in_secs = 1e9  # the roll back of the log probs is not under test
    pipeline.zero_log_probs = torch.zeros(1, 1)
    frames = [_frame(0, [0, 0, 0]), Frame(samples=torch.zeros(3), stream_id=1, length=2)]
    raw_signals = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    pipeline.get_logprobs_given_raw_signals(frames, raw_signals, torch.tensor([0, 3]))

    # row 0 has no left padding, row 1 starts at its third sample; the second frame has one padded sample
    assert seen["buffers"].tolist() == [[0, 1, 2, 3, 4, 5, 6, 7], [11, 12, 13, 14, 15, 8, 9, 10]]
    assert seen["buffer_lens"].tolist() == [8, 8 - 1 - 3]
