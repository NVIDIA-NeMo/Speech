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

from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedFrameStreamer
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

SAMPLE_RATE = 16000
MISSING_FILES = ["/nonexistent/a.wav", "/nonexistent/b.wav", "/nonexistent/c.wav"]


def _streamer(batch_size=2):
    return ContinuousBatchedFrameStreamer(
        sample_rate=SAMPLE_RATE, frame_size_in_secs=0.1, batch_size=batch_size, n_frames_per_stream=1
    )


def _streamed_samples(streamer, num_streams):
    """Concatenates the valid samples of the frames each stream produced, in order."""
    chunks = {sid: [] for sid in range(num_streams)}
    for frames in streamer:
        for frame in frames:
            chunks[frame.stream_id].append(frame.samples[: frame.valid_size])
    return {sid: torch.cat(parts) for sid, parts in chunks.items()}


@pytest.mark.unit
def test_streamer_feeds_each_stream_from_its_preloaded_samples():
    # the files do not exist, so any read from disk would fail; distinct ramps catch streams being mixed up
    samples = [torch.arange(n, dtype=torch.float32) for n in (3500, 1700, 2600)]
    streamer = _streamer()
    streamer.set_audio_filepaths(MISSING_FILES, [ASRRequestOptions()] * 3, audio_samples=samples)

    streamed = _streamed_samples(streamer, num_streams=3)

    for sid, expected in enumerate(samples):
        assert torch.equal(streamed[sid], expected)


@pytest.mark.unit
def test_streamer_rejects_audio_samples_of_another_length_than_the_filepaths():
    streamer = _streamer()
    with pytest.raises(ValueError, match="audio_samples"):
        streamer.set_audio_filepaths(MISSING_FILES, [ASRRequestOptions()] * 3, audio_samples=[torch.zeros(10)])


@pytest.mark.unit
def test_streamer_reports_the_filepath_and_the_valid_audio_duration_of_each_stream():
    # 3500, 1700 and 2600 samples at 16 kHz are 0.21875 s, 0.10625 s and 0.1625 s; the padded last frames must not count
    samples = [torch.zeros(n) for n in (3500, 1700, 2600)]
    streamer = ContinuousBatchedFrameStreamer(
        sample_rate=SAMPLE_RATE, frame_size_in_secs=0.1, batch_size=2, n_frames_per_stream=1, pad_last_frame=True
    )
    streamer.set_audio_filepaths(MISSING_FILES, [ASRRequestOptions()] * 3, audio_samples=samples)

    _streamed_samples(streamer, num_streams=3)

    assert [streamer.get_audio_filepath(sid) for sid in range(3)] == MISSING_FILES
    assert [streamer.get_elapsed_duration(sid) for sid in range(3)] == pytest.approx([0.21875, 0.10625, 0.1625])


class _FakeRequestGenerator:
    def __init__(self):
        self.audio_samples = "not called"

    def set_audio_filepaths(self, audio_filepaths, options, audio_samples=None):
        self.audio_samples = audio_samples

    def set_progress_bar(self, progress_bar):
        pass

    def __iter__(self):
        return iter([])


class _Pipeline(BasePipeline):
    pass


_Pipeline.__abstractmethods__ = frozenset()


@pytest.mark.unit
def test_pipeline_run_hands_the_preloaded_samples_to_the_request_generator():
    pipeline = object.__new__(_Pipeline)
    generator = _FakeRequestGenerator()
    pipeline.get_request_generator = lambda: generator
    pipeline.get_sep = lambda: " "
    pipeline.open_session = pipeline.close_session = lambda: None
    samples = [torch.zeros(10), torch.zeros(20)]

    output = pipeline.run(MISSING_FILES[:2], audio_samples=samples)

    assert generator.audio_samples is samples
    assert output == {}
