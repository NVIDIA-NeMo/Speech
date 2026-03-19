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

import pytest
import torch

from nemo.collections.asr.inference.streaming.framing.mono_stream import MonoStream
from nemo.collections.asr.inference.streaming.framing.multi_stream import MultiStream


@pytest.fixture(scope="module")
def test_audios():
    return torch.ones(83200), torch.ones(118960)


class TestMonoWavStream:

    @pytest.mark.unit
    def test_mono_wav_stream_no_pad(self, test_audios):
        for audio in test_audios:
            stream = MonoStream(16000, 2.5, stream_id=0, pad_last_frame=False)
            stream.load_audio(audio, options=None)
            audio_len_in_samples = stream.samples.shape[0]
            i = 0
            total_samples = 0
            for frame in iter(stream):
                total_samples += len(frame[0].samples)
                i += 1
            assert total_samples == audio_len_in_samples
            assert frame[0].is_last == True

    @pytest.mark.unit
    def test_mono_wav_stream_with_pad(self, test_audios):
        for audio in test_audios:
            stream = MonoStream(16000, 2.5, stream_id=0, pad_last_frame=True)
            stream.load_audio(audio, options=None)
            for frame in iter(stream):
                last_frame_size = frame[0].size
            assert last_frame_size == stream.frame_size


class TestMultiStream:

    @pytest.mark.unit
    def test_multi_stream(self, test_audios):
        multi_stream = MultiStream(n_frames_per_stream=1)
        audio_len_in_samples = {}
        for stream_id, audio in enumerate(test_audios):
            stream = MonoStream(16000, 2.5, stream_id=stream_id, pad_last_frame=False)
            stream.load_audio(audio, options=None)
            multi_stream.add_stream(stream, stream_id=stream_id)
            audio_len_in_samples[stream_id] = stream.samples.shape[0]

        total_samples = {}
        for frames in iter(multi_stream):
            for frame in frames:
                total_samples[frame.stream_id] = total_samples.get(frame.stream_id, 0) + frame.size

        assert total_samples == audio_len_in_samples
