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

from nemo.collections.asr.inference.streaming.buffering.audio_bufferer import AudioBufferer, BatchedAudioBufferer
from nemo.collections.asr.inference.streaming.framing.mono_stream import MonoStream
from nemo.collections.asr.inference.streaming.framing.multi_stream import MultiStream


@pytest.fixture(scope="module")
def test_audios():
    return torch.ones(83200), torch.ones(118960)


class TestAudioBufferer:

    @pytest.mark.unit
    def test_audio_bufferer(self, test_audios):
        for audio in test_audios:
            stream = MonoStream(16000, frame_size_in_secs=2.5, stream_id=0, pad_last_frame=False)
            stream.load_audio(audio, options=None)

            frame_bufferer = AudioBufferer(16000, buffer_size_in_secs=5.0)

            for frame in iter(stream):
                frame = frame[0]
                frame_bufferer.update(frame)
                buffer = frame_bufferer.get_buffer()

                assert len(buffer) == frame_bufferer.buffer_size
                assert torch.allclose(buffer[-frame.size :], frame.samples, atol=1e-5)


class TestBatchedAudioBufferer:

    @pytest.mark.unit
    def test_batched_audio_bufferer(self, test_audios):

        multi_stream = MultiStream(n_frames_per_stream=1)
        for stream_id, audio in enumerate(test_audios):
            stream = MonoStream(16000, 2.5, stream_id=stream_id, pad_last_frame=False)
            stream.load_audio(audio, options=None)
            multi_stream.add_stream(stream, stream_id=stream_id)

        batched_audio_bufferer = BatchedAudioBufferer(16000, buffer_size_in_secs=5.0)

        for frames in iter(multi_stream):
            buffered_frames, left_paddings = batched_audio_bufferer.update(frames)
            for idx, frame in enumerate(frames):
                frame_buffer = buffered_frames[idx]
                assert torch.allclose(frame_buffer[-frame.size :], frame.samples, atol=1e-5)
