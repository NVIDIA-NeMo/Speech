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

from nemo.collections.asr.inference.streaming.buffering.incremental_audio_bufferer import IncrementalAudioBufferer
from nemo.collections.asr.inference.streaming.framing.mono_stream import MonoStream
from nemo.collections.asr.inference.streaming.framing.request import Frame


@pytest.fixture(scope="module")
def test_audios():
    return torch.ones(83200), torch.ones(118960)


def _make_frame(samples: torch.Tensor, stream_id: int = 0, is_first: bool = False, is_last: bool = False) -> Frame:
    return Frame(
        samples=samples,
        stream_id=stream_id,
        is_first=is_first,
        is_last=is_last,
    )


class TestIncrementalAudioBufferer:
    """Tests for IncrementalAudioBufferer."""

    @pytest.mark.unit
    def test_constructor_valid_params(self):
        """Constructor with valid params initializes buffer and capacity."""
        sample_rate = 16000
        buffer_size_in_secs = 5.0
        chunk_size_in_secs = 2.5
        overlap_size_in_secs = 2.5
        buf = IncrementalAudioBufferer(
            sample_rate=sample_rate,
            buffer_size_in_secs=buffer_size_in_secs,
            chunk_size_in_secs=chunk_size_in_secs,
            overlap_size_in_secs=overlap_size_in_secs,
        )
        assert buf.sample_rate == sample_rate
        assert buf.buffer_size == int(buffer_size_in_secs * sample_rate)
        assert buf.chunk_size == int(chunk_size_in_secs * sample_rate)
        assert buf.overlap_size == int(overlap_size_in_secs * sample_rate)
        assert buf.sample_buffer.shape[0] == buf.buffer_size
        assert buf.remaining_capacity == buf.buffer_size
        assert buf.head == 0
        assert not buf.is_full()

    @pytest.mark.unit
    def test_constructor_overlap_negative_raises(self):
        """Overlap < 0 raises ValueError."""
        with pytest.raises(ValueError, match="Overlap size.*must satisfy"):
            IncrementalAudioBufferer(
                sample_rate=16000,
                buffer_size_in_secs=5.0,
                chunk_size_in_secs=2.5,
                overlap_size_in_secs=-0.1,
            )

    @pytest.mark.unit
    def test_constructor_overlap_exceeds_buffer_raises(self):
        """Overlap > buffer_size raises ValueError."""
        with pytest.raises(ValueError, match="Overlap size.*must satisfy"):
            IncrementalAudioBufferer(
                sample_rate=16000,
                buffer_size_in_secs=5.0,
                chunk_size_in_secs=2.5,
                overlap_size_in_secs=6.0,
            )

    @pytest.mark.unit
    def test_constructor_buffer_not_divisible_by_chunk_raises(self):
        """Buffer size not divisible by chunk size raises ValueError."""
        with pytest.raises(ValueError, match="Buffer size.*must be divisible by chunk size"):
            IncrementalAudioBufferer(
                sample_rate=16000,
                buffer_size_in_secs=5.0,
                chunk_size_in_secs=1.7,
                overlap_size_in_secs=1.7,
            )

    @pytest.mark.unit
    def test_constructor_overlap_not_divisible_by_chunk_raises(self):
        """Overlap not divisible by chunk size raises ValueError."""
        with pytest.raises(ValueError, match="Overlap size.*must be divisible by chunk size"):
            IncrementalAudioBufferer(
                sample_rate=16000,
                buffer_size_in_secs=5.0,
                chunk_size_in_secs=2.5,
                overlap_size_in_secs=2.0,
            )

    @pytest.mark.unit
    def test_update_single_frame(self):
        """Single frame update fills start of buffer and decreases remaining_capacity."""
        buf = IncrementalAudioBufferer(
            sample_rate=16000,
            buffer_size_in_secs=5.0,
            chunk_size_in_secs=2.5,
            overlap_size_in_secs=2.5,
        )
        chunk_size = 40000  # 2.5 * 16000
        samples = torch.arange(chunk_size, dtype=torch.float32)
        frame = _make_frame(samples)
        buf.update(frame)
        assert buf.head == chunk_size
        assert buf.remaining_capacity == buf.buffer_size - chunk_size
        assert torch.allclose(buf.sample_buffer[:chunk_size], samples, atol=1e-5)
        assert not buf.is_full()

    @pytest.mark.unit
    def test_update_multiple_frames_until_full(self):
        """Multiple updates fill buffer; is_full() becomes True when capacity is 0."""
        buf = IncrementalAudioBufferer(
            sample_rate=16000,
            buffer_size_in_secs=5.0,
            chunk_size_in_secs=2.5,
            overlap_size_in_secs=2.5,
        )
        chunk_size = 40000
        for i in range(2):
            samples = torch.full((chunk_size,), float(i), dtype=torch.float32)
            frame = _make_frame(samples)
            buf.update(frame)
        assert buf.remaining_capacity == 0
        assert buf.is_full()
        assert buf.head == buf.buffer_size
        assert torch.allclose(buf.sample_buffer[:chunk_size], torch.zeros(chunk_size), atol=1e-5)
        assert torch.allclose(buf.sample_buffer[chunk_size:], torch.ones(chunk_size), atol=1e-5)

    @pytest.mark.unit
    def test_update_frame_exceeds_buffer_raises(self):
        """Frame larger than buffer size raises RuntimeError."""
        buf = IncrementalAudioBufferer(
            sample_rate=16000,
            buffer_size_in_secs=5.0,
            chunk_size_in_secs=2.5,
            overlap_size_in_secs=2.5,
        )
        oversized = torch.zeros(buf.buffer_size + 1)
        frame = _make_frame(oversized)
        with pytest.raises(RuntimeError, match="Frame size.*exceeds buffer size"):
            buf.update(frame)

    @pytest.mark.unit
    def test_incremental_audio_bufferer_with_mono_stream(self, test_audios):
        """Integration: feed frames from MonoStream; buffer contents and paddings are consistent."""
        sample_rate = 16000
        chunk_size_in_secs = 2.5
        buffer_size_in_secs = 5.0
        overlap_size_in_secs = 2.5
        for audio in test_audios:
            stream = MonoStream(sample_rate, frame_size_in_secs=chunk_size_in_secs, stream_id=0, pad_last_frame=False)
            stream.load_audio(audio, options=None)
            buf = IncrementalAudioBufferer(
                sample_rate=sample_rate,
                buffer_size_in_secs=buffer_size_in_secs,
                chunk_size_in_secs=chunk_size_in_secs,
                overlap_size_in_secs=overlap_size_in_secs,
            )
            for frame in iter(stream):
                frame = frame[0]
                buf.update(frame)
                # Newest frame is at [head - frame.size : head]; after update it's at [head - frame.size : head]
                start = buf.head - frame.size
                assert torch.allclose(buf.sample_buffer[start : buf.head], frame.samples, atol=1e-5)
                assert buf.remaining_capacity == max(0, buf.remaining_capacity)
