# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import torch

from nemo.collections.asr.parts.utils import streaming_utils
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer


def _make_streaming_buffer(stream_lengths, pre_encode_cache_size=3):
    buffer_length = max(stream_lengths)
    features = torch.zeros(len(stream_lengths), 1, buffer_length)
    for stream_idx, stream_length in enumerate(stream_lengths):
        features[stream_idx, 0, :stream_length] = (
            torch.arange(1, stream_length + 1, dtype=torch.float32) + 100 * stream_idx
        )

    streaming_buffer = CacheAwareStreamingAudioBuffer.__new__(CacheAwareStreamingAudioBuffer)
    streaming_buffer.buffer = features
    streaming_buffer.buffer_idx = 0
    streaming_buffer.streams_length = torch.tensor(stream_lengths)
    streaming_buffer.step = 0
    streaming_buffer.pad_and_drop_preencoded = False
    streaming_buffer.online_normalization = False
    streaming_buffer.sampling_frames = None
    streaming_buffer.input_features = features.shape[1]
    streaming_buffer.streaming_cfg = SimpleNamespace(
        chunk_size=14,
        shift_size=14,
        pre_encode_cache_size=pre_encode_cache_size,
    )
    return streaming_buffer, features


@pytest.mark.unit
@pytest.mark.parametrize("right_context_size", [0, 8, 24, 40, 104])
@pytest.mark.parametrize("stream_lengths", [(20,), (20, 16)])
def test_iter_with_right_context_preserves_asr_view_and_extends_diar_view(right_context_size, stream_lengths):
    legacy_buffer, _ = _make_streaming_buffer(stream_lengths)
    context_buffer, features = _make_streaming_buffer(stream_lengths)

    legacy_chunks = list(legacy_buffer)
    context_chunks = list(context_buffer.iter_with_right_context(right_context_size))

    assert len(context_chunks) == len(legacy_chunks)
    for step_idx, ((legacy_audio, legacy_lengths), context_chunk) in enumerate(zip(legacy_chunks, context_chunks)):
        asr_audio, asr_lengths, diar_audio, diar_lengths = context_chunk
        torch.testing.assert_close(asr_audio, legacy_audio)
        torch.testing.assert_close(asr_lengths, legacy_lengths)

        if right_context_size == 0:
            assert diar_audio is asr_audio
            assert diar_lengths is asr_lengths
            continue

        cache_size = 3
        chunk_start = step_idx * 14
        expected_width = cache_size + 14 + right_context_size
        assert diar_audio.shape[-1] == expected_width
        expected_lengths = (context_buffer.streams_length - chunk_start + cache_size).clamp(min=0, max=expected_width)
        torch.testing.assert_close(diar_lengths, expected_lengths)

        torch.testing.assert_close(diar_audio[:, :, : asr_audio.shape[-1]], asr_audio)
        future_start = chunk_start + 14
        for stream_idx, stream_length in enumerate(stream_lengths):
            real_future_length = min(right_context_size, max(0, stream_length - future_start))
            torch.testing.assert_close(
                diar_audio[stream_idx, 0, cache_size + 14 : cache_size + 14 + real_future_length],
                features[stream_idx, 0, future_start : future_start + real_future_length],
            )
            assert torch.count_nonzero(diar_audio[stream_idx, 0, cache_size + 14 + real_future_length :]) == 0

    assert legacy_buffer.buffer_idx == context_buffer.buffer_idx
    assert context_buffer.is_buffer_empty()


@pytest.mark.unit
@pytest.mark.parametrize(
    "right_context_size,expected_diar_normalization_lengths",
    [(8, ((20, 16), (9, 5))), (40, ((20, 16), (9, 5)))],
)
def test_diar_view_normalization_uses_real_ragged_lengths(
    monkeypatch, right_context_size, expected_diar_normalization_lengths
):
    streaming_buffer, _ = _make_streaming_buffer((20, 16))
    streaming_buffer.online_normalization = True
    streaming_buffer.model_normalize_type = "per_feature"
    captured_lengths = []

    def capture_normalization(x, seq_len, normalize_type):
        captured_lengths.append(tuple(seq_len.tolist()))
        return x, None, None

    monkeypatch.setattr(streaming_utils, "normalize_batch", capture_normalization)

    list(streaming_buffer.iter_with_right_context(right_context_size))

    diar_normalization_lengths = captured_lengths[1::2]
    assert diar_normalization_lengths == list(expected_diar_normalization_lengths)
