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
import pytest
import torch

from nemo.collections.speechlm2.data.salm_dataset import MultiSpeakerConfig
from nemo.collections.speechlm2.parts.encoder_chunking import (
    _recombine_chunked_audio_embeddings,
    _split_audio_into_chunks,
    _split_spk_targets_into_chunks,
    encode_audio_with_optional_chunking,
)
from tests.collections.speechlm2._chunking_helpers import ChunkingTestPerception


@pytest.mark.parametrize(
    (
        "input_signal_lengths",
        "chunk_size_samples",
        "min_chunk_size_samples",
        "expected_chunk_lens",
        "expected_chunks_per_audio",
        "expected_chunk_spans",
    ),
    [
        (
            [5, 0, 7],
            3,
            2,
            [3, 2, 0, 3, 4],
            [2, 1, 2],
            [(0, 0, 3), (0, 3, 5), (1, 0, 0), (2, 0, 3), (2, 3, 7)],
        ),
        (
            [6],
            2,
            1,
            [2, 2, 2],
            [3],
            [(0, 0, 2), (0, 2, 4), (0, 4, 6)],
        ),
    ],
)
def test_split_audio_into_chunks_returns_spans_independent_of_spk_targets(
    input_signal_lengths,
    chunk_size_samples,
    min_chunk_size_samples,
    expected_chunk_lens,
    expected_chunks_per_audio,
    expected_chunk_spans,
):
    max_signal_len = max(input_signal_lengths, default=0)
    input_signal = torch.arange(len(input_signal_lengths) * max_signal_len, dtype=torch.float32).reshape(
        len(input_signal_lengths), max_signal_len
    )
    chunks, chunk_lens, chunks_per_audio, chunk_spans = _split_audio_into_chunks(
        input_signal=input_signal,
        input_signal_lengths=input_signal_lengths,
        chunk_size_samples=chunk_size_samples,
        min_chunk_size_samples=min_chunk_size_samples,
    )

    assert chunk_lens == expected_chunk_lens
    assert chunks_per_audio == expected_chunks_per_audio
    assert chunk_spans == expected_chunk_spans
    for chunk, (audio_idx, begin, end) in zip(chunks, chunk_spans):
        assert torch.equal(chunk, input_signal[audio_idx, begin:end])

    assert _split_spk_targets_into_chunks(None, input_signal_lengths, chunk_spans) is None


@pytest.mark.parametrize(
    ("chunk_values", "chunk_lens", "chunks_per_audio", "expected_audio_values"),
    [
        (
            [[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]],
            [2, 3],
            [2],
            [[1.0, 2.0, 3.0, 4.0, 5.0]],
        ),
        (
            [[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [10.0, 11.0, 0.0], [12.0, 13.0, 14.0]],
            [2, 3, 2, 3],
            [2, 2],
            [[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 11.0, 12.0, 13.0, 14.0]],
        ),
    ],
)
def test_recombine_chunked_audio_embeddings_reconstructs_original_rows(
    chunk_values,
    chunk_lens,
    chunks_per_audio,
    expected_audio_values,
):
    chunked_embs = torch.tensor(chunk_values, dtype=torch.float32).unsqueeze(-1)
    chunked_emb_lens = torch.tensor(chunk_lens, dtype=torch.long)

    audio_embs = _recombine_chunked_audio_embeddings(chunked_embs, chunked_emb_lens, chunks_per_audio)

    assert len(audio_embs) == len(expected_audio_values)
    for audio_emb, expected_values in zip(audio_embs, expected_audio_values):
        assert torch.equal(audio_emb.squeeze(-1), torch.tensor(expected_values))


@pytest.mark.parametrize(
    (
        "input_signal_lengths",
        "chunk_spans",
        "expected_chunks",
    ),
    [
        (
            [5],
            [(0, 0, 2), (0, 2, 5)],
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0]],
            ],
        ),
        (
            [4],
            [(0, 0, 2), (0, 2, 4)],
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0]],
            ],
        ),
    ],
)
def test_split_spk_targets_into_chunks_uses_chunk_spans(
    input_signal_lengths,
    chunk_spans,
    expected_chunks,
):
    cfg = MultiSpeakerConfig()
    spk_targets = torch.arange(5 * cfg.num_speakers, dtype=torch.float32).reshape(1, 5, cfg.num_speakers)

    chunked_spk_targets = _split_spk_targets_into_chunks(spk_targets, input_signal_lengths, chunk_spans)

    assert torch.equal(chunked_spk_targets, torch.tensor(expected_chunks))


@pytest.mark.parametrize(
    ("audio_values", "audio_len", "expected_chunk_lens"),
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], 5, [2, 3]),
        ([1.0, 2.0, 3.0, 4.0], 4, [2, 2]),
    ],
)
def test_encode_audio_with_optional_chunking_does_not_forward_absent_spk_targets(
    audio_values, audio_len, expected_chunk_lens
):
    perception = ChunkingTestPerception(sampling_rate=2, hop_length=1)
    audios = torch.tensor([audio_values])
    audio_lens = torch.tensor([audio_len], dtype=torch.long)

    embs = encode_audio_with_optional_chunking(
        perception,
        audios,
        audio_lens,
        chunk_size_seconds=1.0,
        sampling_rate=2,
        spk_targets=None,
    )

    chunked_signal, chunked_lens = perception.calls[0]
    assert chunked_signal.shape == (len(expected_chunk_lens), max(expected_chunk_lens))
    assert torch.equal(chunked_lens, torch.tensor(expected_chunk_lens, dtype=torch.long))
    assert perception.spk_targets_calls[0] is None
    assert torch.equal(embs[0].squeeze(-1), audios[0])


@pytest.mark.parametrize(
    ("audio_values", "audio_len", "expected_chunk_lens", "expected_spk_targets"),
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            5,
            [2, 3],
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0]],
            ],
        ),
        (
            [1.0, 2.0, 3.0, 4.0],
            4,
            [2, 2],
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0]],
            ],
        ),
    ],
)
def test_encode_audio_with_optional_chunking_forwards_chunked_spk_targets(
    audio_values, audio_len, expected_chunk_lens, expected_spk_targets
):
    cfg = MultiSpeakerConfig()
    perception = ChunkingTestPerception(sampling_rate=2, hop_length=1)
    audios = torch.tensor([audio_values])
    audio_lens = torch.tensor([audio_len], dtype=torch.long)
    spk_targets = torch.arange(5 * cfg.num_speakers, dtype=torch.float32).reshape(1, 5, cfg.num_speakers)

    embs = encode_audio_with_optional_chunking(
        perception,
        audios,
        audio_lens,
        chunk_size_seconds=1.0,
        sampling_rate=2,
        spk_targets=spk_targets,
    )

    chunked_signal, chunked_lens = perception.calls[0]
    assert chunked_signal.shape == (len(expected_chunk_lens), max(expected_chunk_lens))
    assert torch.equal(chunked_lens, torch.tensor(expected_chunk_lens, dtype=torch.long))
    assert torch.equal(perception.spk_targets_calls[0], torch.tensor(expected_spk_targets))
    assert torch.equal(embs[0].squeeze(-1), audios[0])
