# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.tts.parts.utils.helpers import (
    get_speaker_embeddings_from_waveforms,
    transcribe_with_whisper_from_waveforms,
)

pytestmark = pytest.mark.unit


def test_speaker_receives_resampled_batch_and_true_lengths():
    seen = []

    def forward(input_signal, input_signal_length):
        seen.append(input_signal)
        assert input_signal.shape == (2, 3200)
        assert input_signal_length.tolist() == [1600, 3200]
        assert not input_signal.requires_grad
        return None, torch.ones(2, 3)

    model = SimpleNamespace(cfg=SimpleNamespace(preprocessor=SimpleNamespace(sample_rate=16000)), forward=forward)
    result = get_speaker_embeddings_from_waveforms(
        [torch.ones(2400, requires_grad=True), torch.ones(4800)], 24000, model, 'cpu'
    )
    assert result.shape == (2, 3)
    assert len(seen) == 1


def test_whisper_batches_by_language_and_preserves_order():
    chunks = []

    class Processor:
        def get_decoder_prompt_ids(self, language, task):
            return language

        def __call__(self, arrays, sampling_rate, **kwargs):
            assert sampling_rate == 16000
            assert all(array.shape == (1600,) for array in arrays)
            chunks.append(len(arrays))
            return SimpleNamespace(input_features=torch.tensor([[round(float(wav[100]))] for wav in arrays]))

        def batch_decode(self, ids, **kwargs):
            return [str(value.item()) for value in ids]

    model = SimpleNamespace(dtype=torch.float32, generate=lambda inputs, **kwargs: inputs[:, 0])
    result = transcribe_with_whisper_from_waveforms(
        [torch.full((2400,), float(i)) for i in (1, 2, 3)],
        24000,
        ['en', 'hi', 'en'],
        Processor(),
        model,
        'cpu',
        batch_size=2,
    )
    assert result == ['1.0', '2.0', '3.0']
    assert chunks == [2, 1]


@pytest.mark.parametrize('languages,batch_size', [(['en'], 2), ('en', 0)])
def test_whisper_rejects_invalid_parameters(languages, batch_size):
    with pytest.raises(ValueError):
        transcribe_with_whisper_from_waveforms(
            [torch.ones(100)] * 2, 16000, languages, None, None, 'cpu', batch_size=batch_size
        )
