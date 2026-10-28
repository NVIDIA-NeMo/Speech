# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.aligned_text_head import align_text_turns
from tests.collections.tts.modules.test_aligned_text_head import _DeterministicAligner, _embed_text, _turn_fixture

pytestmark = pytest.mark.unit


class _SpeakerAligner(_DeterministicAligner):
    def __init__(self):
        super().__init__()
        self.use_speaker_conditioning = True
        self.reference_projection = nn.Linear(1, 1, bias=False)
        self.reference_projection.weight.data.fill_(1.0)
        self.references_encoded = 0
        self.packed_speakers = []

    def encode_speaker_reference(self, features, lengths):
        self.references_encoded += 1
        assert not features.requires_grad
        assert lengths.tolist() == [3, 3]
        return self.reference_projection(features.mean(dim=1))

    def forward(self, *args, speaker_embeddings=None):
        result = super().forward(*args)
        self.packed_speakers.append(speaker_embeddings.detach().clone())
        result.loss = result.loss + speaker_embeddings.mean()
        return result


def test_reference_encoder_runs_once_and_gathers_original_samples_after_turn_sorting():
    audio, text, text_lens, turns, warmup = _turn_fixture()
    references = torch.tensor([[[10.0]] * 3, [[30.0]] * 3], requires_grad=True)
    aligner = _SpeakerAligner()
    result = align_text_turns(
        aligner,
        audio,
        text,
        text_lens,
        turns,
        warmup,
        _embed_text,
        pad_id=0,
        use_alignment=True,
        max_turns_per_batch=2,
        max_frames_per_batch=8,
        speaker_reference_features=references,
        speaker_reference_lens=torch.tensor([3, 3]),
    )
    assert aligner.references_encoded == 1
    assert [value.flatten().tolist() for value in aligner.packed_speakers] == [[10.0, 30.0], [10.0]]
    torch.testing.assert_close(result.targets[0, 7:12], torch.tensor([21, 21, 22, 22, 23]))
    result.loss.backward()
    assert references.grad is None and audio.grad is None
    assert aligner.reference_projection.weight.grad.abs().sum() > 0


def test_reference_batch_axis_cannot_be_confused_with_compact_turn_axis():
    audio, text, text_lens, turns, warmup = _turn_fixture()
    with pytest.raises(ValueError, match='conversation batch'):
        align_text_turns(
            _SpeakerAligner(),
            audio,
            text,
            text_lens,
            turns,
            warmup,
            _embed_text,
            pad_id=0,
            use_alignment=True,
            speaker_reference_features=torch.zeros(5, 3, 1),
            speaker_reference_lens=torch.full((5,), 3),
        )
