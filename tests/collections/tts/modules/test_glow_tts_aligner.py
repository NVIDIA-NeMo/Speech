# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.aligner import TinyGlowTTSAligner

pytestmark = pytest.mark.unit


def test_likelihood_includes_flow_jacobian_and_backpropagates():
    torch.manual_seed(21)
    aligner = TinyGlowTTSAligner(
        4,
        6,
        hidden_channels=8,
        num_flows=2,
        num_text_layers=1,
        coupling_num_layers=2,
        coupling_kernel_size=3,
        text_dropout=0.0,
        mas_backend='cpu',
    ).eval()
    audio = torch.randn(2, 7, 4)
    text = torch.randn(2, 1, 6)
    audio_lens = torch.tensor([7, 5])
    text_lens = torch.ones(2, dtype=torch.long)
    with torch.no_grad():
        aligner.flow.actnorms[0].logs.fill_(0.2)
    output = aligner(audio, text, audio_lens, text_lens)
    audio_mask = (torch.arange(7)[None, :] < audio_lens[:, None]).unsqueeze(1).float()
    z, log_det = aligner.flow(audio.transpose(1, 2), audio_mask)
    means = aligner.text_encoder(text.transpose(1, 2), torch.ones(2, 1, 1))
    log_probs = -0.5 * ((z - means).square() + math.log(2.0 * math.pi))
    expected = -((log_probs * audio_mask).sum() + log_det.sum()) / (audio_lens.sum() * 4)
    torch.testing.assert_close(output.loss, expected)
    output.loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in aligner.parameters()
    )
    assert aligner.flow.actnorms[0].logs.grad.abs().sum() > 0
    torch.testing.assert_close(output.durations[:, 0], audio_lens)
    assert (output.token_indices[1, 5:] == -1).all()


def test_likelihood_recovers_acoustically_distinct_segments():
    class IdentityPrior(nn.Module):
        def forward(self, features, mask):
            return features * mask

    aligner = TinyGlowTTSAligner(
        4,
        4,
        hidden_channels=8,
        num_flows=2,
        num_text_layers=1,
        coupling_num_layers=2,
        coupling_kernel_size=3,
        text_dropout=0.0,
        mas_backend='cpu',
    ).eval()
    aligner.text_encoder = IdentityPrior()
    with torch.no_grad():
        for mixer in aligner.flow.mixers:
            mixer.weight.copy_(torch.eye(mixer.n_split))
    text = torch.tensor([[[-3.0] * 4, [0.0] * 4, [3.0] * 4]])
    expected_indices = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2]])
    audio = text[:, expected_indices[0]]
    output = aligner(audio, text, torch.tensor([9]), torch.tensor([3]))
    torch.testing.assert_close(output.token_indices, expected_indices)
    torch.testing.assert_close(output.durations, torch.tensor([[3, 2, 4]]))


@pytest.mark.parametrize(
    'profile',
    [
        {'flow_type': 'legacy'},
        {'flow_type': 'unknown'},
        {'text_encoder_type': 'cnn'},
        {'text_encoder_type': 'unknown'},
    ],
)
def test_unsupported_architecture_profiles_raise(profile):
    with pytest.raises(ValueError, match='supports only'):
        TinyGlowTTSAligner(4, 6, **profile)
