# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import math

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.glow_tts_aligner import TinyGlowTTSAligner, _AudioFlow

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('channels', [4, 5])
def test_flow_inverse_and_jacobian(channels):
    torch.manual_seed(20)
    flow = _AudioFlow(channels, hidden_channels=8, num_flows=2, max_log_scale=2.0).double()
    with torch.no_grad():
        for scales in flow.log_scales:
            scales.normal_(std=0.1)
        for shifts in flow.shifts:
            shifts.normal_(std=0.1)
        for coupling in flow.couplings:
            coupling.output_projection.weight.normal_(std=0.05)
            coupling.output_projection.bias.normal_(std=0.05)
    inputs = torch.randn(1, channels, 3, dtype=torch.double)
    mask = torch.ones(1, 1, 3, dtype=torch.double)
    transformed, log_det = flow(inputs, mask)
    restored, reverse_log_det = flow(transformed, mask, reverse=True)
    torch.testing.assert_close(restored, inputs, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(log_det + reverse_log_det, torch.zeros_like(log_det), atol=1e-9, rtol=1e-9)
    jacobian = torch.autograd.functional.jacobian(lambda value: flow(value.view_as(inputs), mask)[0].flatten(), inputs)
    _, expected_log_det = torch.linalg.slogdet(jacobian.reshape(inputs.numel(), inputs.numel()))
    torch.testing.assert_close(log_det[0], expected_log_det, atol=1e-9, rtol=1e-9)


def test_likelihood_includes_flow_jacobian_and_backpropagates():
    torch.manual_seed(21)
    aligner = TinyGlowTTSAligner(4, 6, hidden_channels=8, num_flows=2, mas_backend='cpu', flow_type='legacy').eval()
    audio = torch.randn(2, 7, 4)
    text = torch.randn(2, 1, 6)
    audio_lens = torch.tensor([7, 5])
    text_lens = torch.ones(2, dtype=torch.long)
    with torch.no_grad():
        aligner.flow.log_scales[0].fill_(0.2)
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
    assert aligner.flow.log_scales[0].grad.abs().sum() > 0
    torch.testing.assert_close(output.durations[:, 0], audio_lens)
    assert (output.token_indices[1, 5:] == -1).all()


def test_padding_cannot_change_valid_alignment_or_density():
    torch.manual_seed(22)
    aligner = TinyGlowTTSAligner(4, 6, hidden_channels=8, num_flows=2, mas_backend='cpu', flow_type='legacy').eval()
    with torch.no_grad():
        for coupling in aligner.flow.couplings:
            coupling.output_projection.weight.normal_(std=0.05)
    audio = torch.randn(1, 7, 4)
    text = torch.randn(1, 3, 6)
    audio_lens, text_lens = torch.tensor([7]), torch.tensor([3])
    unpadded = aligner(audio, text, audio_lens, text_lens)
    padded = aligner(
        torch.cat([audio, 100.0 * torch.randn(1, 4, 4)], dim=1),
        torch.cat([text, 100.0 * torch.randn(1, 3, 6)], dim=1),
        audio_lens,
        text_lens,
    )
    torch.testing.assert_close(padded.loss, unpadded.loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(padded.token_indices[:, :7], unpadded.token_indices)
    torch.testing.assert_close(padded.durations[:, :3], unpadded.durations)
    assert (padded.token_indices[:, 7:] == -1).all()
    assert (padded.durations[:, 3:] == 0).all()


def test_selected_frame_likelihood_and_gradients_match_dense_path_objective():
    torch.manual_seed(24)
    aligner = TinyGlowTTSAligner(4, 6, hidden_channels=8, num_flows=2, mas_backend='cpu', flow_type='legacy').eval()
    with torch.no_grad():
        for coupling in aligner.flow.couplings:
            coupling.output_projection.weight.normal_(std=0.05)
    reference = copy.deepcopy(aligner)
    audio = torch.randn(2, 8, 4, requires_grad=True)
    text = torch.randn(2, 3, 6, requires_grad=True)
    reference_audio = audio.detach().clone().requires_grad_(True)
    reference_text = text.detach().clone().requires_grad_(True)
    audio_lens, text_lens = torch.tensor([8, 6]), torch.tensor([3, 2])
    output = aligner(audio, text, audio_lens, text_lens)
    audio_mask = (torch.arange(8)[None, :] < audio_lens[:, None]).unsqueeze(1).float()
    text_mask = (torch.arange(3)[None, :] < text_lens[:, None]).unsqueeze(1).float()
    z, log_det = reference.flow(reference_audio.transpose(1, 2), audio_mask)
    means = reference.text_encoder(reference_text.transpose(1, 2) * text_mask, text_mask)
    dense_distance = (
        z.square().sum(dim=1).unsqueeze(2)
        + means.square().sum(dim=1).unsqueeze(1)
        - 2.0 * torch.bmm(z.transpose(1, 2), means)
    )
    # Random nonzero distances avoid the numerical clamp boundary in the MAS
    # scores; this checks the exact mathematical fixed-path gradients.
    valid_pairs = audio_mask[:, 0].bool().unsqueeze(2) & text_mask[:, 0].bool().unsqueeze(1)
    assert (dense_distance[valid_pairs] > 0).all()
    dense_log_probs = -0.5 * (dense_distance + 4 * math.log(2.0 * math.pi))
    selected = dense_log_probs.gather(2, output.token_indices.clamp_min(0).unsqueeze(2)).squeeze(2)
    expected = -((selected * audio_mask[:, 0]).sum() + log_det.sum()) / (audio_lens.sum() * 4)
    torch.testing.assert_close(output.loss, expected)
    output.loss.backward()
    expected.backward()
    torch.testing.assert_close(audio.grad, reference_audio.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(text.grad, reference_text.grad, atol=1e-6, rtol=1e-5)
    for actual, dense in zip(aligner.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, dense.grad, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('empty_audio,empty_text', [(False, False), (True, False), (False, True)])
def test_invalid_only_batches_are_finite_and_keep_all_parameter_gradients(empty_audio, empty_text):
    aligner = TinyGlowTTSAligner(4, 6, hidden_channels=8, num_flows=2, mas_backend='cpu', flow_type='legacy')
    audio = torch.randn(2, 0 if empty_audio else 3, 4)
    text = torch.randn(2, 0 if empty_text else 4, 6)
    audio_lens = torch.tensor([audio.size(1), 0])
    text_lens = torch.tensor([text.size(1), 0])
    output = aligner(audio, text, audio_lens, text_lens)
    assert output.loss == 0 and torch.isfinite(output.loss)
    assert not output.valid_turns.any()
    assert (output.token_indices == -1).all()
    assert (output.durations == 0).all()
    output.loss.backward()
    assert all(parameter.grad is not None and (parameter.grad == 0).all() for parameter in aligner.parameters())


def test_likelihood_recovers_acoustically_distinct_segments():
    class IdentityPrior(nn.Module):
        def forward(self, features, mask):
            return features * mask

    aligner = TinyGlowTTSAligner(4, 4, hidden_channels=8, num_flows=2, mas_backend='cpu', flow_type='legacy').eval()
    aligner.text_encoder = IdentityPrior()
    text = torch.tensor([[[-3.0] * 4, [0.0] * 4, [3.0] * 4]])
    expected_indices = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2]])
    audio = text[:, expected_indices[0]]
    output = aligner(audio, text, torch.tensor([9]), torch.tensor([3]))
    torch.testing.assert_close(output.token_indices, expected_indices)
    torch.testing.assert_close(output.durations, torch.tensor([[3, 2, 4]]))


def test_bfloat16_autocast_preserves_float32_likelihood():
    torch.manual_seed(23)
    aligner = TinyGlowTTSAligner(4, 6, hidden_channels=8, num_flows=2, mas_backend='cpu', flow_type='legacy').eval()
    audio, text = torch.randn(1, 7, 4), torch.randn(1, 3, 6)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = aligner(audio, text, torch.tensor([7]), torch.tensor([3]))
    assert output.loss.dtype == torch.float32
    assert torch.isfinite(output.loss)


def test_reference_defaults_do_not_create_a_vocabulary_embedding():
    aligner = TinyGlowTTSAligner(audio_channels=40, text_channels=1024)
    assert sum(parameter.numel() for parameter in aligner.parameters()) < 10_000_000
    assert not any(isinstance(module, nn.Embedding) for module in aligner.modules())
