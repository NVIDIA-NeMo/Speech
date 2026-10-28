# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.glow_tts_aligner import TinyGlowTTSAligner, _STTTSFlow

pytestmark = pytest.mark.unit


def test_reference_flow_has_published_capacity_and_no_duplicate_vocabulary():
    aligner = TinyGlowTTSAligner(audio_channels=80, text_channels=1536)
    assert aligner.flow_type == 'sttts'
    assert len(aligner.flow.couplings) == len(aligner.flow.mixers) == len(aligner.flow.actnorms) == 6
    assert not any(isinstance(module, nn.Embedding) for module in aligner.modules())
    for coupling in aligner.flow.couplings:
        assert coupling.input_projection.conv.out_channels == 128
        assert len(coupling.in_layers) == len(coupling.res_skip_layers) == 4
        assert all(layer.conv.kernel_size == (5,) and layer.conv.dilation == (1,) for layer in coupling.in_layers)
        assert all(layer.conv.out_channels == 256 for layer in coupling.in_layers)
        assert [layer.conv.out_channels for layer in coupling.res_skip_layers] == [256, 256, 256, 128]
        assert hasattr(coupling.input_projection.conv, 'weight_g')
        assert all(hasattr(layer.conv, 'weight_g') for layer in coupling.in_layers)
        assert all(hasattr(layer.conv, 'weight_g') for layer in coupling.res_skip_layers)
        assert coupling.output_projection.weight.eq(0).all()
        assert coupling.output_projection.bias.eq(0).all()


@pytest.mark.parametrize('channels', [4, 5, 8])
def test_flow_inverse_and_full_jacobian_include_learned_mixing(channels):
    torch.manual_seed(110 + channels)
    flow = _small_flow(channels).double()
    _perturb(flow)
    inputs = torch.randn(1, channels, 2, dtype=torch.double)
    mask = torch.ones(1, 1, 2, dtype=torch.double)
    encoded, logdet = flow(inputs, mask)
    restored, inverse_logdet = flow(encoded, mask, reverse=True)
    torch.testing.assert_close(restored, inputs, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(logdet + inverse_logdet, torch.zeros_like(logdet), atol=1e-8, rtol=1e-8)
    jacobian = torch.autograd.functional.jacobian(lambda value: flow(value, mask)[0].flatten(), inputs)
    _, numerical_logdet = torch.linalg.slogdet(jacobian.reshape(inputs.numel(), inputs.numel()))
    torch.testing.assert_close(logdet[0], numerical_logdet, atol=1e-8, rtol=1e-8)


def test_grouped_mixing_crosses_coupling_halves():
    flow = _small_flow(8)
    mixer = flow.mixers[0]
    assert mixer.n_split == 4
    with torch.no_grad():
        # Swap two pairs of rows: determinant +1 and mixing across the two
        # coupling halves, rather than independently inside contiguous groups.
        mixer.weight.copy_(torch.eye(4)[torch.tensor([2, 3, 0, 1])])
    inputs = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    mixed, logdet = mixer(inputs, torch.ones(1, 1, 1))
    expected = torch.cat([inputs[:, 4:], inputs[:, :4]], dim=1)
    torch.testing.assert_close(mixed, expected)
    torch.testing.assert_close(logdet, torch.zeros_like(logdet))


@pytest.mark.parametrize('channels', [5, 8])
def test_mixing_jacobian_counts_repeated_groups_and_valid_frames(channels):
    mixer = _small_flow(channels).mixers[0].double()
    diagonal = torch.linspace(0.8, 1.4, mixer.weight.size(0), dtype=torch.double)
    with torch.no_grad():
        mixer.weight.copy_(torch.diag(diagonal))
    inputs = torch.randn(2, channels, 4, dtype=torch.double)
    lengths = torch.tensor([4, 2])
    mask = (torch.arange(4)[None] < lengths[:, None]).unsqueeze(1).double()
    mixed, logdet = mixer(inputs, mask)
    expected = diagonal.log().sum() * (channels / diagonal.numel()) * lengths
    torch.testing.assert_close(logdet, expected)
    restored, reverse_logdet = mixer(mixed, mask, reverse=True)
    torch.testing.assert_close(restored, inputs * mask, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(logdet + reverse_logdet, torch.zeros_like(logdet), atol=1e-9, rtol=1e-9)


def test_masked_padding_cannot_change_density_or_alignment():
    torch.manual_seed(120)
    aligner = _small_aligner().eval()
    _perturb(aligner.flow)
    audio, text = torch.randn(1, 7, 4), torch.randn(1, 3, 6)
    audio_lens, text_lens = torch.tensor([7]), torch.tensor([3])
    reference = aligner(audio, text, audio_lens, text_lens)
    padded = aligner(
        torch.cat([audio, 100 * torch.randn(1, 5, 4)], dim=1),
        torch.cat([text, 100 * torch.randn(1, 4, 6)], dim=1),
        audio_lens,
        text_lens,
    )
    torch.testing.assert_close(padded.loss, reference.loss, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(padded.token_indices[:, :7], reference.token_indices)
    torch.testing.assert_close(padded.durations[:, :3], reference.durations)
    assert padded.token_indices[:, 7:].eq(-1).all()
    assert padded.durations[:, 3:].eq(0).all()


def test_selected_path_likelihood_and_gradients_match_dense_gaussian_objective():
    torch.manual_seed(121)
    aligner = _small_aligner().eval()
    _perturb(aligner.flow)
    reference = _small_aligner().eval()
    reference.load_state_dict(aligner.state_dict())
    audio = torch.randn(2, 8, 4, requires_grad=True)
    text = torch.randn(2, 3, 6, requires_grad=True)
    reference_audio = audio.detach().clone().requires_grad_(True)
    reference_text = text.detach().clone().requires_grad_(True)
    audio_lens, text_lens = torch.tensor([8, 6]), torch.tensor([3, 2])
    output = aligner(audio, text, audio_lens, text_lens)
    audio_mask = (torch.arange(8)[None] < audio_lens[:, None]).unsqueeze(1).float()
    text_mask = (torch.arange(3)[None] < text_lens[:, None]).unsqueeze(1).float()
    z, logdet = reference.flow(reference_audio.transpose(1, 2), audio_mask)
    means = reference.text_encoder(reference_text.transpose(1, 2) * text_mask, text_mask)
    dense_distance = (
        z.square().sum(dim=1).unsqueeze(2)
        + means.square().sum(dim=1).unsqueeze(1)
        - 2 * torch.bmm(z.transpose(1, 2), means)
    )
    dense_log_probability = -0.5 * (dense_distance + 4 * math.log(2 * math.pi))
    selected = dense_log_probability.gather(2, output.token_indices.clamp_min(0).unsqueeze(2)).squeeze(2)
    expected = -((selected * audio_mask[:, 0]).sum() + logdet.sum()) / (audio_lens.sum() * 4)
    torch.testing.assert_close(output.loss, expected, atol=2e-6, rtol=2e-6)
    output.loss.backward()
    expected.backward()
    torch.testing.assert_close(audio.grad, reference_audio.grad, atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(text.grad, reference_text.grad, atol=3e-6, rtol=2e-5)
    for actual, dense in zip(aligner.parameters(), reference.parameters()):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, dense.grad, atol=3e-6, rtol=2e-5)
    coupling = aligner.flow.couplings[0]
    assert coupling.input_projection.conv.weight_v.grad.abs().sum() > 0
    assert all(layer.conv.weight_v.grad.abs().sum() > 0 for layer in coupling.in_layers)
    assert all(layer.conv.weight_v.grad.abs().sum() > 0 for layer in coupling.res_skip_layers)


@pytest.mark.parametrize('empty_audio,empty_text', [(False, False), (True, False), (False, True)])
def test_invalid_only_batches_keep_zero_gradients_for_every_reference_parameter(empty_audio, empty_text):
    aligner = _small_aligner()
    audio = torch.randn(2, 0 if empty_audio else 3, 4)
    text = torch.randn(2, 0 if empty_text else 4, 6)
    output = aligner(
        audio,
        text,
        torch.tensor([audio.size(1), 0]),
        torch.tensor([text.size(1), 0]),
    )
    assert output.loss == 0 and torch.isfinite(output.loss)
    assert not output.valid_turns.any()
    assert output.token_indices.eq(-1).all()
    assert output.durations.eq(0).all()
    output.loss.backward()
    assert all(parameter.grad is not None and parameter.grad.eq(0).all() for parameter in aligner.parameters())


def test_reference_profile_preserves_audio_grid_under_bfloat16_amp():
    aligner = _small_aligner().eval()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = aligner(torch.randn(2, 9, 4), torch.randn(2, 3, 6), torch.tensor([9, 7]), torch.tensor([3, 2]))
    assert output.loss.dtype == torch.float32 and torch.isfinite(output.loss)
    assert output.token_indices.shape == (2, 9)
    torch.testing.assert_close(output.durations.sum(dim=1), torch.tensor([9, 7]))
    assert output.token_indices[1, 7:].eq(-1).all()


def test_explicit_legacy_profile_restores_original_state_keys_and_predictions():
    kwargs = dict(
        audio_channels=4,
        text_channels=6,
        hidden_channels=8,
        num_flows=2,
        flow_type='legacy',
        text_encoder_type='cnn',
    )
    original = TinyGlowTTSAligner(**kwargs).eval()
    state = original.state_dict()
    assert 'flow.shifts.0' in state and 'flow.log_scales.0' in state
    assert 'flow.couplings.0.hidden_projection.conv.weight' in state
    assert not any(key.startswith(('flow.mixers.', 'flow.actnorms.')) for key in state)
    restored = TinyGlowTTSAligner(**kwargs).eval()
    restored.load_state_dict(state, strict=True)
    audio, text = torch.randn(1, 7, 4), torch.randn(1, 3, 6)
    lengths = torch.tensor([7]), torch.tensor([3])
    expected, actual = original(audio, text, *lengths), restored(audio, text, *lengths)
    torch.testing.assert_close(actual.loss, expected.loss)
    torch.testing.assert_close(actual.token_indices, expected.token_indices)


def _small_flow(channels):
    return _STTTSFlow(
        channels,
        hidden_channels=8,
        num_flows=2,
        coupling_num_layers=2,
        coupling_kernel_size=3,
    )


def _small_aligner():
    return TinyGlowTTSAligner(
        audio_channels=4,
        text_channels=6,
        hidden_channels=8,
        num_flows=2,
        num_text_layers=1,
        coupling_num_layers=2,
        coupling_kernel_size=3,
        mas_backend='cpu',
        flow_type='sttts',
        text_dropout=0.0,
    )


def _perturb(module):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
