# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import math

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.glow_tts_aligner import TinyGlowTTSAligner
from nemo.collections.tts.modules.relative_position_transformer import RelativePositionTransformer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('steps,window', [(3, 4), (11, 4), (5, 0)])
def test_relative_key_and_value_attention_matches_dense_offset_formula_and_gradients(steps, window):
    torch.manual_seed(210 + steps)
    encoder = RelativePositionTransformer(
        hidden_channels=8, filter_channels=32, n_heads=2, n_layers=1, p_dropout=0.0, window_size=window
    ).eval()
    attention = encoder.attn_layers[0]
    reference = copy.deepcopy(attention)
    inputs = torch.randn(2, 8, steps, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    actual = attention(inputs, torch.ones(2, 1, steps))
    expected = _dense_relative_attention(reference, reference_inputs)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, atol=3e-6, rtol=2e-5)
    for actual_parameter, expected_parameter in zip(attention.parameters(), reference.parameters()):
        assert actual_parameter.grad is not None and torch.isfinite(actual_parameter.grad).all()
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, atol=4e-6, rtol=3e-5)
    assert attention.emb_rel_k.grad.abs().sum() > 0
    assert attention.emb_rel_v.grad.abs().sum() > 0


def test_text_prior_reads_past_and_future_beyond_convolution_receptive_field():
    torch.manual_seed(200)
    prior = _small_aligner().text_encoder.eval()
    features = torch.randn(1, 6, 17, requires_grad=True)
    mask = torch.ones(1, 1, 17)
    means = prior(features, mask)
    first_gradient = torch.autograd.grad(means[:, :, 0].square().sum(), features, retain_graph=True)[0]
    last_gradient = torch.autograd.grad(means[:, :, -1].square().sum(), features)[0]
    assert first_gradient[:, :, -1].abs().sum() > 1e-7
    assert last_gradient[:, :, 0].abs().sum() > 1e-7


def test_relative_prior_is_unchanged_by_masked_prefix_and_suffix():
    torch.manual_seed(201)
    prior = _small_aligner().text_encoder.eval()
    features = torch.randn(1, 6, 7)
    expected = prior(features, torch.ones(1, 1, 7))
    padded = torch.cat([100 * torch.randn(1, 6, 5), features, 100 * torch.randn(1, 6, 3)], dim=2)
    mask = torch.zeros(1, 1, 15)
    mask[:, :, 5:12] = 1
    actual = prior(padded, mask)
    torch.testing.assert_close(actual[:, :, 5:12], expected, atol=3e-6, rtol=2e-5)
    assert actual[:, :, :5].eq(0).all()
    assert actual[:, :, 12:].eq(0).all()


def test_mixed_lengths_and_empty_rows_preserve_density_inputs_and_finite_gradients():
    torch.manual_seed(202)
    prior = _small_aligner().text_encoder.eval()
    features = torch.randn(3, 6, 11, requires_grad=True)
    lengths = torch.tensor([4, 11, 0])
    mask = (torch.arange(11)[None] < lengths[:, None]).unsqueeze(1).float()
    expected = prior(features[:1, :, :4].detach(), torch.ones(1, 1, 4))
    means = prior(features, mask)
    torch.testing.assert_close(means[:1, :, :4], expected, atol=3e-6, rtol=2e-5)
    assert means[0, :, 4:].eq(0).all()
    assert means[2].eq(0).all()
    means.square().sum().backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad[0, :, 4:].eq(0).all()
    assert features.grad[2].eq(0).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in prior.parameters())


def test_transformer_prior_remains_small_and_has_no_second_vocabulary_embedding():
    aligner = TinyGlowTTSAligner(audio_channels=80, text_channels=1536)
    assert aligner.text_encoder_type == 'relative_transformer'
    assert aligner.num_text_layers == 2
    assert isinstance(aligner.text_encoder.encoder, RelativePositionTransformer)
    assert len(aligner.text_encoder.encoder.attn_layers) == 2
    assert all(layer.n_heads == 2 and layer.window_size == 4 for layer in aligner.text_encoder.encoder.attn_layers)
    assert sum(parameter.numel() for parameter in aligner.text_encoder.parameters()) < 2_000_000
    assert not any(isinstance(module, nn.Embedding) for module in aligner.text_encoder.modules())


def test_alignment_likelihood_trains_relative_key_and_value_tables():
    torch.manual_seed(204)
    aligner = _small_aligner().eval()
    output = aligner(torch.randn(2, 9, 4), torch.randn(2, 3, 6), torch.tensor([9, 7]), torch.tensor([3, 2]))
    output.loss.backward()
    for layer in aligner.text_encoder.encoder.attn_layers:
        assert layer.emb_rel_k.grad is not None and layer.emb_rel_k.grad.abs().sum() > 0
        assert layer.emb_rel_v.grad is not None and layer.emb_rel_v.grad.abs().sum() > 0


@pytest.mark.parametrize('flow_type', ['legacy', 'sttts'])
def test_cnn_prior_preserves_old_checkpoint_layout_for_both_flows(flow_type):
    kwargs = dict(
        audio_channels=4,
        text_channels=6,
        hidden_channels=8,
        num_flows=2,
        num_text_layers=2,
        coupling_num_layers=2,
        flow_type=flow_type,
        text_encoder_type='cnn',
        mas_backend='cpu',
    )
    original = TinyGlowTTSAligner(**kwargs).eval()
    state = original.state_dict()
    assert 'text_encoder.layers.0.conv.weight' in state
    assert 'text_encoder.norms.0.weight' in state
    assert not any('relative' in key or 'attention' in key for key in state)
    restored = TinyGlowTTSAligner(**kwargs).eval()
    restored.load_state_dict(state, strict=True)
    audio, text = torch.randn(1, 8, 4), torch.randn(1, 3, 6)
    expected = original(audio, text, torch.tensor([8]), torch.tensor([3]))
    actual = restored(audio, text, torch.tensor([8]), torch.tensor([3]))
    torch.testing.assert_close(actual.loss, expected.loss)
    torch.testing.assert_close(actual.token_indices, expected.token_indices)


def test_transformer_prior_checkpoint_roundtrip_preserves_likelihood_and_path():
    torch.manual_seed(203)
    original = _small_aligner().eval()
    restored = _small_aligner().eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    audio, text = torch.randn(2, 9, 4), torch.randn(2, 3, 6)
    lengths = torch.tensor([9, 7]), torch.tensor([3, 2])
    expected, actual = original(audio, text, *lengths), restored(audio, text, *lengths)
    torch.testing.assert_close(actual.loss, expected.loss)
    torch.testing.assert_close(actual.token_indices, expected.token_indices)


@pytest.mark.parametrize(
    'invalid',
    [
        {'text_encoder_type': 'unknown'},
        {'text_num_heads': 0},
        {'text_num_heads': -1},
        {'text_num_heads': True},
        {'text_num_heads': 3},
        {'text_filter_channels': 0},
        {'text_filter_channels': 1.5},
        {'text_kernel_size': 0},
        {'text_kernel_size': 2},
        {'text_dropout': -0.1},
        {'text_dropout': 1.0},
        {'text_dropout': float('nan')},
        {'text_dropout': float('inf')},
        {'text_rel_attn_window_size': -1},
        {'text_rel_attn_window_size': True},
    ],
)
def test_invalid_transformer_configuration_raises_value_error(invalid):
    with pytest.raises(ValueError):
        _small_aligner(**invalid)


def _small_aligner(**overrides):
    kwargs = dict(
        audio_channels=4,
        text_channels=6,
        hidden_channels=8,
        num_flows=2,
        num_text_layers=2,
        coupling_num_layers=2,
        coupling_kernel_size=3,
        text_encoder_type='relative_transformer',
        text_num_heads=2,
        text_dropout=0.0,
        mas_backend='cpu',
    )
    kwargs.update(overrides)
    return TinyGlowTTSAligner(**kwargs)


def _dense_relative_attention(attention, inputs):
    """Direct j-i table lookup, independent of relative-to-absolute reshaping."""
    batch, channels, steps = inputs.shape
    heads = attention.n_heads
    head_channels = channels // heads

    def project(layer):
        return layer(inputs).reshape(batch, heads, head_channels, steps).transpose(2, 3)

    query, key, value = project(attention.conv_q), project(attention.conv_k), project(attention.conv_v)
    positions = torch.arange(steps, device=inputs.device)
    offsets = positions[None, :] - positions[:, None]
    window = attention.window_size
    indices = offsets.clamp(-window, window) + window
    inside_window = offsets.abs().le(window).unsqueeze(-1)
    relative_keys = attention.emb_rel_k[0, indices] * inside_window
    relative_values = attention.emb_rel_v[0, indices] * inside_window
    scores = torch.matmul(query, key.transpose(2, 3))
    scores = scores + torch.einsum('bhid,ijd->bhij', query, relative_keys)
    probabilities = torch.softmax(scores / math.sqrt(head_channels), dim=-1)
    values = torch.matmul(probabilities, value)
    values = values + torch.einsum('bhij,ijd->bhid', probabilities, relative_values)
    return attention.conv_o(values.transpose(2, 3).reshape(batch, channels, steps))
