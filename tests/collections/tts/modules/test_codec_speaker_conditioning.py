# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.tts.modules.codec_speaker_encoder import CodecSpeakerEncoder
from nemo.collections.tts.modules.glow_tts_aligner import TinyGlowTTSAligner

pytestmark = pytest.mark.unit


def test_attentive_statistics_pooling_matches_known_weighted_mean_and_deviation():
    encoder = CodecSpeakerEncoder(2, hidden_channels=2, embedding_dim=4, num_layers=1).eval()
    encoder.norms[0] = nn.Identity()
    with torch.no_grad():
        encoder.input_projection.conv.weight.copy_(torch.eye(2).unsqueeze(2))
        encoder.input_projection.conv.bias.zero_()
        encoder.layers[0].conv.weight.zero_()
        encoder.layers[0].conv.bias.zero_()
        for projection in (encoder.attention[0], encoder.attention[2]):
            projection.weight.copy_(torch.eye(2).unsqueeze(2))
            projection.bias.zero_()
        encoder.output_projection.weight.copy_(torch.eye(4))
        encoder.output_projection.bias.zero_()
    content = torch.tensor([[[1.0, 1.0], [3.0, 2.0], [5.0, 8.0]]])
    padded = torch.cat([content, torch.full((1, 4, 2), 1000.0)], dim=1)
    weights = torch.softmax(content, dim=1)
    means = (content * weights).sum(dim=1)
    deviations = ((content.square() * weights).sum(dim=1) - means.square()).sqrt()
    expected = F.normalize(torch.cat([means, deviations], dim=1), dim=1)
    actual = encoder(padded, torch.tensor([3]))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_codec_speaker_encoder_ignores_padding_and_has_zero_padding_gradients():
    torch.manual_seed(304)
    encoder = CodecSpeakerEncoder(4, hidden_channels=8, embedding_dim=4, num_layers=2).eval()
    inputs = torch.randn(2, 10, 4, requires_grad=True)
    lengths = torch.tensor([4, 10])
    reference = encoder(inputs[:1, :4].detach(), torch.tensor([4]))
    embeddings = encoder(inputs, lengths)
    torch.testing.assert_close(embeddings[:1], reference, atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(2), atol=2e-6, rtol=2e-6)
    coefficients = torch.tensor([[1.0, -0.5, 0.2, 1.5], [0.1, 0.8, -0.9, 0.4]])
    (embeddings * coefficients).sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad[0, 4:].eq(0).all()
    assert inputs.grad[0, :4].abs().sum() > 0
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in encoder.parameters()
    )


@pytest.mark.parametrize('steps', [0, 7])
def test_empty_speaker_references_return_zero_with_finite_zero_gradients(steps):
    encoder = CodecSpeakerEncoder(4, hidden_channels=8, embedding_dim=4, num_layers=2)
    references = torch.randn(2, steps, 4, requires_grad=True)
    embeddings = encoder(references, torch.zeros(2, dtype=torch.long))
    assert embeddings.shape == (2, 4) and embeddings.eq(0).all()
    embeddings.sum().backward()
    assert all(parameter.grad is not None and parameter.grad.eq(0).all() for parameter in encoder.parameters())
    if references.grad is not None:
        assert references.grad.eq(0).all()


def test_mixed_valid_and_empty_speaker_references_are_independent():
    torch.manual_seed(305)
    encoder = CodecSpeakerEncoder(4, hidden_channels=8, embedding_dim=4, num_layers=2).eval()
    references = torch.randn(3, 9, 4, requires_grad=True)
    lengths = torch.tensor([5, 0, 9])
    expected = encoder(references[:1, :5].detach(), torch.tensor([5]))
    actual = encoder(references, lengths)
    torch.testing.assert_close(actual[:1], expected, atol=3e-6, rtol=2e-5)
    assert actual[1].eq(0).all()
    (actual * torch.arange(1.0, 5.0)).sum().backward()
    assert torch.isfinite(references.grad).all() and references.grad[1].eq(0).all()
    assert references.grad[0, 5:].eq(0).all()


def test_disabled_conditioning_preserves_checkpoint_layout_and_predictions():
    torch.manual_seed(300)
    original = _tiny(use_speaker_conditioning=False).eval()
    torch.manual_seed(300)
    default = _tiny().eval()
    assert not default.use_speaker_conditioning
    original_keys = set(original.state_dict())
    assert original_keys == set(default.state_dict())
    assert not any('speaker' in name or 'condition' in name for name in original_keys)
    default.load_state_dict(original.state_dict(), strict=True)
    inputs = _alignment_inputs()
    expected, actual = original(*inputs), default(*inputs)
    torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
    torch.testing.assert_close(actual.token_indices, expected.token_indices)


def test_missing_reference_is_zero_conditioning_without_target_audio_fallback():
    torch.manual_seed(301)
    conditioned = _tiny(use_speaker_conditioning=True).eval()
    plain = _tiny(use_speaker_conditioning=False).eval()
    base_keys = set(plain.state_dict())
    _open_coupling_outputs(conditioned)
    plain.load_state_dict({key: value for key, value in conditioned.state_dict().items() if key in base_keys})
    inputs = _alignment_inputs()
    expected, actual = plain(*inputs), conditioned(*inputs)
    torch.testing.assert_close(actual.loss, expected.loss, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.token_indices, expected.token_indices)
    actual.loss.backward()
    for parameter in conditioned.speaker_encoder.parameters():
        assert parameter.grad is not None and parameter.grad.eq(0).all()


def test_alignment_nll_trains_reference_encoder_and_flow_conditioning():
    torch.manual_seed(302)
    aligner = _tiny(use_speaker_conditioning=True).eval()
    _open_coupling_outputs(aligner)
    reference = torch.randn(2, 11, 4, requires_grad=True)
    output = aligner(
        *_alignment_inputs(),
        speaker_reference_features=reference,
        speaker_reference_lens=torch.tensor([11, 7]),
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    parameters = dict(aligner.named_parameters())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters.values()
    )
    assert any(parameter.grad.abs().sum() > 0 for parameter in aligner.speaker_encoder.parameters())
    assert any(parameter.grad.abs().sum() > 0 for parameter in aligner.speaker_encoder.attention.parameters())
    assert reference.grad is not None and torch.isfinite(reference.grad).all()
    assert reference.grad[1, 7:].eq(0).all()
    condition_parameters = [parameter for name, parameter in parameters.items() if 'condition' in name]
    assert condition_parameters and any(parameter.grad.abs().sum() > 0 for parameter in condition_parameters)


@pytest.mark.parametrize('empty_audio,empty_text', [(False, False), (True, False), (False, True)])
def test_invalid_alignment_turns_keep_every_speaker_parameter_in_zero_gradient_graph(empty_audio, empty_text):
    aligner = _tiny(use_speaker_conditioning=True)
    audio = torch.randn(2, 0 if empty_audio else 3, 4)
    text = torch.randn(2, 0 if empty_text else 4, 6)
    output = aligner(
        audio,
        text,
        torch.tensor([audio.size(1), 0]),
        torch.tensor([text.size(1), 0]),
        speaker_reference_features=torch.randn(2, 7, 4),
        speaker_reference_lens=torch.tensor([7, 5]),
    )
    assert output.loss == 0 and torch.isfinite(output.loss)
    output.loss.backward()
    assert all(parameter.grad is not None and parameter.grad.eq(0).all() for parameter in aligner.parameters())


def test_enabled_conditioning_checkpoint_roundtrip_preserves_path_and_likelihood():
    torch.manual_seed(303)
    original = _tiny(use_speaker_conditioning=True).eval()
    _open_coupling_outputs(original)
    restored = _tiny(use_speaker_conditioning=True).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    kwargs = dict(speaker_reference_features=torch.randn(2, 9, 4), speaker_reference_lens=torch.tensor([9, 6]))
    inputs = _alignment_inputs()
    expected, actual = original(*inputs, **kwargs), restored(*inputs, **kwargs)
    torch.testing.assert_close(actual.loss, expected.loss)
    torch.testing.assert_close(actual.token_indices, expected.token_indices)


def test_prepooled_reference_matches_direct_reference_and_retains_encoder_gradients():
    torch.manual_seed(308)
    aligner = _tiny(use_speaker_conditioning=True).eval()
    _open_coupling_outputs(aligner)
    features, lengths = torch.randn(2, 10, 4), torch.tensor([10, 6])
    inputs = _alignment_inputs()
    direct = aligner(*inputs, speaker_reference_features=features, speaker_reference_lens=lengths)
    embeddings = aligner.encode_speaker_reference(features, lengths)
    assert embeddings.shape == (2, 4) and embeddings.requires_grad
    pooled = aligner(*inputs, speaker_embeddings=embeddings)
    torch.testing.assert_close(pooled.loss, direct.loss, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(pooled.token_indices, direct.token_indices)
    pooled.loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in aligner.speaker_encoder.parameters()
    )
    assert any(parameter.grad.abs().sum() > 0 for parameter in aligner.speaker_encoder.parameters())


def test_prepooled_and_raw_reference_arguments_are_mutually_exclusive():
    aligner = _tiny(use_speaker_conditioning=True)
    with pytest.raises(ValueError):
        aligner(
            *_alignment_inputs(),
            speaker_embeddings=torch.randn(2, 4),
            speaker_reference_features=torch.randn(2, 8, 4),
            speaker_reference_lens=torch.tensor([8, 6]),
        )


def test_reference_speaker_embedding_changes_flow_after_identity_initialization():
    torch.manual_seed(306)
    aligner = _tiny(use_speaker_conditioning=True).eval()
    _open_coupling_outputs(aligner)
    audio = torch.randn(1, 4, 8)
    mask = torch.ones(1, 1, 8)
    first, _ = aligner.flow(audio, mask, speaker_embedding=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    second, _ = aligner.flow(audio, mask, speaker_embedding=torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    assert (first - second).abs().max() > 1e-5


def test_conditioned_flow_inverse_and_jacobian_hold_for_fixed_speaker():
    torch.manual_seed(307)
    aligner = _tiny(use_speaker_conditioning=True)
    _open_coupling_outputs(aligner)
    flow = aligner.flow.double()
    inputs = torch.randn(1, 4, 2, dtype=torch.double)
    mask = torch.ones(1, 1, 2, dtype=torch.double)
    speaker = torch.randn(1, 4, dtype=torch.double)
    encoded, logdet = flow(inputs, mask, speaker_embedding=speaker)
    restored, inverse_logdet = flow(encoded, mask, reverse=True, speaker_embedding=speaker)
    torch.testing.assert_close(restored, inputs, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(logdet + inverse_logdet, torch.zeros_like(logdet), atol=1e-8, rtol=1e-8)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: flow(value, mask, speaker_embedding=speaker)[0].flatten(), inputs
    )
    _, numerical_logdet = torch.linalg.slogdet(jacobian.reshape(inputs.numel(), inputs.numel()))
    torch.testing.assert_close(logdet[0], numerical_logdet, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize('missing', ['features', 'lengths'])
def test_partial_speaker_reference_arguments_raise(missing):
    aligner = _tiny(use_speaker_conditioning=True)
    kwargs = dict(speaker_reference_features=torch.randn(2, 7, 4), speaker_reference_lens=torch.tensor([7, 4]))
    del kwargs['speaker_reference_features' if missing == 'features' else 'speaker_reference_lens']
    with pytest.raises(ValueError):
        aligner(*_alignment_inputs(), **kwargs)


@pytest.mark.parametrize(
    'invalid',
    [
        {'use_speaker_conditioning': 1},
        {'use_speaker_conditioning': 'true'},
        {'speaker_hidden_channels': 0},
        {'speaker_hidden_channels': True},
        {'speaker_embedding_dim': 0},
        {'speaker_embedding_dim': 2.5},
        {'speaker_num_layers': -1},
        {'speaker_num_layers': True},
        {'flow_type': 'legacy'},
    ],
)
def test_invalid_speaker_conditioning_configurations_raise(invalid):
    kwargs = dict(use_speaker_conditioning=True)
    kwargs.update(invalid)
    with pytest.raises(ValueError):
        _tiny(**kwargs)


def _tiny(**overrides):
    kwargs = dict(
        audio_channels=4,
        text_channels=6,
        hidden_channels=8,
        num_flows=2,
        num_text_layers=1,
        coupling_num_layers=2,
        coupling_kernel_size=3,
        text_dropout=0.0,
        dequantization_noise=0.0,
        speaker_hidden_channels=8,
        speaker_embedding_dim=4,
        speaker_num_layers=2,
        mas_backend='cpu',
    )
    kwargs.update(overrides)
    return TinyGlowTTSAligner(**kwargs)


def _alignment_inputs():
    return torch.randn(2, 9, 4), torch.randn(2, 3, 6), torch.tensor([9, 7]), torch.tensor([3, 2])


def _open_coupling_outputs(aligner):
    # Couplings deliberately start as identity; expose the conditional path to
    # test gradients beyond the initially zero final projection.
    with torch.no_grad():
        for coupling in aligner.flow.couplings:
            coupling.output_projection.weight.normal_(mean=0.0, std=0.05)
            coupling.output_projection.bias.normal_(mean=0.0, std=0.02)
