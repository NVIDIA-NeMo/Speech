# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import io
import tarfile

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.modules.aligner import (
    TinyGlowTTSAligner,
    load_pretrained_text_speech_aligner,
    resolve_aligner_config,
)

pytestmark = pytest.mark.unit


def test_defaults_match_current_flow_and_text_encoder_architecture():
    assert resolve_aligner_config()['text_encoder_type'] == 'relative_transformer'
    assert resolve_aligner_config()['flow_type'] == 'sttts'
    assert resolve_aligner_config({})['text_encoder_type'] == 'relative_transformer'
    assert resolve_aligner_config({'hidden_channels': 8})['text_encoder_type'] == 'relative_transformer'
    assert resolve_aligner_config({'hidden_channels': 8})['flow_type'] == 'sttts'
    assert resolve_aligner_config()['use_speaker_conditioning'] is True
    with pytest.raises(ValueError, match='Unknown aligned_text_aligner'):
        resolve_aligner_config({'unexpected_option': True})


@pytest.mark.parametrize('option,value', [('flow_type', 'legacy'), ('text_encoder_type', 'cnn')])
def test_unsupported_checkpoint_architecture_is_rejected_explicitly(option, value):
    with pytest.raises(ValueError, match=option):
        resolve_aligner_config({option: value})


@pytest.mark.parametrize('prefix', ['', 'text_aligner.', 'model.module.text_aligner.', 'alignment.text_aligner.'])
@pytest.mark.parametrize('lightning', [False, True])
def test_only_aligner_weights_are_selected_from_checkpoint(tmp_path, prefix, lightning):
    source, config = _aligner()
    receiver, _ = _aligner()
    model = nn.Module()
    model.text_aligner = receiver
    model.backbone = nn.Linear(6, 6)
    backbone_before = copy.deepcopy(model.backbone.state_dict())
    state = {prefix + key: value.clone() for key, value in source.state_dict().items()}
    if prefix:
        state['backbone.weight'] = torch.full((6, 6), 123.0)
    checkpoint = {'state_dict': state, 'epoch': 4} if lightning else state
    path = tmp_path / 'source.ckpt'
    torch.save(checkpoint, path)
    original = copy.deepcopy(receiver.state_dict())
    receiver.eval()
    rng = torch.random.get_rng_state().clone()
    model.text_aligner, restored_config = load_pretrained_text_speech_aligner(receiver, path, aligner_config=config)
    assert model.text_aligner is not receiver
    assert not model.text_aligner.training
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert restored_config == config
    _assert_state_equal(source.state_dict(), model.text_aligner.state_dict())
    _assert_state_equal(original, receiver.state_dict())
    _assert_state_equal(backbone_before, model.backbone.state_dict())


def test_archive_keeps_receiver_architecture_and_speaker_configuration(tmp_path):
    source, source_config = _aligner(use_speaker_conditioning=True, hidden_channels=12)
    receiver, target_config = _aligner(use_speaker_conditioning=False)
    before = copy.deepcopy(receiver.state_dict())
    path = tmp_path / 'source.nemo'
    source_config = dict(source_config, audio_channels=4, text_channels=6)
    _archive(path, source, source_config)
    replacement, config = load_pretrained_text_speech_aligner(receiver, path, aligner_config=target_config)
    assert replacement.use_speaker_conditioning is False and replacement.speaker_encoder is None
    assert replacement.hidden_channels == 8
    assert config == target_config
    torch.testing.assert_close(
        replacement.text_encoder.input_projection.conv.weight,
        source.text_encoder.input_projection.conv.weight[:8],
        rtol=0,
        atol=0,
    )
    key = 'text_encoder.output_projection.conv.weight'
    torch.testing.assert_close(replacement.state_dict()[key], before[key], rtol=0, atol=0)


def test_lightning_metadata_cannot_disable_receiver_speaker_conditioning(tmp_path):
    source, source_config = _aligner(use_speaker_conditioning=False)
    receiver, target_config = _aligner()
    before = copy.deepcopy(receiver.state_dict())
    path = tmp_path / 'source.ckpt'
    torch.save(
        {
            'state_dict': {'module.text_aligner.' + key: value for key, value in source.state_dict().items()},
            'hyper_parameters': {'cfg': {'model': {'aligned_text_aligner': source_config}}},
        },
        path,
    )
    replacement, config = load_pretrained_text_speech_aligner(receiver, path, aligner_config=target_config)
    assert config == target_config and replacement.speaker_encoder is not None
    for key, value in replacement.state_dict().items():
        expected = source.state_dict().get(key, before[key])
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_archive_ignores_incompatible_configuration_metadata(tmp_path):
    source, source_config = _aligner()
    receiver, config = _aligner()
    before = copy.deepcopy(receiver.state_dict())
    path = tmp_path / 'wrong_dimensions.nemo'
    _archive(path, source, dict(source_config, audio_channels=8, flow_type='unsupported'))
    replacement, restored_config = load_pretrained_text_speech_aligner(receiver, path, aligner_config=config)
    assert replacement.audio_channels == 4 and restored_config == config
    _assert_state_equal(source.state_dict(), replacement.state_dict())
    _assert_state_equal(before, receiver.state_dict())


@pytest.mark.parametrize('failure', ['not_tensor', 'meta'])
def test_invalid_state_cannot_partially_overwrite_receiver(tmp_path, failure):
    source, config = _aligner()
    receiver, _ = _aligner()
    before = copy.deepcopy(receiver.state_dict())
    state = dict(source.state_dict())
    key = next(iter(state))
    if failure == 'meta':
        state[key] = torch.empty_like(state[key], device='meta')
    else:
        state[key] = 7
    path = tmp_path / 'incompatible.pt'
    torch.save({'state_dict': {'text_aligner.' + key: value for key, value in state.items()}}, path)
    rng = torch.random.get_rng_state().clone()
    message = 'meta aligner tensors' if failure == 'meta' else 'non-tensor aligner weights'
    with pytest.raises(ValueError, match=message):
        load_pretrained_text_speech_aligner(receiver, path, aligner_config=config)
    _assert_state_equal(before, receiver.state_dict())
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize('mismatch', ['shape', 'missing', 'unexpected', 'partial_rows'])
def test_partial_checkpoint_keeps_initialized_tensors_and_copies_compatible_weights(tmp_path, mismatch):
    source, config = _aligner()
    receiver, _ = _aligner()
    before = copy.deepcopy(receiver.state_dict())
    original = copy.deepcopy(source.state_dict())
    key = 'text_encoder.input_projection.conv.weight'
    if mismatch == 'shape':
        original[key] = torch.zeros(1)
    elif mismatch == 'missing':
        del original[key]
    elif mismatch == 'partial_rows':
        original[key] = original[key][:3].clone()
    else:
        original['unknown.weight'] = torch.zeros(1)
    path = tmp_path / 'partial.pt'
    torch.save({'state_dict': {'text_aligner.' + name: value for name, value in original.items()}}, path)
    rng = torch.random.get_rng_state().clone()
    replacement, restored_config = load_pretrained_text_speech_aligner(receiver, path, aligner_config=config)
    expected = before[key].clone()
    if mismatch == 'partial_rows':
        expected[:3] = original[key]
    elif mismatch == 'unexpected':
        expected = original[key]
    torch.testing.assert_close(replacement.state_dict()[key], expected, rtol=0, atol=0)
    other = 'text_encoder.output_projection.conv.weight'
    torch.testing.assert_close(replacement.state_dict()[other], original[other], rtol=0, atol=0)
    assert restored_config == config and torch.equal(torch.random.get_rng_state(), rng)
    _assert_state_equal(before, receiver.state_dict())


def test_eighty_channel_donor_loads_into_projected_ninety_six_channel_receiver(tmp_path):
    source, _ = _aligner(audio_channels=80)
    receiver, config = _aligner(audio_channels=80, input_audio_channels=96)
    projection = copy.deepcopy(receiver.audio_projection.state_dict())
    path = tmp_path / 'eighty_channel.pt'
    torch.save({'text_aligner.' + name: value for name, value in source.state_dict().items()}, path)
    replacement, restored_config = load_pretrained_text_speech_aligner(receiver, path, aligner_config=config)
    assert replacement.audio_channels == 80 and replacement.input_audio_channels == 96
    assert restored_config == config
    _assert_state_equal(projection, replacement.audio_projection.state_dict())
    for name, value in source.state_dict().items():
        torch.testing.assert_close(replacement.state_dict()[name], value, rtol=0, atol=0)
    output = replacement(torch.randn(1, 6, 96), torch.randn(1, 2, 6), torch.tensor([6]), torch.tensor([2]))
    assert torch.isfinite(output.loss)


def test_missing_aligner_and_ambiguous_namespaces_are_rejected(tmp_path):
    receiver, config = _aligner()
    path = tmp_path / 'missing.pt'
    torch.save({'state_dict': {'decoder.weight': torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match='no text_aligner weights'):
        load_pretrained_text_speech_aligner(receiver, path, aligner_config=config)
    key, value = next(iter(receiver.state_dict().items()))
    torch.save({'text_aligner.' + key: value, 'alignment.text_aligner.' + key: value}, path)
    with pytest.raises(ValueError, match='multiple text_aligner namespaces'):
        load_pretrained_text_speech_aligner(receiver, path, aligner_config=config)


def test_none_preserves_aligner_object_and_rng():
    receiver, config = _aligner()
    rng = torch.random.get_rng_state().clone()
    replacement, restored_config = load_pretrained_text_speech_aligner(receiver, None, aligner_config=config)
    assert replacement is receiver and restored_config == config
    assert torch.equal(torch.random.get_rng_state(), rng)


def _aligner(**overrides):
    values = dict(
        flow_type='sttts',
        hidden_channels=8,
        num_flows=1,
        num_text_layers=1,
        text_encoder_type='relative_transformer',
        text_num_heads=2,
        text_filter_channels=16,
        text_dropout=0.0,
        coupling_num_layers=1,
        coupling_kernel_size=3,
        flow_n_split=2,
        speaker_hidden_channels=4,
        speaker_embedding_dim=4,
        speaker_num_layers=1,
    )
    values.update(overrides)
    audio_channels = values.pop('audio_channels', 4)
    input_audio_channels = values.pop('input_audio_channels', None)
    config = resolve_aligner_config(values)
    dimensions = dict(audio_channels=audio_channels, text_channels=6)
    if input_audio_channels is not None:
        dimensions['input_audio_channels'] = input_audio_channels
    return TinyGlowTTSAligner(**dimensions, **config), config


def _archive(path, aligner, config):
    weights = io.BytesIO()
    torch.save({'text_aligner.' + key: value for key, value in aligner.state_dict().items()}, weights)
    yaml = OmegaConf.to_yaml(OmegaConf.create({'model': {'aligned_text_aligner': config}})).encode()
    with tarfile.open(path, 'w') as archive:
        for name, content in [('model_config.yaml', yaml), ('model_weights.ckpt', weights.getvalue())]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def _assert_state_equal(expected, actual):
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
