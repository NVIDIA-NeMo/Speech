# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.tts.modules.glow_tts_aligner import TinyGlowTTSAligner

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('checkpoint_format', ['flat', 'lightning', 'lightning_omegaconf', 'bare', 'nemo'])
def test_restores_only_complete_aligner_state(tmp_path, checkpoint_format):
    model = _LoaderModel()
    before = _snapshot(model)
    config_before = OmegaConf.to_container(model.cfg)
    aligner = model.text_aligner
    expected = _replacement_state(aligner)
    full_state = _replacement_state(model)
    full_state.update({f'text_aligner.{key}': value for key, value in expected.items()})
    if checkpoint_format == 'bare':
        checkpoint_state = expected
    elif checkpoint_format in {'lightning', 'lightning_omegaconf'}:
        checkpoint_state = {'state_dict': full_state, 'global_step': 15000, 'optimizer_states': [{'state': {}}]}
        if checkpoint_format == 'lightning_omegaconf':
            checkpoint_state['hyper_parameters'] = {'cfg': OmegaConf.create({'hidden': 4, 'heads': [0, 1]})}
    else:
        checkpoint_state = full_state
    checkpoint = _save_checkpoint(tmp_path, checkpoint_state, archive=checkpoint_format == 'nemo')

    model.restore_text_speech_aligner_from_pretrained_checkpoint(checkpoint)

    assert model.text_aligner is aligner
    assert OmegaConf.to_container(model.cfg) == config_before
    assert model.use_attention_alignment is True
    assert model.use_aligned_text_head is False
    _assert_state_equal(aligner.state_dict(), expected)
    _assert_other_modules_unchanged(model, before)


@pytest.mark.parametrize('prefix', ['module.', 'model.', '_orig_mod.', 'module._orig_mod.model.'])
@pytest.mark.parametrize('bare', [False, True])
def test_accepts_known_checkpoint_wrappers(tmp_path, prefix, bare):
    model = _LoaderModel()
    expected = _replacement_state(model.text_aligner)
    source = expected if bare else {f'text_aligner.{key}': value for key, value in expected.items()}
    source = {prefix + key: value for key, value in source.items()}
    checkpoint = _save_checkpoint(tmp_path, {'state_dict': source})

    model.restore_text_speech_aligner_from_pretrained_checkpoint(checkpoint)

    _assert_state_equal(model.text_aligner.state_dict(), expected)


@pytest.mark.parametrize('invalid_kind', ['missing', 'unexpected', 'shape', 'non_tensor'])
def test_incompatible_aligner_fails_before_copying_any_weights(tmp_path, invalid_kind):
    model = _LoaderModel()
    before = _snapshot(model)
    source = {f'text_aligner.{key}': value for key, value in _replacement_state(model.text_aligner).items()}
    last_key = next(reversed(source))
    if invalid_kind == 'missing':
        del source[last_key]
        message = 'missing keys'
    elif invalid_kind == 'unexpected':
        source['text_aligner.invalid_layer.weight'] = torch.ones(1)
        message = 'unexpected keys'
    elif invalid_kind == 'shape':
        source[last_key] = torch.ones(source[last_key].shape + (1,))
        message = 'incompatible tensors'
    else:
        source[last_key] = 'not a tensor'
        message = 'expected a tensor'
    checkpoint = _save_checkpoint(tmp_path, source)

    with pytest.raises(ValueError, match=message):
        model.restore_text_speech_aligner_from_pretrained_checkpoint(checkpoint)

    _assert_state_equal(model.state_dict(), before)


@pytest.mark.parametrize('source_has_speaker', [False, True])
def test_speaker_conditioning_must_match_without_partial_loading(tmp_path, source_has_speaker):
    model = _LoaderModel(speaker=not source_has_speaker)
    before = _snapshot(model)
    source = _LoaderModel(speaker=source_has_speaker)
    checkpoint = _save_checkpoint(tmp_path, _replacement_state(source))

    with pytest.raises(ValueError, match='including speaker conditioning'):
        model.restore_text_speech_aligner_from_pretrained_checkpoint(checkpoint)

    _assert_state_equal(model.state_dict(), before)


def test_speaker_encoder_is_restored_with_the_rest_of_the_aligner(tmp_path):
    model = _LoaderModel(speaker=True)
    before = _snapshot(model)
    expected = _replacement_state(model.text_aligner)
    assert any(key.startswith('speaker_encoder.') for key in expected)
    checkpoint = _save_checkpoint(tmp_path, {'state_dict': {f'text_aligner.{k}': v for k, v in expected.items()}})

    model.restore_text_speech_aligner_from_pretrained_checkpoint(checkpoint)

    _assert_state_equal(model.text_aligner.state_dict(), expected)
    _assert_other_modules_unchanged(model, before)


@pytest.mark.parametrize('enabled', [False, True])
def test_none_checkpoint_leaves_the_model_unchanged(enabled):
    model = _LoaderModel(enabled=enabled)
    before = _snapshot(model)

    model.restore_text_speech_aligner_from_pretrained_checkpoint(None)

    _assert_state_equal(model.state_dict(), before)
    assert hasattr(model, 'text_aligner') is enabled


def test_loading_cannot_implicitly_enable_alignment(tmp_path):
    model = _LoaderModel(enabled=False)
    before = _snapshot(model)

    with pytest.raises(ValueError, match='requires use_aligned_text_head=true or use_attention_alignment=true'):
        model.restore_text_speech_aligner_from_pretrained_checkpoint(tmp_path / 'not_read.ckpt')

    _assert_state_equal(model.state_dict(), before)
    assert not hasattr(model, 'text_aligner')
    assert model.use_attention_alignment is False


@pytest.mark.parametrize('checkpoint_state', [[], {'state_dict': []}, {'decoder.weight': torch.ones(1)}])
def test_checkpoint_without_an_aligner_fails_clearly(tmp_path, checkpoint_state):
    model = _LoaderModel()
    before = _snapshot(model)
    checkpoint = _save_checkpoint(tmp_path, checkpoint_state)

    with pytest.raises(ValueError, match='state dictionary|no text_aligner weights'):
        model.restore_text_speech_aligner_from_pretrained_checkpoint(checkpoint)

    _assert_state_equal(model.state_dict(), before)


def test_ambiguous_wrapped_aligner_weights_are_rejected(tmp_path):
    model = _LoaderModel()
    before = _snapshot(model)
    source = {f'text_aligner.{key}': value for key, value in _replacement_state(model.text_aligner).items()}
    first_key = next(iter(source))
    source['module.' + first_key] = source[first_key].clone()
    checkpoint = _save_checkpoint(tmp_path, source)

    with pytest.raises(ValueError, match='duplicate unwrapped state key'):
        model.restore_text_speech_aligner_from_pretrained_checkpoint(checkpoint)

    _assert_state_equal(model.state_dict(), before)


def test_attention_alignment_remains_disabled_by_default():
    from tests.collections.tts.models.test_easy_magpietts import _make_easy_magpie_model, tiny_easy_magpie_cfg

    cfg = tiny_easy_magpie_cfg()
    assert 'use_attention_alignment' not in cfg
    model = _make_easy_magpie_model(cfg)
    assert model.use_attention_alignment is False
    assert not hasattr(model, 'text_aligner')
    assert not hasattr(model, 'aligned_text_head')
    repository = Path(__file__).resolve().parents[4]
    example_cfg = OmegaConf.load(repository / 'examples/tts/conf/magpietts/easy_magpietts_lhotse_multiturn.yaml')
    assert example_cfg.model.use_attention_alignment is False
    assert example_cfg.pretrained_model_text_speech_aligner is None


@pytest.mark.parametrize('aligner_checkpoint', [None, '/checkpoints/aligner.nemo'])
def test_entrypoint_loads_optional_aligner_after_full_initialization(monkeypatch, aligner_checkpoint):
    from examples.tts import easy_magpietts

    cfg = OmegaConf.create({'trainer': {}, 'model': {}, 'pretrained_model': '/checkpoints/base.nemo'})
    if aligner_checkpoint is not None:
        cfg.pretrained_model_text_speech_aligner = aligner_checkpoint
    events = []
    model = Mock()
    model.restore_from_pretrained_checkpoint.side_effect = lambda path: events.append(('full', path))
    model.maybe_init_from_pretrained_checkpoint.side_effect = lambda **kwargs: events.append(('maybe', kwargs['cfg']))
    model.restore_text_speech_aligner_from_pretrained_checkpoint.side_effect = lambda path: events.append(
        ('aligner', path)
    )
    trainer = Mock(callbacks=[])
    trainer.fit.side_effect = lambda fitted: events.append(('fit', fitted))
    monkeypatch.setattr(easy_magpietts.mp, 'set_start_method', Mock())
    monkeypatch.setattr(easy_magpietts.pl, 'Trainer', Mock(return_value=trainer))
    monkeypatch.setattr(easy_magpietts.pl.callbacks, 'LearningRateMonitor', Mock())
    monkeypatch.setattr(easy_magpietts, 'exp_manager', Mock())
    monkeypatch.setattr(easy_magpietts, 'EasyMagpieTTSModel', Mock(return_value=model))

    easy_magpietts.main.__wrapped__(cfg)

    expected = [('full', cfg.pretrained_model), ('maybe', cfg)]
    if aligner_checkpoint is not None:
        expected.append(('aligner', aligner_checkpoint))
        model.restore_text_speech_aligner_from_pretrained_checkpoint.assert_called_once_with(aligner_checkpoint)
    else:
        model.restore_text_speech_aligner_from_pretrained_checkpoint.assert_not_called()
    assert events == expected + [('fit', model)]


class _LoaderModel(nn.Module):
    """Exercise the real loader and aligner without constructing a tokenizer or decoder backbone."""

    restore_text_speech_aligner_from_pretrained_checkpoint = (
        EasyMagpieTTSInferenceModel.restore_text_speech_aligner_from_pretrained_checkpoint
    )

    def __init__(self, enabled=True, speaker=False):
        super().__init__()
        self.use_attention_alignment = enabled
        self.use_aligned_text_head = False
        self.cfg = OmegaConf.create({'use_attention_alignment': enabled, 'use_aligned_text_head': False})
        self.decoder = nn.Linear(4, 4)
        self.aligned_text_head = nn.Linear(4, 7)
        self.text_embedding = nn.Embedding(8, 4)
        if enabled:
            self.text_aligner = TinyGlowTTSAligner(
                audio_channels=4,
                text_channels=4,
                hidden_channels=4,
                num_flows=1,
                num_text_layers=1,
                coupling_num_layers=1,
                text_dropout=0.0,
                use_speaker_conditioning=speaker,
                speaker_hidden_channels=4,
                speaker_embedding_dim=4,
                speaker_num_layers=1,
            )


def _snapshot(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _replacement_state(module):
    return {key: torch.full_like(value, 7) for key, value in module.state_dict().items()}


def _assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=lambda message: f'{key}: {message}')


def _assert_other_modules_unchanged(model, before):
    actual = {key: value for key, value in model.state_dict().items() if not key.startswith('text_aligner.')}
    expected = {key: value for key, value in before.items() if not key.startswith('text_aligner.')}
    _assert_state_equal(actual, expected)


def _save_checkpoint(tmp_path, state, archive=False):
    weights = tmp_path / 'model_weights.ckpt'
    torch.save(state, weights)
    if not archive:
        return weights
    checkpoint = tmp_path / 'source.nemo'
    with tarfile.open(checkpoint, 'w:gz') as nemo_archive:
        nemo_archive.add(weights, arcname='model_weights.ckpt')
    return checkpoint
