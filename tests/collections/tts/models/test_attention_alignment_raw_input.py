# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.models import easy_magpietts
from tests.collections.tts.models.test_easy_magpietts_attention_alignment import (
    TinyPhonemeTokenizer,
    _config,
    _make_attention_model,
)

pytestmark = pytest.mark.unit

_UNSET = object()


@pytest.mark.parametrize('factory', ['legacy', 'lhotse_single', 'lhotse_multiturn'])
@pytest.mark.parametrize('mode', ['train', 'val'])
@pytest.mark.parametrize('force', [False, True])
def test_raw_input_policy_reaches_each_factory_without_disabling_model_phoneme_capacity(factory, mode, force):
    model = _model(force=force, multiturn=factory == 'lhotse_multiturn')
    config = _dataset_config(normalized=1.0)
    config_before = OmegaConf.to_container(config)
    vocab_size = model.text_vocab_size
    phoneme_offset = model.text_phoneme_token_offset

    dataset, kwargs = _factory_call(model, config, factory, mode)

    assert kwargs['load_normalized_text_percent'] == (0.0 if force else 1.0)
    assert kwargs['enable_phoneme_text_input'] is (not force)
    if factory == 'lhotse_multiturn':
        assert kwargs['force_raw_text_input'] is force
    else:
        assert dataset.force_raw_text_input is force
    assert OmegaConf.to_container(config) == config_before
    assert model.enable_phoneme_text_input is True
    assert model.cfg.enable_phoneme_text_input is True
    assert model.text_vocab_size == vocab_size
    assert model.text_phoneme_token_offset == phoneme_offset
    assert model.text_phoneme_vocab_size == TinyPhonemeTokenizer.vocab_size
    assert model.partial_phoneme_text_prob == 0.5


@pytest.mark.parametrize('factory', ['legacy', 'lhotse_single', 'lhotse_multiturn'])
def test_force_raw_setting_does_not_activate_when_attention_alignment_is_disabled(factory):
    model = _model(force=True, enabled=False, multiturn=factory == 'lhotse_multiturn')

    dataset, kwargs = _factory_call(model, _dataset_config(normalized=1.0), factory, 'train')

    assert kwargs['load_normalized_text_percent'] == 1.0
    assert kwargs['enable_phoneme_text_input'] is True
    assert model._force_raw_attention_text() is False
    if factory == 'lhotse_multiturn':
        assert kwargs['force_raw_text_input'] is False
    else:
        assert dataset.force_raw_text_input is False
    assert not hasattr(model, 'text_aligner')
    assert not hasattr(model, 'aligned_text_head')


@pytest.mark.parametrize('mode', ['train', 'val'])
@pytest.mark.parametrize('normalized', [0.0, 0.35, 1.0])
def test_legacy_factory_forwards_explicit_top_level_normalization_probability(mode, normalized):
    model = _model()
    config = _dataset_config(normalized=normalized)
    config.dataset.load_normalized_text_percent = 0.7

    _, kwargs = _factory_call(model, config, 'legacy', mode)

    assert kwargs['load_normalized_text_percent'] == normalized
    assert kwargs['enable_phoneme_text_input'] is True
    assert config.dataset.load_normalized_text_percent == 0.7
    assert 'force_raw_text_input' not in kwargs


def test_legacy_factory_preserves_nested_defaults_when_no_top_level_option_is_provided():
    model = _model()
    config = _dataset_config()
    config.dataset.load_normalized_text_percent = 0.65
    before = OmegaConf.to_container(config)

    _, kwargs = _factory_call(model, config, 'legacy', 'val')

    assert 'load_normalized_text_percent' not in kwargs
    assert 'force_raw_text_input' not in kwargs
    assert OmegaConf.to_container(config) == before


def test_force_raw_defaults_to_false_and_preserves_existing_dataset_behavior():
    model = _model()

    _, kwargs = _factory_call(model, _dataset_config(normalized=0.8), 'legacy', 'train')

    assert model.attention_alignment_config['force_raw_text'] is False
    assert model._force_raw_attention_text() is False
    assert kwargs['load_normalized_text_percent'] == 0.8
    assert kwargs['enable_phoneme_text_input'] is True


@pytest.mark.parametrize('value', ['false', 0, 1, None])
def test_force_raw_requires_an_actual_boolean(value):
    with pytest.raises(ValueError, match='attention_alignment.force_raw_text'):
        _model(force=value)


def test_real_multiturn_dataset_forces_raw_selection_and_disables_partial_replacement():
    model = _model(force=True, multiturn=True)
    result = object()
    with (
        patch.object(easy_magpietts, 'FallbackDataset', side_effect=lambda value: value),
        patch.object(easy_magpietts, 'get_lhotse_dataloader_from_config', return_value=result) as loader,
    ):
        assert model.get_lhotse_dataloader(_dataset_config(normalized=1.0), mode='train') is result

    dataset = loader.call_args.kwargs['dataset']
    assert isinstance(dataset, easy_magpietts.MagpieTTSLhotseMultiturnDataset)
    assert dataset.force_raw_text_input is True
    assert dataset.load_normalized_text_percent == 0.0
    assert dataset.enable_phoneme_text_input is False
    assert dataset.partial_phoneme_text_prob == 0.0
    assert dataset.text_phoneme_token_offset == model.text_phoneme_token_offset
    assert model.enable_phoneme_text_input is True
    assert model.partial_phoneme_text_prob == 0.5


def test_force_raw_keeps_base_vocabulary_rows_and_accepts_base_model_weights():
    baseline = _model(force=False)
    forced = _model(force=True)
    before = baseline.state_dict()
    after = forced.state_dict()
    assert before.keys() == after.keys()
    assert all(before[key].shape == after[key].shape for key in before)
    assert baseline.text_vocab_size == forced.text_vocab_size
    assert baseline.text_phoneme_token_offset == forced.text_phoneme_token_offset
    assert baseline.text_phoneme_vocab_size == forced.text_phoneme_vocab_size == TinyPhonemeTokenizer.vocab_size

    forced.load_state_dict(before)

    for key, value in forced.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    device = next(forced.parameters()).device
    ipa = torch.tensor([[forced.text_phoneme_token_offset + 4]], device=device)
    lengths = torch.ones(1, dtype=torch.long, device=device)
    with torch.no_grad():
        embedded = forced.embed_text_tokens(ipa, text_lens=lengths)
    assert torch.isfinite(embedded).all()
    assert embedded.shape == (1, 1, forced.cfg.embedding_dim)


def _model(force=_UNSET, enabled=True, multiturn=False):
    extra = {
        'enable_phoneme_text_input': True,
        'partial_phoneme_text_prob': 0.5,
        'use_multiturn_dataset': multiturn,
        'load_cached_codes_if_available': True,
        'context_duration_min': 0.1,
        'context_duration_max': 0.2,
    }
    if force is not _UNSET:
        extra['attention_alignment'] = {'force_raw_text': force}
    return _make_attention_model(_config(enabled=enabled, extra=extra))


def _dataset_config(normalized=_UNSET):
    config = {'dataset': {}, 'volume_norm': False}
    if normalized is not _UNSET:
        config['load_normalized_text_percent'] = normalized
    return OmegaConf.create(config)


def _factory_call(model, config, factory, mode):
    dataset = SimpleNamespace()
    if factory == 'legacy':
        with patch.object(easy_magpietts, 'safe_instantiate', return_value=dataset) as construct:
            assert model.get_dataset(config, dataset_type=mode) is dataset
        return dataset, construct.call_args.kwargs

    name = 'MagpieTTSLhotseMultiturnDataset' if factory == 'lhotse_multiturn' else 'MagpieTTSLhotseDataset'
    result = object()
    with (
        patch.object(easy_magpietts, name, return_value=dataset) as construct,
        patch.object(easy_magpietts, 'FallbackDataset', side_effect=lambda value: value),
        patch.object(easy_magpietts, 'get_lhotse_dataloader_from_config', return_value=result) as loader,
    ):
        assert model.get_lhotse_dataloader(config, mode=mode) is result
    assert loader.call_args.kwargs['dataset'] is dataset
    return dataset, construct.call_args.kwargs
