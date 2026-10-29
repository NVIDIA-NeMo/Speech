# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.models import easy_magpietts
from tests.collections.tts.models.test_easy_magpietts import _make_easy_magpie_model, _toy_batch, tiny_easy_magpie_cfg
from tests.collections.tts.models.test_easy_magpietts_attention_alignment import (
    _attention_batch,
    _config,
    _make_attention_model,
    _step,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('selector,expected', [('middle', 'middle'), (None, None), ([1], (1,)), ([0, 2], (0, 2))])
def test_layer_selector_reaches_real_decoder_without_changing_conditioning_targets(selector, expected):
    model = _make_attention_model(
        _three_attention_config(
            {'attention_alignment': {'layers': selector, 'look_back_tokens': 2, 'look_ahead_tokens': 3}}
        )
    )
    batch = _attention_batch(model)
    conditioning_before = batch['text'].clone()
    teacher_before = batch['attention_alignment_text_tokens'].clone()

    with (
        torch.no_grad(),
        _step(model, 11000),
        patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as run,
    ):
        output = model.process_batch(**batch)

    guidance = run.call_args.kwargs['attention_alignment']
    assert guidance.layers == expected
    assert guidance.heads == (0,)
    assert guidance.prior_strength == 1.0
    assert guidance.look_back_tokens == 2
    assert guidance.look_ahead_tokens == 3
    assert torch.isfinite(output.loss)
    assert output.attention_alignment_loss > 0
    torch.testing.assert_close(batch['text'], conditioning_before, rtol=0, atol=0)
    torch.testing.assert_close(batch['attention_alignment_text_tokens'], teacher_before, rtol=0, atol=0)


def test_default_selector_is_middle_and_normalized_head_source_is_preserved():
    model = _make_attention_model(_three_attention_config())
    assert model.attention_alignment_config['layers'] == 'middle'
    assert model.attention_alignment_config['look_back_tokens'] == 0
    assert model.attention_alignment_config['look_ahead_tokens'] == 0
    assert model.aligned_text_use_normalized_text is True
    assert model.cfg.get('aligned_text_use_normalized_text', True) is True


@pytest.mark.parametrize('selector', [[], [True], [-1], [3], [1, 1], 'all'])
def test_invalid_layer_selector_fails_before_training(selector):
    with pytest.raises(ValueError, match='layers'):
        _make_attention_model(_three_attention_config({'attention_alignment': {'layers': selector}}))


@pytest.mark.parametrize('name', ['look_back_tokens', 'look_ahead_tokens'])
@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_invalid_attention_token_window_fails_before_training(name, value):
    with pytest.raises(ValueError, match=name):
        _make_attention_model(_config(extra={'attention_alignment': {name: value}}))


def test_alignment_features_default_off_and_do_not_create_guidance_requests():
    model = _make_easy_magpie_model(tiny_easy_magpie_cfg())
    assert model.use_attention_alignment is False
    assert model.use_aligned_text_head is False
    assert model.attention_alignment_inference_config['enabled'] is False
    assert model.aligned_text_use_normalized_text is True
    assert not hasattr(model, 'text_aligner')
    assert not hasattr(model, 'aligned_text_head')

    with torch.no_grad(), patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as run:
        output = model.process_batch(**_toy_batch(model))

    assert run.call_args.kwargs['attention_alignment'] is None
    assert run.call_args.kwargs.get('attention_alignment_inference') is None
    assert output.attention_alignment_loss is None
    assert output.aligner_loss is None
    assert output.aligned_text_loss is None
    assert torch.isfinite(output.loss)


@pytest.mark.parametrize('value', ['false', 0, 1, None])
def test_normalized_head_source_requires_an_actual_boolean(value):
    with pytest.raises(ValueError, match='aligned_text_use_normalized_text'):
        _make_attention_model(_config(extra={'aligned_text_use_normalized_text': value}))


@pytest.mark.parametrize('normalized', [False, True])
@pytest.mark.parametrize('factory', ['legacy', 'lhotse_single', 'lhotse_multiturn'])
def test_transcript_source_option_reaches_each_dataset_factory(normalized, factory):
    model = _make_attention_model(
        _config(
            extra={
                'aligned_text_use_normalized_text': normalized,
                'use_multiturn_dataset': factory == 'lhotse_multiturn',
                'load_cached_codes_if_available': True,
                'context_duration_min': 0.1,
                'context_duration_max': 0.2,
            }
        )
    )
    assert model.aligned_text_use_normalized_text is normalized
    dataset = SimpleNamespace()
    cfg = OmegaConf.create({'dataset': {}, 'volume_norm': False})
    if factory == 'legacy':
        with patch.object(easy_magpietts, 'safe_instantiate', return_value=dataset):
            actual = model.get_dataset(cfg, dataset_type='train')
        assert actual is dataset
        assert actual.aligned_text_use_normalized_text is normalized
        return

    lhotse_class = 'MagpieTTSLhotseMultiturnDataset' if factory == 'lhotse_multiturn' else 'MagpieTTSLhotseDataset'
    result = object()
    with (
        patch.object(easy_magpietts, lhotse_class, return_value=dataset) as construct,
        patch.object(easy_magpietts, 'FallbackDataset', side_effect=lambda value: value),
        patch.object(easy_magpietts, 'get_lhotse_dataloader_from_config', return_value=result) as loader,
    ):
        actual = model.get_lhotse_dataloader(cfg, mode='train')

    assert actual is result
    assert loader.call_args.kwargs['dataset'] is dataset
    if factory == 'lhotse_multiturn':
        assert construct.call_args.kwargs['aligned_text_use_normalized_text'] is normalized
    else:
        assert dataset.aligned_text_use_normalized_text is normalized


def _three_attention_config(extra=None):
    overrides = {
        'nemotron_h_config': {'num_hidden_layers': 5, 'hybrid_override_pattern': '*-*-*'},
    }
    if extra:
        overrides.update(extra)
    return _config(extra=overrides)
