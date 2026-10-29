# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest
import torch

from nemo.collections.tts.models import easy_magpietts
from tests.collections.tts.models.test_easy_magpietts import _make_easy_magpie_model, _toy_batch, tiny_easy_magpie_cfg
from tests.collections.tts.models.test_easy_magpietts_aligned_text import _alignment_batch
from tests.collections.tts.models.test_easy_magpietts_attention_alignment import (
    _attention_batch,
    _config,
    _make_attention_model,
    _step,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope='module')
def plot_model():
    return _make_plot_model()


@pytest.mark.parametrize('step', [0, 11000, 16000])
def test_validation_capture_matches_teacher_turn_and_preserves_predictions_and_losses(plot_model, step):
    model = plot_model
    batch = _attention_batch(model)
    with torch.no_grad(), _step(model, step):
        ordinary = model.process_batch(**batch)
        captured = model.process_batch(**batch, alignment_plot_limit=2)

    assert ordinary.alignment_plots is None
    torch.testing.assert_close(captured.logits, ordinary.logits, rtol=0, atol=0)
    torch.testing.assert_close(captured.loss, ordinary.loss, rtol=0, atol=0)
    torch.testing.assert_close(captured.phoneme_logits, ordinary.phoneme_logits, rtol=0, atol=0)
    torch.testing.assert_close(captured.phoneme_loss, ordinary.phoneme_loss, rtol=0, atol=0)
    torch.testing.assert_close(captured.attention_alignment_loss, ordinary.attention_alignment_loss, rtol=0, atol=0)
    assert len(captured.alignment_plots) == 2
    for index, example in enumerate(captured.alignment_plots):
        assert example['batch_index'] == index
        assert example['sample_start'] == 0
        path = example['mas_token_indices']
        probabilities = example['attention_probs']
        frame_indices = example['attention_frame_indices']
        token_positions = example['attention_token_positions']
        expected_tokens = batch['attention_alignment_text_tokens'][
            index, : batch['attention_alignment_text_lens'][index]
        ]
        torch.testing.assert_close(example['token_ids'], expected_tokens.cpu())
        assert path.ndim == 1 and path.numel() > 0
        assert probabilities.ndim == 2
        assert probabilities.size(0) == frame_indices.numel()
        assert probabilities.size(1) == token_positions.numel()
        assert frame_indices.min() >= 0 and frame_indices.max() < path.numel()
        assert token_positions.min() >= 0 and token_positions.max() < expected_tokens.numel()
        assert torch.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert (probabilities.sum(-1) <= 1.0 + 1e-6).all(), 'Keep full-key probability normalization.'
        for tensor in (path, probabilities, frame_indices, token_positions, example['token_ids']):
            assert tensor.device.type == 'cpu'
            assert not tensor.requires_grad


def test_attention_capture_has_bounded_rows_columns_and_example_count():
    model = _make_plot_model({'max_examples': 1, 'max_audio_frames': 2, 'max_text_tokens': 1})
    with torch.no_grad(), _step(model, 11000):
        output = model.process_batch(**_attention_batch(model), alignment_plot_limit=1)

    assert len(output.alignment_plots) == 1
    example = output.alignment_plots[0]
    assert example['attention_probs'].size(0) <= 2
    assert example['attention_probs'].size(1) <= 1
    assert example['cropped']


@pytest.mark.parametrize(
    'training,grad_enabled,mode', [(True, False, 'val'), (False, True, 'val'), (False, False, 'train')]
)
def test_diagnostic_capture_is_validation_only(plot_model, training, grad_enabled, mode):
    model = plot_model
    model.train(training)
    batch = _attention_batch(model)
    batch['mode'] = mode
    try:
        with torch.set_grad_enabled(grad_enabled), _step(model, 11000):
            output = model.process_batch(**batch, alignment_plot_limit=1)
        assert output.alignment_plots is None
    finally:
        model.eval()


def test_ordinary_models_never_create_alignment_plot_payloads():
    model = _make_easy_magpie_model(tiny_easy_magpie_cfg())
    with torch.no_grad():
        output = model.process_batch(**_toy_batch(model), mode='val', alignment_plot_limit=2)
    assert output.alignment_plots is None
    assert not hasattr(model, 'text_aligner')


@pytest.mark.parametrize('step', [0, 10000])
def test_repeated_head_mode_can_plot_mas_without_changing_predictions_or_enabling_attention_guidance(step):
    model = _make_easy_magpie_model(
        tiny_easy_magpie_cfg(
            {
                'use_aligned_text_head': True,
                'aligned_text_head_dim': 8,
                'aligned_text_aligner': {
                    'hidden_channels': 8,
                    'num_flows': 2,
                    'num_text_layers': 1,
                    'text_dropout': 0.0,
                    'dequantization_noise': 0.0,
                },
            }
        )
    )
    batch = _alignment_batch(model)
    with torch.no_grad(), _step(model, step):
        ordinary = model.process_batch(**batch)
        output = model.process_batch(**batch, alignment_plot_limit=1)
    assert len(output.alignment_plots) == 1
    assert output.alignment_plots[0]['attention_probs'] is None
    assert output.alignment_plots[0]['mas_token_indices'].numel() > 0
    assert not model.use_attention_alignment
    torch.testing.assert_close(output.logits, ordinary.logits, rtol=0, atol=0)
    torch.testing.assert_close(output.loss, ordinary.loss, rtol=0, atol=0)
    torch.testing.assert_close(output.aligned_text_loss, ordinary.aligned_text_loss, rtol=0, atol=0)
    torch.testing.assert_close(output.aligned_text_targets, ordinary.aligned_text_targets, rtol=0, atol=0)


def test_disabled_visualization_ignores_a_manual_capture_request():
    model = _make_plot_model({'enabled': False})
    with (
        torch.no_grad(),
        _step(model, 11000),
        patch.object(
            easy_magpietts, 'AttentionAlignmentCapture', side_effect=AssertionError('Capture must remain disabled')
        ),
    ):
        output = model.process_batch(**_attention_batch(model), alignment_plot_limit=2)
    assert output.alignment_plots is None


@pytest.mark.parametrize(
    'settings',
    [
        {'enabled': 'false'},
        {'enabled': 0},
        {'max_examples': -1},
        {'max_examples': True},
        {'max_audio_frames': 0},
        {'max_audio_frames': 1.5},
        {'max_text_tokens': 0},
        {'max_text_tokens': False},
    ],
)
def test_invalid_visualization_configuration_fails_before_validation(settings):
    with pytest.raises(ValueError, match='alignment_visualization'):
        _make_plot_model(settings)


def test_zero_image_budget_disables_capture_without_enabling_or_disabling_training_guidance():
    model = _make_plot_model({'max_examples': 0})
    with (
        torch.no_grad(),
        _step(model, 11000),
        patch.object(
            easy_magpietts, 'AttentionAlignmentCapture', side_effect=AssertionError('Capture must remain disabled')
        ),
    ):
        output = model.process_batch(**_attention_batch(model), alignment_plot_limit=2)
    assert model.use_attention_alignment
    assert output.alignment_plots is None
    assert output.attention_alignment_loss > 0


@pytest.mark.parametrize('reason', ['disabled', 'rank', 'sanity', 'no_wandb', 'exhausted'])
def test_logging_gate_avoids_capture_without_an_eligible_wandb_validation_context(reason):
    settings = {'enabled': reason != 'disabled'}
    model = _make_plot_model(settings)
    logger = Mock(spec=easy_magpietts.WandbLogger)
    model._alignment_plots_logged = 2 if reason == 'exhausted' else 0
    with _logging_context(
        model,
        [] if reason == 'no_wandb' else [logger],
        rank=1 if reason == 'rank' else 0,
        sanity=reason == 'sanity',
    ):
        assert model._alignment_plot_budget() == 0


def test_wandb_images_have_stable_keys_and_one_budget_for_the_complete_validation_epoch(plot_model):
    model = plot_model
    logger = Mock(spec=easy_magpietts.WandbLogger)
    with torch.no_grad(), _step(model, 11000):
        output = model.process_batch(**_attention_batch(model), alignment_plot_limit=2)
    with _logging_context(model, [logger]), _step(model, 11000):
        model._alignment_plots_logged = 0
        assert model._alignment_plot_budget() == 2
        model._log_alignment_visualizations(output)
        model._log_alignment_visualizations(output)
        assert model._alignment_plot_budget() == 0
        assert logger.log_image.call_count == 2
        keys = [call.kwargs['key'] for call in logger.log_image.call_args_list]
        assert keys == ['val/alignment/turn_0', 'val/alignment/turn_1']
        for call in logger.log_image.call_args_list:
            assert call.kwargs['step'] == 11000
            image = call.kwargs['images'][0]
            assert isinstance(image, np.ndarray) and image.dtype == np.uint8
            assert image.ndim == 3 and image.shape[-1] == 3
            assert 'teacher' in call.kwargs['caption'][0].lower()

        with patch.object(model, '_prepare_text_only_validation_examples', return_value=None):
            model.on_validation_epoch_start()
        assert model._alignment_plot_budget() == 2
        model._log_alignment_visualizations(output)
        assert logger.log_image.call_count == 4


def test_multiple_wandb_loggers_do_not_multiply_the_sample_budget(plot_model):
    model = plot_model
    loggers = [Mock(spec=easy_magpietts.WandbLogger), Mock(spec=easy_magpietts.WandbLogger)]
    with torch.no_grad(), _step(model, 11000):
        output = model.process_batch(**_attention_batch(model), alignment_plot_limit=2)
    with _logging_context(model, loggers):
        model._alignment_plots_logged = 0
        model._log_alignment_visualizations(output)
        assert model._alignment_plots_logged == 2
        assert all(logger.log_image.call_count == 2 for logger in loggers)


def test_real_lightning_log_image_converts_rgb_and_caption_without_creating_a_wandb_run():
    from lightning.pytorch.utilities.rank_zero import rank_zero_only

    logger = Mock(spec=easy_magpietts.WandbLogger)
    pixels = np.zeros((12, 20, 3), dtype=np.uint8)
    caption = 'Teacher-forced evaluation: MAS and attention before the training prior.'

    with patch.object(rank_zero_only, 'rank', 0, create=True):
        easy_magpietts.WandbLogger.log_image(
            logger, key='val/alignment/turn_0', images=[pixels], caption=[caption], step=0
        )

    logger.log_metrics.assert_called_once()
    args, kwargs = logger.log_metrics.call_args
    metrics = args[0]
    step = kwargs.get('step', args[1] if len(args) > 1 else None)
    assert step == 0
    assert list(metrics) == ['val/alignment/turn_0']
    images = metrics['val/alignment/turn_0']
    assert len(images) == 1
    assert isinstance(images[0], easy_magpietts.wandb.Image)
    assert images[0]._caption == caption
    assert images[0].image.size == (20, 12)
    logger.experiment.log.assert_not_called()


def test_validation_step_uses_remaining_epoch_budget_on_later_batches(plot_model):
    model = plot_model
    logger = Mock(spec=easy_magpietts.WandbLogger)
    batch = _attention_batch(model)
    model.validation_step_outputs = []
    model._alignment_plots_logged = 0
    with (
        torch.no_grad(),
        _step(model, 11000),
        _logging_context(model, [logger]),
        patch.object(model, 'log_dict'),
        patch.object(model, 'process_batch', wraps=model.process_batch) as process,
    ):
        model.validation_step(batch, batch_idx=1)
        assert process.call_args.kwargs['alignment_plot_limit'] == 2
        model.validation_step(batch, batch_idx=2)
        assert process.call_args.kwargs['alignment_plot_limit'] == 0
        assert logger.log_image.call_count == 2
    model.validation_step_outputs.clear()


def test_invalid_turns_do_not_spend_the_epoch_image_budget(plot_model):
    model = plot_model
    logger = Mock(spec=easy_magpietts.WandbLogger)
    batch = _attention_batch(model)
    batch['attention_alignment_turn_indices'][:, 2] = batch['attention_alignment_turn_indices'][:, 1]
    with torch.no_grad(), _step(model, 11000):
        output = model.process_batch(**batch, alignment_plot_limit=2)
    with _logging_context(model, [logger]):
        model._alignment_plots_logged = 0
        model._log_alignment_visualizations(output)
        assert model._alignment_plot_budget() == 2
        logger.log_image.assert_not_called()


def _make_plot_model(settings=None):
    values = {'enabled': True, 'max_examples': 2, 'max_audio_frames': 1024, 'max_text_tokens': 256}
    values.update(settings or {})
    return _make_attention_model(_config(extra={'alignment_visualization': values}))


@contextmanager
def _logging_context(model, loggers, rank=0, sanity=False):
    trainer = SimpleNamespace(sanity_checking=sanity, world_size=1)
    with (
        patch.object(type(model), 'loggers', new_callable=PropertyMock, return_value=loggers),
        patch.object(type(model), 'global_rank', new_callable=PropertyMock, return_value=rank),
        patch.object(type(model), 'trainer', new_callable=PropertyMock, return_value=trainer),
    ):
        yield
