# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from tests.collections.tts.models.test_nemotron_ctc_timestamp_head import _batch, _ctc_model
from tests.collections.tts.models.test_nemotron_tts_ctc_head import _native_model
from tests.collections.tts.models.test_nemotron_tts_ctc_word_intervals import MAPPING, _mapping, _word_model
from tests.collections.tts.modules.nemotron_tts.v2.test_training_model import _inputs

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('mode', ['train', 'val'])
def test_singleturn_batch_in_multiturn_model_trims_terminal_eos_only_for_ctc(mode):
    model = _ctc_model(multiturn=True).train(mode == 'train')
    inputs = _inputs(model)
    original_codes = inputs['audio_codes'].clone()
    expected_targets = _prepared_targets(model, inputs)
    assert expected_targets[0, :, -1].eq(model.audio_eos_id).all()
    with patch.object(model.ctc_timestamp_head, 'compute_loss', wraps=model.ctc_timestamp_head.compute_loss) as head:
        output = model.process_batch(**inputs, timestamp_batch=_batch(), mode=mode)
    assert len(model._timestamp_teacher.calls) == head.call_count == 1
    assert head.call_args.args[0].size(0) == 7
    assert head.call_args.kwargs['input_length'] == 14
    metrics = output.ctc_timestamp_metrics
    assert metrics['supervised_turns'] == metrics['trimmed_terminal_eos_turns'] == 1
    assert metrics['supervised_frames'] == 14 and metrics['skipped_turns'] == 0
    if mode == 'val':
        assert metrics['cer_reference_characters'] > 0 and metrics['cer_turns'] == 1
        values = _epoch_ctc_metrics(model, output)
        assert 'val/ctc_timestamp_teacher_cer' in values
        assert 'val/ctc_timestamp_cer' in values
        assert values['val/ctc_timestamp_reference_characters'] > 0
    torch.testing.assert_close(output.audio_codes_target, expected_targets, atol=0, rtol=0)
    torch.testing.assert_close(inputs['audio_codes'], original_codes, atol=0, rtol=0)
    output.ctc_timestamp_loss.backward()
    assert model.ctc_timestamp_head.projection[-1].weight.grad.abs().sum() > 0
    assert model.decoder.layers[2].mixer.q_proj.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model._timestamp_teacher.parameters())


@pytest.mark.parametrize('asr_enabled', [False, True])
def test_native_ctc_and_asr_ctc_both_train_after_singleturn_terminal_eos_is_excluded(asr_enabled):
    model = _native_model(multiturn=True, asr_enabled=asr_enabled).eval()
    inputs = _inputs(model)
    expected_targets = _prepared_targets(model, inputs)
    batch = _batch() if asr_enabled else dict(raw_texts=['selected TTS input'])
    with patch.object(model.tts_ctc_head, 'compute_loss', wraps=model.tts_ctc_head.compute_loss) as head:
        output = model.process_batch(**inputs, timestamp_batch=batch, mode='val')
    assert head.call_count == 1 and len(model._timestamp_teacher.calls) == int(asr_enabled)
    assert head.call_args.args[0].size(0) == 7 and head.call_args.kwargs['input_length'] == 14
    assert head.call_args.args[1].tolist() == [7, 8]
    metrics = output.tts_ctc_metrics
    assert metrics['supervised_turns'] == metrics['trimmed_terminal_eos_turns'] == 1
    assert metrics['supervised_frames'] == 14 and metrics['cer_reference_characters'] > 0
    if asr_enabled:
        assert output.ctc_timestamp_metrics['supervised_turns'] == 1
        assert output.ctc_timestamp_metrics['supervised_frames'] == 14
    torch.testing.assert_close(output.audio_codes_target, expected_targets, atol=0, rtol=0)
    output.tts_ctc_loss.backward()
    assert model.tts_ctc_head.projection[-1].weight.grad.abs().sum() > 0
    assert model.decoder.layers[2].mixer.q_proj.weight.grad.abs().sum() > 0


def test_word_ctc_clips_last_word_window_to_the_acoustic_prefix_before_terminal_eos():
    model = _word_model(multiturn=True, tolerance=0).eval()
    inputs = _inputs(model)
    expected_targets = _prepared_targets(model, inputs)
    with (
        patch(MAPPING, side_effect=_mapping),
        patch.object(model.tts_ctc_head, 'compute_loss', wraps=model.tts_ctc_head.compute_loss) as head,
    ):
        output = model.process_batch(**inputs, timestamp_batch=_batch(), mode='val')
    assert len(model._timestamp_teacher.calls) == head.call_count == 1
    assert head.call_args.kwargs['input_length'] == 14
    assert head.call_args.kwargs['word_intervals'][-1]['end_frame'] == 16
    metrics = output.tts_ctc_metrics
    assert metrics['word_ctc_valid_intervals'] == 2 and metrics['word_ctc_infeasible_intervals'] == 0
    assert metrics['word_ctc_loss'] > 0 and metrics['trimmed_terminal_eos_turns'] == 1
    torch.testing.assert_close(output.audio_codes_target, expected_targets, atol=0, rtol=0)
    output.tts_ctc_loss.backward()
    assert model.tts_ctc_head.projection[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('reason', ['interior_eos', 'non_eos_special', 'nonagent', 'cropped'])
def test_terminal_eos_exception_does_not_accept_interior_special_frames_users_or_cropped_turns(reason):
    model = _ctc_model(multiturn=True).eval()
    inputs = _inputs(model)
    prepare = model.prepare_audio_channel_embeddings
    prepare_targets = model._prepare_timestamp_targets

    def audio_channel(*args, **kwargs):
        values = list(prepare(*args, **kwargs))
        if reason in ('interior_eos', 'non_eos_special'):
            values[2] = values[2].clone()
            values[2][:, :, 3] = model.audio_eos_id if reason == 'interior_eos' else model.audio_bos_id
        return tuple(values)

    def timestamp_targets(*args, **kwargs):
        result = prepare_targets(*args, **kwargs)
        if reason == 'cropped':
            kwargs['ctc_targets'][0]['end_frame'] += 1
        return result

    if reason == 'nonagent':
        inputs['agent_mask'] = torch.tensor([[True, True, True, False, True, True, True, True]])
    with (
        patch.object(model, 'prepare_audio_channel_embeddings', side_effect=audio_channel),
        patch.object(model, '_prepare_timestamp_targets', side_effect=timestamp_targets),
        patch.object(
            model.ctc_timestamp_head, 'compute_loss', side_effect=AssertionError('Invalid turn was supervised')
        ),
    ):
        output = model.process_batch(**inputs, timestamp_batch=_batch(), mode='val')
    assert output.ctc_timestamp_metrics['skipped_turns'] == 1
    assert output.ctc_timestamp_metrics['supervised_turns'] == 0
    assert output.ctc_timestamp_metrics['skipped_cropped_turns'] == int(reason == 'cropped')
    assert output.ctc_timestamp_metrics['skipped_non_audio_turns'] == int(reason != 'cropped')
    assert output.ctc_timestamp_loss == 0
    output.ctc_timestamp_loss.backward()
    assert all(parameter.grad is not None for parameter in model.ctc_timestamp_head.parameters())


def test_terminal_eos_without_remaining_acoustic_frames_is_skipped_with_a_finite_connected_zero():
    model = _ctc_model(multiturn=True).eval()
    inputs = _inputs(model)
    inputs.update(audio_codes=inputs['audio_codes'][:, :, :1], audio_codes_lens=torch.tensor([1]))
    batch = _batch()
    batch.update(audio=batch['audio'][:, :8], audio_lens=torch.tensor([8]))
    with patch.object(model.ctc_timestamp_head, 'compute_loss', side_effect=AssertionError('EOS was used as audio')):
        output = model.process_batch(**inputs, timestamp_batch=batch, mode='val')
    assert output.audio_codes_target.eq(model.audio_eos_id).all()
    assert output.ctc_timestamp_metrics['supervised_turns'] == 0
    assert output.ctc_timestamp_metrics['supervised_frames'] == 0
    assert output.ctc_timestamp_metrics['skipped_turns'] == 1
    assert output.ctc_timestamp_metrics['skipped_empty_turns'] == 1
    assert output.ctc_timestamp_metrics['trimmed_terminal_eos_turns'] == 0
    assert output.ctc_timestamp_loss == 0 and torch.isfinite(output.ctc_timestamp_loss)
    output.ctc_timestamp_loss.backward()
    assert all(parameter.grad is not None for parameter in model.ctc_timestamp_head.parameters())


def _prepared_targets(model, inputs):
    return (
        model.prepare_audio_channel_embeddings(
            inputs['audio_codes'].clone(),
            inputs['audio_codes_lens'].clone(),
            delay=torch.zeros(1, dtype=torch.long),
            speech_eos_mask=inputs['text'].eq(model.interruption_token_id),
        )[2]
        .detach()
        .clone()
    )


def _epoch_ctc_metrics(model, output):
    class MetricsLogged(Exception):
        pass

    with patch('torch.distributed.is_initialized', return_value=False):
        model.on_validation_epoch_start()
        with patch.object(model, 'log_dict'):
            model._log_ctc_timestamp_losses(output, 'val', 1)
        with patch.object(model, 'log_dict', side_effect=MetricsLogged) as log, pytest.raises(MetricsLogged):
            model.on_validation_epoch_end()
    return log.call_args.args[0]
