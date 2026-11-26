# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import PropertyMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.models.nemotron_tts.v2.base import TrainingMode
from tests.collections.tts.models.test_nemotron_asr_word_start_attention import MAPPING, _anchor_model, _anchors
from tests.collections.tts.models.test_nemotron_ctc_timestamp_head import _batch
from tests.collections.tts.models.test_nemotron_timestamp_head import _packed_text
from tests.collections.tts.modules.nemotron_tts.v2.test_streaming_timestamps_model import _init_stream
from tests.collections.tts.modules.nemotron_tts.v2.test_training_model import _inputs

pytestmark = pytest.mark.unit


def test_disabled_word_mask_preserves_sparse_anchor_outputs_and_loss_exactly():
    baseline = _anchor_model()
    observed = _mask_model(enabled=False)
    with patch(MAPPING, side_effect=_anchors):
        expected = baseline.process_batch(**_inputs(baseline), timestamp_batch=_batch(), mode='val')
    with (
        patch(MAPPING, side_effect=_anchors),
        patch.object(observed, 'forward', wraps=observed.forward) as forward,
    ):
        actual = observed.process_batch(**_inputs(observed), timestamp_batch=_batch(), mode='val')
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
    torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
    torch.testing.assert_close(
        actual.asr_word_start_attention_loss, expected.asr_word_start_attention_loss, atol=0, rtol=0
    )
    guidance = forward.call_args.kwargs['attention_alignment']
    assert len(_rows(guidance)) == actual.asr_word_start_attention_metrics['anchors'] == 2
    assert all(turn.band_width is None for turn in guidance.turns)
    assert guidance.prior_strength == 0.0


def test_mask_covers_real_leading_and_trailing_frames_with_word_spans_and_distinct_widths():
    model = _mask_model(ramp_steps=0)
    guidance, metrics = _prepare(model)
    rows = _rows(guidance)
    assert set(rows) == set(range(10, 18))
    assert metrics['anchors'] == 2 and metrics['interior_frames'] == 6
    assert rows[12] == (0, 2, 0.5)
    assert rows[15] == (2, 4, 0.5)
    assert all(rows[frame] == (0, 3, 1.0) for frame in (10, 11, 13, 14))
    assert all(rows[frame] == (1, 4, 1.0) for frame in (16, 17))
    assert all(turn.key_positions.tolist() == [1, 2, 3, 4] for turn in guidance.turns)
    assert guidance.layers == 'middle' and guidance.heads == (0,)
    assert guidance.look_back_tokens == guidance.look_ahead_tokens == 0
    assert guidance.prior_strength == metrics['prior_strength'] == 1.0


def test_colliding_starts_keep_the_full_merged_word_span_at_the_onset():
    model = _mask_model(ramp_steps=0)
    mapping = dict(
        error=None,
        collision_count=1,
        merged_word_groups=0,
        token_coverage=1.0,
        anchors=[
            dict(frame_index=2, token_start=0, token_end=4),
            dict(frame_index=5, token_start=4, token_end=6),
        ],
    )
    guidance, metrics = _prepare(model, token_count=6, mapping=mapping)
    rows = _rows(guidance)
    assert rows[12] == (0, 4, 0.5)
    assert rows[15] == (4, 6, 0.5)
    assert rows[10] == (0, 5, 1.0) and rows[17] == (3, 6, 1.0)
    assert metrics['collision_count'] == 1 and metrics['anchors'] == 2


def test_mask_cannot_rescue_an_unseen_word_through_a_visible_lookback_token():
    model = _mask_model(ramp_steps=0)
    target = _target()
    target['key_positions'] = torch.tensor([1, 2, 50, 51])
    codes = torch.zeros(1, 2, 8, dtype=torch.long)
    codes[:, :, 4] = model.audio_eos_id
    agent_mask = torch.tensor([[True, False, True, True, True, True, True, True]])
    guidance, metrics = _prepare(model, targets=[target], codes=codes, agent_mask=agent_mask)
    assert set(_rows(guidance)) == {10, 12, 13}
    assert metrics['skipped_non_audio_anchors'] == 2
    assert metrics['skipped_causal_anchors'] == 3
    assert metrics['anchors'] == 1 and metrics['interior_frames'] == 2


def test_all_queries_in_overlapping_turns_are_rejected_including_interior_mask_rows():
    model = _mask_model(ramp_steps=0)
    first = _target()
    second = dict(first, turn_index=1)
    guidance, metrics = _prepare(model, targets=[first, second])
    assert guidance is None
    assert metrics['anchors'] == metrics['interior_frames'] == 0
    assert metrics['overlapping_anchors'] == 16


@pytest.mark.parametrize(
    'step,loss_scale,prior',
    [
        (99, 0.0, 0.0),
        (100, 0.0, 0.0),
        (110, 0.5, 0.4),
        (120, 1.0, 0.8),
        (140, 1.0, 0.4),
        (160, 1.0, 0.0),
        (200, 1.0, 0.0),
    ],
)
def test_mask_reuses_loss_ramp_and_prior_annealing_schedule(step, loss_scale, prior):
    model = _mask_model(start_step=100, ramp_steps=20, prior_anneal_steps=40, prior_strength=0.8)
    with patch.object(type(model), 'global_step', new_callable=PropertyMock, return_value=step):
        guidance, metrics = _prepare(model)
    assert metrics['loss_scale'] == pytest.approx(loss_scale)
    assert metrics['prior_strength'] == pytest.approx(prior)
    if guidance is not None:
        assert guidance.prior_strength == pytest.approx(prior)
        assert len(_rows(guidance)) == 8


@pytest.mark.parametrize('text_mode,delay', [('streaming', 1), ('full', 16)])
def test_dense_mask_uses_the_original_multiturn_clock_and_applies_speech_delay_once(text_mode, delay):
    model = _mask_model(multiturn=True, ramp_steps=0)
    turns = []
    inputs = _inputs(model)
    inputs.update(
        text=_packed_text(model, timestamp_turns=turns, fractional=True),
        text_lens=torch.tensor([15]),
        audio_codes=torch.zeros(1, 2, 15, dtype=torch.long),
        audio_codes_lens=torch.tensor([15]),
        context_text_tokens=torch.empty(1, 0, dtype=torch.long),
        context_text_tokens_lens=torch.tensor([0]),
    )
    batch = dict(audio=torch.zeros(1, 120), audio_lens=torch.tensor([120]), timestamp_turns=[turns], languages=['pt'])
    with patch(MAPPING, side_effect=_anchors), patch.object(model, 'forward', wraps=model.forward) as forward:
        output = model.process_batch(
            **inputs, timestamp_batch=batch, mode='val', training_mode=TrainingMode(text_mode, 0, 1, 0)
        )
    rows = _rows(forward.call_args.kwargs['attention_alignment'])
    assert set(rows) == {delay + frame for frame in (*range(2, 7), *range(9, 14))}
    assert output.asr_word_start_attention_metrics['anchors'] == 4
    assert output.asr_word_start_attention_metrics['interior_frames'] == 6
    assert len(model._timestamp_teacher.calls) == 1


def test_mask_training_uses_one_teacher_pass_and_applies_the_loss_ramp_once():
    model = _mask_model(start_step=0, ramp_steps=1000)
    with (
        patch.object(type(model), 'global_step', new_callable=PropertyMock, return_value=500),
        patch(MAPPING, side_effect=_anchors),
        patch.object(model, 'forward', wraps=model.forward) as forward,
        patch(
            'nemo.collections.tts.models.nemotron_tts.v2.model.prepare_attention_alignment_training',
            side_effect=AssertionError('Word mask loaded the flow aligner'),
        ),
    ):
        output = model.process_batch(**_inputs(model), timestamp_batch=_batch(), mode='val')
    metrics = output.asr_word_start_attention_metrics
    guidance = forward.call_args.kwargs['attention_alignment']
    assert len(model._timestamp_teacher.calls) == 1
    assert not model._timestamp_teacher.calls[0]['options'].get('return_ctc', False)
    assert not hasattr(model, 'ctc_timestamp_head') and not hasattr(model, 'tts_ctc_head')
    assert metrics['loss_scale'] == guidance.prior_strength == 0.5
    expected = (
        model.parallel_codebook_loss_scale * output.codebook_loss
        + model.acoustic_codes_predictor_loss_scale * output.acoustic_codes_predictor_loss
        + model.asr_word_start_attention_config['loss_weight'] * 0.5 * output.asr_word_start_attention_loss
    )
    torch.testing.assert_close(output.loss, expected)
    output.asr_word_start_attention_loss.backward()
    assert model.decoder.layers[2].mixer.q_proj.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model._timestamp_teacher.parameters())


def test_mask_is_training_only_and_does_not_load_the_teacher_during_streaming_or_policy_scoring():
    model = _mask_model(ramp_steps=0).eval()
    state = _init_stream(model, use_cfg=False)
    with (
        patch.object(model, '_get_timestamp_teacher', side_effect=AssertionError('Inference loaded ASR')),
        patch.object(model, '_prepare_asr_word_start_attention', side_effect=AssertionError('Inference applied mask')),
    ):
        for token in [model.bos_id, 7, 8, model.eos_id, model.pad_id, model.pad_id, None, None]:
            state, *_ = model.streaming_step(state, None if token is None else torch.full((2,), token))
        output = model.streaming_finalize(state)
        policy = model.process_batch(**_inputs(model), mode='val', return_policy_logits=True)
    assert output.audio_len.numel() == 2 and policy.asr_word_start_attention_loss is None


@pytest.mark.parametrize(
    'key,value',
    [
        ('start_band_width', 0.0),
        ('band_width', -1.0),
        ('look_back_tokens', -1),
        ('look_ahead_tokens', True),
        ('prior_strength', 1.1),
        ('start_step', -1),
        ('ramp_steps', -1),
        ('prior_anneal_steps', 0),
    ],
)
def test_enabled_mask_rejects_invalid_band_and_schedule_settings(key, value):
    model = _anchor_model()
    cfg = OmegaConf.create(dict(asr_word_start_attention=dict(mask=dict(enabled=True, **{key: value}))))
    with pytest.raises(ValueError):
        model._init_asr_word_start_attention(cfg)


def _mask_model(enabled=True, multiturn=False, **overrides):
    model = _anchor_model(multiturn=multiturn)
    model.asr_word_start_attention_config['mask'] = dict(
        enabled=enabled,
        start_band_width=0.5,
        band_width=1.0,
        look_back_tokens=1,
        look_ahead_tokens=1,
        start_step=0,
        ramp_steps=1000,
        prior_anneal_steps=5000,
        prior_strength=1.0,
    )
    model.asr_word_start_attention_config['mask'].update(overrides)
    return model


def _target(token_count=4):
    return dict(
        batch_index=0,
        turn_index=0,
        start_frame=0,
        end_frame=8,
        complete_turn=True,
        token_ids=torch.arange(7, 7 + token_count),
        key_positions=torch.arange(1, 1 + token_count),
        reference_text='alpha beta',
        audio_start_seconds=0.0,
        audio_duration=0.64,
    )


def _prepare(model, token_count=4, targets=None, codes=None, agent_mask=None, mapping=None):
    targets = [_target(token_count)] if targets is None else targets
    records = [
        dict(
            batch_index=target['batch_index'],
            turn_index=target['turn_index'],
            words=[dict(word='reference')],
            error=None,
        )
        for target in targets
    ]
    if mapping is None:
        mapping = dict(
            error=None,
            collision_count=0,
            merged_word_groups=0,
            token_coverage=1.0,
            anchors=[
                dict(frame_index=2, token_start=0, token_end=2),
                dict(frame_index=5, token_start=2, token_end=4),
            ],
        )
    codes = torch.zeros(1, 2, 8, dtype=torch.long) if codes is None else codes
    with patch(MAPPING, return_value=mapping):
        return model._prepare_asr_word_start_attention(
            targets, records, torch.tensor([0]), torch.tensor([10]), codes, torch.tensor([8]), agent_mask=agent_mask
        )


def _rows(guidance):
    rows = {}
    for turn in guidance.turns:
        for query, start, end in zip(
            turn.query_positions.tolist(), turn.token_indices.tolist(), turn.token_end_indices.tolist()
        ):
            assert query not in rows
            rows[query] = (start, end, turn.band_width)
    return rows
