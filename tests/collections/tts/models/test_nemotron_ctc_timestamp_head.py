# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.models.nemotron_tts.v2.base import TrainingMode
from nemo.collections.tts.modules.nemotron_tts.v2.acoustic_codes_predictor import AcousticCodesPredictor
from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import IGNORE_INDEX, CTCTimestampHead
from tests.collections.tts.models.test_nemotron_timestamp_head import _packed_text, _timestamp_model
from tests.collections.tts.modules.nemotron_tts.v2.test_streaming_timestamps_model import _init_stream
from tests.collections.tts.modules.nemotron_tts.v2.test_training_model import _inputs

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('ctc_enabled', [False, True])
@pytest.mark.parametrize('binary_enabled', [False, True])
def test_ctc_and_binary_heads_are_independent_and_share_one_teacher_pass(ctc_enabled, binary_enabled):
    model = _ctc_model(ctc_enabled=ctc_enabled, binary_enabled=binary_enabled)
    teacher = model._timestamp_teacher
    output = model.process_batch(**_inputs(model), timestamp_batch=_batch(), mode='val')
    assert len(teacher.calls) == int(ctc_enabled or binary_enabled)
    assert (output.timestamp_loss is not None) is binary_enabled
    assert (output.ctc_timestamp_loss is not None) is ctc_enabled
    if ctc_enabled:
        assert teacher.calls[0]['options'] == dict(return_ctc=True, align_words=binary_enabled, distillation_topk=2)
        assert output.ctc_timestamp_metrics['supervised_turns'] == 1
        torch.testing.assert_close(
            output.ctc_timestamp_loss,
            model.ctc_timestamp_head.ctc_weight * output.ctc_timestamp_metrics['ctc_loss']
            + model.ctc_timestamp_head.distillation_weight * output.ctc_timestamp_metrics['distillation_loss'],
        )
    elif binary_enabled:
        assert teacher.calls[0]['options'] == {}
    expected = (
        model.parallel_codebook_loss_scale * output.codebook_loss
        + model.acoustic_codes_predictor_loss_scale * output.acoustic_codes_predictor_loss
    )
    if binary_enabled:
        expected += model.timestamp_head_loss_weight * output.timestamp_loss
    if ctc_enabled:
        expected += model.ctc_timestamp_head_loss_weight * output.ctc_timestamp_loss
    torch.testing.assert_close(output.loss, expected)


def test_ctc_loss_trains_the_head_and_backbone_with_a_frozen_unregistered_teacher():
    model = _ctc_model()
    output = model.process_batch(**_inputs(model), timestamp_batch=_batch(), mode='val')
    output.ctc_timestamp_loss.backward()
    assert torch.isfinite(output.ctc_timestamp_loss)
    assert all(parameter.grad is not None for parameter in model.ctc_timestamp_head.parameters())
    assert any(parameter.grad.abs().sum() > 0 for parameter in model.ctc_timestamp_head.parameters())
    assert model.decoder.layers[2].mixer.q_proj.weight.grad.abs().sum() > 0
    assert '_timestamp_teacher' not in model._modules
    assert not any('timestamp_teacher' in name for name in model.state_dict())
    assert all(parameter.grad is None for parameter in model._timestamp_teacher.parameters())


def test_output_only_ctc_preserves_training_audio_logits_and_zero_weight_baseline_loss():
    baseline, observed = _ctc_model(ctc_enabled=False).eval(), _ctc_model().eval()
    observed.ctc_timestamp_head_loss_weight = 0.0
    expected = baseline.process_batch(**_inputs(baseline), mode='val')
    actual = observed.process_batch(**_inputs(observed), timestamp_batch=_batch(), mode='val')
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)


@pytest.mark.parametrize('text_input_mode,offset', [('streaming', 1), ('full', 5)])
def test_ctc_reads_the_existing_audio_prediction_slice_and_excludes_speech_eos(text_input_mode, offset):
    model = _ctc_model()
    inputs = _inputs(model)
    inputs.update(context_text_tokens=torch.empty(1, 0, dtype=torch.long), context_text_tokens_lens=torch.tensor([0]))
    hidden = []
    hook = model.decoder.register_forward_hook(lambda _module, _args, result: hidden.append(result.last_hidden_state))
    try:
        with patch.object(
            model.ctc_timestamp_head, 'compute_loss', wraps=model.ctc_timestamp_head.compute_loss
        ) as head:
            output = model.process_batch(
                **inputs, timestamp_batch=_batch(), mode='val', training_mode=TrainingMode(text_input_mode, 0, 1, 0)
            )
    finally:
        hook.remove()
    assert head.call_count == 1
    torch.testing.assert_close(head.call_args.args[0], hidden[0][0, offset : offset + 8], rtol=0, atol=0)
    assert head.call_args.kwargs['input_length'] == 16
    assert output.audio_codes_target[0, :, 8].eq(model.audio_eos_id).all()
    assert output.ctc_timestamp_metrics['supervised_frames'] == 16


@pytest.mark.parametrize('fractional', [False, True])
def test_multiturn_ctc_keeps_turn_slices_and_references_on_the_original_clock(fractional):
    model = _ctc_model(multiturn=True)
    turns = []
    text = _packed_text(model, timestamp_turns=turns, fractional=fractional)
    audio = torch.arange(120, dtype=torch.float32).unsqueeze(0)
    batch = dict(audio=audio, audio_lens=torch.tensor([120]), timestamp_turns=[turns], languages=['pt'])
    targets = []
    model._prepare_timestamp_targets(batch, text, torch.tensor([15]), 16, ctc_targets=targets)
    assert len(model._timestamp_teacher.calls) == 1 and len(targets) == 2
    call = model._timestamp_teacher.calls[0]
    assert call['texts'] == ['first turn', 'second turn']
    assert call['languages'] == ['pt', 'pt']
    first = 13 if fractional else 16
    torch.testing.assert_close(call['audio'][0], audio[0, first : first + 40])
    torch.testing.assert_close(call['audio'][1], audio[0, 72:112])
    assert [(target['start_frame'], target['end_frame']) for target in targets] == [(2, 7), (9, 14)]
    assert targets[0]['audio_start_seconds'] == pytest.approx(first / 100)
    assert targets[1]['audio_start_seconds'] == pytest.approx(0.72)
    hidden = torch.randn(1, 16, 16, requires_grad=True)
    codes = torch.zeros(1, 2, 16, dtype=torch.long)
    with patch.object(model.ctc_timestamp_head, 'compute_loss', wraps=model.ctc_timestamp_head.compute_loss) as head:
        loss, metrics = model._compute_ctc_timestamp_loss(hidden, targets, codes, torch.tensor([16]))
    assert head.call_count == 2 and metrics['supervised_turns'] == 2
    torch.testing.assert_close(head.call_args_list[0].args[0], hidden[0, 2:7])
    torch.testing.assert_close(head.call_args_list[1].args[0], hidden[0, 9:14])
    loss.backward()
    assert hidden.grad[0, 2:7].abs().sum() > 0 and hidden.grad[0, 9:14].abs().sum() > 0
    assert hidden.grad[0, :2].eq(0).all() and hidden.grad[0, 7:9].eq(0).all()
    assert hidden.grad[0, 14:].eq(0).all()


def test_partial_last_subframe_is_trimmed_and_fractional_leading_subframe_is_masked():
    model = _ctc_model()
    hidden = torch.randn(1, 3, 16, requires_grad=True)
    target = _target(start=0, end=3, duration=0.18)
    codes = torch.zeros(1, 2, 4, dtype=torch.long)
    codes[:, :, 3] = model.audio_eos_id
    with patch.object(model.ctc_timestamp_head, 'compute_loss', wraps=model.ctc_timestamp_head.compute_loss) as head:
        _, metrics = model._compute_ctc_timestamp_loss(hidden, [target], codes, torch.tensor([4]))
    assert head.call_args.kwargs['input_length'] == 5
    assert head.call_args.kwargs['distillation_mask'].tolist() == [True] * 5
    assert metrics['supervised_frames'] == 5
    target = _target(start=0, end=3, duration=0.18, offset=0.02)
    with patch.object(model.ctc_timestamp_head, 'compute_loss', wraps=model.ctc_timestamp_head.compute_loss) as head:
        model._compute_ctc_timestamp_loss(hidden, [target], codes, torch.tensor([4]))
    assert head.call_args.kwargs['distillation_mask'].tolist() == [False, True, True, True, True]
    teacher_rows = head.call_args.args[2]['probabilities'][:, 0]
    torch.testing.assert_close(teacher_rows, target['ctc']['probabilities'][torch.tensor([0, 0, 1, 2, 3]), 0])


@pytest.mark.parametrize('reason', ['eos', 'user', 'padding', 'cropped', 'incomplete'])
def test_nonacoustic_or_incomplete_turns_do_not_train_lexical_ctc(reason):
    model = _ctc_model()
    hidden = torch.randn(1, 5, 16, requires_grad=True)
    target = _target(start=1, end=4, duration=0.24, offset=0.08)
    codes = torch.zeros(1, 2, 5, dtype=torch.long)
    lengths = torch.tensor([5])
    agent_mask = None
    if reason == 'eos':
        codes[:, :, 2] = model.audio_eos_id
    elif reason == 'user':
        agent_mask = torch.tensor([[True, True, False, True, True]])
    elif reason == 'padding':
        lengths[0] = 3
    elif reason == 'cropped':
        target['end_frame'] = 6
    else:
        target['complete_turn'] = False
    with patch.object(model.ctc_timestamp_head, 'compute_loss', side_effect=AssertionError('Invalid turn was used')):
        loss, metrics = model._compute_ctc_timestamp_loss(hidden, [target], codes, lengths, agent_mask)
    assert loss.item() == 0 and metrics['skipped_turns'] == 1 and metrics['supervised_turns'] == 0
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.ctc_timestamp_head.parameters())


def test_acoustic_mask_excludes_teacher_turns_and_crops_preserve_incomplete_metadata():
    model = _ctc_model(multiturn=True)
    turns = []
    text = _packed_text(model, timestamp_turns=turns).repeat(2, 1)
    batch = dict(
        audio=torch.zeros(2, 120),
        audio_lens=torch.tensor([120, 120]),
        timestamp_turns=[turns, turns],
        acoustic_loss_mask=torch.tensor([True, False]),
    )
    targets = []
    model._prepare_timestamp_targets(batch, text, torch.tensor([15, 15]), 12, ctc_targets=targets)
    assert len(targets) == 2 and all(target['batch_index'] == 0 for target in targets)
    assert targets[0]['complete_turn'] and not targets[1]['complete_turn']


@pytest.mark.parametrize('policy', [False, True])
def test_disabled_and_policy_scoring_paths_never_load_the_ctc_teacher(policy):
    model = _ctc_model(ctc_enabled=policy)
    with patch.object(model, '_get_timestamp_teacher', side_effect=AssertionError('Unneeded teacher loaded')):
        output = model.process_batch(**_inputs(model), mode='val', return_policy_logits=policy)
    assert output.ctc_timestamp_loss is None and output.timestamp_loss is None


@pytest.mark.parametrize('use_cfg', [False, True])
def test_streaming_ctc_is_immediate_and_preserves_audio_without_loading_teacher(use_cfg):
    baseline, observed = _ctc_model(ctc_enabled=False).eval(), _ctc_model().eval()
    states = [_init_stream(model, use_cfg=use_cfg) for model in (baseline, observed)]
    events = []
    tokens = [observed.bos_id, 7, 8, observed.eos_id, observed.pad_id, observed.pad_id, None, None]
    with patch.object(observed, '_get_timestamp_teacher', side_effect=AssertionError('Inference loaded teacher')):
        for token in tokens:
            results = [
                model.streaming_step(state, None if token is None else torch.full((2,), token))
                for model, state in zip((baseline, observed), states)
            ]
            states = [result[0] for result in results]
            torch.testing.assert_close(states[0].last_hidden, states[1].last_hidden, rtol=0, atol=0)
            rows = observed.drain_ctc_timestamp_frames(states[1])
            if results[0][1] is not None:
                torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
                assert len(rows) == 4
            else:
                assert not rows
            events.extend(rows)
            assert observed.drain_ctc_timestamp_frames(states[1]) == []
        output = observed.streaming_finalize(states[1])
    assert events
    count = len(events) // 2
    assert output.ctc_timestamp_lens.tolist() == [count, count]
    assert output.ctc_timestamp_tokens.shape == output.ctc_timestamp_probabilities.shape == (2, count)
    for sample in range(2):
        rows = [row for row in events if row['batch_index'] == sample]
        assert [row['frame_index'] for row in rows] == list(range(count))
        assert [row['start_seconds'] for row in rows] == pytest.approx([index * 0.04 for index in range(count)])
        assert all(row['is_blank'] == (row['token_id'] == 5) and 0 <= row['probability'] <= 1 for row in rows)
        assert [row['token_id'] for row in rows] == output.ctc_timestamp_tokens[sample].tolist()


def test_ctc_finalization_excludes_eos_and_keeps_two_ordered_probabilities_per_audio_frame():
    model = _ctc_model().eval()
    history = [torch.tensor([[step, 5], [5, step]]) for step in range(4)]
    probabilities = [torch.tensor([[0.2, 0.8], [0.7, 0.3]]) for _ in history]
    state = SimpleNamespace(
        config=SimpleNamespace(batch_size=2, device=torch.device('cpu')),
        all_predictions=[torch.zeros(2, 1, 1, dtype=torch.long) for _ in history],
        audio_prediction_start_idx=torch.tensor([0, 1]),
        audio_prediction_end_idx=torch.tensor([2, 4]),
        all_ctc_timestamp_predictions=history,
        all_ctc_timestamp_probabilities=probabilities,
    )
    tokens, confidence, lengths = model._finalize_ctc_timestamp_predictions(state)
    assert lengths.tolist() == [4, 6]
    assert tokens.tolist() == [[0, 5, 1, 5, IGNORE_INDEX, IGNORE_INDEX], [5, 1, 5, 2, 5, 3]]
    torch.testing.assert_close(confidence[0], torch.tensor([0.2, 0.8, 0.2, 0.8, 0.0, 0.0]))


@pytest.mark.parametrize('use_cfg', [False, True])
def test_streaming_ctc_excludes_argmax_eos_when_sampled_audio_codes_are_ordinary(use_cfg):
    model = _ctc_model().eval()
    state = _init_stream(model, use_cfg=use_cfg)
    calls = 0

    def predict(current, needs_audio=None):
        nonlocal calls
        calls += 1
        sampled = torch.zeros(2, model.num_audio_codebooks, dtype=torch.long)
        argmax = sampled.clone()
        if calls == 2:
            argmax[:, 0] = model.audio_eos_id
        return sampled, argmax

    model._predict_audio_codes = predict
    for token in [model.bos_id, 7, model.eos_id, model.pad_id, model.pad_id]:
        state, audio, _ = model.streaming_step(state, torch.full((2,), token))
        assert audio is None and model.drain_ctc_timestamp_frames(state) == []
    state, audio, _ = model.streaming_step(state, None)
    assert audio is not None and not state.finished.any()
    rows = model.drain_ctc_timestamp_frames(state)
    assert len(rows) == 4 and all(row['frame_index'] in (0, 1) for row in rows)
    state, audio, _ = model.streaming_step(state, None)
    assert audio.eq(0).all() and state.finished.all()
    assert model.drain_ctc_timestamp_frames(state) == []
    assert state.all_ctc_timestamp_predictions[-1].eq(IGNORE_INDEX).all()
    assert state.all_ctc_timestamp_probabilities[-1].eq(0).all()
    output = model.streaming_finalize(state)
    assert output.ctc_timestamp_lens.tolist() == [2, 2]
    for sample in range(2):
        observed = [row['token_id'] for row in rows if row['batch_index'] == sample]
        assert output.ctc_timestamp_tokens[sample].tolist() == observed


@pytest.mark.parametrize('profile', ['bulk', 'step'])
@pytest.mark.parametrize('use_cfg', [False, True])
def test_ctc_profile_prefill_keeps_aligned_padding_without_emitting_lexical_predictions(profile, use_cfg):
    model = _ctc_model().eval()
    state = _init_stream(model, use_cfg=use_cfg)
    with patch.object(model.ctc_timestamp_head, 'forward', side_effect=AssertionError('Profile ran CTC head')):
        if profile == 'bulk':
            state = model.streaming_prefill(state, torch.tensor([[7, 8, 9], [10, 11, 12]]))
            expected_steps = 3
        else:
            state, audio, _ = model.streaming_step(state, torch.tensor([7, 10]), prefill_like_step=True)
            assert audio is None
            expected_steps = 1
    assert len(state.all_ctc_timestamp_predictions) == expected_steps
    assert len(state.all_ctc_timestamp_probabilities) == expected_steps
    assert sum(prediction.size(-1) for prediction in state.all_predictions) == expected_steps
    assert all(prediction.shape == (2, 2) for prediction in state.all_ctc_timestamp_predictions)
    assert all(prediction.eq(IGNORE_INDEX).all() for prediction in state.all_ctc_timestamp_predictions)
    assert all(probabilities.eq(0).all() for probabilities in state.all_ctc_timestamp_probabilities)
    assert model.drain_ctc_timestamp_frames(state) == []
    tokens, probabilities, lengths = model._finalize_ctc_timestamp_predictions(state)
    assert lengths.tolist() == [0, 0]
    assert tokens.shape == probabilities.shape == (2, 0)


def test_ctc_tokenizer_is_packaged_and_restored_without_constructing_the_acoustic_teacher(tmp_path):
    model = _ctc_model(ctc_enabled=False)
    model.ctc_timestamp_head_config = {}
    metadata = dict(
        num_classes=6,
        blank_id=5,
        frame_seconds=0.04,
        upsampling_factor=2,
        tokenizer_options={},
        model_proto=b'tiny sentencepiece model',
    )
    tokenizer = SimpleNamespace(text_to_ids=lambda text: [1, 2])

    def register(config_path, path):
        assert config_path == 'ctc_timestamp_head.tokenizer_model'
        assert Path(path).read_bytes() == metadata['model_proto']
        return path

    loader_path = 'nemo.collections.tts.modules.nemotron_tts.v2.nemotron_asr_aligner.load_ctc_timestamp_tokenizer'
    with (
        patch(loader_path, return_value=(tokenizer, dict(metadata))) as loader,
        patch.object(model, 'register_artifact', side_effect=register) as artifact,
    ):
        model._init_ctc_timestamp_head()
    loader.assert_called_once()
    assert artifact.call_count == 1
    assert model.cfg.ctc_timestamp_head.num_classes == 6 and model.cfg.ctc_timestamp_head.blank_id == 5
    assert 'model_proto' not in model.cfg.ctc_timestamp_head
    restored = _ctc_model(ctc_enabled=False)
    restored.ctc_timestamp_head_config = OmegaConf.to_container(model.cfg.ctc_timestamp_head)
    assert Path(restored.ctc_timestamp_head_config['tokenizer_model']).is_file()
    with (
        patch(loader_path, return_value=(tokenizer, dict(metadata))),
        patch.object(restored, 'register_artifact', side_effect=register) as artifact,
        patch.object(restored, '_get_timestamp_teacher', side_effect=AssertionError('Teacher loaded')),
    ):
        restored._init_ctc_timestamp_head()
    assert artifact.call_count == 1 and restored.ctc_timestamp_tokenizer is tokenizer


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()), requires_grad=False)
        self.calls = []

    @torch.inference_mode()
    def forward(self, audio, lengths, sample_rate, texts, languages, **options):
        self.calls.append(
            dict(audio=audio.clone(), lengths=lengths.clone(), texts=texts, languages=languages, options=options)
        )
        records = []
        for length, text in zip(lengths, texts):
            duration = int(length) / sample_rate
            record = dict(error=None, words=[dict(word=text, start=0.0, end=duration)])
            if options.get('return_ctc'):
                record['ctc'] = _teacher_targets(duration)
            records.append(record)
        return records


def _teacher_targets(duration):
    frames = max(1, math.ceil(duration / 0.04 - 1e-6))
    probabilities = torch.stack([0.1 + torch.arange(frames) * 0.001, torch.full((frames,), 0.7)], dim=-1)
    return dict(
        token_ids=torch.tensor([1, 2]),
        indices=torch.tensor([[1, 5]]).repeat(frames, 1),
        probabilities=probabilities,
        tail_probabilities=1 - probabilities.sum(-1),
        frame_seconds=0.04,
        num_classes=6,
        blank_id=5,
    )


def _target(start, end, duration, offset=0.0):
    return dict(
        batch_index=0,
        start_frame=start,
        end_frame=end,
        audio_duration=duration,
        audio_start_seconds=offset,
        ctc=_teacher_targets(duration),
        complete_turn=True,
    )


def _batch():
    return dict(
        audio=torch.arange(64, dtype=torch.float32).unsqueeze(0),
        audio_lens=torch.tensor([64]),
        raw_texts=['ground truth transcript'],
        languages=['en'],
    )


def _ctc_model(ctc_enabled=True, binary_enabled=False, multiturn=False):
    model = _timestamp_model(enabled=binary_enabled, multiturn=multiturn)
    model.frame_stacking_factor = 1
    model.codec_model_samples_per_frame = 8
    model.final_proj = nn.Linear(16, model.num_all_tokens_per_codebook * 2)
    model.acoustic_codes_predictor = AcousticCodesPredictor(
        backbone_config=model.decoder.config,
        embed_codes=model.embed_audio_tokens,
        num_audio_codebooks=2,
        audio_eos_id=model.audio_eos_id,
        mask_token_id=model.mask_token_id,
        codebook_size=model.codebook_size,
        prediction_schedule=[1, 1],
        n_layers=1,
    )
    model._predict_audio_codes = lambda state, needs_audio=None: (
        torch.zeros(state.config.batch_size, 2, dtype=torch.long),
        torch.zeros(state.config.batch_size, 2, dtype=torch.long),
    )
    model.use_ctc_timestamp_head = ctc_enabled
    model.ctc_timestamp_head_config = dict(frame_seconds=0.04, distillation_topk=2, loss_weight=0.35)
    model.ctc_timestamp_head_loss_weight = 0.35
    model.ctc_timestamp_tokenizer = SimpleNamespace(ids_to_text=lambda ids: ' '.join(str(token) for token in ids))
    if ctc_enabled:
        model.ctc_timestamp_head = CTCTimestampHead(
            hidden_size=16,
            vocab_size=6,
            blank_id=5,
            projection_size=8,
            chunk_size=2,
            ctc_weight=0.6,
            distillation_weight=0.8,
        )
    object.__setattr__(model, '_timestamp_teacher', _Teacher())
    return model
