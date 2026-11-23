# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from lhotse import AudioSource, CutSet, MonoCut, Recording, SupervisionSegment
from torch import nn

from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import (
    build_token_channel,
    collate_token_channel,
)
from nemo.collections.tts.models.nemotron_tts.v2.base import TrainingMode
from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import (
    CONTINUE,
    END,
    IGNORE_INDEX,
    OUTSIDE,
    START,
    TIMESTAMP_LABELS,
    TimestampHead,
)
from tests.collections.tts.modules.nemotron_tts.v2.test_streaming_timestamps_model import _init_stream
from tests.collections.tts.modules.nemotron_tts.v2.test_training_model import _inputs, _training_model


pytestmark = pytest.mark.unit


def test_dataset_turn_boundaries_recover_exact_gt_audio_and_independent_transcripts():
    model = _timestamp_model(multiturn=True)
    turns = []
    text = _packed_text(model, timestamp_turns=turns)
    audio = torch.arange(120, dtype=torch.float32).unsqueeze(0)
    batch = dict(audio=audio, audio_lens=torch.tensor([120]), languages=['en'], timestamp_turns=[turns])
    labels, valid, metrics = model._prepare_timestamp_targets(batch, text, torch.tensor([30]), 16)
    teacher = model._timestamp_teacher
    assert teacher.calls[0]['texts'] == ['first turn', 'second turn']
    assert teacher.calls[0]['languages'] == ['en', 'en']
    assert teacher.calls[0]['lengths'].tolist() == [40, 40]
    torch.testing.assert_close(teacher.calls[0]['audio'][0], audio[0, 16:56])
    torch.testing.assert_close(teacher.calls[0]['audio'][1], audio[0, 72:112])
    assert labels[0, 2:7].tolist() == [OUTSIDE, START, END, OUTSIDE, OUTSIDE]
    assert labels[0, 9:14].tolist() == [OUTSIDE, START, END, OUTSIDE, OUTSIDE]
    assert valid[0].nonzero().flatten().tolist() == list(range(2, 7)) + list(range(9, 14))
    assert metrics['valid_turns'] == 2


def test_teacher_uses_unpadded_gt_audio_and_raw_reference_even_with_cached_codec_targets():
    model = _timestamp_model(multiturn=False)
    inputs = _inputs(model)
    batch = _raw_batch()
    batch['audio'] = torch.cat([batch['audio'], torch.full((1, 12), 999.0)], dim=1)
    model.process_batch(**inputs, timestamp_batch=batch, mode='val')
    call = model._timestamp_teacher.calls[0]
    assert call['texts'] == ['ground truth transcript']
    assert call['lengths'].tolist() == [32]
    torch.testing.assert_close(call['audio'], batch['audio'][:, :32])
    assert not call['audio'].requires_grad


def test_dataset_packed_turn_labels_land_on_raw_audio_targets_after_bos_and_eos_processing():
    model = _timestamp_model(multiturn=True)
    inputs = _inputs(model)
    turns = []
    inputs.update(
        text=_packed_text(model, timestamp_turns=turns),
        text_lens=torch.tensor([15]),
        audio_codes=torch.arange(60).reshape(1, 2, 30) % model.codebook_size,
        audio_codes_lens=torch.tensor([30]),
        context_text_tokens=torch.empty(1, 0, dtype=torch.long),
        context_text_tokens_lens=torch.tensor([0]),
    )
    batch = dict(
        audio=torch.arange(120, dtype=torch.float32)[None], audio_lens=torch.tensor([120]), timestamp_turns=[turns]
    )
    output = model.process_batch(
        **inputs, timestamp_batch=batch, training_mode=TrainingMode('streaming', 0, 1, 0), mode='val'
    )
    assert output.audio_codes_target.shape[-1] == 15
    # The artificial stacked BOS was removed from targets; target t=2 contains raw codec frames 4 and 5.
    assert output.audio_codes_target[0, :, 2].tolist() == [4, 5, 2, 3]
    assert output.timestamp_targets[0, 2:7].tolist() == [OUTSIDE, START, END, OUTSIDE, OUTSIDE]
    assert output.timestamp_targets[0, 9:14].tolist() == [OUTSIDE, START, END, OUTSIDE, OUTSIDE]
    assert not output.timestamp_mask[(output.audio_codes_target >= model.codebook_size).any(dim=1)].any()


def test_packed_teacher_requires_complete_source_metadata_instead_of_reconstructing_tokens():
    model = _timestamp_model(multiturn=True)
    text = _packed_text(model)
    with pytest.raises(ValueError, match='timestamp_turns'):
        model._prepare_timestamp_targets(
            dict(audio=torch.zeros(1, 120), audio_lens=torch.tensor([120])), text, torch.tensor([30]), 16
        )
    assert not model._timestamp_teacher.calls


def test_source_metadata_keeps_chosen_full_transcript_and_fractional_waveform_offset():
    model = _timestamp_model(multiturn=True)
    turns = []
    text = _packed_text(model, timestamp_turns=turns, fractional=True, normalized=True)
    text[0, 5] = model.pad_id  # The full reference survives even when packed conditioning is truncated.
    assert turns[0] == dict(text='first turn', start=0.13, end=0.53, language=None)
    model._timestamp_teacher.records = [
        {'words': [{'start': 0.05, 'end': 0.18}], 'error': None},
        {'words': [{'start': 0.08, 'end': 0.24}], 'error': None},
    ]
    audio = torch.arange(120, dtype=torch.float32)[None]
    labels, valid, metrics = model._prepare_timestamp_targets(
        dict(audio=audio, audio_lens=torch.tensor([120]), timestamp_turns=[turns]),
        text,
        torch.tensor([30]),
        16,
    )
    call = model._timestamp_teacher.calls[0]
    assert call['texts'] == ['first turn', 'second turn']
    torch.testing.assert_close(call['audio'][0], audio[0, 13:53])
    # Global word times .18-.31 quantize to [frame 2, frame 4), using the .13 source origin.
    assert labels[0, 2:7].tolist() == [START, END, OUTSIDE, OUTSIDE, OUTSIDE]
    assert valid[0, 2:7].all() and metrics['valid_turns'] == 2


def test_optional_dataset_collector_keeps_default_token_channel_and_groups_turns_per_cut():
    model = _timestamp_model(multiturn=True)
    kwargs = dict(
        cuts=CutSet.from_cuts([_packed_cut()]),
        tokenizer=model.tokenizer,
        frame_length=0.08,
        roles={'agent'},
        tokenizer_names=['bpe'],
        pad_id=model.pad_id,
        eos_id=model.eos_id,
        bos_id=model.bos_id,
        interruption_token_id=model.interruption_token_id,
    )
    baseline, baseline_lens = collate_token_channel(**kwargs)
    turns = []
    observed, observed_lens = collate_token_channel(**kwargs, timestamp_turns=turns)
    assert torch.equal(baseline, observed) and torch.equal(baseline_lens, observed_lens)
    assert [[turn['text'] for turn in sample] for sample in turns] == [['first turn', 'second turn']]


def test_one_step_audio_delay_reads_the_audio_prediction_state_and_masks_special_and_user_frames():
    model = _timestamp_model(multiturn=False)
    inputs = _inputs(model)
    inputs['context_text_tokens'] = torch.empty(1, 0, dtype=torch.long)
    inputs['context_text_tokens_lens'] = torch.tensor([0])
    mode = TrainingMode('streaming', 0, 1, 0)
    # Lexical token 8 arrives at time 2, after the first audio prediction state at time 1.
    assert inputs['text'][0, 2].item() == 8
    hidden, projected = [], []
    decoder_hook = model.decoder.register_forward_hook(
        lambda _module, _args, output: hidden.append(output.last_hidden_state)
    )
    head_hook = model.timestamp_head.register_forward_pre_hook(lambda _module, args: projected.append(args[0]))
    try:
        output = model.process_batch(
            **inputs,
            timestamp_batch=_raw_batch(),
            mode='val',
            training_mode=mode,
            agent_mask=torch.tensor([[True, False, True, True]]),
        )
    finally:
        decoder_hook.remove()
        head_hook.remove()
    frames = output.audio_codes_target.size(-1)
    torch.testing.assert_close(projected[0], hidden[0][:, 1 : 1 + frames], rtol=0, atol=0)
    assert output.timestamp_targets.shape == (1, frames)
    assert output.timestamp_targets[0].tolist() == [OUTSIDE, IGNORE_INDEX, END, OUTSIDE, IGNORE_INDEX]
    assert output.timestamp_mask[0].tolist() == [True, False, True, True, False]
    special = (output.audio_codes_target >= model.codebook_size).any(dim=1)
    assert not output.timestamp_mask[special].any()


def test_timestamp_loss_trains_head_and_backbone_while_teacher_is_excluded_from_state_and_optimizer():
    model = _timestamp_model(multiturn=False)
    output = model.process_batch(**_inputs(model), timestamp_batch=_raw_batch(), mode='val')
    output.timestamp_loss.backward()
    assert model.timestamp_head.projection[-1].weight.grad.abs().sum() > 0
    assert model.decoder.layers[2].mixer.q_proj.weight.grad.abs().sum() > 0
    teacher = model._timestamp_teacher
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters())
    assert '_timestamp_teacher' not in model._modules
    assert not any('timestamp_teacher' in name for name in model.state_dict())
    assert any(name.startswith('timestamp_head.') for name in model.state_dict())
    assert model._get_timestamp_teacher() is teacher
    model.setup_optimizer_param_groups()
    optimized = {id(parameter) for group in model._optimizer_param_groups for parameter in group['params']}
    assert id(teacher.weight) not in optimized
    assert id(model.timestamp_head.projection[-1].weight) in optimized


def test_external_teacher_is_constructed_only_once_when_first_requested():
    model = _timestamp_model(multiturn=False)
    object.__delattr__(model, '_timestamp_teacher')
    with patch(
        'nemo.collections.tts.modules.nemotron_tts.v2.nemotron_asr_aligner.NemotronASRTimestampAligner',
        return_value=_FrozenTeacher(),
    ) as construct:
        assert construct.call_count == 0
        first, second = model._get_timestamp_teacher(), model._get_timestamp_teacher()
    assert first is second and construct.call_count == 1
    assert '_timestamp_teacher' not in model._modules


def test_zero_timestamp_loss_weight_preserves_audio_logits_and_base_loss():
    baseline, observed = _timestamp_model(enabled=False, multiturn=False), _timestamp_model(multiturn=False)
    observed.timestamp_head_loss_weight = 0.0
    baseline_output = baseline.process_batch(**_inputs(baseline), mode='val')
    observed_output = observed.process_batch(**_inputs(observed), timestamp_batch=_raw_batch(), mode='val')
    torch.testing.assert_close(baseline_output.logits, observed_output.logits, rtol=0, atol=0)
    torch.testing.assert_close(baseline_output.loss, observed_output.loss, rtol=0, atol=0)
    assert observed_output.timestamp_loss.requires_grad
    assert baseline_output.timestamp_loss is None


def test_class_metrics_expose_missing_boundaries_when_every_prediction_is_continue():
    model = _timestamp_model(multiturn=False)
    model._timestamp_teacher.records = [{'words': [{'start': 0.0, 'end': 0.32}], 'error': None}]
    with torch.no_grad():
        model.timestamp_head.projection[-1].weight.zero_()
        model.timestamp_head.projection[-1].bias.zero_()
        model.timestamp_head.projection[-1].bias[CONTINUE] = 10
    output = model.process_batch(**_inputs(model), timestamp_batch=_raw_batch(), mode='val')
    metrics = output.timestamp_metrics
    assert metrics['turns'].dtype == torch.long
    assert metrics['continue_count'] == metrics['continue_correct'] == 2
    assert metrics['continue_recall'] == 1
    assert metrics['start_count'] == metrics['end_count'] == 1
    assert metrics['start_recall'] == metrics['end_recall'] == 0
    assert metrics['accuracy'] == 0.5


def test_validation_timestamp_metrics_use_global_frame_totals_and_reset_between_epochs():
    model = _timestamp_model(multiturn=False)
    model.run_val_inference = False
    model.validation_step_outputs = [
        dict(
            val_loss=torch.tensor(1.0),
            val_codebook_loss=torch.tensor(1.0),
            val_acoustic_codes_predictor_loss=torch.tensor(1.0),
        )
    ]
    batches = [([0, 1, 0, 1, 0], [0, 1, 0, 1, 0]), ([0, 0, 6, 0, 2], [0, 0, 3, 0, 0])]
    logs = []
    with (
        patch.object(model, 'log'),
        patch.object(model, 'log_dict', side_effect=lambda values, **_: logs.append(values)),
    ):
        model.on_validation_epoch_start()
        for batch_size, (counts, correct) in zip((1, 3), batches):
            metrics = dict(accuracy=torch.tensor(0.99), supervised_frames=torch.tensor(sum(counts)))
            for index, name in enumerate(TIMESTAMP_LABELS):
                metrics.update(
                    {
                        f'{name.lower()}_count': torch.tensor(counts[index]),
                        f'{name.lower()}_correct': torch.tensor(correct[index]),
                        f'{name.lower()}_recall': torch.tensor(0.99),
                    }
                )
            model._log_timestamp_losses(
                SimpleNamespace(timestamp_loss=torch.tensor(1.0), timestamp_metrics=metrics), 'val', batch_size
            )
        assert not any('val/timestamp_accuracy' in values for values in logs)
        model.on_validation_epoch_end()
        final = logs[-1]
        assert final['val/timestamp_supervised_frames'] == 10
        assert final['val/timestamp_accuracy'] == 0.5
        assert final['val/timestamp_start_count'] == final['val/timestamp_start_correct'] == 1
        assert final['val/timestamp_start_recall'] == 1
        assert final['val/timestamp_continue_count'] == 6
        assert final['val/timestamp_continue_recall'] == 0.5
        assert '_timestamp_validation_totals' not in model._buffers
        assert '_timestamp_validation_totals' not in model.state_dict()
        model.on_validation_epoch_start()
        assert not model._timestamp_validation_totals.any()


def test_invalid_teacher_turn_and_colliding_frames_are_ignored_without_retiming():
    model = _timestamp_model(multiturn=True)
    model._timestamp_teacher.records = [
        {'words': [], 'error': 'unreliable'},
        {
            'words': [{'start': 0.0, 'end': 0.16}, {'start': 0.08, 'end': 0.32}],
            'error': None,
        },
    ]
    turns = []
    text = _packed_text(model, timestamp_turns=turns)
    labels, valid, metrics = model._prepare_timestamp_targets(
        dict(audio=torch.zeros(1, 120), audio_lens=torch.tensor([120]), timestamp_turns=[turns]),
        text,
        torch.tensor([30]),
        16,
    )
    assert not valid[0, :9].any()
    assert labels[0, 9:14].tolist() == [START, IGNORE_INDEX, CONTINUE, END, OUTSIDE]
    assert metrics['invalid_turns'] == 1 and metrics['collision_frames'] == 1


def test_code_only_batches_require_gt_waveform_and_policy_scoring_never_loads_teacher():
    model = _timestamp_model(multiturn=False)
    with pytest.raises(ValueError, match='ground-truth'):
        model.process_batch(**_inputs(model), timestamp_batch={}, mode='val')
    with pytest.raises(ValueError, match='speaker-reference'):
        model.process_batch(
            **_inputs(model),
            timestamp_batch=dict(audio_16khz=torch.zeros(1, 100), audio_lens_16khz=torch.tensor([100])),
            mode='val',
        )
    with patch.object(model, '_get_timestamp_teacher', side_effect=AssertionError('Policy loaded timestamp teacher')):
        output = model.process_batch(**_inputs(model), return_policy_logits=True, mode='val')
    assert output.timestamp_loss is None


@pytest.mark.parametrize('use_cfg', [False, True])
def test_streaming_classes_are_passive_and_never_load_teacher(use_cfg):
    baseline = _timestamp_model(enabled=False).eval()
    observed = _timestamp_model().eval()
    states = [_init_stream(model, use_cfg=use_cfg) for model in (baseline, observed)]
    tokens = [observed.bos_id, 7, 8, observed.eos_id, observed.pad_id, observed.pad_id, None, None]
    with patch.object(observed, '_get_timestamp_teacher', side_effect=AssertionError('Inference loaded teacher')):
        for token in tokens:
            with torch.no_grad():
                for parameter in observed.timestamp_head.parameters():
                    parameter.normal_()
            results = [
                model.streaming_step(state, None if token is None else torch.full((2,), token))
                for model, state in zip((baseline, observed), states)
            ]
            states = [result[0] for result in results]
            torch.testing.assert_close(states[0].last_hidden, states[1].last_hidden, rtol=0, atol=0)
            if results[0][1] is not None:
                torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
        output = observed.streaming_finalize(states[1])
    assert output.timestamp_classes.shape[0] == 2
    assert (output.timestamp_classes_lens > 0).all()
    assert ((output.timestamp_classes >= 0) & (output.timestamp_classes < 5)).all()


class _ReferenceTokenizer:
    pad = 0

    def encode(self, text, tokenizer_name=None):
        return {'first turn': [7, 8], 'second turn': [9, 10]}[text]


class _FrozenTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()), requires_grad=False)
        self.calls = []
        self.records = None

    @torch.inference_mode()
    def forward(self, audio, lengths, sample_rate, texts, languages):
        self.calls.append(
            dict(
                audio=audio.clone(), lengths=lengths.clone(), sample_rate=sample_rate, texts=texts, languages=languages
            )
        )
        if self.records is not None:
            return self.records
        return [{'words': [{'word': text, 'start': 0.08, 'end': 0.24}], 'error': None} for text in texts]


def _timestamp_model(enabled=True, multiturn=True):
    model = _training_model(enabled=False, multiturn=multiturn)
    model._device = next(model.parameters()).device
    model.use_timestamp_head = enabled
    model.timestamp_head_config = {'projection_size': 128, 'loss_weight': 0.1, 'aligner': {}}
    model.timestamp_head_loss_weight = 0.1
    model.text_conditioning_tokenizer_name = 'bpe'
    model.tokenizer = _ReferenceTokenizer()
    if enabled:
        model.timestamp_head = TimestampHead(16)
        object.__setattr__(model, '_timestamp_teacher', _FrozenTeacher())
    return model


def _raw_batch():
    return dict(
        audio=torch.arange(32, dtype=torch.float32).unsqueeze(0),
        audio_lens=torch.tensor([32]),
        raw_texts=['ground truth transcript'],
        languages=['en'],
    )


def _packed_cut(fractional=False, normalized=False):
    return MonoCut(
        id='packed',
        start=0.0,
        duration=1.2,
        channel=0,
        recording=Recording(
            id='gt',
            sources=[AudioSource(type='file', channels=[0], source='unused.wav')],
            sampling_rate=100,
            num_samples=120,
            duration=1.2,
        ),
        supervisions=[
            SupervisionSegment(
                id='first',
                recording_id='gt',
                start=0.13 if fractional else 0.16,
                duration=0.40,
                text='raw first' if normalized else 'first turn',
                speaker='agent',
                custom={'normalized_text': 'first turn'} if normalized else None,
            ),
            SupervisionSegment(
                id='second', recording_id='gt', start=0.72, duration=0.40, text='second turn', speaker='agent'
            ),
        ],
    )


def _packed_text(model, timestamp_turns=None, fractional=False, normalized=False):
    text = build_token_channel(
        _packed_cut(fractional=fractional, normalized=normalized),
        model.tokenizer,
        frame_length=0.08,
        roles={'agent'},
        tokenizer_name='bpe',
        pad_id=model.pad_id,
        eos_id=model.eos_id,
        bos_id=model.bos_id,
        interruption_token_id=model.interruption_token_id,
        timestamp_turns=timestamp_turns,
    )
    return text.unsqueeze(0)
