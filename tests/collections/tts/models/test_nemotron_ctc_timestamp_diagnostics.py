# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.collections.tts.models.test_nemotron_ctc_timestamp_head import _batch, _ctc_model
from tests.collections.tts.models.test_nemotron_timestamp_head import _packed_text
from tests.collections.tts.modules.nemotron_tts.v2.test_training_model import _inputs

pytestmark = pytest.mark.unit


def test_validation_decodes_the_existing_teacher_and_student_passes_and_keeps_the_selected_tts_input():
    model = _diagnostic_model()
    batch = _batch()
    batch.update(raw_texts=['AaB!'], normalized_texts=['a different reference'])
    calls = []

    def forward(hidden):
        calls.append(hidden.size(0))
        labels = torch.full((hidden.size(0) * 2,), 5, dtype=torch.long)
        if labels.numel():
            labels[:6] = torch.tensor([0, 0, 5, 0, 5, 1])
        logits = hidden.new_full((labels.numel(), 6), -8)
        logits.scatter_(-1, labels[:, None], 8)
        return logits.unflatten(0, (hidden.size(0), 2)) + hidden.sum() * 0

    with patch.object(model.ctc_timestamp_head, 'forward', side_effect=forward):
        output = model.process_batch(
            **_inputs(model), timestamp_batch=batch, mode='val', return_ctc_timestamp_diagnostics=True
        )
    assert len(model._timestamp_teacher.calls) == 1
    assert [frames for frames in calls if frames] == [8]
    assert output.ctc_timestamp_metrics['cer_errors'] == 0
    assert output.ctc_timestamp_metrics['cer_reference_characters'] == 3
    assert output.ctc_timestamp_metrics['cer_turns'] == 1
    assert output.ctc_timestamp_metrics['teacher_cer_errors'] == 3
    (diagnostic,) = output.ctc_timestamp_diagnostics
    assert diagnostic['reference_text'] == 'AaB!'
    assert diagnostic['student_text'] == 'aab' and diagnostic['teacher_text'] == ''
    assert diagnostic['student_cer'] == 0 and diagnostic['teacher_cer'] == 1
    assert len(diagnostic['student_token_ids']) == 16
    assert diagnostic['frame_start_seconds'][0] == 0
    assert diagnostic['frame_end_seconds'][-1] == pytest.approx(0.64)
    for key in ('student_token_ids', 'student_probabilities', 'valid_mask', 'teacher_probabilities'):
        assert diagnostic[key].device.type == 'cpu' and not diagnostic[key].requires_grad


def test_training_does_not_decode_or_capture_ctc_diagnostics():
    model = _diagnostic_model()
    with (
        patch.object(model.ctc_timestamp_tokenizer, 'ids_to_text', side_effect=AssertionError('Training decoded CTC')),
        patch.object(model.ctc_timestamp_head, 'compute_loss', wraps=model.ctc_timestamp_head.compute_loss) as head,
    ):
        output = model.process_batch(**_inputs(model), timestamp_batch=_batch(), mode='train')
    assert not head.call_args.kwargs.get('return_predictions', False)
    assert output.ctc_timestamp_diagnostics is None
    assert 'cer_errors' not in output.ctc_timestamp_metrics


@pytest.mark.parametrize('reference', ['$10', 'ten dollars'])
def test_ctc_metadata_preserves_the_exact_reference_chosen_for_tts(reference):
    model = _diagnostic_model()
    targets = []
    batch = _batch()
    batch.update(raw_texts=[reference], normalized_texts=['unused normalized text'])
    model._prepare_timestamp_targets(batch, _inputs(model)['text'], torch.tensor([8]), 9, ctc_targets=targets)
    assert targets[0]['reference_text'] == reference
    assert model._timestamp_teacher.calls[0]['texts'] == [reference]


def test_multiturn_ctc_diagnostics_use_independent_turn_references_and_fractional_times():
    model = _diagnostic_model(multiturn=True)
    turns = []
    text = _packed_text(model, timestamp_turns=turns, fractional=True, normalized=True)
    targets = []
    batch = dict(audio=torch.zeros(1, 120), audio_lens=torch.tensor([120]), timestamp_turns=[turns])
    model._prepare_timestamp_targets(batch, text, torch.tensor([15]), 16, ctc_targets=targets)
    assert [target['reference_text'] for target in targets] == ['first turn', 'second turn']
    assert [target['turn_index'] for target in targets] == [0, 1]
    diagnostics = []
    _, metrics = model._compute_ctc_timestamp_loss(
        torch.randn(1, 16, 16),
        targets,
        torch.zeros(1, 2, 16, dtype=torch.long),
        torch.tensor([16]),
        collect_predictions=True,
        diagnostics=diagnostics,
    )
    assert metrics['cer_turns'] == 2
    assert metrics['cer_reference_characters'] == len('first turn') + len('second turn')
    assert [(row['batch_index'], row['turn_index']) for row in diagnostics] == [(0, 0), (0, 1)]
    assert diagnostics[0]['frame_start_seconds'][0] == pytest.approx(0.03)
    assert diagnostics[0]['frame_end_seconds'][-1] <= 0.4
    assert diagnostics[1]['frame_start_seconds'][0] == pytest.approx(0.0)


def test_ctc_decoding_resets_at_each_turn_and_counts_all_characters_beyond_the_plot_limit():
    model = _diagnostic_model()
    captured = [_captured('a', [5, 5], [0, 5], sample=index, turn=0) for index in range(6)]
    captured.append(_captured('bbbbbbbbb', [1, 5] * 9, [1, 5] * 9, sample=6, turn=0))
    diagnostics = []
    counts = model._collect_ctc_timestamp_predictions(captured, diagnostics)
    assert counts == dict(cer_errors=6, teacher_cer_errors=0, cer_reference_characters=15, cer_turns=7)
    assert len(diagnostics) == 3
    # A blank-separated repeated character must survive normal CTC collapse.
    repeated = model._collect_ctc_timestamp_predictions([_captured('aa', [0, 0, 5, 0], [0, 5, 0, 0])])
    assert repeated['cer_errors'] == repeated['teacher_cer_errors'] == 0
    # Resetting on an invalid frame prevents adjacent turns or cropped rows from collapsing together.
    invalid = _captured('aa', [0, 1, 0], [0, 1, 0], valid=[True, False, True])
    assert model._collect_ctc_timestamp_predictions([invalid])['cer_errors'] == 0


def test_ctc_cer_uses_existing_tts_normalization_without_spoken_number_expansion():
    model = _diagnostic_model()
    assert model._ctc_timestamp_cer_counts('A-B!', 'a b') == (0, 3)
    assert model._ctc_timestamp_cer_counts('São Paulo.', 'são paulo') == (0, 9)
    errors, count = model._ctc_timestamp_cer_counts('$10', 'ten dollars')
    assert count == 2 and errors > count
    assert model._ctc_timestamp_cer_counts('!!!', '') == (0, 0)
    assert model._ctc_timestamp_cer_counts('!!!', 'ab') == (2, 0)


def test_ctc_cer_aggregates_character_counts_across_batches_and_ranks():
    model = _diagnostic_model()
    model.run_val_inference = False
    model._trainer = SimpleNamespace(strategy=SimpleNamespace(model=SimpleNamespace(require_backward_grad_sync=True)))
    model.validation_step_outputs = [
        dict(
            val_loss=torch.tensor(1.0),
            val_codebook_loss=torch.tensor(1.0),
            val_acoustic_codes_predictor_loss=torch.tensor(1.0),
        )
    ]
    logs = []
    with (
        patch.object(model, 'log'),
        patch.object(model, 'log_dict', side_effect=lambda values, **_: logs.append(values)),
    ):
        model.on_validation_epoch_start()
        for errors, teacher_errors, characters in [(1, 0, 1), (0, 2, 9)]:
            output = SimpleNamespace(
                ctc_timestamp_loss=torch.tensor(1.0),
                ctc_timestamp_metrics=dict(
                    cer_errors=errors,
                    teacher_cer_errors=teacher_errors,
                    cer_reference_characters=characters,
                    cer_turns=1,
                ),
            )
            model._log_ctc_timestamp_losses(output, 'val', 1)
        assert model._ctc_timestamp_validation_totals[:4].tolist() == [1, 2, 10, 2]
        assert not any('val/ctc_timestamp_cer' in values for values in logs)
        model._ctc_timestamp_validation_totals[4:] = torch.tensor([4, 10, 2])

        def reduce(totals):
            totals += torch.tensor([4, 3, 90, 3, 1, 90, 3])

        with (
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.all_reduce', side_effect=reduce) as reduction,
        ):
            model.on_validation_epoch_end()
        assert reduction.call_count == 1
        final = next(values for values in logs if 'val/ctc_timestamp_cer' in values)
        assert float(final['val/ctc_timestamp_cer']) == pytest.approx(0.05)
        assert float(final['val/ctc_timestamp_teacher_cer']) == pytest.approx(0.05)
        assert float(final['val/ctc_timestamp_generated_cer']) == pytest.approx(0.05)
        assert final['val/ctc_timestamp_reference_characters'] == 100
        model.on_validation_epoch_start()
        assert not model._ctc_timestamp_validation_totals.any()
        assert not any('ctc_timestamp_validation_totals' in key for key in model.state_dict())


def test_generated_ctc_cer_trims_padding_and_uses_the_exact_tts_input():
    model = _diagnostic_model()
    model.on_validation_epoch_start()
    generated = SimpleNamespace(
        ctc_timestamp_tokens=torch.tensor([[0, 0, 5, 1, 2, 2], [1, 5, 1, 2, 2, 2]]),
        ctc_timestamp_lens=torch.tensor([4, 3]),
    )
    model._update_generated_ctc_timestamp_cer(
        dict(raw_texts=['AB!', 'bb'], normalized_texts=['different', 'also different']), generated
    )
    assert model._ctc_timestamp_validation_totals[4:].tolist() == [0, 4, 2]
    model._update_generated_ctc_timestamp_cer(
        dict(timestamp_turns=[[dict(text='a'), dict(text='b')], [dict(text='bb')]]), generated
    )
    assert model._ctc_timestamp_validation_totals[4:].tolist() == [0, 6, 3]


def test_ctc_media_saves_teacher_student_plots_and_transcripts_without_running_models(tmp_path):
    model = _diagnostic_model()
    diagnostics = []
    model._collect_ctc_timestamp_predictions([_captured('ab', [0, 5, 1, 5], [0, 5, 1, 5])], diagnostics)
    output = SimpleNamespace(ctc_timestamp_diagnostics=diagnostics)
    model._trainer = SimpleNamespace(log_dir=str(tmp_path), global_step=42)
    prefix = 'nemo.collections.tts.models.nemotron_tts.v2.model'
    with (
        patch.object(model, '_get_timestamp_teacher', side_effect=AssertionError('Media reran teacher')),
        patch.object(model.ctc_timestamp_head, 'forward', side_effect=AssertionError('Media reran student')),
        patch(f'{prefix}.wandb.Image') as image,
        patch(f'{prefix}.wandb.Table') as table,
    ):
        media = model._log_ctc_timestamp_examples(output)
    assert image.call_count == table.call_count == 1
    assert 'CTC_Timestamp/Alignment_0' in media and 'CTC_Timestamp/Transcripts' in media
    directory = tmp_path / 'val_audios/ctc_timestamps_step_42'
    assert (directory / 'sample_0_turn_0.png').is_file()
    report = json.loads((directory / 'sample_0_turn_0.json').read_text())
    assert report['reference_text'] == report['teacher_text'] == report['student_text'] == 'ab'
    assert report['student_cer'] == report['teacher_cer'] == 0
    assert report['frame_start_seconds'] == pytest.approx([0, 0.04, 0.08, 0.12])
    assert report['student_token_ids'] == [0, 5, 1, 5]


class _Tokenizer:
    pieces = ['a', 'b', 'c', ' ', 'd']

    def ids_to_text(self, ids):
        return ''.join(self.pieces[index] for index in ids)

    def ids_to_tokens(self, ids):
        return [self.pieces[index] for index in ids]


def _diagnostic_model(multiturn=False):
    model = _ctc_model(multiturn=multiturn)
    model.ctc_timestamp_tokenizer = _Tokenizer()
    model.ctc_timestamp_head.chunk_size = 128
    return model


def _captured(reference, predicted, teacher, *, sample=0, turn=0, valid=None):
    assert len(predicted) == len(teacher)
    length = len(predicted)
    teacher = torch.tensor(teacher)
    alternate = torch.where(teacher.eq(5), 0, 5)
    aligned = dict(
        indices=torch.stack([teacher, alternate], dim=-1),
        probabilities=torch.tensor([[0.9, 0.1]]).repeat(length, 1),
        tail_probabilities=torch.zeros(length),
    )
    target = dict(
        batch_index=sample,
        turn_index=turn,
        reference_text=reference,
        audio_duration=length * 0.04,
        ctc=dict(token_ids=torch.tensor([0, 1]), frame_seconds=0.04),
    )
    result = dict(predicted_ids=torch.tensor(predicted), predicted_probabilities=torch.full((length,), 0.9))
    mask = torch.ones(length, dtype=torch.bool) if valid is None else torch.tensor(valid)
    return target, result, aligned, mask, torch.arange(length).double() * 0.04
