# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from lhotse import AudioSource, CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.utils import compute_num_frames
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import (
    build_token_channel,
    collate_token_channel,
)
from nemo.collections.tts.models.nemotron_tts.v2.base import TrainingMode
from nemo.collections.tts.models.nemotron_tts.v2.model import NemotronTTSModel
from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import (
    END,
    IGNORE_INDEX,
    NONEND,
    TIMESTAMP_LABELS,
    TimestampHead,
)
from tests.collections.tts.modules.nemotron_tts.v2.test_streaming_timestamps_model import _init_stream
from tests.collections.tts.modules.nemotron_tts.v2.test_training_model import _inputs, _training_model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('multiturn', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_timestamp_dataset_requirements_are_available_during_modelpt_initialization(multiturn, enabled):
    class LoaderConstructed(Exception):
        pass

    config = OmegaConf.create(
        dict(
            codecmodel_path='unused.nemo',
            text_tokenizers={'bpe': {}},
            use_multiturn_dataset=multiturn,
            load_cached_codes_if_available=True,
            context_duration_min=3.0,
            context_duration_max=3.0,
            train_ds=dict(use_lhotse=True, volume_norm=True, dataset={}),
        )
    )
    if enabled:
        config.use_timestamp_head = True
    codec = SimpleNamespace(
        sample_rate=100, output_sample_rate=100, num_codebooks=2, codebook_size=16, samples_per_frame=4
    )
    observed = {}
    original_loader = NemotronTTSModel.get_lhotse_dataloader

    def record_loader(model, *args, **kwargs):
        observed['enabled'] = getattr(model, 'use_timestamp_head', False)
        observed['head_constructed'] = hasattr(model, 'timestamp_head')
        return original_loader(model, *args, **kwargs)

    def restore_codec(*args, **kwargs):
        return OmegaConf.create({}) if kwargs.get('return_config') else codec

    with (
        patch(
            'nemo.collections.tts.models.nemotron_tts.v2.base.AudioCodecModel.restore_from', side_effect=restore_codec
        ),
        patch(
            'nemo.collections.tts.models.nemotron_tts.v2.base.setup_tokenizers',
            return_value=SimpleNamespace(tokens=list(range(16)), pad=0),
        ),
        patch('nemo.collections.tts.models.nemotron_tts.v2.model.MagpieTTSLhotseMultiturnDataset') as mt_dataset,
        patch('nemo.collections.tts.models.nemotron_tts.v2.model.MagpieTTSLhotseDataset') as st_dataset,
        patch(
            'nemo.collections.tts.models.nemotron_tts.v2.model.FallbackDataset', side_effect=lambda dataset: dataset
        ),
        patch(
            'nemo.collections.tts.models.nemotron_tts.v2.model.get_lhotse_dataloader_from_config',
            side_effect=LoaderConstructed,
        ) as loader,
        patch.object(NemotronTTSModel, 'get_lhotse_dataloader', record_loader),
        pytest.raises(LoaderConstructed),
    ):
        NemotronTTSModel(cfg=config)

    assert observed == dict(enabled=enabled, head_constructed=False)
    loader.assert_called_once()
    if multiturn:
        assert mt_dataset.call_args.kwargs['return_timestamp_supervision'] is enabled
        st_dataset.assert_not_called()
    else:
        assert st_dataset.call_args.kwargs['load_cached_codes_if_available'] is (not enabled)
        mt_dataset.assert_not_called()


def test_dataset_turn_boundaries_recover_exact_gt_audio_and_independent_transcripts():
    model = _timestamp_model(multiturn=True)
    turns = []
    text = _packed_text(model, timestamp_turns=turns)
    audio = torch.arange(120, dtype=torch.float32).unsqueeze(0)
    batch = dict(audio=audio, audio_lens=torch.tensor([120]), languages=['en'], timestamp_turns=[turns])
    labels, valid, metrics, terminal = model._prepare_timestamp_targets(batch, text, torch.tensor([30]), 16)
    teacher = model._timestamp_teacher
    assert teacher.calls[0]['texts'] == ['first turn', 'second turn']
    assert teacher.calls[0]['languages'] == ['en', 'en']
    assert teacher.calls[0]['lengths'].tolist() == [40, 40]
    torch.testing.assert_close(teacher.calls[0]['audio'][0], audio[0, 16:56])
    torch.testing.assert_close(teacher.calls[0]['audio'][1], audio[0, 72:112])
    assert labels[0, 2:7].tolist() == [NONEND, NONEND, END, NONEND, NONEND]
    assert labels[0, 9:14].tolist() == [NONEND, NONEND, END, NONEND, NONEND]
    assert valid[0].nonzero().flatten().tolist() == list(range(2, 7)) + [8] + list(range(9, 14)) + [15]
    assert terminal[0].nonzero().flatten().tolist() == [8, 15]
    assert labels[terminal].tolist() == [END, END]
    assert metrics['valid_turns'] == 2


def test_terminal_end_survives_the_next_turns_first_nonend_frame():
    model = _timestamp_model(multiturn=True)
    turns = []
    text = _packed_text(model, timestamp_turns=turns)
    turns[1].update(start=0.64, end=1.04, start_frame=8, end_frame=13)
    labels, valid, metrics, terminal = model._prepare_timestamp_targets(
        dict(audio=torch.zeros(1, 120), audio_lens=torch.tensor([120]), timestamp_turns=[turns]),
        text,
        torch.tensor([30]),
        16,
    )
    # The first turn's terminal shares frame 8 with the next turn's initial NONEND.
    assert terminal[0].nonzero().flatten().tolist() == [8, 14]
    assert labels[0, 8:13].tolist() == [END, NONEND, END, NONEND, NONEND]
    assert valid[terminal].all() and (labels[terminal] == END).all()
    assert metrics['valid_turns'] == 2 and metrics['collision_frames'] == 0


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
    assert output.timestamp_targets[0, 2:7].tolist() == [NONEND, NONEND, END, NONEND, NONEND]
    assert output.timestamp_targets[0, 9:14].tolist() == [NONEND, NONEND, END, NONEND, NONEND]
    eos = (output.audio_codes_target == model.audio_eos_id).any(dim=1)
    # The first text marker injects EOS at target 8 despite the speech delay.
    # The second alignment-derived terminal at 15 lies past this target crop.
    assert eos[0].nonzero().flatten().tolist() == [8, 14]
    assert output.timestamp_targets[0, 8] == END and output.timestamp_mask[0, 8]
    # A separate target EOS cannot create an alignment-derived END label.
    assert output.timestamp_targets[0, 14] == IGNORE_INDEX and not output.timestamp_mask[0, 14]
    other_specials = (output.audio_codes_target >= model.codebook_size).any(dim=1) & ~eos
    assert output.timestamp_targets[eos & output.timestamp_mask].tolist() == [END]
    assert not output.timestamp_mask[other_specials].any()


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
    assert turns[0] == dict(
        text='first turn', start=0.13, end=0.53, language=None, start_frame=2, end_frame=7, has_end_marker=True
    )
    model._timestamp_teacher.records = [
        {'words': [{'start': 0.0, 'end': 0.18}, {'start': 0.18, 'end': 0.4}], 'error': None},
        {'words': [{'start': 0.0, 'end': 0.24}, {'start': 0.24, 'end': 0.4}], 'error': None},
    ]
    audio = torch.arange(120, dtype=torch.float32)[None]
    labels, valid, metrics, terminal = model._prepare_timestamp_targets(
        dict(audio=audio, audio_lens=torch.tensor([120]), timestamp_turns=[turns]),
        text,
        torch.tensor([30]),
        16,
    )
    call = model._timestamp_teacher.calls[0]
    assert call['texts'] == ['first turn', 'second turn']
    torch.testing.assert_close(call['audio'][0], audio[0, 13:53])
    # Global internal end .31 rounds to .32, so its preceding acoustic frame is 3.
    assert labels[0, 2:7].tolist() == [NONEND, END, NONEND, NONEND, NONEND]
    assert valid[0, 2:7].all() and metrics['valid_turns'] == 2
    # Terminal alignment .53 maps to native multiturn EOS round(.53 / .08) + 1 = 8.
    assert terminal[0].nonzero().flatten().tolist() == [8, 15]
    assert labels[0, 8] == END and valid[0, 8]


@pytest.mark.parametrize('fractional', [False, True])
def test_early_internal_end_stays_in_the_first_retained_turn_frame(fractional):
    model = _timestamp_model(multiturn=True)
    turns = []
    text = _packed_text(model, timestamp_turns=turns, fractional=fractional)
    model._timestamp_teacher.records = [
        {'words': [{'start': 0.0, 'end': 0.05}, {'start': 0.05, 'end': 0.4}], 'error': None},
        {'words': [{'start': 0.0, 'end': 0.05}, {'start': 0.05, 'end': 0.4}], 'error': None},
    ]
    labels, valid, metrics, terminal = model._prepare_timestamp_targets(
        dict(audio=torch.zeros(1, 120), audio_lens=torch.tensor([120]), timestamp_turns=[turns]),
        text,
        torch.tensor([30]),
        16,
    )
    # For the .13s origin, .18s is before the first retained frame's right edge;
    # it still closes the first word in that frame, instead of leaking to frame 1.
    assert labels[0, 2:7].tolist() == [END, NONEND, NONEND, NONEND, NONEND]
    assert labels[0, 9:14].tolist() == [END, NONEND, NONEND, NONEND, NONEND]
    assert valid[0, 2] and valid[0, 9] and metrics['valid_turns'] == 2


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


@pytest.mark.parametrize(
    'start,end,source_rate,legacy_metadata',
    [(0.72, 0.92, 24000, False), (0.72, 0.92, 24000, True), (0.0, 1.16, 24000, False), (0.0, 0.119979, 22050, False)],
)
def test_terminal_alignment_reuses_the_original_cut_grid_at_half_frames(start, end, source_rate, legacy_metadata):
    model = _timestamp_model(multiturn=True)
    model.sample_rate, model.codec_model_samples_per_frame = 24000, 960
    cut = _packed_cut()
    cut.duration = cut.recording.duration = 2.0
    cut.recording.sampling_rate = source_rate
    cut.recording.num_samples = 2 * source_rate
    cut.supervisions = cut.supervisions[:1]
    cut.supervisions[0].start, cut.supervisions[0].duration = start, end - start
    turns = []
    text = _packed_text(model, timestamp_turns=turns, cut=cut)
    expected_frame = compute_num_frames(cut.supervisions[0].end, 0.08, source_rate)
    if 'end_frame' in turns[0]:
        assert turns[0]['end_frame'] == expected_frame
    assert text[0, expected_frame] == model.interruption_token_id
    if legacy_metadata:
        for key in ('start_frame', 'end_frame', 'has_end_marker'):
            turns[0].pop(key, None)
    duration = (round(end * model.sample_rate) - round(start * model.sample_rate)) / model.sample_rate
    model._timestamp_teacher.records = [{'words': [{'start': 0.0, 'end': duration}], 'error': None}]
    labels, valid, metrics, terminal = model._prepare_timestamp_targets(
        dict(audio=torch.zeros(1, 48000), audio_lens=torch.tensor([48000]), timestamp_turns=[turns]),
        text,
        torch.tensor([50]),
        26,
    )
    assert terminal[0].nonzero().flatten().tolist() == [expected_frame + 1]
    assert labels[terminal].tolist() == [END] and valid[terminal].all()
    assert metrics['valid_turns'] == 1 and metrics['missing_end_markers'] == 0


@pytest.mark.parametrize('marker_problem', ['overwritten', 'truncated'])
def test_missing_packed_end_marker_keeps_word_labels_without_a_terminal_target(marker_problem):
    model = _timestamp_model(multiturn=True)
    cut = _packed_cut()
    if marker_problem == 'overwritten':
        cut.supervisions[1].start = cut.supervisions[0].end
    else:
        cut.supervisions = cut.supervisions[:1]
        cut.duration = cut.recording.duration = cut.supervisions[0].end
        cut.recording.num_samples = round(cut.duration * cut.sampling_rate)
    turns = []
    text = _packed_text(model, timestamp_turns=turns, cut=cut)
    assert turns[0]['has_end_marker'] is False
    samples = round(cut.duration * model.sample_rate)
    labels, valid, metrics, terminal = model._prepare_timestamp_targets(
        dict(audio=torch.zeros(1, samples), audio_lens=torch.tensor([samples]), timestamp_turns=[turns]),
        text,
        torch.tensor([30]),
        16,
    )
    assert not terminal[0, turns[0]['end_frame'] + 1]
    assert (labels[0, turns[0]['start_frame'] : turns[0]['end_frame']] == END).any()
    assert metrics['missing_end_markers'] == 1 and metrics['valid_turns'] == len(turns)
    assert valid.any()


def test_original_packing_grid_does_not_hide_an_incorrect_teacher_endpoint():
    model = _timestamp_model(multiturn=True)
    turns = []
    text = _packed_text(model, timestamp_turns=turns)
    model._timestamp_teacher.records = [
        {'words': [{'start': 0.0, 'end': 0.08}], 'error': None},
        {'words': [{'start': 0.0, 'end': 0.4}], 'error': None},
    ]
    _, _, _, terminal = model._prepare_timestamp_targets(
        dict(audio=torch.zeros(1, 120), audio_lens=torch.tensor([120]), timestamp_turns=[turns]),
        text,
        torch.tensor([30]),
        16,
    )
    assert terminal[0, 4] and not terminal[0, turns[0]['end_frame'] + 1]


@pytest.mark.parametrize('text_input_mode,audio_delay', [('streaming', 1), ('full', 5)])
def test_audio_delay_keeps_internal_and_terminal_ends_on_the_audio_prediction_grid(text_input_mode, audio_delay):
    model = _timestamp_model(multiturn=False)
    inputs = _inputs(model)
    inputs['context_text_tokens'] = torch.empty(1, 0, dtype=torch.long)
    inputs['context_text_tokens_lens'] = torch.tensor([0])
    mode = TrainingMode(text_input_mode, 0, 1, 0)
    # Full-text conditioning adds all four text positions before the speech delay.
    assert audio_delay == (int(inputs['text_lens'][0]) if text_input_mode == 'full' else 0) + 1
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
    torch.testing.assert_close(projected[0], hidden[0][:, audio_delay : audio_delay + frames], rtol=0, atol=0)
    assert output.timestamp_targets.shape == (1, frames)
    # The .24-second end belongs to acoustic frame 2, whose hidden state is delay+2.
    assert output.timestamp_targets[0].tolist() == [NONEND, IGNORE_INDEX, END, NONEND, END]
    assert output.timestamp_mask[0].tolist() == [True, False, True, True, True]
    special = (output.audio_codes_target >= model.codebook_size).any(dim=1)
    eos = (output.audio_codes_target == model.audio_eos_id).any(dim=1)
    assert eos[0].nonzero().flatten().tolist() == [4]
    assert output.timestamp_targets[eos].tolist() == [END]
    torch.testing.assert_close(projected[0][0, 4], hidden[0][0, audio_delay + 4], rtol=0, atol=0)
    assert not output.timestamp_mask[special & ~eos].any()


def test_terminal_target_comes_from_alignment_and_must_match_the_audio_eos_clock():
    model = _timestamp_model(multiturn=False)
    # The waveform/codes contain four acoustic frames, but the supplied alignment
    # falsely places its final boundary after the first frame.
    model._timestamp_teacher.records = [{'words': [{'start': 0.0, 'end': 0.08}], 'error': None}]
    inputs = _inputs(model)
    labels, valid, _, terminal = model._prepare_timestamp_targets(
        _raw_batch(), inputs['text'], inputs['audio_codes_lens'], 5
    )
    assert terminal[0].nonzero().flatten().tolist() == [1]
    assert valid[0, 1] and labels[0, 1] == END
    assert not terminal[0, 4]
    with pytest.raises(ValueError, match=r'(terminal|EOS|alignment)'):
        model.process_batch(**inputs, timestamp_batch=_raw_batch(), mode='val')


def test_a_terminal_outside_the_retained_target_grid_is_ignored_without_moving_it():
    model = _timestamp_model(multiturn=False)
    inputs = _inputs(model)
    labels, valid, _, terminal = model._prepare_timestamp_targets(
        _raw_batch(), inputs['text'], inputs['audio_codes_lens'], 4
    )
    assert labels[0].tolist() == [NONEND, NONEND, END, NONEND]
    assert valid[0].all() and not terminal.any()


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


def test_class_metrics_expose_missing_word_ends_when_every_prediction_is_nonend():
    model = _timestamp_model(multiturn=False)
    model._timestamp_teacher.records = [
        {'words': [{'start': 0.0, 'end': 0.08}, {'start': 0.08, 'end': 0.32}], 'error': None}
    ]
    with torch.no_grad():
        model.timestamp_head.projection[-1].weight.zero_()
        model.timestamp_head.projection[-1].bias.fill_(-10)
    output = model.process_batch(**_inputs(model), timestamp_batch=_raw_batch(), mode='val')
    metrics = output.timestamp_metrics
    assert metrics['turns'].dtype == torch.long
    assert metrics['nonend_count'] == metrics['nonend_correct'] == 3
    assert metrics['nonend_recall'] == 1
    assert metrics['end_count'] == 2
    assert metrics['end_recall'] == 0
    assert metrics['accuracy'] == 0.6


@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.parametrize('all_masked', [False, True])
def test_training_timestamp_metrics_share_one_detached_reduction(distributed, all_masked):
    model = _timestamp_model(multiturn=False)
    output = SimpleNamespace(
        timestamp_loss=torch.tensor(0.0 if all_masked else 0.7, requires_grad=True),
        timestamp_metrics={
            'valid_turns': torch.tensor(0 if all_masked else 2),
            'supervised_frames': torch.tensor(0 if all_masked else 9),
            'end_recall': torch.tensor(0.0 if all_masked else 0.5),
        },
    )
    local = torch.stack([output.timestamp_loss.detach(), *output.timestamp_metrics.values()]).float()
    other_rank = torch.tensor([0.9, 4.0, 17.0, 0.8])
    calls = []

    def reduce(packed, reduce_op):
        assert reduce_op == 'mean'
        assert not packed.requires_grad and packed.dtype == torch.float32
        calls.append(packed.clone())
        packed.add_(other_rank).div_(2)
        return packed

    model._trainer = SimpleNamespace(strategy=SimpleNamespace(reduce=reduce)) if distributed else None
    with patch.object(model, 'log_dict') as log:
        model._log_timestamp_losses(output, 'train', batch_size=3)

    assert len(calls) == int(distributed)
    logged = log.call_args.args[0]
    assert list(logged) == [
        'train/timestamp_loss',
        'train/timestamp_valid_turns',
        'train/timestamp_supervised_frames',
        'train/timestamp_end_recall',
    ]
    expected = (local + other_rank) / 2 if distributed else local
    torch.testing.assert_close(torch.stack(list(logged.values())), expected)
    torch.testing.assert_close(
        torch.stack([output.timestamp_loss.detach(), *output.timestamp_metrics.values()]).float(), local
    )
    assert output.timestamp_loss.requires_grad
    assert log.call_args.kwargs == dict(on_step=True, on_epoch=False, batch_size=3, sync_dist=False)


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
    batches = [([0, 2], [0, 2]), ([8, 0], [3, 0])]
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
        assert final['val/timestamp_end_count'] == final['val/timestamp_end_correct'] == 2
        assert final['val/timestamp_end_recall'] == 1
        assert final['val/timestamp_nonend_count'] == 8
        assert final['val/timestamp_nonend_recall'] == 0.375
        assert '_timestamp_validation_totals' not in model._buffers
        assert '_timestamp_validation_totals' not in model.state_dict()
        model.on_validation_epoch_start()
        assert not model._timestamp_validation_totals.any()


def test_invalid_teacher_turn_is_ignored_and_shared_ends_remain_binary_positive():
    model = _timestamp_model(multiturn=True)
    model._timestamp_teacher.records = [
        {'words': [], 'error': 'unreliable'},
        {
            'words': [{'start': 0.0, 'end': 0.05}, {'start': 0.05, 'end': 0.06}, {'start': 0.06, 'end': 0.4}],
            'error': None,
        },
    ]
    turns = []
    text = _packed_text(model, timestamp_turns=turns)
    labels, valid, metrics, terminal = model._prepare_timestamp_targets(
        dict(audio=torch.zeros(1, 120), audio_lens=torch.tensor([120]), timestamp_turns=[turns]),
        text,
        torch.tensor([30]),
        16,
    )
    assert not valid[0, :9].any()
    assert labels[0, 9:14].tolist() == [END, NONEND, NONEND, NONEND, NONEND]
    assert metrics['invalid_turns'] == 1 and metrics['collision_frames'] == 1
    assert metrics['collision_words'] == 1 and valid[0, 9]


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
    assert ((output.timestamp_classes >= 0) & (output.timestamp_classes < 2)).all()


def test_timestamp_finalization_retains_eos_predictions_without_changing_other_passive_heads():
    model = _timestamp_model().eval()
    stacking = model.frame_stacking_factor
    history = [
        torch.tensor([NONEND, NONEND, NONEND]),
        torch.tensor([NONEND, NONEND, NONEND]),
        torch.tensor([END, NONEND, NONEND]),
        torch.tensor([NONEND, END, NONEND]),
    ]
    state = SimpleNamespace(
        config=SimpleNamespace(batch_size=3, device=torch.device('cpu')),
        all_predictions=[torch.zeros(3, 1, stacking, dtype=torch.long) for _ in history],
        audio_prediction_start_idx=torch.tensor([0, stacking, 0]),
        # First item has EOS in an EOS-only step. Second item has an acoustic
        # subframe before EOS in its final stacked step. Third has not emitted EOS.
        audio_prediction_end_idx=torch.tensor([2 * stacking, 4 * stacking - 1, -1]),
    )
    previous, previous_lens = model._finalize_frame_predictions(state, history, IGNORE_INDEX)
    observed, observed_lens = model._finalize_frame_predictions(state, history, IGNORE_INDEX, include_eos=True)
    assert previous_lens.tolist() == [2, 3, 4]
    assert observed_lens.tolist() == [3, 3, 4]
    assert observed[0, 2] == END and observed[0, 3] == IGNORE_INDEX
    assert observed[1, 2] == END
    torch.testing.assert_close(previous[2], observed[2], rtol=0, atol=0)


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
        return [
            {
                'words': [
                    {'word': text, 'start': 0.0, 'end': 0.24},
                    {'word': 'last', 'start': 0.24, 'end': int(length) / sample_rate},
                ],
                'error': None,
            }
            for text, length in zip(texts, lengths)
        ]


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


def _packed_text(model, timestamp_turns=None, fractional=False, normalized=False, cut=None):
    text = build_token_channel(
        _packed_cut(fractional=fractional, normalized=normalized) if cut is None else cut,
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
