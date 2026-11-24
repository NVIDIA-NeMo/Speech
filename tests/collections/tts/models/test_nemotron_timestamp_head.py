# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from lhotse import AudioSource, CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.utils import compute_num_frames
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

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
@pytest.mark.parametrize('lhotse', [False, True])
def test_timestamp_dataset_requirements_are_available_during_modelpt_initialization(multiturn, enabled, lhotse):
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
            train_ds=dict(
                use_lhotse=lhotse,
                volume_norm=True,
                dataset={},
                dataloader_params=dict(batch_size=2, num_workers=2),
            ),
        )
    )
    if enabled:
        config.use_timestamp_head = True
    codec = SimpleNamespace(
        sample_rate=100, output_sample_rate=100, num_codebooks=2, codebook_size=16, samples_per_frame=4
    )
    observed = {}
    loader_method = 'get_lhotse_dataloader' if lhotse else 'get_dataset'
    original_loader = getattr(NemotronTTSModel, loader_method)

    def record_loader(model, *args, **kwargs):
        observed['enabled'] = getattr(model, 'use_timestamp_head', False)
        observed['head_constructed'] = hasattr(model, 'timestamp_head')
        result = original_loader(model, *args, **kwargs)
        if not lhotse:
            raise LoaderConstructed
        return result

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
        patch('nemo.collections.tts.models.nemotron_tts.v2.model.safe_instantiate') as manifest_dataset,
        patch(
            'nemo.collections.tts.models.nemotron_tts.v2.model.FallbackDataset', side_effect=lambda dataset: dataset
        ),
        patch(
            'nemo.collections.tts.models.nemotron_tts.v2.model.get_lhotse_dataloader_from_config',
            side_effect=LoaderConstructed,
        ) as loader,
        patch.object(NemotronTTSModel, loader_method, record_loader),
        pytest.raises(LoaderConstructed),
    ):
        NemotronTTSModel(cfg=config)

    assert observed == dict(enabled=enabled, head_constructed=False)
    if not lhotse:
        manifest_dataset.assert_called_once()
        assert manifest_dataset.call_args.kwargs['load_cached_codes_if_available'] is (not enabled)
        loader.assert_not_called()
    elif multiturn:
        loader.assert_called_once()
        assert mt_dataset.call_args.kwargs['return_timestamp_supervision'] is enabled
        st_dataset.assert_not_called()
    else:
        loader.assert_called_once()
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
    assert valid[0].nonzero().flatten().tolist() == list(range(2, 7)) + list(range(9, 14))
    assert terminal[0].nonzero().flatten().tolist() == [8, 15]
    assert labels[terminal].tolist() == [IGNORE_INDEX, IGNORE_INDEX]
    assert metrics['valid_turns'] == 2


def test_acoustic_loss_mask_filters_whole_samples_without_per_turn_tensor_truth_checks():
    model = _timestamp_model(multiturn=True)
    turns = []
    text = _packed_text(model, timestamp_turns=turns).repeat(3, 1)
    audio = torch.arange(360, dtype=torch.float32).reshape(3, 120)
    batch = dict(
        audio=audio,
        audio_lens=torch.tensor([120, 120, 120]),
        timestamp_turns=[turns, turns, turns],
        languages=['en', 'es', 'pt'],
        acoustic_loss_mask=torch.tensor([True, False, True]),
    )
    with patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Per-turn CUDA synchronization')):
        labels, valid, metrics, terminal = model._prepare_timestamp_targets(
            batch, text, torch.tensor([30, 30, 30]), 16
        )
    call = model._timestamp_teacher.calls[0]
    assert call['texts'] == ['first turn', 'second turn', 'first turn', 'second turn']
    assert call['languages'] == ['en', 'en', 'pt', 'pt']
    expected_audio = torch.stack([audio[0, 16:56], audio[0, 72:112], audio[2, 16:56], audio[2, 72:112]])
    torch.testing.assert_close(call['audio'], expected_audio)
    assert metrics['turns'] == 6 and metrics['valid_turns'] == 4 and metrics['invalid_turns'] == 2
    torch.testing.assert_close(labels[0], labels[2])
    torch.testing.assert_close(valid[0], valid[2])
    assert labels[1].eq(IGNORE_INDEX).all() and not valid[1].any() and not terminal[1].any()


def test_terminal_candidates_preserve_the_next_turns_interior_labels():
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
    assert labels[0, 8:13].tolist() == [NONEND, NONEND, END, NONEND, NONEND]
    assert valid[0, 8] and not valid[0, 14]
    assert metrics['valid_turns'] == 2 and metrics['collision_frames'] == 0


@pytest.mark.parametrize('multiturn', [False, True])
def test_teacher_uses_unpadded_gt_audio_and_raw_reference_even_with_cached_codec_targets(multiturn):
    model = _timestamp_model(multiturn=multiturn)
    inputs = _inputs(model)
    batch = _raw_batch()
    batch['audio'] = torch.cat([batch['audio'], torch.full((1, 12), 999.0)], dim=1)
    model.process_batch(**inputs, timestamp_batch=batch, mode='val')
    call = model._timestamp_teacher.calls[0]
    assert call['texts'] == ['ground truth transcript']
    assert call['lengths'].tolist() == [32]
    torch.testing.assert_close(call['audio'], batch['audio'][:, :32])
    assert not call['audio'].requires_grad


def test_multiturn_model_validates_singleturn_batch_with_timestamp_supervision():
    model = _timestamp_model(multiturn=True)
    model._trainer = SimpleNamespace(world_size=1, global_rank=0, local_rank=0, loggers=[])
    model.run_val_inference = False
    model.validation_step_outputs = []
    batch = {**_inputs(model), **_raw_batch()}
    with patch.object(model, 'log_dict') as log:
        model.on_validation_epoch_start()
        output = model.validation_step(batch, batch_idx=1)
    assert torch.isfinite(output['val_loss'])
    assert model._timestamp_teacher.calls[0]['texts'] == batch['raw_texts']
    assert model._timestamp_validation_totals[0].sum() > 0
    assert any('val/timestamp_loss' in call.args[0] for call in log.call_args_list)


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
    # Actual speech EOS remains supervised even when the alignment endpoint was cropped.
    assert output.timestamp_targets[0, 14] == END and output.timestamp_mask[0, 14]
    other_specials = (output.audio_codes_target >= model.codebook_size).any(dim=1) & ~eos
    assert output.timestamp_targets[eos & output.timestamp_mask].tolist() == [END, END]
    assert output.timestamp_metrics['eos_alignment_total_count'] == 2
    assert output.timestamp_metrics['eos_alignment_missing_count'] == 1
    assert output.timestamp_metrics['eos_alignment_extra_count'] == 0
    assert not output.timestamp_mask[other_specials].any()


@pytest.mark.parametrize('layout_field', ['task', 'agent_mask'])
def test_packed_teacher_requires_complete_source_metadata_instead_of_reconstructing_tokens(layout_field):
    model = _timestamp_model(multiturn=True)
    text = _packed_text(model)
    batch = dict(audio=torch.zeros(1, 120), audio_lens=torch.tensor([120]), raw_texts=['packed transcript'])
    batch[layout_field] = ['tts'] if layout_field == 'task' else torch.ones_like(text, dtype=torch.bool)
    with pytest.raises(ValueError, match='timestamp_turns'):
        model._prepare_timestamp_targets(batch, text, torch.tensor([30]), 16)
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
    assert labels[0, 8] == IGNORE_INDEX and not valid[0, 8]


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
    assert labels[terminal].tolist() == [IGNORE_INDEX] and not valid[terminal].any()
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


def test_missing_alignment_terminal_is_reported_while_actual_eos_is_supervised():
    model = _timestamp_model(multiturn=False)
    # The waveform/codes contain four acoustic frames, but the supplied alignment
    # falsely places its final boundary after the first frame.
    model._timestamp_teacher.records = [{'words': [{'start': 0.0, 'end': 0.08}], 'error': None}]
    inputs = _inputs(model)
    labels, valid, _, terminal = model._prepare_timestamp_targets(
        _raw_batch(), inputs['text'], inputs['audio_codes_lens'], 5
    )
    assert terminal[0].nonzero().flatten().tolist() == [1]
    assert valid[0, 1] and labels[0, 1] == NONEND
    assert not terminal[0, 4]
    output = model.process_batch(**inputs, timestamp_batch=_raw_batch(), mode='val')
    assert output.timestamp_targets[0, 4] == END and output.timestamp_mask[0, 4]
    assert output.timestamp_targets[0, 1] == NONEND and output.timestamp_mask[0, 1]
    assert output.timestamp_metrics['eos_alignment_total_count'] == 1
    assert output.timestamp_metrics['eos_alignment_missing_count'] == 1
    assert output.timestamp_metrics['eos_alignment_extra_count'] == 1


def test_a_terminal_outside_the_retained_target_grid_is_ignored_without_moving_it():
    model = _timestamp_model(multiturn=False)
    inputs = _inputs(model)
    labels, valid, _, terminal = model._prepare_timestamp_targets(
        _raw_batch(), inputs['text'], inputs['audio_codes_lens'], 4
    )
    assert labels[0].tolist() == [NONEND, NONEND, END, NONEND]
    assert valid[0].all() and not terminal.any()


def test_timestamp_diagnostics_are_opt_in_detached_and_reuse_the_training_alignment():
    model = _timestamp_model(multiturn=False).eval()
    baseline = model.process_batch(**_inputs(model), timestamp_batch=_raw_batch(), mode='val')
    logits = []
    hook = model.timestamp_head.register_forward_hook(lambda _module, _args, output: logits.append(output.detach()))
    try:
        output = model.process_batch(
            **_inputs(model), timestamp_batch=_raw_batch(), mode='val', return_timestamp_diagnostics=True
        )
    finally:
        hook.remove()
    assert baseline.timestamp_diagnostics is None
    assert len(model._timestamp_teacher.calls) == 2 and len(logits) == 1
    torch.testing.assert_close(output.loss, baseline.loss, rtol=0, atol=0)
    torch.testing.assert_close(output.logits, baseline.logits, rtol=0, atol=0)
    (diagnostic,) = output.timestamp_diagnostics
    assert (diagnostic['batch_index'], diagnostic['turn_index']) == (0, 0)
    assert [word['word'] for word in diagnostic['words']] == ['ground truth transcript', 'last']
    assert diagnostic['frame_indices'].tolist() == [0, 1, 2, 3, 4]
    assert diagnostic['frame_end_seconds'].tolist() == pytest.approx([0.08, 0.16, 0.24, 0.32, 0.32])
    torch.testing.assert_close(diagnostic['end_probabilities'], logits[0][0].sigmoid(), rtol=0, atol=0)
    torch.testing.assert_close(diagnostic['targets'], output.timestamp_targets[0])
    torch.testing.assert_close(diagnostic['valid_mask'], output.timestamp_mask[0])
    assert diagnostic['eos_mask'].tolist() == [False, False, False, False, True]
    assert diagnostic['audio_duration'] == pytest.approx(0.32)
    for key in ('frame_indices', 'frame_end_seconds', 'end_probabilities', 'targets', 'valid_mask', 'eos_mask'):
        assert diagnostic[key].device.type == 'cpu' and not diagnostic[key].requires_grad


@pytest.mark.parametrize('text_input_mode', ['streaming', 'full'])
def test_timestamp_diagnostics_keep_fractional_turn_times_without_adding_the_speech_delay(text_input_mode):
    model = _timestamp_model(multiturn=True).eval()
    turns = []
    inputs = _inputs(model)
    inputs.update(
        text=_packed_text(model, timestamp_turns=turns, fractional=True),
        text_lens=torch.tensor([15]),
        audio_codes=torch.arange(60).reshape(1, 2, 30) % model.codebook_size,
        audio_codes_lens=torch.tensor([30]),
        context_text_tokens=torch.empty(1, 0, dtype=torch.long),
        context_text_tokens_lens=torch.tensor([0]),
    )
    batch = dict(audio=torch.zeros(1, 120), audio_lens=torch.tensor([120]), timestamp_turns=[turns])
    output = model.process_batch(
        **inputs,
        timestamp_batch=batch,
        mode='val',
        training_mode=TrainingMode(text_input_mode, 0, 1, 0),
        return_timestamp_diagnostics=True,
    )
    first, second = output.timestamp_diagnostics
    assert [(row['batch_index'], row['turn_index']) for row in output.timestamp_diagnostics] == [(0, 0), (0, 1)]
    assert (first['wave_start'], first['wave_end']) == (13, 53)
    assert first['frame_indices'].tolist() == [2, 3, 4, 5, 6, 8]
    assert first['frame_end_seconds'].tolist() == pytest.approx([0.11, 0.19, 0.27, 0.35, 0.4, 0.4])
    assert second['frame_indices'].tolist() == [9, 10, 11, 12, 13, 14]
    assert second['frame_end_seconds'].max() <= second['audio_duration']
    assert len(model._timestamp_teacher.calls) == 1


@pytest.mark.parametrize('invalid_first', [False, True])
def test_timestamp_diagnostics_collect_at_most_three_valid_turns_without_changing_target_returns(invalid_first):
    model = _timestamp_model(multiturn=True)
    turns = []
    text = _packed_text(model, timestamp_turns=turns).repeat(3, 1)
    batch = dict(audio=torch.zeros(3, 120), audio_lens=torch.tensor([120, 120, 120]), timestamp_turns=[turns] * 3)
    if invalid_first:
        model._timestamp_teacher.records = [
            {'words': [], 'error': 'invalid_reference'},
            *[{'words': [{'word': 'only', 'start': 0.0, 'end': 0.4}], 'error': None} for _ in range(5)],
        ]
    diagnostics = []
    result = model._prepare_timestamp_targets(batch, text, torch.tensor([30, 30, 30]), 16, diagnostics=diagnostics)
    assert len(result) == 4 and len(diagnostics) == 3
    expected = [(0, 1), (1, 0), (1, 1)] if invalid_first else [(0, 0), (0, 1), (1, 0)]
    assert [(row['batch_index'], row['turn_index']) for row in diagnostics] == expected
    assert len(model._timestamp_teacher.calls) == 1


def test_timestamp_diagnostics_show_unsupervised_audio_but_exclude_ineligible_prediction_frames():
    model = _timestamp_model(multiturn=True).eval()
    turns = []
    inputs = _inputs(model)
    inputs.update(
        text=_packed_text(model, timestamp_turns=turns),
        text_lens=torch.tensor([15]),
        audio_codes=torch.arange(60).reshape(1, 2, 30) % model.codebook_size,
        audio_codes_lens=torch.tensor([30]),
        context_text_tokens=torch.empty(1, 0, dtype=torch.long),
        context_text_tokens_lens=torch.tensor([0]),
    )
    turns[1].update(start=0.32, end=0.72, start_frame=4, end_frame=9)
    agent_mask = torch.ones(1, 15, dtype=torch.bool)
    agent_mask[0, 3] = False
    output = model.process_batch(
        **inputs,
        timestamp_batch=dict(audio=torch.zeros(1, 120), audio_lens=torch.tensor([120]), timestamp_turns=[turns]),
        agent_mask=agent_mask,
        mode='val',
        training_mode=TrainingMode('streaming', 0, 1, 0),
        return_timestamp_diagnostics=True,
    )
    diagnostic = output.timestamp_diagnostics[0]
    assert 3 not in diagnostic['frame_indices'].tolist()
    overlapping = diagnostic['frame_indices'] == 4
    assert overlapping.any() and not diagnostic['valid_mask'][overlapping].any()
    assert torch.isfinite(diagnostic['end_probabilities'][overlapping]).all()


@pytest.mark.parametrize('enabled,policy', [(False, False), (True, True)])
def test_timestamp_diagnostics_do_not_activate_a_disabled_or_policy_teacher(enabled, policy):
    model = _timestamp_model(enabled=enabled, multiturn=False)
    with patch.object(model, '_get_timestamp_teacher', side_effect=AssertionError('Unused teacher was loaded')):
        output = model.process_batch(
            **_inputs(model), mode='val', return_policy_logits=policy, return_timestamp_diagnostics=True
        )
    assert output.timestamp_diagnostics is None


@pytest.mark.parametrize(
    'rank,batch_index,has_wandb,enabled,collect',
    [
        (0, 0, True, True, True),
        (0, 1, True, True, False),
        (1, 0, True, True, False),
        (0, 0, False, True, False),
        (0, 0, True, False, False),
    ],
)
def test_validation_collects_timestamp_media_only_for_the_first_wandb_batch(
    rank, batch_index, has_wandb, enabled, collect
):
    model = _timestamp_model(enabled=enabled, multiturn=True).eval()
    model._fabric = None
    logger = MagicMock(spec=WandbLogger)
    model._trainer = SimpleNamespace(
        world_size=2, global_rank=rank, local_rank=rank, global_step=100, loggers=[logger] if has_wandb else []
    )
    model.run_val_inference = False
    model.validation_step_outputs = []
    with (
        patch.object(model, 'log_dict'),
        patch.object(model, 'log_val_audio_example', return_value={}),
        patch.object(model, '_log_timestamp_head_examples', return_value={'Timestamp/Stub': 'media'}) as media,
        patch.object(model, 'process_batch', wraps=model.process_batch) as process,
    ):
        model.on_validation_epoch_start()
        output = model.validation_step({**_inputs(model), **_raw_batch()}, batch_idx=batch_index)
    assert torch.isfinite(output['val_loss'])
    assert process.call_args.kwargs.get('return_timestamp_diagnostics', False) is collect
    assert media.call_count == int(collect)
    if collect:
        assert media.call_args.args[0].timestamp_diagnostics
        assert logger.experiment.log.call_args.args[0]['Timestamp/Stub'] == 'media'


@pytest.mark.parametrize('predict_eos', [False, True])
def test_timestamp_media_saves_marked_copies_without_forcing_predicted_ends(tmp_path, predict_eos):
    import json

    import numpy as np
    import soundfile as sf

    model = _timestamp_model(multiturn=False).eval()
    batch = _raw_batch()
    batch['audio'].zero_()
    with torch.no_grad():
        model.timestamp_head.projection[-1].weight.zero_()
        model.timestamp_head.projection[-1].bias.fill_(-10)
    output = model.process_batch(
        **_inputs(model), timestamp_batch=batch, mode='val', return_timestamp_diagnostics=True
    )
    model._trainer = SimpleNamespace(log_dir=str(tmp_path), global_step=100)
    classes = torch.zeros(1, 15, dtype=torch.long)
    classes[0, 2] = classes[0, 14] = END
    classes[0, 12] = END if predict_eos else NONEND
    generated = SimpleNamespace(
        predicted_audio=torch.zeros(1, 104),
        predicted_audio_lens=torch.tensor([96]),
        predicted_timestamp_classes=classes,
        predicted_timestamp_classes_lens=torch.tensor([13]),
        predicted_codes_lens=torch.tensor([24]),
    )
    generated.predicted_audio[0, 96:] = 0.9
    original_gt, original_generated = batch['audio'].clone(), generated.predicted_audio.clone()
    prefix = 'nemo.collections.tts.models.nemotron_tts.v2.model'
    with (
        patch(f'{prefix}.wandb.Image') as image,
        patch(f'{prefix}.wandb.Audio') as audio,
        patch.object(model, '_get_timestamp_teacher', side_effect=AssertionError('Media reran the ASR teacher')),
    ):
        ground_truth_media = model._log_timestamp_head_examples(output, batch)
        generated_media = model._log_timestamp_head_examples(output, batch, generated=generated)
    assert 'Timestamp/Alignment_0' in ground_truth_media and image.call_count == 1
    assert 'Timestamp/Generated_predicted_0' in generated_media and audio.call_count == 3
    directory = tmp_path / 'val_audios' / 'timestamps_step_100'
    assert (directory / 'sample_0_turn_0.png').is_file()
    teacher_forced = json.loads((directory / 'sample_0_turn_0.json').read_text())
    predicted = json.loads((directory / 'generated_sample_0.json').read_text())
    assert teacher_forced['predicted_end_seconds'] == []
    assert predicted['predicted_end_seconds'] == pytest.approx([0.24, 0.96] if predict_eos else [0.24])
    assert len(predicted['predicted_classes']) == 13
    assert predicted['frame_end_seconds'][-1] == pytest.approx(0.96)
    assert teacher_forced['sample_rate'] == predicted['sample_rate'] == 100
    assert predicted['audio_duration'] == pytest.approx(0.96)
    assert predicted['frame_seconds'] == pytest.approx(0.08)
    for name, expected_frames in [
        ('gt_reference_sample_0_turn_0.wav', 32),
        ('gt_predicted_sample_0_turn_0.wav', 32),
        ('generated_predicted_sample_0.wav', 96),
    ]:
        info = sf.info(directory / name)
        assert info.frames == expected_frames and info.samplerate == 100
    marked_reference, _ = sf.read(directory / 'gt_reference_sample_0_turn_0.wav', dtype='float32')
    unmarked_predictions, _ = sf.read(directory / 'gt_predicted_sample_0_turn_0.wav', dtype='float32')
    assert np.any(marked_reference) and not np.any(unmarked_predictions)
    torch.testing.assert_close(batch['audio'], original_gt, rtol=0, atol=0)
    torch.testing.assert_close(generated.predicted_audio, original_generated, rtol=0, atol=0)


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


def test_disabled_local_count_loss_preserves_bce_and_never_computes_windows():
    baseline = _timestamp_model(multiturn=False).eval()
    observed = _timestamp_model(
        multiturn=False, local_count_loss={'enabled': False, 'weight': 20.0, 'window_size': 3}
    ).eval()
    with patch.object(
        observed.timestamp_head,
        'compute_local_count_loss',
        side_effect=AssertionError('Disabled local count loss must not compute windows'),
    ):
        expected = baseline.process_batch(**_inputs(baseline), timestamp_batch=_raw_batch(), mode='val')
        actual = observed.process_batch(**_inputs(observed), timestamp_batch=_raw_batch(), mode='val')
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(actual.timestamp_loss, expected.timestamp_loss, rtol=0, atol=0)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    assert actual.timestamp_metrics.keys() == expected.timestamp_metrics.keys()
    assert 'local_count_loss' not in actual.timestamp_metrics


@pytest.mark.parametrize('text_input_mode,audio_delay', [('streaming', 1), ('full', 5)])
def test_local_count_loss_uses_the_masked_audio_grid_and_keeps_terminal_eos(text_input_mode, audio_delay):
    model = _timestamp_model(multiturn=False, local_count_loss={'enabled': True, 'weight': 2.0, 'window_size': 3})
    inputs = _inputs(model)
    inputs['context_text_tokens'] = torch.empty(1, 0, dtype=torch.long)
    inputs['context_text_tokens_lens'] = torch.tensor([0])
    hidden, projected = [], []
    decoder_hook = model.decoder.register_forward_hook(
        lambda _module, _args, output: hidden.append(output.last_hidden_state)
    )
    head_hook = model.timestamp_head.register_forward_pre_hook(lambda _module, args: projected.append(args[0]))
    logits = torch.logit(torch.tensor([[0.1, 0.99, 0.8, 0.7, 0.6]], requires_grad=True))
    try:
        with (
            patch.object(model.timestamp_head, 'forward', return_value=logits),
            patch.object(
                model.timestamp_head,
                'compute_local_count_loss',
                wraps=model.timestamp_head.compute_local_count_loss,
            ) as count,
        ):
            output = model.process_batch(
                **inputs,
                timestamp_batch=_raw_batch(),
                mode='val',
                training_mode=TrainingMode(text_input_mode, 0, 1, 0),
                agent_mask=torch.tensor([[True, False, True, True]]),
            )
    finally:
        decoder_hook.remove()
        head_hook.remove()
    frames = output.audio_codes_target.size(-1)
    torch.testing.assert_close(projected[0], hidden[0][:, audio_delay : audio_delay + frames], rtol=0, atol=0)
    assert output.timestamp_targets.tolist() == [[NONEND, IGNORE_INDEX, END, NONEND, END]]
    assert output.timestamp_mask.tolist() == [[True, False, True, True, True]]
    torch.testing.assert_close(count.call_args.args[1], output.timestamp_targets)
    torch.testing.assert_close(count.call_args.args[2], output.timestamp_mask)
    assert count.call_args.kwargs['eos_mask'].tolist() == [[False, False, False, False, True]]
    bce = F.binary_cross_entropy_with_logits(
        logits[output.timestamp_mask], output.timestamp_targets[output.timestamp_mask].float()
    )
    # The sole complete usable window has probabilities .8+.7+.6 and two END targets.
    expected_count = torch.tensor((0.8 + 0.7 + 0.6 - 2.0) ** 2)
    torch.testing.assert_close(output.timestamp_metrics['bce_loss'], bce.detach())
    torch.testing.assert_close(output.timestamp_metrics['local_count_loss'], expected_count)
    torch.testing.assert_close(output.timestamp_metrics['local_count_loss_weighted'], 2 * expected_count)
    torch.testing.assert_close(output.timestamp_loss, bce + 2 * expected_count)
    base_loss = (
        model.parallel_codebook_loss_scale * output.codebook_loss
        + model.acoustic_codes_predictor_loss_scale * output.acoustic_codes_predictor_loss
    )
    torch.testing.assert_close(output.loss, base_loss + model.timestamp_head_loss_weight * output.timestamp_loss)
    for name in ('bce_loss', 'local_count_loss', 'local_count_loss_weighted'):
        assert not output.timestamp_metrics[name].requires_grad


@pytest.mark.parametrize('name', ['local', 'cumulative'])
def test_count_loss_trains_head_and_backbone_and_reuses_metric_reduction(name):
    config = {'enabled': True, **({'window_size': 3} if name == 'local' else {})}
    model = _timestamp_model(multiturn=False, **{f'{name}_count_loss': config})
    counts = []
    method = f'compute_{name}_count_loss'
    compute = getattr(model.timestamp_head, method)

    def collect(*args, **kwargs):
        value = compute(*args, **kwargs)
        counts.append(value)
        return value

    with patch.object(model.timestamp_head, method, side_effect=collect):
        output = model.process_batch(**_inputs(model), timestamp_batch=_raw_batch(), mode='val')
    # The detached logging values do not sever the actual count-loss graph.
    counts[0].backward()
    assert model.timestamp_head.projection[-1].weight.grad.abs().sum() > 0
    assert model.decoder.layers[2].mixer.q_proj.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model._timestamp_teacher.parameters())
    reduced = []

    def reduce(packed, reduce_op):
        assert reduce_op == 'mean' and not packed.requires_grad
        reduced.append(packed)
        return packed

    model._trainer = SimpleNamespace(strategy=SimpleNamespace(reduce=reduce))
    with patch.object(model, 'log_dict') as log:
        model._log_timestamp_losses(output, 'train', batch_size=1)
    assert len(reduced) == 1
    for key in ('bce_loss', f'{name}_count_loss', f'{name}_count_loss_weighted'):
        torch.testing.assert_close(log.call_args.args[0][f'train/timestamp_{key}'], output.timestamp_metrics[key])


@pytest.mark.parametrize('local_enabled', [False, True])
def test_disabled_cumulative_count_preserves_existing_loss_and_never_computes_prefixes(local_enabled):
    local = {'enabled': local_enabled, 'window_size': 3}
    baseline = _timestamp_model(multiturn=False, local_count_loss=local).eval()
    observed = _timestamp_model(
        multiturn=False,
        local_count_loss=local,
        cumulative_count_loss={'enabled': False, 'weight': 100.0, 'delta': 10.0},
    ).eval()
    with patch.object(
        observed.timestamp_head,
        'compute_cumulative_count_loss',
        side_effect=AssertionError('Disabled cumulative count loss must not compute prefixes'),
    ):
        expected = baseline.process_batch(**_inputs(baseline), timestamp_batch=_raw_batch(), mode='val')
        actual = observed.process_batch(**_inputs(observed), timestamp_batch=_raw_batch(), mode='val')
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(actual.timestamp_loss, expected.timestamp_loss, rtol=0, atol=0)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    assert actual.timestamp_metrics.keys() == expected.timestamp_metrics.keys()
    assert 'cumulative_count_loss' not in actual.timestamp_metrics


@pytest.mark.parametrize('text_input_mode,audio_delay', [('streaming', 1), ('full', 5)])
@pytest.mark.parametrize(
    'local_enabled,cumulative_enabled', [(False, False), (False, True), (True, False), (True, True)]
)
def test_count_losses_coexist_on_the_masked_audio_prediction_grid(
    text_input_mode, audio_delay, local_enabled, cumulative_enabled
):
    model = _timestamp_model(
        multiturn=False,
        local_count_loss={'enabled': local_enabled, 'weight': 2.0, 'window_size': 3},
        cumulative_count_loss={'enabled': cumulative_enabled, 'weight': 0.1, 'delta': 1.0},
    )
    inputs = _inputs(model)
    inputs['context_text_tokens'] = torch.empty(1, 0, dtype=torch.long)
    inputs['context_text_tokens_lens'] = torch.tensor([0])
    hidden, projected = [], []
    decoder_hook = model.decoder.register_forward_hook(
        lambda _module, _args, output: hidden.append(output.last_hidden_state)
    )
    head_hook = model.timestamp_head.register_forward_pre_hook(lambda _module, args: projected.append(args[0]))
    values = torch.logit(torch.tensor([[0.1, 0.99, 0.8, 0.7, 0.6]], requires_grad=True))
    try:
        with (
            patch.object(model.timestamp_head, 'forward', return_value=values),
            patch.object(
                model.timestamp_head,
                'compute_cumulative_count_loss',
                wraps=model.timestamp_head.compute_cumulative_count_loss,
            ) as cumulative,
        ):
            output = model.process_batch(
                **inputs,
                timestamp_batch=_raw_batch(),
                mode='val',
                training_mode=TrainingMode(text_input_mode, 0, 1, 0),
                agent_mask=torch.tensor([[True, False, True, True]]),
            )
    finally:
        decoder_hook.remove()
        head_hook.remove()
    steps = output.audio_codes_target.size(-1)
    torch.testing.assert_close(projected[0], hidden[0][:, audio_delay : audio_delay + steps], rtol=0, atol=0)
    assert output.timestamp_targets.tolist() == [[NONEND, IGNORE_INDEX, END, NONEND, END]]
    assert output.timestamp_mask.tolist() == [[True, False, True, True, True]]
    assert cumulative.call_count == int(cumulative_enabled)
    bce = F.binary_cross_entropy_with_logits(
        values[output.timestamp_mask], output.timestamp_targets[output.timestamp_mask].float()
    )
    expected = bce
    metrics = output.timestamp_metrics
    if local_enabled or cumulative_enabled:
        torch.testing.assert_close(metrics['bce_loss'], bce.detach())
        assert not metrics['bce_loss'].requires_grad
    else:
        assert 'bce_loss' not in metrics
    if local_enabled:
        expected = expected + 2 * 0.01
        torch.testing.assert_close(metrics['local_count_loss'], torch.tensor(0.01))
    if cumulative_enabled:
        torch.testing.assert_close(cumulative.call_args.args[1], output.timestamp_targets)
        torch.testing.assert_close(cumulative.call_args.args[2], output.timestamp_mask)
        assert cumulative.call_args.kwargs['eos_mask'].tolist() == [[False, False, False, False, True]]
        # The masked frame splits regions of lengths one and three; EOS belongs to the latter.
        raw = torch.tensor((0.1**2 + ((-0.2) ** 2 + 0.5**2 + 0.1**2) / 3) / (2 * 4))
        torch.testing.assert_close(metrics['cumulative_count_loss'], raw)
        torch.testing.assert_close(metrics['cumulative_count_loss_weighted'], 0.1 * raw)
        assert not metrics['cumulative_count_loss'].requires_grad
        assert not metrics['cumulative_count_loss_weighted'].requires_grad
        expected = expected + 0.1 * raw
    else:
        assert 'cumulative_count_loss' not in metrics
    torch.testing.assert_close(output.timestamp_loss, expected)
    base = (
        model.parallel_codebook_loss_scale * output.codebook_loss
        + model.acoustic_codes_predictor_loss_scale * output.acoustic_codes_predictor_loss
    )
    torch.testing.assert_close(output.loss, base + model.timestamp_head_loss_weight * expected)


def test_zero_outer_timestamp_weight_preserves_base_loss_with_local_count_enabled():
    baseline = _timestamp_model(enabled=False, multiturn=False).eval()
    observed = _timestamp_model(multiturn=False, local_count_loss={'enabled': True, 'window_size': 3}).eval()
    observed.timestamp_head_loss_weight = 0.0
    expected = baseline.process_batch(**_inputs(baseline), mode='val')
    actual = observed.process_batch(**_inputs(observed), timestamp_batch=_raw_batch(), mode='val')
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    assert actual.timestamp_loss.requires_grad and actual.timestamp_metrics['local_count_loss'] > 0


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


@pytest.mark.parametrize(
    'local_counts,other_counts,world_size,expected_percent',
    [((1, 1), (9, 0), 2, 10.0), ((1, 1), (0, 0), 16, 100.0), ((0, 0), (1, 0), 16, 0.0), ((0, 0), (0, 0), 16, 0.0)],
)
def test_eos_missing_percentage_uses_the_ratio_of_globally_reduced_counts(
    local_counts, other_counts, world_size, expected_percent
):
    model = _timestamp_model(multiturn=False)
    output = SimpleNamespace(
        timestamp_loss=torch.tensor(0.7),
        timestamp_metrics={
            'eos_alignment_total_count': torch.tensor(local_counts[0]),
            'eos_alignment_missing_count': torch.tensor(local_counts[1]),
            'eos_alignment_extra_count': torch.tensor(1),
        },
    )
    calls = []

    def reduce(packed, reduce_op):
        assert reduce_op == 'mean' and packed.numel() == 4
        calls.append(packed.clone())
        return (packed + torch.tensor([0.0, *other_counts, 0.0])) / world_size

    model._trainer = SimpleNamespace(strategy=SimpleNamespace(reduce=reduce))
    with patch.object(model, 'log_dict') as log:
        model._log_timestamp_losses(output, 'train', batch_size=1)
    assert len(calls) == 1
    logged = log.call_args.args[0]
    percent = logged['train/timestamp_eos_alignment_missing_percent']
    assert torch.isfinite(percent) and percent.item() == pytest.approx(expected_percent)
    assert logged['train/timestamp_eos_alignment_extra_count'].item() == pytest.approx(1 / world_size)
    assert log.call_args.kwargs['sync_dist'] is False


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


def _timestamp_model(enabled=True, multiturn=True, local_count_loss=None, cumulative_count_loss=None):
    model = _training_model(enabled=False, multiturn=multiturn)
    model._device = next(model.parameters()).device
    model.use_timestamp_head = enabled
    model.timestamp_head_config = {'projection_size': 128, 'loss_weight': 0.1, 'aligner': {}}
    model.timestamp_head_loss_weight = 0.1
    model.text_conditioning_tokenizer_name = 'bpe'
    model.tokenizer = _ReferenceTokenizer()
    if enabled:
        model.timestamp_head = TimestampHead(
            16, local_count_loss=local_count_loss, cumulative_count_loss=cumulative_count_loss
        )
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
