# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.models.nemotron_tts.v2.base import TrainingMode
from nemo.collections.tts.models.nemotron_tts.v2.model import NemotronTTSModel
from tests.collections.tts.models.test_nemotron_ctc_timestamp_head import _batch
from tests.collections.tts.models.test_nemotron_timestamp_head import _packed_text
from tests.collections.tts.models.test_nemotron_tts_ctc_head import _native_model
from tests.collections.tts.modules.nemotron_tts.v2.test_streaming_timestamps_model import _init_stream
from tests.collections.tts.modules.nemotron_tts.v2.test_training_model import _inputs

pytestmark = pytest.mark.unit

MAPPING = 'nemo.collections.tts.models.nemotron_tts.v2.model.map_tts_ctc_word_intervals'


@pytest.mark.parametrize('asr_enabled', [False, True])
@pytest.mark.parametrize('word_enabled', [False, True])
def test_word_ctc_reuses_one_teacher_pass_and_leaves_disabled_path_unchanged(asr_enabled, word_enabled):
    model = _word_model(asr_enabled=asr_enabled, enabled=word_enabled)
    with (
        patch(MAPPING, side_effect=_mapping) as mapping,
        patch.object(model.tts_ctc_head, 'compute_loss', wraps=model.tts_ctc_head.compute_loss) as native,
    ):
        output = model.process_batch(**_inputs(model), timestamp_batch=_batch(), mode='val')
    assert len(model._timestamp_teacher.calls) == int(asr_enabled or word_enabled)
    assert mapping.call_count == int(word_enabled)
    assert native.call_count == 1
    assert ('word_intervals' in native.call_args.kwargs) is word_enabled
    if word_enabled:
        assert output.tts_ctc_metrics['word_ctc_valid_intervals'] == 2
        torch.testing.assert_close(
            output.tts_ctc_loss,
            output.tts_ctc_metrics['ctc_loss']
            + model.tts_ctc_word_interval_config['weight'] * output.tts_ctc_metrics['word_ctc_loss'],
        )
    if asr_enabled:
        options = model._timestamp_teacher.calls[0]['options']
        assert options['return_ctc'] and options['align_words'] is word_enabled


def test_word_records_are_attached_by_sample_and_turn_keys_and_include_fractional_phase():
    model = _word_model(multiturn=True, tolerance=0.0)
    turns = []
    text = _packed_text(model, timestamp_turns=turns, fractional=True)
    targets = model._prepare_tts_ctc_targets(
        dict(timestamp_turns=[turns]), text, torch.tensor([15]), torch.tensor([15]), 16
    )
    records = [
        dict(batch_index=0, turn_index=1, words=[dict(word='second')]),
        dict(batch_index=1, turn_index=0, words=[dict(word='other sample')]),
        dict(batch_index=0, turn_index=0, words=[dict(word='first')]),
    ]
    with patch(MAPPING, side_effect=_mapping) as mapping:
        metrics = model._attach_tts_ctc_word_intervals(targets, records)
    assert [call.args[3][0]['word'] for call in mapping.call_args_list] == ['first', 'second']
    assert [call.args[2].tolist() for call in mapping.call_args_list] == [[7, 8], [9, 10]]
    assert targets[0]['word_intervals'] == [
        dict(token_start=0, token_end=1, start_frame=-1, end_frame=5),
        dict(token_start=1, token_end=2, start_frame=4, end_frame=10),
    ]
    assert targets[1]['word_intervals'] == [
        dict(token_start=0, token_end=1, start_frame=0, end_frame=5),
        dict(token_start=1, token_end=2, start_frame=5, end_frame=10),
    ]
    assert metrics['word_interval_mapping_turns'] == 2
    assert metrics['word_interval_mapped_tokens'] == metrics['word_interval_target_tokens'] == 4


@pytest.mark.parametrize('text_mode,delay', [('streaming', 1), ('full', 16)])
def test_fractional_multiturn_word_windows_use_audio_hidden_states_after_speech_delay_once(text_mode, delay):
    model = _word_model(multiturn=True, tolerance=0.0)
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
    batch = dict(
        audio=torch.arange(120, dtype=torch.float32).unsqueeze(0),
        audio_lens=torch.tensor([120]),
        timestamp_turns=[turns],
        languages=['pt'],
    )
    hidden = []
    hook = model.decoder.register_forward_hook(lambda _module, _args, result: hidden.append(result.last_hidden_state))
    try:
        with (
            patch(MAPPING, side_effect=_mapping),
            patch.object(model.tts_ctc_head, 'compute_loss', wraps=model.tts_ctc_head.compute_loss) as native,
        ):
            output = model.process_batch(
                **inputs, timestamp_batch=batch, mode='val', training_mode=TrainingMode(text_mode, 0, 1, 0)
            )
    finally:
        hook.remove()
    assert len(model._timestamp_teacher.calls) == 1 and native.call_count == 2
    for call, start, end in zip(native.call_args_list, [2, 9], [7, 14]):
        torch.testing.assert_close(call.args[0], hidden[0][0, delay + start : delay + end], atol=0, rtol=0)
    assert native.call_args_list[0].kwargs['word_intervals'][0]['start_frame'] == -1
    assert output.tts_ctc_metrics['word_ctc_valid_intervals'] == 4


@pytest.mark.parametrize('failure', ['teacher', 'mapping'])
def test_unusable_word_alignment_keeps_native_global_ctc_supervision(failure):
    model = _word_model()
    teacher = model._timestamp_teacher
    original_forward = teacher.forward

    def forward(*args, **kwargs):
        records = original_forward(*args, **kwargs)
        if failure == 'teacher':
            for record in records:
                record.update(error='alignment failed', words=[])
        return records

    mapping_result = dict(error='input_token_mismatch', segments=[], merged_word_groups=0, token_coverage=0.0)
    with patch.object(teacher, 'forward', side_effect=forward), patch(MAPPING, return_value=mapping_result) as mapping:
        output = model.process_batch(**_inputs(model), timestamp_batch=_batch(), mode='val')
    assert mapping.call_count == int(failure == 'mapping')
    metrics = output.tts_ctc_metrics
    assert metrics['supervised_turns'] == 1 and metrics['word_ctc_valid_intervals'] == 0
    assert metrics[f'word_interval_{"teacher" if failure == "teacher" else "mapping"}_failed_turns'] == 1
    torch.testing.assert_close(output.tts_ctc_loss, metrics['ctc_loss'])
    output.tts_ctc_loss.backward()
    assert model.tts_ctc_head.projection[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_local_word_loss_trains_native_head_and_backbone_without_teacher_gradients():
    model = _word_model(asr_enabled=True)
    with patch(MAPPING, side_effect=_mapping):
        output = model.process_batch(**_inputs(model), timestamp_batch=_batch(), mode='val')
    assert output.tts_ctc_metrics['word_ctc_loss'] > 0
    output.tts_ctc_loss.backward()
    assert model.tts_ctc_head.projection[-1].weight.grad.abs().sum() > 0
    assert model.decoder.layers[2].mixer.q_proj.weight.grad.abs().sum() > 0
    assert model.ctc_timestamp_head.projection[-1].weight.grad is None
    assert all(parameter.grad is None for parameter in model._timestamp_teacher.parameters())


def test_word_supervision_does_not_load_asr_or_map_words_during_streaming_or_policy_scoring():
    model = _word_model().eval()
    state = _init_stream(model, use_cfg=False)
    with (
        patch.object(model, '_get_timestamp_teacher', side_effect=AssertionError('Inference loaded ASR')),
        patch(MAPPING, side_effect=AssertionError('Inference mapped word targets')),
    ):
        for token in [model.bos_id, 7, 8, model.eos_id, model.pad_id, model.pad_id, None, None]:
            state, *_ = model.streaming_step(state, None if token is None else torch.full((2,), token))
            model.drain_ctc_timestamp_frames(state, source='tts')
        output = model.streaming_finalize(state)
        policy = model.process_batch(**_inputs(model), mode='val', return_policy_logits=True)
    assert output.tts_ctc_tokens is not None
    assert policy.tts_ctc_loss is None


@pytest.mark.parametrize('multiturn', [False, True])
def test_word_teacher_waveform_requirement_is_available_before_modelpt_constructs_dataloader(multiturn):
    class LoaderConstructed(Exception):
        pass

    cfg = OmegaConf.create(
        dict(
            codecmodel_path='unused.nemo',
            text_tokenizers={'bpe': {}},
            use_multiturn_dataset=multiturn,
            load_cached_codes_if_available=True,
            context_duration_min=3.0,
            context_duration_max=3.0,
            use_tts_ctc_head=True,
            tts_ctc_head=dict(word_interval=dict(enabled=True)),
            train_ds=dict(
                use_lhotse=True, volume_norm=True, dataset={}, dataloader_params=dict(batch_size=2, num_workers=2)
            ),
        )
    )
    codec = SimpleNamespace(
        sample_rate=100, output_sample_rate=100, num_codebooks=2, codebook_size=16, samples_per_frame=8
    )
    observed = {}
    original = NemotronTTSModel.get_lhotse_dataloader

    def loader(model, *args, **kwargs):
        observed.update(enabled=model.use_tts_ctc_word_intervals, head_constructed=hasattr(model, 'tts_ctc_head'))
        return original(model, *args, **kwargs)

    def restore(*args, **kwargs):
        return OmegaConf.create({}) if kwargs.get('return_config') else codec

    prefix = 'nemo.collections.tts.models.nemotron_tts.v2'
    with (
        patch(f'{prefix}.base.AudioCodecModel.restore_from', side_effect=restore),
        patch(f'{prefix}.base.setup_tokenizers', return_value=SimpleNamespace(tokens=list(range(60)), pad=0)),
        patch(f'{prefix}.model.MagpieTTSLhotseMultiturnDataset') as mt,
        patch(f'{prefix}.model.MagpieTTSLhotseDataset') as st,
        patch(f'{prefix}.model.FallbackDataset', side_effect=lambda dataset: dataset),
        patch(f'{prefix}.model.get_lhotse_dataloader_from_config', side_effect=LoaderConstructed),
        patch.object(NemotronTTSModel, 'get_lhotse_dataloader', loader),
        pytest.raises(LoaderConstructed),
    ):
        NemotronTTSModel(cfg)
    assert observed == dict(enabled=True, head_constructed=False)
    dataset = mt if multiturn else st
    if multiturn:
        assert dataset.call_args.kwargs['return_timestamp_supervision']
    else:
        assert dataset.call_args.kwargs['load_cached_codes_if_available'] is False


def _mapping(tokenizer, text, ids, words, duration):
    assert len(ids) == 2
    return dict(
        error=None,
        merged_word_groups=0,
        token_coverage=1.0,
        segments=[
            dict(token_start=0, token_end=1, start=0.0, end=duration / 2),
            dict(token_start=1, token_end=2, start=duration / 2, end=duration),
        ],
    )


def _word_model(asr_enabled=False, enabled=True, multiturn=False, tolerance=0.08):
    model = _native_model(asr_enabled=asr_enabled, multiturn=multiturn)
    model.use_tts_ctc_word_intervals = enabled
    model.tts_ctc_word_interval_config = dict(enabled=enabled, weight=1.0, tolerance_seconds=tolerance)
    return model
