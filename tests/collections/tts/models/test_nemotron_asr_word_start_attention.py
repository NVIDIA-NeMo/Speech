# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
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
from tests.collections.tts.models.test_nemotron_tts_ctc_word_intervals import MAPPING as WORD_MAPPING
from tests.collections.tts.models.test_nemotron_tts_ctc_word_intervals import _mapping
from tests.collections.tts.modules.nemotron_tts.v2.test_streaming_timestamps_model import _init_stream
from tests.collections.tts.modules.nemotron_tts.v2.test_training_model import _inputs

pytestmark = pytest.mark.unit

MAPPING = 'nemo.collections.tts.models.nemotron_tts.v2.model.map_asr_word_start_anchors'


@pytest.mark.parametrize('layers,resolved', [('middle', 'middle'), ([1], (1,))])
def test_word_start_config_resolves_omegaconf_lists_and_selects_main_backbone(layers, resolved):
    model = _anchor_model()
    cfg = OmegaConf.create(
        dict(
            anchor_weight=0.2,
            asr_word_start_attention=dict(
                loss_weight='${anchor_weight}', layers=layers, heads=[0], query_chunk_size=64
            ),
        )
    )
    with patch('nemo.collections.tts.models.nemotron_tts.v2.model.logging.info') as log:
        model._init_asr_word_start_attention(cfg)
    assert model.asr_word_start_attention_config == dict(
        loss_weight=0.2,
        layers=resolved,
        heads=(0,),
        query_chunk_size=64,
    )
    assert 'main-backbone attention blocks (2,), heads (0,)' in log.call_args.args[0]


@pytest.mark.parametrize(
    'setting,value,message',
    [
        ('layers', [3], 'layer index exceeds'),
        ('layers', [-1], 'nonnegative attention-layer indices'),
        ('heads', [2], 'heads exceeds'),
        ('heads', [-1], 'nonnegative head indices'),
        ('loss_weight', -0.1, 'finite and nonnegative'),
        ('loss_weight', float('nan'), 'finite and nonnegative'),
        ('loss_weight', True, 'finite and nonnegative'),
    ],
)
def test_word_start_config_rejects_invalid_selection_and_weight(setting, value, message):
    model = _anchor_model()
    cfg = OmegaConf.create(dict(asr_word_start_attention={setting: value}))
    with pytest.raises(ValueError, match=message):
        model._init_asr_word_start_attention(cfg)


def test_disabled_word_start_config_does_not_validate_backbone_selection():
    model = _anchor_model(enabled=False)
    cfg = OmegaConf.create(dict(asr_word_start_attention=dict(layers=[999], heads=[999])))
    with patch(
        'nemo.collections.tts.models.nemotron_tts.v2.model.AttentionAlignmentGuidance',
        side_effect=AssertionError('Disabled feature validated the backbone selection'),
    ):
        model._init_asr_word_start_attention(cfg)
    assert model.asr_word_start_attention_config == dict(layers=[999], heads=[999])


@pytest.mark.parametrize('asr_enabled,native_enabled', [(False, False), (True, False), (False, True), (True, True)])
def test_word_start_attention_is_independent_of_ctc_heads_and_shares_the_teacher(asr_enabled, native_enabled):
    model = _anchor_model(asr_enabled=asr_enabled, native_enabled=native_enabled)
    with (
        patch(MAPPING, side_effect=_anchors) as mapping,
        patch.object(model.tokenizer, 'encode', side_effect=AssertionError('Conditioning IDs were retokenized')),
        patch.object(model, 'forward', wraps=model.forward) as forward,
        patch(
            'nemo.collections.tts.models.nemotron_tts.v2.model.prepare_attention_alignment_training',
            side_effect=AssertionError('Word starts loaded the flow aligner'),
        ),
    ):
        output = model.process_batch(**_inputs(model), timestamp_batch=_batch(), mode='val')
    assert len(model._timestamp_teacher.calls) == mapping.call_count == 1
    options = model._timestamp_teacher.calls[0]['options']
    assert options.get('return_ctc', False) is asr_enabled
    assert options.get('align_words', True)
    guidance = forward.call_args.kwargs['attention_alignment']
    assert guidance.layers == 'middle' and guidance.heads == (0,)
    assert guidance.prior_strength == 0.0
    assert guidance.turns[0].key_positions.tolist() == [3, 4]
    assert guidance.turns[0].query_positions.tolist() == [7, 11]
    assert guidance.turns[0].token_indices.tolist() == [0, 1]
    assert guidance.turns[0].token_end_indices.tolist() == [1, 2]
    assert output.asr_word_start_attention_metrics['anchors'] == 2
    assert output.asr_word_start_attention_loss.requires_grad
    assert (output.ctc_timestamp_loss is not None) is asr_enabled
    assert (output.tts_ctc_loss is not None) is native_enabled
    expected = (
        model.parallel_codebook_loss_scale * output.codebook_loss
        + model.acoustic_codes_predictor_loss_scale * output.acoustic_codes_predictor_loss
        + model.asr_word_start_attention_config['loss_weight'] * output.asr_word_start_attention_loss
    )
    if asr_enabled:
        expected += model.ctc_timestamp_head_loss_weight * output.ctc_timestamp_loss
    if native_enabled:
        expected += model.tts_ctc_loss_weight * output.tts_ctc_loss
    torch.testing.assert_close(output.loss, expected)
    assert not hasattr(model, 'text_aligner')
    assert hasattr(model, 'ctc_timestamp_head') is asr_enabled
    assert hasattr(model, 'tts_ctc_head') is native_enabled


def test_word_start_and_word_interval_losses_reuse_one_frozen_teacher_forward():
    model = _anchor_model(asr_enabled=True, native_enabled=True)
    model.use_tts_ctc_word_intervals = True
    model.tts_ctc_word_interval_config = dict(enabled=True, weight=1.0, tolerance_seconds=0.08)
    with patch(MAPPING, side_effect=_anchors), patch(WORD_MAPPING, side_effect=_mapping):
        output = model.process_batch(**_inputs(model), timestamp_batch=_batch(), mode='val')
    assert len(model._timestamp_teacher.calls) == 1
    assert output.asr_word_start_attention_loss > 0
    assert output.tts_ctc_metrics['word_ctc_valid_intervals'] == 2
    assert output.ctc_timestamp_metrics['supervised_turns'] == 1


def test_word_start_loss_trains_only_selected_middle_query_head_and_upstream_backbone():
    model = _anchor_model(asr_enabled=True, native_enabled=True)
    with patch(MAPPING, side_effect=_anchors):
        output = model.process_batch(**_inputs(model), timestamp_batch=_batch(), mode='val')
    output.asr_word_start_attention_loss.backward()
    middle = model.decoder.layers[2].mixer
    assert middle.q_proj.weight.grad[:8].abs().sum() > 0
    assert middle.q_proj.weight.grad[8:].eq(0).all()
    assert middle.k_proj.weight.grad.abs().sum() > 0
    assert middle.v_proj.weight.grad is None or middle.v_proj.weight.grad.eq(0).all()
    later = model.decoder.layers[4].mixer.q_proj.weight.grad
    assert later is None or later.eq(0).all()
    assert all(parameter.grad is None for parameter in model.acoustic_codes_predictor.parameters())
    assert model.ctc_timestamp_head.projection[-1].weight.grad is None
    assert model.tts_ctc_head.projection[-1].weight.grad is None
    assert all(parameter.grad is None for parameter in model._timestamp_teacher.parameters())


@pytest.mark.parametrize('text_mode,delay', [('streaming', 1), ('full', 16)])
def test_packed_word_anchors_restore_fractional_turn_and_speech_delay_once(text_mode, delay):
    model = _anchor_model(multiturn=True)
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
    with (
        patch(MAPPING, side_effect=_anchors) as mapping,
        patch.object(model, 'forward', wraps=model.forward) as forward,
    ):
        output = model.process_batch(
            **inputs, timestamp_batch=batch, mode='val', training_mode=TrainingMode(text_mode, 0, 1, 0)
        )
    assert mapping.call_count == 2 and len(model._timestamp_teacher.calls) == 1
    guidance = forward.call_args.kwargs['attention_alignment']
    assert [turn.query_positions.tolist() for turn in guidance.turns] == [
        [2 + delay, 4 + delay],
        [9 + delay, 12 + delay],
    ]
    assert [turn.key_positions.tolist() for turn in guidance.turns] == [[3, 4], [10, 11]]
    assert [call.args[2].tolist() for call in mapping.call_args_list] == [[7, 8], [9, 10]]
    assert output.asr_word_start_attention_metrics['anchors'] == 4


@pytest.mark.parametrize('reason', ['causal', 'eos', 'user', 'incomplete', 'overlap', 'mapping', 'teacher', 'dropout'])
def test_word_anchor_filters_preserve_causality_and_reject_invalid_supervision(reason):
    model = _anchor_model()
    inputs = _inputs(model)
    targets = model._prepare_tts_ctc_targets(
        _batch(),
        inputs['text'],
        inputs['text_lens'],
        inputs['audio_codes_lens'],
        9,
        include_key_positions=True,
    )
    records = [dict(batch_index=0, turn_index=0, words=[dict(word='reference')], error=None)]
    codes = torch.zeros(1, 2, 8, dtype=torch.long)
    agent_mask = None
    if reason == 'causal':
        targets[0]['key_positions'] = torch.tensor([12, 13])
    elif reason == 'eos':
        codes[:, :, 4] = model.audio_eos_id
    elif reason == 'user':
        agent_mask = torch.tensor([[True, True, True, True, False, True, True, True]])
    elif reason == 'incomplete':
        targets[0]['complete_turn'] = False
    elif reason == 'overlap':
        targets.append(dict(targets[0], turn_index=1))
        records.append(dict(records[0], turn_index=1))
    elif reason == 'teacher':
        records[0]['error'] = 'alignment failed'
    mapping_result = dict(anchors=[], error='input_token_mismatch', merged_word_groups=0, collision_count=0)
    with patch(MAPPING, side_effect=_anchors if reason != 'mapping' else None, return_value=mapping_result):
        guidance, metrics = model._prepare_asr_word_start_attention(
            targets,
            records,
            torch.tensor([0]),
            torch.tensor([5]),
            codes,
            torch.tensor([8]),
            agent_mask=agent_mask,
            dropout_conditioning=reason == 'dropout',
        )
    expected = 1 if reason in ('eos', 'user') else 0
    assert metrics['anchors'] == expected
    if expected:
        assert guidance.turns[0].query_positions.tolist() == [5]
    else:
        assert guidance is None
    if reason == 'overlap':
        assert metrics['overlapping_anchors'] == 4
    if reason == 'causal':
        assert metrics['skipped_causal_anchors'] == 2


def test_zero_anchor_weight_preserves_audio_outputs_and_disabled_mode_loads_no_teacher():
    baseline = _anchor_model(enabled=False)
    observed = _anchor_model(weight=0.0)
    with (
        patch.object(baseline, '_get_timestamp_teacher', side_effect=AssertionError('Disabled feature loaded ASR')),
        patch.object(
            baseline,
            '_prepare_asr_word_start_attention',
            side_effect=AssertionError('Disabled feature mapped anchors'),
        ),
    ):
        expected = baseline.process_batch(**_inputs(baseline), mode='val')
    with patch(MAPPING, side_effect=_anchors):
        actual = observed.process_batch(**_inputs(observed), timestamp_batch=_batch(), mode='val')
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
    torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
    assert expected.asr_word_start_attention_loss is None
    assert set(baseline.state_dict()) == set(observed.state_dict())


def test_streaming_and_policy_scoring_never_load_or_apply_word_start_teacher():
    model = _anchor_model().eval()
    state = _init_stream(model, use_cfg=False)
    with (
        patch.object(model, '_get_timestamp_teacher', side_effect=AssertionError('Inference loaded ASR')),
        patch.object(
            model, '_prepare_asr_word_start_attention', side_effect=AssertionError('Inference mapped anchors')
        ),
    ):
        for token in [model.bos_id, 7, 8, model.eos_id, model.pad_id, model.pad_id, None, None]:
            state, *_ = model.streaming_step(state, None if token is None else torch.full((2,), token))
        output = model.streaming_finalize(state)
        policy = model.process_batch(**_inputs(model), mode='val', return_policy_logits=True)
    assert output.audio_len.numel() == 2
    assert policy.asr_word_start_attention_loss is None


@pytest.mark.parametrize('multiturn', [False, True])
def test_anchor_waveform_and_turn_requirements_are_available_before_modelpt_loader_setup(multiturn):
    class LoaderConstructed(Exception):
        pass

    cfg = _constructor_config(multiturn)
    codec = SimpleNamespace(
        sample_rate=100, output_sample_rate=100, num_codebooks=2, codebook_size=16, samples_per_frame=8
    )
    observed = {}
    original = NemotronTTSModel.get_lhotse_dataloader

    def loader(model, *args, **kwargs):
        observed.update(enabled=model.use_asr_word_start_attention, head_constructed=hasattr(model, 'tts_ctc_head'))
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


def _constructor_config(multiturn):
    return OmegaConf.create(
        dict(
            codecmodel_path='unused.nemo',
            text_tokenizers={'bpe': {}},
            use_multiturn_dataset=multiturn,
            load_cached_codes_if_available=True,
            context_duration_min=3.0,
            context_duration_max=3.0,
            use_asr_word_start_attention=True,
            train_ds=dict(
                use_lhotse=True, volume_norm=True, dataset={}, dataloader_params=dict(batch_size=2, num_workers=2)
            ),
        )
    )


def _anchors(tokenizer, text, ids, words, duration, frame_seconds, audio_start_seconds=0.0):
    assert ids.tolist() in ([7, 8], [9, 10])
    return dict(
        error=None,
        collision_count=0,
        merged_word_groups=0,
        token_coverage=1.0,
        anchors=[
            dict(frame_index=math.floor(audio_start_seconds / frame_seconds + 0.5 + 1e-8), token_start=0, token_end=1),
            dict(
                frame_index=math.floor((audio_start_seconds + duration / 2) / frame_seconds + 0.5 + 1e-8),
                token_start=1,
                token_end=2,
            ),
        ],
    )


def _anchor_model(*, enabled=True, asr_enabled=False, native_enabled=False, multiturn=False, weight=0.1):
    model = _native_model(native_enabled=native_enabled, asr_enabled=asr_enabled, multiturn=multiturn)
    model.use_asr_word_start_attention = enabled
    model.asr_word_start_attention_config = dict(loss_weight=weight, layers='middle', heads=(0,), query_chunk_size=2)
    return model
