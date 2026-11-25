# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.models.nemotron_tts.v2.base import TrainingMode
from nemo.collections.tts.models.nemotron_tts.v2.model import NemotronTTSModel
from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import CTCTimestampHead
from tests.collections.tts.models.test_nemotron_ctc_timestamp_head import _batch, _ctc_model
from tests.collections.tts.models.test_nemotron_timestamp_head import _packed_text
from tests.collections.tts.modules.nemotron_tts.v2.test_streaming_timestamps_model import _init_stream
from tests.collections.tts.modules.nemotron_tts.v2.test_training_model import _inputs

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('native_enabled', [False, True])
@pytest.mark.parametrize('asr_enabled', [False, True])
def test_native_and_asr_ctc_heads_enable_independently_and_share_the_backbone(native_enabled, asr_enabled):
    model = _native_model(native_enabled=native_enabled, asr_enabled=asr_enabled)
    original = _inputs(model)
    batch = _batch() if asr_enabled else dict(raw_texts=['selected TTS input'])
    output = model.process_batch(**original, timestamp_batch=batch, mode='val')
    assert (output.tts_ctc_loss is not None) is native_enabled
    assert (output.ctc_timestamp_loss is not None) is asr_enabled
    assert len(model._timestamp_teacher.calls) == int(asr_enabled)
    expected = (
        model.parallel_codebook_loss_scale * output.codebook_loss
        + model.acoustic_codes_predictor_loss_scale * output.acoustic_codes_predictor_loss
    )
    if native_enabled:
        expected += model.tts_ctc_loss_weight * output.tts_ctc_loss
        assert output.tts_ctc_metrics['supervised_turns'] == 1
    if asr_enabled:
        expected += model.ctc_timestamp_head_loss_weight * output.ctc_timestamp_loss
    torch.testing.assert_close(output.loss, expected)


def test_native_ctc_uses_actual_conditioning_ids_without_retokenizing_the_reference():
    model = _native_model()
    inputs = _inputs(model)
    inputs['text'] = torch.tensor([[model.bos_id, 11, 12, model.pad_id, model.eos_id]])
    inputs['text_lens'] = torch.tensor([5])
    batch = dict(raw_texts=['text whose standalone tokenization could differ'], normalized_texts=['unused text'])
    with (
        patch.object(model.tokenizer, 'encode', side_effect=AssertionError('Reference text was retokenized')),
        patch.object(model, '_get_timestamp_teacher', side_effect=AssertionError('Native CTC loaded ASR')),
        patch.object(model.tts_ctc_head, 'compute_loss', wraps=model.tts_ctc_head.compute_loss) as native,
    ):
        output = model.process_batch(**inputs, timestamp_batch=batch, mode='val')
    assert native.call_count == 1
    assert native.call_args.args[1].tolist() == [11, 12]
    assert len(native.call_args.args) < 3 or native.call_args.args[2] is None
    assert native.call_args.kwargs.get('teacher') is None
    assert native.call_args.kwargs.get('distillation_mask') is None
    assert output.tts_ctc_metrics['supervised_turns'] == 1


@pytest.mark.parametrize('multiturn', [False, True])
def test_native_ctc_skips_the_whole_turn_when_conditioning_contains_phoneme_ids(multiturn):
    model = _native_model(multiturn=multiturn)
    model.enable_phoneme_text_input = True
    model.text_phoneme_token_offset = 64
    model.text_phoneme_vocab_size = 4
    if multiturn:
        turns = []
        text = _packed_text(model, timestamp_turns=turns, fractional=True)
        text[text == 8] = model.text_phoneme_token_offset
        turns[0]['input_token_ids'] = [7, model.text_phoneme_token_offset]
        batch = dict(timestamp_turns=[turns])
    else:
        text = torch.tensor([[model.bos_id, 7, model.text_phoneme_token_offset, 8, model.eos_id]])
        batch = dict(raw_texts=['partly phoneme-conditioned input'])
    with patch.object(model.tokenizer, 'encode', side_effect=AssertionError('Incomplete input was reconstructed')):
        targets = model._prepare_tts_ctc_targets(
            batch,
            text,
            torch.tensor([text.size(1)]),
            torch.tensor([16]),
            16,
        )
    assert [target['token_ids'].tolist() for target in targets] == ([[9, 10]] if multiturn else [])
    hidden = torch.randn(1, 16, 16, requires_grad=True)
    with patch.object(model.tts_ctc_head, 'compute_loss', wraps=model.tts_ctc_head.compute_loss) as native:
        loss, metrics = model._compute_tts_ctc_loss(
            hidden,
            targets,
            torch.zeros(1, 2, 16, dtype=torch.long),
            torch.tensor([16]),
        )
    assert native.call_count == metrics['supervised_turns'] == int(multiturn)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.tts_ctc_head.parameters())
    if multiturn:
        assert hidden.grad[0, 2:7].eq(0).all() and hidden.grad[0, 9:14].abs().sum() > 0


@pytest.mark.parametrize('asr_enabled', [False, True])
def test_native_ctc_gradients_reach_its_head_and_backbone_and_keep_the_teacher_frozen(asr_enabled):
    model = _native_model(asr_enabled=asr_enabled)
    batch = _batch() if asr_enabled else dict(raw_texts=['selected TTS input'])
    output = model.process_batch(**_inputs(model), timestamp_batch=batch, mode='val')
    output.tts_ctc_loss.backward()
    assert model.tts_ctc_head.projection[-1].weight.grad.abs().sum() > 0
    assert model.decoder.layers[2].mixer.q_proj.weight.grad.abs().sum() > 0
    if asr_enabled:
        assert model.ctc_timestamp_head.projection[-1].weight.grad is None
    assert all(parameter.grad is None for parameter in model._timestamp_teacher.parameters())
    assert not any('timestamp_teacher' in name for name in model.state_dict())
    assert any(name.startswith('tts_ctc_head.') for name in model.state_dict())


@pytest.mark.parametrize('text_input_mode,delay', [('streaming', 1), ('full', 5)])
def test_both_ctc_heads_read_the_same_speech_delayed_audio_slice(text_input_mode, delay):
    model = _native_model(asr_enabled=True)
    inputs = _inputs(model)
    inputs.update(context_text_tokens=torch.empty(1, 0, dtype=torch.long), context_text_tokens_lens=torch.tensor([0]))
    hidden = []
    hook = model.decoder.register_forward_hook(lambda _module, _args, result: hidden.append(result.last_hidden_state))
    try:
        with (
            patch.object(model.tts_ctc_head, 'compute_loss', wraps=model.tts_ctc_head.compute_loss) as native,
            patch.object(model.ctc_timestamp_head, 'compute_loss', wraps=model.ctc_timestamp_head.compute_loss) as asr,
        ):
            model.process_batch(
                **inputs, timestamp_batch=_batch(), mode='val', training_mode=TrainingMode(text_input_mode, 0, 1, 0)
            )
    finally:
        hook.remove()
    torch.testing.assert_close(native.call_args.args[0], hidden[0][0, delay : delay + 8], atol=0, rtol=0)
    torch.testing.assert_close(native.call_args.args[0], asr.call_args.args[0], atol=0, rtol=0)
    assert native.call_args.kwargs.get('input_length', 16) == 16


def test_multiturn_native_ctc_preserves_actual_input_ids_and_independent_turn_slices_without_waveforms():
    model = _native_model(multiturn=True)
    turns = []
    text = _packed_text(model, timestamp_turns=turns, fractional=True)
    assert [turn['input_token_ids'] for turn in turns] == [[7, 8], [9, 10]]
    batch = dict(timestamp_turns=[turns], languages=['pt'])
    with patch.object(model.tokenizer, 'encode', side_effect=AssertionError('Packed reference was retokenized')):
        targets = model._prepare_tts_ctc_targets(batch, text, torch.tensor([15]), torch.tensor([15]), 16)
    hidden = torch.randn(1, 16, 16, requires_grad=True)
    with patch.object(model.tts_ctc_head, 'compute_loss', wraps=model.tts_ctc_head.compute_loss) as native:
        loss, metrics = model._compute_tts_ctc_loss(
            hidden, targets, torch.zeros(1, 2, 16, dtype=torch.long), torch.tensor([16])
        )
    assert native.call_count == 2 and metrics['supervised_turns'] == 2
    assert [call.args[1].tolist() for call in native.call_args_list] == [[7, 8], [9, 10]]
    torch.testing.assert_close(native.call_args_list[0].args[0], hidden[0, 2:7])
    torch.testing.assert_close(native.call_args_list[1].args[0], hidden[0, 9:14])
    loss.backward()
    assert hidden.grad[0, 2:7].abs().sum() > 0 and hidden.grad[0, 9:14].abs().sum() > 0
    assert hidden.grad[0, :2].eq(0).all() and hidden.grad[0, 7:9].eq(0).all()
    assert hidden.grad[0, 14:].eq(0).all()


@pytest.mark.parametrize('reason', ['eos', 'user', 'padding'])
def test_native_ctc_does_not_train_a_complete_transcript_on_partial_or_nonagent_audio(reason):
    model = _native_model()
    inputs = _inputs(model)
    targets = model._prepare_tts_ctc_targets(
        dict(raw_texts=['selected input']),
        inputs['text'],
        inputs['text_lens'],
        inputs['audio_codes_lens'],
        9,
    )
    hidden = torch.randn(1, 9, 16, requires_grad=True)
    codes = torch.zeros(1, 2, 9, dtype=torch.long)
    lengths = torch.tensor([9])
    mask = None
    if reason == 'eos':
        codes[:, :, 3] = model.audio_eos_id
    elif reason == 'user':
        mask = torch.tensor([[True, True, True, False, True, True, True, True, True]])
    else:
        lengths[0] = 5
    with patch.object(
        model.tts_ctc_head, 'compute_loss', side_effect=AssertionError('An incomplete transcript was used')
    ):
        loss, metrics = model._compute_tts_ctc_loss(hidden, targets, codes, lengths, mask)
    assert metrics['supervised_turns'] == 0 and metrics['skipped_turns'] == 1 and loss == 0
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.tts_ctc_head.parameters())


@pytest.mark.parametrize('policy', [False, True])
def test_native_disabled_or_policy_scoring_preserves_the_existing_path(policy):
    model = _native_model(native_enabled=policy)
    with (
        patch.object(
            model, '_prepare_tts_ctc_targets', side_effect=AssertionError('Disabled native CTC prepared data')
        ),
        patch.object(model, '_get_timestamp_teacher', side_effect=AssertionError('Disabled native CTC loaded ASR')),
    ):
        output = model.process_batch(**_inputs(model), mode='val', return_policy_logits=policy)
    assert output.tts_ctc_loss is None and output.ctc_timestamp_loss is None


@pytest.mark.parametrize('asr_enabled', [False, True])
def test_native_zero_weight_preserves_the_existing_audio_predictions_and_loss(asr_enabled):
    baseline = _native_model(native_enabled=False, asr_enabled=asr_enabled)
    observed = _native_model(asr_enabled=asr_enabled)
    observed.tts_ctc_loss_weight = 0
    batch = _batch() if asr_enabled else dict(raw_texts=['selected input'])
    expected = baseline.process_batch(**_inputs(baseline), timestamp_batch=batch, mode='val')
    actual = observed.process_batch(**_inputs(observed), timestamp_batch=batch, mode='val')
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
    torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)


@pytest.mark.parametrize('use_cfg', [False, True])
def test_both_streaming_heads_are_passive_and_drain_independently(use_cfg):
    baseline = _native_model(native_enabled=False, asr_enabled=True).eval()
    observed = _native_model(asr_enabled=True).eval()
    states = [_init_stream(model, use_cfg=use_cfg) for model in (baseline, observed)]
    native_rows, asr_rows = [], []
    seen_hidden = []
    hooks = [
        head.register_forward_pre_hook(lambda _module, args: seen_hidden.append(args[0]))
        for head in (observed.tts_ctc_head, observed.ctc_timestamp_head)
    ]
    try:
        with patch.object(
            observed, '_get_timestamp_teacher', side_effect=AssertionError('Streaming loaded ASR teacher')
        ):
            for token in [observed.bos_id, 7, 8, observed.eos_id, observed.pad_id, observed.pad_id, None, None]:
                results = [
                    model.streaming_step(state, None if token is None else torch.full((2,), token))
                    for model, state in zip((baseline, observed), states)
                ]
                states = [result[0] for result in results]
                torch.testing.assert_close(states[0].last_hidden, states[1].last_hidden, atol=0, rtol=0)
                if results[0][1] is not None:
                    torch.testing.assert_close(results[0][1], results[1][1], atol=0, rtol=0)
                tts = observed.drain_ctc_timestamp_frames(states[1], source='tts')
                assert observed.drain_ctc_timestamp_frames(states[1], source='tts') == []
                asr = observed.drain_ctc_timestamp_frames(states[1])
                assert len(tts) == len(asr)
                native_rows.extend(tts)
                asr_rows.extend(asr)
            final = observed.streaming_finalize(states[1])
    finally:
        for hook in hooks:
            hook.remove()
    assert native_rows and asr_rows
    assert all(hidden.size(0) == 2 for hidden in seen_hidden)
    assert all(row['is_blank'] == (row['token_id'] == 60) for row in native_rows)
    assert all(row['is_blank'] == (row['token_id'] == 5) for row in asr_rows)
    assert final.tts_ctc_tokens.shape == final.tts_ctc_probabilities.shape
    assert final.tts_ctc_lens.tolist() == final.ctc_timestamp_lens.tolist()
    for sample in range(2):
        rows = [row for row in native_rows if row['batch_index'] == sample]
        assert [row['frame_index'] for row in rows] == list(range(int(final.tts_ctc_lens[sample])))
        assert [row['token_id'] for row in rows] == final.tts_ctc_tokens[sample].tolist()


def test_native_validation_cer_uses_total_character_errors_instead_of_mean_batch_rates():
    model = _native_model(asr_enabled=True)
    with patch('torch.distributed.is_initialized', return_value=False):
        model.on_validation_epoch_start()
    with patch.object(model, 'log_dict') as log:
        for errors, characters in [(1, 2), (1, 18)]:
            model._log_tts_ctc_losses(
                SimpleNamespace(
                    tts_ctc_loss=torch.tensor(2.0),
                    tts_ctc_metrics=dict(
                        ctc_loss=torch.tensor(2.0),
                        distillation_loss=torch.tensor(0.0),
                        cer_errors=errors,
                        cer_reference_characters=characters,
                        cer_turns=1,
                    ),
                ),
                'val',
                1,
            )
    assert model._tts_ctc_validation_totals.tolist() == [2, 20, 2, 0, 0, 0]
    assert model._ctc_timestamp_validation_totals.eq(0).all()
    assert all('val/tts_ctc_distillation_loss' not in call.args[0] for call in log.call_args_list)

    class NativeMetricsLogged(Exception):
        pass

    with (
        patch('torch.distributed.is_initialized', return_value=False),
        patch.object(model, 'log_dict', side_effect=NativeMetricsLogged) as epoch_log,
        pytest.raises(NativeMetricsLogged),
    ):
        model.on_validation_epoch_end()
    values = epoch_log.call_args.args[0]
    assert float(values['val/tts_ctc_cer']) == pytest.approx(0.1)
    assert int(values['val/tts_ctc_evaluated_turns']) == 2
    assert epoch_log.call_args.kwargs['sync_dist'] is False


@pytest.mark.parametrize('multiturn', [False, True])
def test_native_generated_cer_uses_selected_input_and_excludes_ambiguous_multiturn_references(multiturn):
    model = _native_model(asr_enabled=True)
    with patch('torch.distributed.is_initialized', return_value=False):
        model.on_validation_epoch_start()
    batch = dict(raw_texts=['7 8', '9 10'], normalized_texts=['unused normalized reference'] * 2)
    if multiturn:
        batch['timestamp_turns'] = [[dict(text='7 8')], [dict(text='ambiguous first'), dict(text='ambiguous second')]]
    generated = SimpleNamespace(
        tts_ctc_tokens=torch.tensor([[7, 7, 60, 8, 8, 59], [9, 9, 60, 10, 10, 59]]),
        tts_ctc_lens=torch.tensor([5, 5]),
    )
    with patch.object(model, '_get_timestamp_teacher', side_effect=AssertionError('Generated native CER loaded ASR')):
        model._update_generated_ctc_timestamp_cer(batch, generated, source='tts')
    characters = model._ctc_timestamp_cer_counts('7 8', '7 8')[1]
    if not multiturn:
        characters += model._ctc_timestamp_cer_counts('9 10', '9 10')[1]
    assert model._tts_ctc_validation_totals.tolist() == [0, 0, 0, 0, characters, 1 if multiturn else 2]
    assert model._ctc_timestamp_validation_totals.eq(0).all()


@pytest.mark.parametrize('multiturn', [False, True])
def test_native_flag_is_available_before_modelpt_builds_the_dataloader(multiturn):
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
        observed['enabled'] = model.use_tts_ctc_head
        observed['head_constructed'] = hasattr(model, 'tts_ctc_head')
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
    assert dataset.call_args.kwargs['load_cached_codes_if_available']
    if multiturn:
        assert dataset.call_args.kwargs['return_timestamp_supervision']


class _NativeTokenizer:
    pad = 0
    tokens = list(range(60))

    def encode(self, text, tokenizer_name=None):
        return {'first turn': [7, 8], 'second turn': [9, 10]}[text]

    def ids_to_text(self, ids):
        return ' '.join(str(token) for token in ids)

    def ids_to_tokens(self, ids):
        return [str(token) for token in ids]


def _native_model(native_enabled=True, asr_enabled=False, multiturn=False):
    model = _ctc_model(ctc_enabled=asr_enabled, multiturn=multiturn)
    model.tokenizer = _NativeTokenizer()
    model.use_tts_ctc_head = native_enabled
    model.tts_ctc_head_config = dict(projection_size=8, loss_weight=0.2, chunk_size=2, frame_seconds=0.04)
    model.tts_ctc_loss_weight = 0.2
    model.tts_ctc_tokenizer = _NativeTokenizer()
    if native_enabled:
        model.tts_ctc_head = CTCTimestampHead(16, 61, 60, projection_size=8, chunk_size=2, distillation_weight=0)
    return model
