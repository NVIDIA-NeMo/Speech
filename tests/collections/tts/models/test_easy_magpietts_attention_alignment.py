# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch

from nemo.collections.tts.models import easy_magpietts_inference
from tests.collections.tts.models.test_easy_magpietts import _make_easy_magpie_model, _toy_batch, tiny_easy_magpie_cfg

pytestmark = pytest.mark.unit


class TinyPhonemeTokenizer:
    """Fixed IDs sufficient to exercise the existing phoneme embedding/loss path."""

    vocab_size = 16
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    pad = 0


@pytest.fixture(scope='module')
def attention_model():
    with torch.device('cuda' if torch.cuda.is_available() else 'cpu'):
        return _make_attention_model(_config())


@pytest.mark.parametrize(
    'step,expected',
    [
        (0, (0.0, 0.0)),
        (9999, (0.0, 0.0)),
        (10000, (0.0, 0.0)),
        (10500, (0.5, 0.5)),
        (11000, (1.0, 1.0)),
        (13500, (1.0, 0.5)),
        (16000, (1.0, 0.0)),
        (20000, (1.0, 0.0)),
    ],
)
def test_teacher_warmup_ramp_and_prior_anneal_are_continuous(attention_model, step, expected):
    with _step(attention_model, step):
        assert attention_model._attention_alignment_schedule() == pytest.approx(expected)


@pytest.mark.parametrize('step', [0, 16000])
def test_no_prior_preserves_pretrained_decoder_predictions(attention_model, step):
    model = attention_model
    with torch.device(_device(model)):
        baseline = _make_attention_model(_config(enabled=False))
    common_state = {name: model.state_dict()[name] for name in baseline.state_dict()}
    baseline.load_state_dict(common_state)
    batch = _attention_batch(model)
    baseline_batch = {name: value for name, value in batch.items() if not name.startswith('attention_alignment_')}

    with torch.no_grad(), _step(model, step):
        original = baseline.process_batch(**baseline_batch)
        guided = model.process_batch(**batch)

    torch.testing.assert_close(guided.logits, original.logits, rtol=0, atol=0)
    torch.testing.assert_close(guided.codebook_loss, original.codebook_loss, rtol=0, atol=0)
    assert original.phoneme_loss is not None and original.phoneme_loss > 0
    torch.testing.assert_close(guided.phoneme_loss, original.phoneme_loss, rtol=0, atol=0)
    torch.testing.assert_close(guided.phoneme_logits, original.phoneme_logits, rtol=0, atol=0)
    assert guided.aligner_loss is not None and torch.isfinite(guided.aligner_loss)
    assert guided.aligned_text_loss is None
    assert not hasattr(model, 'aligned_text_head')
    if step == 16000:
        assert guided.attention_alignment_loss > 0


def test_attention_guidance_works_without_repeated_text_head_or_feedback(attention_model):
    model = attention_model
    with _step(model, 11000):
        output = model.process_batch(**_attention_batch(model))

    assert torch.isfinite(output.loss)
    assert output.aligned_text_loss is None
    assert output.attention_alignment_loss > 0
    assert output.attention_alignment_metrics['valid_turns'] == 2
    assert output.attention_alignment_metrics['causal_frames'] > 0
    expected = output.codebook_loss + model.phoneme_loss_weight * output.phoneme_loss
    expected = expected + model.aligner_loss_weight * output.aligner_loss
    expected = expected + model.attention_alignment_config['loss_weight'] * output.attention_alignment_loss
    torch.testing.assert_close(output.loss, expected)


def test_attention_loss_cannot_train_the_teacher_and_teacher_nll_cannot_train_the_decoder(attention_model):
    model = attention_model
    model.zero_grad(set_to_none=True)
    with _step(model, 11000):
        output = model.process_batch(**_attention_batch(model))

    output.attention_alignment_loss.backward(retain_graph=True)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.decoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.text_aligner.parameters())
    assert all(parameter.grad is None for parameter in model._codec_model.parameters())

    model.zero_grad(set_to_none=True)
    output.aligner_loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.text_aligner.parameters()
    )
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    assert all(parameter.grad is None for parameter in model._codec_model.parameters())
    assert all(
        parameter.grad is None for name, parameter in model.named_parameters() if not name.startswith('text_aligner.')
    )


def test_restored_turn_positions_include_context_and_audio_delay_and_exclude_future_keys(attention_model):
    model = attention_model
    device = _device(model)
    teacher = SimpleNamespace(
        token_indices=torch.tensor([[0, 1, 1, 1, -1, -1, -1, -1, -1, -1, 0, 0, 1, 1]], device=device),
        turn_count=torch.tensor(2, device=device),
        valid_turn_count=torch.tensor(2, device=device),
    )
    with torch.device(device):
        tokens = torch.tensor([[7, 8], [9, 10]])
        lengths = torch.tensor([2, 2])
        spans = torch.tensor([[0, 0, 4], [0, 10, 14]])
        positions = torch.tensor([[0, 3], [10, 12]])
        codes = torch.zeros(1, model.num_audio_codebooks, 14, dtype=torch.long)
        audio_lengths = torch.tensor([14])
        context = torch.tensor([7])
        delay = torch.tensor([8])
    with patch.object(model, '_prepare_aligned_text_targets', return_value=teacher), _step(model, 10500):
        _, guidance, metrics = model._prepare_attention_alignment_targets(
            codes, audio_lengths, codes, tokens, lengths, spans, positions, context, delay
        )

    assert guidance.prior_strength == pytest.approx(0.5)
    first, second = guidance.turns
    assert first.batch_index == second.batch_index == 0
    assert first.key_positions.tolist() == [7, 10]
    assert first.query_positions.tolist() == [8, 10, 11]
    assert first.token_indices.tolist() == [0, 1, 1]
    assert second.key_positions.tolist() == [17, 19]
    assert second.query_positions.tolist() == [18, 19, 20, 21]
    assert second.token_indices.tolist() == [0, 0, 1, 1]
    assert metrics['teacher_frames'] == 8
    assert metrics['causal_frames'] == 7


def test_conditioning_dropout_suppresses_guidance_but_keeps_training_teacher(attention_model):
    model = attention_model
    batch = _attention_batch(model)
    device = _device(model)
    teacher = SimpleNamespace(
        token_indices=torch.tensor([[0, 0, 0, 0], [0, 0, 0, -1]], device=device),
        turn_count=torch.tensor(2, device=device),
        valid_turn_count=torch.tensor(2, device=device),
    )
    with patch.object(model, '_prepare_aligned_text_targets', return_value=teacher) as prepare, _step(model, 11000):
        result, request, _ = model._prepare_attention_alignment_targets(
            batch['audio_codes'],
            batch['audio_codes_lens'],
            batch['audio_codes'],
            batch['attention_alignment_text_tokens'],
            batch['attention_alignment_text_lens'],
            batch['attention_alignment_turn_indices'],
            batch['attention_alignment_key_positions'],
            torch.zeros(2, dtype=torch.long, device=device),
            torch.zeros(2, dtype=torch.long, device=device),
            dropout_conditional_input=True,
        )

    assert result is teacher
    assert request is None
    prepare.assert_called_once()


def test_text_only_batches_keep_all_teacher_parameters_in_ddp(attention_model):
    model = attention_model
    model.zero_grad(set_to_none=True)
    model._aligned_text_zero_loss(torch.zeros((), device=_device(model))).backward()
    assert all(
        parameter.grad is not None and parameter.grad.eq(0).all() for parameter in model.text_aligner.parameters()
    )


def test_attention_guidance_rejects_repeated_head_mode():
    with pytest.raises(ValueError, match='use_aligned_text_head|separate experimental|mutually exclusive'):
        _make_attention_model(_config(extra={'use_aligned_text_head': True}))


@pytest.mark.parametrize(
    'overrides,message',
    [
        ({'attention_alignment': {'heads': [4]}}, 'outside the decoder head count'),
        ({'nemotron_h_config': {'hybrid_override_pattern': '-'}}, 'requires at least one attention layer'),
    ],
)
def test_invalid_attention_guidance_decoder_configuration_fails_before_training(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_attention_model(_config(extra=overrides))


def test_missing_exact_conditioning_metadata_fails_instead_of_using_normalized_targets(attention_model):
    batch = _attention_batch(attention_model)
    del batch['attention_alignment_key_positions']
    with pytest.raises(ValueError, match='exact conditioning BPE'):
        attention_model.process_batch(**batch)


def test_stale_teacher_tokens_cannot_silently_supervise_different_conditioning_keys(attention_model):
    batch = _attention_batch(attention_model)
    batch['attention_alignment_text_tokens'][0, 0] += 1
    with pytest.raises(ValueError, match='actual conditioning BPE IDs and positions'):
        attention_model.process_batch(**batch)


def _config(enabled=True, extra=None):
    values = {
        'use_attention_alignment': enabled,
        'use_aligned_text_head': False,
        'aligned_text_feedback': False,
        'phoneme_tokenizer': {
            '_target_': ('tests.collections.tts.models.test_easy_magpietts_attention_alignment.TinyPhonemeTokenizer')
        },
        'phoneme_stacking_factor': 1,
        'attention_alignment': {
            'loss_weight': 0.1,
            'start_step': 10000,
            'ramp_steps': 1000,
            'prior_anneal_steps': 5000,
            'band_width': 1.0,
            'heads': [0],
            'query_chunk_size': 2,
        },
        'aligned_text_aligner': {
            'flow_type': 'sttts',
            'text_encoder_type': 'relative_transformer',
            'text_dropout': 0.0,
            'hidden_channels': 8,
            'num_flows': 2,
            'num_text_layers': 1,
            'dequantization_noise': 0.0,
        },
    }
    values.update(extra or {})
    return tiny_easy_magpie_cfg(values)


def _attention_batch(model):
    with torch.device(_device(model)):
        batch = _toy_batch(model)
        content = []
        positions = []
        for row, length in enumerate(batch['text_lens'].tolist()):
            ids = batch['text'][row, :length]
            mask = ids.ge(0) & ids.lt(model.bos_id) & ids.ne(model.pad_id)
            positions.append(mask.nonzero(as_tuple=True)[0])
            content.append(ids[mask])
        width = max(map(len, content))
        tokens = torch.full((len(content), width), model.pad_id, dtype=torch.long)
        keys = torch.full_like(tokens, -1)
        for row, (ids, key_positions) in enumerate(zip(content, positions)):
            tokens[row, : ids.numel()] = ids
            keys[row, : ids.numel()] = key_positions
        batch['attention_alignment_text_tokens'] = tokens
        batch['attention_alignment_text_lens'] = torch.tensor(list(map(len, content)))
        batch['attention_alignment_key_positions'] = keys
        batch['attention_alignment_turn_indices'] = torch.tensor([[0, 0, 4], [1, 0, 3]])
        batch['phoneme_tokens'] = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]])
        batch['phoneme_tokens_lens'] = torch.tensor([4, 3])
        batch['mode'] = 'val'
    return batch


def _step(model, value):
    return patch.object(type(model), 'global_step', new_callable=PropertyMock, return_value=value)


def _make_attention_model(config):
    instantiate = easy_magpietts_inference.safe_instantiate
    tokenizer_target = 'tests.collections.tts.models.test_easy_magpietts_attention_alignment.TinyPhonemeTokenizer'

    def instantiate_test_tokenizer_only(value, *args, **kwargs):
        if hasattr(value, 'get') and value.get('_target_') == tokenizer_target:
            return TinyPhonemeTokenizer()
        return instantiate(value, *args, **kwargs)

    # The production safe-instantiation allowlist stays intact. Only this fixed
    # in-memory test tokenizer bypasses Hydra; all other targets use the real API.
    with patch.object(easy_magpietts_inference, 'safe_instantiate', side_effect=instantiate_test_tokenizer_only):
        return _make_easy_magpie_model(config)


def _device(model):
    # Lightning's cached device can remain CPU when a module is constructed
    # under torch.device('cuda') without a subsequent explicit .to() call.
    return next(model.parameters()).device
