# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from nemo.collections.tts.modules.attention_alignment import (
    AttentionAlignmentCapture,
    AttentionAlignmentGuidance,
    AttentionAlignmentTurn,
    apply_attention_alignment,
)
from nemo.collections.tts.modules.nemotron_h_decoder import NemotronHFlashAttention2, NemotronHForCausalLM
from tests.collections.tts.modules.test_attention_alignment import _config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('prior_strength', [0.0, 1.0])
def test_capture_is_prior_free_detached_and_preserves_full_key_normalization(prior_strength):
    query = torch.zeros(1, 2, 8, 2, requires_grad=True)
    key = torch.zeros_like(query, requires_grad=True)
    value = torch.arange(8, dtype=torch.float32)[None, None, :, None].expand_as(query)
    collector = AttentionAlignmentCapture()
    guidance = AttentionAlignmentGuidance([_turn()], prior_strength=prior_strength, capture=collector)
    with torch.no_grad():
        ordinary = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        output, loss = apply_attention_alignment(query, key, value, ordinary, guidance)
        plain_output, plain_loss = apply_attention_alignment(
            query, key, value, ordinary, replace(guidance, capture=None)
        )

    torch.testing.assert_close(output, plain_output, rtol=0, atol=0)
    torch.testing.assert_close(loss, plain_loss, rtol=0, atol=0)
    record = collector.records[0]
    expected = torch.tensor([[0.25, 0.25, 0.0], [1 / 6, 1 / 6, 1 / 6], [0.125, 0.125, 0.125]])
    torch.testing.assert_close(record['probabilities'], expected)
    assert not record['probabilities'].requires_grad
    assert record['probabilities'].device.type == 'cpu'
    assert record['prior_free'] is True
    assert record['probabilities'].sum(dim=-1).max() < 1
    assert record['turn_index'] == 0 and record['batch_index'] == 0
    assert record['turn_query_start'] == 3 and record['turn_query_end'] == 8
    assert record['cropped_audio'] is False and record['cropped_text'] is False


def test_capture_caps_turns_frames_and_text_with_honest_leading_excerpt_metadata():
    query = torch.zeros(2, 2, 8, 2)
    collector = AttentionAlignmentCapture(max_turns=1, max_audio_frames=1, max_text_tokens=2)
    guidance = AttentionAlignmentGuidance([_turn(), replace(_turn(), batch_index=1)], capture=collector)
    with torch.no_grad():
        apply_attention_alignment(query, query, query, query, guidance)

    assert len(collector.records) == 1
    record = collector.records[0]
    assert record['probabilities'].shape == (1, 2)
    assert record['query_positions'].tolist() == [3]
    assert record['key_positions'].tolist() == [0, 2]
    assert record['token_indices'].tolist() == [0]
    assert record['total_query_rows'] == record['eligible_query_rows'] == 3
    assert record['total_text_tokens'] == 3
    assert record['cropped_audio'] is True and record['cropped_text'] is True


@pytest.mark.parametrize('mask_kind', ['padding', 'boolean', 'additive'])
def test_capture_never_assigns_probability_to_future_or_masked_keys(mask_kind):
    query = torch.zeros(1, 2, 8, 2)
    padding = torch.ones(1, 8, dtype=torch.bool)
    padding[:, 2] = False
    mask = padding
    if mask_kind != 'padding':
        mask = padding[:, None, None, :].expand(-1, 1, 8, -1)
        if mask_kind == 'additive':
            mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf)
    collector = AttentionAlignmentCapture()
    with torch.no_grad():
        apply_attention_alignment(
            query, query, query, query, AttentionAlignmentGuidance([_turn()], capture=collector), attention_mask=mask
        )

    record = collector.records[0]
    assert record['query_positions'].tolist() == [3, 7]
    torch.testing.assert_close(record['probabilities'], torch.tensor([[1 / 3, 0.0, 0.0], [1 / 7, 0.0, 1 / 7]]))


@pytest.mark.parametrize('layers,block,ordinal', [('middle', 2, 1), (None, 0, 0), ((2,), 4, 2)])
@pytest.mark.parametrize('capture_only', [False, True])
def test_decoder_captures_first_selected_layer_and_head_without_changing_predictions(
    layers, block, ordinal, capture_only
):
    torch.manual_seed(2)
    model = NemotronHForCausalLM(_config(num_hidden_layers=5, hybrid_override_pattern='*-*-*')).eval()
    inputs = torch.randn(1, 8, 16)
    collector = AttentionAlignmentCapture()
    request = AttentionAlignmentGuidance(
        [_turn()], layers=layers, heads=(2, 1), capture=collector, capture_only=capture_only
    )
    with torch.no_grad():
        baseline = model(inputs_embeds=inputs, use_cache=False)
        captured = model(inputs_embeds=inputs, use_cache=False, attention_alignment=request)

    torch.testing.assert_close(captured.logits, baseline.logits, rtol=0, atol=0)
    assert len(collector.records) == 1
    record = collector.records[0]
    assert record['decoder_block_index'] == block
    assert record['attention_layer_ordinal'] == ordinal
    assert record['head'] == 2
    if capture_only:
        assert captured.attention_alignment_loss == 0
    else:
        assert captured.attention_alignment_loss > 0


def test_capture_only_preserves_logits_and_zero_loss_on_the_flash_layout_path():
    import nemo.collections.tts.modules.nemotron_h_decoder as decoder_module

    model = NemotronHForCausalLM(_config()).eval()
    for layer in model.backbone.layers:
        if layer.block_type == 'attention':
            mixer = NemotronHFlashAttention2(model.config, layer.layer_idx)
            mixer.load_state_dict(layer.mixer.state_dict())
            layer.mixer = mixer
    model.eval()
    assert all(not layer.mixer.training for layer in model.backbone.layers if layer.block_type == 'attention')
    model.config._attn_implementation = 'flash_attention_2'
    inputs = torch.randn(1, 8, 16)
    collector = AttentionAlignmentCapture()
    padding = torch.ones(1, 8, dtype=torch.bool)
    padding[:, 6] = False

    def flash(query, key, value, attention_mask, query_length, **kwargs):
        causal = torch.arange(query_length)[:, None] >= torch.arange(query_length)[None, :]
        mask = causal[None, None] & attention_mask[:, None, None, :]
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=mask
        ).transpose(1, 2)

    with (
        torch.no_grad(),
        patch.object(decoder_module, 'FLASH_ATTN_AVAILABLE', True),
        patch.object(decoder_module, '_flash_attention_forward', side_effect=flash),
    ):
        baseline = model(inputs_embeds=inputs, use_cache=False, attention_mask=padding)
        captured = model(
            inputs_embeds=inputs,
            use_cache=False,
            attention_mask=padding,
            attention_alignment=AttentionAlignmentGuidance([_turn()], capture=collector, capture_only=True),
        )
    torch.testing.assert_close(captured.logits, baseline.logits, rtol=0, atol=0)
    assert captured.attention_alignment_loss == 0
    assert collector.records[0]['probabilities'].shape == (3, 3)


@pytest.mark.parametrize('model_training,grad_enabled', [(True, False), (False, True)])
def test_decoder_rejects_capture_during_training_or_gradient_recording(model_training, grad_enabled):
    model = NemotronHForCausalLM(_config()).train(model_training)
    collector = AttentionAlignmentCapture()
    with torch.set_grad_enabled(grad_enabled), pytest.raises(ValueError, match='eval mode and no_grad'):
        model(
            inputs_embeds=torch.randn(1, 8, 16),
            use_cache=False,
            attention_alignment=AttentionAlignmentGuidance([_turn()], capture=collector),
        )
    assert collector.records == []


@pytest.mark.parametrize(
    'settings',
    [
        {'capture_only': True},
        {'capture': AttentionAlignmentCapture(), 'capture_only': True, 'prior_strength': 0.1},
        {'capture': []},
        {'capture_only': 1},
    ],
)
def test_invalid_capture_requests_fail_clearly(settings):
    with pytest.raises(ValueError, match='capture'):
        AttentionAlignmentGuidance([], **settings)


@pytest.mark.parametrize('field,value', [('max_turns', 0), ('max_audio_frames', True), ('max_text_tokens', -1)])
def test_capture_limits_require_positive_integers(field, value):
    with pytest.raises(ValueError, match=field):
        AttentionAlignmentCapture(**{field: value})


def test_adjacent_turn_plots_restore_variable_context_delays_and_do_not_mix_batch_rows():
    from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel

    arguments, records = _visualization_inputs()
    # Capture order need not match transcript order, and equal absolute cache
    # positions in different batch rows must never identify the same turn.
    arguments['capture'] = AttentionAlignmentCapture(max_turns=3, records=list(reversed(records)))
    examples = EasyMagpieTTSModel._prepare_alignment_visualizations(
        SimpleNamespace(frame_stacking_factor=1), **arguments, limit=3
    )

    assert len(examples) == 3
    for index, (example, record) in enumerate(zip(examples, records)):
        assert example['turn_index'] == index
        assert example['batch_index'] == (0 if index < 2 else 1)
        assert example['sample_start'] == (0, 4, 2)[index]
        assert example['attention_frame_indices'].tolist() == [0, 1, 2, 3]
        assert example['attention_token_positions'].tolist() == list(range((2, 3, 2)[index]))
        torch.testing.assert_close(example['attention_probs'], record['probabilities'], rtol=0, atol=0)
        torch.testing.assert_close(example['mas_token_indices'], record['token_indices'], rtol=0, atol=0)
        torch.testing.assert_close(example['token_ids'], arguments['tokens'][index, : (2, 3, 2)[index]])


def test_paired_attention_plot_has_priority_over_an_earlier_mas_only_turn():
    from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel

    arguments, records = _visualization_inputs()
    # The first MAS turn can lack visible conditioning keys while a later turn
    # has a genuine captured head. A small image budget should show that pair.
    arguments['capture'] = AttentionAlignmentCapture(records=[records[1]])
    examples = EasyMagpieTTSModel._prepare_alignment_visualizations(
        SimpleNamespace(frame_stacking_factor=1), **arguments, limit=1
    )

    assert len(examples) == 1
    assert examples[0]['turn_index'] == 1
    torch.testing.assert_close(examples[0]['attention_probs'], records[1]['probabilities'], rtol=0, atol=0)


def _visualization_inputs():
    spans = torch.tensor([[0, 0, 4], [0, 4, 8], [1, 2, 6]])
    path = torch.full((2, 8), -1, dtype=torch.long)
    paths = ([0, 0, 1, 1], [0, 1, 1, 2], [0, 0, 1, 1])
    # Effective audio delays include different context lengths and speech delays.
    delays = torch.tensor([5, 9])
    records = []
    for index, ((sample, start, end), centers) in enumerate(zip(spans.tolist(), paths)):
        path[sample, start:end] = torch.tensor(centers)
        count = (2, 3, 2)[index]
        records.append(
            {
                'batch_index': sample,
                'query_positions': torch.arange(start, end) + delays[sample],
                'key_positions': torch.arange(count),
                'token_indices': torch.tensor(centers),
                'probabilities': torch.full((4, count), 0.05 * (index + 1)),
                'attention_layer_ordinal': 1,
                'decoder_block_index': 2,
                'head': 0,
                'cropped_audio': False,
                'cropped_text': False,
            }
        )
    arguments = {
        'alignment': SimpleNamespace(token_indices=path),
        'tokens': torch.tensor([[11, 12, 0], [21, 22, 23], [31, 32, 0]]),
        'token_lens': torch.tensor([2, 3, 2]),
        'turn_indices': spans,
        'audio_lens': torch.tensor([8, 6]),
        'audio_delay': delays,
    }
    return arguments, records


def _turn():
    return AttentionAlignmentTurn(
        batch_index=0,
        query_positions=torch.tensor([3, 5, 7]),
        key_positions=torch.tensor([0, 2, 4]),
        token_indices=torch.tensor([0, 1, 2]),
    )
