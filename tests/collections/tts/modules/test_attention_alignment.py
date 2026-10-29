# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from nemo.collections.tts.modules.attention_alignment import (
    AttentionAlignmentGuidance,
    AttentionAlignmentTurn,
    apply_attention_alignment,
)
from nemo.collections.tts.modules.nemotron_h_decoder import (
    NemotronHConfig,
    NemotronHFlashAttention2,
    NemotronHForCausalLM,
)


def _turn(queries=(3, 4, 5), keys=(0, 2), centers=(0, 0, 1), batch=0, device='cpu'):
    return AttentionAlignmentTurn(
        batch_index=batch,
        query_positions=torch.tensor(queries, device=device),
        key_positions=torch.tensor(keys, device=device),
        token_indices=torch.tensor(centers, device=device),
    )


def _config(**kwargs):
    settings = dict(
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        hybrid_override_pattern='-*-*',
        attention_dropout=0.0,
        vocab_size=32,
        _attn_implementation='sdpa',
    )
    settings.update(kwargs)
    return NemotronHConfig(**settings)


def test_guidance_loss_uses_actual_full_key_normalization_and_zero_prior_is_identity():
    q = torch.zeros(1, 2, 6, 2, requires_grad=True)
    k = torch.zeros_like(q, requires_grad=True)
    v = torch.randn_like(q)
    ordinary = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    guidance = AttentionAlignmentGuidance([_turn()], heads=(0,), query_chunk_size=1)
    actual, loss = apply_attention_alignment(q, k, v, ordinary, guidance)
    assert actual is ordinary
    assert torch.equal(actual, ordinary)
    assert loss.item() == pytest.approx((math.log(4) + math.log(5) + math.log(6)) / 3, abs=1e-6)
    assert torch.isfinite(loss)
    loss.backward()
    assert q.grad is not None and k.grad is not None


def test_alignment_loss_penalizes_attention_to_acoustic_history_outside_text():
    q = torch.ones(1, 1, 6, 1)
    k = torch.zeros_like(q)
    k[:, :, 5] = 10.0
    v = torch.ones_like(q)
    ordinary = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    _, loss = apply_attention_alignment(
        q, k, v, ordinary, AttentionAlignmentGuidance([_turn(queries=(5,), keys=(0, 1), centers=(0,))])
    )
    # Text-only renormalization would report log(2), hiding the head's failure to read text.
    assert loss > 9.9


def test_guidance_loss_alone_trains_queries_and_keys_without_value_shortcut():
    torch.manual_seed(4)
    q = torch.randn(1, 2, 6, 3, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    ordinary = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    _, loss = apply_attention_alignment(q, k, v, ordinary, AttentionAlignmentGuidance([_turn()]))
    loss.backward()
    assert q.grad.abs().sum() > 0
    assert k.grad.abs().sum() > 0
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
    assert v.grad is None


def test_prior_changes_actual_context_only_for_selected_heads_and_queries():
    q = torch.zeros(1, 2, 6, 1, requires_grad=True)
    k = torch.zeros_like(q, requires_grad=True)
    v = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1).repeat(1, 2, 1, 1).requires_grad_()
    ordinary = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    guidance = AttentionAlignmentGuidance(
        [_turn(queries=(3,), keys=(0, 1), centers=(0,))], prior_strength=1.0, band_width=0.05
    )
    actual, loss = apply_attention_alignment(q, k, v, ordinary, guidance)
    assert actual[0, 0, 3, 0] < 1e-4
    assert ordinary[0, 0, 3, 0].item() == pytest.approx(1.5)
    assert torch.equal(actual[:, 1], ordinary[:, 1])
    assert torch.equal(actual[0, 0, :3], ordinary[0, 0, :3])
    assert torch.equal(actual[0, 0, 4:], ordinary[0, 0, 4:])
    (actual.square().sum() + loss).backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (q, k, v))


def test_future_and_padding_teacher_keys_are_excluded_without_nan():
    q = torch.zeros(1, 2, 6, 1, requires_grad=True)
    k = torch.zeros_like(q, requires_grad=True)
    v = torch.randn_like(q)
    padding = torch.tensor([[1, 1, 1, 1, 0, 0]])
    ordinary = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    guidance = AttentionAlignmentGuidance([_turn(queries=(1, 3, 5), keys=(2, 4), centers=(0, 0, 1))])
    actual, loss = apply_attention_alignment(q, k, v, ordinary, guidance, attention_mask=padding)
    assert actual is ordinary
    assert loss.item() == pytest.approx(math.log(4), abs=1e-6)
    loss.backward()
    assert torch.isfinite(q.grad).all()


def test_all_invalid_teacher_frames_return_differentiable_zero():
    q = torch.randn(1, 2, 6, 2, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q)
    ordinary = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    guidance = AttentionAlignmentGuidance([_turn(queries=(1,), keys=(2, 4), centers=(0,))], prior_strength=1.0)
    actual, loss = apply_attention_alignment(q, k, v, ordinary, guidance)
    assert torch.equal(actual, ordinary)
    assert loss == 0
    loss.backward()
    assert q.grad is not None and torch.equal(q.grad, torch.zeros_like(q))


def test_multiple_turns_restore_context_to_original_batch_and_positions():
    torch.manual_seed(42)
    q = torch.randn(2, 2, 8, 2, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    ordinary = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    guidance = AttentionAlignmentGuidance(
        [
            _turn(queries=(2, 3), keys=(0, 1), centers=(0, 1)),
            _turn(queries=(6, 7), keys=(4, 5), centers=(0, 1)),
            _turn(queries=(4, 5), keys=(1, 3), centers=(0, 1), batch=1),
        ],
        prior_strength=0.7,
        query_chunk_size=1,
    )
    actual, loss = apply_attention_alignment(q, k, v, ordinary, guidance)
    assert torch.equal(actual[:, 1], ordinary[:, 1])
    assert torch.equal(actual[0, 0, 4:6], ordinary[0, 0, 4:6])
    assert torch.equal(actual[1, 0, :4], ordinary[1, 0, :4])
    assert not torch.equal(actual[1, 0, 4:6], ordinary[1, 0, 4:6])
    (loss + actual.square().mean()).backward()
    assert all(torch.isfinite(tensor.grad).all() for tensor in (q, k, v))


@pytest.mark.parametrize('prior_strength', [0.0, 0.7])
@pytest.mark.parametrize('layers', ['middle', (0, 2)])
@pytest.mark.parametrize('window', [(0, 0), (1, 2)])
def test_decoder_guidance_and_gradient_checkpointing_have_matching_gradients(prior_strength, layers, window):
    torch.manual_seed(0)
    ordinary_model = NemotronHForCausalLM(_config(num_hidden_layers=7, hybrid_override_pattern='-*-*-*-')).train()
    checkpointed_model = copy.deepcopy(ordinary_model)
    checkpointed_model.backbone.gradient_checkpointing = True
    inputs = torch.randn(1, 6, 16)
    guidance = AttentionAlignmentGuidance(
        [_turn()],
        prior_strength=prior_strength,
        query_chunk_size=1,
        layers=layers,
        look_back_tokens=window[0],
        look_ahead_tokens=window[1],
    )
    actual = ordinary_model(inputs_embeds=inputs, attention_alignment=guidance, use_cache=False)
    recomputed = checkpointed_model(inputs_embeds=inputs, attention_alignment=guidance, use_cache=False)
    torch.testing.assert_close(actual.logits, recomputed.logits)
    torch.testing.assert_close(actual.attention_alignment_loss, recomputed.attention_alignment_loss)
    (actual.logits.square().mean() + actual.attention_alignment_loss).backward()
    (recomputed.logits.square().mean() + recomputed.attention_alignment_loss).backward()
    for (name, parameter), (other_name, other_parameter) in zip(
        ordinary_model.named_parameters(), checkpointed_model.named_parameters()
    ):
        assert name == other_name
        assert (parameter.grad is None) == (other_parameter.grad is None)
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, other_parameter.grad, atol=1e-6, rtol=1e-4)


def test_zero_prior_preserves_full_decoder_logits_and_adds_no_checkpoint_parameters():
    torch.manual_seed(0)
    model = NemotronHForCausalLM(_config()).eval()
    inputs = torch.randn(1, 6, 16)
    old_keys = set(model.state_dict())
    ordinary = model(inputs_embeds=inputs, use_cache=False)
    guided = model(
        inputs_embeds=inputs,
        attention_alignment=AttentionAlignmentGuidance([_turn()]),
        use_cache=False,
    )
    assert ordinary.attention_alignment_loss is None
    assert guided.attention_alignment_loss is not None
    assert torch.equal(ordinary.logits, guided.logits)
    assert set(model.state_dict()) == old_keys


def test_flash_guidance_keeps_padding_mask_compact_and_uses_flash_baseline():
    import nemo.collections.tts.modules.nemotron_h_decoder as decoder_module

    model = NemotronHForCausalLM(_config()).eval()
    attention_blocks = [layer for layer in model.backbone.layers if layer.block_type == 'attention']
    for block in attention_blocks:
        replacement = NemotronHFlashAttention2(model.config, block.layer_idx)
        replacement.load_state_dict(block.mixer.state_dict())
        block.mixer = replacement
    model.config._attn_implementation = 'flash_attention_2'
    inputs = torch.randn(1, 512, 16)
    padding = torch.ones(1, 512, dtype=torch.bool)
    padding[:, -2:] = False
    observed_masks = []

    def fake_flash(q, k, v, attention_mask, query_length, **kwargs):
        assert attention_mask.ndim == 2
        observed_masks.append(attention_mask.shape)
        causal = torch.arange(query_length)[:, None] >= torch.arange(query_length)[None, :]
        mask = causal[None, None] & attention_mask[:, None, None, :]
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        ).transpose(1, 2)

    with (
        patch.object(decoder_module, 'FLASH_ATTN_AVAILABLE', True),
        patch.object(decoder_module, '_flash_attention_forward', side_effect=fake_flash),
    ):
        output = model(
            inputs_embeds=inputs,
            attention_mask=padding,
            attention_alignment=AttentionAlignmentGuidance(
                [_turn(queries=(400, 401), keys=(0, 3), centers=(0, 1))], prior_strength=0.5
            ),
            use_cache=False,
        )
    assert len(observed_masks) == 2
    assert torch.isfinite(output.attention_alignment_loss)


def test_guidance_rejects_cached_decode_and_rswa_without_altering_disabled_paths():
    model = NemotronHForCausalLM(_config()).eval()
    guidance = AttentionAlignmentGuidance([_turn()])
    with pytest.raises(ValueError, match='uncached'):
        model(inputs_embeds=torch.randn(1, 6, 16), attention_alignment=guidance, use_cache=True)
    sparse = NemotronHForCausalLM(_config(rswa_window=4)).eval()
    with pytest.raises(ValueError, match='R-SWA'):
        sparse(inputs_embeds=torch.randn(1, 6, 16), attention_alignment=guidance, use_cache=False)


@pytest.mark.parametrize(
    'settings',
    [
        {'prior_strength': -0.1},
        {'prior_strength': 1.1},
        {'band_width': 0.0},
        {'heads': ()},
        {'heads': (0, 0)},
        {'query_chunk_size': 0},
        {'layers': 'first'},
        {'layers': ()},
        {'layers': (0, 0)},
        {'layers': (-1,)},
        {'layers': (True,)},
        {'layers': ('1',)},
        {'layers': True},
        {'layers': 0},
        {'layers': {0: 1}},
        {'layers': {0, 1}},
        {'look_back_tokens': -1},
        {'look_ahead_tokens': -1},
        {'look_back_tokens': True},
        {'look_ahead_tokens': False},
        {'look_back_tokens': 1.0},
        {'look_ahead_tokens': '1'},
    ],
)
def test_guidance_configuration_is_validated(settings):
    with pytest.raises(ValueError):
        AttentionAlignmentGuidance([], **settings)


def test_overlapping_queries_fail_instead_of_silently_overwriting_guidance():
    q = torch.randn(1, 2, 6, 2)
    with pytest.raises(ValueError, match='overlap'):
        apply_attention_alignment(q, q, q, q, AttentionAlignmentGuidance([_turn(), _turn()]))


@pytest.mark.parametrize('layers,expected', [('middle', (3,)), ([1], (3,)), ((0, 2), (1, 5)), (None, (1, 3, 5))])
def test_only_selected_attention_layers_receive_the_prior_and_contribute_to_the_loss(layers, expected):
    import nemo.collections.tts.modules.nemotron_h_decoder as decoder_module

    torch.manual_seed(13)
    model = NemotronHForCausalLM(_config(num_hidden_layers=7, hybrid_override_pattern='-*-*-*-')).eval()
    active_layer = {}
    observed_requests = {}
    observed_losses = {}
    handles = []

    def observe_layer(module, args, kwargs):
        active_layer['index'] = module.layer_idx
        observed_requests[module.layer_idx] = kwargs.get('attention_alignment')

    for layer in model.backbone.layers:
        if layer.block_type == 'attention':
            handles.append(layer.mixer.register_forward_pre_hook(observe_layer, with_kwargs=True))

    def observe_guidance(query, key, value, ordinary_output, guidance, **kwargs):
        output, loss = apply_attention_alignment(query, key, value, ordinary_output, guidance, **kwargs)
        observed_losses[active_layer['index']] = loss
        assert not torch.equal(output[:, 0, 3:], ordinary_output[:, 0, 3:])
        assert torch.equal(output[:, 1:], ordinary_output[:, 1:])
        return output, loss

    request = AttentionAlignmentGuidance([_turn()], prior_strength=0.7, heads=(0,), layers=layers)
    try:
        with patch.object(decoder_module, 'apply_attention_alignment', side_effect=observe_guidance):
            output = model(inputs_embeds=torch.randn(1, 6, 16), attention_alignment=request, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    assert tuple(observed_losses) == expected
    assert tuple(index for index, guidance in observed_requests.items() if guidance is not None) == expected
    for index in (1, 3, 5):
        assert observed_requests[index] is (request if index in expected else None)
    torch.testing.assert_close(output.attention_alignment_loss, torch.stack(list(observed_losses.values())).mean())


@pytest.mark.parametrize('count,expected', [(1, (0,)), (2, (1,)), (3, (1,)), (4, (2,))])
def test_middle_selector_counts_attention_layers_only(count, expected):
    assert AttentionAlignmentGuidance([]).resolve_layers(count) == expected


@pytest.mark.parametrize('pattern,layers', [('---', 'middle'), ('*-*', (2,))])
def test_guidance_rejects_missing_or_out_of_range_attention_layers(pattern, layers):
    model = NemotronHForCausalLM(_config(num_hidden_layers=len(pattern), hybrid_override_pattern=pattern)).eval()
    with pytest.raises(ValueError, match='attention layer|attention-layer count'):
        model(
            inputs_embeds=torch.randn(1, 6, 16),
            attention_alignment=AttentionAlignmentGuidance([_turn()], layers=layers),
            use_cache=False,
        )


def test_asymmetric_token_window_has_flat_tolerance_and_gaussian_edges():
    query = torch.zeros(1, 1, 8, 8, requires_grad=True)
    key = torch.zeros_like(query, requires_grad=True)
    value = torch.eye(8).reshape(1, 1, 8, 8).requires_grad_()
    ordinary = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    guidance = AttentionAlignmentGuidance(
        [_turn(queries=(7,), keys=tuple(range(8)), centers=(3,))],
        prior_strength=1.0,
        look_back_tokens=1,
        look_ahead_tokens=2,
    )

    output, loss = apply_attention_alignment(query, key, value, ordinary, guidance)

    # The plateau spans positions 2..5: one before the center and two after it.
    distance = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
    target = torch.exp(-0.5 * distance.square())
    target = target / target.sum()
    torch.testing.assert_close(output[0, 0, 7], target)
    assert torch.equal(output[0, 0, 7, 2:6], output[0, 0, 7, 2].expand(4))
    assert output[0, 0, 7, 0] < output[0, 0, 7, 1] < output[0, 0, 7, 2]
    assert loss.item() == pytest.approx(math.log(8), abs=1e-6)
    (loss + output.square().mean()).backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (query, key, value))


def test_zero_token_window_reproduces_the_original_gaussian_loss_and_prior():
    torch.manual_seed(24)
    query = torch.randn(1, 1, 7, 7, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.eye(7).reshape(1, 1, 7, 7)
    ordinary = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    guidance = AttentionAlignmentGuidance(
        [_turn(queries=(6,), keys=tuple(range(7)), centers=(2,))], prior_strength=1.0
    )

    output, loss = apply_attention_alignment(query, key, value, ordinary, guidance)

    scores = query[0, 0, 6] @ key[0, 0].T / math.sqrt(query.size(-1))
    offsets = torch.arange(7) - 2
    old_target = torch.exp(-0.5 * offsets.float().square())
    old_target = old_target / old_target.sum()
    expected_loss = -(old_target * F.log_softmax(scores, dim=-1)).sum()
    expected_output = F.softmax(scores + old_target.clamp_min(1e-6).log(), dim=-1)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(output[0, 0, 6], expected_output)
    explicit_zero, explicit_zero_loss = apply_attention_alignment(
        query,
        key,
        value,
        ordinary,
        AttentionAlignmentGuidance(guidance.turns, prior_strength=1.0, look_back_tokens=0, look_ahead_tokens=0),
    )
    assert torch.equal(output, explicit_zero)
    assert torch.equal(loss, explicit_zero_loss)


def test_asymmetric_token_window_supervises_the_same_soft_target_used_by_the_prior():
    torch.manual_seed(19)
    query = torch.randn(1, 1, 8, 8, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.eye(8).reshape(1, 1, 8, 8)
    ordinary = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    guidance = AttentionAlignmentGuidance(
        [_turn(queries=(7,), keys=tuple(range(8)), centers=(3,))],
        prior_strength=1.0,
        look_back_tokens=1,
        look_ahead_tokens=2,
    )

    output, loss = apply_attention_alignment(query, key, value, ordinary, guidance)

    scores = query[0, 0, 7] @ key[0, 0].T / math.sqrt(query.size(-1))
    distance = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
    target = torch.exp(-0.5 * distance.square())
    target = target / target.sum()
    torch.testing.assert_close(loss, -(target * F.log_softmax(scores, dim=-1)).sum())
    torch.testing.assert_close(output[0, 0, 7], F.softmax(scores + target.log(), dim=-1))


def test_token_window_respects_causality_padding_and_turn_key_boundaries():
    query = torch.zeros(1, 1, 10, 10, requires_grad=True)
    key = torch.zeros_like(query, requires_grad=True)
    value = torch.eye(10).reshape(1, 1, 10, 10)
    ordinary = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    padding = torch.ones(1, 10, dtype=torch.bool)
    padding[:, 3] = False
    guidance = AttentionAlignmentGuidance(
        [_turn(queries=(4,), keys=(1, 2, 3, 5, 6), centers=(1,))],
        prior_strength=1.0,
        look_back_tokens=100,
        look_ahead_tokens=100,
    )

    output, loss = apply_attention_alignment(query, key, value, ordinary, guidance, attention_mask=padding)

    probabilities = output[0, 0, 4]
    torch.testing.assert_close(probabilities[1:3], torch.full((2,), 0.5), atol=2e-6, rtol=0)
    assert probabilities[3] == 0  # Masked padding stays invisible.
    assert torch.equal(probabilities[5:], torch.zeros(5))  # Look-ahead never reveals future keys.
    assert (probabilities[torch.tensor([0, 4])] < 2e-6).all()  # Other turns receive only the existing soft floor.
    assert loss.item() == pytest.approx(math.log(4), abs=1e-6)
    loss.backward()
    assert torch.isfinite(query.grad).all() and torch.isfinite(key.grad).all()
