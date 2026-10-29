# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from nemo.collections.tts.modules.attention_alignment import AttentionAlignmentGuidance
from nemo.collections.tts.modules.attention_alignment_inference import (
    AttentionAlignmentInferenceState,
    apply_attention_alignment_inference,
)
from nemo.collections.tts.modules.nemotron_h_decoder import (
    HybridMambaAttentionDynamicCache,
    NemotronHConfig,
    NemotronHFlashAttention2,
    NemotronHForCausalLM,
)

pytestmark = pytest.mark.unit


def test_inference_prior_scales_current_turn_text_only_preserving_other_heads_and_cfg_rows():
    query = torch.zeros(2, 2, 1, 8)
    key = torch.zeros(2, 2, 8, 8)
    value = torch.eye(8).reshape(1, 1, 8, 8).repeat(2, 2, 1, 1)
    ordinary = F.scaled_dot_product_attention(query, key, value)
    state = AttentionAlignmentInferenceState(1, look_back_tokens=0, look_ahead_tokens=1, prior_epsilon=0.1)
    for position in (1, 3, 4, 6):
        state.add_text_token(0, position)

    output = apply_attention_alignment_inference(query, key, value, ordinary, state.request([0]))

    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 0.1, 1.0])
    torch.testing.assert_close(output[0, 0, 0], weights / weights.sum())
    assert torch.equal(output[0, 1], ordinary[0, 1])
    assert torch.equal(output[1], ordinary[1])
    assert state.anchors == [0]
    assert state.last_attended_tokens == [0]


def test_progress_is_local_and_nondecreasing_while_actual_attention_can_look_back_or_overcome_finite_prior():
    state = AttentionAlignmentInferenceState(1, look_back_tokens=1, look_ahead_tokens=1, prior_epsilon=0.1)
    for position in range(5):
        state.add_text_token(0, position)
    query = torch.ones(1, 1, 1, 1)
    value = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1)
    for scores, expected_anchor, expected_peak in (
        ([0.0, 5.0, 10.0, 100.0, 0.0], 1, 3),
        ([10.0, 0.0, 0.0, 0.0, 0.0], 1, 0),
        ([0.0, 0.0, 20.0, 0.0, 0.0], 2, 2),
    ):
        key = torch.tensor(scores).reshape(1, 1, 5, 1)
        ordinary = F.scaled_dot_product_attention(query, key, value)
        output = apply_attention_alignment_inference(query, key, value, ordinary, state.request())
        assert torch.isfinite(output).all()
        assert state.anchors == [expected_anchor]
        assert state.last_attended_tokens == [expected_peak]


@pytest.mark.parametrize('mask_format', ['padding', 'boolean', 'additive'])
def test_inference_window_respects_absolute_cache_causality_padding_and_turn_reset(mask_format):
    query = torch.zeros(1, 1, 1, 7)
    key = torch.zeros(1, 1, 7, 7)
    value = torch.eye(7).reshape(1, 1, 7, 7)
    padding = torch.ones(1, 7, dtype=torch.bool)
    padding[:, 3] = False
    if mask_format == 'padding':
        attention_mask = padding
    elif mask_format == 'boolean':
        attention_mask = padding[:, None, None]
    else:
        attention_mask = torch.zeros(1, 1, 1, 7).masked_fill(~padding[:, None, None], -torch.inf)
    causal = (torch.arange(7) <= 4) & padding[0]
    ordinary = F.scaled_dot_product_attention(query, key, value, attn_mask=causal.reshape(1, 1, 1, 7))
    state = AttentionAlignmentInferenceState(1, look_back_tokens=0, look_ahead_tokens=0, prior_epsilon=0.1)
    for position in (0, 2, 3, 5, 6):
        state.add_text_token(0, position)

    output = apply_attention_alignment_inference(
        query, key, value, ordinary, state.request(), cache_position=torch.tensor([4]), attention_mask=attention_mask
    )

    expected = torch.tensor([1.0, 1.0, 0.1, 0.0, 1.0, 0.0, 0.0])
    torch.testing.assert_close(output[0, 0, 0], expected / expected.sum())
    assert torch.equal(output[0, 0, 0, 5:], torch.zeros(2))
    assert output[0, 0, 0, 3] == 0
    state.anchors[0] = 2
    state.reset_turn(0)
    assert state.text_key_positions == [[]]
    assert state.anchors == [0]
    assert state.last_attended_tokens == [-1]
    state.add_text_token(0, 4)
    reset_output = apply_attention_alignment_inference(
        query, key, value, ordinary, state.request(), cache_position=torch.tensor([4]), attention_mask=attention_mask
    )
    # Previous-turn text at keys 0 and 2 is ordinary history after resetting metadata.
    assert torch.equal(reset_output, ordinary)


def test_neutral_prior_empty_metadata_and_inactive_rows_preserve_original_output():
    torch.manual_seed(4)
    query = torch.randn(2, 3, 1, 2)
    key = torch.randn(2, 3, 5, 2)
    value = torch.randn_like(key)
    ordinary = F.scaled_dot_product_attention(query, key, value)
    state = AttentionAlignmentInferenceState(2, prior_epsilon=1.0, look_ahead_tokens=0)
    assert torch.equal(apply_attention_alignment_inference(query, key, value, ordinary, state.request()), ordinary)
    for position in range(5):
        state.add_text_token(0, position)
        state.add_text_token(1, position)
    assert torch.equal(apply_attention_alignment_inference(query, key, value, ordinary, state.request([])), ordinary)
    assert state.last_attended_tokens == [-1, -1]
    assert torch.equal(apply_attention_alignment_inference(query, key, value, ordinary, state.request([0])), ordinary)
    assert state.last_attended_tokens[0] >= 0
    assert state.last_attended_tokens[1] == -1


def test_real_cached_decoder_routes_to_middle_layer_only_without_an_alignment_teacher():
    import nemo.collections.tts.modules.nemotron_h_decoder as decoder_module

    torch.manual_seed(18)
    model = NemotronHForCausalLM(_config()).eval()
    state = AttentionAlignmentInferenceState(1, look_ahead_tokens=0)
    for position in (0, 1, 2):
        state.add_text_token(0, position)
    inputs = torch.randn(2, 1, 16)
    requests = {}
    handles = []

    def observe(module, args, kwargs):
        requests[module.layer_idx] = kwargs.get('attention_alignment_inference')

    for layer in model.backbone.layers:
        if layer.block_type == 'attention':
            handles.append(layer.mixer.register_forward_pre_hook(observe, with_kwargs=True))
    try:
        with torch.no_grad():
            prefill = model(inputs_embeds=torch.randn(2, 4, 16), use_cache=True, cache_position=torch.arange(4))
            baseline_cache = copy.deepcopy(prefill.past_key_values)
            baseline = model(
                inputs_embeds=inputs,
                past_key_values=baseline_cache,
                cache_position=torch.tensor([4]),
                use_cache=True,
            )
            requests.clear()
            request = state.request([0])
            with patch.object(
                decoder_module, 'apply_attention_alignment_inference', wraps=apply_attention_alignment_inference
            ) as apply:
                guided = model(
                    inputs_embeds=inputs,
                    past_key_values=prefill.past_key_values,
                    cache_position=torch.tensor([4]),
                    use_cache=True,
                    attention_alignment_inference=request,
                )
    finally:
        for handle in handles:
            handle.remove()
    assert apply.call_count == 1
    assert requests == {1: None, 3: request, 5: None}
    assert guided.attention_alignment_loss is None
    assert guided.past_key_values.get_seq_length(3) == 5
    assert torch.equal(guided.logits[1], baseline.logits[1])
    assert torch.isfinite(guided.logits).all()
    assert state.last_attended_tokens[0] >= 0


def test_flash_cached_path_reuses_expanded_grouped_keys_for_the_selected_head():
    import nemo.collections.tts.modules.nemotron_h_decoder as decoder_module

    config = _config()
    cache = HybridMambaAttentionDynamicCache(config, 1, torch.float32, device='cpu')
    attention = NemotronHFlashAttention2(config, layer_idx=3).eval()
    state = AttentionAlignmentInferenceState(1, look_ahead_tokens=0)
    for position in (0, 1, 2):
        state.add_text_token(0, position)
    observed_lengths = []

    def fake_flash(query, key, value, attention_mask, query_length, **kwargs):
        observed_lengths.append((query_length, key.size(1)))
        positions = torch.arange(key.size(1) - query_length, key.size(1))
        visible = torch.arange(key.size(1))[None, :] <= positions[:, None]
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=visible[None, None]
        ).transpose(1, 2)

    with (
        patch.object(decoder_module, 'FLASH_ATTN_AVAILABLE', True),
        patch.object(decoder_module, '_flash_attention_forward', side_effect=fake_flash),
        patch.object(
            decoder_module, 'apply_attention_alignment_inference', wraps=apply_attention_alignment_inference
        ) as apply,
        torch.no_grad(),
    ):
        attention(torch.randn(1, 4, 16), past_key_value=cache, cache_position=torch.arange(4))
        output, loss, returned_cache = attention(
            torch.randn(1, 1, 16),
            past_key_value=cache,
            cache_position=torch.tensor([4]),
            attention_alignment_inference=state.request(),
        )
    assert observed_lengths == [(4, 4), (1, 5)]
    assert apply.call_count == 1
    assert output.shape == (1, 1, 16) and torch.isfinite(output).all()
    assert returned_cache is cache
    assert loss is None and state.last_attended_tokens[0] >= 0


def test_decoder_resolves_missing_cache_position_and_rejects_noncontiguous_positions_before_cache_mutation():
    model = NemotronHForCausalLM(_config()).eval()
    state = AttentionAlignmentInferenceState(1)
    state.add_text_token(0, 0)
    with torch.no_grad():
        prefill = model(inputs_embeds=torch.randn(1, 3, 16), use_cache=True, cache_position=torch.arange(3))
        cache = prefill.past_key_values
        for position in (torch.tensor([2]), torch.tensor([4]), torch.tensor([3.0]), torch.tensor([[3]])):
            with pytest.raises(ValueError, match='absolute cache timeline'):
                model(
                    inputs_embeds=torch.randn(1, 1, 16),
                    use_cache=True,
                    past_key_values=cache,
                    cache_position=position,
                    attention_alignment_inference=state.request(),
                )
            assert all(cache.get_seq_length(index) == 3 for index in (1, 3, 5))
        output = model(
            inputs_embeds=torch.randn(1, 1, 16),
            use_cache=True,
            past_key_values=cache,
            attention_alignment_inference=state.request(),
        )
    assert all(cache.get_seq_length(index) == 4 for index in (1, 3, 5))
    assert torch.isfinite(output.logits).all()


@pytest.mark.parametrize(
    'settings',
    [
        {'batch_size': 0},
        {'batch_size': True},
        {'look_back_tokens': -1},
        {'look_ahead_tokens': True},
        {'look_ahead_tokens': 1.0},
        {'prior_epsilon': 0.0},
        {'prior_epsilon': -0.1},
        {'prior_epsilon': 1.1},
        {'prior_epsilon': float('nan')},
        {'prior_epsilon': True},
        {'prior_epsilon': '0.1'},
        {'layer': 'last'},
        {'layer': -1},
        {'layer': True},
        {'head': -1},
        {'head': True},
    ],
)
def test_inference_settings_are_validated(settings):
    with pytest.raises(ValueError):
        AttentionAlignmentInferenceState(**{'batch_size': 1, **settings})


def test_registered_keys_requests_and_resolved_layers_are_validated():
    state = AttentionAlignmentInferenceState(2)
    assert state.resolve_layer(3) == 1
    assert state.resolve_layer(4) == 2
    state.add_text_token(0, 2)
    for bad_index in (-1, 2, True):
        with pytest.raises(ValueError):
            state.add_text_token(bad_index, 3)
        with pytest.raises(ValueError):
            state.reset_turn(bad_index)
    for bad_position in (-1, 1, 2, True):
        with pytest.raises(ValueError):
            state.add_text_token(0, bad_position)
    for bad_indices in ((0, 0), (2,), {0}):
        with pytest.raises(ValueError):
            state.request(bad_indices)
    with pytest.raises(ValueError):
        state.resolve_layer(0)
    with pytest.raises(ValueError):
        AttentionAlignmentInferenceState(1, layer=3).resolve_layer(3)


@pytest.mark.parametrize('invalid', ['teacher', 'training', 'multi_query', 'uncached', 'rswa', 'head', 'layer'])
def test_decoder_rejects_incompatible_inference_requests(invalid):
    model = NemotronHForCausalLM(_config(rswa_window=4 if invalid == 'rswa' else None)).eval()
    state = AttentionAlignmentInferenceState(
        1, head=4 if invalid == 'head' else 0, layer=3 if invalid == 'layer' else 1
    )
    settings = dict(
        inputs_embeds=torch.randn(1, 2 if invalid == 'multi_query' else 1, 16),
        attention_alignment_inference=state.request(),
        use_cache=invalid != 'uncached',
    )
    if invalid == 'teacher':
        settings['attention_alignment'] = AttentionAlignmentGuidance([])
    if invalid == 'training':
        model.train()
    with pytest.raises(ValueError):
        model(**settings)


def _config(**kwargs):
    settings = dict(
        hidden_size=16,
        num_hidden_layers=7,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        hybrid_override_pattern='-*-*-*-',
        attention_dropout=0.0,
        vocab_size=32,
        _attn_implementation='sdpa',
    )
    settings.update(kwargs)
    return NemotronHConfig(**settings)
