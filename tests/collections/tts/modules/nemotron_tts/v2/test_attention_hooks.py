# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.tts.modules import nemotron_h_decoder
from nemo.collections.tts.modules.attention_alignment import AttentionAlignmentGuidance, AttentionAlignmentTurn
from nemo.collections.tts.modules.nemotron_h_decoder import (
    HybridMambaAttentionDynamicCache,
    NemotronHConfig,
    NemotronHFlashAttention2,
    NemotronHModel,
)
from nemo.collections.tts.modules.nemotron_tts.v2.acoustic_codes_predictor import AcousticCodesPredictorBlock

pytestmark = pytest.mark.unit


def test_middle_is_main_backbone_middle_with_extra_predictor_attention_layers():
    composite = nn.Module()
    composite.backbone = NemotronHModel(_config()).eval()
    composite.predictor = AcousticCodesPredictorBlock(_config(), 0, 2, 17, n_layers=2).eval()
    observations = []
    with ExitStack() as stack, torch.no_grad():
        main = [
            stack.enter_context(patch.object(layer, 'forward', wraps=layer.forward))
            for layer in composite.backbone.layers
        ]
        extra = [
            stack.enter_context(patch.object(layer, 'forward', wraps=layer.forward))
            for layer in composite.predictor.layers
        ]
        output = composite.backbone(
            inputs_embeds=torch.randn(2, 7, 16),
            use_cache=False,
            attention_observer=lambda *args, **kwargs: observations.append(args[0].shape),
        )
        composite.predictor(output.last_hidden_state)
    assert len(observations) == 1
    assert [
        index for index, call in enumerate(main) if call.call_args.kwargs.get('attention_observer') is not None
    ] == [2]
    assert all(call.call_args.kwargs.get('attention_observer') is None for call in extra)
    assert all(call.call_args.kwargs.get('attention_alignment') is None for call in extra)


@pytest.mark.parametrize('checkpointed', [False, True])
def test_selected_attention_loss_preserves_hidden_at_zero_prior_and_backpropagates(checkpointed):
    model = NemotronHModel(_config()).train()
    model.gradient_checkpointing = checkpointed
    inputs = torch.randn(2, 7, 16, requires_grad=True)
    ordinary = model(inputs_embeds=inputs, use_cache=False)
    guided = model(inputs_embeds=inputs, use_cache=False, attention_alignment=_guidance(inputs.device))
    torch.testing.assert_close(ordinary.last_hidden_state, guided.last_hidden_state, rtol=0, atol=0)
    assert ordinary.attention_alignment_loss is None
    loss = guided.attention_alignment_loss
    assert loss.requires_grad and torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    middle = model.layers[2].mixer
    assert middle.q_proj.weight.grad.abs().sum() > 0
    assert middle.k_proj.weight.grad.abs().sum() > 0
    # Only head zero is directly supervised; the later attention layer does not
    # participate when backpropagating only the middle-head alignment loss.
    assert model.layers[4].mixer.q_proj.weight.grad is None


@pytest.mark.parametrize('implementation', ['sdpa', 'flash_attention_2'])
def test_cached_attention_observer_is_read_only_and_includes_current_key(implementation):
    torch.manual_seed(19)
    baseline = NemotronHModel(_config(implementation)).eval()
    observed = NemotronHModel(_config(implementation)).eval()
    observed.load_state_dict(baseline.state_dict())
    cache = [None, None]
    observations = []
    inputs = [torch.randn(2, 3, 16), torch.randn(2, 1, 16), torch.randn(2, 1, 16)]
    with patch.object(nemotron_h_decoder, 'FLASH_ATTN_AVAILABLE', False), torch.no_grad():
        for index, features in enumerate(inputs):
            start = 0 if index == 0 else index + 2
            position = torch.arange(start, start + features.size(1))
            ordinary = baseline(inputs_embeds=features, use_cache=True, cache_params=cache[0], cache_position=position)
            captured = observed(
                inputs_embeds=features,
                use_cache=True,
                cache_params=cache[1],
                cache_position=position,
                attention_observer=lambda q, k, **kwargs: observations.append((q.shape, k.shape, kwargs)),
            )
            cache = [ordinary.past_key_values, captured.past_key_values]
            torch.testing.assert_close(ordinary.last_hidden_state, captured.last_hidden_state, rtol=0, atol=0)
            assert observations[-1][1][-2] == start + features.size(1)
            assert observations[-1][0][1] == observations[-1][1][1] == 2
    assert len(observations) == 3


def test_training_alignment_is_never_applied_to_cached_inference():
    model = NemotronHModel(_config()).eval()
    with pytest.raises(ValueError, match='cached inference'):
        model(inputs_embeds=torch.randn(1, 7, 16), use_cache=True, attention_alignment=_guidance('cpu'))
    model.train()
    with pytest.raises(ValueError, match='evaluation mode'):
        model(inputs_embeds=torch.randn(1, 7, 16), use_cache=False, attention_observer=lambda *args, **kwargs: None)


def test_flash_wrapper_observes_expanded_cached_keys_and_preserves_outputs():
    config = _config('flash_attention_2')
    observations = []
    with (
        patch.object(nemotron_h_decoder, 'FLASH_ATTN_AVAILABLE', True),
        patch.object(nemotron_h_decoder, 'is_flash_attn_greater_or_equal_2_10', return_value=True),
        patch.object(nemotron_h_decoder, '_flash_attention_forward', side_effect=_reference_flash_forward) as kernel,
        torch.no_grad(),
    ):
        baseline = NemotronHFlashAttention2(config, layer_idx=0).eval()
        observed = NemotronHFlashAttention2(config, layer_idx=0).eval()
        observed.load_state_dict(baseline.state_dict())
        caches = [HybridMambaAttentionDynamicCache(config, batch_size=2) for _ in range(2)]
        for start, length in [(0, 3), (3, 1)]:
            features = torch.randn(2, length, 16)
            padding_mask = torch.ones(2, start + length, dtype=torch.long)
            padding_mask[1, 0] = 0
            position = torch.arange(start, start + length)
            ordinary = baseline(
                features, attention_mask=padding_mask, past_key_value=caches[0], cache_position=position
            )
            captured = observed(
                features,
                attention_mask=padding_mask,
                past_key_value=caches[1],
                cache_position=position,
                attention_observer=lambda q, k, **kwargs: observations.append((q.shape, k.shape, kwargs)),
            )
            torch.testing.assert_close(ordinary[0], captured[0], rtol=0, atol=0)
            query_shape, key_shape, options = observations[-1]
            assert query_shape == (2, 2, length, 8)
            assert key_shape == (2, 2, start + length, 8)
            assert options['attention_mask'] is padding_mask
            assert options['cache_position'] is position
            assert all(cache.get_seq_length() == start + length for cache in caches)
    assert len(observations) == 2 and kernel.call_count == 4


@pytest.mark.parametrize('prior_strength', [0.0, 0.6])
def test_flash_wrapper_training_alignment_preserves_zero_prior_and_backpropagates(prior_strength):
    with (
        patch.object(nemotron_h_decoder, 'FLASH_ATTN_AVAILABLE', True),
        patch.object(nemotron_h_decoder, 'is_flash_attn_greater_or_equal_2_10', return_value=True),
        patch.object(nemotron_h_decoder, '_flash_attention_forward', side_effect=_reference_flash_forward) as kernel,
    ):
        attention = NemotronHFlashAttention2(_config('flash_attention_2'), layer_idx=0).train()
        features = torch.randn(2, 7, 16, requires_grad=True)
        ordinary = attention(features)
        guidance = _guidance(features.device)
        guidance.prior_strength = prior_strength
        guided = attention(features, attention_alignment=guidance)
        if prior_strength == 0:
            torch.testing.assert_close(ordinary[0], guided[0], rtol=0, atol=0)
        else:
            assert not torch.equal(ordinary[0], guided[0])
        loss = guided[1]
        assert torch.isfinite(loss) and loss.item() > 0
        loss.backward()
        assert attention.q_proj.weight.grad.abs().sum() > 0
        assert attention.k_proj.weight.grad.abs().sum() > 0
    assert kernel.call_count == 2


def _config(implementation='sdpa'):
    return NemotronHConfig(
        hidden_size=16,
        num_hidden_layers=5,
        hybrid_override_pattern='*-*-*',
        vocab_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        mamba_num_heads=2,
        mamba_head_dim=8,
        ssm_state_size=8,
        n_groups=1,
        attention_dropout=0.0,
        _attn_implementation=implementation,
    )


def _guidance(device):
    return AttentionAlignmentGuidance(
        turns=[
            AttentionAlignmentTurn(
                batch_index=0,
                query_positions=torch.tensor([3, 4, 5, 6], device=device),
                key_positions=torch.tensor([0, 1, 2], device=device),
                token_indices=torch.tensor([0, 0, 1, 2], device=device),
            )
        ],
        prior_strength=0.0,
        heads=(0,),
        layers='middle',
        query_chunk_size=2,
    )


def _reference_flash_forward(
    query,
    key,
    value,
    attention_mask,
    query_length,
    dropout,
    sliding_window,
    is_causal,
    use_top_left_mask,
):
    """Exercise the Flash wrapper's layout through a deterministic SDPA kernel."""
    batch, length, _, _ = query.shape
    assert query_length == length and sliding_window is None
    visible = torch.ones(batch, 1, length, key.size(1), device=query.device, dtype=torch.bool)
    if is_causal:
        positions = torch.arange(key.size(1) - length, key.size(1), device=query.device)
        visible &= torch.arange(key.size(1), device=query.device)[None, :] <= positions[:, None]
    if attention_mask is not None:
        visible &= attention_mask[:, None, None, :].bool()
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=visible,
        dropout_p=dropout,
    ).transpose(1, 2)
