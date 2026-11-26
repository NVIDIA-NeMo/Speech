# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from unittest.mock import patch

import pytest
import torch

from nemo.collections.tts.modules.attention_alignment import (
    AttentionAlignmentGuidance,
    AttentionAlignmentTurn,
    apply_attention_alignment,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('strength', [0.0, 0.25, 1.0])
def test_soft_word_band_prior_and_weighted_mass_match_the_same_unnormalized_weights(strength):
    q = torch.zeros(1, 1, 8, 8)
    values = torch.eye(8).reshape_as(q)
    ordinary = torch.full_like(q, 0.125)
    turn = _turn((7,), tuple(range(8)), (2,), (4,), 0.5)
    output, loss = apply_attention_alignment(
        q, q, values, ordinary, AttentionAlignmentGuidance([turn], prior_strength=strength)
    )
    distances = torch.tensor([2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    weights = torch.exp(-0.5 * (distances / 0.5).square())
    expected_loss = -(weights.sum() / 8).log()
    prior = (1 - strength) + strength * weights.clamp_min(1e-6)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(output[0, 0, 7], prior / prior.sum())
    assert torch.equal(output[0, 0, :7], ordinary[0, 0, :7])
    if strength == 0:
        assert output is ordinary


def test_soft_word_band_preserves_freedom_to_redistribute_attention_inside_a_word():
    q = torch.ones(1, 1, 4, 1)
    losses = []
    for probabilities in ((0.9, 0.1), (0.5, 0.5)):
        key = torch.tensor([*probabilities, 0.5, 0.5]).log().reshape_as(q)
        _, loss = apply_attention_alignment(
            q,
            key,
            key,
            q,
            AttentionAlignmentGuidance([_turn((3,), (0, 1), (0,), (2,), 0.5)]),
        )
        losses.append(loss)
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=1e-6)
    assert losses[0].item() == pytest.approx(math.log(2), abs=1e-6)


def test_band_mass_retains_audio_context_probability_and_uses_raw_not_normalized_band_weights():
    q = torch.zeros(1, 1, 8, 8)
    values = torch.eye(8).reshape_as(q)
    _, loss = apply_attention_alignment(
        q,
        q,
        values,
        values,
        AttentionAlignmentGuidance([_turn((7,), (0, 2, 4), (1,), (2,), 0.5)]),
    )
    expected = math.log(8) - math.log(1 + 2 * math.exp(-2))
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_narrow_word_start_band_and_wider_interior_band_use_the_whole_word_plateau():
    q = torch.zeros(1, 1, 8, 8)
    values = torch.eye(8).reshape_as(q)
    ordinary = torch.full_like(q, 0.125)
    narrow, narrow_loss = apply_attention_alignment(
        q,
        q,
        values,
        ordinary,
        AttentionAlignmentGuidance([_turn((7,), tuple(range(8)), (2,), (4,), 0.5)], prior_strength=1.0),
    )
    wide, wide_loss = apply_attention_alignment(
        q,
        q,
        values,
        ordinary,
        AttentionAlignmentGuidance([_turn((7,), tuple(range(8)), (1,), (5,), 1.0)], prior_strength=1.0),
    )
    assert narrow_loss > wide_loss
    assert narrow[0, 0, 7, 2] == narrow[0, 0, 7, 3]
    assert wide[0, 0, 7, 1:5].eq(wide[0, 0, 7, 1]).all()
    assert wide[0, 0, 7, 1] > narrow[0, 0, 7, 1]


@pytest.mark.parametrize(
    'device,dtype',
    [
        ('cpu', torch.float32),
        ('cpu', torch.bfloat16),
        pytest.param('cuda', torch.float32, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')),
        pytest.param(
            'cuda',
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason='BF16 CUDA'
            ),
        ),
    ],
)
@pytest.mark.parametrize('strength', [0.0, 0.6])
def test_soft_band_masked_reference_values_and_all_gradients(device, dtype, strength):
    torch.manual_seed(37)
    q = torch.randn(1, 3, 9, 4, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    ordinary = torch.randn_like(q, requires_grad=True)
    turn = _turn((3, 7), (0, 2, 5, 6), (0, 1), (2, 4), 0.75, device=device)
    mask = torch.zeros(1, 3, 9, 9, device=device)
    mask[0, 0, 3, 0] = -torch.inf  # A later word key remains visible.
    mask[0, 2, 3, [0, 2]] = -torch.inf  # This head has no visible word plateau.
    mask[0, 2, 7, :] = -torch.inf
    output, loss = apply_attention_alignment(
        q,
        k,
        v,
        ordinary,
        AttentionAlignmentGuidance([turn], prior_strength=strength, heads=(0, 2), query_chunk_size=1),
        attention_mask=mask,
    )
    reference_q = q.detach().float().cpu().requires_grad_(True)
    reference_k = k.detach().float().cpu().requires_grad_(True)
    reference_v = v.detach().float().cpu().requires_grad_(True)
    reference_ordinary = ordinary.detach().cpu().requires_grad_(True)
    expected_output, expected_loss = _reference(
        reference_q, reference_k, reference_v, reference_ordinary, [turn], (0, 2), mask.cpu(), strength
    )
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss.cpu(), expected_loss, atol=4e-6, rtol=2e-5)
    torch.testing.assert_close(
        output.cpu(), expected_output, atol=1e-3 if dtype == torch.bfloat16 else 4e-6, rtol=1e-2
    )
    assert torch.equal(output[:, 1], ordinary[:, 1])
    assert torch.equal(output[0, 2, [3, 7]], ordinary[0, 2, [3, 7]])
    if strength == 0:
        assert output is ordinary
    (loss + output.float().square().mean()).backward()
    (expected_loss + expected_output.float().square().mean()).backward()
    for actual, expected in ((q, reference_q), (k, reference_k), (ordinary, reference_ordinary)):
        torch.testing.assert_close(
            actual.grad.cpu(),
            expected.grad.to(dtype),
            atol=1e-3 if dtype == torch.bfloat16 else 4e-6,
            rtol=1e-2,
        )
    if strength:
        torch.testing.assert_close(
            v.grad.cpu(),
            reference_v.grad.to(dtype),
            atol=1e-3 if dtype == torch.bfloat16 else 4e-6,
            rtol=1e-2,
        )
    else:
        assert v.grad is None and reference_v.grad is None


def test_soft_band_loss_is_measured_before_the_forward_prior():
    torch.manual_seed(11)
    q = torch.randn(1, 2, 8, 3, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q)
    ordinary = torch.randn_like(q)
    turn = _turn((4, 7), (0, 2, 3, 6), (0, 2), (2, 4), 0.5)
    free_output, free_loss = apply_attention_alignment(q, k, v, ordinary, AttentionAlignmentGuidance([turn]))
    guided_output, guided_loss = apply_attention_alignment(
        q, k, v, ordinary, AttentionAlignmentGuidance([turn], prior_strength=1.0)
    )
    assert free_output is ordinary and torch.equal(free_loss, guided_loss)
    assert not torch.equal(guided_output[0, 0, [4, 7]], ordinary[0, 0, [4, 7]])


@pytest.mark.parametrize('strength', [0.0, 1.0])
def test_fully_masked_band_heads_keep_ordinary_output_and_finite_zero_gradients(strength):
    q = torch.randn(1, 2, 8, 3, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    ordinary = torch.randn_like(q, requires_grad=True)
    turn = _turn((7,), (0, 2), (0,), (2,), 0.5)
    output, loss = apply_attention_alignment(
        q,
        k,
        v,
        ordinary,
        AttentionAlignmentGuidance([turn], prior_strength=strength, heads=None),
        attention_mask=torch.zeros(1, 2, 8, 8, dtype=torch.bool),
    )
    assert torch.equal(output, ordinary) and torch.isfinite(loss) and loss == 0
    (loss + output.sum()).backward()
    assert q.grad.eq(0).all() and k.grad.eq(0).all()
    assert torch.equal(ordinary.grad, torch.ones_like(ordinary))
    assert v.grad is None or v.grad.eq(0).all()


@pytest.mark.parametrize('width', [0, -1.0, float('inf'), float('nan'), True, '0.5'])
def test_per_turn_band_width_must_be_a_finite_positive_scalar(width):
    q = torch.zeros(1, 1, 8, 2)
    turn = _turn((7,), (0, 2), (0,), (2,), width)
    with pytest.raises(ValueError, match='band_width'):
        apply_attention_alignment(q, q, q, q, AttentionAlignmentGuidance([turn]))


def test_per_turn_band_requires_span_ends_and_plain_spans_still_reject_a_prior():
    q = torch.zeros(1, 1, 8, 2)
    turn = _turn((7,), (0, 2), (0,), None, 0.5)
    with pytest.raises(ValueError, match='band_width'):
        apply_attention_alignment(q, q, q, q, AttentionAlignmentGuidance([turn]))
    turn = _turn((7,), (0, 2), (0,), (2,), None)
    with pytest.raises(ValueError, match='prior_strength=0'):
        apply_attention_alignment(q, q, q, q, AttentionAlignmentGuidance([turn], prior_strength=1.0))


def test_band_prior_reuses_each_chunk_score_matrix_for_both_loss_and_context():
    q = torch.randn(1, 1, 32, 4)
    turn = _turn((10, 20, 30), (0, 2, 4), (0, 0, 0), (3, 3, 3), 1.0)
    with torch.no_grad(), patch('torch.matmul', wraps=torch.matmul) as matmul:
        output, loss = apply_attention_alignment(
            q, q, q, q, AttentionAlignmentGuidance([turn], prior_strength=1.0, query_chunk_size=2)
        )
    assert torch.isfinite(loss) and output.shape == q.shape
    score_calls = [call for call in matmul.call_args_list if call.args[0].size(-1) == 4]
    assert len(score_calls) == 2 and matmul.call_count == 4
    assert [call.args[0].size(1) for call in score_calls] == [2, 1]


def _turn(rows, keys, starts, ends, width, device='cpu'):
    return AttentionAlignmentTurn(
        batch_index=0,
        query_positions=torch.tensor(rows, dtype=torch.long, device=device),
        key_positions=torch.tensor(keys, dtype=torch.long, device=device),
        token_indices=torch.tensor(starts, dtype=torch.long, device=device),
        token_end_indices=None if ends is None else torch.tensor(ends, dtype=torch.long, device=device),
        band_width=width,
    )


def _reference(q, k, v, ordinary, turns, heads, mask, strength):
    output = ordinary.clone() if strength else ordinary
    losses = []
    for turn in turns:
        keys = turn.key_positions.cpu()
        for row, first, end in zip(
            turn.query_positions.tolist(), turn.token_indices.tolist(), turn.token_end_indices.tolist()
        ):
            indices = torch.arange(len(keys))
            distance = (first - indices).clamp_min(0) + (indices - end + 1).clamp_min(0)
            log_weights = -0.5 * (distance.float() / turn.band_width).square()
            for head in heads:
                scores = q[0, head, row] @ k[0, head].T / math.sqrt(q.size(-1))
                selected_mask = mask[0, head, row]
                visible = (torch.arange(k.size(2)) <= row) & torch.isfinite(selected_mask)
                if not visible[keys[first:end]].any():
                    continue
                scores = (scores + selected_mask).masked_fill(~visible, -torch.inf)
                word_scores = (scores[keys] + log_weights).masked_fill(~visible[keys], -torch.inf)
                losses.append(scores.logsumexp(-1) - word_scores.logsumexp(-1))
                if strength:
                    prior = torch.full_like(scores, 1e-6)
                    prior[keys] = log_weights.exp().clamp_min(1e-6)
                    prior = 1 - strength + strength * prior
                    probabilities = (scores + prior.log()).softmax(-1)
                    output[0, head, row] = (probabilities @ v[0, head]).to(ordinary.dtype)
    loss = torch.stack(losses).mean() if losses else q.sum() * 0 + k.sum() * 0
    return output, loss
