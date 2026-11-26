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
@pytest.mark.parametrize('chunk_size', [1, 4])
def test_word_mass_matches_full_key_reference_values_and_gradients(device, dtype, chunk_size):
    torch.manual_seed(42)
    q = torch.randn(2, 3, 9, 4, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    ordinary = torch.randn_like(q, requires_grad=True)
    turns = [
        _turn((3, 7), (0, 1, 2, 5, 6), (0, 2), (2, 5), device=device),
        _turn((4,), (0, 3, 6), (1,), (3,), batch=1, device=device),
    ]
    actual, loss = apply_attention_alignment(
        q, k, v, ordinary, AttentionAlignmentGuidance(turns, heads=(0, 2), query_chunk_size=chunk_size)
    )
    reference_q = q.detach().float().cpu().requires_grad_(True)
    reference_k = k.detach().float().cpu().requires_grad_(True)
    reference = _reference(reference_q, reference_k, turns, (0, 2))
    assert actual is ordinary and loss.dtype == torch.float32
    torch.testing.assert_close(loss.cpu(), reference, atol=3e-6, rtol=2e-5)
    loss.backward()
    reference.backward()
    for actual_gradient, expected in ((q.grad, reference_q.grad), (k.grad, reference_k.grad)):
        torch.testing.assert_close(
            actual_gradient.cpu(),
            expected.to(dtype),
            atol=1e-3 if dtype == torch.bfloat16 else 3e-6,
            rtol=1e-2,
        )
    assert v.grad is None and ordinary.grad is None


def test_word_loss_is_invariant_to_redistribution_among_word_bpe_keys():
    q = torch.ones(1, 1, 4, 1)
    losses = []
    for word_probabilities in ((0.9, 0.1), (0.5, 0.5)):
        k = torch.tensor([*word_probabilities, 0.5, 0.5]).log().reshape_as(q)
        ordinary = torch.randn_like(q)
        output, loss = apply_attention_alignment(
            q, k, k, ordinary, AttentionAlignmentGuidance([_turn((3,), (0, 1), (0,), (2,))])
        )
        assert output is ordinary
        losses.append(loss)
    torch.testing.assert_close(losses[0], losses[1], atol=1e-6, rtol=0)
    assert losses[0].item() == pytest.approx(math.log(2), abs=1e-6)


def test_word_mass_keeps_context_and_audio_keys_in_the_normalizer():
    q = torch.ones(1, 1, 7, 1)
    k = torch.zeros_like(q)
    k[:, :, 6] = 10
    _, loss = apply_attention_alignment(q, k, k, q, AttentionAlignmentGuidance([_turn((6,), (0, 1), (0,), (2,))]))
    assert loss > 9.0


def test_sparse_word_loss_trains_only_anchor_queries_and_selected_head_keys():
    torch.manual_seed(6)
    q = torch.randn(1, 3, 8, 3, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    ordinary = torch.randn_like(q, requires_grad=True)
    output, loss = apply_attention_alignment(
        q,
        k,
        v,
        ordinary,
        AttentionAlignmentGuidance([_turn((3, 6), (0, 2, 5), (0, 1), (2, 3))], heads=(1,), query_chunk_size=1),
    )
    assert output is ordinary
    loss.backward()
    assert q.grad[:, (0, 2)].eq(0).all() and k.grad[:, (0, 2)].eq(0).all()
    assert q.grad[0, 1, [0, 1, 2, 4, 5, 7]].eq(0).all()
    assert q.grad[0, 1, [3, 6]].abs().sum() > 0
    assert k.grad[0, 1, 1].abs().sum() > 0  # A visible non-word key still competes for attention.
    assert k.grad[0, 1, 7].eq(0).all()
    assert v.grad is None and ordinary.grad is None


@pytest.mark.parametrize('mask_kind', ['padding', 'head_boolean', 'head_additive', 'broadcast'])
def test_word_spans_use_any_visible_key_and_respect_padding_causality_and_head_masks(mask_kind):
    torch.manual_seed(13)
    q = torch.randn(1, 2, 8, 3, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    turn = _turn((2, 4, 7), (0, 2, 6), (0, 0, 1), (2, 2, 3))
    if mask_kind == 'padding':
        mask = torch.tensor([[False, True, True, True, True, True, True, False]])
    elif mask_kind == 'broadcast':
        mask = torch.ones(1, 1, 1, 8, dtype=torch.bool)
        mask[..., 0] = False
    else:
        mask = torch.ones(1, 2, 8, 8, dtype=torch.bool)
        mask[0, 0, 2, 0] = False  # Later keys in the same span remain usable.
        mask[0, 0, 4, :] = False  # An entirely masked head/query must stay finite.
        mask[0, 1, 4, [0, 2]] = False  # No word key, despite visible context keys.
        mask[0, 0, 7, 2] = False
        if mask_kind == 'head_additive':
            mask = torch.zeros(mask.shape).masked_fill(~mask, -torch.inf)
            mask[0, 0, 2, 2] = 1.25
    _, actual = apply_attention_alignment(
        q, k, k, q, AttentionAlignmentGuidance([turn], heads=(0, 1), query_chunk_size=2), attention_mask=mask
    )
    expected = _reference(q, k, [turn], (0, 1), mask)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    actual.backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()


@pytest.mark.parametrize('kind', ['future', 'all_masked', 'word_masked'])
def test_invisible_word_spans_return_finite_graph_connected_zero(kind):
    q = torch.randn(1, 2, 8, 3, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    mask = None
    turn = _turn((2,), (6, 7), (0,), (2,))
    if kind != 'future':
        turn = _turn((7,), (0, 2), (0,), (2,))
        mask = torch.ones(1, 2, 8, 8, dtype=torch.bool)
        if kind == 'all_masked':
            mask.zero_()
        else:
            mask[..., [0, 2]] = False
    output, loss = apply_attention_alignment(q, k, k, q, AttentionAlignmentGuidance([turn]), attention_mask=mask)
    assert output is q and torch.isfinite(loss) and loss == 0
    loss.backward()
    assert q.grad.eq(0).all() and k.grad.eq(0).all()


@pytest.mark.parametrize('starts,ends', [((-1,), (1,)), ((1,), (1,)), ((2,), (1,)), ((0,), (4,))])
def test_word_span_bounds_are_validated(starts, ends):
    q = torch.zeros(1, 1, 8, 2)
    with pytest.raises(ValueError, match='Word-span indices'):
        apply_attention_alignment(q, q, q, q, AttentionAlignmentGuidance([_turn((7,), (0, 2, 4), starts, ends)]))


@pytest.mark.parametrize('invalid', ['shape', 'dtype', 'device'])
def test_word_span_end_indices_require_matching_integer_shape_and_device(invalid):
    turn = _turn((7,), (0, 2), (0,), (2,))
    turn.token_end_indices = {
        'shape': torch.tensor([[2]]),
        'dtype': torch.tensor([2.0]),
        'device': torch.empty(1, dtype=torch.long, device='meta'),
    }[invalid]
    q = torch.zeros(1, 1, 8, 2)
    with pytest.raises(ValueError):
        apply_attention_alignment(q, q, q, q, AttentionAlignmentGuidance([turn]))


def test_word_span_mode_rejects_an_attention_prior():
    q = torch.zeros(1, 1, 8, 2)
    with pytest.raises(ValueError, match='prior_strength=0'):
        apply_attention_alignment(
            q, q, q, q, AttentionAlignmentGuidance([_turn((7,), (0, 2), (0,), (2,))], prior_strength=0.1)
        )


def test_word_supervision_projects_only_sparse_query_chunks_once():
    q = torch.randn(1, 1, 64, 4)
    turn = _turn((10, 20, 30, 50, 60), (0, 1, 2), (0, 0, 0, 0, 0), (3, 3, 3, 3, 3))
    with torch.no_grad(), patch('torch.matmul', wraps=torch.matmul) as matmul:
        output, loss = apply_attention_alignment(q, q, q, q, AttentionAlignmentGuidance([turn], query_chunk_size=2))
    assert output is q and torch.isfinite(loss)
    assert matmul.call_count == 3
    assert [call.args[0].size(1) for call in matmul.call_args_list] == [2, 2, 1]


@pytest.mark.parametrize('word_first', [False, True])
def test_word_spans_and_legacy_gaussian_targets_share_the_loss_average(word_first):
    q = torch.zeros(1, 1, 6, 2)
    gaussian = _turn((3,), (0, 1), (0,), None)
    word = _turn((5,), (0, 1), (0,), (2,))
    turns = [word, gaussian] if word_first else [gaussian, word]
    output, loss = apply_attention_alignment(q, q, q, q, AttentionAlignmentGuidance(turns))
    assert output is q
    assert loss.item() == pytest.approx((math.log(4) + math.log(3)) / 2, abs=1e-6)


def _turn(queries, keys, starts, ends, batch=0, device='cpu'):
    return AttentionAlignmentTurn(
        batch_index=batch,
        query_positions=torch.tensor(queries, dtype=torch.long, device=device),
        key_positions=torch.tensor(keys, dtype=torch.long, device=device),
        token_indices=torch.tensor(starts, dtype=torch.long, device=device),
        token_end_indices=None if ends is None else torch.tensor(ends, dtype=torch.long, device=device),
    )


def _reference(q, k, turns, heads, mask=None):
    losses = []
    for turn in turns:
        keys = turn.key_positions.tolist()
        for row, start, end in zip(
            turn.query_positions.tolist(), turn.token_indices.tolist(), turn.token_end_indices.tolist()
        ):
            if mask is not None and mask.ndim == 2 and not mask[turn.batch_index, row]:
                continue
            for head in heads:
                scores = (q[turn.batch_index, head, row].float() @ k[turn.batch_index, head].float().T) / math.sqrt(
                    q.size(-1)
                )
                visible = torch.arange(k.size(2), device=q.device) <= row
                if mask is not None:
                    if mask.ndim == 2:
                        selected = mask[turn.batch_index].bool()
                    else:
                        selected = mask[
                            turn.batch_index if mask.size(0) > 1 else 0,
                            head if mask.size(1) > 1 else 0,
                            row if mask.size(2) > 1 else 0,
                        ]
                    if selected.dtype.is_floating_point:
                        visible &= torch.isfinite(selected) & (selected > torch.finfo(selected.dtype).min / 2)
                        scores = scores + selected
                    else:
                        visible &= selected.bool()
                word_keys = [key for key in keys[start:end] if visible[key]]
                if word_keys:
                    losses.append(
                        scores.masked_fill(~visible, -torch.inf).logsumexp(-1) - scores[word_keys].logsumexp(-1)
                    )
    return torch.stack(losses).mean() if losses else q.float().sum() * 0 + k.float().sum() * 0
