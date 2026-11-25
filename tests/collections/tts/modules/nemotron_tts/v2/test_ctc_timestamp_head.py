# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
from torch.nn import functional as F

from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import CTCTimestampHead

pytestmark = pytest.mark.unit


def _teacher(frames, classes=9, blank_id=8, topk=3):
    probabilities = torch.randn(frames, classes).softmax(-1)
    candidates = probabilities.clone()
    candidates[:, blank_id] = -1
    indices = candidates.topk(min(topk, classes - 1), dim=-1).indices
    indices = torch.cat([indices, torch.full((frames, 1), blank_id)], dim=-1)
    selected = probabilities.gather(-1, indices)
    return {
        'indices': indices,
        'probabilities': selected,
        'tail_probabilities': (1 - selected.sum(-1)).clamp_min(0),
    }


def _reference(head, hidden, targets, teacher, valid=None, length=None, start=0):
    logits = head(hidden).flatten(0, 1).float()
    length = logits.size(0) if length is None else length
    logits = logits[:length]
    log_probs = logits.log_softmax(-1)
    ctc = F.ctc_loss(
        log_probs[start:].unsqueeze(1),
        targets,
        torch.tensor([length - start]),
        torch.tensor([targets.numel()]),
        blank=head.blank_id,
        reduction='sum',
    ) / max(targets.numel(), 1)
    indices = teacher['indices'][:length]
    probabilities = teacher['probabilities'][:length]
    tail = teacher['tail_probabilities'][:length]
    selected = log_probs.gather(-1, indices)
    terms = probabilities * (probabilities.clamp_min(torch.finfo(torch.float32).tiny).log() - selected)
    kl = terms.sum(-1)
    if indices.size(-1) < head.vocab_size:
        other = logits.scatter(-1, indices, float('-inf')).logsumexp(-1) - logits.logsumexp(-1)
        kl = kl + tail * (tail.clamp_min(torch.finfo(torch.float32).tiny).log() - other)
    valid = torch.ones(length, dtype=torch.bool) if valid is None else valid[:length]
    kl = kl[valid].sum() / valid.sum().clamp_min(1)
    return ctc, kl


def test_ctc_head_returns_two_ordered_predictions_and_streams_without_future_context():
    torch.manual_seed(1)
    head = CTCTimestampHead(5, 9, 8, projection_size=7)
    hidden = torch.randn(2, 6, 5)
    logits = head(hidden)
    assert logits.shape == (2, 6, 2, 9)
    online = torch.stack([head(hidden[:, step]) for step in range(hidden.size(1))], dim=1)
    torch.testing.assert_close(logits, online)
    modified = hidden.clone()
    modified[:, 3:] = 100
    torch.testing.assert_close(head(modified)[:, :3], logits[:, :3])
    assert not torch.allclose(logits[:, :, 0], logits[:, :, 1])


@pytest.mark.parametrize('blank_id', [0, 4, 8])
@pytest.mark.parametrize('topk', [3, 8])
@pytest.mark.parametrize('chunk_size', [1, 4])
def test_compact_ctc_and_sparse_kl_match_full_projection_losses_and_gradients(blank_id, topk, chunk_size):
    torch.manual_seed(2)
    head = CTCTimestampHead(5, 9, blank_id, projection_size=7, chunk_size=chunk_size)
    reference = copy.deepcopy(head)
    hidden = torch.randn(6, 5, requires_grad=True)
    full_hidden = hidden.detach().clone().requires_grad_(True)
    labels = [value for value in range(9) if value != blank_id]
    targets = torch.tensor([labels[1], labels[1], labels[3], labels[5]])
    teacher = _teacher(12, blank_id=blank_id, topk=topk)
    valid = torch.tensor([True, False, True, True, False, True, True, False, True, True, True, False])
    result = head.compute_loss(hidden, targets, teacher, distillation_mask=valid)
    ctc, kl = _reference(reference, full_hidden, targets, teacher, valid)
    torch.testing.assert_close(result['ctc_loss'], ctc)
    torch.testing.assert_close(result['distillation_loss'], kl)
    result['loss'].backward()
    (ctc + kl).backward()
    torch.testing.assert_close(hidden.grad, full_hidden.grad, atol=2e-6, rtol=2e-5)
    for parameter, expected in zip(head.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-6, rtol=2e-5)
    assert result['ctc_valid'] and result['infeasible_count'] == 0
    assert result['supervised_subframes'] == 12
    assert result['distilled_subframes'] == valid.sum()


def test_compact_ctc_supports_transcript_using_every_nonblank_class():
    torch.manual_seed(3)
    head = CTCTimestampHead(5, 4, 2, projection_size=7, chunk_size=1)
    hidden = torch.randn(3, 5, requires_grad=True)
    targets = torch.tensor([3, 0, 1])
    teacher = _teacher(6, classes=4, blank_id=2, topk=3)
    result = head.compute_loss(hidden, targets, teacher)
    expected, _ = _reference(head, hidden, targets, teacher)
    torch.testing.assert_close(result['ctc_loss'], expected)
    result['loss'].backward()
    assert torch.isfinite(hidden.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in head.parameters())


def test_partial_last_audio_frame_excludes_padded_ctc_subframe():
    torch.manual_seed(4)
    head = CTCTimestampHead(5, 9, 8, projection_size=7, chunk_size=2)
    hidden = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([2, 4])
    teacher = _teacher(8)
    result = head.compute_loss(hidden, targets, teacher, input_length=7)
    ctc, kl = _reference(head, hidden, targets, teacher, length=7)
    torch.testing.assert_close(result['ctc_loss'], ctc)
    torch.testing.assert_close(result['distillation_loss'], kl)
    assert result['supervised_subframes'] == 7
    changed = {key: value.clone() for key, value in teacher.items()}
    changed['probabilities'][7] = float('nan')
    changed['tail_probabilities'][7] = float('nan')
    observed = head.compute_loss(hidden, targets, changed, input_length=7)
    torch.testing.assert_close(observed['loss'], result['loss'], atol=0, rtol=0)


def test_partial_first_and_last_subframes_match_full_vocabulary_ctc_slice_and_gradients():
    torch.manual_seed(7)
    head = CTCTimestampHead(5, 9, 8, projection_size=7, chunk_size=2)
    reference = copy.deepcopy(head)
    hidden = torch.randn(5, 5, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    targets = torch.tensor([2, 2, 4])
    teacher = _teacher(10)
    valid = torch.tensor([False, True, True, True, False, True, True, True, True, False])
    result = head.compute_loss(hidden, targets, teacher, valid, input_length=9, input_start=1)
    ctc, kl = _reference(reference, reference_hidden, targets, teacher, valid, length=9, start=1)
    torch.testing.assert_close(result['ctc_loss'], ctc)
    torch.testing.assert_close(result['distillation_loss'], kl)
    assert result['supervised_subframes'] == 8
    assert result['distilled_subframes'] == valid[:9].sum()
    result['loss'].backward()
    (ctc + kl).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=2e-6, rtol=2e-5)
    for parameter, expected in zip(head.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-6, rtol=2e-5)


def test_repeated_target_feasibility_uses_length_after_partial_first_subframe():
    head = CTCTimestampHead(5, 9, 8, projection_size=7)
    hidden = torch.randn(4, 5)
    targets, teacher = torch.tensor([1, 1, 1, 1]), _teacher(8)
    full = head.compute_loss(hidden, targets, teacher, input_length=7)
    cropped = head.compute_loss(hidden, targets, teacher, input_length=7, input_start=1)
    assert full['ctc_valid'] is True and full['infeasible_count'] == 0
    assert cropped['ctc_valid'] is False and cropped['infeasible_count'] == 1
    assert cropped['ctc_loss'] == 0
    assert cropped['supervised_subframes'] == 6
    torch.testing.assert_close(cropped['distillation_loss'], full['distillation_loss'])


def test_empty_ctc_slice_retains_independently_masked_distillation():
    head = CTCTimestampHead(5, 9, 8, projection_size=7)
    hidden = torch.randn(3, 5, requires_grad=True)
    result = head.compute_loss(hidden, torch.empty(0, dtype=torch.long), _teacher(6), input_length=5, input_start=5)
    assert result['ctc_valid'] is True and result['ctc_loss'] == 0
    assert result['supervised_subframes'] == 0 and result['distilled_subframes'] == 5
    result['loss'].backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0


def test_infeasible_repeated_transcript_is_reported_and_keeps_distillation_gradients():
    head = CTCTimestampHead(5, 9, 8, projection_size=7)
    hidden = torch.randn(2, 5, requires_grad=True)
    result = head.compute_loss(hidden, torch.tensor([1, 1, 1]), _teacher(4))
    assert result['ctc_valid'] is False and result['infeasible_count'] == 1
    assert result['ctc_loss'] == 0 and result['distillation_loss'] > 0
    result['loss'].backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_empty_turn_returns_graph_zero_connected_to_every_parameter():
    head = CTCTimestampHead(5, 9, 8, projection_size=7)
    hidden = torch.empty(0, 5, requires_grad=True)
    result = head.compute_loss(hidden, torch.empty(0, dtype=torch.long), _teacher(0))
    assert result['loss'].dtype == torch.float32
    assert result['loss'] == 0 and result['ctc_valid'] is True
    result['loss'].backward()
    assert hidden.grad is not None
    for parameter in head.parameters():
        assert parameter.grad is not None and parameter.grad.eq(0).all()


def test_empty_transcript_trains_all_blank_with_finite_gradients():
    head = CTCTimestampHead(5, 9, 8, projection_size=7)
    hidden = torch.randn(3, 5, requires_grad=True)
    targets = torch.empty(0, dtype=torch.long)
    teacher = _teacher(6)
    result = head.compute_loss(hidden, targets, teacher)
    ctc, _ = _reference(head, hidden, targets, teacher)
    torch.testing.assert_close(result['ctc_loss'], ctc)
    result['loss'].backward()
    assert torch.isfinite(hidden.grad).all()


def test_teacher_tensors_are_detached_and_fully_masked_kl_is_zero():
    head = CTCTimestampHead(5, 9, 8, projection_size=7)
    teacher = _teacher(6)
    teacher['probabilities'].requires_grad_(True)
    teacher['tail_probabilities'].requires_grad_(True)
    hidden = torch.randn(3, 5, requires_grad=True)
    result = head.compute_loss(
        hidden, torch.tensor([1, 3]), teacher, distillation_mask=torch.zeros(6, dtype=torch.bool)
    )
    assert result['distillation_loss'] == 0
    result['loss'].backward()
    assert teacher['probabilities'].grad is None and teacher['tail_probabilities'].grad is None
    assert torch.isfinite(hidden.grad).all()


def test_low_precision_head_preserves_fp32_loss_arithmetic_and_fp32_residual_input():
    torch.manual_seed(5)
    head = CTCTimestampHead(5, 9, 8, projection_size=7, chunk_size=1).to(dtype=torch.bfloat16)
    hidden = torch.randn(3, 5, requires_grad=True)
    teacher = _teacher(6)
    result = head.compute_loss(hidden, torch.tensor([2, 3]), teacher)
    for name in ('loss', 'ctc_loss', 'distillation_loss'):
        assert result[name].dtype == torch.float32 and torch.isfinite(result[name])
    result['loss'].backward()
    assert torch.isfinite(hidden.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())


def test_loss_weights_apply_to_raw_components():
    head = CTCTimestampHead(5, 9, 8, projection_size=7, ctc_weight=0.3, distillation_weight=2.0)
    result = head.compute_loss(torch.randn(3, 5), torch.tensor([2, 3]), _teacher(6))
    torch.testing.assert_close(result['loss'], 0.3 * result['ctc_loss'] + 2 * result['distillation_loss'])


def test_saturated_blank_and_zero_teacher_tail_keep_finite_losses_and_gradients():
    head = CTCTimestampHead(5, 9, 8, projection_size=7, chunk_size=1)
    with torch.no_grad():
        head.projection[-1].weight.zero_()
        head.projection[-1].bias.fill_(-10000)
        head.projection[-1].bias.view(2, 9)[:, 8] = 10000
    hidden = torch.randn(3, 5, requires_grad=True)
    teacher = {
        'indices': torch.tensor([[1, 8]]).expand(6, -1),
        'probabilities': torch.tensor([[0.0, 1.0]]).expand(6, -1),
        'tail_probabilities': torch.zeros(6),
    }
    result = head.compute_loss(hidden, torch.tensor([1]), teacher)
    assert torch.isfinite(result['loss']) and result['distillation_loss'] == 0
    result['loss'].backward()
    assert torch.isfinite(hidden.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())


@pytest.mark.parametrize(
    'options',
    [
        {'hidden_size': 0},
        {'vocab_size': 0},
        {'projection_size': False},
        {'chunk_size': -1},
        {'blank_id': -1},
        {'blank_id': 9},
        {'ctc_weight': -1},
        {'distillation_weight': float('nan')},
    ],
)
def test_invalid_head_configuration(options):
    arguments = dict(hidden_size=5, vocab_size=9, blank_id=8)
    arguments.update(options)
    with pytest.raises(ValueError):
        CTCTimestampHead(**arguments)


@pytest.mark.parametrize('targets', [torch.tensor([8]), torch.tensor([-1]), torch.tensor([9])])
def test_invalid_ctc_transcript_is_rejected(targets):
    head = CTCTimestampHead(5, 9, 8)
    with pytest.raises(ValueError):
        head.compute_loss(torch.randn(3, 5), targets, _teacher(6))


@pytest.mark.parametrize('length', [-1, 7, True, 1.5])
def test_invalid_subframe_length_is_rejected(length):
    head = CTCTimestampHead(5, 9, 8)
    with pytest.raises(ValueError):
        head.compute_loss(torch.randn(3, 5), torch.tensor([1]), _teacher(6), input_length=length)


@pytest.mark.parametrize('start', [-1, 6, True, 1.5])
def test_invalid_subframe_start_is_rejected(start):
    head = CTCTimestampHead(5, 9, 8)
    with pytest.raises(ValueError):
        head.compute_loss(torch.randn(3, 5), torch.tensor([1]), _teacher(6), input_length=5, input_start=start)
