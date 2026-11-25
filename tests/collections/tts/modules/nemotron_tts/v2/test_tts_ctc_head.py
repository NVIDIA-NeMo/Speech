# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
from torch.nn import functional as F

from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import CTCTimestampHead

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('blank_id', [0, 4, 8])
@pytest.mark.parametrize('chunk_size', [1, 4])
def test_plain_ctc_matches_full_vocabulary_loss_and_all_gradients(blank_id, chunk_size):
    torch.manual_seed(7)
    head = CTCTimestampHead(
        5,
        9,
        blank_id,
        projection_size=7,
        chunk_size=chunk_size,
        ctc_weight=0.6,
        distillation_weight=0.0,
    )
    reference = copy.deepcopy(head)
    hidden = torch.randn(7, 5, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    pieces = [value for value in range(9) if value != blank_id]
    targets = torch.tensor([pieces[0], pieces[1], pieces[1], pieces[2], pieces[3]])
    observed = head.compute_loss(hidden, targets, input_start=1, input_length=13)
    log_probs = reference(reference_hidden).flatten(0, 1).float().log_softmax(-1)
    expected = (
        F.ctc_loss(
            log_probs[1:13, None],
            targets,
            torch.tensor([12]),
            torch.tensor([targets.numel()]),
            blank=blank_id,
            reduction='sum',
        )
        / targets.numel()
    )
    torch.testing.assert_close(observed['ctc_loss'], expected)
    torch.testing.assert_close(observed['loss'], 0.6 * expected)
    assert observed['distillation_loss'] == 0 and observed['distilled_subframes'] == 0
    observed['loss'].backward()
    (0.6 * expected).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=2e-6, rtol=2e-5)
    for actual, expected_parameter in zip(head.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected_parameter.grad, atol=2e-6, rtol=2e-5)


def test_plain_ctc_retains_two_ordered_predictions_from_the_same_chunks():
    head = CTCTimestampHead(5, 9, 8, projection_size=7, chunk_size=2, distillation_weight=0)
    hidden = torch.randn(5, 5, requires_grad=True)
    calls = []
    hook = head.register_forward_hook(lambda _module, args, _output: calls.append(args[0].size(0)))
    try:
        result = head.compute_loss(hidden, torch.tensor([1, 2]), input_length=9, return_predictions=True)
    finally:
        hook.remove()
    assert calls == [2, 2, 1]
    with torch.no_grad():
        probabilities, ids = head(hidden).flatten(0, 1)[:9].float().softmax(-1).max(-1)
    torch.testing.assert_close(result['predicted_ids'], ids)
    torch.testing.assert_close(result['predicted_probabilities'], probabilities)
    assert not result['predicted_probabilities'].requires_grad
    result['loss'].backward()
    assert torch.isfinite(hidden.grad).all()


def test_plain_ctc_requires_distillation_to_be_disabled():
    head = CTCTimestampHead(5, 9, 8)
    with pytest.raises(ValueError, match='teacher probabilities'):
        head.compute_loss(torch.randn(3, 5), torch.tensor([1]))


def test_plain_ctc_rejects_a_distillation_mask_without_a_teacher():
    head = CTCTimestampHead(5, 9, 8, distillation_weight=0)
    with pytest.raises(ValueError, match='distillation_mask'):
        head.compute_loss(torch.randn(3, 5), torch.tensor([1]), distillation_mask=torch.ones(6, dtype=torch.bool))


def test_empty_plain_ctc_returns_a_parameter_connected_zero_and_empty_predictions():
    head = CTCTimestampHead(5, 9, 8, projection_size=7, distillation_weight=0)
    hidden = torch.empty(0, 5, requires_grad=True)
    result = head.compute_loss(hidden, torch.empty(0, dtype=torch.long), return_predictions=True)
    assert result['loss'] == 0 and result['distillation_loss'] == 0
    assert result['predicted_ids'].shape == result['predicted_probabilities'].shape == (0,)
    result['loss'].backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_plain_ctc_low_precision_parameters_keep_fp32_losses_and_finite_gradients():
    head = CTCTimestampHead(
        5,
        9,
        8,
        projection_size=7,
        chunk_size=2,
        distillation_weight=0,
    ).bfloat16()
    hidden = torch.randn(5, 5, requires_grad=True)
    result = head.compute_loss(hidden, torch.tensor([1, 2]), return_predictions=True)
    assert result['loss'].dtype == result['predicted_probabilities'].dtype == torch.float32
    result['loss'].backward()
    assert torch.isfinite(hidden.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in head.parameters())


def test_plain_ctc_infeasible_transcript_remains_explicit_and_parameter_connected():
    head = CTCTimestampHead(5, 9, 8, projection_size=7, distillation_weight=0)
    hidden = torch.randn(1, 5, requires_grad=True)
    result = head.compute_loss(hidden, torch.tensor([1, 1]))
    assert not result['ctc_valid'] and result['infeasible_count'] == 1
    assert result['loss'] == 0
    result['loss'].backward()
    assert all(parameter.grad is not None for parameter in head.parameters())
