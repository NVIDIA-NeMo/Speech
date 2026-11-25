# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import CTCTimestampHead

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    'device',
    [
        'cpu',
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason='CUDA is unavailable',
            ),
        ),
    ],
)
@pytest.mark.parametrize('blank_id', [0, 4, 8])
def test_word_ctc_matches_full_vocabulary_cpu_values_and_all_gradients(device, blank_id):
    torch.manual_seed(7)
    reference = CTCTimestampHead(5, 9, blank_id, projection_size=7, chunk_size=3, distillation_weight=0)
    head = copy.deepcopy(reference).to(device)
    reference_hidden = torch.randn(7, 5, requires_grad=True)
    hidden = reference_hidden.detach().clone().to(device).requires_grad_(True)
    pieces = [token for token in range(9) if token != blank_id]
    targets = torch.tensor([pieces[0], pieces[1], pieces[1], pieces[2], pieces[3]])
    intervals = [_word(0, 1, 1, 4), _word(1, 3, 3, 9), _word(3, 5, 8, 13)]
    calls = []
    hook = head.register_forward_hook(lambda _module, args, _result: calls.append(args[0].size(0)))
    try:
        result = head.compute_loss(
            hidden,
            targets,
            input_start=1,
            input_length=13,
            word_intervals=intervals,
            return_predictions=True,
        )
    finally:
        hook.remove()
    assert calls == [3, 3, 1]
    logits = reference(reference_hidden).flatten(0, 1).float().log_softmax(-1)
    expected_global = _ctc(logits[1:13], targets, blank_id)
    expected_word = torch.stack(
        [
            _ctc(
                logits[word['start_frame'] : word['end_frame']],
                targets[word['token_start'] : word['token_end']],
                blank_id,
            )
            for word in intervals
        ]
    ).mean()
    torch.testing.assert_close(result['ctc_loss'].cpu(), expected_global, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(result['word_ctc_loss'].cpu(), expected_word, atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(result['loss'], result['ctc_loss'], atol=0, rtol=0)
    assert result['word_ctc_valid_intervals'] == 3
    assert result['word_ctc_infeasible_intervals'] == result['word_ctc_skipped_intervals'] == 0
    (result['loss'] + 0.7 * result['word_ctc_loss']).backward()
    (expected_global + 0.7 * expected_word).backward()
    torch.testing.assert_close(hidden.grad.cpu(), reference_hidden.grad, atol=3e-5, rtol=3e-4)
    for actual, expected in zip(head.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad.cpu(), expected.grad, atol=3e-5, rtol=3e-4)


def test_word_ctc_penalizes_early_copying_that_global_ctc_accepts():
    head = _logits_head(4)
    early = torch.full((8, 4), -8.0)
    early[:, 0] = 8
    early[0, [0, 1]] = torch.tensor([-8.0, 8.0])
    early[1, [0, 2]] = torch.tensor([-8.0, 8.0])
    timed = torch.full((8, 4), -3.0)
    timed[:, 0] = 3
    timed[2, [0, 1]] = torch.tensor([-3.0, 3.0])
    timed[6, [0, 2]] = torch.tensor([-3.0, 3.0])
    targets = torch.tensor([1, 2])
    intervals = [_word(0, 1, 2, 4), _word(1, 2, 6, 8)]
    results = [head.compute_loss(values.reshape(4, 8), targets, word_intervals=intervals) for values in (early, timed)]
    assert results[0]['ctc_loss'] < results[1]['ctc_loss']
    assert results[0]['word_ctc_loss'] > results[1]['word_ctc_loss'] + 5


def test_word_ctc_keeps_other_word_and_nontranscript_probability_mass():
    head = _logits_head(4)
    logits = torch.tensor(
        [[0.0, 1.0, 5.0, 3.0], [0.0, 1.0, 5.0, 3.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], requires_grad=True
    )
    result = head.compute_loss(logits.reshape(2, 8), torch.tensor([1, 2]), word_intervals=[_word(0, 1, 0, 2)])
    expected = _ctc(logits[:2].log_softmax(-1), torch.tensor([1]), 0)
    torch.testing.assert_close(result['word_ctc_loss'], expected)
    result['word_ctc_loss'].backward()
    assert logits.grad[:2, 2:].gt(0).all()
    assert logits.grad[2:].eq(0).all()


def test_word_ctc_clips_partial_boundary_subframes_and_reports_bad_or_infeasible_intervals():
    head = CTCTimestampHead(5, 7, 6, projection_size=8, distillation_weight=0)
    hidden = torch.randn(5, 5, requires_grad=True)
    targets = torch.tensor([1, 2, 2])
    intervals = [
        _word(0, 1, -5, 4),
        _word(1, 2, 7, 20),
        _word(1, 3, 6, 8),
        _word(0, 0, 2, 4),
        _word(0, 4, 2, 8),
        _word(0, 1, -9, -1),
        _word(0, 1, 4, 4),
    ]
    result = head.compute_loss(hidden, targets, input_start=2, input_length=9, word_intervals=intervals)
    full = head(hidden).flatten(0, 1).float().log_softmax(-1)
    expected = (_ctc(full[2:4], targets[:1], 6) + _ctc(full[7:9], targets[1:2], 6)) / 2
    torch.testing.assert_close(result['word_ctc_loss'], expected)
    assert result['word_ctc_valid_intervals'] == 2
    assert result['word_ctc_infeasible_intervals'] == 1
    assert result['word_ctc_skipped_intervals'] == 4
    result['word_ctc_loss'].backward()
    assert torch.isfinite(hidden.grad).all()


def test_word_ctc_retains_the_global_blank_constraint_between_repeated_words():
    head = _logits_head(3)
    logits = torch.tensor([[-6.0, 6.0, -6.0]] * 4)
    result = head.compute_loss(
        logits.reshape(2, 6), torch.tensor([1, 1]), word_intervals=[_word(0, 1, 0, 2), _word(1, 2, 2, 4)]
    )
    assert result['word_ctc_loss'] < 0.001
    assert result['loss'] > result['word_ctc_loss'] + 4
    assert result['ctc_valid']


@pytest.mark.parametrize('empty_hidden', [False, True])
def test_empty_or_infeasible_word_ctc_is_a_finite_parameter_connected_zero(empty_hidden):
    head = CTCTimestampHead(5, 7, 6, projection_size=8, distillation_weight=0)
    hidden = torch.randn(0 if empty_hidden else 1, 5, requires_grad=True)
    result = head.compute_loss(hidden, torch.tensor([1, 1]), word_intervals=[_word(0, 2, 0, 2)])
    assert result['word_ctc_loss'] == 0 and torch.isfinite(result['word_ctc_loss'])
    assert result['word_ctc_valid_intervals'] == 0
    assert result['word_ctc_skipped_intervals'] == int(empty_hidden)
    assert result['word_ctc_infeasible_intervals'] == int(not empty_hidden)
    result['word_ctc_loss'].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())


def test_word_ctc_uses_bounded_batched_dp_without_reprojecting_each_word():
    head = CTCTimestampHead(5, 7, 6, projection_size=8, chunk_size=128, distillation_weight=0)
    hidden = torch.randn(66, 5)
    intervals = torch.tensor([[token, token + 1, 2 * token, 2 * token + 2] for token in range(65)])
    calls = []
    hook = head.register_forward_hook(lambda _module, args, _result: calls.append(args[0].size(0)))
    try:
        with patch('torch.nn.functional.ctc_loss', wraps=F.ctc_loss) as ctc:
            result = head.compute_loss(hidden, torch.ones(65, dtype=torch.long), word_intervals=intervals)
    finally:
        hook.remove()
    assert calls == [66]
    assert [call.args[0].size(1) for call in ctc.call_args_list] == [1, 64, 1]
    assert result['word_ctc_valid_intervals'] == 65


def test_disabled_and_empty_word_intervals_leave_existing_objective_unchanged():
    head = CTCTimestampHead(5, 7, 6, projection_size=8, distillation_weight=0)
    hidden = torch.randn(5, 5)
    with patch('torch.nn.functional.ctc_loss', wraps=F.ctc_loss) as ctc:
        baseline = head.compute_loss(hidden, torch.tensor([1, 2]))
        observed = head.compute_loss(hidden, torch.tensor([1, 2]), word_intervals=[])
    assert ctc.call_count == 2
    assert not any(key.startswith('word_ctc') for key in baseline)
    for key, expected in baseline.items():
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(observed[key], expected, atol=0, rtol=0)
        else:
            assert observed[key] == expected
    assert observed['word_ctc_loss'] == 0


def test_word_ctc_bf16_parameters_keep_fp32_loss_and_finite_gradients():
    head = CTCTimestampHead(5, 7, 6, projection_size=8, distillation_weight=0).bfloat16()
    hidden = torch.randn(5, 5, requires_grad=True)
    result = head.compute_loss(hidden, torch.tensor([1, 2]), word_intervals=[_word(0, 1, 0, 4), _word(1, 2, 4, 10)])
    assert result['word_ctc_loss'].dtype == torch.float32
    result['word_ctc_loss'].backward()
    assert torch.isfinite(hidden.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in head.parameters())


def _word(token_start, token_end, start_frame, end_frame):
    return dict(token_start=token_start, token_end=token_end, start_frame=start_frame, end_frame=end_frame)


def _ctc(log_probs, targets, blank_id):
    return (
        F.ctc_loss(
            log_probs[:, None],
            targets,
            torch.tensor([log_probs.size(0)]),
            torch.tensor([targets.numel()]),
            blank=blank_id,
            reduction='sum',
        )
        / targets.numel()
    )


def _logits_head(vocab_size):
    head = CTCTimestampHead(2 * vocab_size, vocab_size, 0, distillation_weight=0)
    head.forward = lambda values: values.unflatten(-1, (2, vocab_size))
    return head
