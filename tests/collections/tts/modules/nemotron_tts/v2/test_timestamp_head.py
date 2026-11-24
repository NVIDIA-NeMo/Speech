# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch.nn import functional as F

from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import (
    END,
    IGNORE_INDEX,
    NONEND,
    TIMESTAMP_LABELS,
    TimestampHead,
    reconcile_timestamp_eos,
    word_intervals_to_frame_labels,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('aligned', [[], [2], [3], [4], [5], [6]])
def test_eos_reconciliation_repairs_shifted_terminals_without_new_off_eos_events(aligned):
    labels = torch.tensor([[NONEND, END, NONEND, NONEND, NONEND, IGNORE_INDEX, IGNORE_INDEX]])
    valid = labels != IGNORE_INDEX
    terminal = torch.zeros_like(valid)
    terminal[:, aligned] = True
    eos = torch.zeros_like(valid)
    eos[:, 4] = True
    audio = ~eos
    eligible = torch.ones_like(valid)
    originals = [tensor.clone() for tensor in (labels, valid, terminal, eos, audio, eligible)]

    targets, mask, stats = reconcile_timestamp_eos(labels, valid, terminal, eos, audio, eligible)

    assert targets.tolist() == [[NONEND, END, NONEND, NONEND, END, IGNORE_INDEX, IGNORE_INDEX]]
    assert mask.tolist() == [[True, True, True, True, True, False, False]]
    assert stats['eos_alignment_total_count'] == 1
    assert stats['eos_alignment_missing_count'] == int(aligned != [4])
    assert stats['eos_alignment_extra_count'] == int(bool(aligned) and aligned != [4])
    for original, observed in zip(originals, (labels, valid, terminal, eos, audio, eligible)):
        torch.testing.assert_close(observed, original, rtol=0, atol=0)


def test_eos_reconciliation_counts_missing_and_extra_endpoints_and_excludes_ineligible_steps():
    labels = torch.full((2, 7), NONEND, dtype=torch.long)
    valid = torch.ones_like(labels, dtype=torch.bool)
    terminal = torch.tensor([[False, False, True, False, False, True, False], [False] * 7])
    eos = torch.tensor(
        [[False, False, True, False, True, False, True], [False, True, False, False, False, False, False]]
    )
    eligible = torch.tensor([[True, True, True, True, True, True, False], [False] * 7])
    audio = torch.ones_like(valid)
    audio[:, 0] = False  # Non-EOS control frames never gain supervision.
    targets, mask, stats = reconcile_timestamp_eos(labels, valid, terminal, eos, audio, eligible)
    assert targets.tolist() == [[IGNORE_INDEX, NONEND, END, NONEND, END, NONEND, IGNORE_INDEX], [IGNORE_INDEX] * 7]
    assert mask.tolist() == [[False, True, True, True, True, True, False], [False] * 7]
    assert {name: value.item() for name, value in stats.items()} == {
        'eos_alignment_total_count': 2,
        'eos_alignment_missing_count': 1,
        'eos_alignment_extra_count': 1,
    }


def test_eos_reconciliation_recovers_a_terminal_when_the_teacher_has_no_usable_frames():
    labels = torch.full((1, 4), IGNORE_INDEX, dtype=torch.long)
    absent = torch.zeros_like(labels, dtype=torch.bool)
    eos = torch.tensor([[False, False, False, True]])
    targets, mask, stats = reconcile_timestamp_eos(labels, absent, absent, eos, ~eos, ~absent)
    assert targets.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, END]]
    assert mask.tolist() == [[False, False, False, True]]
    assert stats['eos_alignment_missing_count'] == stats['eos_alignment_total_count'] == 1


def test_eos_reconciliation_leaves_no_eos_batches_without_a_terminal_event():
    labels = torch.tensor([[NONEND, END, NONEND]])
    valid = torch.ones_like(labels, dtype=torch.bool)
    absent = torch.zeros_like(valid)
    targets, mask, stats = reconcile_timestamp_eos(labels, valid, absent, absent, valid, valid)
    torch.testing.assert_close(targets, labels)
    torch.testing.assert_close(mask, valid)
    assert not any(value.item() for value in stats.values())


def test_eos_reconciliation_keeps_all_masked_loss_connected_and_zero():
    labels = torch.full((2, 3), IGNORE_INDEX, dtype=torch.long)
    absent = torch.zeros_like(labels, dtype=torch.bool)
    targets, mask, stats = reconcile_timestamp_eos(labels, absent, ~absent, ~absent, absent, absent)
    assert not mask.any() and not any(value.item() for value in stats.values())
    logits = torch.randn(2, 3, requires_grad=True)
    loss = TimestampHead(4).compute_loss(logits, targets, mask)
    loss.backward()
    assert loss.item() == 0 and not logits.grad.any()


@pytest.mark.parametrize(
    'intervals,expected',
    [
        ([], [NONEND] * 4),
        ([(0.08, 0.32)], [NONEND] * 4),
        ([(0.08, 0.16), (0.16, 0.32)], [NONEND, END, NONEND, NONEND]),
        ([(0.0, 0.08), (0.08, 0.24), (0.24, 0.32)], [END, NONEND, END, NONEND]),
        ([(-0.08, 0.16), (0.16, 0.4)], [NONEND, END, NONEND, NONEND]),
        ([(0.0, 0.4), (0.4, 0.5)], [NONEND] * 4),
    ],
)
def test_only_internal_word_ends_are_positive_and_initial_silence_has_no_event(intervals, expected):
    result = _labels(intervals)
    assert result.labels.tolist() == expected
    assert result.valid_mask.tolist() == [True] * 4
    assert result.collision_count == result.collision_word_count == result.invalid_word_count == 0


@pytest.mark.parametrize('duration', [0.001, 0.01, 0.04, 0.08, 0.4, 4.0])
def test_single_word_has_no_internal_end_and_leaves_terminal_supervision_to_the_caller(duration):
    result = _labels([(0.09, 0.09 + duration)])
    assert result.labels.tolist() == [NONEND] * 4
    assert result.valid_mask.all()


@pytest.mark.parametrize(
    'end,frame',
    [(0.001, 0), (0.039, 0), (0.04, 0), (0.08, 0), (0.119, 0), (0.12, 1), (0.16, 1), (0.24, 2)],
)
def test_internal_ends_label_the_preceding_frame_with_forward_rounding_ties(end, frame):
    result = _labels([(0.0, end), (end, 0.32)])
    assert result.labels.nonzero().flatten().tolist() == [frame]
    if end in (0.08, 0.16, 0.24):
        assert (frame + 1) * 0.08 == end


def test_decimal_roundoff_does_not_shift_an_end_or_create_spurious_collisions():
    result = _labels([(0.0, 0.56), (0.56, 0.64), (0.64, 0.72)], num_frames=10)
    assert result.labels.tolist() == [NONEND] * 6 + [END, END, NONEND, NONEND]
    assert result.collision_count == 0
    result = _labels([(0.0, 0.15), (0.15, 0.3)], frame_seconds=0.1)
    assert result.labels.tolist() == [NONEND, END, NONEND, NONEND]


@pytest.mark.parametrize(
    'intervals,expected,collision_bins,extra_ends',
    [
        ([(0.0, 0.01), (0.01, 0.02), (0.02, 0.32)], [END, NONEND, NONEND, NONEND], 1, 1),
        ([(0.0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.32)], [END, NONEND, NONEND, NONEND], 1, 2),
        (
            [(0.0, 0.01), (0.01, 0.02), (0.02, 0.17), (0.17, 0.18), (0.18, 0.32)],
            [END, END, NONEND, NONEND],
            2,
            2,
        ),
    ],
)
def test_multiple_internal_ends_share_a_valid_end_bin_without_shifting(
    intervals, expected, collision_bins, extra_ends
):
    result = _labels(intervals)
    assert result.labels.tolist() == expected
    assert result.valid_mask.all()
    assert result.collision_count == collision_bins
    assert result.collision_word_count == extra_ends


def test_last_partial_audio_bin_retains_an_internal_end_that_rounds_to_its_right_edge():
    result = _labels([(0.0, 0.29), (0.29, 0.5)], audio_duration=0.30)
    assert result.labels.tolist() == [NONEND, NONEND, NONEND, END]
    assert result.valid_mask.all()


def test_padding_crop_and_audio_eos_do_not_invent_end_events():
    result = _labels([(0.0, 0.15), (0.15, 0.16), (0.16, 0.4)], audio_duration=0.16)
    assert result.labels.tolist() == [NONEND, END, IGNORE_INDEX, IGNORE_INDEX]
    assert result.valid_mask.tolist() == [True, True, False, False]
    result = _labels([(0.0, 0.16), (0.16, 0.5)], num_frames=2, audio_duration=1.0)
    assert result.labels.tolist() == [NONEND, END]
    result = _labels([(0.0, 0.16), (0.16, 0.3)], audio_duration=0.16)
    assert result.labels.tolist() == [NONEND, NONEND, IGNORE_INDEX, IGNORE_INDEX]
    assert _labels([], num_frames=0).labels.shape == (0,)
    assert _labels([], audio_duration=0).labels.tolist() == [IGNORE_INDEX] * 4


def test_the_last_word_end_is_reserved_for_caller_terminal_supervision():
    result = _labels([(0.0, 0.08), (0.08, 0.16)], audio_duration=1.0)
    assert result.labels.tolist() == [END, NONEND, NONEND, NONEND]


@pytest.mark.parametrize(
    'word',
    [
        {},
        {'start': None, 'end': 0.1},
        {'start': 'invalid', 'end': 0.1},
        {'start': float('nan'), 'end': 0.1},
        {'start': 0.0, 'end': float('inf')},
        {'start': 0.1, 'end': 0.1},
        {'start': 0.2, 'end': 0.1},
    ],
)
def test_invalid_word_masks_the_whole_turn_instead_of_teaching_false_nonends(word):
    result = word_intervals_to_frame_labels([{'start': 0.0, 'end': 0.08}, word], num_frames=4)
    assert result.labels.tolist() == [IGNORE_INDEX] * 4
    assert not result.valid_mask.any()
    assert result.invalid_word_count == 1


@pytest.mark.parametrize(
    'kwargs',
    [
        {'num_frames': -1},
        {'num_frames': 1.5},
        {'num_frames': True},
        {'frame_seconds': 0},
        {'frame_seconds': float('nan')},
        {'audio_duration': -0.1},
        {'audio_duration': float('inf')},
    ],
)
def test_invalid_frame_grid_is_rejected(kwargs):
    with pytest.raises(ValueError):
        word_intervals_to_frame_labels([], **({'num_frames': 4} | kwargs))


def test_head_has_one_logit_per_frame_and_preserves_its_input():
    head = TimestampHead(hidden_size=12)
    assert TIMESTAMP_LABELS == ('NONEND', 'END')
    assert isinstance(head.projection[0], torch.nn.LayerNorm)
    assert head.projection[1].out_features == 128
    assert isinstance(head.projection[2], torch.nn.GELU)
    assert head.projection[3].out_features == 1
    hidden = torch.randn(2, 4, 12, requires_grad=True)
    before = hidden.detach().clone()
    logits = head(hidden)
    assert logits.shape == (2, 4)
    assert torch.equal(hidden, before)
    head.compute_loss(logits, torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0]])).backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert head.projection[3].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('pos_weight', [1.0, 3.0])
def test_zero_logit_bce_matches_analytic_loss_and_gradient(pos_weight):
    head = TimestampHead(4, pos_weight=pos_weight)
    logits = torch.zeros(2, requires_grad=True)
    loss = head.compute_loss(logits, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(loss, torch.tensor(math.log(2) * (1 + pos_weight) / 2))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([0.25, -pos_weight / 4]))


def test_bce_is_fp32_and_only_uses_unmasked_nonignored_frames():
    head = TimestampHead(4, pos_weight=2.5)
    logits = torch.tensor([[0.5, float('nan'), float('nan'), -0.5]], dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([[END, NONEND, IGNORE_INDEX, NONEND]])
    valid = torch.tensor([[True, False, True, True]])
    expected = F.binary_cross_entropy_with_logits(
        logits[0, [0, 3]].float(), labels[0, [0, 3]].float(), pos_weight=torch.tensor(2.5)
    )
    loss = head.compute_loss(logits, labels, valid)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 1:3].count_nonzero() == 0
    assert logits.grad[0, [0, 3]].abs().sum() > 0


@pytest.mark.parametrize('selection', ['ignore_index', 'mask', 'empty'])
def test_all_masked_loss_is_finite_and_differentiable_even_with_nan_logits(selection):
    head = TimestampHead(4)
    logits = torch.full((0,) if selection == 'empty' else (2, 3), float('nan'), requires_grad=True)
    labels = torch.full_like(logits, IGNORE_INDEX if selection == 'ignore_index' else 999)
    mask = torch.zeros_like(labels, dtype=torch.bool) if selection == 'mask' else None
    loss = head.compute_loss(logits, labels, mask)
    assert loss.dtype == torch.float32 and loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None and logits.grad.count_nonzero() == 0


def test_loss_dtype_is_fp32_even_when_global_default_is_float64():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        head = TimestampHead(4)
        assert head.pos_weight.dtype == torch.float32
        loss = head.compute_loss(torch.zeros(2, requires_grad=True), torch.tensor([0.0, 1.0]))
        assert loss.dtype == torch.float32
        torch.testing.assert_close(loss, torch.tensor(math.log(2), dtype=torch.float32))
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize('pos_weight', [0, -1, float('inf'), float('nan'), [1], True])
def test_invalid_positive_weights_are_rejected(pos_weight):
    with pytest.raises(ValueError, match='pos_weight'):
        TimestampHead(4, pos_weight=pos_weight)


def test_loss_rejects_mismatched_shapes_and_nonboolean_masks():
    head = TimestampHead(4)
    with pytest.raises(ValueError, match='same shape'):
        head.compute_loss(torch.zeros(2, 1), torch.zeros(2))
    with pytest.raises(ValueError, match='valid_mask'):
        head.compute_loss(torch.zeros(2), torch.zeros(2), torch.ones(2))


def _labels(intervals, **kwargs):
    words = [{'start': start, 'end': end} for start, end in intervals]
    return word_intervals_to_frame_labels(words, **({'num_frames': 4} | kwargs))
