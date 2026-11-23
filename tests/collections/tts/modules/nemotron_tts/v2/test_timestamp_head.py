# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional as F

from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import (
    CONTINUE,
    END,
    IGNORE_INDEX,
    OUTSIDE,
    SINGLE,
    START,
    TIMESTAMP_LABELS,
    TimestampHead,
    word_intervals_to_frame_labels,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    'intervals,expected',
    [
        ([], [OUTSIDE] * 4),
        ([(0.01, 0.07)], [SINGLE, OUTSIDE, OUTSIDE, OUTSIDE]),
        ([(0.08, 0.16)], [OUTSIDE, SINGLE, OUTSIDE, OUTSIDE]),
        ([(0.00, 0.16)], [START, END, OUTSIDE, OUTSIDE]),
        ([(0.00, 0.32)], [START, CONTINUE, CONTINUE, END]),
        ([(0.02, 0.20)], [START, CONTINUE, END, OUTSIDE]),
        ([(0.00, 0.08), (0.24, 0.32)], [SINGLE, OUTSIDE, OUTSIDE, SINGLE]),
        ([(0.00, 0.16), (0.16, 0.32)], [START, END, START, END]),
        ([(0.00, 0.10), (0.10, 0.25)], [SINGLE, START, END, OUTSIDE]),
    ],
)
def test_labels_silence_gaps_and_word_extents(intervals, expected):
    result = _labels(intervals)
    assert result.labels.tolist() == expected
    assert result.valid_mask.tolist() == [True] * 4
    assert result.collision_count == result.invalid_word_count == 0
    assert result.collapsed_word_count == 0


def test_exact_decimal_boundaries_do_not_create_spurious_shared_bins():
    result = _labels([(0.56, 0.64), (0.64, 0.72)], num_frames=10)
    assert result.labels.tolist() == [OUTSIDE] * 7 + [SINGLE, SINGLE, OUTSIDE]
    assert result.collision_count == 0


def test_continuous_words_share_one_quantized_boundary_without_losing_start_or_end_labels():
    result = _labels([(0.0, 0.080000000001), (0.080000000001, 0.16)])
    assert result.labels.tolist() == [SINGLE, SINGLE, OUTSIDE, OUTSIDE]
    assert result.collision_count == 0


@pytest.mark.parametrize(
    'interval,midpoint_bin',
    [((0.01, 0.02), 0), ((0.09, 0.10), 1), ((0.25, 0.26), 3), ((0.005, 0.035), 0), ((0.245, 0.275), 3)],
)
def test_isolated_subframe_words_are_single_in_their_original_midpoint_bin(interval, midpoint_bin):
    result = _labels([interval])
    assert result.collapsed_word_count == 1
    assert result.labels[midpoint_bin] == SINGLE
    assert result.valid_mask.all() and result.collision_count == 0
    assert result.labels.eq(OUTSIDE).sum() == 3


def test_collapsed_words_mask_ambiguity_without_shifting_or_stretching_neighbors():
    result = _labels([(0.0, 0.08), (0.08, 0.10), (0.10, 0.24)])
    assert result.labels.tolist() == [SINGLE, IGNORE_INDEX, END, OUTSIDE]
    assert result.collapsed_word_count == 1 and result.collision_count == 1


def test_one_ctc_frame_word_does_not_mask_an_uncontested_neighbor():
    result = _labels([(0.04, 0.08), (0.08, 0.24)])
    assert result.labels.tolist() == [SINGLE, START, END, OUTSIDE]
    assert result.valid_mask.all()
    assert result.collapsed_word_count == 1 and result.collision_count == 0


@pytest.mark.parametrize('intervals', [[(0.0, 0.04), (0.04, 0.08)], [(0.01, 0.02), (0.025, 0.035)]])
def test_two_subframe_words_competing_for_one_bin_are_ignored(intervals):
    result = _labels(intervals)
    assert result.labels.tolist() == [IGNORE_INDEX, OUTSIDE, OUTSIDE, OUTSIDE]
    assert result.valid_mask.tolist() == [False, True, True, True]
    assert result.collision_count == 1


@pytest.mark.parametrize(
    'interval,kwargs,expected',
    [
        ((-0.01, 0.01), {}, [IGNORE_INDEX, OUTSIDE, OUTSIDE, OUTSIDE]),
        ((0.09, 0.11), {'audio_duration': 0.1}, [OUTSIDE, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]),
        ((0.15, 0.17), {'num_frames': 2}, [OUTSIDE, IGNORE_INDEX]),
        ((0.15, 0.17), {'num_frames': 2, 'audio_duration': 1.0}, [OUTSIDE, IGNORE_INDEX]),
        ((0.02, 0.10), {'num_frames': 1, 'audio_duration': 0.07}, [IGNORE_INDEX]),
        ((0.17, 0.19), {'num_frames': 2, 'audio_duration': 1.0}, [OUTSIDE, OUTSIDE]),
    ],
)
def test_partial_words_cannot_acquire_single_labels_at_the_crop(interval, kwargs, expected):
    result = _labels([interval], **kwargs)
    assert result.labels.tolist() == expected
    assert not result.labels.eq(SINGLE).any()


def test_nearest_boundary_quantization_moves_boundaries_by_at_most_half_a_frame():
    result = _labels([(0.039, 0.201), (0.201, 0.319)])
    assert result.labels.tolist() == [START, CONTINUE, END, SINGLE]
    assert result.collision_count == 0


@pytest.mark.parametrize(
    'intervals,expected,collisions',
    [
        ([(0.00, 0.16), (0.08, 0.24)], [START, IGNORE_INDEX, END, OUTSIDE], 1),
        ([(0.00, 0.08), (0.01, 0.07)], [IGNORE_INDEX, OUTSIDE, OUTSIDE, OUTSIDE], 1),
        ([(0.00, 0.08), (0.01, 0.07), (0.02, 0.06)], [IGNORE_INDEX, OUTSIDE, OUTSIDE, OUTSIDE], 1),
        ([(0.00, 0.32), (0.08, 0.24)], [START, IGNORE_INDEX, IGNORE_INDEX, END], 2),
    ],
)
def test_shared_bins_are_masked_without_moving_any_boundary(intervals, expected, collisions):
    result = _labels(intervals)
    assert result.labels.tolist() == expected
    assert result.valid_mask.tolist() == [label != IGNORE_INDEX for label in expected]
    assert result.collision_count == collisions


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
def test_invalid_word_invalidates_whole_turn_instead_of_teaching_false_outside(word):
    result = word_intervals_to_frame_labels([{'start': 0.0, 'end': 0.08}, word], num_frames=4)
    assert result.labels.tolist() == [IGNORE_INDEX] * 4
    assert not result.valid_mask.any()
    assert result.invalid_word_count == 1


def test_cropped_word_keeps_its_original_boundary_classes():
    assert _labels([(0.0, 0.4)], num_frames=2).labels.tolist() == [START, CONTINUE]
    assert _labels([(-0.1, 0.2)], num_frames=2).labels.tolist() == [CONTINUE, CONTINUE]
    assert _labels([(-0.08, 0.16)], num_frames=2).labels.tolist() == [CONTINUE, END]


def test_audio_duration_masks_padding_and_keeps_the_partial_final_bin():
    result = _labels([(0.0, 0.4)], audio_duration=0.10)
    assert result.labels.tolist() == [START, CONTINUE, IGNORE_INDEX, IGNORE_INDEX]
    assert result.valid_mask.tolist() == [True, True, False, False]
    assert _labels([(0.12, 0.14)], audio_duration=0.1).labels.tolist() == [
        OUTSIDE,
        OUTSIDE,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]


def test_empty_audio_and_words_outside_the_crop():
    assert _labels([], num_frames=0).labels.shape == (0,)
    assert _labels([], audio_duration=0).labels.tolist() == [IGNORE_INDEX] * 4
    assert _labels([(-0.2, -0.1), (0.32, 0.4)]).labels.tolist() == [OUTSIDE] * 4


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


def test_head_is_the_requested_small_output_projection_and_preserves_inputs():
    head = TimestampHead(hidden_size=12)
    assert TIMESTAMP_LABELS == ('OUTSIDE', 'START', 'CONTINUE', 'END', 'SINGLE')
    assert isinstance(head.projection[0], torch.nn.LayerNorm)
    assert head.projection[1].out_features == 128
    assert isinstance(head.projection[2], torch.nn.GELU)
    assert head.projection[3].out_features == 5
    hidden = torch.randn(2, 4, 12, requires_grad=True)
    before = hidden.detach().clone()
    logits = head(hidden)
    assert logits.shape == (2, 4, 5)
    assert torch.equal(hidden, before)
    head.compute_loss(logits, torch.tensor([[0, 1, 2, 3], [4, 0, 1, 3]])).backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert head.projection[3].weight.grad.abs().sum() > 0


def test_ce_is_fp32_and_only_uses_unmasked_nonignored_frames():
    head = TimestampHead(hidden_size=4)
    logits = torch.randn(1, 4, 5, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([[START, END, IGNORE_INDEX, OUTSIDE]])
    valid = torch.tensor([[True, False, True, True]])
    expected = F.cross_entropy(logits[0, [0, 3]].float(), labels[0, [0, 3]])
    loss = head.compute_loss(logits, labels, valid)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 1:3].count_nonzero() == 0
    assert logits.grad[0, [0, 3]].abs().sum() > 0


def test_class_weights_use_weighted_mean_normalization():
    weights = [0.25, 2, 1, 3, 4]
    head = TimestampHead(hidden_size=4, class_weights=weights)
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.tensor([[OUTSIDE, START, CONTINUE], [END, SINGLE, IGNORE_INDEX]])
    expected = F.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1), weight=torch.tensor(weights))
    torch.testing.assert_close(head.compute_loss(logits, labels), expected)


@pytest.mark.parametrize('zero_weight', [False, True])
def test_all_masked_or_zero_weight_loss_is_finite_and_differentiable(zero_weight):
    weights = [0, 1, 1, 1, 1] if zero_weight else None
    head = TimestampHead(hidden_size=4, class_weights=weights)
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.full((2, 3), OUTSIDE if zero_weight else IGNORE_INDEX)
    loss = head.compute_loss(logits, labels)
    assert loss.dtype == torch.float32 and loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None and logits.grad.count_nonzero() == 0


def test_masked_nan_logits_and_arbitrary_masked_labels_do_not_contaminate_loss():
    head = TimestampHead(hidden_size=4)
    logits = torch.full((1, 2, 5), float('nan'), requires_grad=True)
    loss = head.compute_loss(logits, torch.full((1, 2), 999), torch.zeros(1, 2, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert logits.grad.count_nonzero() == 0


@pytest.mark.parametrize('weights', [[1, 2], [1, 1, 1, 1, float('nan')], [0] * 5, [-1, 1, 1, 1, 1]])
def test_invalid_class_weights_are_rejected(weights):
    with pytest.raises(ValueError, match='class_weights'):
        TimestampHead(4, class_weights=weights)


def _labels(intervals, **kwargs):
    words = [{'start': start, 'end': end} for start, end in intervals]
    return word_intervals_to_frame_labels(words, **({'num_frames': 4} | kwargs))
