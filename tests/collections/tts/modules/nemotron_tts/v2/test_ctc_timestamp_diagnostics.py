# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io

import numpy as np
import pytest
import torch

from nemo.collections.tts.modules.nemotron_tts.v2.timestamp_head import (
    IGNORE_INDEX,
    decode_ctc_tokens,
    plot_ctc_timestamp_alignment,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('tensor', [False, True])
def test_greedy_ctc_collapses_before_removing_blank_and_resets_at_ignored_frames(tensor):
    tokens = [1, 1, 5, 1, 2, 2, IGNORE_INDEX, 2, 0, 0]
    if tensor:
        tokens = torch.tensor(tokens)
    assert decode_ctc_tokens(tokens, 5, _Tokenizer()) == '1 1 2 2 0'


def test_greedy_ctc_masked_gaps_break_runs_without_adding_invalid_predictions():
    tokens = torch.tensor([1, 1, 2, 1, 5, 1, 2, 2])
    valid = torch.tensor([True, False, False, True, True, True, True, True])
    assert decode_ctc_tokens(tokens, 5, _Tokenizer(), valid) == '1 1 1 2'
    assert decode_ctc_tokens(torch.full((4,), 5), 5, _Tokenizer()) == ''
    assert decode_ctc_tokens([], 5, _Tokenizer()) == ''
    assert decode_ctc_tokens(tokens, 5, _Tokenizer(), torch.zeros(8, dtype=torch.bool)) == ''


@pytest.mark.parametrize(
    'tokens,mask',
    [
        (torch.zeros(2, 2, dtype=torch.long), None),
        (torch.zeros(2, dtype=torch.long), torch.ones(2, 1, dtype=torch.bool)),
        (torch.zeros(2, dtype=torch.long), torch.ones(2)),
        ([1, 2], [True]),
    ],
)
def test_greedy_ctc_rejects_incompatible_frame_shapes(tokens, mask):
    with pytest.raises(ValueError):
        decode_ctc_tokens(tokens, 5, _Tokenizer(), mask)


def test_ctc_comparison_plots_token_identity_confidence_and_preserves_blank_and_mask():
    import matplotlib.pyplot as plt

    diagnostic = _diagnostic()
    original = {key: value.clone() for key, value in diagnostic.items() if isinstance(value, torch.Tensor)}
    figure = plot_ctc_timestamp_alignment(diagnostic, _Tokenizer())
    try:
        teacher, student, confidence = figure.axes[:3]
        labels = [tick.get_text() for tick in teacher.get_yticklabels()]
        assert labels == ['<blank>', '1: one', '2: two', '4: extra']
        # Repeated reference occurrence 1 shares the same vocabulary row.
        assert sum(label.startswith('1:') for label in labels) == 1
        teacher_matrix = teacher.collections[0].get_array().reshape(4, 4)
        student_matrix = student.collections[0].get_array().reshape(4, 4)
        assert teacher_matrix[1, 0] == pytest.approx(0.8)
        assert teacher_matrix[2, 2] == pytest.approx(0.6)
        assert teacher_matrix[1, 3] == pytest.approx(0.7)
        assert np.ma.getmaskarray(teacher_matrix)[:, 1].all()
        assert student_matrix[3, 2] == pytest.approx(0.9)
        assert np.ma.getmaskarray(student_matrix)[:, 1].all()
        assert teacher.get_xlim() == pytest.approx((0, 0.16))
        assert confidence.lines[2].get_label() == 'Teacher P(blank)'
        assert confidence.lines[2].get_ydata()[2] == pytest.approx(0.2)
        assert any('not forced alignments' in text.get_text() for text in figure.texts)
        assert any('TTS input:' in text.get_text() for text in figure.texts)
        stream = io.BytesIO()
        figure.savefig(stream, format='png')
        assert stream.getvalue().startswith(b'\x89PNG\r\n\x1a\n')
    finally:
        plt.close(figure)
    for key, value in original.items():
        torch.testing.assert_close(diagnostic[key], value, rtol=0, atol=0)


def test_ctc_comparison_display_limits_preserve_diagnostic_and_expose_undisplayed_tokens():
    import matplotlib.pyplot as plt

    diagnostic = _diagnostic()
    figure = plot_ctc_timestamp_alignment(diagnostic, _Tokenizer(), max_frames=2, max_tokens=1)
    try:
        labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        assert labels == ['<blank>', '1: one', 'other / undisplayed IDs']
        assert figure.axes[0].collections[0].get_array().size == 6
        assert any('2/4 frames' in text.get_text() for text in figure.texts)
        assert diagnostic['student_token_ids'].numel() == 4
    finally:
        plt.close(figure)


@pytest.mark.parametrize('empty', [False, True])
def test_ctc_comparison_keeps_real_time_intervals_when_frames_have_gaps_or_are_empty(empty):
    import matplotlib.pyplot as plt

    diagnostic = _diagnostic()
    if empty:
        for key, value in diagnostic.items():
            if isinstance(value, torch.Tensor):
                diagnostic[key] = value[:0]
    else:
        diagnostic['frame_start_seconds'][2:] += 0.08
        diagnostic['frame_end_seconds'][2:] += 0.08
    figure = plot_ctc_timestamp_alignment(diagnostic, _Tokenizer())
    try:
        if empty:
            assert not figure.axes[0].collections
            assert any('No CTC frames' in text.get_text() for text in figure.axes[0].texts)
        else:
            assert len(figure.axes[0].collections) == 2
            assert figure.axes[0].get_xlim() == pytest.approx((0, 0.24))
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    'changes',
    [
        {'valid_mask': torch.ones(3, dtype=torch.bool)},
        {'teacher_probabilities': torch.ones(4, 1)},
        {'teacher_indices': torch.empty(4, 0, dtype=torch.long), 'teacher_probabilities': torch.empty(4, 0)},
        {'frame_start_seconds': torch.tensor([0.0, 0.04, float('nan'), 0.12])},
        {'frame_end_seconds': torch.tensor([0.04, 0.03, 0.12, 0.16])},
        {'frame_start_seconds': torch.tensor([0.0, 0.02, 0.08, 0.12])},
    ],
)
def test_ctc_comparison_rejects_incompatible_shapes_or_time_grid(changes):
    with pytest.raises(ValueError):
        plot_ctc_timestamp_alignment({**_diagnostic(), **changes}, _Tokenizer())


@pytest.mark.parametrize('options', [{'max_frames': 0}, {'max_frames': 1.5}, {'max_tokens': False}])
def test_ctc_comparison_rejects_invalid_display_limits(options):
    with pytest.raises(ValueError):
        plot_ctc_timestamp_alignment(_diagnostic(), _Tokenizer(), **options)


class _Tokenizer:
    def ids_to_text(self, ids):
        return ' '.join(str(token) for token in ids)

    def ids_to_tokens(self, ids):
        pieces = {0: 'zero', 1: 'one', 2: 'two', 3: '$10', 4: 'extra'}
        return [pieces[token] for token in ids]


def _diagnostic():
    return dict(
        reference_text='one two one',
        reference_token_ids=[1, 2, 1],
        blank_id=5,
        frame_start_seconds=torch.tensor([0.0, 0.04, 0.08, 0.12], dtype=torch.float64),
        frame_end_seconds=torch.tensor([0.04, 0.08, 0.12, 0.16], dtype=torch.float64),
        valid_mask=torch.tensor([True, False, True, True]),
        teacher_indices=torch.tensor([[1, 5], [5, 2], [2, 5], [1, 5]]),
        teacher_probabilities=torch.tensor([[0.8, 0.1], [0.7, 0.2], [0.6, 0.2], [0.7, 0.1]]),
        teacher_tail_probabilities=torch.tensor([0.1, 0.1, 0.2, 0.2]),
        student_token_ids=torch.tensor([1, 5, 4, 1]),
        student_probabilities=torch.tensor([0.6, 0.8, 0.9, 0.7]),
    )
