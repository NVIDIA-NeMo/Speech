# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np
import pytest

from nemo.collections.tts.parts.utils.alignment_visualization import _attention_raster, render_alignment_comparison

pytestmark = pytest.mark.unit


def test_mas_and_attention_render_as_bounded_rgb_with_shared_axes():
    import matplotlib.pyplot as plt

    path = np.repeat(np.arange(4), 3)
    probabilities = np.eye(4, dtype=np.float32)[path] * 0.7
    before = plt.get_fignums()
    image = render_alignment_comparison(
        path,
        probabilities,
        token_labels=['one', 'thousand', 'dollars', '.'],
        title='Eval turn: layer 12, head 0',
        frames_per_second=12.5,
    )
    assert image.shape == (600, 1200, 3)
    assert image.dtype == np.uint8
    assert image.std() > 20
    assert plt.get_fignums() == before


@pytest.mark.parametrize('path', [[], [-1, -1, -1], [0, 0, 1]])
def test_missing_attention_and_empty_or_unavailable_mas_are_rendered_explicitly(path):
    image = render_alignment_comparison(path)
    assert image.shape == (600, 1200, 3)
    assert image.dtype == np.uint8


def test_sparse_capture_retains_original_positions_and_full_attention_mass():
    probabilities = np.array([[0.1, 0.2], [0.3, np.nan]], dtype=np.float32)
    actual = _attention_raster(
        probabilities,
        np.array([1, 4]),
        np.array([0, 3]),
        (6, 5),
        6,
        5,
    )
    assert actual[1, 0] == pytest.approx(0.1)
    assert actual[1, 3] == pytest.approx(0.2)
    assert actual[4, 0] == pytest.approx(0.3)
    assert actual.mask[4, 3]
    assert actual.mask[2].all()
    assert actual.mask[:, 1].all()
    assert actual[1].sum() == pytest.approx(0.3), 'The renderer must not renormalize text-only attention.'
    image = render_alignment_comparison(
        [0, 0, 1, 2, 3, 4],
        probabilities,
        [1, 4],
        attention_token_positions=[0, 3],
        token_labels=['a', 'b', 'c', 'd', 'e'],
    )
    assert image.shape == (600, 1200, 3)


@pytest.mark.parametrize('limits,expected', [((64, 32), (64, 32)), ((100_000, 10_000), (512, 256))])
def test_long_mas_path_never_allocates_the_full_frame_by_token_matrix(limits, expected):
    path = np.arange(100_000, dtype=np.int64) // 10
    captured = []
    original = np.zeros

    def observe_shape(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2:
            captured.append(shape)
        return original(shape, *args, **kwargs)

    with patch('nemo.collections.tts.parts.utils.alignment_visualization.np.zeros', side_effect=observe_shape):
        image = render_alignment_comparison(path, max_frames=limits[0], max_tokens=limits[1])
    assert image.shape == (600, 1200, 3)
    assert expected in captured
    assert (100_000, 10_000) not in captured


def test_figure_is_cleared_when_rendering_fails():
    from matplotlib.figure import Figure

    cleared = []
    original = Figure.clear

    def observe_clear(figure, *args, **kwargs):
        cleared.append(figure)
        return original(figure, *args, **kwargs)

    with (
        patch.object(Figure, 'clear', observe_clear),
        patch('matplotlib.backends.backend_agg.FigureCanvasAgg.draw', side_effect=RuntimeError('render failure')),
        pytest.raises(RuntimeError, match='render failure'),
    ):
        render_alignment_comparison([0, 0, 1])
    assert cleared and not cleared[-1].axes


@pytest.mark.parametrize(
    'kwargs',
    [
        {'mas_token_indices': [[0]]},
        {'mas_token_indices': [0.5]},
        {'mas_token_indices': [-2]},
        {'mas_token_indices': [0], 'attention_probs': [[1.1]]},
        {'mas_token_indices': [0], 'attention_probs': [[-0.1]]},
        {'mas_token_indices': [0], 'attention_probs': [[0.2]], 'attention_frame_indices': [0, 1]},
        {'mas_token_indices': [0], 'attention_probs': [[0.2, 0.3]], 'attention_token_positions': [1, 1]},
        {'mas_token_indices': [0], 'attention_probs': [[0.2]], 'attention_frame_indices': [-1]},
        {'mas_token_indices': [0], 'frames_per_second': 0},
        {'mas_token_indices': [0], 'max_frames': False},
        {'mas_token_indices': [0], 'max_tokens': 0},
    ],
)
def test_invalid_display_inputs_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        render_alignment_comparison(**kwargs)
