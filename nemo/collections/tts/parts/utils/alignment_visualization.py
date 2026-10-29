# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded validation images comparing a MAS path with a decoder attention head."""

import math

import numpy as np


def render_alignment_comparison(
    mas_token_indices,
    attention_probs=None,
    attention_frame_indices=None,
    *,
    token_labels=None,
    title='',
    frames_per_second=None,
    attention_token_positions=None,
    max_frames=512,
    max_tokens=256,
):
    """Return a 1200x600 RGB image without allocating a full audio-by-text matrix.

    The MAS path has one token position per turn-local audio frame; ``-1`` marks
    an unavailable alignment. Attention is ``[captured_frames, captured_tokens]``.
    Its optional position vectors place a sparse capture on the same full-turn
    axes as MAS. Probabilities retain their original full-key normalization.

    Long sequences are maximum-binned for display only, at most 512 frames by
    256 tokens; the size arguments may reduce these limits. Uncaptured cells are
    gray, rather than zero attention. Passing no attention shows a labeled
    placeholder. Inputs should already be detached CPU arrays.
    """
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    for name, value in (('max_frames', max_frames), ('max_tokens', max_tokens)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    max_frames = min(max_frames, 512)
    max_tokens = min(max_tokens, 256)
    if frames_per_second is not None and (
        isinstance(frames_per_second, bool)
        or not isinstance(frames_per_second, (int, float))
        or not math.isfinite(frames_per_second)
        or frames_per_second <= 0
    ):
        raise ValueError('frames_per_second must be finite and positive')

    path = np.asarray(mas_token_indices)
    if path.ndim != 1 or (path.size and path.dtype.kind not in 'iu') or np.any(path < -1):
        raise ValueError('mas_token_indices must be a one-dimensional integer path with -1 for unavailable frames')
    labels = None if token_labels is None else list(token_labels)
    num_frames = path.size
    num_tokens = max(int(path.max(initial=-1)) + 1, 0 if labels is None else len(labels))
    probabilities = None
    if attention_probs is not None:
        probabilities = np.asarray(attention_probs)
        if probabilities.ndim != 2 or probabilities.dtype.kind not in 'fiu':
            raise ValueError('attention_probs must be a numeric frame-by-token matrix')
        finite = probabilities[np.isfinite(probabilities)]
        if np.any((finite < 0) | (finite > 1 + 1e-6)):
            raise ValueError('attention_probs must contain probabilities in [0, 1] or non-finite masked values')
        frames = _positions(attention_frame_indices, probabilities.shape[0], 'attention_frame_indices')
        tokens = _positions(attention_token_positions, probabilities.shape[1], 'attention_token_positions')
        num_frames = max(num_frames, int(frames.max(initial=-1)) + 1)
        num_tokens = max(num_tokens, int(tokens.max(initial=-1)) + 1)
    elif attention_frame_indices is not None or attention_token_positions is not None:
        raise ValueError('Attention positions require attention_probs')

    frame_extent, token_extent = max(num_frames, 1), max(num_tokens, 1)
    raster_shape = (min(frame_extent, max_frames), min(token_extent, max_tokens))
    mas_image = np.zeros(raster_shape, dtype=np.float32)
    mas_rows = np.zeros(raster_shape[0], dtype=bool)
    valid = path >= 0
    if valid.any():
        rows = _bin_positions(np.flatnonzero(valid), frame_extent, raster_shape[0])
        columns = _bin_positions(path[valid], token_extent, raster_shape[1])
        mas_image[rows, columns] = 1.0
        mas_rows[rows] = True
    mas_image = np.ma.array(mas_image, mask=np.broadcast_to(~mas_rows[:, None], raster_shape))

    attention_image = None
    if probabilities is not None:
        attention_image = _attention_raster(probabilities, frames, tokens, raster_shape, frame_extent, token_extent)

    fig = Figure(figsize=(12, 6), dpi=100, layout='constrained')
    canvas = FigureCanvasAgg(fig)
    try:
        axes = fig.subplots(1, 2, sharex=True, sharey=True)
        if title:
            fig.suptitle(str(title)[:300], fontsize=11, wrap=True)
        colormap = colormaps['viridis'].copy()
        colormap.set_bad('#d9d9d9')
        scale = 1.0 if frames_per_second is None else 1.0 / frames_per_second
        extent = (0, token_extent, 0, frame_extent * scale)
        panels = (('MAS hard alignment', mas_image), ('Guided-head attention (before prior)', attention_image))
        for axis, (panel_title, values) in zip(axes, panels):
            if values is None:
                values = np.ma.masked_all(raster_shape, dtype=np.float32)
            artist = axis.imshow(
                values,
                origin='lower',
                aspect='auto',
                interpolation='nearest',
                cmap=colormap,
                vmin=0.0,
                vmax=1.0,
                extent=extent,
            )
            axis.set_title(panel_title, fontsize=10)
            axis.set_xlabel('Text BPE token position')
            if num_tokens:
                ticks = np.unique(np.linspace(0, num_tokens - 1, min(12, num_tokens), dtype=np.int64))
                axis.set_xticks(ticks + 0.5)
                tick_labels = [str(index) for index in ticks]
                if labels is not None:
                    tick_labels = [
                        f'{index}: {_short_label(labels[index])}' if index < len(labels) else str(index)
                        for index in ticks
                    ]
                axis.set_xticklabels(
                    tick_labels, rotation=55 if labels is not None else 0, ha='right', fontsize=7, parse_math=False
                )
            fig.colorbar(artist, ax=axis, fraction=0.045, pad=0.025)
        axes[0].set_ylabel('Audio frame in turn' if frames_per_second is None else 'Audio time in turn (s)')
        if not valid.any():
            axes[0].text(0.5, 0.5, 'No valid MAS path', ha='center', va='center', transform=axes[0].transAxes)
        if attention_image is None:
            axes[1].text(0.5, 0.5, 'No guided attention head', ha='center', va='center', transform=axes[1].transAxes)
        elif not np.isfinite(probabilities).any():
            axes[1].text(0.5, 0.5, 'No captured attention', ha='center', va='center', transform=axes[1].transAxes)
        footer = 'Gray: uncaptured / unavailable. Long sequences are max-binned for display.'
        fig.supxlabel(footer, fontsize=8)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()
    finally:
        # Agg figures are never registered with pyplot's global figure manager.
        fig.clear()


def _positions(values, size, name):
    positions = np.arange(size, dtype=np.int64) if values is None else np.asarray(values)
    if (
        positions.ndim != 1
        or positions.size != size
        or (positions.size and positions.dtype.kind not in 'iu')
        or np.any(positions < 0)
        or np.any(positions[1:] <= positions[:-1])
    ):
        raise ValueError(f'{name} must contain one strictly increasing nonnegative integer position per matrix entry')
    return positions


def _bin_positions(positions, extent, bins):
    return np.minimum(positions.astype(np.int64) * bins // extent, bins - 1)


def _attention_raster(probabilities, frames, tokens, shape, frame_extent, token_extent):
    values = np.zeros(shape, dtype=np.float32)
    missing = np.ones(shape, dtype=bool)
    if not frames.size or not tokens.size:
        return np.ma.array(values, mask=missing)
    rows = _bin_positions(frames, frame_extent, shape[0])
    columns = _bin_positions(tokens, token_extent, shape[1])
    starts = np.r_[0, np.flatnonzero(np.diff(columns)) + 1]
    raster_columns = columns[starts]
    for row, samples in zip(rows, probabilities):
        binned = np.maximum.reduceat(np.where(np.isfinite(samples), samples, -np.inf), starts)
        valid = np.isfinite(binned)
        selected = raster_columns[valid]
        values[row, selected] = np.maximum(values[row, selected], binned[valid])
        missing[row, selected] = False
    return np.ma.array(values, mask=missing)


def _short_label(value):
    text = str(value).replace('\n', r'\n').replace('\t', r'\t')
    return text if len(text) <= 18 else text[:17] + '…'
