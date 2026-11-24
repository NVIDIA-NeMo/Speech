# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched CUDA CTC Viterbi search for frozen timestamp alignment.

Each CUDA program keeps one utterance's score frontier in registers, advances
through its valid frames, then follows byte backpointers. Scores use FP32 and
the reference's strict stay/step/skip tie order. Optional band starts preserve
the existing CPU anchor-guided search and its exact state-space restriction.
"""

from __future__ import annotations

import math

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


MAX_SUPPORTED_STATES = 4096


if triton is not None:

    @triton.jit(
        do_not_specialize=[
            'LOG_BATCH_STRIDE',
            'LOG_FRAME_STRIDE',
            'LABEL_BATCH_STRIDE',
            'MAX_FRAMES',
            'MAX_STATES',
            'MAX_WIDTH',
            'COLUMNS',
        ]
    )
    def _ctc_viterbi_kernel(
        LogProbs,
        Labels,
        FrameLengths,
        StateLengths,
        BandStarts,
        BandWidths,
        Backpointers,
        Paths,
        Scores,
        LOG_BATCH_STRIDE,
        LOG_FRAME_STRIDE,
        LABEL_BATCH_STRIDE,
        MAX_FRAMES,
        MAX_STATES,
        MAX_WIDTH,
        COLUMNS,
        BANDED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(0).to(tl.int64)
        frames = tl.load(FrameLengths + batch)
        states = tl.load(StateLengths + batch)
        width = tl.load(BandWidths + batch) if BANDED else states
        offsets = tl.arange(0, BLOCK)
        active = (
            (frames > 0)
            & (frames <= MAX_FRAMES)
            & (states > 0)
            & (states <= MAX_STATES)
            & (width > 0)
            & (width <= MAX_WIDTH)
            & (width <= states)
        )
        if BANDED:
            active &= tl.load(BandStarts + batch * MAX_FRAMES) == 0
        if active:
            valid = offsets < width
            targets = tl.load(Labels + batch * LABEL_BATCH_STRIDE + offsets, valid, other=0)
            previous_targets = tl.load(
                Labels + batch * LABEL_BATCH_STRIDE + offsets - 2,
                valid & (offsets >= 2),
                other=0,
            )
            can_skip = valid & (offsets >= 2) & (targets != 0) & (targets != previous_targets)
            valid_columns = (targets >= 0) & (targets < COLUMNS)
            previous = tl.load(
                LogProbs + batch * LOG_BATCH_STRIDE + targets,
                valid & (offsets < 2) & valid_columns,
                other=-float('inf'),
            ).to(tl.float32)
            previous_start = tl.full((), 0, tl.int32)
            for frame in range(1, frames):
                start = tl.load(BandStarts + batch * MAX_FRAMES + frame).to(tl.int32) if BANDED else 0
                delta = start - previous_start
                state_offsets = start + offsets
                valid_band = (start >= 0) & (start + width <= states) & (delta >= 0) & (delta <= 2)
                if BANDED:
                    valid_states = valid & (state_offsets >= 0) & (state_offsets < states)
                    targets = tl.load(Labels + batch * LABEL_BATCH_STRIDE + state_offsets, valid_states, other=0)
                    previous_targets = tl.load(
                        Labels + batch * LABEL_BATCH_STRIDE + state_offsets - 2,
                        valid_states & (state_offsets >= 2),
                        other=0,
                    )
                    can_skip = valid & (state_offsets >= 2) & (targets != 0) & (targets != previous_targets)
                    valid_columns = (targets >= 0) & (targets < COLUMNS)
                stay_index, one_index, two_index = offsets + delta, offsets + delta - 1, offsets + delta - 2
                best = tl.gather(previous, tl.minimum(tl.maximum(stay_index, 0), BLOCK - 1), axis=0)
                one = tl.gather(previous, tl.minimum(tl.maximum(one_index, 0), BLOCK - 1), axis=0)
                two = tl.gather(previous, tl.minimum(tl.maximum(two_index, 0), BLOCK - 1), axis=0)
                best = tl.where((stay_index >= 0) & (stay_index < width), best, -float('inf'))
                one = tl.where((one_index >= 0) & (one_index < width), one, -float('inf'))
                two = tl.where((two_index >= 0) & (two_index < width) & can_skip, two, -float('inf'))
                take_one = one > best
                best = tl.where(take_one, one, best)
                move = tl.where(take_one, 1, 0)
                take_two = two > best
                best = tl.where(take_two, two, best)
                move = tl.where(take_two, 2, move)
                emission = tl.load(
                    LogProbs + batch * LOG_BATCH_STRIDE + frame * LOG_FRAME_STRIDE + targets,
                    valid & valid_columns & valid_band,
                    other=-float('inf'),
                ).to(tl.float32)
                previous = best + emission
                tl.store(
                    Backpointers + (batch * MAX_FRAMES + frame) * MAX_WIDTH + offsets,
                    move.to(tl.uint8),
                    valid,
                )
                previous_start = start

            last_offset, penultimate_offset = states - 1 - previous_start, states - 2 - previous_start
            last = tl.sum(tl.where(offsets == last_offset, previous, 0.0), axis=0)
            last = tl.where((last_offset >= 0) & (last_offset < width), last, -float('inf'))
            penultimate = tl.sum(tl.where(offsets == penultimate_offset, previous, 0.0), axis=0)
            penultimate = tl.where(
                (states > 1) & (penultimate_offset >= 0) & (penultimate_offset < width), penultimate, -float('inf')
            )
            finish_penultimate = penultimate > last
            score = tl.where(finish_penultimate, penultimate, last)
            tl.store(Scores + batch, score)
            if (score > -float('inf')) & (score < float('inf')):
                state = tl.where(finish_penultimate, states - 2, states - 1)
                # The scalar traceback can read a predecessor written by another warp.
                tl.debug_barrier()
                for reverse_frame in range(frames):
                    frame = frames - 1 - reverse_frame
                    tl.store(Paths + batch * MAX_FRAMES + frame, state)
                    if frame > 0:
                        start = tl.load(BandStarts + batch * MAX_FRAMES + frame) if BANDED else 0
                        move = tl.load(Backpointers + (batch * MAX_FRAMES + frame) * MAX_WIDTH + state - start)
                        state -= move.to(tl.int32)


def batched_ctc_viterbi(
    log_probs, labels, frame_lengths, state_lengths, *, band_starts=None, band_widths=None, max_band_width=None
):
    """Return best paths ``(B,T)`` and path scores ``(B,)`` on CUDA.

    ``log_probs`` is contiguous FP32 ``(B,T,C)`` with blank column zero.
    ``labels`` is contiguous int32/int64 ``(B,S)`` containing blank-expanded
    target columns: ``[0, token, 0, token, ..., 0]``. Length vectors are
    contiguous int32/int64 ``(B,)`` on the same device. Floating inputs must
    contain finite values or negative infinity, never NaN/positive infinity.

    Paths contain target-state indices, including blank states. Padding and
    unreachable paths are ``-1``; their scores are negative infinity. Empty
    dimensions return these values without launching a kernel. An invalid
    per-record length is treated as unreachable, with no out-of-bounds access.
    Shapes above ``MAX_SUPPORTED_STATES`` and non-CUDA execution are unsupported
    so callers can retain their existing CPU implementation for those cases.

    Optional contiguous integer ``band_starts[B,T]`` and ``band_widths[B]``
    restrict each frame to a moving band. Starts must begin at zero, advance by
    zero, one, or two, and keep the band inside valid states. ``max_band_width``
    is a host-side upper bound on all widths; it avoids a device synchronization
    for launch sizing. Without that bound, padded target width is used.

    This function is inference-only. It neither normalizes emissions nor applies
    speaker priors or frame selection.
    """
    _validate_inputs(log_probs, labels, frame_lengths, state_lengths)
    batch, frames, columns = log_probs.shape
    states = labels.shape[1]
    banded = band_starts is not None
    if banded != (band_widths is not None):
        raise ValueError('band_starts and band_widths must be provided together')
    width = states
    if banded:
        _validate_bands(band_starts, band_widths, log_probs)
        width = states if max_band_width is None else max_band_width
        if states and (isinstance(width, bool) or not isinstance(width, int) or not 0 < width <= states):
            raise ValueError('max_band_width must be a positive integer no greater than padded target width')
    paths = torch.full((batch, frames), -1, dtype=torch.int32, device=log_probs.device)
    scores = torch.full((batch,), -float('inf'), dtype=torch.float32, device=log_probs.device)
    if not batch or not frames or not states:
        return paths, scores
    if not log_probs.is_cuda:
        raise NotImplementedError('CTC Viterbi requires CUDA')
    if triton is None:
        raise NotImplementedError('CUDA CTC Viterbi requires Triton')
    if states > MAX_SUPPORTED_STATES:
        raise NotImplementedError(f'CUDA CTC Viterbi supports at most {MAX_SUPPORTED_STATES} target states')
    if not columns:
        return paths, scores
    block = triton.next_power_of_2(width)
    backpointers = torch.empty((batch, frames, width), dtype=torch.uint8, device=log_probs.device)
    with torch.cuda.device(log_probs.device):
        _ctc_viterbi_kernel[(batch,)](
            log_probs,
            labels,
            frame_lengths,
            state_lengths,
            band_starts if banded else frame_lengths,
            band_widths if banded else state_lengths,
            backpointers,
            paths,
            scores,
            log_probs.stride(0),
            log_probs.stride(1),
            labels.stride(0),
            frames,
            states,
            width,
            columns,
            banded,
            block,
            num_warps=8 if block >= 2048 else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return paths, scores


def gpu_prepared_paths(prepared, device):
    """Precompute eligible CPU-prepared mono-speaker paths on CUDA.

    Returns ``(record_index, stream_index) -> (path, score, band_starts)`` with
    NumPy paths/starts for the existing formatter. Unsupported records retain
    their native CPU path. Native anchor construction stays on the host, so
    long-turn bands and short-turn full searches have identical semantics.
    """
    if triton is None or torch.device(device).type != 'cuda':
        return {}
    if not prepared['config']['epsilon'] <= 1 or not math.isfinite(prepared['config']['speaker_logprob_weight']):
        return {}
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import _ctc_anchor_band_starts

    eligible, paths = [], {}
    for index, record in enumerate(prepared['records']):
        streams = record['streams']
        if len(streams) != 1 or record['speaker_probs'] is not None or record['diarization_labels'] is not None:
            continue
        ctc = record['ctc_log_probs']
        labels = np.asarray(streams[0]['labels'], np.int64)
        frames, states = record['ctc_length'], len(labels)
        if (
            not 0 < states <= MAX_SUPPORTED_STATES
            or not 0 < frames <= ctc.shape[0]
            or ctc.dtype != torch.float32
            or ctc.device.type != 'cpu'
        ):
            continue
        half = states if frames < 1000 else min(states, math.ceil(0.10 * states + 100))
        width = min(states, 2 * half + 1)
        target = labels[1::2]
        if frames < len(target) + np.count_nonzero(target[1:] == target[:-1]):
            starts = np.clip(np.rint(np.linspace(0, states - 1, frames)).astype(np.int64) - half, 0, states - width)
            paths[index, 0] = (np.full(frames, -1, np.int64), -np.inf, starts)
            continue
        starts = (
            np.zeros(frames, np.int64)
            if width == states
            else _ctc_anchor_band_starts(ctc[:frames].numpy().argmax(axis=1), target, states, half, width)
        )
        eligible.append((index, ctc[:frames], labels, starts, width))
    if not eligible:
        return paths
    batch, frames, columns = (
        len(eligible),
        max(row[1].shape[0] for row in eligible),
        max(row[1].shape[1] for row in eligible),
    )
    states, width = max(len(row[2]) for row in eligible), max(row[4] for row in eligible)
    emissions = torch.full((batch, frames, columns), -float('inf'), dtype=torch.float32, device='cpu')
    labels = torch.zeros((batch, states), dtype=torch.long, device='cpu')
    starts = torch.zeros((batch, frames), dtype=torch.long, device='cpu')
    for row, (_, ctc, target, band, _) in enumerate(eligible):
        emissions[row, : ctc.shape[0], : ctc.shape[1]] = ctc
        labels[row, : len(target)] = torch.from_numpy(target)
        starts[row, : len(band)] = torch.from_numpy(band)
    device = torch.device(device)
    bands = {}
    if any(row[4] < len(row[2]) for row in eligible):
        bands = dict(
            band_starts=starts.to(device=device),
            band_widths=torch.tensor([row[4] for row in eligible], dtype=torch.long, device=device),
            max_band_width=width,
        )
    gpu_paths, scores = batched_ctc_viterbi(
        emissions.to(device=device),
        labels.to(device=device),
        torch.tensor([row[1].shape[0] for row in eligible], dtype=torch.long, device=device),
        torch.tensor([len(row[2]) for row in eligible], dtype=torch.long, device=device),
        **bands,
    )
    gpu_paths, scores = gpu_paths.cpu().numpy(), scores.cpu().numpy()
    for row, (index, ctc, _, band, _) in enumerate(eligible):
        paths[index, 0] = (gpu_paths[row, : ctc.shape[0]].astype(np.int64), float(scores[row]), band)
    return paths


def _validate_inputs(log_probs, labels, frame_lengths, state_lengths):
    if log_probs.ndim != 3 or log_probs.dtype != torch.float32:
        raise ValueError('log_probs must be FP32 with shape (batch, frames, columns)')
    batch = log_probs.shape[0]
    if labels.ndim != 2 or labels.shape[0] != batch:
        raise ValueError('labels must have shape (batch, padded target states)')
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError('labels must use int32 or int64 column indices')
    for name, lengths in (('frame_lengths', frame_lengths), ('state_lengths', state_lengths)):
        if lengths.ndim != 1 or lengths.shape[0] != batch or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError(f'{name} must be an int32/int64 vector with one entry per record')
    for tensor in (log_probs, labels, frame_lengths, state_lengths):
        if tensor.device != log_probs.device:
            raise ValueError('All CTC Viterbi inputs must be on the same device')
        if not tensor.is_contiguous():
            raise ValueError('CTC Viterbi inputs must be contiguous')


def _validate_bands(starts, widths, log_probs):
    if starts.shape != log_probs.shape[:2] or starts.dtype not in (torch.int32, torch.int64):
        raise ValueError('band_starts must be integer and have shape (batch, frames)')
    if widths.shape != (log_probs.shape[0],) or widths.dtype not in (torch.int32, torch.int64):
        raise ValueError('band_widths must be an integer vector with one entry per record')
    if any(tensor.device != log_probs.device or not tensor.is_contiguous() for tensor in (starts, widths)):
        raise ValueError('Band tensors must be contiguous and on the log_probs device')
