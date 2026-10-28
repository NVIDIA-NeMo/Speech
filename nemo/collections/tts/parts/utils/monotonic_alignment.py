# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact monotonic alignment search for independently batched speech turns.

The CUDA backend keeps the dynamic-programming row in registers and stores only
one byte per lattice cell for traceback. Lengths and strides remain runtime
values, so independently padded turns share kernels within each power-of-two
text-width bucket. It returns token indices rather than a dense one-hot path.
The CPU backend reuses NeMo's existing Numba MAS.
"""

import numpy as np
import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


@torch.no_grad()
def maximum_path_indices(log_scores, text_lens, audio_lens, backend='auto'):
    """Find the maximum-score monotonic, non-skipping path for every turn.

    Args:
        log_scores: Floating-point acoustic scores, shaped ``(B, T, N)``.
            Scores are accumulated in float32, including for bf16/fp16 inputs.
        text_lens: Integer text lengths, shaped ``(B,)``.
        audio_lens: Integer audio lengths, shaped ``(B,)``.
        backend: ``auto``, ``cpu`` or ``triton``. Auto uses Triton for CUDA
            tensors when available and padded text length is at most 4096;
            otherwise it uses the existing CPU implementation. Explicit
            ``triton`` requests raise rather than silently falling back.

    Returns:
        Long tensor shaped ``(B, T)`` on the score device, containing one text
        position per frame. Padding and invalid turns are ``-1``. Zero-length,
        out-of-bounds and ``T < N`` turns are invalid. A path with a nonfinite
        total score is also invalid. Every valid path starts at position zero,
        ends at ``N - 1`` and advances by zero or one at each frame. Ties choose
        advancement during traceback, matching NeMo's ``mas_width1``.

    This discrete search is intentionally nondifferentiable. Gather the selected
    scores from the original tensor to train the acoustic likelihood.
    """
    if log_scores.ndim != 3 or not torch.is_floating_point(log_scores):
        raise ValueError('log_scores must be a floating-point tensor shaped (B, T, N)')
    batch_size, max_audio, max_text = log_scores.shape
    if text_lens.shape != (batch_size,) or audio_lens.shape != (batch_size,):
        raise ValueError('text_lens and audio_lens must have shape (B,)')
    if text_lens.dtype not in (torch.int32, torch.int64) or audio_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError('text_lens and audio_lens must contain integers')
    if backend not in ('auto', 'cpu', 'triton'):
        raise ValueError(f'Unsupported MAS backend: {backend!r}')
    use_triton = log_scores.is_cuda and TRITON_AVAILABLE and max_text <= 4096
    if backend == 'triton' and not use_triton:
        raise ValueError('Triton MAS requires CUDA, Triton, and padded text length <= 4096')
    path = torch.full((batch_size, max_audio), -1, device=log_scores.device, dtype=torch.long)
    if batch_size == 0 or max_audio == 0 or max_text == 0:
        return path
    if backend == 'cpu' or not use_triton:
        return _maximum_path_cpu(log_scores, text_lens, audio_lens, path)

    text_lens = text_lens.to(device=log_scores.device).contiguous()
    audio_lens = audio_lens.to(device=log_scores.device).contiguous()
    decisions = torch.empty(log_scores.shape, device=log_scores.device, dtype=torch.uint8)
    valid = torch.empty(batch_size, device=log_scores.device, dtype=torch.uint8)
    block_text = triton.next_power_of_2(max_text)
    _mas_forward[(batch_size,)](
        log_scores,
        text_lens,
        audio_lens,
        decisions,
        valid,
        *log_scores.stride(),
        MAX_AUDIO=max_audio,
        MAX_TEXT=max_text,
        BLOCK_TEXT=block_text,
        num_warps=4 if block_text >= 128 else 1,
    )
    _mas_traceback[(batch_size,)](
        decisions, text_lens, audio_lens, valid, path, MAX_AUDIO=max_audio, MAX_TEXT=max_text, num_warps=1
    )
    return path


def _maximum_path_cpu(log_scores, text_lens, audio_lens, path):
    from nemo.collections.tts.parts.utils.helpers import b_mas

    scores = log_scores.detach().float().cpu().numpy()
    text_lengths = text_lens.detach().cpu().numpy()
    audio_lengths = audio_lens.detach().cpu().numpy()
    batch_size, max_audio, max_text = scores.shape
    valid = (
        (text_lengths > 0)
        & (text_lengths <= max_text)
        & (audio_lengths >= text_lengths)
        & (audio_lengths <= max_audio)
    )
    # The legacy width-one traceback assumes at least two text positions. A
    # one-token turn has exactly one path and needs no dynamic programming.
    single = valid & (text_lengths == 1)
    path_cpu = np.full((batch_size, max_audio), -1, dtype=np.int64)
    for index in np.flatnonzero(single):
        length = audio_lengths[index]
        if np.isfinite(scores[index, :length, 0].sum(dtype=np.float32)):
            path_cpu[index, :length] = 0
    indices = np.flatnonzero(valid & ~single)
    if indices.size:
        selected_scores = np.ascontiguousarray(scores[indices, None])
        selected_text_lens = np.ascontiguousarray(text_lengths[indices])
        selected_audio_lens = np.ascontiguousarray(audio_lengths[indices])
        hard_path = b_mas(selected_scores, selected_text_lens, selected_audio_lens, width=1)
        for local_index, batch_index in enumerate(indices):
            length = audio_lengths[batch_index]
            positions = hard_path[local_index, 0, :length].argmax(axis=-1)
            total = scores[batch_index, np.arange(length), positions].sum(dtype=np.float32)
            if np.isfinite(total):
                path_cpu[batch_index, :length] = positions
    path.copy_(torch.from_numpy(path_cpu).to(device=path.device))
    return path


if TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=['STRIDE_BATCH', 'STRIDE_AUDIO', 'STRIDE_TEXT', 'MAX_AUDIO', 'MAX_TEXT'])
    def _mas_forward(
        scores,
        text_lens,
        audio_lens,
        decisions,
        valid,
        STRIDE_BATCH,
        STRIDE_AUDIO,
        STRIDE_TEXT,
        MAX_AUDIO,
        MAX_TEXT,
        BLOCK_TEXT: tl.constexpr,
    ):
        batch = tl.program_id(0)
        num_text = tl.load(text_lens + batch).to(tl.int32)
        num_audio = tl.load(audio_lens + batch).to(tl.int32)
        turn_valid = (num_text > 0) & (num_text <= MAX_TEXT) & (num_audio >= num_text) & (num_audio <= MAX_AUDIO)
        tl.store(valid + batch, 0)
        if turn_valid:
            token = tl.arange(0, BLOCK_TEXT)
            previous = tl.where(token == 0, 0.0, -float('inf'))
            for frame in range(num_audio):
                advanced = tl.gather(previous, tl.maximum(token - 1, 0), axis=0)
                advanced = tl.where(token > 0, advanced, -float('inf'))
                moved = (advanced >= previous) & (token > 0)
                score = tl.load(
                    scores + batch * STRIDE_BATCH + frame * STRIDE_AUDIO + token * STRIDE_TEXT,
                    mask=token < num_text,
                    other=-float('inf'),
                ).to(tl.float32)
                current = tl.maximum(previous, advanced) + score
                reachable = (token < num_text) & (token <= frame) & (num_text - 1 - token <= num_audio - 1 - frame)
                previous = tl.where(reachable, current, -float('inf'))
                tl.store(
                    decisions + (batch * MAX_AUDIO + frame) * MAX_TEXT + token,
                    moved.to(tl.uint8),
                    mask=token < num_text,
                )
            final_score = tl.sum(tl.where(token == num_text - 1, previous, 0.0), axis=0)
            tl.store(valid + batch, (final_score > -float('inf')) & (final_score < float('inf')))

    @triton.jit(do_not_specialize=['MAX_AUDIO', 'MAX_TEXT'])
    def _mas_traceback(
        decisions,
        text_lens,
        audio_lens,
        valid,
        path,
        MAX_AUDIO,
        MAX_TEXT,
    ):
        batch = tl.program_id(0)
        if tl.load(valid + batch):
            token = tl.load(text_lens + batch).to(tl.int32) - 1
            num_audio = tl.load(audio_lens + batch).to(tl.int32)
            for step in range(num_audio):
                frame = num_audio - 1 - step
                tl.store(path + batch * MAX_AUDIO + frame, token)
                moved = tl.load(decisions + (batch * MAX_AUDIO + frame) * MAX_TEXT + token).to(tl.int32)
                token -= moved
