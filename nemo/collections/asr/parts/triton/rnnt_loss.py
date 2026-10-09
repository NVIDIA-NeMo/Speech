# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared Triton dynamic programming for exact standard RNN-T losses."""

from __future__ import annotations

import torch

from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl


def _validate_transition_inputs(
    target_scores: torch.Tensor,
    blank_scores: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> None:
    if target_scores.ndim != 3 or blank_scores.ndim != 3:
        raise ValueError("RNN-T transition scores must have shape [B, T, U] and [B, T, U + 1]")
    if blank_scores.shape[:2] != target_scores.shape[:2] or blank_scores.shape[2] != target_scores.shape[2] + 1:
        raise ValueError(
            f"Incompatible target/blank score shapes: {tuple(target_scores.shape)} and {tuple(blank_scores.shape)}"
        )
    batch = target_scores.shape[0]
    if source_lengths.shape != (batch,) or target_lengths.shape != (batch,):
        raise ValueError("source_lengths and target_lengths must have shape [B]")
    if any(tensor.device != target_scores.device for tensor in (blank_scores, source_lengths, target_lengths)):
        raise ValueError("Transition scores and length tensors must be on the same device")


if TRITON_AVAILABLE:

    @triton.jit
    def _logaddexp(left, right):
        maximum = tl.maximum(left, right)
        value = maximum + tl.log(tl.exp(left - maximum) + tl.exp(right - maximum))
        return tl.where(maximum == -float("inf"), -float("inf"), value)

    @triton.jit
    def _rnnt_loss_kernel(
        target_scores_ptr,
        blank_scores_ptr,
        source_lengths_ptr,
        target_lengths_ptr,
        alpha_ptr,
        losses_ptr,
        target_occupation_ptr,
        blank_occupation_ptr,
        max_source,
        max_target,
        fastemit_scale: tl.constexpr,
        block_target: tl.constexpr,
    ):
        """Score one utterance with diagonal wavefronts and save transition occupancies.

        Each GPU program handles one utterance, with target positions distributed
        across lanes. Each diagonal depends only on the previous diagonal;
        neighbouring lanes exchange its values through gathers. The reverse pass
        combines beta with stored alpha to compute blank and target occupancies
        for autograd backward.
        """
        batch_idx = tl.program_id(0)
        source_len = tl.load(source_lengths_ptr + batch_idx)
        target_len = tl.load(target_lengths_ptr + batch_idx)
        valid_lengths = (source_len >= 1) & (source_len <= max_source) & (target_len >= 0) & (target_len <= max_target)
        num_diagonals = tl.where(valid_lengths, source_len + target_len, 0)
        symbols = tl.arange(0, block_target)
        valid_symbol = valid_lengths & (symbols <= target_len)
        previous = tl.full((block_target,), -float("inf"), tl.float32)

        for diagonal in tl.range(0, num_diagonals):
            time_idx = diagonal - symbols
            valid_state = valid_symbol & (time_idx >= 0) & (time_idx < source_len)
            blank_offset = (batch_idx * max_source + time_idx - 1) * (max_target + 1) + symbols
            blank = tl.load(
                blank_scores_ptr + blank_offset,
                mask=valid_state & (time_idx > 0),
                other=-float("inf"),
            )

            target_offset = (batch_idx * max_source + time_idx) * max_target + symbols - 1
            target = tl.load(
                target_scores_ptr + target_offset,
                mask=valid_state & (symbols > 0),
                other=-float("inf"),
            )
            left = tl.gather(previous, tl.maximum(symbols - 1, 0), axis=0)
            current = _logaddexp(previous + blank, left + target)
            current = tl.where((time_idx == 0) & (symbols == 0), 0.0, current)
            current = tl.where(valid_state, current, -float("inf"))
            alpha_offset = (batch_idx * max_source + time_idx) * (max_target + 1) + symbols
            tl.store(alpha_ptr + alpha_offset, current, mask=valid_state)
            previous = current

        final_blank_offset = (batch_idx * max_source + source_len - 1) * (max_target + 1) + target_len
        final_blank = tl.load(
            blank_scores_ptr + final_blank_offset,
            mask=valid_lengths,
            other=-float("inf"),
        )
        log_likelihood = tl.sum(tl.where(symbols == target_len, previous, 0.0), axis=0) + final_blank
        tl.store(losses_ptr + batch_idx, -log_likelihood * fastemit_scale)
        previous = tl.full((block_target,), -float("inf"), tl.float32)

        for reverse_diagonal in tl.range(0, num_diagonals):
            time_idx = num_diagonals - 1 - reverse_diagonal - symbols
            valid_state = valid_symbol & (time_idx >= 0) & (time_idx < source_len)
            offset = (batch_idx * max_source + time_idx) * (max_target + 1) + symbols
            blank = tl.load(blank_scores_ptr + offset, mask=valid_state, other=-float("inf"))
            target_offset = (batch_idx * max_source + time_idx) * max_target + symbols
            target = tl.load(
                target_scores_ptr + target_offset,
                mask=valid_state & (symbols < target_len),
                other=-float("inf"),
            )
            right = tl.gather(previous, tl.minimum(symbols + 1, block_target - 1), axis=0)
            beta = _logaddexp(previous + blank, right + target)
            last_time = time_idx == source_len - 1
            beta = tl.where(last_time & (symbols == target_len), blank, beta)
            beta = tl.where(valid_state, beta, -float("inf"))
            alpha = tl.load(alpha_ptr + offset, mask=valid_state, other=-float("inf"))
            blank_occupation = tl.where(
                last_time,
                tl.where(symbols == target_len, 1.0, 0.0),
                tl.exp(alpha + blank + previous - log_likelihood),
            )
            tl.store(blank_occupation_ptr + offset, blank_occupation, mask=valid_state)
            target_occupation = tl.exp(alpha + target + right - log_likelihood)
            tl.store(
                target_occupation_ptr + target_offset,
                target_occupation,
                mask=valid_state & (symbols < target_len),
            )
            previous = beta


class _RNNTLossTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        target_scores,
        blank_scores,
        source_lengths,
        target_lengths,
        fastemit_lambda,
        loss_grad_scale,
    ):
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is required for RNN-T CUDA training")
        if not target_scores.is_cuda:
            raise RuntimeError("Triton RNN-T training requires CUDA tensors")
        _validate_transition_inputs(target_scores, blank_scores, source_lengths, target_lengths)

        target_scores = target_scores.contiguous()
        blank_scores = blank_scores.contiguous()
        source_lengths = source_lengths.to(dtype=torch.int32).contiguous()
        target_lengths = target_lengths.to(dtype=torch.int32).contiguous()
        batch, max_source, max_target = target_scores.shape
        if fastemit_lambda < 0.0:
            raise ValueError("fastemit_lambda must be nonnegative")
        block_target = triton.next_power_of_2(max_target + 1)
        fastemit_scale = 1.0 + float(fastemit_lambda)

        alpha = torch.empty_like(blank_scores, dtype=torch.float32)
        losses = torch.empty((batch,), device=target_scores.device, dtype=torch.float32)
        # The recurrence only writes valid positions; padding must have zero gradient.
        target_occupation = torch.zeros_like(target_scores, dtype=torch.float32)
        blank_occupation = torch.zeros_like(blank_scores, dtype=torch.float32)
        _rnnt_loss_kernel[(batch,)](
            target_scores,
            blank_scores,
            source_lengths,
            target_lengths,
            alpha,
            losses,
            target_occupation,
            blank_occupation,
            max_source=max_source,
            max_target=max_target,
            fastemit_scale=fastemit_scale,
            block_target=block_target,
            # Use one lane per target position up to the 16-warp limit.
            num_warps=min(max(block_target // 32, 1), 16),
        )
        ctx.save_for_backward(target_occupation, blank_occupation)
        ctx.fastemit_scale = fastemit_scale
        # Held outside save_for_backward: it is written during backward, and the version
        # counter would reject a saved tensor that changed after forward.
        ctx.loss_grad_scale = loss_grad_scale
        return losses

    @staticmethod
    def backward(ctx, grad_losses):
        target_occupation, blank_occupation = ctx.saved_tensors
        if ctx.loss_grad_scale is not None:
            # The score producer's backward uses this scale for gradient clamping.
            # It runs after the loss backward because it consumes the score gradients.
            ctx.loss_grad_scale.copy_(grad_losses.detach())
        scale = -grad_losses.float().reshape(-1, 1, 1)
        target_grad = target_occupation * (scale * ctx.fastemit_scale)
        blank_grad = blank_occupation * scale
        return target_grad, blank_grad, None, None, None, None


def rnnt_loss_triton(
    target_scores: torch.Tensor,
    blank_scores: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    fastemit_lambda: float = 0.0,
    loss_grad_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return exact per-sample RNN-T losses from blank and target scores.

    Args:
        loss_grad_scale: optional ``[B]`` float32 buffer. Backward writes ``grad_losses`` into it --
            the objective's gradient with respect to each per-sample loss, which the reduction and any
            AMP scale determine -- so a producer of the scores can recover the unit scale its own
            gradients were computed at. Only gradient clamping needs it.
    """
    return _RNNTLossTriton.apply(
        target_scores, blank_scores, source_lengths, target_lengths, fastemit_lambda, loss_grad_scale
    )
