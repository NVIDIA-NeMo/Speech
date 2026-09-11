# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
    def _log_semiring_compose(a_left, b_left, a_right, b_right):
        # Compose F(x) = logadd(a, b + x). The inclusive scan's ``a``
        # component is the RNN-T recurrence value at that lane.
        return _logaddexp(a_right, b_right + a_left), b_right + b_left

    @triton.jit
    def _rnnt_loss_kernel(
        target_scores_ptr,
        blank_scores_ptr,
        source_lengths_ptr,
        target_lengths_ptr,
        alpha_ptr,
        losses_ptr,
        beta_ptr,
        blank_occupation_ptr,
        max_source,
        max_target,
        fastemit_scale: tl.constexpr,
        block_target: tl.constexpr,
    ):
        """Score one utterance's lattice and leave the occupations backward consumes.

        Both passes walk source time sequentially and resolve the whole target axis in one scan over
        the log semiring, so a lane holds a target state rather than a time step. The reverse pass
        reads back the alphas the forward stored, which is what lets it emit each blank transition's
        posterior beside beta.
        """
        batch_idx = tl.program_id(0)
        source_len = tl.load(source_lengths_ptr + batch_idx)
        target_len = tl.load(target_lengths_ptr + batch_idx)
        valid_lengths = (source_len >= 1) & (source_len <= max_source) & (target_len >= 0) & (target_len <= max_target)
        symbols = tl.arange(0, block_target)
        valid_symbol = valid_lengths & (symbols <= target_len)
        previous = tl.where(symbols == 0, 0.0, -float("inf"))

        for time_idx in tl.range(0, max_source):
            active_time = valid_lengths & (time_idx < source_len)
            blank_offset = (batch_idx * max_source + time_idx - 1) * (max_target + 1) + symbols
            from_blank = previous + tl.load(
                blank_scores_ptr + blank_offset,
                mask=(time_idx > 0) & active_time & valid_symbol,
                other=0.0,
            )
            base = tl.where(
                time_idx == 0,
                tl.where(symbols == 0, 0.0, -float("inf")),
                from_blank,
            )

            target_offset = (batch_idx * max_source + time_idx) * max_target + symbols - 1
            step = tl.load(
                target_scores_ptr + target_offset,
                mask=active_time & (symbols > 0) & (symbols <= target_len),
                other=-float("inf"),
            )
            current, _ = tl.associative_scan((base, step), axis=0, combine_fn=_log_semiring_compose)
            current = tl.where(valid_symbol, current, -float("inf"))

            alpha_offset = (batch_idx * max_source + time_idx) * (max_target + 1) + symbols
            tl.store(alpha_ptr + alpha_offset, current, mask=active_time & valid_symbol)
            previous = tl.where(active_time, current, previous)

        final_alpha = tl.sum(tl.where(symbols == target_len, previous, 0.0), axis=0)
        final_blank_offset = (batch_idx * max_source + source_len - 1) * (max_target + 1) + target_len
        final_blank = tl.load(
            blank_scores_ptr + final_blank_offset,
            mask=valid_lengths,
            other=-float("inf"),
        )
        log_likelihood = final_alpha + final_blank
        tl.store(losses_ptr + batch_idx, -log_likelihood * fastemit_scale)
        # The beta pass below walks the target axis in reverse, so each lane reads alpha values
        # that a different lane wrote. This orders those stores against the reads instead of
        # leaning on the barriers the scan lowering happens to emit. It is not a debug aid.
        tl.debug_barrier()

        symbols = max_target - tl.arange(0, block_target)
        valid_symbol = valid_lengths & (symbols >= 0) & (symbols <= target_len)
        beta_next = tl.full((block_target,), -float("inf"), tl.float32)

        for reverse_time in tl.range(0, max_source):
            time_idx = max_source - reverse_time - 1
            active_time = valid_lengths & (time_idx < source_len)
            blank_offset = (batch_idx * max_source + time_idx) * (max_target + 1) + symbols
            blank = tl.load(
                blank_scores_ptr + blank_offset,
                mask=active_time & valid_symbol,
                other=0.0,
            )
            last_time = time_idx == source_len - 1
            base = tl.where(
                last_time,
                tl.where(symbols == target_len, blank, -float("inf")),
                blank + beta_next,
            )
            target_offset = (batch_idx * max_source + time_idx) * max_target + symbols
            target = tl.load(
                target_scores_ptr + target_offset,
                mask=active_time & (symbols >= 0) & (symbols < target_len),
                other=-float("inf"),
            )
            step = tl.where((symbols >= 0) & (symbols < target_len), target, -float("inf"))
            beta, _ = tl.associative_scan((base, step), axis=0, combine_fn=_log_semiring_compose)
            beta = tl.where(valid_symbol, beta, -float("inf"))

            beta_offset = (batch_idx * max_source + time_idx) * (max_target + 1) + symbols
            alpha = tl.load(
                alpha_ptr + beta_offset,
                mask=active_time & valid_symbol,
                other=-float("inf"),
            )
            blank_occupation = tl.where(
                last_time,
                tl.where(symbols == target_len, 1.0, 0.0),
                tl.exp(alpha + blank + beta_next - log_likelihood),
            )
            tl.store(
                blank_occupation_ptr + blank_offset,
                tl.where(active_time & valid_symbol, blank_occupation, 0.0),
                mask=(symbols >= 0) & (symbols <= max_target),
            )
            tl.store(beta_ptr + beta_offset, beta, mask=active_time & valid_symbol)
            beta_next = tl.where(active_time, beta, beta_next)

    @triton.jit
    def _rnnt_target_occupation_kernel(
        target_scores_ptr,
        source_lengths_ptr,
        target_lengths_ptr,
        alpha_ptr,
        beta_ptr,
        losses_ptr,
        target_occupation_ptr,
        max_source,
        max_target,
        fastemit_scale: tl.constexpr,
        block_target: tl.constexpr,
    ):
        """Posterior of each target transition, from the alphas and betas the loss kernel stored."""
        batch_idx = tl.program_id(0)
        time_idx = tl.program_id(1)
        symbols = tl.arange(0, block_target)
        source_len = tl.load(source_lengths_ptr + batch_idx)
        target_len = tl.load(target_lengths_ptr + batch_idx)
        valid_lengths = (source_len >= 1) & (source_len <= max_source) & (target_len >= 0) & (target_len <= max_target)
        valid_state = valid_lengths & (time_idx < source_len) & (symbols < target_len)
        alpha_offset = (batch_idx * max_source + time_idx) * (max_target + 1) + symbols
        target_offset = (batch_idx * max_source + time_idx) * max_target + symbols
        alpha = tl.load(alpha_ptr + alpha_offset, mask=valid_state, other=-float("inf"))
        target = tl.load(target_scores_ptr + target_offset, mask=valid_state, other=-float("inf"))
        beta_after = tl.load(beta_ptr + alpha_offset + 1, mask=valid_state, other=-float("inf"))
        log_likelihood = -tl.load(losses_ptr + batch_idx) / fastemit_scale
        occupation = tl.exp(alpha + target + beta_after - log_likelihood)
        tl.store(
            target_occupation_ptr + target_offset,
            tl.where(valid_state, occupation, 0.0),
            mask=symbols < max_target,
        )


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
        beta = torch.empty_like(blank_scores, dtype=torch.float32)
        losses = torch.empty((batch,), device=target_scores.device, dtype=torch.float32)
        target_occupation = torch.empty_like(target_scores, dtype=torch.float32)
        blank_occupation = torch.empty_like(blank_scores, dtype=torch.float32)
        _rnnt_loss_kernel[(batch,)](
            target_scores,
            blank_scores,
            source_lengths,
            target_lengths,
            alpha,
            losses,
            beta,
            blank_occupation,
            max_source=max_source,
            max_target=max_target,
            fastemit_scale=fastemit_scale,
            block_target=block_target,
            # The scan wants one target state per lane: fewer warps make each lane walk several
            # states in sequence, and more spend longer combining across warps than they save.
            # The floor keeps a small target axis from launching too narrow to fill the device.
            num_warps=min(max(block_target // 32, 4), 16),
        )
        # Nothing to score when every transcript in the batch is empty.
        if max_target > 0:
            _rnnt_target_occupation_kernel[(batch, max_source)](
                target_scores,
                source_lengths,
                target_lengths,
                alpha,
                beta,
                losses,
                target_occupation,
                max_source=max_source,
                max_target=max_target,
                fastemit_scale=fastemit_scale,
                block_target=triton.next_power_of_2(max_target),
            )
        ctx.save_for_backward(target_occupation, blank_occupation)
        ctx.fastemit_scale = fastemit_scale
        # Held outside save_for_backward: it is written during backward, and the version
        # counter would reject a saved tensor that changed after forward.
        ctx.loss_grad_scale = loss_grad_scale
        ctx.mark_non_differentiable(target_occupation, blank_occupation)
        return losses, target_occupation, blank_occupation

    @staticmethod
    def backward(ctx, grad_losses, _grad_target_occupation, _grad_blank_occupation):
        target_occupation, blank_occupation = ctx.saved_tensors
        if ctx.loss_grad_scale is not None:
            # Publish this backward's per-sample scale for whoever produced the scores.
            # Score gradients feed this loss, so their backward always runs after ours.
            ctx.loss_grad_scale.copy_(grad_losses.detach())
        scale = grad_losses.float().reshape(-1, 1, 1)
        target_grad = -target_occupation * (scale * ctx.fastemit_scale)
        blank_grad = -blank_occupation * scale
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
    losses, _, _ = _RNNTLossTriton.apply(
        target_scores, blank_scores, source_lengths, target_lengths, fastemit_lambda, loss_grad_scale
    )
    return losses
