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


import torch
import triton
import triton.language as tl

from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs_triton import (
    _check_logits,
    _rnnt_logprobs_bwd,
    _rnnt_logprobs_fwd,
    rnnt_logprobs_triton,
)
from nemo.collections.asr.parts.turbo_transducer.utils_triton import cuda_device_guard, log_add_exp

_MIN_DIAGONAL_BLOCK_SIZE = 32


# Shape-dependent integer arguments are not specialized to avoid recompilation for each new batch shape
# (Triton specializes integers equal to 1 and divisible by 16 by default).
@triton.jit(do_not_specialize=["max_src_len", "max_tgt_len_plus_1"])
def _rnnt_fwd_kernel(
    target_logprobs_ptr,
    blank_logprobs_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    alpha_out_ptr,
    loss_batch_out_ptr,
    max_src_len,
    max_tgt_len_plus_1,
    BLOCK_SIZE: tl.constexpr,
    PARALLELIZE_OVER_SRC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Forward kernel for RNN-T loss.

    Calculations are performed in float32 or float64 based on USE_FP64.
    PARALLELIZE_OVER_SRC: if True, offsets enumerate src positions; if False, tgt positions.
    Only alpha values for valid positions are written (and later read), alpha does not require initialization.

    The loss is NaN for the lengths that exceed the size of the lattice (no out-of-bounds memory access),
    and infinite if there is no alignment with non-zero probability.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    NEG_INF = -float("inf")  # score for unreachable states

    src_len = tl.load(src_lengths_ptr + batch_i).to(tl.int64)
    tgt_len = tl.load(tgt_lengths_ptr + batch_i).to(tl.int64)

    if src_len > max_src_len or tgt_len < 0 or tgt_len >= max_tgt_len_plus_1:
        # invalid lengths: the lattice is outside the tensors
        tl.store(loss_batch_out_ptr + batch_i, float("nan"))
        return

    if src_len <= 0:
        # no valid alignments for empty source: loss is infinite
        tl.store(loss_batch_out_ptr + batch_i, float("inf"))
        return

    # Base offset for this batch element in [B, T, U+1] tensors
    batch_offset = batch_i * max_src_len * max_tgt_len_plus_1

    # alpha[0, 0] = 0.0
    tl.store(alpha_out_ptr + batch_offset, 0.0)

    num_diags = src_len + tgt_len
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)

    for diag_i_i32 in tl.range(1, num_diags):
        diag_i = diag_i_i32.to(tl.int64)
        if PARALLELIZE_OVER_SRC:
            src_offsets = offsets
            tgt_offsets = diag_i - offsets
        else:
            tgt_offsets = offsets
            src_offsets = diag_i - offsets
        # Mask: valid positions on this diagonal
        mask = (src_offsets >= 0) & (src_offsets < src_len) & (tgt_offsets >= 0) & (tgt_offsets <= tgt_len)

        # Blank predecessor: alpha[t-1, u] + blank_logprobs[t-1, u] (valid when t > 0)
        blank_mask = mask & (src_offsets > 0)
        blank_pred_idx = batch_offset + (src_offsets - 1) * max_tgt_len_plus_1 + tgt_offsets
        blank_alpha = tl.load(alpha_out_ptr + blank_pred_idx, mask=blank_mask, other=NEG_INF).to(compute_dtype)
        blank_lp = tl.load(blank_logprobs_ptr + blank_pred_idx, mask=blank_mask, other=0.0).to(compute_dtype)
        blank_score = tl.where(blank_mask, blank_alpha + blank_lp, NEG_INF)

        # Emit predecessor: alpha[t, u-1] + target_logprobs[t, u-1] (valid when u > 0)
        emit_mask = mask & (tgt_offsets > 0)
        emit_pred_idx = batch_offset + src_offsets * max_tgt_len_plus_1 + (tgt_offsets - 1)
        emit_alpha = tl.load(alpha_out_ptr + emit_pred_idx, mask=emit_mask, other=NEG_INF).to(compute_dtype)
        emit_lp = tl.load(target_logprobs_ptr + emit_pred_idx, mask=emit_mask, other=0.0).to(compute_dtype)
        emit_score = tl.where(emit_mask, emit_alpha + emit_lp, NEG_INF)

        alpha_diag = log_add_exp(blank_score, emit_score)

        # Store alpha values
        cur_idx = batch_offset + src_offsets * max_tgt_len_plus_1 + tgt_offsets
        tl.store(alpha_out_ptr + cur_idx, alpha_diag, mask=mask)
        # Barrier needed: cross-warp store-load dependency between consecutive diagonals
        tl.debug_barrier()

    # Loss = -(alpha[src_len-1, tgt_len] + blank_logprobs[src_len-1, tgt_len])
    final_idx = batch_offset + (src_len - 1) * max_tgt_len_plus_1 + tgt_len
    final_alpha = tl.load(alpha_out_ptr + final_idx).to(compute_dtype)
    final_blank_lp = tl.load(blank_logprobs_ptr + final_idx).to(compute_dtype)
    # NB: loss is infinite if there is no alignment with non-zero probability (the final state is unreachable)
    loss = -(final_alpha + final_blank_lp)
    tl.store(loss_batch_out_ptr + batch_i, loss)


@triton.jit(do_not_specialize=["max_src_len", "max_tgt_len_plus_1"])
def _rnnt_bwd_kernel(
    target_logprobs_ptr,
    blank_logprobs_ptr,
    alpha_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    grad_rnnt_loss_ptr,
    target_logprobs_grad_out_ptr,
    blank_logprobs_grad_out_ptr,
    max_src_len,
    max_tgt_len_plus_1,
    BLOCK_SIZE: tl.constexpr,
    PARALLELIZE_OVER_SRC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Backward kernel for RNN-T loss.

    Calculations are performed in float32 or float64 based on USE_FP64.
    Beta values are kept in a register tensor (beta_diag) instead of global memory.
    On each diagonal, one successor is at the same offset (aligned — direct register access)
    and the other is at offset+1 (shifted — accessed via tl.gather).
    PARALLELIZE_OVER_SRC: if True, offsets enumerate src positions; if False, tgt positions.

    The gradient is zero for utterances with invalid lengths or infinite loss (see the forward kernel).
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    NEG_INF = -float("inf")  # score for unreachable states

    src_len = tl.load(src_lengths_ptr + batch_i).to(tl.int64)
    tgt_len = tl.load(tgt_lengths_ptr + batch_i).to(tl.int64)
    grad_rnnt_loss = tl.load(grad_rnnt_loss_ptr + batch_i).to(compute_dtype)
    if grad_rnnt_loss == 0.0 or src_len <= 0:
        # gradients are zero-initialized
        return
    if src_len > max_src_len or tgt_len < 0 or tgt_len >= max_tgt_len_plus_1:
        # invalid lengths: the lattice is outside the tensors (loss is NaN)
        return

    batch_offset = batch_i * max_src_len * max_tgt_len_plus_1

    # Initialize beta[src_len-1, tgt_len] = blank_logprobs[src_len-1, tgt_len]
    final_idx = batch_offset + (src_len - 1) * max_tgt_len_plus_1 + tgt_len
    final_blank_lp = tl.load(blank_logprobs_ptr + final_idx).to(compute_dtype)

    # log_like = alpha[src_len-1, tgt_len] + blank_logprobs[src_len-1, tgt_len]
    final_alpha = tl.load(alpha_ptr + final_idx).to(compute_dtype)
    log_like = final_alpha + final_blank_lp
    if log_like == NEG_INF:
        # no alignment with non-zero probability (loss is infinite)
        return

    # Gradient at final state: blank_grad[src_len-1, tgt_len] = -1.0
    tl.store(blank_logprobs_grad_out_ptr + final_idx, -grad_rnnt_loss)

    num_diags = src_len + tgt_len
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)

    # Initialize beta_diag for the final diagonal (only one valid position)
    if PARALLELIZE_OVER_SRC:
        init_pos = src_len - 1
    else:
        init_pos = tgt_len
    beta_diag = tl.where(offsets == init_pos, final_blank_lp, NEG_INF).to(compute_dtype)

    # Precompute shifted gather index: (offsets+1) % BLOCK_SIZE
    # Wraparound at BLOCK_SIZE is always masked out by blank_mask/emit_mask
    shifted_offsets = (offsets + 1) % BLOCK_SIZE

    # Reverse diagonal loop: d from (num_diags - 2) down to 0
    for diag_rev_i_i32 in tl.range(0, num_diags - 1):
        diag_i = num_diags - 2 - diag_rev_i_i32.to(tl.int64)
        if PARALLELIZE_OVER_SRC:
            src_offsets = offsets
            tgt_offsets = diag_i - offsets
        else:
            tgt_offsets = offsets
            src_offsets = diag_i - offsets
        mask = (src_offsets >= 0) & (src_offsets < src_len) & (tgt_offsets >= 0) & (tgt_offsets <= tgt_len)

        # Blank successor: beta[t+1, u] (valid when t+1 < src_len)
        # When PARALLELIZE_OVER_SRC: (t+1,u) is at offset+1 on prev diagonal → shifted (gather)
        # When not: (t+1,u) is at same offset on prev diagonal → aligned (direct)
        blank_mask = mask & (src_offsets + 1 < src_len)
        if PARALLELIZE_OVER_SRC:
            blank_beta = beta_diag.gather(shifted_offsets, axis=0)
        else:
            blank_beta = beta_diag
        cur_idx = batch_offset + src_offsets * max_tgt_len_plus_1 + tgt_offsets
        blank_lp = tl.load(blank_logprobs_ptr + cur_idx, mask=blank_mask, other=0.0).to(compute_dtype)
        blank_score = tl.where(blank_mask, blank_beta + blank_lp, NEG_INF)

        # Emit successor: beta[t, u+1] (valid when u+1 <= tgt_len)
        # When PARALLELIZE_OVER_SRC: (t,u+1) is at same offset on prev diagonal → aligned (direct)
        # When not: (t,u+1) is at offset+1 on prev diagonal → shifted (gather)
        emit_mask = mask & (tgt_offsets + 1 <= tgt_len)
        if PARALLELIZE_OVER_SRC:
            emit_beta = beta_diag
        else:
            emit_beta = beta_diag.gather(shifted_offsets, axis=0)
        emit_lp = tl.load(target_logprobs_ptr + cur_idx, mask=emit_mask, other=0.0).to(compute_dtype)
        emit_score = tl.where(emit_mask, emit_beta + emit_lp, NEG_INF)

        beta_diag = tl.where(mask, log_add_exp(blank_score, emit_score), NEG_INF)

        # Fused gradient computation
        # blank_grad[t, u] = -exp(alpha[t, u] + beta[t+1, u] + blank_logprobs[t, u] - log_like)
        alpha_val = tl.load(alpha_ptr + cur_idx, mask=mask, other=NEG_INF).to(compute_dtype)
        blank_grad = tl.where(
            blank_mask,
            -tl.exp(alpha_val + blank_beta + blank_lp - log_like) * grad_rnnt_loss,
            0.0,
        )
        tl.store(blank_logprobs_grad_out_ptr + cur_idx, blank_grad, mask=mask)

        # target_grad[t, u] = -exp(alpha[t, u] + beta[t, u+1] + target_logprobs[t, u] - log_like)
        target_grad = tl.where(
            emit_mask,
            -tl.exp(alpha_val + emit_beta + emit_lp - log_like) * grad_rnnt_loss,
            0.0,
        )
        tl.store(target_logprobs_grad_out_ptr + cur_idx, target_grad, mask=mask)
        # Barrier needed: cross-warp gather dependency between consecutive diagonals
        tl.debug_barrier()


class TritonRnntLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        target_logprobs: torch.Tensor,
        blank_logprobs: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_lengths: torch.Tensor,
        fastemit_lambda: float = 0.0,
    ):
        """
        RNN-T loss calculation from target/blank log-probs. Forward pass.

        Args:
            ctx: ctx object for storing the context
            target_logprobs: logprobs for target labels of size [B, T, U+1]
            blank_logprobs: logprobs for blank labels of size [B, T, U+1]
            src_lengths: source lengths of size [B]
            tgt_lengths: target lengths of size [B]
            fastemit_lambda: Float scaling factor for FastEmit regularization.

        Returns:
            loss of size [B]. The loss is NaN for utterances with lengths exceeding the size of the lattice,
            and infinite for utterances without alignments with non-zero probability (including empty source);
            the gradient is zero for such utterances.
        """
        if target_logprobs.shape != blank_logprobs.shape or target_logprobs.dim() != 3:
            raise ValueError(
                "Expected target and blank log-probs of the same shape [B, T, U+1], "
                f"got {list(target_logprobs.shape)} and {list(blank_logprobs.shape)}"
            )
        # log-probs are small (no vocabulary dimension), making them contiguous is cheap
        target_logprobs = target_logprobs.contiguous()
        blank_logprobs = blank_logprobs.contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()

        loss_batch, alpha = _rnnt_loss_fwd(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            fastemit_lambda=fastemit_lambda,
        )

        ctx.save_for_backward(target_logprobs, blank_logprobs, alpha, src_lengths, tgt_lengths)
        ctx.fastemit_lambda = fastemit_lambda
        return loss_batch

    @staticmethod
    def backward(ctx, grad_rnnt_loss):
        """
        Backward pass for RNN-T loss
        """
        (target_logprobs, blank_logprobs, alpha, src_lengths, tgt_lengths) = ctx.saved_tensors
        target_logprobs_grad, blank_logprobs_grad = _rnnt_loss_bwd(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            alpha=alpha,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            grad_rnnt_loss=grad_rnnt_loss.to(alpha.dtype).contiguous(),
            fastemit_lambda=ctx.fastemit_lambda,
        )
        return target_logprobs_grad, blank_logprobs_grad, None, None, None


class TritonRnntLossWithClampFunction(torch.autograd.Function):
    """
    RNN-T loss calculation from logits with clamping of the gradient for the logits.

    The gradient of each per-utterance loss is clamped before it is multiplied by the upstream gradient
    (as in `warprnnt_numba`), so the clamping does not depend on the reduction of the loss and the batch size.
    The threshold for the gradient is known only if both steps (logits -> log-probs -> loss) are in the same
    autograd function. The function uses the same kernels as `RnntLogProbs` followed by `TritonRnntLossFunction`.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_lengths: torch.Tensor,
        blank_id: int,
        fastemit_lambda: float,
        clamp: float,
    ):
        """
        Args:
            ctx: ctx object for storing the context
            logits: Joint tensor of size [B, T, U+1, D], raw logits (not after log-softmax)
            targets: targets of size [B, U]
            src_lengths: source lengths of size [B]
            tgt_lengths: target lengths of size [B]
            blank_id: blank index
            fastemit_lambda: Float scaling factor for FastEmit regularization.
            clamp: positive threshold, the gradient of the per-utterance loss is clamped to [-clamp, clamp]

        Returns:
            loss of size [B], see `TritonRnntLossFunction`
        """
        if clamp <= 0.0:
            raise ValueError(f"`clamp` must be positive, got {clamp}")
        _check_logits(logits=logits, blank_id=blank_id)
        targets = targets.contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()

        target_logprobs, blank_logprobs, shifted_log_sum_exp = _rnnt_logprobs_fwd(
            logits=logits, targets=targets, blank_id=blank_id, src_lengths=src_lengths, tgt_lengths=tgt_lengths
        )
        loss_batch, alpha = _rnnt_loss_fwd(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            fastemit_lambda=fastemit_lambda,
        )

        ctx.save_for_backward(
            logits, targets, src_lengths, tgt_lengths, shifted_log_sum_exp, target_logprobs, blank_logprobs, alpha
        )
        ctx.blank_id = blank_id
        ctx.fastemit_lambda = fastemit_lambda
        ctx.clamp = clamp
        return loss_batch

    @staticmethod
    def backward(ctx, grad_rnnt_loss):
        """
        Backward pass: gradient for the logits, clamped to [-clamp, clamp] before applying the upstream gradient
        """
        (logits, targets, src_lengths, tgt_lengths, shifted_log_sum_exp, target_logprobs, blank_logprobs, alpha) = (
            ctx.saved_tensors
        )
        grad_rnnt_loss = grad_rnnt_loss.to(alpha.dtype).contiguous()
        target_logprobs_grad, blank_logprobs_grad = _rnnt_loss_bwd(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            alpha=alpha,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            grad_rnnt_loss=grad_rnnt_loss,
            fastemit_lambda=ctx.fastemit_lambda,
        )
        # The gradients above are already multiplied by the upstream gradient: clamping `grad / upstream`
        # to [-clamp, clamp] is equivalent to clamping `grad` to [-clamp * |upstream|, clamp * |upstream|]
        grad_logits = _rnnt_logprobs_bwd(
            logits=logits,
            targets=targets,
            blank_id=ctx.blank_id,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            shifted_log_sum_exp=shifted_log_sum_exp,
            grad_target_scores=target_logprobs_grad,
            grad_blank_scores=blank_logprobs_grad,
            grad_clamp=ctx.clamp * grad_rnnt_loss.abs(),
        )
        return grad_logits, None, None, None, None, None, None


def rnnt_loss_from_logprobs_triton(
    target_logprobs: torch.Tensor,
    blank_logprobs: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
    fastemit_lambda: float = 0.0,
) -> torch.Tensor:
    """
    RNN-T loss in Triton

    Args:
        target_logprobs: target log probabilities of size [B, T, U+1] (padded)
        blank_logprobs: blank log probabilities of size [B, T, U+1]
        src_lengths: source lengths of size [B]
        tgt_lengths: target lengths of size [B]
        fastemit_lambda: Float scaling factor for FastEmit regularization. Default 0.0 (disabled).
    Returns:
        tensor of size [B] with RNN-T loss
    """
    loss_batch = TritonRnntLossFunction.apply(
        target_logprobs, blank_logprobs, src_lengths, tgt_lengths, fastemit_lambda
    )
    return loss_batch


def rnnt_loss_triton(
    blank_id: int,
    logits: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
    fastemit_lambda: float = 0.0,
    clamp: float = 0.0,
) -> torch.Tensor:
    """
    RNN-T loss in Triton

    Args:
        blank_id: blank index
        logits: Joint tensor of size [B, T, U+1, D], raw logits (not after log-softmax)
        targets: targets of size [B, U]
        src_lengths: source lengths of size [B]
        tgt_lengths: target lengths of size [B]
        fastemit_lambda: Float scaling factor for FastEmit regularization. Default 0.0 (disabled).
        clamp: when positive, the gradient of each per-utterance loss for the logits is clamped
            to [-clamp, clamp] (before multiplying by the upstream gradient). Default 0.0 (disabled).
    Returns:
        tensor of size [B] with RNN-T loss
    """
    if clamp > 0.0:
        return TritonRnntLossWithClampFunction.apply(
            logits, targets, src_lengths, tgt_lengths, blank_id, fastemit_lambda, clamp
        )
    target_logprobs, blank_logprobs = rnnt_logprobs_triton(
        logits=logits,
        targets=targets,
        blank_id=blank_id,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )
    loss_batch = rnnt_loss_from_logprobs_triton(
        target_logprobs=target_logprobs,
        blank_logprobs=blank_logprobs,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
        fastemit_lambda=fastemit_lambda,
    )
    return loss_batch


def _rnnt_loss_fwd(
    target_logprobs: torch.Tensor,
    blank_logprobs: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
    fastemit_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Launch the forward kernel for contiguous inputs;
    returns the loss of size [B] and alpha (forward variables) of size [B, T, U+1] for backward.
    """
    use_fp64 = target_logprobs.dtype == torch.float64
    float_dtype = torch.float64 if use_fp64 else torch.float32
    batch_size, src_max_length, tgt_max_length_plus_1 = target_logprobs.shape

    # the kernel reads only alpha values that are written before, no need to initialize
    alpha = torch.empty(
        [batch_size, src_max_length, tgt_max_length_plus_1], dtype=float_dtype, device=target_logprobs.device
    )
    loss_batch = torch.empty([batch_size], dtype=float_dtype, device=target_logprobs.device)

    with cuda_device_guard(target_logprobs):
        _rnnt_fwd_kernel[(batch_size,)](
            target_logprobs_ptr=target_logprobs,
            blank_logprobs_ptr=blank_logprobs,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            alpha_out_ptr=alpha,
            loss_batch_out_ptr=loss_batch,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            BLOCK_SIZE=_get_diagonal_block_size(src_max_length, tgt_max_length_plus_1),
            PARALLELIZE_OVER_SRC=src_max_length <= tgt_max_length_plus_1,
            USE_FP64=use_fp64,
        )

    # FastEmit regularization: loss = (1 + fastemit_lambda) * base_loss
    if fastemit_lambda != 0.0:
        loss_batch = loss_batch * (1.0 + fastemit_lambda)
    return loss_batch, alpha


def _rnnt_loss_bwd(
    target_logprobs: torch.Tensor,
    blank_logprobs: torch.Tensor,
    alpha: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
    grad_rnnt_loss: torch.Tensor,
    fastemit_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Launch the backward kernel; returns the gradients for target and blank log-probs, both of size [B, T, U+1].
    `grad_rnnt_loss` is the (contiguous) upstream gradient of size [B] in the dtype of `alpha`.
    """
    batch_size, src_max_length, tgt_max_length_plus_1 = target_logprobs.shape
    target_logprobs_grad = torch.zeros(
        [batch_size, src_max_length, tgt_max_length_plus_1], dtype=alpha.dtype, device=target_logprobs.device
    )
    blank_logprobs_grad = torch.zeros(
        [batch_size, src_max_length, tgt_max_length_plus_1], dtype=alpha.dtype, device=target_logprobs.device
    )

    with cuda_device_guard(target_logprobs):
        _rnnt_bwd_kernel[(batch_size,)](
            target_logprobs_ptr=target_logprobs,
            blank_logprobs_ptr=blank_logprobs,
            alpha_ptr=alpha,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            grad_rnnt_loss_ptr=grad_rnnt_loss,
            target_logprobs_grad_out_ptr=target_logprobs_grad,
            blank_logprobs_grad_out_ptr=blank_logprobs_grad,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            BLOCK_SIZE=_get_diagonal_block_size(src_max_length, tgt_max_length_plus_1),
            PARALLELIZE_OVER_SRC=src_max_length <= tgt_max_length_plus_1,
            USE_FP64=alpha.dtype == torch.float64,
        )

    # FastEmit: scale emit (target) gradients by (1 + fastemit_lambda), blank unchanged
    if fastemit_lambda != 0.0:
        target_logprobs_grad = target_logprobs_grad * (1.0 + fastemit_lambda)
    return target_logprobs_grad, blank_logprobs_grad


def _get_diagonal_block_size(src_max_length: int, tgt_max_length_plus_1: int) -> int:
    """
    Block size to process a diagonal of the RNN-T lattice.
    The minimum block size is limited to avoid recompiling the kernels for each small shape.
    """
    return max(_MIN_DIAGONAL_BLOCK_SIZE, triton.next_power_of_2(min(src_max_length, tgt_max_length_plus_1)))
