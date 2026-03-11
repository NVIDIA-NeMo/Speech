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

from nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs_triton import rnnt_logprobs_triton
from nemo.collections.asr.parts.rnnt_triton.utils_triton import log_add_exp


@triton.jit
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
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    NEG_INF = -1e38

    src_len = tl.load(src_lengths_ptr + batch_i).to(tl.int64)
    tgt_len = tl.load(tgt_lengths_ptr + batch_i).to(tl.int64)

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
    loss = -(final_alpha + final_blank_lp)
    tl.store(loss_batch_out_ptr + batch_i, loss)


@triton.jit
def _rnnt_bwd_kernel(
    target_logprobs_ptr,
    blank_logprobs_ptr,
    alpha_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
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
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    NEG_INF = -1e38

    src_len = tl.load(src_lengths_ptr + batch_i).to(tl.int64)
    tgt_len = tl.load(tgt_lengths_ptr + batch_i).to(tl.int64)

    batch_offset = batch_i * max_src_len * max_tgt_len_plus_1

    # Initialize beta[src_len-1, tgt_len] = blank_logprobs[src_len-1, tgt_len]
    final_idx = batch_offset + (src_len - 1) * max_tgt_len_plus_1 + tgt_len
    final_blank_lp = tl.load(blank_logprobs_ptr + final_idx).to(compute_dtype)

    # log_like = alpha[src_len-1, tgt_len] + blank_logprobs[src_len-1, tgt_len]
    final_alpha = tl.load(alpha_ptr + final_idx).to(compute_dtype)
    log_like = final_alpha + final_blank_lp

    # Gradient at final state: blank_grad[src_len-1, tgt_len] = -1.0
    tl.store(blank_logprobs_grad_out_ptr + final_idx, -1.0)

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
            -tl.exp(alpha_val + blank_beta + blank_lp - log_like),
            0.0,
        )
        tl.store(blank_logprobs_grad_out_ptr + cur_idx, blank_grad, mask=mask)

        # target_grad[t, u] = -exp(alpha[t, u] + beta[t, u+1] + target_logprobs[t, u] - log_like)
        target_grad = tl.where(
            emit_mask,
            -tl.exp(alpha_val + emit_beta + emit_lp - log_like),
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
            loss of size [B]
        """
        assert target_logprobs.is_contiguous()
        assert blank_logprobs.is_contiguous()
        use_fp64 = target_logprobs.dtype == torch.float64
        float_dtype = torch.float64 if use_fp64 else torch.float32
        batch_size, src_max_length, tgt_max_length_plus_1 = target_logprobs.shape

        alpha = torch.full(
            [batch_size, src_max_length, tgt_max_length_plus_1],
            fill_value=-1e38,
            dtype=float_dtype,
            device=target_logprobs.device,
        )
        loss_batch = torch.empty([batch_size], dtype=float_dtype, device=target_logprobs.device)

        parallelize_over_src = src_max_length <= tgt_max_length_plus_1
        BLOCK_SIZE = triton.next_power_of_2(min(src_max_length, tgt_max_length_plus_1))
        _rnnt_fwd_kernel[(batch_size,)](
            target_logprobs_ptr=target_logprobs,
            blank_logprobs_ptr=blank_logprobs,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            alpha_out_ptr=alpha,
            loss_batch_out_ptr=loss_batch,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            BLOCK_SIZE=BLOCK_SIZE,
            PARALLELIZE_OVER_SRC=parallelize_over_src,
            USE_FP64=use_fp64,
        )

        # FastEmit regularization: loss = (1 + fastemit_lambda) * base_loss
        if fastemit_lambda != 0.0:
            loss_batch = loss_batch * (1.0 + fastemit_lambda)

        ctx.save_for_backward(target_logprobs, blank_logprobs, alpha, src_lengths, tgt_lengths)
        ctx.use_fp64 = use_fp64
        ctx.fastemit_lambda = fastemit_lambda
        return loss_batch

    @staticmethod
    def backward(ctx, grad_rnnt_loss):
        """
        Backward pass for RNN-T loss
        """
        (target_logprobs, blank_logprobs, alpha, src_lengths, tgt_lengths) = ctx.saved_tensors
        use_fp64 = ctx.use_fp64
        fastemit_lambda = ctx.fastemit_lambda
        float_dtype = torch.float64 if use_fp64 else torch.float32
        batch_size, src_max_length, tgt_max_length_plus_1 = target_logprobs.shape

        target_logprobs_grad = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1],
            dtype=float_dtype,
            device=target_logprobs.device,
        )
        blank_logprobs_grad = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1],
            dtype=float_dtype,
            device=target_logprobs.device,
        )

        parallelize_over_src = src_max_length <= tgt_max_length_plus_1
        BLOCK_SIZE = triton.next_power_of_2(min(src_max_length, tgt_max_length_plus_1))
        _rnnt_bwd_kernel[(batch_size,)](
            target_logprobs_ptr=target_logprobs,
            blank_logprobs_ptr=blank_logprobs,
            alpha_ptr=alpha,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            target_logprobs_grad_out_ptr=target_logprobs_grad,
            blank_logprobs_grad_out_ptr=blank_logprobs_grad,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            BLOCK_SIZE=BLOCK_SIZE,
            PARALLELIZE_OVER_SRC=parallelize_over_src,
            USE_FP64=use_fp64,
        )

        # FastEmit: scale emit (target) gradients by (1 + fastemit_lambda), blank unchanged
        if fastemit_lambda != 0.0:
            target_logprobs_grad = target_logprobs_grad * (1.0 + fastemit_lambda)

        # Multiply by upstream gradient
        grad_rnnt_loss = grad_rnnt_loss.to(float_dtype).view(-1, 1, 1)
        target_logprobs_grad = target_logprobs_grad * grad_rnnt_loss
        blank_logprobs_grad = blank_logprobs_grad * grad_rnnt_loss

        return target_logprobs_grad, blank_logprobs_grad, None, None, None


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
    Returns:
        tensor of size [B] with RNN-T loss
    """
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
