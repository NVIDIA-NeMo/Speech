# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


@triton.jit
def _rnnt_logprobs_fwd_kernel(
    logits_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    max_source_len: int,
    max_target_len_plus_1: int,
    num_labels: int,  # vocab size (with blank)
    blank_id: int,
    target_scores_out_ptr,
    blank_scores_out_ptr,
    log_sum_exp_scores_out_ptr,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Forward kernel for RNN-T log probs. Stores result in `target_scores_out_ptr` and `blank_scores_out_ptr`.
    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    # load lengths for source/target
    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.load(tgt_lengths_ptr + batch_i)

    if source_i >= source_len or target_i > target_len:
        # no calculations required
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32

    # calculate offset in [B, T, U+1, V] tensor for the current vector with target logits
    flat_index_grid = (batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i
    flat_index_logits = flat_index_grid * num_labels
    logits_ptr += flat_index_logits
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(compute_dtype)
    # stable log softmax calculation
    logits_max = tl.max(logits, axis=0)
    logits_minus_max = logits - logits_max
    log_sum_exp = tl.log(tl.sum(tl.exp(logits_minus_max), axis=0)) + logits_max
    blank_logit = tl.load(logits_ptr + blank_id).to(compute_dtype)

    tl.store(blank_scores_out_ptr + flat_index_grid, blank_logit - log_sum_exp)
    tl.store(log_sum_exp_scores_out_ptr + flat_index_grid, log_sum_exp)

    # calculate log prob for target if needed
    if target_i < target_len:
        target_id = tl.load(targets_ptr + batch_i * (max_target_len_plus_1 - 1) + target_i)
        target_logit = tl.load(logits_ptr + target_id).to(compute_dtype)
        tl.store(target_scores_out_ptr + flat_index_grid, target_logit - log_sum_exp)


@triton.jit
def _rnnt_logprobs_bwd_kernel(
    logits_ptr,
    grad_logits_out_ptr,
    targets_ptr,
    log_sum_exp_scores_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    max_source_len: int,
    max_target_len_plus_1: int,
    num_labels: int,
    blank_id: int,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Backward kernel for RNN-T log probs. Stores result in `grad_target_scores_ptr` and `grad_blank_scores_ptr`.
    We recalculate part of the forward here to avoid using extra memory in forward.
    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    # load lengths for source/target
    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.load(tgt_lengths_ptr + batch_i)
    if source_i >= source_len or target_i > target_len:
        # no calculations required
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32

    # calculate offset in [B, T, U+1, V] tensor for the current vector with target logits/grad_logits
    flat_index_grid = (batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i
    flat_index_logits = ((batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i) * num_labels
    logits_ptr += flat_index_logits
    grad_logits_out_ptr += flat_index_logits

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(compute_dtype)
    # stable log softmax calculation
    log_sum_exp = tl.load(log_sum_exp_scores_ptr + flat_index_grid)
    log_softmax = logits - log_sum_exp
    # softmax for gradient
    softmax = tl.exp(log_softmax)

    blank_grad = tl.load(grad_blank_scores_ptr + flat_index_grid).to(compute_dtype)
    target_i_valid = target_i < target_len
    target_grad = tl.load(grad_target_scores_ptr + flat_index_grid, mask=target_i_valid, other=0.0).to(compute_dtype)
    target_id = tl.load(targets_ptr + batch_i * (max_target_len_plus_1 - 1) + target_i, mask=target_i_valid, other=-1)

    grad_not_in_targets = (-softmax) * (blank_grad + target_grad)
    grad = tl.where(col_offsets == blank_id, blank_grad + grad_not_in_targets, grad_not_in_targets)
    grad = tl.where(col_offsets == target_id, target_grad + grad_not_in_targets, grad)
    tl.store(grad_logits_out_ptr + col_offsets, grad, mask=mask)


class RnntLogProbs(torch.autograd.Function):
    """
    Function to calculate log probabilities for target and blank labels for RNN-T, supporting torch.autograd.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        blank_id: int,
        src_lengths: torch.Tensor | None,
        tgt_lengths: torch.Tensor | None,
    ):
        """

        Args:
            ctx: ctx object for storing the context
            logits: Joint tensor of size [B, T, U+1, D]
            targets: Targets of size [B, U]
            blank_id: id of the blank output
            src_lengths: optional tensor with lengths for source utterances
            tgt_lengths: optional tensor with lengths for targets

        Returns:

        """
        assert logits.is_contiguous()  # logits are huge, so here we just check if logits are contiguous
        targets = targets.contiguous()
        device = logits.device
        num_labels = logits.shape[-1]
        if blank_id < 0 or blank_id >= num_labels:
            raise ValueError(
                f"blank_id ({blank_id}) must be in range [0, {num_labels - 1}] for logits with {num_labels} labels."
            )
        # Use float64 if input is float64, otherwise float32
        use_fp64 = logits.dtype == torch.float64
        float_dtype = torch.float64 if use_fp64 else torch.float32

        target_scores = torch.zeros(logits.shape[:-1], dtype=float_dtype, device=device)
        blank_scores = torch.zeros_like(target_scores)
        log_sum_exp_scores = torch.empty_like(target_scores)
        if src_lengths is None:
            src_lengths = torch.full([logits.shape[0]], fill_value=logits.shape[1], dtype=torch.int, device=device)
        else:
            src_lengths = src_lengths.contiguous()
        if tgt_lengths is None:
            tgt_lengths = torch.full(
                [logits.shape[0]], fill_value=logits.shape[2] - 1, dtype=torch.int, device=device
            )
        else:
            tgt_lengths = tgt_lengths.contiguous()

        # run Triton kernel
        _rnnt_logprobs_fwd_kernel[(logits.shape[0], logits.shape[1], logits.shape[2])](
            logits_ptr=logits,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            max_source_len=logits.shape[1],
            max_target_len_plus_1=logits.shape[2],
            num_labels=num_labels,
            blank_id=blank_id,
            target_scores_out_ptr=target_scores,
            blank_scores_out_ptr=blank_scores,
            log_sum_exp_scores_out_ptr=log_sum_exp_scores,
            BLOCK_SIZE=triton.next_power_of_2(logits.shape[-1]),
            USE_FP64=use_fp64,
        )

        # saving for backward
        ctx.save_for_backward(logits, targets, src_lengths, tgt_lengths, log_sum_exp_scores)
        ctx.blank_id = blank_id
        ctx.use_fp64 = use_fp64
        return target_scores, blank_scores

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        """
        Backward calculation for RNN-T log-probs.

        Args:
            ctx: ctx object for storing the context
            grad_target_scores: upstream gradient for targets
            grad_blank_scores:  upstream gradient for blank scores

        Returns:
            gradient for logits, None for all other arguments for `forward`
        """
        (logits, targets, src_lengths, tgt_lengths, log_sum_exp_scores) = ctx.saved_tensors
        blank_id = ctx.blank_id
        use_fp64 = ctx.use_fp64
        grad_logits = torch.zeros_like(logits)
        _rnnt_logprobs_bwd_kernel[(logits.shape[0], logits.shape[1], logits.shape[2])](
            logits_ptr=logits,
            grad_logits_out_ptr=grad_logits,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            targets_ptr=targets,
            log_sum_exp_scores_ptr=log_sum_exp_scores,
            max_source_len=logits.shape[1],
            max_target_len_plus_1=logits.shape[2],
            num_labels=logits.shape[3],
            blank_id=blank_id,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            BLOCK_SIZE=triton.next_power_of_2(logits.shape[-1]),
            USE_FP64=use_fp64,
        )
        return grad_logits, None, None, None, None


def rnnt_logprobs_triton(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    src_lengths: torch.Tensor | None = None,
    tgt_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given logits, calculate log probabilities for blank and target labels needed for transducer loss calculation.
    Optimized implementation in Triton.

    Args:
        logits: Joint tensor of size [B, T, U+1, D]
        targets: Targets of size [B, U]
        blank_id: id of the blank output
        src_lengths: optional tensor with lengths for source utterances
        tgt_lengths: optional tensor with lengths for targets

    Returns:
        Tuple of tensors with log probabilities for targets and blank labels, both of size [B, T, U+1].
        For the non-existent targets (U+1 or beyond tgt_lengths) output is zero.
    """
    return RnntLogProbs.apply(logits, targets, blank_id, src_lengths, tgt_lengths)
