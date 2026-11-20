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

from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency_triton import _kl_div_bwd_kernel, _kl_div_fwd_kernel
from nemo.collections.asr.parts.turbo_transducer.types import TransducerLossInputs
from nemo.collections.asr.parts.turbo_transducer.utils_triton import cuda_device_guard


# Shape-dependent integer arguments are not specialized to avoid recompilation for each new batch shape
# (Triton specializes integers equal to 1 and divisible by 16 by default).
@triton.jit(do_not_specialize=["max_source_len", "max_target_len_plus_1", "num_targets", "blank_id"])
def _rnnt_logprobs_fwd_kernel(
    logits_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    max_source_len: int,
    max_target_len_plus_1: int,
    num_labels: int,  # vocab size (with blank)
    blank_id: int,
    num_targets: int,  # size of the last dimension of (contiguous) targets
    target_scores_out_ptr,
    blank_scores_out_ptr,
    shifted_log_sum_exp_out_ptr,  # log-sum-exp of logits shifted by the maximum
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Forward kernel for RNN-T log probs. Stores result in `target_scores_out_ptr` and `blank_scores_out_ptr`.
    Calculations are performed in float32 or float64 based on USE_FP64.

    The kernel does not access memory outside the tensors for invalid inputs: target lengths are limited
    by the size of the lattice; the target score is NaN if the target is outside the targets tensor
    or outside the vocabulary.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    # load lengths for source/target
    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.minimum(tl.load(tgt_lengths_ptr + batch_i), max_target_len_plus_1 - 1)

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
    # stable log softmax calculation; NB: all values are computed relative to the maximum logit,
    # adding the maximum to log-sum-exp loses precision for logits with large magnitude
    logits_max = tl.max(logits, axis=0)
    logits_minus_max = logits - logits_max
    shifted_log_sum_exp = tl.log(tl.sum(tl.exp(logits_minus_max), axis=0))
    blank_logit = tl.load(logits_ptr + blank_id).to(compute_dtype)

    tl.store(blank_scores_out_ptr + flat_index_grid, (blank_logit - logits_max) - shifted_log_sum_exp)
    tl.store(shifted_log_sum_exp_out_ptr + flat_index_grid, shifted_log_sum_exp)

    # calculate log prob for target if needed
    if target_i < target_len:
        target_id = tl.load(targets_ptr + batch_i * num_targets + target_i, mask=target_i < num_targets, other=-1)
        target_id_valid = (target_id >= 0) & (target_id < num_labels)
        target_logit = tl.load(logits_ptr + target_id, mask=target_id_valid, other=float("nan")).to(compute_dtype)
        tl.store(target_scores_out_ptr + flat_index_grid, (target_logit - logits_max) - shifted_log_sum_exp)


@triton.jit(do_not_specialize=["max_source_len", "max_target_len_plus_1", "num_targets", "blank_id"])
def _rnnt_logprobs_bwd_kernel(
    logits_ptr,
    grad_logits_out_ptr,
    targets_ptr,
    shifted_log_sum_exp_ptr,  # log-sum-exp of logits shifted by the maximum (from forward)
    src_lengths_ptr,
    tgt_lengths_ptr,
    max_source_len: int,
    max_target_len_plus_1: int,
    num_labels: int,
    blank_id: int,
    num_targets: int,  # size of the last dimension of (contiguous) targets
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_clamp_ptr,  # per-utterance thresholds for gradient clamping of size [B], used only if USE_GRAD_CLAMP
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_GRAD_CLAMP: tl.constexpr,
):
    """
    Backward kernel for RNN-T log probs. Stores result in `grad_logits_out_ptr`.
    We recalculate part of the forward here to avoid using extra memory in forward.
    The kernel writes the full gradient tensor (zeros for positions outside the lengths or with zero upstream
    gradient), so the output buffer does not require initialization.
    Calculations are performed in float32 or float64 based on USE_FP64.
    If USE_GRAD_CLAMP is set, each element of the gradient for the utterance `b` is clamped to
    `[-grad_clamp[b], grad_clamp[b]]` (NaN values are preserved); otherwise `grad_clamp_ptr` is not used.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    flat_index_grid = (batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i
    # offset in [B, T, U+1, V] tensor for the current vector with grad_logits
    grad_logits_out_ptr += flat_index_grid * num_labels
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels

    # load lengths for source/target; target length is limited by the size of the lattice (see forward kernel)
    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.minimum(tl.load(tgt_lengths_ptr + batch_i), max_target_len_plus_1 - 1)
    if source_i >= source_len or target_i > target_len:
        # no calculations required, gradient is zero
        tl.store(grad_logits_out_ptr + col_offsets, tl.zeros([BLOCK_SIZE], dtype=tl.float32), mask=mask)
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32

    blank_grad = tl.load(grad_blank_scores_ptr + flat_index_grid).to(compute_dtype)
    target_i_valid = target_i < target_len
    target_grad = tl.load(grad_target_scores_ptr + flat_index_grid, mask=target_i_valid, other=0.0).to(compute_dtype)
    if blank_grad == 0.0 and target_grad == 0.0:
        tl.store(grad_logits_out_ptr + col_offsets, tl.zeros([BLOCK_SIZE], dtype=tl.float32), mask=mask)
        return

    logits_ptr += flat_index_grid * num_labels
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(compute_dtype)
    # stable log softmax calculation (relative to the maximum logit, see forward kernel)
    shifted_log_sum_exp = tl.load(shifted_log_sum_exp_ptr + flat_index_grid)
    log_softmax = (logits - tl.max(logits, axis=0)) - shifted_log_sum_exp
    # softmax for gradient
    softmax = tl.exp(log_softmax)

    target_id = tl.load(
        targets_ptr + batch_i * num_targets + target_i, mask=target_i_valid & (target_i < num_targets), other=-1
    )

    grad_not_in_targets = (-softmax) * (blank_grad + target_grad)
    grad = tl.where(col_offsets == blank_id, blank_grad + grad_not_in_targets, grad_not_in_targets)
    grad = tl.where(col_offsets == target_id, target_grad + grad_not_in_targets, grad)
    if USE_GRAD_CLAMP:
        grad_clamp = tl.load(grad_clamp_ptr + batch_i).to(compute_dtype)
        grad = tl.clamp(grad, -grad_clamp, grad_clamp, propagate_nan=tl.PropagateNan.ALL)
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
            Tuple of tensors with log probabilities for targets and blank labels, both of size [B, T, U+1]
        """
        _check_logits(logits=logits, blank_id=blank_id)
        batch_size, max_source_len, max_target_len_plus_1, _ = logits.shape
        device = logits.device
        if src_lengths is None:
            src_lengths = torch.full([batch_size], fill_value=max_source_len, dtype=torch.long, device=device)
        if tgt_lengths is None:
            tgt_lengths = torch.full(
                [batch_size], fill_value=max_target_len_plus_1 - 1, dtype=torch.long, device=device
            )
        targets = targets.contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()

        target_scores, blank_scores, shifted_log_sum_exp = _rnnt_logprobs_fwd(
            logits=logits, targets=targets, blank_id=blank_id, src_lengths=src_lengths, tgt_lengths=tgt_lengths
        )

        # saving for backward
        ctx.save_for_backward(logits, targets, src_lengths, tgt_lengths, shifted_log_sum_exp)
        ctx.blank_id = blank_id
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
        (logits, targets, src_lengths, tgt_lengths, shifted_log_sum_exp) = ctx.saved_tensors
        grad_logits = _rnnt_logprobs_bwd(
            logits=logits,
            targets=targets,
            blank_id=ctx.blank_id,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            shifted_log_sum_exp=shifted_log_sum_exp,
            grad_target_scores=grad_target_scores,
            grad_blank_scores=grad_blank_scores,
        )
        return grad_logits, None, None, None, None


class RnntLogProbsWithConsistency(torch.autograd.Function):
    """
    Function to calculate log probabilities for target and blank labels for RNN-T together with the
    per-position consistency (KL) loss, supporting torch.autograd.

    The logits are expected to represent a double batch: rows `[0, B)` are the teacher pass, rows
    `[B, 2B)` are the student pass (the same utterances with different augmentations). Both losses are
    computed from the same logits tensor, and the backward pass fills a single gradient buffer, which
    avoids materializing separate (full-size) gradients for each loss.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        blank_id: int,
        src_lengths: torch.Tensor,
        tgt_lengths: torch.Tensor,
        consistency_mask: torch.Tensor,
        consistency_symmetrical: bool,
    ):
        """
        Args:
            ctx: ctx object for storing the context
            logits: Joint tensor of size [2B, T, U+1, V] (double batch: teacher first, student second)
            targets: Targets of size [2B, U]
            blank_id: id of the blank output
            src_lengths: tensor with lengths for source utterances, [2B]
            tgt_lengths: tensor with lengths for targets, [2B]
            consistency_mask: mask for consistency loss positions, [B, T, U+1]
            consistency_symmetrical: if True, compute symmetric KL divergence

        Returns:
            Tuple (target log-probs [2B, T, U+1], blank log-probs [2B, T, U+1], consistency KL [B, T, U+1])
        """
        _check_logits(logits=logits, blank_id=blank_id)
        batch_size, max_source_len, max_target_len_plus_1, num_labels = logits.shape
        if batch_size % 2 != 0:
            raise ValueError(f"Consistency loss requires a double batch, got odd batch size {batch_size}")
        half_batch_size = batch_size // 2

        targets = targets.contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()
        consistency_mask = consistency_mask.contiguous()

        target_scores, blank_scores, shifted_log_sum_exp = _rnnt_logprobs_fwd(
            logits=logits, targets=targets, blank_id=blank_id, src_lengths=src_lengths, tgt_lengths=tgt_lengths
        )

        use_fp64 = logits.dtype == torch.float64
        consistency_kl = torch.zeros(
            [half_batch_size, max_source_len, max_target_len_plus_1],
            dtype=torch.float64 if use_fp64 else torch.float32,
            device=logits.device,
        )
        with cuda_device_guard(logits):
            _kl_div_fwd_kernel[(half_batch_size, max_source_len, max_target_len_plus_1)](
                teacher_logits_ptr=logits[:half_batch_size],
                student_logits_ptr=logits[half_batch_size:],
                mask_ptr=consistency_mask,
                max_source_len=max_source_len,
                max_target_len_plus_1=max_target_len_plus_1,
                num_labels=num_labels,
                kl_loss_out_ptr=consistency_kl,
                BLOCK_SIZE=triton.next_power_of_2(num_labels),
                USE_FP64=use_fp64,
                SYMMETRIC=consistency_symmetrical,
            )

        # saving for backward
        ctx.save_for_backward(logits, targets, src_lengths, tgt_lengths, shifted_log_sum_exp, consistency_mask)
        ctx.blank_id = blank_id
        ctx.consistency_symmetrical = consistency_symmetrical
        return target_scores, blank_scores, consistency_kl

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores, grad_consistency_kl):
        """
        Backward calculation for RNN-T log-probs and consistency loss.

        Both gradients are accumulated in a single buffer for the logits, which is the main purpose
        of fusing the two losses into one autograd function.

        Args:
            ctx: ctx object for storing the context
            grad_target_scores: upstream gradient for targets
            grad_blank_scores: upstream gradient for blank scores
            grad_consistency_kl: upstream gradient for the per-position consistency loss

        Returns:
            gradient for logits, None for all other arguments for `forward`
        """
        (logits, targets, src_lengths, tgt_lengths, shifted_log_sum_exp, consistency_mask) = ctx.saved_tensors
        batch_size, max_source_len, max_target_len_plus_1, num_labels = logits.shape
        half_batch_size = batch_size // 2

        # NB: the RNN-T part is computed first (it writes the whole gradient), the consistency part accumulates
        grad_logits = _rnnt_logprobs_bwd(
            logits=logits,
            targets=targets,
            blank_id=ctx.blank_id,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            shifted_log_sum_exp=shifted_log_sum_exp,
            grad_target_scores=grad_target_scores,
            grad_blank_scores=grad_blank_scores,
        )
        with cuda_device_guard(logits):
            _kl_div_bwd_kernel[(half_batch_size, max_source_len, max_target_len_plus_1)](
                teacher_logits_ptr=logits[:half_batch_size],
                student_logits_ptr=logits[half_batch_size:],
                grad_kl_loss_ptr=grad_consistency_kl.contiguous(),
                mask_ptr=consistency_mask,
                max_source_len=max_source_len,
                max_target_len_plus_1=max_target_len_plus_1,
                num_labels=num_labels,
                teacher_grad_out_ptr=grad_logits[:half_batch_size],
                student_grad_out_ptr=grad_logits[half_batch_size:],
                BLOCK_SIZE=triton.next_power_of_2(num_labels),
                USE_FP64=logits.dtype == torch.float64,
                SYMMETRIC=ctx.consistency_symmetrical,
                ACCUMULATE=True,
            )
        return grad_logits, None, None, None, None, None, None


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
        For invalid targets (outside the vocabulary or the targets tensor) output is NaN.
    """
    return RnntLogProbs.apply(logits, targets, blank_id, src_lengths, tgt_lengths)


def rnnt_logprobs_with_consistency_triton(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
    consistency_mask: torch.Tensor,
    consistency_symmetrical: bool = True,
) -> TransducerLossInputs:
    """
    Given logits for a double batch, calculate log probabilities for blank and target labels along with
    the per-position consistency (KL) loss between the halves of the batch. Optimized implementation in Triton.

    Args:
        logits: Joint tensor of size [2B, T, U+1, V] (teacher first half, student second half)
        targets: Targets of size [2B, U]
        blank_id: id of the blank output
        src_lengths: tensor with lengths for source utterances, [2B]
        tgt_lengths: tensor with lengths for targets, [2B]
        consistency_mask: mask for consistency loss positions, [B, T, U+1]
        consistency_symmetrical: if True, compute symmetric KL divergence

    Returns:
        `TransducerLossInputs` with target/blank log probabilities of size [2B, T, U+1] and the
        consistency loss of size [B, T, U+1].
    """
    return TransducerLossInputs(
        *RnntLogProbsWithConsistency.apply(
            logits, targets, blank_id, src_lengths, tgt_lengths, consistency_mask, consistency_symmetrical
        )
    )


def _check_logits(logits: torch.Tensor, blank_id: int):
    """Validate the Joint output before launching the kernels."""
    if logits.dim() != 4:
        raise ValueError(f"Expected logits of shape [B, T, U+1, V], got shape {list(logits.shape)}")
    # logits are huge, so we do not make them contiguous implicitly
    if not logits.is_contiguous():
        raise ValueError("Logits must be contiguous")
    num_labels = logits.shape[-1]
    if blank_id < 0 or blank_id >= num_labels:
        raise ValueError(
            f"blank_id ({blank_id}) must be in range [0, {num_labels - 1}] for logits with {num_labels} labels."
        )


def _rnnt_logprobs_fwd(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Launch the forward kernel; returns target log-probs, blank log-probs,
    and log-sum-exp of logits shifted by the maximum (for backward).
    """
    batch_size, max_source_len, max_target_len_plus_1, num_labels = logits.shape
    float_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    target_scores = torch.zeros(logits.shape[:-1], dtype=float_dtype, device=logits.device)
    blank_scores = torch.zeros_like(target_scores)
    shifted_log_sum_exp = torch.empty_like(target_scores)
    with cuda_device_guard(logits):
        _rnnt_logprobs_fwd_kernel[(batch_size, max_source_len, max_target_len_plus_1)](
            logits_ptr=logits,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            max_source_len=max_source_len,
            max_target_len_plus_1=max_target_len_plus_1,
            num_labels=num_labels,
            blank_id=blank_id,
            num_targets=targets.shape[1],
            target_scores_out_ptr=target_scores,
            blank_scores_out_ptr=blank_scores,
            shifted_log_sum_exp_out_ptr=shifted_log_sum_exp,
            BLOCK_SIZE=triton.next_power_of_2(num_labels),
            USE_FP64=float_dtype == torch.float64,
        )
    return target_scores, blank_scores, shifted_log_sum_exp


def _rnnt_logprobs_bwd(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
    shifted_log_sum_exp: torch.Tensor,
    grad_target_scores: torch.Tensor,
    grad_blank_scores: torch.Tensor,
    grad_clamp: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Launch the backward kernel; returns the gradient for logits.
    If `grad_clamp` (per-utterance thresholds of size [B]) is passed, the gradient for the utterance `b`
    is clamped to `[-grad_clamp[b], grad_clamp[b]]`.
    """
    batch_size, max_source_len, max_target_len_plus_1, num_labels = logits.shape
    # the kernel writes all the values (including zeros), no need to initialize the gradient
    grad_logits = torch.empty_like(logits)
    with cuda_device_guard(logits):
        _rnnt_logprobs_bwd_kernel[(batch_size, max_source_len, max_target_len_plus_1)](
            logits_ptr=logits,
            grad_logits_out_ptr=grad_logits,
            targets_ptr=targets,
            shifted_log_sum_exp_ptr=shifted_log_sum_exp,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            max_source_len=max_source_len,
            max_target_len_plus_1=max_target_len_plus_1,
            num_labels=num_labels,
            blank_id=blank_id,
            num_targets=targets.shape[1],
            # upstream gradients can be non-contiguous (e.g., expanded after `sum()`)
            grad_target_scores_ptr=grad_target_scores.contiguous(),
            grad_blank_scores_ptr=grad_blank_scores.contiguous(),
            grad_clamp_ptr=grad_clamp,
            BLOCK_SIZE=triton.next_power_of_2(num_labels),
            USE_FP64=logits.dtype == torch.float64,
            USE_GRAD_CLAMP=grad_clamp is not None,
        )
    return grad_logits
