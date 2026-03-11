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

EPS = 1e-5

WEIGHTED_NONE = 0
WEIGHTED_P_NON_BLANK = 1
WEIGHTED_P_NON_BLANK_WITH_GRAD = 2


def _get_weighted_mode(weighted: str | None) -> int:
    if weighted is None:
        return WEIGHTED_NONE
    if weighted == "p_non_blank":
        return WEIGHTED_P_NON_BLANK
    if weighted == "p_non_blank_with_grad":
        return WEIGHTED_P_NON_BLANK_WITH_GRAD
    raise NotImplementedError(f"Unsupported weighted mode: {weighted}")


@triton.jit
def _kl_div_fwd_kernel(
    teacher_logits_ptr,
    student_logits_ptr,
    mask_ptr,
    blank_id: int,
    max_source_len: int,
    max_target_len_plus_1: int,
    num_labels: int,  # vocab size (with blank)
    kl_loss_out_ptr,
    weighted_denominator_out_ptr,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    WEIGHTED_MODE: tl.constexpr,
):
    """
    Forward kernel for KL divergence loss.

    When SYMMETRIC=False: computes KL(P||Q) = sum(P * (log P - log Q))
    When SYMMETRIC=True: computes 0.5 * (KL(P||Q) + KL(Q||P)) = 0.5 * sum((P - Q) * (log P - log Q))

    Calculations are performed in float32 or float64 based on USE_FP64.
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    idx_no_vocab = (batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i
    mask_value = tl.load(mask_ptr + idx_no_vocab)
    if not mask_value:
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32

    # calculate offset in [B, T, U+1, V] tensor
    idx_vocab_start = idx_no_vocab * num_labels
    teacher_logits_ptr += idx_vocab_start
    student_logits_ptr += idx_vocab_start
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels
    teacher_logits = tl.load(teacher_logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(compute_dtype)
    student_logits = tl.load(student_logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(compute_dtype)

    # stable log softmax calculation for teacher
    teacher_logits_max = tl.max(teacher_logits, axis=0)
    teacher_logits_minus_max = teacher_logits - teacher_logits_max
    teacher_denominator = tl.log(tl.sum(tl.exp(teacher_logits_minus_max), axis=0))
    teacher_log_softmax = teacher_logits_minus_max - teacher_denominator
    teacher_softmax = tl.exp(teacher_log_softmax)

    # stable log softmax calculation for student
    student_logits_max = tl.max(student_logits, axis=0)
    student_logits_minus_max = student_logits - student_logits_max
    student_denominator = tl.log(tl.sum(tl.exp(student_logits_minus_max), axis=0))
    student_log_softmax = student_logits_minus_max - student_denominator

    if SYMMETRIC:
        # symmetric KL: 0.5 * sum((P - Q) * (log P - log Q))
        student_softmax = tl.exp(student_log_softmax)
        prob_diff = teacher_softmax - student_softmax
        log_prob_diff = teacher_log_softmax - student_log_softmax
        kl_per_vocab = prob_diff * log_prob_diff
        kl_loss_value = 0.5 * tl.sum(tl.where(mask, kl_per_vocab, 0.0), axis=0)
    else:
        # non-symmetric KL: sum(P * (log P - log Q))
        kl_per_vocab = teacher_softmax * (teacher_log_softmax - student_log_softmax)
        kl_loss_value = tl.sum(tl.where(mask, kl_per_vocab, 0.0), axis=0)

    if WEIGHTED_MODE == 0:
        tl.store(kl_loss_out_ptr + idx_no_vocab, kl_loss_value)
        return

    blank_mask = col_offsets == blank_id
    teacher_blank_prob = tl.sum(tl.where(blank_mask, teacher_softmax, 0.0), axis=0)
    if SYMMETRIC:
        student_softmax = tl.exp(student_log_softmax)
        student_blank_prob = tl.sum(tl.where(blank_mask, student_softmax, 0.0), axis=0)
        weight = 1.0 - 0.5 * (teacher_blank_prob + student_blank_prob)
    else:
        weight = 1.0 - teacher_blank_prob

    tl.store(kl_loss_out_ptr + idx_no_vocab, kl_loss_value * weight)
    tl.store(weighted_denominator_out_ptr + idx_no_vocab, weight)


@triton.jit
def _kl_div_bwd_kernel(
    teacher_logits_ptr,
    student_logits_ptr,
    grad_kl_loss_ptr,
    mask_ptr,
    weighted_denominator_ptr,
    weighted_kl_loss_ptr,
    max_source_len: int,
    max_target_len_plus_1: int,
    num_labels: int,  # vocab size (with blank)
    blank_id: int,
    teacher_grad_out_ptr,
    student_grad_out_ptr,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    WEIGHTED_MODE: tl.constexpr,
    WITH_WEIGHT_GRAD: tl.constexpr,
    STORE_TEACHER_GRAD: tl.constexpr,
    EPSILON: tl.constexpr,
):
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    idx_no_vocab = (batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i
    mask_value = tl.load(mask_ptr + idx_no_vocab)
    if not mask_value:
        return

    compute_dtype = tl.float64 if USE_FP64 else tl.float32

    # calculate offset in [B, T, U+1, V] tensor
    idx_vocab_start = idx_no_vocab * num_labels
    teacher_logits_ptr += idx_vocab_start
    student_logits_ptr += idx_vocab_start
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels
    teacher_logits = tl.load(teacher_logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(compute_dtype)
    student_logits = tl.load(student_logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(compute_dtype)

    # stable log softmax calculation for teacher
    teacher_logits_max = tl.max(teacher_logits, axis=0)
    teacher_logits_minus_max = teacher_logits - teacher_logits_max
    teacher_denominator = tl.log(tl.sum(tl.exp(teacher_logits_minus_max), axis=0))
    teacher_softmax = tl.exp(teacher_logits_minus_max - teacher_denominator)

    # stable log softmax calculation for student
    student_logits_max = tl.max(student_logits, axis=0)
    student_logits_minus_max = student_logits - student_logits_max
    student_denominator = tl.log(tl.sum(tl.exp(student_logits_minus_max), axis=0))
    student_softmax = tl.exp(student_logits_minus_max - student_denominator)

    # compute gradient: (Q - P) where Q = student_softmax, P = teacher_softmax
    prob_diff = student_softmax - teacher_softmax

    if WEIGHTED_MODE == 0:
        upstream_grad = tl.load(grad_kl_loss_ptr + idx_no_vocab).to(compute_dtype)
        if SYMMETRIC:
            # symmetric: student_grad = 0.5 * upstream * (Q - P), teacher_grad = -student_grad
            student_grad = 0.5 * upstream_grad * prob_diff
            teacher_grad = -student_grad
        else:
            # non-symmetric: student_grad = upstream * (Q - P), no teacher gradient
            student_grad = upstream_grad * prob_diff
            teacher_grad = student_grad * 0.0
    else:
        # Recompute KL and weight for weighted modes.
        if SYMMETRIC:
            prob_diff_fwd = teacher_softmax - student_softmax
            log_prob_diff = (teacher_logits_minus_max - teacher_denominator) - (
                student_logits_minus_max - student_denominator
            )
            kl_per_vocab = prob_diff_fwd * log_prob_diff
            kl_loss_value = 0.5 * tl.sum(tl.where(mask, kl_per_vocab, 0.0), axis=0)
        else:
            teacher_log_softmax = teacher_logits_minus_max - teacher_denominator
            student_log_softmax = student_logits_minus_max - student_denominator
            kl_per_vocab = teacher_softmax * (teacher_log_softmax - student_log_softmax)
            kl_loss_value = tl.sum(tl.where(mask, kl_per_vocab, 0.0), axis=0)

        blank_mask = col_offsets == blank_id
        blank_indicator = tl.where(blank_mask, 1.0, 0.0).to(compute_dtype)
        teacher_blank_prob = tl.sum(tl.where(blank_mask, teacher_softmax, 0.0), axis=0)
        if SYMMETRIC:
            student_blank_prob = tl.sum(tl.where(blank_mask, student_softmax, 0.0), axis=0)
            weight = 1.0 - 0.5 * (teacher_blank_prob + student_blank_prob)
        else:
            weight = 1.0 - teacher_blank_prob

        upstream_grad = tl.load(grad_kl_loss_ptr + batch_i).to(compute_dtype)
        denominator = tl.load(weighted_denominator_ptr + batch_i).to(compute_dtype)
        inv_norm = 1.0 / (denominator + EPSILON)
        batch_kl_loss = tl.load(weighted_kl_loss_ptr + batch_i).to(compute_dtype)

        # KL-path gradient scaling
        kl_upstream_grad = upstream_grad * weight * inv_norm
        if SYMMETRIC:
            student_grad = 0.5 * kl_upstream_grad * prob_diff
            teacher_grad = -student_grad
        else:
            student_grad = kl_upstream_grad * prob_diff
            teacher_grad = student_grad * 0.0

        if WITH_WEIGHT_GRAD:
            weight_upstream_grad = upstream_grad * (kl_loss_value - batch_kl_loss) * inv_norm
            if SYMMETRIC:
                teacher_grad += 0.5 * weight_upstream_grad * teacher_blank_prob * (teacher_softmax - blank_indicator)
                student_grad += 0.5 * weight_upstream_grad * student_blank_prob * (student_softmax - blank_indicator)
            else:
                teacher_grad += weight_upstream_grad * teacher_blank_prob * (teacher_softmax - blank_indicator)

    if STORE_TEACHER_GRAD:
        teacher_grad_out_ptr += idx_vocab_start
        tl.store(teacher_grad_out_ptr + col_offsets, teacher_grad, mask=mask)

    student_grad_out_ptr += idx_vocab_start
    tl.store(student_grad_out_ptr + col_offsets, student_grad, mask=mask)


class FusedKLDivTriton(torch.autograd.Function):
    """
    Function to calculate KL divergence for RNN-T, supporting torch.autograd.

    When symmetric=False: computes KL(P||Q), only student receives gradients by default.
    When symmetric=True: computes 0.5 * (KL(P||Q) + KL(Q||P)), both teacher and student receive gradients
    """

    @staticmethod
    def forward(
        ctx,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        mask: torch.Tensor,
        symmetric: bool = False,
        blank_id: int | None = None,
        weighted: str | None = None,
    ):
        """
        Args:
            ctx: ctx object for storing the context
            teacher_logits: Joint tensor of size [B, T, U+1, D]
            student_logits: Joint tensor of size [B, T, U+1, D]
            mask: mask tensor [B, T, U+1]
            symmetric: if True, compute symmetric KL divergence

        Returns:
            loss of size [B, T, U+1] for standard mode,
            or [B] for weighted modes
        """
        assert teacher_logits.is_contiguous()
        assert student_logits.is_contiguous()
        assert teacher_logits.shape == student_logits.shape
        use_fp64 = teacher_logits.dtype == torch.float64
        weighted_mode = _get_weighted_mode(weighted)
        acc_dtype = torch.float64 if use_fp64 else torch.float32

        if weighted_mode == WEIGHTED_NONE:
            kl_loss = teacher_logits.new_zeros(teacher_logits.shape[:-1])
            dummy_weighted = torch.empty([1], device=teacher_logits.device, dtype=acc_dtype)
            _kl_div_fwd_kernel[(teacher_logits.shape[0], teacher_logits.shape[1], teacher_logits.shape[2])](
                teacher_logits_ptr=teacher_logits,
                student_logits_ptr=student_logits,
                mask_ptr=mask,
                blank_id=-1,
                max_source_len=teacher_logits.shape[1],
                max_target_len_plus_1=teacher_logits.shape[2],
                num_labels=teacher_logits.shape[3],
                kl_loss_out_ptr=kl_loss,
                weighted_denominator_out_ptr=dummy_weighted,
                BLOCK_SIZE=triton.next_power_of_2(teacher_logits.shape[-1]),
                USE_FP64=use_fp64,
                SYMMETRIC=symmetric,
                WEIGHTED_MODE=weighted_mode,
            )
            ctx.save_for_backward(teacher_logits, student_logits, mask)
        else:
            if blank_id is None:
                raise ValueError("blank_id is required for weighted KL loss")
            if blank_id < 0 or blank_id >= teacher_logits.shape[-1]:
                raise ValueError(f"blank_id must be in [0, {teacher_logits.shape[-1] - 1}], got {blank_id}")

            weighted_numerator_map = torch.zeros(
                teacher_logits.shape[:-1], device=teacher_logits.device, dtype=acc_dtype
            )
            weighted_denominator_map = torch.zeros_like(weighted_numerator_map)

            _kl_div_fwd_kernel[(teacher_logits.shape[0], teacher_logits.shape[1], teacher_logits.shape[2])](
                teacher_logits_ptr=teacher_logits,
                student_logits_ptr=student_logits,
                mask_ptr=mask,
                blank_id=blank_id,
                max_source_len=teacher_logits.shape[1],
                max_target_len_plus_1=teacher_logits.shape[2],
                num_labels=teacher_logits.shape[3],
                kl_loss_out_ptr=weighted_numerator_map,
                weighted_denominator_out_ptr=weighted_denominator_map,
                BLOCK_SIZE=triton.next_power_of_2(teacher_logits.shape[-1]),
                USE_FP64=use_fp64,
                SYMMETRIC=symmetric,
                WEIGHTED_MODE=weighted_mode,
            )

            # Return per-batch weighted KL values (shape [B]).
            weighted_numerator = weighted_numerator_map.sum(dim=(1, 2))
            weighted_denominator = weighted_denominator_map.sum(dim=(1, 2))
            kl_loss = weighted_numerator / (weighted_denominator + EPS)
            ctx.save_for_backward(teacher_logits, student_logits, mask, weighted_denominator, kl_loss)

        ctx.use_fp64 = use_fp64
        ctx.symmetric = symmetric
        ctx.weighted_mode = weighted_mode
        ctx.blank_id = -1 if blank_id is None else blank_id
        return kl_loss

    @staticmethod
    def backward(ctx, grad_kl_loss):
        """
        Backward calculation for KL divergence loss.

        Args:
            ctx: ctx object for storing the context
            grad_kl_loss: upstream gradient [B, T, U+1] or [B] for weighted modes

        Returns:
            Gradients for teacher_logits (None if not symmetric), student_logits, mask (None), symmetric (None)
        """
        weighted_mode = ctx.weighted_mode
        if weighted_mode == WEIGHTED_NONE:
            teacher_logits, student_logits, mask = ctx.saved_tensors
        else:
            teacher_logits, student_logits, mask, weighted_denominator, weighted_kl_loss = ctx.saved_tensors

        use_fp64 = ctx.use_fp64
        symmetric = ctx.symmetric
        acc_dtype = torch.float64 if use_fp64 else torch.float32

        if weighted_mode == WEIGHTED_NONE:
            weighted_denominator = torch.empty([1], device=teacher_logits.device, dtype=acc_dtype)
            weighted_kl_loss = torch.empty([1], device=teacher_logits.device, dtype=acc_dtype)

        student_grad_logits = torch.zeros_like(student_logits)
        need_teacher_grad = symmetric or weighted_mode == WEIGHTED_P_NON_BLANK_WITH_GRAD
        teacher_grad_logits = torch.zeros_like(teacher_logits) if need_teacher_grad else student_grad_logits  # dummy
        grad_kl_loss = grad_kl_loss.contiguous()

        _kl_div_bwd_kernel[(teacher_logits.shape[0], teacher_logits.shape[1], teacher_logits.shape[2])](
            teacher_logits_ptr=teacher_logits,
            student_logits_ptr=student_logits,
            grad_kl_loss_ptr=grad_kl_loss,
            mask_ptr=mask,
            weighted_denominator_ptr=weighted_denominator,
            weighted_kl_loss_ptr=weighted_kl_loss,
            max_source_len=teacher_logits.shape[1],
            max_target_len_plus_1=teacher_logits.shape[2],
            num_labels=teacher_logits.shape[3],
            blank_id=ctx.blank_id,
            teacher_grad_out_ptr=teacher_grad_logits,
            student_grad_out_ptr=student_grad_logits,
            BLOCK_SIZE=triton.next_power_of_2(teacher_logits.shape[-1]),
            USE_FP64=use_fp64,
            SYMMETRIC=symmetric,
            WEIGHTED_MODE=weighted_mode,
            WITH_WEIGHT_GRAD=weighted_mode == WEIGHTED_P_NON_BLANK_WITH_GRAD,
            STORE_TEACHER_GRAD=need_teacher_grad,
            EPSILON=EPS,
        )

        if need_teacher_grad:
            return teacher_grad_logits, student_grad_logits, None, None, None, None
        else:
            return None, student_grad_logits, None, None, None, None


def kl_loss_triton(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    symmetrical: bool = False,
    blank_id: int | None = None,
    weighted: str | None = None,
) -> torch.Tensor:
    """
    Memory-efficient implementation of kl-div loss for RNN-T in Triton

    Args:
        teacher_logits: Joint tensor of size [B, T, U+1, D]
        student_logits: Joint tensor of size [B, T, U+1, D]
        mask: mask tensor [B, T, U+1]
        symmetrical: if loss is symmetrical

    Returns:
        tensor of size [B, T, U+1] with consistency loss in standard mode,
        or [B] in weighted modes
    """
    weighted_mode = _get_weighted_mode(weighted)
    if symmetrical:
        return FusedKLDivTriton.apply(teacher_logits, student_logits, mask, True, blank_id, weighted)
    if weighted_mode == WEIGHTED_P_NON_BLANK_WITH_GRAD:
        # Non-symmetric weighted-with-grad: teacher receives gradients through the weight.
        return FusedKLDivTriton.apply(teacher_logits, student_logits, mask, False, blank_id, weighted)
    # Non-symmetric: only student receives gradients, teacher is detached.
    return FusedKLDivTriton.apply(teacher_logits.detach(), student_logits, mask, False, blank_id, weighted)
