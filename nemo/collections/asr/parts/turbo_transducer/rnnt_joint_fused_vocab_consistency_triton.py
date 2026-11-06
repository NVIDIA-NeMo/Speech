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

"""
Triton `fused_vocab` Joint with the consistency (KL) loss computed inside the kernels.

The joint hidden tensor is a double batch [2B, T, U+1, hidden]: rows [0, B) are the teacher pass, rows [B, 2B) are the
student pass of the same utterances (identical lengths and targets). The KL divergence at each lattice position needs
both full distributions over the vocabulary, so each program processes *pairs* of rows: a teacher row and the
corresponding student row, offset by the half of the flattened batch. Both rows share each loaded weight tile.

Forward computes target/blank log-probabilities for both rows and the per-position KL without materializing logits.
Backward recomputes logits for both rows and adds the consistency gradient to the RNN-T gradient wrt logits before
the single contraction with the vocabulary projection weight (atomic or chunked algorithm, see
`FusedVocabBackwardMode`). With backward pruning, one target window is selected per teacher/student frame pair, and
the consistency gradient is also restricted to this window (the loss value is computed exactly).
"""

import torch
import triton
import triton.language as tl

from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab_triton import (
    _CHUNKED_BWD_ROWS_MULTIPLE,
    _CHUNKED_BWD_VOCAB_PADDING,
    FusedVocabBackwardMode,
    _addmm_full_precision_,
    _copy_pruned_hidden,
    _get_lattice_offsets,
    _load_lattice_row_grads,
    _load_lattice_rows,
    _normalize_backward_options,
    _select_pruning_starts,
)
from nemo.collections.asr.parts.turbo_transducer.types import TransducerLossInputs
from nemo.collections.asr.parts.turbo_transducer.utils_triton import matmul

# Size of the grad logits buffer (teacher and student rows) used by CHUNKED backward
CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES = 128 * 1024**2
# Logit for padded vocabulary columns in forward: exp underflows to zero, finite to keep differences zero
_PADDED_VOCAB_LOGIT: tl.constexpr = tl.constexpr(-1.0e30)


def _fwd_autotune_configs():
    # Two [PAIR_BLOCK, VOCAB_BLOCK] logits accumulators (teacher, student) => half of the row block of the
    # single-row kernel keeps register pressure similar. (PAIR_BLOCK, HB, VB, num_warps, num_stages)
    candidates = [
        (32, 32, 128, 4, 3),
        (64, 32, 64, 4, 3),
        (64, 32, 128, 4, 3),
        (64, 32, 128, 8, 3),
        (64, 64, 128, 8, 3),
        (128, 32, 64, 8, 3),
        (64, 64, 64, 4, 3),  # best for 1025, 8193 (RTX PRO 6000 Blackwell)
    ]
    return [
        triton.Config({"PAIR_BLOCK": pb, "HIDDEN_BLOCK": hb, "VOCAB_BLOCK": vb}, num_warps=nw, num_stages=ns)
        for pb, hb, vb, nw, ns in candidates
    ]


def _bwd_autotune_configs():
    # (PAIR_BLOCK, HB, VB, num_warps, num_stages)
    candidates = [
        (64, 32, 64, 4, 2),
        (64, 32, 128, 8, 2),
        (64, 32, 256, 8, 2),
        (128, 32, 64, 8, 2),
        (128, 32, 128, 8, 2),
        (128, 32, 128, 8, 1),
        (64, 32, 128, 4, 2),  # best for 1025, 8193 (RTX PRO 6000 Blackwell)
    ]
    return [
        triton.Config({"PAIR_BLOCK": pb, "HIDDEN_BLOCK": hb, "VOCAB_BLOCK": vb}, num_warps=nw, num_stages=ns)
        for pb, hb, vb, nw, ns in candidates
    ]


def _bwd_chunked_grad_logits_autotune_configs():
    # (PAIR_BLOCK, HB, VB, num_warps, num_stages)
    candidates = [
        (32, 64, 128, 4, 4),
        (64, 32, 128, 4, 4),
        (64, 64, 128, 8, 3),
        (64, 32, 256, 8, 3),
        (128, 32, 128, 8, 3),
        (128, 64, 64, 8, 3),
        (64, 32, 128, 4, 3),  # best for 8193 (RTX PRO 6000 Blackwell)
        (64, 32, 64, 4, 4),  # best for 1025 (RTX PRO 6000 Blackwell)
    ]
    return [
        triton.Config(
            {"PAIR_BLOCK": pb, "HIDDEN_BLOCK": hb, "VOCAB_BLOCK": vb, "GROUP_SIZE": 8}, num_warps=nw, num_stages=ns
        )
        for pb, hb, vb, nw, ns in candidates
    ]


@triton.autotune(
    configs=_fwd_autotune_configs(),
    key=["hidden_dim", "vocab_size", "USE_HIGH_PRECISION_MM", "MM_FULL_PRECISION", "SYMMETRICAL"],
    cache_results=True,
)
@triton.jit
def _rnnt_joint_fused_vocab_consistency_fwd_triton_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    target_logprobs_out_ptr,
    blank_logprobs_out_ptr,
    log_sum_exp_out_ptr,
    consistency_kl_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    pair_flattened_size: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    SYMMETRICAL: tl.constexpr,
    USE_HIGH_PRECISION_MM: tl.constexpr,
    MM_FULL_PRECISION: tl.constexpr = False,
):
    """
    Teacher rows are [0, pair_flattened_size) of the flattened batch, student rows are offset by pair_flattened_size.
    Log-softmax and KL accumulators are computed online over vocabulary blocks with running max and exp-sums:
    KL(teacher || student) = sum(p_teacher * (z_teacher - z_student)) - (lse_teacher - lse_student)
    Target and blank logits are computed after the vocabulary loop as dot products with the gathered weight rows:
    extracting them from each block of logits needs more registers than the two logits accumulators leave.
    """
    compute_dtype = tl.float32
    matmul_dtype = (
        tl.bfloat16
        if (
            joint_hidden_ptr.dtype.element_ty == tl.bfloat16
            and (not USE_HIGH_PRECISION_MM)
            and (not MM_FULL_PRECISION)
        )
        else compute_dtype
    )

    pair_start = tl.program_id(axis=0) * PAIR_BLOCK
    teacher_offsets = pair_start + tl.arange(0, PAIR_BLOCK)
    pair_mask = teacher_offsets < pair_flattened_size
    student_offsets = teacher_offsets + pair_flattened_size

    # teacher and student rows have identical lengths and targets => load lattice rows once (also saves registers)
    blank_mask, target_mask, targets = _load_lattice_rows(
        teacher_offsets, pair_mask, targets_ptr, src_lengths_ptr, tgt_lengths_ptr, max_src_len, max_tgt_len_plus_1
    )

    vocab_offsets_in_block = tl.arange(0, VOCAB_BLOCK)
    teacher_max = tl.full([PAIR_BLOCK], value=float("-inf"), dtype=compute_dtype)
    teacher_exp_sum = tl.zeros([PAIR_BLOCK], dtype=compute_dtype)
    teacher_kl_sum = tl.zeros([PAIR_BLOCK], dtype=compute_dtype)
    student_max = tl.full([PAIR_BLOCK], value=float("-inf"), dtype=compute_dtype)
    student_exp_sum = tl.zeros([PAIR_BLOCK], dtype=compute_dtype)
    student_kl_sum = tl.zeros([PAIR_BLOCK], dtype=compute_dtype)

    NUM_HIDDEN_ITERS: tl.constexpr = tl.cdiv(hidden_dim, HIDDEN_BLOCK)
    HIDDEN_RESET: tl.constexpr = NUM_HIDDEN_ITERS * HIDDEN_BLOCK

    teacher_hidden_block_ptr = tl.make_block_ptr(
        base=joint_hidden_ptr,
        shape=(2 * pair_flattened_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(pair_start, 0),
        block_shape=(PAIR_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    student_hidden_block_ptr = tl.make_block_ptr(
        base=joint_hidden_ptr,
        shape=(2 * pair_flattened_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(pair_flattened_size + pair_start, 0),
        block_shape=(PAIR_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(vocab_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(0, 0),
        block_shape=(VOCAB_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    bias_block_ptr = tl.make_block_ptr(
        base=bias_ptr,
        shape=(vocab_size,),
        strides=(1,),
        offsets=(0,),
        block_shape=(VOCAB_BLOCK,),
        order=(0,),
    )

    for vocab_start in tl.range(0, vocab_size, VOCAB_BLOCK):
        bias_chunk = tl.load(bias_block_ptr, boundary_check=(0,)).to(compute_dtype)
        teacher_logits = tl.zeros([PAIR_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]
        student_logits = tl.zeros([PAIR_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]
        for _ in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            # weight tile is shared by teacher and student rows
            weight_chunk_t = tl.load(weight_block_ptr, boundary_check=(0, 1)).T.to(matmul_dtype)
            teacher_hidden_chunk = tl.load(teacher_hidden_block_ptr, boundary_check=(0, 1), cache_modifier=".cv").to(
                matmul_dtype
            )
            teacher_logits += matmul(teacher_hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM)
            student_hidden_chunk = tl.load(student_hidden_block_ptr, boundary_check=(0, 1), cache_modifier=".cv").to(
                matmul_dtype
            )
            student_logits += matmul(student_hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM)

            teacher_hidden_block_ptr = tl.advance(teacher_hidden_block_ptr, (0, HIDDEN_BLOCK))
            student_hidden_block_ptr = tl.advance(student_hidden_block_ptr, (0, HIDDEN_BLOCK))
            weight_block_ptr = tl.advance(weight_block_ptr, (0, HIDDEN_BLOCK))

        teacher_hidden_block_ptr = tl.advance(teacher_hidden_block_ptr, (0, -HIDDEN_RESET))
        student_hidden_block_ptr = tl.advance(student_hidden_block_ptr, (0, -HIDDEN_RESET))
        weight_block_ptr = tl.advance(weight_block_ptr, (VOCAB_BLOCK, -HIDDEN_RESET))
        bias_block_ptr = tl.advance(bias_block_ptr, (VOCAB_BLOCK,))

        if vocab_start + VOCAB_BLOCK > vocab_size:
            # only the last block has padded vocab columns. Their logits are zero for both rows (zero-padded loads);
            # a large finite negative value excludes them from the exp-sums and keeps the logits difference zero
            vocab_mask = (vocab_start + vocab_offsets_in_block) < vocab_size
            teacher_logits = tl.where(vocab_mask[None, :], teacher_logits, _PADDED_VOCAB_LOGIT)
            student_logits = tl.where(vocab_mask[None, :], student_logits, _PADDED_VOCAB_LOGIT)
        logits_diff = teacher_logits - student_logits

        teacher_max, teacher_exp_sum, teacher_kl_sum = _online_exp_sums_update(
            teacher_logits, logits_diff, teacher_max, teacher_exp_sum, teacher_kl_sum, WITH_WEIGHTED_SUM=True
        )
        student_max, student_exp_sum, student_kl_sum = _online_exp_sums_update(
            student_logits, -logits_diff, student_max, student_exp_sum, student_kl_sum, WITH_WEIGHTED_SUM=SYMMETRICAL
        )

    teacher_target_logits, teacher_blank_logits = _target_blank_logits(
        joint_hidden_ptr,
        teacher_offsets,
        weight_ptr,
        bias_ptr,
        targets,
        blank_mask,
        blank_id,
        hidden_dim,
        HIDDEN_BLOCK,
    )
    student_target_logits, student_blank_logits = _target_blank_logits(
        joint_hidden_ptr,
        student_offsets,
        weight_ptr,
        bias_ptr,
        targets,
        blank_mask,
        blank_id,
        hidden_dim,
        HIDDEN_BLOCK,
    )

    teacher_lse = teacher_max + tl.log(teacher_exp_sum)
    student_lse = student_max + tl.log(student_exp_sum)
    if SYMMETRICAL:
        # 0.5 * sum((p_teacher - p_student) * (log p_teacher - log p_student)); log-sum-exp terms cancel out
        consistency_kl = 0.5 * (teacher_kl_sum / teacher_exp_sum + student_kl_sum / student_exp_sum)
    else:
        consistency_kl = teacher_kl_sum / teacher_exp_sum - (teacher_lse - student_lse)

    tl.store(blank_logprobs_out_ptr + teacher_offsets, teacher_blank_logits - teacher_lse, mask=blank_mask)
    tl.store(target_logprobs_out_ptr + teacher_offsets, teacher_target_logits - teacher_lse, mask=target_mask)
    tl.store(log_sum_exp_out_ptr + teacher_offsets, teacher_lse, mask=blank_mask)
    tl.store(blank_logprobs_out_ptr + student_offsets, student_blank_logits - student_lse, mask=blank_mask)
    tl.store(target_logprobs_out_ptr + student_offsets, student_target_logits - student_lse, mask=target_mask)
    tl.store(log_sum_exp_out_ptr + student_offsets, student_lse, mask=blank_mask)
    tl.store(consistency_kl_out_ptr + teacher_offsets, consistency_kl, mask=blank_mask)


@triton.autotune(
    configs=_bwd_autotune_configs(),
    key=["hidden_dim", "vocab_size", "USE_HIGH_PRECISION_MM", "MM_FULL_PRECISION", "PRUNING_RANGE", "SYMMETRICAL"],
    reset_to_zero=["grad_joint_hidden_out_ptr", "grad_weight_out_ptr", "grad_bias_out_ptr"],
    cache_results=True,
)
@triton.jit
def _rnnt_joint_fused_vocab_consistency_bwd_triton_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    log_sum_exp_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_consistency_kl_ptr,
    grad_joint_hidden_out_ptr,
    grad_weight_out_ptr,
    grad_bias_out_ptr,
    pruning_starts_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    pair_flattened_size: int,
    student_lattice_offset: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    SYMMETRICAL: tl.constexpr,
    USE_HIGH_PRECISION_MM: tl.constexpr = False,
    MM_FULL_PRECISION: tl.constexpr = False,
    PRUNING_RANGE: tl.constexpr = 0,
):
    """
    Atomic backward for (pair block, vocab block) tiles. `pair_flattened_size` is the number of processed teacher
    rows (B * T * (U+1), or B * T * PRUNING_RANGE with pruning); `student_lattice_offset` = B * T * (U+1).
    """
    compute_dtype = tl.float32
    matmul_dtype = (
        tl.bfloat16
        if (
            joint_hidden_ptr.dtype.element_ty == tl.bfloat16
            and (not USE_HIGH_PRECISION_MM)
            and (not MM_FULL_PRECISION)
        )
        else compute_dtype
    )

    # all vocab blocks for a pair block execute consecutively (see the single-row kernel)
    num_pair_blocks = tl.num_programs(axis=0)
    num_vocab_blocks = tl.num_programs(axis=1)
    linear_program_index = tl.program_id(axis=1) * num_pair_blocks + tl.program_id(axis=0)
    pair_block_index = linear_program_index // num_vocab_blocks
    vocab_block_index = linear_program_index % num_vocab_blocks
    vocab_block_start = vocab_block_index * VOCAB_BLOCK
    pair_start = pair_block_index * PAIR_BLOCK

    vocab_offsets = vocab_block_start + tl.arange(0, VOCAB_BLOCK)
    vocab_mask = vocab_offsets < vocab_size

    NUM_HIDDEN_ITERS: tl.constexpr = tl.cdiv(hidden_dim, HIDDEN_BLOCK)
    HIDDEN_RESET: tl.constexpr = NUM_HIDDEN_ITERS * HIDDEN_BLOCK

    teacher_hidden_block_ptr = tl.make_block_ptr(
        base=joint_hidden_ptr,
        shape=(2 * student_lattice_offset, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(pair_start, 0),
        block_shape=(PAIR_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    student_hidden_block_ptr = tl.make_block_ptr(
        base=joint_hidden_ptr,
        shape=(2 * student_lattice_offset, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(student_lattice_offset + pair_start, 0),
        block_shape=(PAIR_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(vocab_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(vocab_block_start, 0),
        block_shape=(VOCAB_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)

    pair_offsets = pair_start + tl.arange(0, PAIR_BLOCK)
    pair_mask = pair_offsets < pair_flattened_size
    teacher_lattice_offsets = _get_lattice_offsets(
        pair_offsets, pair_mask, pruning_starts_ptr, max_tgt_len_plus_1, PRUNING_RANGE
    )
    student_lattice_offsets = teacher_lattice_offsets + student_lattice_offset
    blank_mask, target_mask, targets = _load_lattice_rows(
        teacher_lattice_offsets,
        pair_mask,
        targets_ptr,
        src_lengths_ptr,
        tgt_lengths_ptr,
        max_src_len,
        max_tgt_len_plus_1,
    )
    teacher_lse, teacher_grad_target, teacher_grad_blank = _load_lattice_row_grads(
        teacher_lattice_offsets,
        blank_mask,
        target_mask,
        log_sum_exp_ptr,
        grad_target_scores_ptr,
        grad_blank_scores_ptr,
        compute_dtype,
    )
    student_lse, student_grad_target, student_grad_blank = _load_lattice_row_grads(
        student_lattice_offsets,
        blank_mask,
        target_mask,
        log_sum_exp_ptr,
        grad_target_scores_ptr,
        grad_blank_scores_ptr,
        compute_dtype,
    )
    grad_consistency_kl = tl.load(grad_consistency_kl_ptr + teacher_lattice_offsets, mask=blank_mask, other=0.0).to(
        compute_dtype
    )
    if PRUNING_RANGE:
        # pruned lattice rows: labels from the masked load fail to compile in Triton 3.7, see `_load_targets_unmasked`
        targets = _load_targets_unmasked(
            teacher_lattice_offsets, targets_ptr, student_lattice_offset, max_src_len, max_tgt_len_plus_1
        )

    teacher_rows = teacher_lattice_offsets.to(tl.int64)
    student_rows = student_lattice_offsets.to(tl.int64)
    teacher_logits = tl.zeros([PAIR_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]
    student_logits = tl.zeros([PAIR_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]

    # Inner loop 1: recompute logits for teacher and student rows
    for hidden_start in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        if PRUNING_RANGE:
            teacher_hidden_chunk = _load_hidden_rows(
                joint_hidden_ptr, teacher_rows, blank_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
            ).to(matmul_dtype)
            student_hidden_chunk = _load_hidden_rows(
                joint_hidden_ptr, student_rows, blank_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
            ).to(matmul_dtype)
        else:
            teacher_hidden_chunk = tl.load(teacher_hidden_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
            student_hidden_chunk = tl.load(student_hidden_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
            teacher_hidden_block_ptr = tl.advance(teacher_hidden_block_ptr, (0, HIDDEN_BLOCK))
            student_hidden_block_ptr = tl.advance(student_hidden_block_ptr, (0, HIDDEN_BLOCK))
        weight_chunk_t = tl.load(weight_block_ptr, boundary_check=(0, 1)).T.to(matmul_dtype)
        teacher_logits += matmul(teacher_hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM)
        student_logits += matmul(student_hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM)
        weight_block_ptr = tl.advance(weight_block_ptr, (0, HIDDEN_BLOCK))

    teacher_grad_logits, student_grad_logits = _paired_grad_logits(
        teacher_logits,
        student_logits,
        teacher_lse,
        student_lse,
        teacher_grad_target,
        teacher_grad_blank,
        student_grad_target,
        student_grad_blank,
        grad_consistency_kl,
        targets,
        blank_mask,
        vocab_offsets,
        vocab_block_start + VOCAB_BLOCK > vocab_size,
        vocab_size,
        blank_id,
        SYMMETRICAL,
    )
    grad_bias_acc = tl.sum(teacher_grad_logits, axis=0) + tl.sum(student_grad_logits, axis=0)

    # Inner loop 2: grad joint hidden, reverse iteration for cache reuse
    teacher_grad_logits_matmul = teacher_grad_logits.to(matmul_dtype)
    student_grad_logits_matmul = student_grad_logits.to(matmul_dtype)
    for r_hidden_idx in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        hidden_start = HIDDEN_RESET - HIDDEN_BLOCK - r_hidden_idx
        hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < hidden_dim

        weight_block_ptr = tl.advance(weight_block_ptr, (0, -HIDDEN_BLOCK))
        weight_chunk = tl.load(weight_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

        # need atomic: different vocab parts write to the same place
        teacher_grad_hidden_delta = matmul(
            teacher_grad_logits_matmul, weight_chunk, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM
        )
        tl.atomic_add(
            grad_joint_hidden_out_ptr + teacher_rows[:, None] * hidden_dim + hidden_offsets[None, :],
            teacher_grad_hidden_delta.to(grad_joint_hidden_out_ptr.dtype.element_ty),
            mask=blank_mask[:, None] & hidden_mask[None, :],
            sem="relaxed",
        )
        student_grad_hidden_delta = matmul(
            student_grad_logits_matmul, weight_chunk, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM
        )
        tl.atomic_add(
            grad_joint_hidden_out_ptr + student_rows[:, None] * hidden_dim + hidden_offsets[None, :],
            student_grad_hidden_delta.to(grad_joint_hidden_out_ptr.dtype.element_ty),
            mask=blank_mask[:, None] & hidden_mask[None, :],
            sem="relaxed",
        )

    # Inner loop 3: grad weight; teacher and student contributions are summed before a single atomic
    for r_hidden_idx in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        hidden_start = HIDDEN_RESET - HIDDEN_BLOCK - r_hidden_idx
        hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < hidden_dim

        if PRUNING_RANGE:
            teacher_hidden_chunk_t = _load_hidden_rows(
                joint_hidden_ptr, teacher_rows, blank_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
            ).T.to(matmul_dtype)
            student_hidden_chunk_t = _load_hidden_rows(
                joint_hidden_ptr, student_rows, blank_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
            ).T.to(matmul_dtype)
        else:
            teacher_hidden_block_ptr = tl.advance(teacher_hidden_block_ptr, (0, -HIDDEN_BLOCK))
            student_hidden_block_ptr = tl.advance(student_hidden_block_ptr, (0, -HIDDEN_BLOCK))
            teacher_hidden_chunk_t = tl.load(teacher_hidden_block_ptr, boundary_check=(0, 1)).T.to(matmul_dtype)
            student_hidden_chunk_t = tl.load(student_hidden_block_ptr, boundary_check=(0, 1)).T.to(matmul_dtype)

        grad_weight_delta = matmul(
            teacher_hidden_chunk_t, teacher_grad_logits_matmul, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM
        )
        grad_weight_delta += matmul(
            student_hidden_chunk_t, student_grad_logits_matmul, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM
        )
        # need atomic: different batch parts write to the same place
        tl.atomic_add(
            grad_weight_out_ptr + vocab_offsets[:, None] * hidden_dim + hidden_offsets[None, :],
            grad_weight_delta.T.to(grad_weight_out_ptr.dtype.element_ty),
            mask=vocab_mask[:, None] & hidden_mask[None, :],
            sem="relaxed",
        )

    tl.atomic_add(grad_bias_out_ptr + vocab_offsets, grad_bias_acc, mask=vocab_mask, sem="relaxed")


@triton.autotune(
    configs=_bwd_chunked_grad_logits_autotune_configs(),
    key=["hidden_dim", "vocab_size", "USE_HIGH_PRECISION_MM", "MM_FULL_PRECISION", "PRUNING_RANGE", "SYMMETRICAL"],
    reset_to_zero=["grad_bias_out_ptr"],
    cache_results=True,
)
@triton.jit
def _rnnt_joint_fused_vocab_consistency_bwd_grad_logits_triton_kernel(
    hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    log_sum_exp_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_consistency_kl_ptr,
    grad_logits_out_ptr,
    grad_bias_out_ptr,
    pruning_starts_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    student_lattice_offset: int,
    hidden_rows: int,
    teacher_hidden_row_start: int,
    student_hidden_row_start: int,
    chunk_start: int,
    chunk_size: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    padded_vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SYMMETRICAL: tl.constexpr,
    USE_HIGH_PRECISION_MM: tl.constexpr = False,
    MM_FULL_PRECISION: tl.constexpr = False,
    PRUNING_RANGE: tl.constexpr = 0,
):
    """
    Recompute logits for teacher pair rows [chunk_start, chunk_start + chunk_size) and the corresponding student rows,
    store grad logits into grad_logits_out [2 * chunk_size, padded_vocab_size]: teacher rows first, then student rows.
    Hidden rows are read from `hidden_ptr` [hidden_rows, hidden_dim] starting at `teacher_hidden_row_start` and
    `student_hidden_row_start` (the dense joint hidden, or the packed pruned rows of the chunk).
    """
    compute_dtype = tl.float32
    matmul_dtype = (
        tl.bfloat16
        if (hidden_ptr.dtype.element_ty == tl.bfloat16 and (not USE_HIGH_PRECISION_MM) and (not MM_FULL_PRECISION))
        else compute_dtype
    )

    # Grouped ordering: GROUP_SIZE consecutive pair blocks share vocab blocks for L2 reuse
    program_index = tl.program_id(axis=0)
    num_pair_blocks = tl.cdiv(chunk_size, PAIR_BLOCK)
    NUM_VOCAB_BLOCKS: tl.constexpr = (padded_vocab_size + VOCAB_BLOCK - 1) // VOCAB_BLOCK
    num_programs_in_group = GROUP_SIZE * NUM_VOCAB_BLOCKS
    first_pair_block_index = (program_index // num_programs_in_group) * GROUP_SIZE
    current_group_size = min(num_pair_blocks - first_pair_block_index, GROUP_SIZE)
    program_index_in_group = program_index % num_programs_in_group
    pair_block_index = first_pair_block_index + program_index_in_group % current_group_size
    vocab_block_index = program_index_in_group // current_group_size

    chunk_offsets = pair_block_index * PAIR_BLOCK + tl.arange(0, PAIR_BLOCK)
    chunk_mask = chunk_offsets < chunk_size
    teacher_lattice_offsets = _get_lattice_offsets(
        chunk_start + chunk_offsets, chunk_mask, pruning_starts_ptr, max_tgt_len_plus_1, PRUNING_RANGE
    )
    student_lattice_offsets = teacher_lattice_offsets + student_lattice_offset
    vocab_start = vocab_block_index * VOCAB_BLOCK
    vocab_offsets = vocab_start + tl.arange(0, VOCAB_BLOCK)
    vocab_mask = vocab_offsets < vocab_size

    teacher_hidden_block_ptr = tl.make_block_ptr(
        base=hidden_ptr,
        shape=(hidden_rows, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(teacher_hidden_row_start + pair_block_index * PAIR_BLOCK, 0),
        block_shape=(PAIR_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    student_hidden_block_ptr = tl.make_block_ptr(
        base=hidden_ptr,
        shape=(hidden_rows, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(student_hidden_row_start + pair_block_index * PAIR_BLOCK, 0),
        block_shape=(PAIR_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    # transposed view of weight: [hidden_dim, vocab_size]
    weight_t_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(hidden_dim, vocab_size),
        strides=(1, hidden_dim),
        offsets=(0, vocab_start),
        block_shape=(HIDDEN_BLOCK, VOCAB_BLOCK),
        order=(0, 1),
    )

    teacher_logits = tl.zeros([PAIR_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
    student_logits = tl.zeros([PAIR_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
    for _ in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        weight_chunk_t = tl.load(weight_t_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
        teacher_hidden_chunk = tl.load(teacher_hidden_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
        teacher_logits += matmul(teacher_hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM)
        student_hidden_chunk = tl.load(student_hidden_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
        student_logits += matmul(student_hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM)
        teacher_hidden_block_ptr = tl.advance(teacher_hidden_block_ptr, (0, HIDDEN_BLOCK))
        student_hidden_block_ptr = tl.advance(student_hidden_block_ptr, (0, HIDDEN_BLOCK))
        weight_t_block_ptr = tl.advance(weight_t_block_ptr, (HIDDEN_BLOCK, 0))

    bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)
    teacher_logits += bias_chunk[None, :]
    student_logits += bias_chunk[None, :]

    blank_mask, target_mask, targets = _load_lattice_rows(
        teacher_lattice_offsets,
        chunk_mask,
        targets_ptr,
        src_lengths_ptr,
        tgt_lengths_ptr,
        max_src_len,
        max_tgt_len_plus_1,
    )
    teacher_lse, teacher_grad_target, teacher_grad_blank = _load_lattice_row_grads(
        teacher_lattice_offsets,
        blank_mask,
        target_mask,
        log_sum_exp_ptr,
        grad_target_scores_ptr,
        grad_blank_scores_ptr,
        compute_dtype,
    )
    student_lse, student_grad_target, student_grad_blank = _load_lattice_row_grads(
        student_lattice_offsets,
        blank_mask,
        target_mask,
        log_sum_exp_ptr,
        grad_target_scores_ptr,
        grad_blank_scores_ptr,
        compute_dtype,
    )
    grad_consistency_kl = tl.load(grad_consistency_kl_ptr + teacher_lattice_offsets, mask=blank_mask, other=0.0).to(
        compute_dtype
    )

    teacher_grad_logits, student_grad_logits = _paired_grad_logits(
        teacher_logits,
        student_logits,
        teacher_lse,
        student_lse,
        teacher_grad_target,
        teacher_grad_blank,
        student_grad_target,
        student_grad_blank,
        grad_consistency_kl,
        targets,
        blank_mask,
        vocab_offsets,
        vocab_start + VOCAB_BLOCK > vocab_size,
        vocab_size,
        blank_id,
        SYMMETRICAL,
    )

    tl.atomic_add(
        grad_bias_out_ptr + vocab_offsets,
        tl.sum(teacher_grad_logits, axis=0) + tl.sum(student_grad_logits, axis=0),
        mask=vocab_mask,
        sem="relaxed",
    )
    output_mask = chunk_mask[:, None] & (vocab_offsets < padded_vocab_size)[None, :]
    teacher_out_rows = chunk_offsets.to(tl.int64)
    tl.store(
        grad_logits_out_ptr + teacher_out_rows[:, None] * padded_vocab_size + vocab_offsets[None, :],
        teacher_grad_logits.to(grad_logits_out_ptr.dtype.element_ty),
        mask=output_mask,
    )
    student_out_rows = teacher_out_rows + chunk_size
    tl.store(
        grad_logits_out_ptr + student_out_rows[:, None] * padded_vocab_size + vocab_offsets[None, :],
        student_grad_logits.to(grad_logits_out_ptr.dtype.element_ty),
        mask=output_mask,
    )


class _RnntJointFusedVocabConsistencyTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        joint_hidden: torch.Tensor,
        targets: torch.Tensor,
        tgt_lengths: torch.Tensor,
        src_lengths: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        blank_id: int,
        symmetrical: bool,
        use_high_precision_mm: bool,
        input_grad_full_precision: bool,
        weight_grad_full_precision: bool,
        backward_mode: FusedVocabBackwardMode,
        mm_full_precision: bool,
        backward_pruning_range_size: int | None,
    ):
        float_dtype = torch.float64 if joint_hidden.dtype == torch.float64 else torch.float32

        batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
        half_batch_size = batch_size // 2
        pair_flattened_size = half_batch_size * src_max_length * tgt_max_length_plus_1
        vocab_size = weight.shape[0]
        device = joint_hidden.device

        joint_hidden = joint_hidden.contiguous()
        targets = targets.contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        # Cast once, rather than reloading and converting FP32 master weights in every vocabulary tile.
        matmul_weight = (
            weight.to(torch.bfloat16)
            if joint_hidden.dtype == torch.bfloat16 and not mm_full_precision and not use_high_precision_mm
            else weight
        )

        target_logprobs = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1], dtype=float_dtype, device=device
        )
        blank_logprobs = torch.zeros_like(target_logprobs)
        log_sum_exp_scores = torch.empty_like(target_logprobs)
        consistency_kl = torch.zeros(
            [half_batch_size, src_max_length, tgt_max_length_plus_1], dtype=float_dtype, device=device
        )

        fwd_grid = lambda META: (triton.cdiv(pair_flattened_size, META["PAIR_BLOCK"]),)
        _rnnt_joint_fused_vocab_consistency_fwd_triton_kernel[fwd_grid](
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=matmul_weight,
            bias_ptr=bias,
            target_logprobs_out_ptr=target_logprobs,
            blank_logprobs_out_ptr=blank_logprobs,
            log_sum_exp_out_ptr=log_sum_exp_scores,
            consistency_kl_out_ptr=consistency_kl,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            pair_flattened_size=pair_flattened_size,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            SYMMETRICAL=symmetrical,
            USE_HIGH_PRECISION_MM=use_high_precision_mm,
            MM_FULL_PRECISION=mm_full_precision,
        )

        ctx.save_for_backward(
            joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, log_sum_exp_scores, matmul_weight
        )
        ctx.blank_id = blank_id
        ctx.symmetrical = symmetrical
        ctx.use_high_precision_mm = use_high_precision_mm
        ctx.mm_full_precision = mm_full_precision
        ctx.input_grad_full_precision = input_grad_full_precision
        ctx.weight_grad_full_precision = weight_grad_full_precision
        ctx.backward_mode = backward_mode
        ctx.backward_pruning_range_size = backward_pruning_range_size
        return target_logprobs, blank_logprobs, consistency_kl

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores, grad_consistency_kl):
        (joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, log_sum_exp_scores, matmul_weight) = (
            ctx.saved_tensors
        )
        float_dtype = torch.float64 if joint_hidden.dtype == torch.float64 else torch.float32
        grad_target_scores = grad_target_scores.contiguous()
        grad_blank_scores = grad_blank_scores.contiguous()
        grad_consistency_kl = grad_consistency_kl.contiguous()

        backward_pruning_range_size = ctx.backward_pruning_range_size
        pruning_starts = None
        if backward_pruning_range_size is not None:
            pruning_starts = _select_pruning_starts(
                grad_target_scores,
                grad_blank_scores,
                src_lengths,
                tgt_lengths,
                backward_pruning_range_size,
                paired=True,
            )

        batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
        half_batch_size = batch_size // 2
        student_lattice_offset = half_batch_size * src_max_length * tgt_max_length_plus_1
        pair_flattened_size = half_batch_size * src_max_length * (backward_pruning_range_size or tgt_max_length_plus_1)
        vocab_size = weight.shape[0]
        device = joint_hidden.device

        if ctx.backward_mode == FusedVocabBackwardMode.CHUNKED:
            grad_joint_hidden, grad_weight, grad_bias = _rnnt_joint_fused_vocab_consistency_bwd_chunked(
                joint_hidden=joint_hidden,
                targets=targets,
                src_lengths=src_lengths,
                tgt_lengths=tgt_lengths,
                weight=weight,
                bias=bias,
                log_sum_exp_scores=log_sum_exp_scores,
                grad_target_scores=grad_target_scores,
                grad_blank_scores=grad_blank_scores,
                grad_consistency_kl=grad_consistency_kl,
                blank_id=ctx.blank_id,
                symmetrical=ctx.symmetrical,
                use_high_precision_mm=ctx.use_high_precision_mm,
                mm_full_precision=ctx.mm_full_precision,
                matmul_weight=matmul_weight,
                pruning_starts=pruning_starts,
                backward_pruning_range_size=backward_pruning_range_size,
            )
        else:
            grad_joint_hidden = torch.zeros(
                joint_hidden.shape,
                dtype=float_dtype if ctx.input_grad_full_precision else joint_hidden.dtype,
                device=device,
            )
            grad_weight = torch.zeros(
                [vocab_size, hidden_dim],
                dtype=float_dtype if ctx.weight_grad_full_precision else weight.dtype,
                device=device,
            )
            grad_bias = torch.zeros(
                [vocab_size], dtype=float_dtype if ctx.weight_grad_full_precision else bias.dtype, device=device
            )
            bwd_grid = lambda META: (
                triton.cdiv(pair_flattened_size, META["PAIR_BLOCK"]),
                triton.cdiv(vocab_size, META["VOCAB_BLOCK"]),
            )
            _rnnt_joint_fused_vocab_consistency_bwd_triton_kernel[bwd_grid](
                joint_hidden_ptr=joint_hidden,
                targets_ptr=targets,
                src_lengths_ptr=src_lengths,
                tgt_lengths_ptr=tgt_lengths,
                weight_ptr=matmul_weight,
                bias_ptr=bias,
                log_sum_exp_ptr=log_sum_exp_scores,
                grad_target_scores_ptr=grad_target_scores,
                grad_blank_scores_ptr=grad_blank_scores,
                grad_consistency_kl_ptr=grad_consistency_kl,
                grad_joint_hidden_out_ptr=grad_joint_hidden,
                grad_weight_out_ptr=grad_weight,
                grad_bias_out_ptr=grad_bias,
                pruning_starts_ptr=pruning_starts,
                max_src_len=src_max_length,
                max_tgt_len_plus_1=tgt_max_length_plus_1,
                pair_flattened_size=pair_flattened_size,
                student_lattice_offset=student_lattice_offset,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                blank_id=ctx.blank_id,
                SYMMETRICAL=ctx.symmetrical,
                USE_HIGH_PRECISION_MM=ctx.use_high_precision_mm,
                MM_FULL_PRECISION=ctx.mm_full_precision,
                PRUNING_RANGE=backward_pruning_range_size or 0,
            )
            grad_joint_hidden = grad_joint_hidden.to(joint_hidden.dtype)
            grad_weight = grad_weight.to(weight.dtype)
            grad_bias = grad_bias.to(bias.dtype)

        return (
            grad_joint_hidden,
            None,
            None,
            None,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rnnt_joint_fused_vocab_consistency_triton(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    consistency_symmetrical: bool = True,
    use_high_precision_mm: bool = False,
    input_grad_full_precision: bool = False,
    weight_grad_full_precision: bool = False,
    backward_mode: str | FusedVocabBackwardMode | None = None,
    mm_full_precision: bool = False,
    backward_pruning_range_size: int | None = None,
) -> TransducerLossInputs:
    """
    Compute target/blank log probabilities and the per-position consistency loss from the joint hidden of a double
    batch without materializing logits [2B, T, U+1, V].

    Rows [0, B) of the double batch are the teacher pass, rows [B, 2B) are the student pass of the same utterances
    (identical lengths and targets in both halves; not checked to avoid host-device synchronization, the kernels
    read lengths and targets of the teacher half for both rows of a pair).
    The consistency loss is KL(teacher || student), or 0.5 * (KL(teacher || student) + KL(student || teacher))
    if `consistency_symmetrical`. Its gradient treats the other distribution as a constant target in each KL
    direction: student gets `coef * (p_student - p_teacher)`, teacher gets the negated value when symmetrical
    (`coef = 0.5`), otherwise no consistency gradient (`coef = 1`).

    Args: see `rnnt_joint_fused_vocab_triton`. With backward pruning, one target window per frame is shared by the
        teacher and the student rows (selected by their summed gradient importance), and the consistency gradient
        is restricted to this window too; the loss values are computed exactly.

    Returns:
        `TransducerLossInputs`: target/blank log-probabilities [2B, T, U+1], consistency loss [B, T, U+1]
        (zero at invalid positions of the teacher).
    """
    if joint_hidden.shape[0] % 2 != 0:
        raise ValueError(f"Consistency loss requires a double batch, got odd batch size {joint_hidden.shape[0]}")
    backward_mode, backward_pruning_range_size = _normalize_backward_options(
        joint_hidden, backward_mode, backward_pruning_range_size
    )
    target_logprobs, blank_logprobs, consistency_kl = _RnntJointFusedVocabConsistencyTritonFunction.apply(
        joint_hidden,
        targets,
        tgt_lengths,
        src_lengths,
        weight,
        bias,
        blank_id,
        consistency_symmetrical,
        use_high_precision_mm,
        input_grad_full_precision,
        weight_grad_full_precision,
        backward_mode,
        mm_full_precision,
        backward_pruning_range_size,
    )
    return TransducerLossInputs(
        target_logprobs=target_logprobs, blank_logprobs=blank_logprobs, consistency_kl=consistency_kl
    )


def _rnnt_joint_fused_vocab_consistency_bwd_chunked(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    log_sum_exp_scores: torch.Tensor,
    grad_target_scores: torch.Tensor,
    grad_blank_scores: torch.Tensor,
    grad_consistency_kl: torch.Tensor,
    blank_id: int,
    symmetrical: bool,
    use_high_precision_mm: bool,
    mm_full_precision: bool,
    matmul_weight: torch.Tensor,
    pruning_starts: torch.Tensor | None = None,
    backward_pruning_range_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Chunked backward over teacher/student pairs: the Triton kernel writes grad logits for the chunk pairs
    [2 * chunk_size, padded_vocab_size] (teacher rows first) to a bounded buffer, then GEMMs compute grad joint hidden
    and accumulate grad weight in full precision. No atomics except for grad bias.
    """
    float_dtype = torch.float64 if joint_hidden.dtype == torch.float64 else torch.float32
    batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
    half_batch_size = batch_size // 2
    dense_batch_size = batch_size * src_max_length * tgt_max_length_plus_1
    student_lattice_offset = half_batch_size * src_max_length * tgt_max_length_plus_1
    pair_flattened_size = half_batch_size * src_max_length * (backward_pruning_range_size or tgt_max_length_plus_1)
    vocab_size = weight.shape[0]
    device = joint_hidden.device

    use_low_precision_mm = joint_hidden.dtype == torch.bfloat16 and not mm_full_precision and not use_high_precision_mm
    grad_logits_dtype = torch.bfloat16 if use_low_precision_mm else float_dtype
    padded_vocab_size = triton.cdiv(vocab_size, _CHUNKED_BWD_VOCAB_PADDING) * _CHUNKED_BWD_VOCAB_PADDING
    # the buffer holds both teacher and student rows of the chunk
    chunk_pairs = CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES // (2 * padded_vocab_size * grad_logits_dtype.itemsize)
    chunk_pairs = max(chunk_pairs // _CHUNKED_BWD_ROWS_MULTIPLE, 1) * _CHUNKED_BWD_ROWS_MULTIPLE
    chunk_pairs = max(min(chunk_pairs, pair_flattened_size), 1)

    joint_hidden_flat = joint_hidden.view(dense_batch_size, hidden_dim)
    # zero rows for padded vocab: grad logits there are zeros, padding only keeps GEMM operands aligned
    weight_padded = torch.zeros([padded_vocab_size, hidden_dim], dtype=grad_logits_dtype, device=device)
    weight_padded[:vocab_size] = matmul_weight
    grad_logits_buffer = torch.empty([2 * chunk_pairs, padded_vocab_size], dtype=grad_logits_dtype, device=device)

    # each element of grad joint hidden is computed by a single GEMM => store directly in the output dtype
    grad_joint_hidden = (
        torch.zeros_like(joint_hidden) if backward_pruning_range_size else torch.empty_like(joint_hidden)
    )
    grad_joint_hidden_flat = grad_joint_hidden.view(dense_batch_size, hidden_dim)
    if backward_pruning_range_size:
        packed_hidden = torch.empty([2 * chunk_pairs, hidden_dim], dtype=grad_logits_dtype, device=device)
        packed_grad_hidden = torch.empty_like(packed_hidden)
    grad_weight_padded = torch.zeros([padded_vocab_size, hidden_dim], dtype=float_dtype, device=device)
    grad_bias = torch.zeros([vocab_size], dtype=float_dtype, device=device)
    # contiguous views of the halves of the double batch: lattice offsets of pairs index both of them
    joint_hidden_halves = (joint_hidden[:half_batch_size], joint_hidden[half_batch_size:])
    grad_joint_hidden_halves = (grad_joint_hidden[:half_batch_size], grad_joint_hidden[half_batch_size:])

    for chunk_start in range(0, pair_flattened_size, chunk_pairs):
        chunk_size = min(chunk_pairs, pair_flattened_size - chunk_start)
        if backward_pruning_range_size:
            for half_index, joint_hidden_half in enumerate(joint_hidden_halves):
                _copy_pruned_hidden(
                    joint_hidden_half,
                    packed_hidden[half_index * chunk_size :],
                    pruning_starts,
                    src_lengths,
                    tgt_lengths,
                    chunk_start,
                    chunk_size,
                    backward_pruning_range_size,
                    scatter=False,
                )
            hidden, hidden_rows = packed_hidden, 2 * chunk_size
            teacher_hidden_row_start, student_hidden_row_start = 0, chunk_size
        else:
            hidden, hidden_rows = joint_hidden, dense_batch_size
            teacher_hidden_row_start, student_hidden_row_start = chunk_start, student_lattice_offset + chunk_start
        grid = lambda META: (
            triton.cdiv(chunk_size, META["PAIR_BLOCK"]) * triton.cdiv(padded_vocab_size, META["VOCAB_BLOCK"]),
        )
        _rnnt_joint_fused_vocab_consistency_bwd_grad_logits_triton_kernel[grid](
            hidden_ptr=hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=matmul_weight,
            bias_ptr=bias,
            log_sum_exp_ptr=log_sum_exp_scores,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            grad_consistency_kl_ptr=grad_consistency_kl,
            grad_logits_out_ptr=grad_logits_buffer,
            grad_bias_out_ptr=grad_bias,
            pruning_starts_ptr=pruning_starts,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            student_lattice_offset=student_lattice_offset,
            hidden_rows=hidden_rows,
            teacher_hidden_row_start=teacher_hidden_row_start,
            student_hidden_row_start=student_hidden_row_start,
            chunk_start=chunk_start,
            chunk_size=chunk_size,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            padded_vocab_size=padded_vocab_size,
            blank_id=blank_id,
            SYMMETRICAL=symmetrical,
            USE_HIGH_PRECISION_MM=use_high_precision_mm,
            MM_FULL_PRECISION=mm_full_precision,
            PRUNING_RANGE=backward_pruning_range_size or 0,
        )
        grad_logits_chunk = grad_logits_buffer[: 2 * chunk_size]
        if backward_pruning_range_size:
            # packed rows of the teacher and the student are contiguous => one GEMM for each gradient
            torch.mm(grad_logits_chunk, weight_padded, out=packed_grad_hidden[: 2 * chunk_size])
            for half_index, grad_joint_hidden_half in enumerate(grad_joint_hidden_halves):
                _copy_pruned_hidden(
                    grad_joint_hidden_half,
                    packed_grad_hidden[half_index * chunk_size :],
                    pruning_starts,
                    src_lengths,
                    tgt_lengths,
                    chunk_start,
                    chunk_size,
                    backward_pruning_range_size,
                    scatter=True,
                )
            _addmm_full_precision_(grad_weight_padded, grad_logits_chunk.T, packed_hidden[: 2 * chunk_size])
            continue

        for half_index, row_start in enumerate((chunk_start, student_lattice_offset + chunk_start)):
            grad_logits_half = grad_logits_chunk[half_index * chunk_size : (half_index + 1) * chunk_size]
            grad_joint_hidden_rows = grad_joint_hidden_flat[row_start : row_start + chunk_size]
            if grad_joint_hidden_rows.dtype == grad_logits_dtype:
                torch.mm(grad_logits_half, weight_padded, out=grad_joint_hidden_rows)
            else:
                grad_joint_hidden_rows.copy_(torch.mm(grad_logits_half, weight_padded))
            _addmm_full_precision_(
                grad_weight_padded,
                grad_logits_half.T,
                joint_hidden_flat[row_start : row_start + chunk_size].to(grad_logits_dtype),
            )

    grad_weight = grad_weight_padded[:vocab_size].to(weight.dtype)
    grad_bias = grad_bias.to(bias.dtype)
    return grad_joint_hidden, grad_weight, grad_bias


@triton.jit
def _online_exp_sums_update(
    block_logits,
    block_values,
    running_max,
    running_exp_sum,
    running_weighted_sum,
    WITH_WEIGHTED_SUM: tl.constexpr,
):
    """
    Online update over vocabulary blocks: exp_sum = sum(exp(logits - max)),
    weighted_sum = sum(exp(logits - max) * values); both are rescaled when the running max changes.
    """
    new_max = tl.maximum(running_max, tl.max(block_logits, axis=-1))
    scale = tl.exp(running_max - new_max)
    block_exp = tl.exp(block_logits - new_max[:, None])
    running_exp_sum = running_exp_sum * scale + tl.sum(block_exp, axis=-1)
    if WITH_WEIGHTED_SUM:
        running_weighted_sum = running_weighted_sum * scale + tl.sum(block_exp * block_values, axis=-1)
    return new_max, running_exp_sum, running_weighted_sum


@triton.jit
def _target_blank_logits(
    joint_hidden_ptr,
    rows,
    weight_ptr,
    bias_ptr,
    targets,
    blank_mask,
    blank_id: tl.constexpr,
    hidden_dim: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
):
    """
    Target and blank logits of the hidden rows: dot products with the gathered weight rows (CUDA cores).
    Bias and weight rows are gathered without the target mask: labels are zero (a valid row) at positions without
    a target, and their target logits are not stored. With the target mask, `TritonGPURemoveLayoutConversions`
    in Triton 3.7 fails ("'tt.load' op failed to verify that mask type matches ptr type").
    """
    compute_dtype = tl.float32
    target_logits = tl.load(bias_ptr + targets).to(compute_dtype)
    blank_logits = tl.zeros_like(target_logits) + tl.load(bias_ptr + blank_id).to(compute_dtype)
    hidden_rows = rows.to(tl.int64) * hidden_dim
    target_weight_rows = targets.to(tl.int64) * hidden_dim
    for hidden_start in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < hidden_dim
        hidden_chunk = tl.load(
            joint_hidden_ptr + hidden_rows[:, None] + hidden_offsets[None, :],
            mask=blank_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(compute_dtype)
        target_weight_chunk = tl.load(
            weight_ptr + target_weight_rows[:, None] + hidden_offsets[None, :],
            mask=hidden_mask[None, :],
            other=0.0,
        ).to(compute_dtype)
        blank_weight_chunk = tl.load(
            weight_ptr + blank_id * hidden_dim + hidden_offsets, mask=hidden_mask, other=0.0
        ).to(compute_dtype)
        target_logits += tl.sum(hidden_chunk * target_weight_chunk, axis=1)
        blank_logits += tl.sum(hidden_chunk * blank_weight_chunk[None, :], axis=1)
    return target_logits, blank_logits


@triton.jit
def _paired_grad_logits(
    teacher_logits,
    student_logits,
    teacher_lse,
    student_lse,
    teacher_grad_target,
    teacher_grad_blank,
    student_grad_target,
    student_grad_blank,
    grad_consistency_kl,
    targets,
    row_mask,
    vocab_offsets,
    has_padded_vocab,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    SYMMETRICAL: tl.constexpr,
):
    """
    Gradient wrt teacher/student logits of the RNN-T log-probabilities and the consistency loss.
    The consistency gradient treats the other distribution as a constant target in each KL direction:
    student += coef * grad_kl * (p_student - p_teacher); teacher -= the same value when symmetrical (coef = 0.5),
    otherwise teacher gets no consistency gradient (coef = 1).
    Terms are grouped by probabilities with per-row coefficients to keep few [rows, vocab] tensors alive.
    Gradients are zero for invalid rows and padded vocabulary columns without masking the [rows, vocab] outputs:
    probabilities are zero there (+inf log-sum-exp, -inf logits), upstream gradients of invalid rows are zero.
    """
    if has_padded_vocab:
        vocab_mask = vocab_offsets < vocab_size
        teacher_logits = tl.where(vocab_mask[None, :], teacher_logits, -float("inf"))
        student_logits = tl.where(vocab_mask[None, :], student_logits, -float("inf"))
    teacher_lse = tl.where(row_mask, teacher_lse, float("inf"))
    student_lse = tl.where(row_mask, student_lse, float("inf"))
    consistency_coef = (0.5 if SYMMETRICAL else 1.0) * grad_consistency_kl
    teacher_probabilities = tl.exp(teacher_logits - teacher_lse[:, None])
    student_probabilities = tl.exp(student_logits - student_lse[:, None])
    is_target_col = vocab_offsets[None, :] == targets[:, None]
    is_blank_col = (vocab_offsets == blank_id)[None, :]

    teacher_coef = -(teacher_grad_target + teacher_grad_blank)
    if SYMMETRICAL:
        teacher_coef += consistency_coef
    teacher_grad_logits = teacher_coef[:, None] * teacher_probabilities
    if SYMMETRICAL:
        teacher_grad_logits -= consistency_coef[:, None] * student_probabilities
    teacher_grad_logits += tl.where(is_target_col, teacher_grad_target[:, None], 0.0)
    teacher_grad_logits += tl.where(is_blank_col, teacher_grad_blank[:, None], 0.0)

    student_coef = consistency_coef - (student_grad_target + student_grad_blank)
    student_grad_logits = (
        student_coef[:, None] * student_probabilities - consistency_coef[:, None] * teacher_probabilities
    )
    student_grad_logits += tl.where(is_target_col, student_grad_target[:, None], 0.0)
    student_grad_logits += tl.where(is_blank_col, student_grad_blank[:, None], 0.0)
    return teacher_grad_logits, student_grad_logits


@triton.jit
def _load_hidden_rows(hidden_ptr, rows, row_mask, hidden_start, hidden_dim: tl.constexpr, HIDDEN_BLOCK: tl.constexpr):
    """Load [rows, HIDDEN_BLOCK] chunk of the hidden tensor for arbitrary (int64) row indices"""
    hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
    return tl.load(
        hidden_ptr + rows[:, None] * hidden_dim + hidden_offsets[None, :],
        mask=row_mask[:, None] & (hidden_offsets < hidden_dim)[None, :],
        other=0.0,
    )


@triton.jit
def _load_targets_unmasked(lattice_offsets, targets_ptr, lattice_size, max_src_len, max_tgt_len_plus_1):
    """
    Target labels for flat offsets in the [B, T, U+1] lattice without a masked load: the index is clamped in-bounds,
    labels at positions without a target are arbitrary (callers zero their target gradients).
    Workaround for Triton 3.7: when the labels from a masked load (`_load_lattice_rows`) are broadcast against
    the vocabulary block, `TritonGPURemoveLayoutConversions` fails with
    "'tt.load' op failed to verify that mask type matches ptr type".
    """
    max_tgt_len = max_tgt_len_plus_1 - 1
    clamped_offsets = tl.minimum(lattice_offsets, lattice_size - 1)
    batch_index = clamped_offsets // (max_src_len * max_tgt_len_plus_1)
    target_index = tl.minimum(clamped_offsets % max_tgt_len_plus_1, max_tgt_len - 1)
    if max_tgt_len > 0:
        targets = tl.load(targets_ptr + batch_index * max_tgt_len + target_index)
    else:
        # empty targets tensor: nothing to load
        targets = tl.zeros_like(target_index).to(targets_ptr.dtype.element_ty)
    return targets
