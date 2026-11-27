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
With the pruned lattice (`pruning_range_starts`), the joint hidden is given only for the pruning ranges shared by
the teacher and the student: [2B, T, range_size, hidden]; both losses are computed only inside the ranges.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab_triton import (
    _AUTOTUNE_MIN_GRAD_LOGITS,
    _CHUNKED_BWD_ROWS_MULTIPLE,
    _CHUNKED_BWD_VOCAB_PADDING,
    FusedVocabBackwardMode,
    _addmm_full_precision_,
    _autotune_size_class,
    _check_atomic_accumulators,
    _check_joint_inputs,
    _copy_pruned_hidden,
    _exclude_utterances_without_grad,
    _get_lattice_offsets,
    _has_utterance_grad,
    _load_lattice_row_grads,
    _load_lattice_rows,
    _normalize_backward_options,
    _normalize_pruning_range_starts,
    _select_pruning_starts,
)
from nemo.collections.asr.parts.turbo_transducer.types import TransducerLossInputs
from nemo.collections.asr.parts.turbo_transducer.utils_triton import (
    autotune,
    block_offsets_int64,
    can_use_tensor_descriptors,
    cuda_device_guard,
    launch_kernel_on_device,
    load_rows_block,
    make_tensor_descriptor,
    make_tile_configs,
    matmul,
)

# Size of the grad logits buffer (teacher and student rows) used by CHUNKED backward
CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES = 128 * 1024**2
# Logit for padded vocabulary columns in forward: exp underflows to zero, finite to keep differences zero
_PADDED_VOCAB_LOGIT: tl.constexpr = tl.constexpr(-1.0e30)


# Autotune configs: (PAIR_BLOCK, HIDDEN_BLOCK, VOCAB_BLOCK, num_warps, num_stages), see the comments for the configs
# of the kernels without the consistency loss. Two tiles of logits [PAIR_BLOCK, VOCAB_BLOCK] (teacher, student)
# are kept in registers: 2 * PAIR_BLOCK * VOCAB_BLOCK / (32 * num_warps) must not exceed 128.
_BLOCK_NAMES = ["PAIR_BLOCK", "HIDDEN_BLOCK", "VOCAB_BLOCK"]


def _fwd_autotune_configs():
    tiles = [
        (32, 32, 128, 4, 3),
        (64, 32, 64, 4, 3),
        (64, 32, 128, 4, 3),
        (64, 32, 128, 8, 3),
        (64, 64, 128, 8, 3),
        (128, 32, 64, 8, 3),
        (64, 64, 64, 4, 3),  # best for 1025, 8193 (RTX PRO 6000 Blackwell)
        # Larger vocab tiles, best on RTX A6000 (sm_86): 10% faster for vocab 1025, 18% for vocab 16385
        (32, 32, 256, 4, 3),
        (64, 32, 256, 8, 3),
    ]
    tensor_descriptor_tiles = [
        (64, 64, 128, 4, 3),  # best on GB300 (sm_103) for vocab 1025 - 16385
        (64, 64, 64, 4, 3),
        (64, 64, 256, 8, 3),
    ]
    return make_tile_configs(_BLOCK_NAMES, tiles, tensor_descriptor_tiles)


def _bwd_autotune_configs():
    tiles = [
        (64, 32, 64, 4, 2),
        (64, 32, 128, 8, 2),
        (64, 32, 256, 8, 2),
        (128, 32, 64, 8, 2),
        (128, 32, 128, 8, 2),
        (128, 32, 128, 8, 1),
        (64, 32, 128, 4, 2),  # best for 1025, 8193 (RTX PRO 6000 Blackwell)
    ]
    return make_tile_configs(_BLOCK_NAMES, tiles)


def _bwd_chunked_grad_logits_autotune_configs():
    tiles = [
        (32, 64, 128, 4, 4),
        (64, 32, 128, 4, 4),
        (64, 64, 128, 8, 3),
        (64, 32, 256, 8, 3),
        (128, 32, 128, 8, 3),
        (128, 64, 64, 8, 3),
        (64, 32, 128, 4, 3),  # best for 8193 (RTX PRO 6000 Blackwell)
        (64, 32, 64, 4, 4),  # best for 1025 (RTX PRO 6000 Blackwell)
    ]
    tensor_descriptor_tiles = [
        (64, 64, 64, 4, 3),  # best on GB300 (sm_103) for vocab 16385
        (64, 64, 128, 4, 3),
        (64, 32, 64, 4, 3),  # best on GB300 (sm_103) for vocab 1025
    ]
    return make_tile_configs(_BLOCK_NAMES, tiles, tensor_descriptor_tiles, GROUP_SIZE=8)


@autotune(
    configs=_fwd_autotune_configs(),
    key=[
        "hidden_dim",
        "vocab_size",
        "USE_HIGH_PRECISION_MM",
        "MM_FULL_PRECISION",
        "SYMMETRICAL",
        "TENSOR_DESCRIPTORS_ALLOWED",
        "autotune_size_class",
    ],
)
# In all kernels, shape-dependent integer arguments are not specialized to avoid recompilation
# for each new batch shape (Triton specializes integers equal to 1 and divisible by 16 by default).
# `autotune_size_class` is not used by the kernels: it is only a part of the autotune key.
@triton.jit(
    do_not_specialize=[
        "max_src_len",
        "max_tgt_len_plus_1",
        "pair_flattened_size",
        "student_lattice_offset",
        "autotune_size_class",
    ]
)
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
    pruning_starts_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    pair_flattened_size: int,
    student_lattice_offset: int,
    autotune_size_class: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    SYMMETRICAL: tl.constexpr,
    USE_HIGH_PRECISION_MM: tl.constexpr,
    MM_FULL_PRECISION: tl.constexpr = False,
    PRUNING_RANGE: tl.constexpr = 0,
    TENSOR_DESCRIPTORS: tl.constexpr = False,
    TENSOR_DESCRIPTORS_ALLOWED: tl.constexpr = False,
):
    """
    Teacher rows are [0, pair_flattened_size) of the flattened batch, student rows are offset by pair_flattened_size.
    The rows are the positions of the lattice [B, T, U+1], or, with `PRUNING_RANGE`, the positions of the pruning
    ranges [B, T, PRUNING_RANGE] (pruned lattice). The log-probabilities are always stored to the lattice
    [B, T, U+1], `student_lattice_offset` = B * T * (U+1) is the offset of the student outputs;
    the consistency loss is stored for the rows.
    Log-softmax and KL accumulators are computed online over vocabulary blocks with running max and exp-sums:
    KL(teacher || student) = sum(p_teacher * (z_teacher - z_student)) - (lse_teacher - lse_student)
    Target and blank logits are computed after the vocabulary loop from the gathered weight rows: extracting them
    from each block of logits needs more registers than the two logits accumulators leave (the best configs for
    the vocab 16385 are 35-40% slower, RTX A6000). They must be consistent with the logits used for the log-sum-exp,
    otherwise the log-probability of a label with the probability close to 1 can be positive (negative loss):
    - the matmul does not round its operands (BF16 operands, or `USE_HIGH_PRECISION_MM`): dot products computed
      by CUDA cores, see `_target_blank_logits`. They differ from the logits of the blocks only by the order of
      the FP32 accumulation (~1e-7 of the magnitude of the logits);
    - the matmul rounds FP32 operands to TF32 (`MATMUL_ROUNDS_OPERANDS_TO_TF32`; the error is ~1e-3 of
      the magnitude of the logits, which is larger than the log-probabilities of the confident predictions):
      the same matmul for the labels of the rows, see `_target_blank_logits_matmul`. It is not used in the first
      case: the forward kernel is 0.6% (vocab 1025) - 4.7% (vocab 16385) slower with it for BF16 (RTX A6000).
    In both cases the log-probabilities are limited by 0.
    With `TENSOR_DESCRIPTORS` (autotune config) and `TENSOR_DESCRIPTORS_ALLOWED` (see `can_use_tensor_descriptors`),
    the tiles of the joint hidden and of the weight are loaded with tensor descriptors, otherwise - with pointers
    and explicit masks.
    """
    USE_TENSOR_DESCRIPTORS: tl.constexpr = TENSOR_DESCRIPTORS and TENSOR_DESCRIPTORS_ALLOWED
    compute_dtype = tl.float32
    # NB: the flags go first, Triton 3.3 does not support `and` with a comparison of dtypes on the left
    matmul_dtype = (
        tl.bfloat16
        if not (USE_HIGH_PRECISION_MM or MM_FULL_PRECISION) and joint_hidden_ptr.dtype.element_ty == tl.bfloat16
        else compute_dtype
    )

    # The matmul (tensor cores) rounds FP32 operands to TF32, 10 bits of mantissa instead of 23, unless the IEEE
    # precision is requested (`USE_HIGH_PRECISION_MM`); BF16 operands are used without rounding.
    # The flag selects how the target and blank logits are computed after the loop over the vocabulary: they must
    # be computed from the same (rounded) operands as the blocks of logits, see the docstring.
    # NB: the flag goes first, see above
    MATMUL_ROUNDS_OPERANDS_TO_TF32: tl.constexpr = not USE_HIGH_PRECISION_MM and matmul_dtype != tl.bfloat16

    pair_start = tl.program_id(axis=0) * PAIR_BLOCK
    teacher_offsets = pair_start + tl.arange(0, PAIR_BLOCK)
    pair_mask = teacher_offsets < pair_flattened_size
    student_offsets = teacher_offsets + pair_flattened_size

    teacher_lattice_offsets = _get_lattice_offsets(
        teacher_offsets, pair_mask, pruning_starts_ptr, max_tgt_len_plus_1, PRUNING_RANGE
    )
    student_lattice_offsets = teacher_lattice_offsets + student_lattice_offset
    # teacher and student rows have identical lengths and targets => load lattice rows once (also saves registers)
    blank_mask, target_mask, targets = _load_lattice_rows(
        teacher_lattice_offsets,
        pair_mask,
        targets_ptr,
        src_lengths_ptr,
        tgt_lengths_ptr,
        max_src_len,
        max_tgt_len_plus_1,
    )

    vocab_offsets_in_block = tl.arange(0, VOCAB_BLOCK)
    teacher_rows = block_offsets_int64(pair_start, PAIR_BLOCK)
    student_rows = block_offsets_int64(pair_start + pair_flattened_size, PAIR_BLOCK)
    if USE_TENSOR_DESCRIPTORS:
        # the loads are bounded by the shapes of the descriptors: zeros outside the matrices, as for the masked loads;
        # the rows of the teacher end with the first half of the joint hidden
        teacher_hidden_descriptor = make_tensor_descriptor(
            joint_hidden_ptr,
            shape=[pair_flattened_size, hidden_dim],
            strides=[hidden_dim, 1],
            block_shape=[PAIR_BLOCK, HIDDEN_BLOCK],
        )
        student_hidden_descriptor = make_tensor_descriptor(
            joint_hidden_ptr,
            shape=[2 * pair_flattened_size, hidden_dim],
            strides=[hidden_dim, 1],
            block_shape=[PAIR_BLOCK, HIDDEN_BLOCK],
        )
        weight_descriptor = make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_dim],
            strides=[hidden_dim, 1],
            block_shape=[VOCAB_BLOCK, HIDDEN_BLOCK],
        )
    teacher_max = tl.full([PAIR_BLOCK], value=float("-inf"), dtype=compute_dtype)
    teacher_exp_sum = tl.zeros([PAIR_BLOCK], dtype=compute_dtype)
    teacher_kl_sum = tl.zeros([PAIR_BLOCK], dtype=compute_dtype)
    student_max = tl.full([PAIR_BLOCK], value=float("-inf"), dtype=compute_dtype)
    student_exp_sum = tl.zeros([PAIR_BLOCK], dtype=compute_dtype)
    student_kl_sum = tl.zeros([PAIR_BLOCK], dtype=compute_dtype)

    for vocab_start in tl.range(0, vocab_size, VOCAB_BLOCK):
        vocab_offsets = vocab_start + vocab_offsets_in_block
        vocab_mask = vocab_offsets < vocab_size
        weight_rows = block_offsets_int64(vocab_start, VOCAB_BLOCK)
        bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)
        teacher_logits = tl.zeros([PAIR_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]
        student_logits = tl.zeros([PAIR_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]
        for hidden_start in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            # weight tile is shared by teacher and student rows
            if USE_TENSOR_DESCRIPTORS:
                weight_chunk_t = weight_descriptor.load([vocab_start, hidden_start]).T.to(matmul_dtype)
                teacher_hidden_chunk = teacher_hidden_descriptor.load([pair_start, hidden_start]).to(matmul_dtype)
            else:
                weight_chunk_t = load_rows_block(
                    weight_ptr, weight_rows, vocab_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
                ).T.to(matmul_dtype)
                teacher_hidden_chunk = load_rows_block(
                    joint_hidden_ptr,
                    teacher_rows,
                    pair_mask,
                    hidden_start,
                    hidden_dim,
                    HIDDEN_BLOCK,
                    CACHE_VOLATILE=True,
                ).to(matmul_dtype)
            teacher_logits += matmul(teacher_hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM)
            if USE_TENSOR_DESCRIPTORS:
                student_hidden_chunk = student_hidden_descriptor.load(
                    [pair_start + pair_flattened_size, hidden_start]
                ).to(matmul_dtype)
            else:
                student_hidden_chunk = load_rows_block(
                    joint_hidden_ptr,
                    student_rows,
                    pair_mask,
                    hidden_start,
                    hidden_dim,
                    HIDDEN_BLOCK,
                    CACHE_VOLATILE=True,
                ).to(matmul_dtype)
            student_logits += matmul(student_hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM)

        if vocab_start + VOCAB_BLOCK > vocab_size:
            # only the last block has padded vocab columns. Their logits are zero for both rows (zero-padded loads);
            # a large finite negative value excludes them from the exp-sums and keeps the logits difference zero
            teacher_logits = tl.where(vocab_mask[None, :], teacher_logits, _PADDED_VOCAB_LOGIT)
            student_logits = tl.where(vocab_mask[None, :], student_logits, _PADDED_VOCAB_LOGIT)
        logits_diff = teacher_logits - student_logits

        teacher_max, teacher_exp_sum, teacher_kl_sum = _online_exp_sums_update(
            teacher_logits, logits_diff, teacher_max, teacher_exp_sum, teacher_kl_sum, WITH_WEIGHTED_SUM=True
        )
        student_max, student_exp_sum, student_kl_sum = _online_exp_sums_update(
            student_logits, -logits_diff, student_max, student_exp_sum, student_kl_sum, WITH_WEIGHTED_SUM=SYMMETRICAL
        )

    teacher_lse = teacher_max + tl.log(teacher_exp_sum)
    student_lse = student_max + tl.log(student_exp_sum)
    if MATMUL_ROUNDS_OPERANDS_TO_TF32:
        # the same matmul as for the blocks of logits, only for the labels of the rows: the same rounding
        teacher_target_logits, teacher_blank_logits = _target_blank_logits_matmul(
            joint_hidden_ptr,
            teacher_offsets,
            weight_ptr,
            bias_ptr,
            targets,
            blank_mask,
            blank_id,
            hidden_dim,
            vocab_size,
            PAIR_BLOCK,
            HIDDEN_BLOCK,
        )
        student_target_logits, student_blank_logits = _target_blank_logits_matmul(
            joint_hidden_ptr,
            student_offsets,
            weight_ptr,
            bias_ptr,
            targets,
            blank_mask,
            blank_id,
            hidden_dim,
            vocab_size,
            PAIR_BLOCK,
            HIDDEN_BLOCK,
        )
    else:
        # no rounding in the matmul: dot products (faster than the matmul for the labels of the rows)
        teacher_target_logits, teacher_blank_logits = _target_blank_logits(
            joint_hidden_ptr,
            teacher_offsets,
            weight_ptr,
            bias_ptr,
            targets,
            blank_mask,
            blank_id,
            hidden_dim,
            vocab_size,
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
            vocab_size,
            HIDDEN_BLOCK,
        )
    if SYMMETRICAL:
        # 0.5 * sum((p_teacher - p_student) * (log p_teacher - log p_student)); log-sum-exp terms cancel out
        consistency_kl = 0.5 * (teacher_kl_sum / teacher_exp_sum + student_kl_sum / student_exp_sum)
    else:
        consistency_kl = teacher_kl_sum / teacher_exp_sum - (teacher_lse - student_lse)

    tl.store(
        blank_logprobs_out_ptr + teacher_lattice_offsets,
        _limit_logprobs(teacher_blank_logits - teacher_lse),
        mask=blank_mask,
    )
    tl.store(
        target_logprobs_out_ptr + teacher_lattice_offsets,
        _limit_logprobs(teacher_target_logits - teacher_lse),
        mask=target_mask,
    )
    tl.store(log_sum_exp_out_ptr + teacher_lattice_offsets, teacher_lse, mask=blank_mask)
    tl.store(
        blank_logprobs_out_ptr + student_lattice_offsets,
        _limit_logprobs(student_blank_logits - student_lse),
        mask=blank_mask,
    )
    tl.store(
        target_logprobs_out_ptr + student_lattice_offsets,
        _limit_logprobs(student_target_logits - student_lse),
        mask=target_mask,
    )
    tl.store(log_sum_exp_out_ptr + student_lattice_offsets, student_lse, mask=blank_mask)
    tl.store(consistency_kl_out_ptr + teacher_offsets, consistency_kl, mask=blank_mask)


@autotune(
    configs=_bwd_autotune_configs(),
    key=[
        "hidden_dim",
        "vocab_size",
        "USE_HIGH_PRECISION_MM",
        "MM_FULL_PRECISION",
        "PRUNING_RANGE",
        "PACKED_HIDDEN",
        "SYMMETRICAL",
        "autotune_size_class",
    ],
    reset_to_zero=["grad_joint_hidden_out_ptr", "grad_weight_out_ptr", "grad_bias_out_ptr"],
)
@triton.jit(
    do_not_specialize=[
        "max_src_len",
        "max_tgt_len_plus_1",
        "pair_flattened_size",
        "student_lattice_offset",
        "autotune_size_class",
    ]
)
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
    autotune_size_class: int,
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
    PACKED_HIDDEN: tl.constexpr = False,
):
    """
    Atomic backward for (pair block, vocab block) tiles. `pair_flattened_size` is the number of processed teacher
    rows (B * T * (U+1), or B * T * PRUNING_RANGE with pruning); `student_lattice_offset` = B * T * (U+1).
    With pruning, the joint hidden and its gradient are either dense [2B, T, U+1, hidden_dim] (rows of the ranges
    are gathered/scattered), or, with `PACKED_HIDDEN`, contain only the rows of the ranges
    [2B, T, PRUNING_RANGE, hidden_dim].
    """
    compute_dtype = tl.float32
    # NB: the flags go first, Triton 3.3 does not support `and` with a comparison of dtypes on the left
    matmul_dtype = (
        tl.bfloat16
        if not (USE_HIGH_PRECISION_MM or MM_FULL_PRECISION) and joint_hidden_ptr.dtype.element_ty == tl.bfloat16
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
    weight_rows = block_offsets_int64(vocab_block_start, VOCAB_BLOCK)

    # start of the last hidden block: the loops for the gradients iterate over hidden blocks in reverse order
    LAST_HIDDEN_START: tl.constexpr = (tl.cdiv(hidden_dim, HIDDEN_BLOCK) - 1) * HIDDEN_BLOCK

    bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)

    pair_offsets = pair_start + tl.arange(0, PAIR_BLOCK)
    pair_mask = pair_offsets < pair_flattened_size
    teacher_lattice_offsets = _get_lattice_offsets(
        pair_offsets, pair_mask, pruning_starts_ptr, max_tgt_len_plus_1, PRUNING_RANGE
    )
    student_lattice_offsets = teacher_lattice_offsets + student_lattice_offset
    blank_mask, target_mask, _ = _load_lattice_rows(
        teacher_lattice_offsets,
        pair_mask,
        targets_ptr,
        src_lengths_ptr,
        tgt_lengths_ptr,
        max_src_len,
        max_tgt_len_plus_1,
    )
    # labels without a mask: the labels from the masked load fail to compile in Triton 3.7
    targets = _load_targets_unmasked(
        teacher_lattice_offsets, targets_ptr, student_lattice_offset, max_src_len, max_tgt_len_plus_1
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

    # rows of the joint hidden and of its gradient
    if PRUNING_RANGE != 0 and not PACKED_HIDDEN:
        # rows of the pruning ranges are gathered from / scattered to the dense joint hidden
        teacher_rows = teacher_lattice_offsets.to(tl.int64)
        student_rows = student_lattice_offsets.to(tl.int64)
    else:
        teacher_rows = block_offsets_int64(pair_start, PAIR_BLOCK)
        student_rows = block_offsets_int64(pair_start + pair_flattened_size, PAIR_BLOCK)
    teacher_logits = tl.zeros([PAIR_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]
    student_logits = tl.zeros([PAIR_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]

    # Inner loop 1: recompute logits for teacher and student rows
    for hidden_start in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        teacher_hidden_chunk = load_rows_block(
            joint_hidden_ptr, teacher_rows, blank_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
        ).to(matmul_dtype)
        student_hidden_chunk = load_rows_block(
            joint_hidden_ptr, student_rows, blank_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
        ).to(matmul_dtype)
        weight_chunk_t = load_rows_block(
            weight_ptr, weight_rows, vocab_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
        ).T.to(matmul_dtype)
        teacher_logits += matmul(teacher_hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM)
        student_logits += matmul(student_hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM)

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
        (vocab_block_start + VOCAB_BLOCK > vocab_size)
        | _has_excluded_rows(teacher_lattice_offsets, pair_mask, src_lengths_ptr, max_src_len, max_tgt_len_plus_1),
        vocab_offsets,
        vocab_size,
        blank_id,
        SYMMETRICAL,
    )
    grad_bias_acc = tl.sum(teacher_grad_logits, axis=0) + tl.sum(student_grad_logits, axis=0)

    # Inner loop 2: grad joint hidden, reverse iteration for cache reuse
    teacher_grad_logits_matmul = teacher_grad_logits.to(matmul_dtype)
    student_grad_logits_matmul = student_grad_logits.to(matmul_dtype)
    for r_hidden_idx in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        hidden_start = LAST_HIDDEN_START - r_hidden_idx
        hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < hidden_dim

        weight_chunk = load_rows_block(weight_ptr, weight_rows, vocab_mask, hidden_start, hidden_dim, HIDDEN_BLOCK).to(
            matmul_dtype
        )

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
        hidden_start = LAST_HIDDEN_START - r_hidden_idx
        hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < hidden_dim

        teacher_hidden_chunk_t = load_rows_block(
            joint_hidden_ptr, teacher_rows, blank_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
        ).T.to(matmul_dtype)
        student_hidden_chunk_t = load_rows_block(
            joint_hidden_ptr, student_rows, blank_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
        ).T.to(matmul_dtype)

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


@autotune(
    configs=_bwd_chunked_grad_logits_autotune_configs(),
    key=[
        "hidden_dim",
        "vocab_size",
        "USE_HIGH_PRECISION_MM",
        "MM_FULL_PRECISION",
        "PRUNING_RANGE",
        "SYMMETRICAL",
        "TENSOR_DESCRIPTORS_ALLOWED",
        "autotune_size_class",
    ],
    reset_to_zero=["grad_bias_out_ptr"],
)
@triton.jit(
    do_not_specialize=[
        "max_src_len",
        "max_tgt_len_plus_1",
        "student_lattice_offset",
        "teacher_hidden_row_start",
        "student_hidden_row_start",
        "chunk_start",
        "chunk_size",
        "autotune_size_class",
    ]
)
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
    teacher_hidden_row_start: int,
    student_hidden_row_start: int,
    chunk_start: int,
    chunk_size: int,
    autotune_size_class: int,
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
    TENSOR_DESCRIPTORS: tl.constexpr = False,
    TENSOR_DESCRIPTORS_ALLOWED: tl.constexpr = False,
):
    """
    Recompute logits for teacher pair rows [chunk_start, chunk_start + chunk_size) and the corresponding student rows,
    store grad logits into grad_logits_out [2 * chunk_size, padded_vocab_size]: teacher rows first, then student rows.
    Hidden rows are read from `hidden_ptr` [*, hidden_dim] starting at `teacher_hidden_row_start` and
    `student_hidden_row_start` (the dense joint hidden, or the packed pruned rows of the chunk).
    With `TENSOR_DESCRIPTORS` (autotune config) and `TENSOR_DESCRIPTORS_ALLOWED` (see `can_use_tensor_descriptors`),
    the tiles of the hidden, of the weight and of the grad logits are loaded / stored with tensor descriptors,
    otherwise - with pointers and explicit masks.
    """
    USE_TENSOR_DESCRIPTORS: tl.constexpr = TENSOR_DESCRIPTORS and TENSOR_DESCRIPTORS_ALLOWED
    compute_dtype = tl.float32
    # NB: the flags go first, Triton 3.3 does not support `and` with a comparison of dtypes on the left
    matmul_dtype = (
        tl.bfloat16
        if not (USE_HIGH_PRECISION_MM or MM_FULL_PRECISION) and hidden_ptr.dtype.element_ty == tl.bfloat16
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

    chunk_block_start = pair_block_index * PAIR_BLOCK
    chunk_offsets = chunk_block_start + tl.arange(0, PAIR_BLOCK)
    chunk_mask = chunk_offsets < chunk_size
    teacher_lattice_offsets = _get_lattice_offsets(
        chunk_start + chunk_offsets, chunk_mask, pruning_starts_ptr, max_tgt_len_plus_1, PRUNING_RANGE
    )
    student_lattice_offsets = teacher_lattice_offsets + student_lattice_offset
    vocab_start = vocab_block_index * VOCAB_BLOCK
    vocab_offsets = vocab_start + tl.arange(0, VOCAB_BLOCK)
    vocab_mask = vocab_offsets < vocab_size
    weight_rows = block_offsets_int64(vocab_start, VOCAB_BLOCK)

    teacher_hidden_rows = block_offsets_int64(teacher_hidden_row_start + chunk_block_start, PAIR_BLOCK)
    student_hidden_rows = block_offsets_int64(student_hidden_row_start + chunk_block_start, PAIR_BLOCK)
    if USE_TENSOR_DESCRIPTORS:
        # the loads and the stores are bounded by the shapes of the descriptors, as with the explicit masks:
        # the rows of the teacher and of the student end with the chunk, the vocabulary of the grad logits is padded
        teacher_hidden_descriptor = make_tensor_descriptor(
            hidden_ptr,
            shape=[teacher_hidden_row_start + chunk_size, hidden_dim],
            strides=[hidden_dim, 1],
            block_shape=[PAIR_BLOCK, HIDDEN_BLOCK],
        )
        student_hidden_descriptor = make_tensor_descriptor(
            hidden_ptr,
            shape=[student_hidden_row_start + chunk_size, hidden_dim],
            strides=[hidden_dim, 1],
            block_shape=[PAIR_BLOCK, HIDDEN_BLOCK],
        )
        weight_descriptor = make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_dim],
            strides=[hidden_dim, 1],
            block_shape=[VOCAB_BLOCK, HIDDEN_BLOCK],
        )
        teacher_grad_logits_descriptor = make_tensor_descriptor(
            grad_logits_out_ptr,
            shape=[chunk_size, padded_vocab_size],
            strides=[padded_vocab_size, 1],
            block_shape=[PAIR_BLOCK, VOCAB_BLOCK],
        )
        student_grad_logits_descriptor = make_tensor_descriptor(
            grad_logits_out_ptr,
            shape=[2 * chunk_size, padded_vocab_size],
            strides=[padded_vocab_size, 1],
            block_shape=[PAIR_BLOCK, VOCAB_BLOCK],
        )

    teacher_logits = tl.zeros([PAIR_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
    student_logits = tl.zeros([PAIR_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
    for hidden_start in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        if USE_TENSOR_DESCRIPTORS:
            weight_chunk_t = weight_descriptor.load([vocab_start, hidden_start]).T.to(matmul_dtype)
            teacher_hidden_chunk = teacher_hidden_descriptor.load(
                [teacher_hidden_row_start + chunk_block_start, hidden_start]
            ).to(matmul_dtype)
        else:
            weight_chunk_t = load_rows_block(
                weight_ptr, weight_rows, vocab_mask, hidden_start, hidden_dim, HIDDEN_BLOCK, TRANSPOSED=True
            ).to(matmul_dtype)
            teacher_hidden_chunk = load_rows_block(
                hidden_ptr, teacher_hidden_rows, chunk_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
            ).to(matmul_dtype)
        teacher_logits += matmul(teacher_hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM)
        if USE_TENSOR_DESCRIPTORS:
            student_hidden_chunk = student_hidden_descriptor.load(
                [student_hidden_row_start + chunk_block_start, hidden_start]
            ).to(matmul_dtype)
        else:
            student_hidden_chunk = load_rows_block(
                hidden_ptr, student_hidden_rows, chunk_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
            ).to(matmul_dtype)
        student_logits += matmul(student_hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM)

    bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)
    teacher_logits += bias_chunk[None, :]
    student_logits += bias_chunk[None, :]

    blank_mask, target_mask, _ = _load_lattice_rows(
        teacher_lattice_offsets,
        chunk_mask,
        targets_ptr,
        src_lengths_ptr,
        tgt_lengths_ptr,
        max_src_len,
        max_tgt_len_plus_1,
    )
    # labels without a mask: the labels from the masked load fail to compile in Triton 3.7
    targets = _load_targets_unmasked(
        teacher_lattice_offsets, targets_ptr, student_lattice_offset, max_src_len, max_tgt_len_plus_1
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
        (vocab_start + VOCAB_BLOCK > vocab_size)
        | _has_excluded_rows(teacher_lattice_offsets, chunk_mask, src_lengths_ptr, max_src_len, max_tgt_len_plus_1),
        vocab_offsets,
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
    if USE_TENSOR_DESCRIPTORS:
        teacher_grad_logits_descriptor.store(
            [chunk_block_start, vocab_start], teacher_grad_logits.to(grad_logits_out_ptr.dtype.element_ty)
        )
        student_grad_logits_descriptor.store(
            [chunk_size + chunk_block_start, vocab_start], student_grad_logits.to(grad_logits_out_ptr.dtype.element_ty)
        )
    else:
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
        pruning_range_starts: torch.Tensor | None = None,
    ):

        # number of rows of the joint hidden for a frame: U+1, or the pruning range size for the pruned lattice
        batch_size, src_max_length, num_frame_rows, hidden_dim = joint_hidden.shape
        packed_hidden = pruning_range_starts is not None
        tgt_max_length_plus_1 = targets.shape[1] + 1 if packed_hidden else num_frame_rows
        half_batch_size = batch_size // 2
        pair_flattened_size = half_batch_size * src_max_length * num_frame_rows
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

        # pruned lattice: zero probability outside the pruning ranges
        target_logprobs = torch.full(
            [batch_size, src_max_length, tgt_max_length_plus_1],
            fill_value=float("-inf") if packed_hidden else 0.0,
            dtype=torch.float32,
            device=device,
        )
        blank_logprobs = target_logprobs.clone()
        log_sum_exp_scores = torch.empty_like(target_logprobs)
        # the consistency loss is computed for the rows of the joint hidden
        consistency_kl = torch.zeros(
            [half_batch_size, src_max_length, num_frame_rows], dtype=torch.float32, device=device
        )

        fwd_grid = lambda META: (triton.cdiv(pair_flattened_size, META["PAIR_BLOCK"]),)
        launch_kernel_on_device(
            _rnnt_joint_fused_vocab_consistency_fwd_triton_kernel,
            fwd_grid,
            joint_hidden,
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
            pruning_starts_ptr=pruning_range_starts,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            pair_flattened_size=pair_flattened_size,
            autotune_size_class=_autotune_size_class(2 * pair_flattened_size),
            student_lattice_offset=half_batch_size * src_max_length * tgt_max_length_plus_1,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            SYMMETRICAL=symmetrical,
            USE_HIGH_PRECISION_MM=use_high_precision_mm,
            MM_FULL_PRECISION=mm_full_precision,
            PRUNING_RANGE=num_frame_rows if packed_hidden else 0,
            TENSOR_DESCRIPTORS_ALLOWED=can_use_tensor_descriptors(joint_hidden, matmul_weight),
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
        ctx.pruning_range_starts = pruning_range_starts
        return target_logprobs, blank_logprobs, consistency_kl

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_target_scores, grad_blank_scores, grad_consistency_kl):
        (joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, log_sum_exp_scores, matmul_weight) = (
            ctx.saved_tensors
        )
        grad_target_scores = grad_target_scores.contiguous()
        grad_blank_scores = grad_blank_scores.contiguous()
        grad_consistency_kl = grad_consistency_kl.contiguous()
        # a pair of utterances (teacher, student) is excluded only if it has no upstream gradient at all
        half_batch_size = joint_hidden.shape[0] // 2
        has_utterance_grad = _has_utterance_grad(torch.logical_or(grad_target_scores, grad_blank_scores))
        has_pair_grad = (
            has_utterance_grad[:half_batch_size]
            | has_utterance_grad[half_batch_size:]
            | _has_utterance_grad(grad_consistency_kl)
        )
        src_lengths = _exclude_utterances_without_grad(src_lengths, has_pair_grad.repeat(2))

        # pruning ranges: given for the pruned lattice (packed joint hidden), or selected by the gradients
        pruning_starts = ctx.pruning_range_starts
        packed_hidden = pruning_starts is not None
        pruning_range_size = joint_hidden.shape[2] if packed_hidden else ctx.backward_pruning_range_size
        if pruning_range_size is not None and not packed_hidden:
            pruning_starts = _select_pruning_starts(
                grad_target_scores,
                grad_blank_scores,
                src_lengths,
                tgt_lengths,
                pruning_range_size,
                paired=True,
            )

        batch_size, src_max_length, _, hidden_dim = joint_hidden.shape
        tgt_max_length_plus_1 = log_sum_exp_scores.shape[2]
        if packed_hidden:
            # the kernels read the upstream gradients from the lattice [B, T, U+1]
            target_indices = pruning_starts.long().unsqueeze(-1) + torch.arange(
                pruning_range_size, device=pruning_starts.device
            )
            grad_consistency_kl = torch.zeros_like(grad_target_scores[: batch_size // 2]).scatter_(
                2, target_indices, grad_consistency_kl.to(grad_target_scores.dtype)
            )
        student_lattice_offset = half_batch_size * src_max_length * tgt_max_length_plus_1
        pair_flattened_size = half_batch_size * src_max_length * (pruning_range_size or tgt_max_length_plus_1)
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
                pruning_range_size=pruning_range_size,
                packed_hidden=packed_hidden,
            )
        else:
            grad_joint_hidden = torch.zeros(
                joint_hidden.shape,
                dtype=torch.float32 if ctx.input_grad_full_precision else joint_hidden.dtype,
                device=device,
            )
            grad_weight = torch.zeros(
                [vocab_size, hidden_dim],
                dtype=torch.float32 if ctx.weight_grad_full_precision else weight.dtype,
                device=device,
            )
            grad_bias = torch.zeros(
                [vocab_size], dtype=torch.float32 if ctx.weight_grad_full_precision else bias.dtype, device=device
            )
            _check_atomic_accumulators(grad_joint_hidden, grad_weight, grad_bias)
            bwd_grid = lambda META: (
                triton.cdiv(pair_flattened_size, META["PAIR_BLOCK"]),
                triton.cdiv(vocab_size, META["VOCAB_BLOCK"]),
            )
            with cuda_device_guard(joint_hidden):
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
                    autotune_size_class=_autotune_size_class(2 * pair_flattened_size),
                    student_lattice_offset=student_lattice_offset,
                    hidden_dim=hidden_dim,
                    vocab_size=vocab_size,
                    blank_id=ctx.blank_id,
                    SYMMETRICAL=ctx.symmetrical,
                    USE_HIGH_PRECISION_MM=ctx.use_high_precision_mm,
                    MM_FULL_PRECISION=ctx.mm_full_precision,
                    PRUNING_RANGE=pruning_range_size or 0,
                    PACKED_HIDDEN=packed_hidden,
                )
            grad_joint_hidden = grad_joint_hidden.to(joint_hidden.dtype)
            grad_weight = grad_weight.to(weight.dtype)
            grad_bias = grad_bias.to(bias.dtype)

        return (grad_joint_hidden, None, None, None, grad_weight, grad_bias) + (None,) * 9


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
    pruning_range_starts: torch.Tensor | None = None,
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
        With the pruned lattice, `pruning_range_starts` [B, T] are the starts of the pruning ranges shared by
        the teacher and the student rows, the joint hidden is [2B, T, range_size, hidden_dim]. The log-probabilities
        are `-inf` outside the ranges, and the consistency loss is computed only for the positions of the ranges.

    Returns:
        `TransducerLossInputs`: target/blank log-probabilities [2B, T, U+1], consistency loss [B, T, U+1]
        (zero at invalid positions of the teacher). With the pruned lattice, the consistency loss is
        [B, T, range_size]: the loss at `range_index` of a frame is the loss of the target position
        `start + range_index`.
    """
    if joint_hidden.shape[0] % 2 != 0:
        raise ValueError(f"Consistency loss requires a double batch, got odd batch size {joint_hidden.shape[0]}")
    _check_joint_inputs(
        joint_hidden, targets, tgt_lengths, src_lengths, weight, bias, blank_id, pruning_range_starts is not None
    )
    pruning_range_starts = _normalize_pruning_range_starts(
        joint_hidden, targets, pruning_range_starts, backward_pruning_range_size, paired=True
    )
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
        pruning_range_starts,
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
    pruning_range_size: int | None = None,
    packed_hidden: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Chunked backward over teacher/student pairs: the Triton kernel writes grad logits for the chunk pairs
    [2 * chunk_size, padded_vocab_size] (teacher rows first) to a bounded buffer, then GEMMs compute grad joint hidden
    and accumulate grad weight in full precision. No atomics except for grad bias.

    With `pruning_range_size`, only the rows of the pruning ranges are processed. They are gathered from the dense
    joint hidden [2B, T, U+1, hidden_dim] to a buffer (and the gradient is scattered back), unless the joint hidden
    is `packed_hidden`: [2B, T, pruning_range_size, hidden_dim], only the rows of the ranges.
    """
    batch_size, src_max_length, num_frame_rows, hidden_dim = joint_hidden.shape
    tgt_max_length_plus_1 = log_sum_exp_scores.shape[2]
    half_batch_size = batch_size // 2
    num_hidden_rows = batch_size * src_max_length * num_frame_rows
    student_hidden_offset = num_hidden_rows // 2  # first row of the student half of the joint hidden
    student_lattice_offset = half_batch_size * src_max_length * tgt_max_length_plus_1
    pair_flattened_size = half_batch_size * src_max_length * (pruning_range_size or tgt_max_length_plus_1)
    gather_hidden = pruning_range_size is not None and not packed_hidden
    vocab_size = weight.shape[0]
    device = joint_hidden.device

    use_low_precision_mm = joint_hidden.dtype == torch.bfloat16 and not mm_full_precision and not use_high_precision_mm
    grad_logits_dtype = torch.bfloat16 if use_low_precision_mm else torch.float32
    padded_vocab_size = triton.cdiv(vocab_size, _CHUNKED_BWD_VOCAB_PADDING) * _CHUNKED_BWD_VOCAB_PADDING
    # the buffer holds both teacher and student rows of the chunk
    chunk_pairs = CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES // (2 * padded_vocab_size * grad_logits_dtype.itemsize)
    chunk_pairs = max(chunk_pairs // _CHUNKED_BWD_ROWS_MULTIPLE, 1) * _CHUNKED_BWD_ROWS_MULTIPLE
    chunk_pairs = max(min(chunk_pairs, pair_flattened_size), 1)
    # NB: the size class of the first (largest) chunk is used for all chunks: autotuning resets the bias gradient
    # accumulated by the kernel, so it must be possible only for the first chunk (the same autotune key)
    autotune_size_class = int(2 * chunk_pairs * padded_vocab_size >= _AUTOTUNE_MIN_GRAD_LOGITS)

    joint_hidden_flat = joint_hidden.view(num_hidden_rows, hidden_dim)
    # zero rows for padded vocab: grad logits there are zeros, padding only keeps GEMM operands aligned
    weight_padded = torch.zeros([padded_vocab_size, hidden_dim], dtype=grad_logits_dtype, device=device)
    weight_padded[:vocab_size] = matmul_weight
    grad_logits_buffer = torch.empty([2 * chunk_pairs, padded_vocab_size], dtype=grad_logits_dtype, device=device)

    # each element of grad joint hidden is computed by a single GEMM => store directly in the output dtype
    grad_joint_hidden = torch.zeros_like(joint_hidden) if gather_hidden else torch.empty_like(joint_hidden)
    grad_joint_hidden_flat = grad_joint_hidden.view(num_hidden_rows, hidden_dim)
    if gather_hidden:
        gathered_hidden = torch.empty([2 * chunk_pairs, hidden_dim], dtype=grad_logits_dtype, device=device)
        gathered_grad_hidden = torch.empty_like(gathered_hidden)
    grad_weight_padded = torch.zeros([padded_vocab_size, hidden_dim], dtype=torch.float32, device=device)
    grad_bias = torch.zeros([vocab_size], dtype=torch.float32, device=device)
    # contiguous views of the halves of the double batch: lattice offsets of pairs index both of them
    joint_hidden_halves = (joint_hidden[:half_batch_size], joint_hidden[half_batch_size:])
    grad_joint_hidden_halves = (grad_joint_hidden[:half_batch_size], grad_joint_hidden[half_batch_size:])
    # all chunks must have the same autotune key
    tensor_descriptors_allowed = can_use_tensor_descriptors(
        gathered_hidden if gather_hidden else joint_hidden, matmul_weight, grad_logits_buffer
    )

    # GEMMs use the dtypes selected above also if backward is called inside autocast
    with cuda_device_guard(joint_hidden), torch.autocast(joint_hidden.device.type, enabled=False):
        for chunk_start in range(0, pair_flattened_size, chunk_pairs):
            chunk_size = min(chunk_pairs, pair_flattened_size - chunk_start)
            if gather_hidden:
                for half_index, joint_hidden_half in enumerate(joint_hidden_halves):
                    _copy_pruned_hidden(
                        joint_hidden_half,
                        gathered_hidden[half_index * chunk_size :],
                        pruning_starts,
                        src_lengths,
                        tgt_lengths,
                        chunk_start,
                        chunk_size,
                        pruning_range_size,
                        scatter=False,
                    )
                hidden = gathered_hidden
                teacher_hidden_row_start, student_hidden_row_start = 0, chunk_size
            else:
                hidden = joint_hidden
                teacher_hidden_row_start, student_hidden_row_start = chunk_start, student_hidden_offset + chunk_start
            grid = lambda META: (
                triton.cdiv(chunk_size, META["PAIR_BLOCK"]) * triton.cdiv(padded_vocab_size, META["VOCAB_BLOCK"]),
            )
            launch_kernel_on_device(
                _rnnt_joint_fused_vocab_consistency_bwd_grad_logits_triton_kernel,
                grid,
                joint_hidden,
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
                teacher_hidden_row_start=teacher_hidden_row_start,
                student_hidden_row_start=student_hidden_row_start,
                chunk_start=chunk_start,
                chunk_size=chunk_size,
                autotune_size_class=autotune_size_class,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                padded_vocab_size=padded_vocab_size,
                blank_id=blank_id,
                SYMMETRICAL=symmetrical,
                USE_HIGH_PRECISION_MM=use_high_precision_mm,
                MM_FULL_PRECISION=mm_full_precision,
                PRUNING_RANGE=pruning_range_size or 0,
                TENSOR_DESCRIPTORS_ALLOWED=tensor_descriptors_allowed,
            )
            grad_logits_chunk = grad_logits_buffer[: 2 * chunk_size]
            if gather_hidden:
                # gathered rows of the teacher and the student are contiguous => one GEMM for each gradient
                torch.mm(grad_logits_chunk, weight_padded, out=gathered_grad_hidden[: 2 * chunk_size])
                for half_index, grad_joint_hidden_half in enumerate(grad_joint_hidden_halves):
                    _copy_pruned_hidden(
                        grad_joint_hidden_half,
                        gathered_grad_hidden[half_index * chunk_size :],
                        pruning_starts,
                        src_lengths,
                        tgt_lengths,
                        chunk_start,
                        chunk_size,
                        pruning_range_size,
                        scatter=True,
                    )
                _addmm_full_precision_(grad_weight_padded, grad_logits_chunk.T, gathered_hidden[: 2 * chunk_size])
                continue

            for half_index, row_start in enumerate((chunk_start, student_hidden_offset + chunk_start)):
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
    vocab_size: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
):
    """
    Target and blank logits of the hidden rows: dot products with the gathered weight rows (CUDA cores).
    Bias and weight rows are gathered without the target mask. With the target mask,
    `TritonGPURemoveLayoutConversions` in Triton 3.7 fails ("'tt.load' op failed to verify that mask type
    matches ptr type"). Instead, the labels are clamped to the vocabulary: the parameters are never read outside
    the tensors for the positions without a target (label -1) and for the labels outside the vocabulary (invalid
    input). The target logit is NaN for a label outside the vocabulary.
    """
    compute_dtype = tl.float32
    target_is_valid = (targets >= 0) & (targets < vocab_size)
    safe_targets = tl.minimum(tl.maximum(targets, 0), vocab_size - 1)
    target_logits = tl.load(bias_ptr + safe_targets).to(compute_dtype)
    blank_logits = tl.zeros_like(target_logits) + tl.load(bias_ptr + blank_id).to(compute_dtype)
    hidden_rows = rows.to(tl.int64) * hidden_dim
    target_weight_rows = safe_targets.to(tl.int64) * hidden_dim
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
    return tl.where(target_is_valid, target_logits, float("nan")), blank_logits


@triton.jit
def _target_blank_logits_matmul(
    joint_hidden_ptr,
    rows,
    weight_ptr,
    bias_ptr,
    targets,
    blank_mask,
    blank_id: tl.constexpr,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
):
    """
    Target and blank logits of the hidden rows for FP32 matmul operands: the logits of the labels of the rows,
    computed as the blocks of logits in the loop over the vocabulary (bias, then the same matmul for each block
    of the hidden dimension, the operands are rounded to TF32 by the matmul). The dot products
    of `_target_blank_logits` are computed from the operands that are not rounded.
    The labels of the columns are the targets of the rows: [rows, hidden] x [hidden, rows] => the target logit
    of a row is on the diagonal. The weight row of the blank is the first column of a block of the minimal size.
    The labels are clamped to the vocabulary, the target logit is NaN for a label outside the vocabulary,
    see `_target_blank_logits`.
    """
    compute_dtype = tl.float32
    BLANK_COLUMNS: tl.constexpr = 16  # minimal block of the matmul; all columns except the first are zeros
    blank_column_offsets = tl.arange(0, BLANK_COLUMNS)
    is_blank_column = blank_column_offsets == 0
    target_is_valid = (targets >= 0) & (targets < vocab_size)
    safe_targets = tl.minimum(tl.maximum(targets, 0), vocab_size - 1)
    target_bias = tl.load(bias_ptr + safe_targets).to(compute_dtype)
    target_logits_block = tl.zeros([PAIR_BLOCK, PAIR_BLOCK], dtype=compute_dtype) + target_bias[None, :]
    blank_logits_block = tl.zeros([PAIR_BLOCK, BLANK_COLUMNS], dtype=compute_dtype) + tl.load(bias_ptr + blank_id).to(
        compute_dtype
    )
    hidden_rows = rows.to(tl.int64) * hidden_dim
    target_weight_rows = safe_targets.to(tl.int64) * hidden_dim
    for hidden_start in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < hidden_dim
        hidden_chunk = tl.load(
            joint_hidden_ptr + hidden_rows[:, None] + hidden_offsets[None, :],
            mask=blank_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(compute_dtype)
        # weight rows in the transposed layout: [HIDDEN_BLOCK, labels].
        # NB: pointers and masks of the full shape, Triton < 3.5 does not broadcast them to each other
        target_weight_chunk_t = tl.load(
            weight_ptr + target_weight_rows[None, :] + hidden_offsets[:, None],
            mask=hidden_mask[:, None] & (safe_targets >= 0)[None, :],
            other=0.0,
        ).to(compute_dtype)
        blank_weight_chunk_t = tl.load(
            weight_ptr + blank_id * hidden_dim + hidden_offsets[:, None] + 0 * blank_column_offsets[None, :],
            mask=hidden_mask[:, None] & is_blank_column[None, :],
            other=0.0,
        ).to(compute_dtype)
        target_logits_block += matmul(hidden_chunk, target_weight_chunk_t)
        blank_logits_block += matmul(hidden_chunk, blank_weight_chunk_t)
    row_indices = tl.arange(0, PAIR_BLOCK)
    target_logits = tl.sum(tl.where(row_indices[:, None] == row_indices[None, :], target_logits_block, 0.0), axis=1)
    blank_logits = tl.sum(tl.where(is_blank_column[None, :], blank_logits_block, 0.0), axis=1)
    return tl.where(target_is_valid, target_logits, float("nan")), blank_logits


@triton.jit
def _limit_logprobs(logprobs):
    """Log-probabilities limited by 0. NaN values are preserved (`tl.minimum` can replace NaN with the limit)"""
    return tl.where(logprobs > 0.0, 0.0, logprobs)


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
    mask_logits,
    vocab_offsets,
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
    The logits are masked only if `mask_logits` is set (a scalar, computed for the tile): for the padded vocabulary
    columns, and for all invalid rows if the tile contains the rows of an excluded utterance, see
    `_has_excluded_rows` (the recomputed logits of these rows can be non-finite: NaN probabilities for +inf
    log-sum-exp). Masking each tile makes the chunked backward ~1% slower (RTX A6000, vocab 1025).
    """
    if mask_logits:
        logits_mask = row_mask[:, None] & (vocab_offsets < vocab_size)[None, :]
        teacher_logits = tl.where(logits_mask, teacher_logits, -float("inf"))
        student_logits = tl.where(logits_mask, student_logits, -float("inf"))
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
def _has_excluded_rows(lattice_offsets, row_mask, src_lengths_ptr, max_src_len, max_tgt_len_plus_1):
    """
    Whether the tile contains a row of an utterance with zero source length (scalar): backward excludes
    the utterances without upstream gradients by their source lengths, see `_exclude_utterances_without_grad`
    """
    batch_index = lattice_offsets // (max_src_len * max_tgt_len_plus_1)
    source_length = tl.load(src_lengths_ptr + batch_index, mask=row_mask, other=1)
    return tl.min(source_length, axis=0) == 0


@triton.jit
def _load_targets_unmasked(lattice_offsets, targets_ptr, lattice_size, max_src_len, max_tgt_len_plus_1):
    """
    Target labels for flat offsets in the [B, T, U+1] lattice without a masked load: the index is clamped in-bounds,
    labels at positions without a target are arbitrary (callers zero their target gradients).
    Used by the backward kernels for all Triton versions. Workaround for Triton 3.7: when the labels from a masked
    load (`_load_lattice_rows`) are broadcast against the vocabulary block, the kernels can't be compiled
    ("PassManager::run failed": `TritonGPURemoveLayoutConversions` fails with "'tt.load' op failed to verify that
    mask type matches ptr type") for the tiles of 32+ rows with 2+ warps, i.e., for all production configs.
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
