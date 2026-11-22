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
from torch.autograd.function import once_differentiable

from nemo.collections.asr.parts.turbo_transducer.types import FusedVocabBackwardMode
from nemo.collections.asr.parts.turbo_transducer.utils_triton import (
    BF16_ATOMICS_AVAILABLE,
    autotune,
    block_offsets_int64,
    cuda_device_guard,
    load_rows_block,
    log_add_exp,
    matmul,
)


# Size of the grad logits buffer used by CHUNKED backward; 128 MiB keeps GEMMs efficient for vocab sizes 1k-8k
CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES = 128 * 1024**2
_CHUNKED_BWD_ROWS_MULTIPLE = 256
_CHUNKED_BWD_VOCAB_PADDING = 64  # padded vocab rows keep GEMM operands aligned (e.g., vocab 8193 -> 8256)

# The configs of the kernels are autotuned by timing them on the first batch, and the result is cached on disk.
# A small batch (e.g., a smoke run with 2 utterances) does not load the GPU enough to compare the configs.
# The kernels are autotuned separately for the launches below and above these sizes (`autotune_size_class`,
# a part of the autotune key), so that a small first batch does not select the config for the full-size batches.
_AUTOTUNE_MIN_ROWS = 2**17  # rows of the joint hidden: forward and atomic backward kernels
_AUTOTUNE_MIN_GRAD_LOGITS = 2**24  # elements of the grad logits of a chunk: chunked backward kernel


# Autotune configs: (FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK, VOCAB_BLOCK, num_warps, num_stages).
# The best config is selected by Triton on the first call of a kernel for each combination of (hidden dim, vocab size,
# precision flags, dtypes) by timing each config on the actual batch; with Triton 3.4+ the result is cached on disk
# (the key of the cache includes the GPU architecture). The best tiles depend on the GPU, the vocab size and the
# version of Triton (register allocation of the bundled `ptxas`), so the lists cover different tile shapes; configs
# that do not fit a GPU (shared memory) are skipped by the autotuner.
# Rule for new configs: the tile of logits [FLATTENED_BATCH_BLOCK, VOCAB_BLOCK] is kept in registers, and
# FLATTENED_BATCH_BLOCK * VOCAB_BLOCK / (32 * num_warps) elements per thread must not exceed 128 (255 registers
# per thread on all GPUs). Otherwise, the kernel is 2-50 times slower (measured on RTX A6000).
def _fwd_autotune_configs():
    candidates = [
        (64, 32, 64, 4, 3),  # small tiles, low register pressure
        (128, 32, 128, 4, 3),
        (128, 64, 128, 8, 3),
        (256, 32, 128, 8, 3),
        # Larger vocab tiles: the joint hidden tile is reloaded for each vocab tile.
        # Best on RTX A6000 (sm_86): 4% faster for vocab 1025, 18% for vocab 16385 than the configs above
        (64, 32, 256, 4, 3),
        (128, 32, 256, 8, 3),
    ]
    return [
        triton.Config(
            {"FLATTENED_BATCH_BLOCK": fb, "HIDDEN_BLOCK": hb, "VOCAB_BLOCK": vb}, num_warps=nw, num_stages=ns
        )
        for fb, hb, vb, nw, ns in candidates
    ]


def _bwd_autotune_configs():
    # Atomic backward: 2 loops over the hidden blocks with the tile of grad logits kept in registers, atomic adds
    candidates = [
        (64, 32, 64, 8, 2),  # small tiles, low register pressure
        (128, 32, 256, 8, 2),
        (128, 32, 256, 8, 1),
        (256, 32, 128, 8, 2),
        (256, 32, 128, 8, 1),
    ]
    return [
        triton.Config(
            {"FLATTENED_BATCH_BLOCK": fb, "HIDDEN_BLOCK": hb, "VOCAB_BLOCK": vb}, num_warps=nw, num_stages=ns
        )
        for fb, hb, vb, nw, ns in candidates
    ]


def _bwd_chunked_grad_logits_autotune_configs():
    # GEMM-like kernel ([FLATTENED_BATCH_BLOCK, hidden] x [hidden, VOCAB_BLOCK] + epilogue), atomics only for grad bias
    candidates = [
        (128, 32, 128, 4, 4),  # best on RTX A6000 (sm_86) for vocab 1025 - 16385
        (128, 64, 128, 4, 3),
        (128, 64, 128, 8, 3),
        (128, 32, 256, 8, 3),
        (128, 64, 256, 8, 3),
        (256, 32, 128, 8, 3),
        (256, 64, 128, 8, 3),
        (64, 64, 128, 4, 4),
        (128, 64, 64, 4, 4),
        (64, 32, 256, 4, 4),
    ]
    return [
        triton.Config(
            {"FLATTENED_BATCH_BLOCK": fb, "HIDDEN_BLOCK": hb, "VOCAB_BLOCK": vb, "GROUP_SIZE": 8},
            num_warps=nw,
            num_stages=ns,
        )
        for fb, hb, vb, nw, ns in candidates
    ]


@autotune(
    configs=_fwd_autotune_configs(),
    key=["hidden_dim", "vocab_size", "USE_HIGH_PRECISION_MM", "MM_FULL_PRECISION", "autotune_size_class"],
)
# In all kernels, shape-dependent integer arguments are not specialized to avoid recompilation
# for each new batch shape (Triton specializes integers equal to 1 and divisible by 16 by default).
# `autotune_size_class` is not used by the kernels: it is only a part of the autotune key, see `_AUTOTUNE_MIN_ROWS`.
@triton.jit(do_not_specialize=["max_src_len", "max_tgt_len_plus_1", "flattened_batch_size", "autotune_size_class"])
def _rnnt_joint_fused_vocab_fwd_triton_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    target_logprobs_out_ptr,
    blank_logprobs_out_ptr,
    log_sum_exp_out_ptr,
    pruning_starts_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    flattened_batch_size: int,
    autotune_size_class: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    FLATTENED_BATCH_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_HIGH_PRECISION_MM: tl.constexpr,
    MM_FULL_PRECISION: tl.constexpr = False,
    PRUNING_RANGE: tl.constexpr = 0,
):
    """
    Forward for the rows of the joint hidden [flattened_batch_size, hidden_dim]. The rows are the positions of
    the lattice [B, T, U+1], or, with `PRUNING_RANGE`, the positions of the pruning ranges [B, T, PRUNING_RANGE]
    (pruned lattice). The outputs are always stored to the lattice [B, T, U+1].
    """
    compute_dtype = tl.float32
    # NB: the flags go first, Triton 3.3 does not support `and` with a comparison of dtypes on the left
    matmul_dtype = (
        tl.bfloat16
        if not (USE_HIGH_PRECISION_MM or MM_FULL_PRECISION) and joint_hidden_ptr.dtype.element_ty == tl.bfloat16
        else compute_dtype
    )

    flattened_batch_block_index = tl.program_id(axis=0)
    flattened_batch_start = flattened_batch_block_index * FLATTENED_BATCH_BLOCK
    flattened_batch_offsets = flattened_batch_start + tl.arange(0, FLATTENED_BATCH_BLOCK)
    flattened_batch_valid_mask = flattened_batch_offsets < flattened_batch_size

    lattice_offsets = _get_lattice_offsets(
        flattened_batch_offsets, flattened_batch_valid_mask, pruning_starts_ptr, max_tgt_len_plus_1, PRUNING_RANGE
    )
    output_blank_mask, output_target_mask, targets = _load_lattice_rows(
        lattice_offsets,
        flattened_batch_valid_mask,
        targets_ptr,
        src_lengths_ptr,
        tgt_lengths_ptr,
        max_src_len,
        max_tgt_len_plus_1,
    )

    vocab_offsets_in_block = tl.arange(0, VOCAB_BLOCK)
    hidden_rows = block_offsets_int64(flattened_batch_start, FLATTENED_BATCH_BLOCK)

    log_sum_exp_score = tl.full([FLATTENED_BATCH_BLOCK], value=float("-inf"), dtype=compute_dtype)
    blank_logits = tl.zeros([FLATTENED_BATCH_BLOCK], dtype=compute_dtype)
    target_logits = tl.zeros([FLATTENED_BATCH_BLOCK], dtype=compute_dtype)

    # Outer loop over vocab chunks
    for vocab_start in tl.range(0, vocab_size, VOCAB_BLOCK):
        vocab_offsets = vocab_start + vocab_offsets_in_block
        vocab_mask = vocab_offsets < vocab_size
        weight_rows = block_offsets_int64(vocab_start, VOCAB_BLOCK)
        contains_blank = vocab_start <= blank_id and blank_id < vocab_start + VOCAB_BLOCK

        bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)

        block_logits = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]
        for hidden_start in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            # hidden chunk: do not cache
            hidden_chunk = load_rows_block(
                joint_hidden_ptr,
                hidden_rows,
                flattened_batch_valid_mask,
                hidden_start,
                hidden_dim,
                HIDDEN_BLOCK,
                CACHE_VOLATILE=True,
            ).to(matmul_dtype)
            weight_chunk_t = load_rows_block(
                weight_ptr, weight_rows, vocab_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
            ).T.to(matmul_dtype)

            block_logits += matmul(hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM).to(
                compute_dtype
            )

        # Mask invalid vocab positions
        block_logits = tl.where(vocab_mask[None, :], block_logits, -float("inf"))

        # Online log-sum-exp
        block_logits_max = tl.max(block_logits, axis=-1)  # [TILE]
        block_lse = tl.log(tl.sum(tl.exp(block_logits - block_logits_max[:, None]), axis=-1)) + block_logits_max
        log_sum_exp_score = log_add_exp(log_sum_exp_score, block_lse)

        # Extract blank and target logits from this chunk
        target_logits += tl.sum(tl.where(vocab_offsets[None, :] == targets[:, None], block_logits, 0.0), axis=-1)
        if contains_blank:
            blank_logits += tl.sum(tl.where((vocab_offsets == blank_id)[None, :], block_logits, 0.0), axis=-1)

    tl.store(
        blank_logprobs_out_ptr + lattice_offsets,
        blank_logits - log_sum_exp_score,
        mask=output_blank_mask,
    )
    tl.store(
        target_logprobs_out_ptr + lattice_offsets,
        target_logits - log_sum_exp_score,
        mask=output_target_mask,
    )
    tl.store(
        log_sum_exp_out_ptr + lattice_offsets,
        log_sum_exp_score,
        mask=output_blank_mask,
    )


@autotune(
    configs=_bwd_autotune_configs(),
    key=[
        "hidden_dim",
        "vocab_size",
        "USE_HIGH_PRECISION_MM",
        "MM_FULL_PRECISION",
        "PRUNING_RANGE",
        "PACKED_HIDDEN",
        "autotune_size_class",
    ],
    reset_to_zero=["grad_joint_hidden_out_ptr", "grad_weight_out_ptr", "grad_bias_out_ptr"],
)
@triton.jit(do_not_specialize=["max_src_len", "max_tgt_len_plus_1", "flattened_batch_size", "autotune_size_class"])
def _rnnt_joint_fused_vocab_bwd_triton_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    log_sum_exp_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_joint_hidden_out_ptr,
    grad_weight_out_ptr,
    grad_bias_out_ptr,
    pruning_starts_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    flattened_batch_size: int,
    autotune_size_class: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    FLATTENED_BATCH_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_HIGH_PRECISION_MM: tl.constexpr = False,
    MM_FULL_PRECISION: tl.constexpr = False,
    PRUNING_RANGE: tl.constexpr = 0,
    PACKED_HIDDEN: tl.constexpr = False,
):
    """
    Atomic backward for (flattened batch block, vocab block) tiles. With `PRUNING_RANGE`, only the positions of
    the pruning ranges [B, T, PRUNING_RANGE] are processed (`flattened_batch_size` rows). The joint hidden and its
    gradient are either dense [B, T, U+1, hidden_dim] (rows of the ranges are gathered/scattered), or, with
    `PACKED_HIDDEN`, contain only the rows of the ranges [B, T, PRUNING_RANGE, hidden_dim].
    """
    compute_dtype = tl.float32
    # NB: the flags go first, Triton 3.3 does not support `and` with a comparison of dtypes on the left
    matmul_dtype = (
        tl.bfloat16
        if not (USE_HIGH_PRECISION_MM or MM_FULL_PRECISION) and joint_hidden_ptr.dtype.element_ty == tl.bfloat16
        else compute_dtype
    )
    # CUDA enumerates the x grid dimension fastest. Remap that physical order so all vocab blocks for a
    # flattened-batch tile execute consecutively, preserving joint-hidden and gradient cache locality.
    num_flattened_batch_blocks = tl.num_programs(axis=0)
    num_vocab_blocks = tl.num_programs(axis=1)
    linear_program_index = tl.program_id(axis=1) * num_flattened_batch_blocks + tl.program_id(axis=0)
    flattened_batch_index = linear_program_index // num_vocab_blocks
    vocab_block_index = linear_program_index % num_vocab_blocks
    vocab_block_start = vocab_block_index * VOCAB_BLOCK
    flattened_batch_start = flattened_batch_index * FLATTENED_BATCH_BLOCK

    vocab_offsets = vocab_block_start + tl.arange(0, VOCAB_BLOCK)
    vocab_mask = vocab_offsets < vocab_size
    weight_rows = block_offsets_int64(vocab_block_start, VOCAB_BLOCK)

    # start of the last hidden block: the loops for the gradients iterate over hidden blocks in reverse order
    LAST_HIDDEN_START: tl.constexpr = (tl.cdiv(hidden_dim, HIDDEN_BLOCK) - 1) * HIDDEN_BLOCK

    bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)

    flattened_batch_offsets = flattened_batch_start + tl.arange(0, FLATTENED_BATCH_BLOCK)
    flattened_batch_mask = flattened_batch_offsets < flattened_batch_size
    lattice_offsets = _get_lattice_offsets(
        flattened_batch_offsets, flattened_batch_mask, pruning_starts_ptr, max_tgt_len_plus_1, PRUNING_RANGE
    )
    output_blank_mask, output_target_mask, targets = _load_lattice_rows(
        lattice_offsets,
        flattened_batch_mask,
        targets_ptr,
        src_lengths_ptr,
        tgt_lengths_ptr,
        max_src_len,
        max_tgt_len_plus_1,
    )
    lse, grad_target, grad_blank = _load_lattice_row_grads(
        lattice_offsets,
        output_blank_mask,
        output_target_mask,
        log_sum_exp_ptr,
        grad_target_scores_ptr,
        grad_blank_scores_ptr,
        compute_dtype,
    )

    # rows of the joint hidden and of its gradient
    if PRUNING_RANGE != 0 and not PACKED_HIDDEN:
        # rows of the pruning ranges are gathered from / scattered to the dense joint hidden
        flattened_batch_flat_indices = lattice_offsets.to(tl.int64)
    else:
        flattened_batch_flat_indices = block_offsets_int64(flattened_batch_start, FLATTENED_BATCH_BLOCK)
    logits_block = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]

    # Inner loop 1: recompute logits
    for hidden_start in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        # hidden_chunk: [FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK]
        hidden_chunk = load_rows_block(
            joint_hidden_ptr, flattened_batch_flat_indices, output_blank_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
        ).to(matmul_dtype)
        # weight_chunk: [VOCAB_BLOCK, HIDDEN_BLOCK]
        weight_chunk_t = load_rows_block(
            weight_ptr, weight_rows, vocab_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
        ).T.to(matmul_dtype)

        logits_block += matmul(hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM).to(
            compute_dtype
        )

    # Compute grad_logits
    probabilities_block = tl.exp(logits_block - lse[:, None])  # [FLATTENED_BATCH_BLOCK, VOCAB_BLOCK]
    grad_logits_block = _rnnt_grad_logits(
        probabilities_block, grad_target, grad_blank, targets, vocab_offsets, blank_id
    )
    grad_logits_block = tl.where(output_blank_mask[:, None] & vocab_mask[None, :], grad_logits_block, 0.0)
    grad_bias_acc = tl.sum(grad_logits_block, axis=0)

    # Inner loop 2: compute grad_hidden and grad_weight in reverse iteration for cache reuse
    grad_logits_matmul = grad_logits_block.to(matmul_dtype)
    for r_hidden_idx in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        hidden_start = LAST_HIDDEN_START - r_hidden_idx
        hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < hidden_dim

        weight_chunk = load_rows_block(weight_ptr, weight_rows, vocab_mask, hidden_start, hidden_dim, HIDDEN_BLOCK).to(
            matmul_dtype
        )

        grad_hidden_delta = matmul(grad_logits_matmul, weight_chunk, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM).to(
            grad_joint_hidden_out_ptr.dtype.element_ty
        )

        # need atomic: different vocab parts write to the same place
        tl.atomic_add(
            grad_joint_hidden_out_ptr + flattened_batch_flat_indices[:, None] * hidden_dim + hidden_offsets[None, :],
            grad_hidden_delta,
            mask=output_blank_mask[:, None] & hidden_mask[None, :],
            sem="relaxed",  # no need to guarantee order of adding - no read inside kernel
        )

    for r_hidden_idx in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        hidden_start = LAST_HIDDEN_START - r_hidden_idx
        hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < hidden_dim

        joint_hidden_chunk_t = load_rows_block(
            joint_hidden_ptr, flattened_batch_flat_indices, output_blank_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
        ).T.to(matmul_dtype)

        grad_weight_delta = matmul(
            joint_hidden_chunk_t, grad_logits_matmul, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM
        ).T.to(grad_weight_out_ptr.dtype.element_ty)

        # need atomic: different batch parts write to the same place
        tl.atomic_add(
            grad_weight_out_ptr + vocab_offsets[:, None] * hidden_dim + hidden_offsets[None, :],
            grad_weight_delta,
            mask=vocab_mask[:, None] & hidden_mask[None, :],
            sem="relaxed",  # no need to guarantee order of adding - no read inside kernel
        )

    # Atomic add into global grads; different batch parts write to the same place
    tl.atomic_add(
        grad_bias_out_ptr + vocab_offsets,
        grad_bias_acc,
        mask=vocab_mask,
        sem="relaxed",  # no need to guarantee order of adding - no read inside kernel
    )


@autotune(
    configs=_bwd_chunked_grad_logits_autotune_configs(),
    key=[
        "hidden_dim",
        "vocab_size",
        "USE_HIGH_PRECISION_MM",
        "MM_FULL_PRECISION",
        "PRUNING_RANGE",
        "autotune_size_class",
    ],
    reset_to_zero=["grad_bias_out_ptr"],
)
@triton.jit(
    do_not_specialize=["max_src_len", "max_tgt_len_plus_1", "chunk_start", "chunk_size", "autotune_size_class"]
)
def _rnnt_joint_fused_vocab_bwd_grad_logits_triton_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    log_sum_exp_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_logits_out_ptr,
    grad_bias_out_ptr,
    pruning_starts_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    chunk_start: int,
    chunk_size: int,
    autotune_size_class: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    padded_vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    FLATTENED_BATCH_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    USE_HIGH_PRECISION_MM: tl.constexpr = False,
    MM_FULL_PRECISION: tl.constexpr = False,
    PRUNING_RANGE: tl.constexpr = 0,
):
    """
    Recompute logits for rows [chunk_start, chunk_start + chunk_size) of the flattened batch and store grad logits
    into grad_logits_out [chunk_size, padded_vocab_size] (zeros for padded vocab and invalid rows).
    Grad bias is accumulated with atomics (one atomic per vocab column per program).
    """
    compute_dtype = tl.float32
    # NB: the flags go first, Triton 3.3 does not support `and` with a comparison of dtypes on the left
    matmul_dtype = (
        tl.bfloat16
        if not (USE_HIGH_PRECISION_MM or MM_FULL_PRECISION) and joint_hidden_ptr.dtype.element_ty == tl.bfloat16
        else compute_dtype
    )

    # Grouped ordering: GROUP_SIZE consecutive flattened-batch blocks share vocab blocks for L2 reuse
    program_index = tl.program_id(axis=0)
    num_flattened_batch_blocks = tl.cdiv(chunk_size, FLATTENED_BATCH_BLOCK)
    NUM_VOCAB_BLOCKS: tl.constexpr = (padded_vocab_size + VOCAB_BLOCK - 1) // VOCAB_BLOCK
    num_programs_in_group = GROUP_SIZE * NUM_VOCAB_BLOCKS
    first_flattened_batch_block_index = (program_index // num_programs_in_group) * GROUP_SIZE
    current_group_size = min(num_flattened_batch_blocks - first_flattened_batch_block_index, GROUP_SIZE)
    program_index_in_group = program_index % num_programs_in_group
    flattened_batch_block_index = first_flattened_batch_block_index + program_index_in_group % current_group_size
    vocab_block_index = program_index_in_group // current_group_size

    chunk_offsets = flattened_batch_block_index * FLATTENED_BATCH_BLOCK + tl.arange(0, FLATTENED_BATCH_BLOCK)
    chunk_mask = chunk_offsets < chunk_size
    flattened_batch_offsets = chunk_start + chunk_offsets
    lattice_offsets = _get_lattice_offsets(
        flattened_batch_offsets, chunk_mask, pruning_starts_ptr, max_tgt_len_plus_1, PRUNING_RANGE
    )
    vocab_start = vocab_block_index * VOCAB_BLOCK
    vocab_offsets = vocab_start + tl.arange(0, VOCAB_BLOCK)
    vocab_mask = vocab_offsets < vocab_size
    weight_rows = block_offsets_int64(vocab_start, VOCAB_BLOCK)

    # with pruning the joint hidden contains only the rows of the chunk, otherwise - all rows of the flattened batch
    hidden_row_start = flattened_batch_block_index * FLATTENED_BATCH_BLOCK
    if not PRUNING_RANGE:
        hidden_row_start += chunk_start
    hidden_rows = block_offsets_int64(hidden_row_start, FLATTENED_BATCH_BLOCK)

    logits_block = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype)
    for hidden_start in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        hidden_chunk = load_rows_block(
            joint_hidden_ptr, hidden_rows, chunk_mask, hidden_start, hidden_dim, HIDDEN_BLOCK
        ).to(matmul_dtype)
        weight_chunk_t = load_rows_block(
            weight_ptr, weight_rows, vocab_mask, hidden_start, hidden_dim, HIDDEN_BLOCK, TRANSPOSED=True
        ).to(matmul_dtype)
        logits_block += matmul(hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM).to(
            compute_dtype
        )

    bias_chunk = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(compute_dtype)
    logits_block += bias_chunk[None, :]

    output_blank_mask, output_target_mask, targets = _load_lattice_rows(
        lattice_offsets, chunk_mask, targets_ptr, src_lengths_ptr, tgt_lengths_ptr, max_src_len, max_tgt_len_plus_1
    )
    lse, grad_target, grad_blank = _load_lattice_row_grads(
        lattice_offsets,
        output_blank_mask,
        output_target_mask,
        log_sum_exp_ptr,
        grad_target_scores_ptr,
        grad_blank_scores_ptr,
        compute_dtype,
    )

    probabilities_block = tl.exp(logits_block - lse[:, None])
    grad_logits_block = _rnnt_grad_logits(
        probabilities_block, grad_target, grad_blank, targets, vocab_offsets, blank_id
    )
    grad_logits_block = tl.where(output_blank_mask[:, None] & vocab_mask[None, :], grad_logits_block, 0.0)

    tl.atomic_add(
        grad_bias_out_ptr + vocab_offsets,
        tl.sum(grad_logits_block, axis=0),
        mask=vocab_mask,
        sem="relaxed",  # no need to guarantee order of adding - no read inside kernel
    )
    tl.store(
        grad_logits_out_ptr + chunk_offsets.to(tl.int64)[:, None] * padded_vocab_size + vocab_offsets[None, :],
        grad_logits_block.to(grad_logits_out_ptr.dtype.element_ty),
        mask=chunk_mask[:, None] & (vocab_offsets < padded_vocab_size)[None, :],
    )


class _RnntJointFusedVocabTritonFunction(torch.autograd.Function):
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
        use_high_precision_mm: bool = False,
        input_grad_full_precision: bool = False,
        weight_grad_full_precision: bool = False,
        backward_mode: FusedVocabBackwardMode = FusedVocabBackwardMode.ATOMIC,
        mm_full_precision: bool = False,
        backward_pruning_range_size: int | None = None,
        pruning_range_starts: torch.Tensor | None = None,
    ):

        # number of rows of the joint hidden for a frame: U+1, or the pruning range size for the pruned lattice
        batch_size, src_max_length, num_frame_rows, hidden_dim = joint_hidden.shape
        packed_hidden = pruning_range_starts is not None
        tgt_max_length_plus_1 = targets.shape[1] + 1 if packed_hidden else num_frame_rows
        flattened_batch_size = batch_size * src_max_length * num_frame_rows
        vocab_size = weight.shape[0]
        device = joint_hidden.device

        joint_hidden = joint_hidden.contiguous()
        targets = targets.contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        # Cast once, rather than reloading and converting FP32 master weights in every vocabulary tile.
        # Keep the original parameter for its version check and FP32 gradient storage.
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

        fwd_grid = lambda META: (triton.cdiv(flattened_batch_size, META["FLATTENED_BATCH_BLOCK"]),)

        with cuda_device_guard(joint_hidden):
            _rnnt_joint_fused_vocab_fwd_triton_kernel[fwd_grid](
                joint_hidden_ptr=joint_hidden,
                targets_ptr=targets,
                src_lengths_ptr=src_lengths,
                tgt_lengths_ptr=tgt_lengths,
                weight_ptr=matmul_weight,
                bias_ptr=bias,
                target_logprobs_out_ptr=target_logprobs,
                blank_logprobs_out_ptr=blank_logprobs,
                log_sum_exp_out_ptr=log_sum_exp_scores,
                pruning_starts_ptr=pruning_range_starts,
                max_src_len=src_max_length,
                max_tgt_len_plus_1=tgt_max_length_plus_1,
                flattened_batch_size=flattened_batch_size,
                autotune_size_class=_autotune_size_class(flattened_batch_size),
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                blank_id=blank_id,
                USE_HIGH_PRECISION_MM=use_high_precision_mm,
                MM_FULL_PRECISION=mm_full_precision,
                PRUNING_RANGE=num_frame_rows if packed_hidden else 0,
            )

        # NB: the targets are checked here, not in the kernel: a check before / after the loop over the vocabulary
        # makes the forward kernel 2% / 13% slower (interleaved A/B with the best configs, RTX A6000)
        _set_nan_for_invalid_targets_(target_logprobs, targets, tgt_lengths, vocab_size)

        ctx.save_for_backward(
            joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, log_sum_exp_scores, matmul_weight
        )
        ctx.pruning_range_starts = pruning_range_starts
        ctx.blank_id = blank_id
        ctx.use_high_precision_mm = use_high_precision_mm
        ctx.mm_full_precision = mm_full_precision
        ctx.input_grad_full_precision = input_grad_full_precision
        ctx.weight_grad_full_precision = weight_grad_full_precision
        ctx.backward_mode = backward_mode
        ctx.backward_pruning_range_size = backward_pruning_range_size
        return target_logprobs, blank_logprobs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_target_scores, grad_blank_scores):
        (joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, log_sum_exp_scores, matmul_weight) = (
            ctx.saved_tensors
        )
        blank_id = ctx.blank_id
        use_high_precision_mm = ctx.use_high_precision_mm
        mm_full_precision = ctx.mm_full_precision
        input_grad_full_precision = ctx.input_grad_full_precision
        if input_grad_full_precision is None:
            input_grad_full_precision = use_high_precision_mm
        weight_grad_full_precision = ctx.weight_grad_full_precision
        if weight_grad_full_precision is None:
            weight_grad_full_precision = use_high_precision_mm

        grad_target_scores = grad_target_scores.contiguous()
        grad_blank_scores = grad_blank_scores.contiguous()
        # pruning ranges: given for the pruned lattice (packed joint hidden), or selected by the gradients
        pruning_starts = ctx.pruning_range_starts
        packed_hidden = pruning_starts is not None
        pruning_range_size = joint_hidden.shape[2] if packed_hidden else ctx.backward_pruning_range_size
        if pruning_range_size is not None and not packed_hidden:
            pruning_starts = _select_pruning_starts(
                grad_target_scores, grad_blank_scores, src_lengths, tgt_lengths, pruning_range_size
            )

        if ctx.backward_mode == FusedVocabBackwardMode.CHUNKED:
            grad_joint_hidden, grad_weight, grad_bias = _rnnt_joint_fused_vocab_bwd_chunked(
                joint_hidden=joint_hidden,
                targets=targets,
                src_lengths=src_lengths,
                tgt_lengths=tgt_lengths,
                weight=weight,
                bias=bias,
                log_sum_exp_scores=log_sum_exp_scores,
                grad_target_scores=grad_target_scores,
                grad_blank_scores=grad_blank_scores,
                blank_id=blank_id,
                use_high_precision_mm=use_high_precision_mm,
                mm_full_precision=mm_full_precision,
                matmul_weight=matmul_weight,
                pruning_starts=pruning_starts,
                pruning_range_size=pruning_range_size,
                packed_hidden=packed_hidden,
            )
            return (grad_joint_hidden, None, None, None, grad_weight, grad_bias) + (None,) * 8

        batch_size, src_max_length, _, hidden_dim = joint_hidden.shape
        tgt_max_length_plus_1 = log_sum_exp_scores.shape[2]
        flattened_batch_size = batch_size * src_max_length * (pruning_range_size or tgt_max_length_plus_1)
        vocab_size = weight.shape[0]
        device = joint_hidden.device

        grad_joint_hidden = torch.zeros(
            joint_hidden.shape,
            dtype=torch.float32 if input_grad_full_precision else joint_hidden.dtype,
            device=device,
        )

        grad_weight = torch.zeros(
            [vocab_size, hidden_dim],
            dtype=torch.float32 if weight_grad_full_precision else weight.dtype,
            device=device,
        )
        grad_bias = torch.zeros(
            [vocab_size], dtype=torch.float32 if weight_grad_full_precision else bias.dtype, device=device
        )

        _check_atomic_accumulators(grad_joint_hidden, grad_weight, grad_bias)

        bwd_grid = lambda META: (
            triton.cdiv(flattened_batch_size, META["FLATTENED_BATCH_BLOCK"]),
            triton.cdiv(vocab_size, META["VOCAB_BLOCK"]),
        )

        with cuda_device_guard(joint_hidden):
            _rnnt_joint_fused_vocab_bwd_triton_kernel[bwd_grid](
                joint_hidden_ptr=joint_hidden,
                targets_ptr=targets,
                src_lengths_ptr=src_lengths,
                tgt_lengths_ptr=tgt_lengths,
                weight_ptr=matmul_weight,
                bias_ptr=bias,
                log_sum_exp_ptr=log_sum_exp_scores,
                grad_target_scores_ptr=grad_target_scores,
                grad_blank_scores_ptr=grad_blank_scores,
                grad_joint_hidden_out_ptr=grad_joint_hidden,
                grad_weight_out_ptr=grad_weight,
                grad_bias_out_ptr=grad_bias,
                pruning_starts_ptr=pruning_starts,
                max_src_len=src_max_length,
                max_tgt_len_plus_1=tgt_max_length_plus_1,
                flattened_batch_size=flattened_batch_size,
                autotune_size_class=_autotune_size_class(flattened_batch_size),
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                blank_id=blank_id,
                USE_HIGH_PRECISION_MM=use_high_precision_mm,
                MM_FULL_PRECISION=mm_full_precision,
                PRUNING_RANGE=pruning_range_size or 0,
                PACKED_HIDDEN=packed_hidden,
            )

        # convert grad to desired dtype
        grad_weight = grad_weight.to(weight.dtype)
        grad_bias = grad_bias.to(bias.dtype)
        grad_joint_hidden = grad_joint_hidden.to(joint_hidden.dtype)

        return (grad_joint_hidden, None, None, None, grad_weight, grad_bias) + (None,) * 8


def rnnt_joint_fused_vocab_triton(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    use_high_precision_mm: bool = False,
    input_grad_full_precision: bool = False,
    weight_grad_full_precision: bool = False,
    backward_mode: str | FusedVocabBackwardMode | None = None,
    mm_full_precision: bool = False,
    backward_pruning_range_size: int | None = None,
    pruning_range_starts: torch.Tensor | None = None,
):
    """
    Compute target and blank log probabilities from joint hidden without materializing logits [B, T, U+1, V].

    Args:
        joint_hidden: joint hidden [B, T, U+1, hidden_dim]; with ``pruning_range_starts`` - joint hidden of
            the pruned lattice [B, T, range_size, hidden_dim]
        mm_full_precision: Use FP32 matmul operands even for BF16 activations. Otherwise BF16 activations use
            BF16 operands, including when the projection parameters are FP32 master weights. Parameter gradients
            retain their original dtype. FP32 matmuls use Triton's default input precision and PyTorch's global
            matmul precision setting in the chunked GEMMs.
        use_high_precision_mm: Use FP32 operands and IEEE input precision in Triton. This implies full precision
            operands independently of ``mm_full_precision``, preserving the existing high-precision behavior.
        backward_mode: backward algorithm, see `FusedVocabBackwardMode`. If None, selected automatically:
            ATOMIC on GPUs with native bf16 atomics (compute capability >= 9.0), CHUNKED otherwise.
            input_grad_full_precision / weight_grad_full_precision affect only the ATOMIC mode
            (CHUNKED always accumulates grad weight in full precision and rounds each gradient once).
        backward_pruning_range_size: Optional positive number of contiguous target positions retained per frame in
            backward. Windows maximize abs(target gradient) + abs(blank gradient); both outgoing gradients
            are retained without renormalization. Forward is unchanged, so this approximates the full-loss
            gradient. FastEmit produces weighted importance rather than unweighted state posteriors.
            None, non-positive widths, or a range covering the padded target dimension use dense backward.
        pruning_range_starts: Optional starts of the pruning ranges [B, T] for the pruned lattice: the joint hidden
            is given only for ``range_size`` contiguous target positions per frame, row ``range_index`` of a frame
            is the target position ``start + range_index``. Forward and backward are computed only for these
            positions. The outputs have the shape of the full lattice [B, T, U+1] with U = ``targets.shape[1]``
            and are ``-inf`` outside the ranges (zero probability): the loss computed from them is the loss of
            the pruned lattice, and its gradient is exact. The ranges must be inside the lattice and connected
            for a finite loss, see ``rnnt_pruning``. Incompatible with ``backward_pruning_range_size``.
    """
    _check_joint_inputs(
        joint_hidden, targets, tgt_lengths, src_lengths, weight, bias, blank_id, pruning_range_starts is not None
    )
    pruning_range_starts = _normalize_pruning_range_starts(
        joint_hidden, targets, pruning_range_starts, backward_pruning_range_size
    )
    backward_mode, backward_pruning_range_size = _normalize_backward_options(
        joint_hidden, backward_mode, backward_pruning_range_size
    )
    target_logprobs, blank_logprobs = _RnntJointFusedVocabTritonFunction.apply(
        joint_hidden,
        targets,
        tgt_lengths,
        src_lengths,
        weight,
        bias,
        blank_id,
        use_high_precision_mm,
        input_grad_full_precision,
        weight_grad_full_precision,
        backward_mode,
        mm_full_precision,
        backward_pruning_range_size,
        pruning_range_starts,
    )
    return target_logprobs, blank_logprobs


def _check_joint_inputs(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    pruned_lattice: bool,
):
    """
    Check the dtypes and the shapes of the inputs of the fused Joint. The kernels index all tensors by the shape
    of the joint hidden, so inconsistent shapes would silently read wrong data.
    The values (lengths, labels) are not checked to avoid host-device synchronization.
    """
    if joint_hidden.dim() != 4:
        raise ValueError(f"Expected joint hidden [B, T, U+1, hidden_dim], got {list(joint_hidden.shape)}")
    if joint_hidden.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(
            f"Joint hidden must be float32, bfloat16 or float16, got {joint_hidden.dtype} "
            "(log-probabilities and gradients are computed in float32)"
        )
    batch_size, _, num_frame_rows, hidden_dim = joint_hidden.shape
    if weight.dim() != 2 or weight.shape[1] != hidden_dim or tuple(bias.shape) != (weight.shape[0],):
        raise ValueError(
            f"Expected weight [vocab_size, {hidden_dim}] and bias [vocab_size] for joint hidden "
            f"{list(joint_hidden.shape)}, got weight {list(weight.shape)}, bias {list(bias.shape)}"
        )
    if not 0 <= blank_id < weight.shape[0]:
        raise ValueError(f"Blank id {blank_id} is outside the vocabulary of size {weight.shape[0]}")
    if targets.dim() != 2 or targets.shape[0] != batch_size or targets.is_floating_point():
        raise ValueError(
            f"Expected integer targets [{batch_size}, U] for joint hidden {list(joint_hidden.shape)}, "
            f"got {targets.dtype} {list(targets.shape)}"
        )
    # full lattice: the labels are read for the target positions of the joint hidden (not read if U = 0)
    if not pruned_lattice and num_frame_rows > 1 and targets.shape[1] != num_frame_rows - 1:
        raise ValueError(
            f"Expected targets [{batch_size}, {num_frame_rows - 1}] for joint hidden {list(joint_hidden.shape)} "
            f"([B, T, U+1, hidden_dim]), got {list(targets.shape)}"
        )
    if tuple(src_lengths.shape) != (batch_size,) or tuple(tgt_lengths.shape) != (batch_size,):
        raise ValueError(
            f"Expected source and target lengths [{batch_size}], "
            f"got {list(src_lengths.shape)} and {list(tgt_lengths.shape)}"
        )


def _set_nan_for_invalid_targets_(
    target_logprobs: torch.Tensor, targets: torch.Tensor, tgt_lengths: torch.Tensor, vocab_size: int
):
    """
    Set NaN log-probability for the targets (inside the target lengths) that are invalid: labels outside
    the vocabulary, and the target positions after the last target of the tensor `[B, U]` for a target length
    larger than U. The loss of such an utterance is NaN (it can be masked by the loss), as for the loss computed
    from the Joint output. The labels after the target length (padding) are ignored.
    No host-device synchronization.

    Args:
        target_logprobs: log-probabilities of the targets, [B, T, U+1], modified in-place
        targets: targets, [B, U] (not used if U = 0)
        tgt_lengths: target lengths, [B]
        vocab_size: vocabulary size with blank
    """
    num_targets = target_logprobs.shape[2] - 1
    labels = targets[:, :num_targets]
    # no label for the last target position
    is_invalid = torch.ones(target_logprobs.shape[::2], dtype=torch.bool, device=target_logprobs.device)
    is_invalid[:, :num_targets] = (labels < 0) | (labels >= vocab_size)
    is_inside_length = torch.arange(num_targets + 1, device=target_logprobs.device) < tgt_lengths.unsqueeze(1)
    target_logprobs.masked_fill_((is_invalid & is_inside_length).unsqueeze(1), float("nan"))


def _autotune_size_class(num_rows: int) -> int:
    """Class of the number of rows of the joint hidden for the autotune key, see `_AUTOTUNE_MIN_ROWS`"""
    return int(num_rows >= _AUTOTUNE_MIN_ROWS)


def _normalize_pruning_range_starts(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    pruning_range_starts: torch.Tensor | None,
    backward_pruning_range_size: int | None,
    paired: bool = False,
) -> torch.Tensor | None:
    """
    Validate the starts of the pruning ranges for the joint hidden of the pruned lattice [B, T, range_size, hidden]
    and convert them to contiguous int32 [B, T] ([B // 2, T] if `paired`: ranges are shared by the halves of the batch)
    """
    if pruning_range_starts is None:
        return None
    if backward_pruning_range_size is not None:
        raise ValueError("`pruning_range_starts` and `backward_pruning_range_size` can't be used together")
    expected_shape = (joint_hidden.shape[0] // 2 if paired else joint_hidden.shape[0], joint_hidden.shape[1])
    if pruning_range_starts.is_floating_point() or tuple(pruning_range_starts.shape) != expected_shape:
        raise ValueError(
            f"Expected integer `pruning_range_starts` of shape {list(expected_shape)}, "
            f"got {pruning_range_starts.dtype} {list(pruning_range_starts.shape)}"
        )
    if joint_hidden.shape[2] > targets.shape[1] + 1:
        raise ValueError(
            f"Pruning range size {joint_hidden.shape[2]} (joint hidden {list(joint_hidden.shape)}) exceeds "
            f"the number of target positions {targets.shape[1] + 1} (targets {list(targets.shape)})"
        )
    return pruning_range_starts.to(torch.int32).contiguous()


def _normalize_backward_options(
    joint_hidden: torch.Tensor,
    backward_mode: str | FusedVocabBackwardMode | None,
    backward_pruning_range_size: int | None,
) -> tuple[FusedVocabBackwardMode, int | None]:
    """Resolve the automatic backward mode; disable pruning for non-positive widths or widths covering U+1"""
    if backward_pruning_range_size is not None:
        if type(backward_pruning_range_size) is not int:
            raise ValueError("backward_pruning_range_size must be an integer or None")
        if backward_pruning_range_size <= 0 or backward_pruning_range_size >= joint_hidden.shape[2]:
            backward_pruning_range_size = None
    if backward_mode is None:
        backward_mode = _get_default_backward_mode(joint_hidden.device)
    return FusedVocabBackwardMode(backward_mode), backward_pruning_range_size


def _get_default_backward_mode(device: torch.device) -> FusedVocabBackwardMode:
    # Native bf16 atomic add (`red.global.add.noftz.bf16x2`) requires sm_90+; on older GPUs Triton emits
    # compare-and-swap loops, which makes the ATOMIC backward several times slower than the CHUNKED one.
    # Triton < 3.4 does not support BF16 atomics at all.
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 9 and BF16_ATOMICS_AVAILABLE:
        return FusedVocabBackwardMode.ATOMIC
    return FusedVocabBackwardMode.CHUNKED


def _check_atomic_accumulators(*accumulators: torch.Tensor):
    """Atomic backward accumulates the gradients with `tl.atomic_add`: BF16 accumulators require Triton 3.4+"""
    if not BF16_ATOMICS_AVAILABLE and any(accumulator.dtype == torch.bfloat16 for accumulator in accumulators):
        raise RuntimeError(
            "Atomic backward of the fused Joint with BF16 gradients requires Triton 3.4+ (BF16 atomics), "
            f"found Triton {triton.__version__}. Update Triton or use the chunked backward (`backward_mode='chunked'`)."
        )


def _rnnt_joint_fused_vocab_bwd_chunked(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: torch.Tensor,
    tgt_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    log_sum_exp_scores: torch.Tensor,
    grad_target_scores: torch.Tensor,
    grad_blank_scores: torch.Tensor,
    blank_id: int,
    use_high_precision_mm: bool,
    mm_full_precision: bool = False,
    matmul_weight: torch.Tensor | None = None,
    pruning_starts: torch.Tensor | None = None,
    pruning_range_size: int | None = None,
    packed_hidden: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward by row chunks of the flattened batch: Triton kernel recomputes logits and writes grad logits
    [chunk_size, padded_vocab_size] to a bounded buffer, then GEMMs compute grad joint hidden for the chunk
    and accumulate grad weight in full precision. No atomics except for grad bias.

    With `pruning_range_size`, only the rows of the pruning ranges are processed. They are gathered from the dense
    joint hidden [B, T, U+1, hidden_dim] to a buffer (and the gradient is scattered back), unless the joint hidden
    is `packed_hidden`: [B, T, pruning_range_size, hidden_dim], only the rows of the ranges.
    """
    batch_size, src_max_length, num_frame_rows, hidden_dim = joint_hidden.shape
    tgt_max_length_plus_1 = log_sum_exp_scores.shape[2]
    num_hidden_rows = batch_size * src_max_length * num_frame_rows
    flattened_batch_size = batch_size * src_max_length * (pruning_range_size or tgt_max_length_plus_1)
    gather_hidden = pruning_range_size is not None and not packed_hidden
    vocab_size = weight.shape[0]
    device = joint_hidden.device

    use_low_precision_mm = joint_hidden.dtype == torch.bfloat16 and not mm_full_precision and not use_high_precision_mm
    grad_logits_dtype = torch.bfloat16 if use_low_precision_mm else torch.float32
    if matmul_weight is None:
        matmul_weight = weight.to(torch.bfloat16) if use_low_precision_mm else weight
    padded_vocab_size = triton.cdiv(vocab_size, _CHUNKED_BWD_VOCAB_PADDING) * _CHUNKED_BWD_VOCAB_PADDING
    chunk_rows = CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES // (padded_vocab_size * grad_logits_dtype.itemsize)
    chunk_rows = max(chunk_rows // _CHUNKED_BWD_ROWS_MULTIPLE, 1) * _CHUNKED_BWD_ROWS_MULTIPLE
    chunk_rows = max(min(chunk_rows, flattened_batch_size), 1)
    # NB: the size class of the first (largest) chunk is used for all chunks. Autotuning resets the bias gradient
    # accumulated by the kernel (`reset_to_zero`), so it must be possible only for the first chunk:
    # all chunks must have the same autotune key
    autotune_size_class = int(chunk_rows * padded_vocab_size >= _AUTOTUNE_MIN_GRAD_LOGITS)

    joint_hidden_flat = joint_hidden.view(num_hidden_rows, hidden_dim)
    # zero rows for padded vocab: grad logits there are zeros, padding only keeps GEMM operands aligned
    weight_padded = torch.zeros([padded_vocab_size, hidden_dim], dtype=grad_logits_dtype, device=device)
    weight_padded[:vocab_size] = matmul_weight
    grad_logits_buffer = torch.empty([chunk_rows, padded_vocab_size], dtype=grad_logits_dtype, device=device)

    # each element of grad joint hidden is computed by a single GEMM => store directly in the output dtype
    grad_joint_hidden = torch.zeros_like(joint_hidden) if gather_hidden else torch.empty_like(joint_hidden)
    grad_joint_hidden_flat = grad_joint_hidden.view(num_hidden_rows, hidden_dim)
    if gather_hidden:
        gathered_hidden = torch.empty([chunk_rows, hidden_dim], dtype=grad_logits_dtype, device=device)
        gathered_grad_hidden = torch.empty_like(gathered_hidden)
    grad_weight_padded = torch.zeros([padded_vocab_size, hidden_dim], dtype=torch.float32, device=device)
    grad_bias = torch.zeros([vocab_size], dtype=torch.float32, device=device)

    # GEMMs use the dtypes selected above also if backward is called inside autocast
    with cuda_device_guard(joint_hidden), torch.autocast(joint_hidden.device.type, enabled=False):
        for chunk_start in range(0, flattened_batch_size, chunk_rows):
            chunk_size = min(chunk_rows, flattened_batch_size - chunk_start)
            if gather_hidden:
                _copy_pruned_hidden(
                    joint_hidden,
                    gathered_hidden,
                    pruning_starts,
                    src_lengths,
                    tgt_lengths,
                    chunk_start,
                    chunk_size,
                    pruning_range_size,
                    scatter=False,
                )
                joint_hidden_chunk = gathered_hidden[:chunk_size]
            else:
                joint_hidden_chunk = joint_hidden_flat[chunk_start : chunk_start + chunk_size].to(grad_logits_dtype)
            grid = lambda META: (
                triton.cdiv(chunk_size, META["FLATTENED_BATCH_BLOCK"])
                * triton.cdiv(padded_vocab_size, META["VOCAB_BLOCK"]),
            )
            _rnnt_joint_fused_vocab_bwd_grad_logits_triton_kernel[grid](
                # with pruning the kernel reads the rows of the chunk, otherwise - the whole joint hidden
                joint_hidden_ptr=joint_hidden_chunk if pruning_range_size else joint_hidden,
                targets_ptr=targets,
                src_lengths_ptr=src_lengths,
                tgt_lengths_ptr=tgt_lengths,
                weight_ptr=matmul_weight,
                bias_ptr=bias,
                log_sum_exp_ptr=log_sum_exp_scores,
                grad_target_scores_ptr=grad_target_scores,
                grad_blank_scores_ptr=grad_blank_scores,
                grad_logits_out_ptr=grad_logits_buffer,
                grad_bias_out_ptr=grad_bias,
                pruning_starts_ptr=pruning_starts,
                max_src_len=src_max_length,
                max_tgt_len_plus_1=tgt_max_length_plus_1,
                chunk_start=chunk_start,
                chunk_size=chunk_size,
                autotune_size_class=autotune_size_class,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                padded_vocab_size=padded_vocab_size,
                blank_id=blank_id,
                USE_HIGH_PRECISION_MM=use_high_precision_mm,
                MM_FULL_PRECISION=mm_full_precision,
                PRUNING_RANGE=pruning_range_size or 0,
            )
            grad_logits_chunk = grad_logits_buffer[:chunk_size]
            grad_joint_hidden_chunk = (
                gathered_grad_hidden[:chunk_size]
                if gather_hidden
                else grad_joint_hidden_flat[chunk_start : chunk_start + chunk_size]
            )
            if grad_joint_hidden_chunk.dtype == grad_logits_dtype:
                torch.mm(grad_logits_chunk, weight_padded, out=grad_joint_hidden_chunk)
            else:
                grad_joint_hidden_chunk.copy_(torch.mm(grad_logits_chunk, weight_padded))
            if gather_hidden:
                _copy_pruned_hidden(
                    grad_joint_hidden,
                    gathered_grad_hidden,
                    pruning_starts,
                    src_lengths,
                    tgt_lengths,
                    chunk_start,
                    chunk_size,
                    pruning_range_size,
                    scatter=True,
                )
            _addmm_full_precision_(grad_weight_padded, grad_logits_chunk.T, joint_hidden_chunk)

    grad_weight = grad_weight_padded[:vocab_size].to(weight.dtype)
    grad_bias = grad_bias.to(bias.dtype)
    return grad_joint_hidden, grad_weight, grad_bias


def _addmm_full_precision_(accumulator: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor):
    """accumulator += mat1 @ mat2, accumulator can be in higher precision than mat1, mat2 (no rounding of the sum)"""
    if mat1.dtype == accumulator.dtype:
        accumulator.addmm_(mat1, mat2)
        return
    try:
        # bf16 operands, fp32 output: GEMM epilogue adds to the fp32 accumulator directly
        torch.addmm(accumulator, mat1, mat2, out_dtype=accumulator.dtype, out=accumulator)
    except TypeError:
        # older torch without out_dtype: one bf16 rounding per chunk result, chunks are still summed in fp32
        accumulator += torch.mm(mat1, mat2)


def _select_pruning_starts(grad_target, grad_blank, src_lengths, tgt_lengths, range_size, paired=False):
    """
    Select the start of the target window for each frame, maximizing abs(target grad) + abs(blank grad) in the window.
    With `paired`, the batch is a double batch [teacher; student] with identical lengths in both halves:
    one window is selected per teacher/student frame pair from their summed importance, output is [B // 2, T].
    """
    batch_size, max_src_len, max_tgt_len_plus_1 = grad_blank.shape
    num_frames = (batch_size // 2 if paired else batch_size) * max_src_len
    starts = torch.empty([num_frames // max_src_len, max_src_len], dtype=torch.int32, device=grad_blank.device)
    with cuda_device_guard(grad_blank):
        _select_pruning_starts_kernel[(num_frames,)](
            grad_target,
            grad_blank,
            src_lengths,
            tgt_lengths,
            starts,
            max_src_len,
            max_tgt_len_plus_1,
            num_frames,
            range_size,
            BLOCK_SIZE=triton.next_power_of_2(max_tgt_len_plus_1),
            PAIRED=paired,
        )
    return starts


@triton.jit(do_not_specialize=["max_src_len", "max_tgt_len_plus_1", "paired_frame_offset"])
def _select_pruning_starts_kernel(
    grad_target_ptr,
    grad_blank_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    starts_out_ptr,
    max_src_len,
    max_tgt_len_plus_1,
    paired_frame_offset,
    range_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAIRED: tl.constexpr = False,
):
    frame_index = tl.program_id(0)
    batch_index = frame_index // max_src_len
    src_index = frame_index % max_src_len
    src_length = tl.load(src_lengths_ptr + batch_index)
    tgt_length = tl.load(tgt_lengths_ptr + batch_index)
    tgt_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = (src_index < src_length) & (tgt_offsets < max_tgt_len_plus_1)
    grad_target = tl.load(
        grad_target_ptr + frame_index * max_tgt_len_plus_1 + tgt_offsets,
        mask=row_mask & (tgt_offsets < tgt_length),
        other=0.0,
    )
    grad_blank = tl.load(
        grad_blank_ptr + frame_index * max_tgt_len_plus_1 + tgt_offsets,
        mask=row_mask & (tgt_offsets <= tgt_length),
        other=0.0,
    )
    importance = tl.abs(grad_target) + tl.abs(grad_blank)
    if PAIRED:
        # student frame of the pair: same lengths, offset by the half of the double batch
        paired_offsets = (frame_index + paired_frame_offset) * max_tgt_len_plus_1 + tgt_offsets
        paired_grad_target = tl.load(
            grad_target_ptr + paired_offsets, mask=row_mask & (tgt_offsets < tgt_length), other=0.0
        )
        paired_grad_blank = tl.load(
            grad_blank_ptr + paired_offsets, mask=row_mask & (tgt_offsets <= tgt_length), other=0.0
        )
        importance += tl.abs(paired_grad_target) + tl.abs(paired_grad_blank)
    prefix = tl.cumsum(importance)
    end_indices = tl.minimum(tgt_offsets + range_size - 1, BLOCK_SIZE - 1)
    before_indices = tl.maximum(tgt_offsets - 1, 0)
    window_importance = tl.gather(prefix, end_indices, 0) - tl.where(
        tgt_offsets > 0, tl.gather(prefix, before_indices, 0), 0.0
    )
    candidate_mask = tgt_offsets <= tl.maximum(tgt_length + 1 - range_size, 0)
    window_importance = tl.where(candidate_mask, window_importance, float('-inf'))
    # argmax breaks ties toward the first window, including zero-gradient and padded frames.
    start = tl.argmax(window_importance, axis=0, tie_break_left=True)
    # non-finite gradients (e.g., NaN loss): argmax can return a window that is not a candidate
    start = tl.minimum(start, tl.maximum(tgt_length + 1 - range_size, 0))
    tl.store(starts_out_ptr + frame_index, start.to(tl.int32))


@triton.jit
def _load_lattice_rows(
    lattice_offsets, row_mask, targets_ptr, src_lengths_ptr, tgt_lengths_ptr, max_src_len, max_tgt_len_plus_1
):
    """
    For flat offsets in the [B, T, U+1] lattice, return masks of the valid blank/target outputs and target labels.
    The label is -1 (not a label) for the positions without a target. These are also the positions after the last
    target of the tensor `[B, U]` for a target length larger than U (invalid input, rejected by the loss):
    the labels are never read outside the tensor.
    """
    source_target_block_size = max_src_len * max_tgt_len_plus_1
    batch_index = lattice_offsets // source_target_block_size
    batch_offsets = lattice_offsets - batch_index * source_target_block_size
    source_index = batch_offsets // max_tgt_len_plus_1
    target_index = batch_offsets - source_index * max_tgt_len_plus_1

    source_length = tl.load(src_lengths_ptr + batch_index, mask=row_mask, other=0)
    target_length = tl.load(tgt_lengths_ptr + batch_index, mask=row_mask, other=0)

    source_mask = source_index < source_length
    output_blank_mask = row_mask & source_mask & (target_index <= target_length)
    output_target_mask = row_mask & source_mask & (target_index < target_length)
    targets = tl.load(
        targets_ptr + batch_index * (max_tgt_len_plus_1 - 1) + target_index,
        mask=output_target_mask & (target_index < max_tgt_len_plus_1 - 1),
        other=-1,
    )
    return output_blank_mask, output_target_mask, targets


@triton.jit
def _load_lattice_row_grads(
    lattice_offsets,
    output_blank_mask,
    output_target_mask,
    log_sum_exp_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    compute_dtype: tl.constexpr,
):
    """Load log-sum-exp (saved in forward) and upstream target/blank gradients for lattice rows"""
    lse = tl.load(log_sum_exp_ptr + lattice_offsets, mask=output_blank_mask, other=0.0).to(compute_dtype)
    grad_target = tl.load(grad_target_scores_ptr + lattice_offsets, mask=output_target_mask, other=0.0)
    grad_blank = tl.load(grad_blank_scores_ptr + lattice_offsets, mask=output_blank_mask, other=0.0)
    return lse, grad_target.to(compute_dtype), grad_blank.to(compute_dtype)


@triton.jit
def _rnnt_grad_logits(probabilities, grad_target, grad_blank, targets, vocab_offsets, blank_id: tl.constexpr):
    """
    Gradient wrt logits [rows, vocab block] of target/blank log-probabilities given their upstream gradients [rows]
    """
    return (
        -((grad_target + grad_blank)[:, None] * probabilities)
        + tl.where((vocab_offsets == blank_id)[None, :], grad_blank[:, None], 0.0)
        + tl.where(vocab_offsets[None, :] == targets[:, None], grad_target[:, None], 0.0)
    )


@triton.jit
def _get_lattice_offsets(row_offsets, row_mask, pruning_starts_ptr, max_tgt_len_plus_1, PRUNING_RANGE: tl.constexpr):
    if PRUNING_RANGE:
        frame_offsets = row_offsets // PRUNING_RANGE
        starts = tl.load(pruning_starts_ptr + frame_offsets, mask=row_mask, other=0)
        # the range is inside the frame for any starts: offsets outside the lattice are not possible
        starts = tl.minimum(tl.maximum(starts, 0), max_tgt_len_plus_1 - PRUNING_RANGE)
        return frame_offsets * max_tgt_len_plus_1 + starts + row_offsets % PRUNING_RANGE
    else:
        return row_offsets


def _copy_pruned_hidden(
    dense,
    packed,
    starts,
    src_lengths,
    tgt_lengths,
    chunk_start,
    chunk_size,
    range_size,
    scatter,
):
    with cuda_device_guard(dense):
        _copy_pruned_hidden_kernel[(triton.cdiv(chunk_size, 32), triton.cdiv(dense.shape[-1], 128))](
            dense,
            packed,
            starts,
            src_lengths,
            tgt_lengths,
            dense.shape[1],
            dense.shape[2],
            dense.shape[3],
            chunk_start,
            chunk_size,
            PRUNING_RANGE=range_size,
            SCATTER=scatter,
        )


@triton.jit(do_not_specialize=["max_src_len", "max_tgt_len_plus_1", "chunk_start", "chunk_size"])
def _copy_pruned_hidden_kernel(
    dense_ptr,
    packed_ptr,
    starts_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    max_src_len,
    max_tgt_len_plus_1,
    hidden_dim: tl.constexpr,
    chunk_start,
    chunk_size,
    PRUNING_RANGE: tl.constexpr,
    SCATTER: tl.constexpr,
):
    chunk_offsets = tl.program_id(0) * 32 + tl.arange(0, 32)
    row_mask = chunk_offsets < chunk_size
    row_offsets = chunk_start + chunk_offsets
    lattice_offsets = _get_lattice_offsets(row_offsets, row_mask, starts_ptr, max_tgt_len_plus_1, PRUNING_RANGE)
    batch_indices = row_offsets // (max_src_len * PRUNING_RANGE)
    src_indices = row_offsets // PRUNING_RANGE % max_src_len
    tgt_indices = lattice_offsets % max_tgt_len_plus_1
    src_lengths = tl.load(src_lengths_ptr + batch_indices, mask=row_mask, other=0)
    tgt_lengths = tl.load(tgt_lengths_ptr + batch_indices, mask=row_mask, other=0)
    valid_rows = row_mask & (src_indices < src_lengths) & (tgt_indices <= tgt_lengths)
    hidden_offsets = tl.program_id(1) * 128 + tl.arange(0, 128)
    dense_offsets = lattice_offsets.to(tl.int64)[:, None] * hidden_dim + hidden_offsets[None, :]
    packed_offsets = chunk_offsets.to(tl.int64)[:, None] * hidden_dim + hidden_offsets[None, :]
    valid_mask = valid_rows[:, None] & (hidden_offsets < hidden_dim)[None, :]
    packed_mask = row_mask[:, None] & (hidden_offsets < hidden_dim)[None, :]
    if SCATTER:
        values = tl.load(packed_ptr + packed_offsets, mask=valid_mask, other=0.0)
        tl.store(dense_ptr + dense_offsets, values, mask=valid_mask)
    else:
        values = tl.load(dense_ptr + dense_offsets, mask=valid_mask, other=0.0)
        tl.store(packed_ptr + packed_offsets, values, mask=packed_mask)
