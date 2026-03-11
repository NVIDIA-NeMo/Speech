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


from nemo.collections.asr.parts.rnnt_triton.utils_triton import log_add_exp, matmul


@triton.jit
def _rnnt_joint_vocab_fwd_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    target_logprobs_out_ptr,
    blank_logprobs_out_ptr,
    log_sum_exp_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    flattened_batch_size: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    FLATTENED_BATCH_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = (
        compute_dtype
        if USE_HIGH_PRECISION
        else (tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype)
    )

    flattened_batch_block_index = tl.program_id(axis=0)
    flattened_batch_start = flattened_batch_block_index * FLATTENED_BATCH_BLOCK
    flattened_batch_offsets = flattened_batch_start + tl.arange(0, FLATTENED_BATCH_BLOCK)
    flattened_batch_valid_mask = flattened_batch_offsets < flattened_batch_size

    source_target_block_size = max_src_len * max_tgt_len_plus_1
    batch_index = flattened_batch_offsets // source_target_block_size
    batch_offsets = flattened_batch_offsets - batch_index * source_target_block_size
    source_index = batch_offsets // max_tgt_len_plus_1
    target_index = batch_offsets - source_index * max_tgt_len_plus_1

    source_length = tl.load(src_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)
    target_length = tl.load(tgt_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)

    source_mask = source_index < source_length
    target_valid_mask = target_index <= target_length
    target_label_mask = target_index < target_length
    output_blank_mask = flattened_batch_valid_mask & source_mask & target_valid_mask
    output_target_mask = flattened_batch_valid_mask & source_mask & target_label_mask

    vocab_offsets_in_block = tl.arange(0, VOCAB_BLOCK)

    log_sum_exp_score = tl.full([FLATTENED_BATCH_BLOCK], value=float("-inf"), dtype=compute_dtype)
    blank_logits = tl.zeros([FLATTENED_BATCH_BLOCK], dtype=compute_dtype)
    target_logits = tl.zeros([FLATTENED_BATCH_BLOCK], dtype=compute_dtype)

    max_target_len = max_tgt_len_plus_1 - 1
    targets = tl.load(
        targets_ptr + batch_index * max_target_len + target_index,
        mask=flattened_batch_valid_mask & target_label_mask,
        other=0,
    )

    # Create block pointers once before the loops
    NUM_HIDDEN_ITERS: tl.constexpr = (hidden_dim + HIDDEN_BLOCK - 1) // HIDDEN_BLOCK
    HIDDEN_RESET: tl.constexpr = NUM_HIDDEN_ITERS * HIDDEN_BLOCK

    joint_hidden_block_ptr = tl.make_block_ptr(
        base=joint_hidden_ptr,
        shape=(flattened_batch_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(flattened_batch_start, 0),
        block_shape=(FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK),
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

    # Outer loop over vocab chunks
    for vocab_start in tl.range(0, vocab_size, VOCAB_BLOCK):
        vocab_offsets = vocab_start + vocab_offsets_in_block
        vocab_mask = vocab_offsets < vocab_size

        bias_chunk = tl.load(bias_block_ptr, boundary_check=(0,)).to(compute_dtype)

        block_logits = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]
        for _ in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            hidden_chunk = tl.load(joint_hidden_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
            weight_chunk = tl.load(weight_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

            block_logits += matmul(
                hidden_chunk, weight_chunk.T, USE_FP64=USE_FP64, USE_HIGH_PRECISION=USE_HIGH_PRECISION
            ).to(compute_dtype)

            joint_hidden_block_ptr = tl.advance(joint_hidden_block_ptr, (0, HIDDEN_BLOCK))
            weight_block_ptr = tl.advance(weight_block_ptr, (0, HIDDEN_BLOCK))

        # Reset hidden dim, advance vocab dim for next iteration
        joint_hidden_block_ptr = tl.advance(joint_hidden_block_ptr, (0, -HIDDEN_RESET))
        weight_block_ptr = tl.advance(weight_block_ptr, (VOCAB_BLOCK, -HIDDEN_RESET))
        bias_block_ptr = tl.advance(bias_block_ptr, (VOCAB_BLOCK,))

        block_logits = tl.where(vocab_mask[None, :], block_logits, -float("inf"))

        # Online log-sum-exp
        block_logits_max = tl.max(block_logits, axis=-1)  # [TILE]
        block_lse = tl.log(tl.sum(tl.exp(block_logits - block_logits_max[:, None]), axis=-1)) + block_logits_max
        log_sum_exp_score = log_add_exp(log_sum_exp_score, block_lse)

        # Extract blank and target logits from this chunk
        blank_logits += tl.sum(tl.where((vocab_offsets == blank_id)[None, :], block_logits, 0.0), axis=-1)
        target_logits += tl.sum(tl.where(vocab_offsets[None, :] == targets[:, None], block_logits, 0.0), axis=-1)

    tl.store(
        blank_logprobs_out_ptr + flattened_batch_offsets,
        blank_logits - log_sum_exp_score,
        mask=output_blank_mask,
    )

    tl.store(
        target_logprobs_out_ptr + flattened_batch_offsets,
        target_logits - log_sum_exp_score,
        mask=output_target_mask,
    )
    tl.store(
        log_sum_exp_out_ptr + flattened_batch_offsets,
        log_sum_exp_score,
        mask=output_blank_mask,
    )


@triton.jit
def _rnnt_joint_vocab_partial_hidden_bwd_kernel(
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
    max_src_len: int,
    max_tgt_len_plus_1: int,
    flattened_batch_size: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    FLATTENED_BATCH_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    NUM_HIDDEN_BLOCKS: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = (
        compute_dtype
        if USE_HIGH_PRECISION
        else (tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype)
    )

    flattened_batch_block_index = tl.program_id(axis=0)
    flattened_batch_start = flattened_batch_block_index * FLATTENED_BATCH_BLOCK
    flattened_batch_offsets = flattened_batch_start + tl.arange(0, FLATTENED_BATCH_BLOCK)
    flattened_batch_valid_mask = flattened_batch_offsets < flattened_batch_size

    source_target_block_size = max_src_len * max_tgt_len_plus_1
    batch_index = flattened_batch_offsets // source_target_block_size
    batch_offsets = flattened_batch_offsets - batch_index * source_target_block_size
    source_index = batch_offsets // max_tgt_len_plus_1
    target_index = batch_offsets - source_index * max_tgt_len_plus_1

    source_length = tl.load(src_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)
    target_length = tl.load(tgt_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)

    source_mask = source_index < source_length
    target_valid_mask = target_index <= target_length
    target_label_mask = target_index < target_length
    output_blank_mask = flattened_batch_valid_mask & source_mask & target_valid_mask
    output_target_mask = flattened_batch_valid_mask & source_mask & target_label_mask

    vocab_offsets_in_block = tl.arange(0, VOCAB_BLOCK)

    max_target_len = max_tgt_len_plus_1 - 1
    targets = tl.load(
        targets_ptr + batch_index * max_target_len + target_index,
        mask=output_target_mask,
        other=0,
    )

    lse = tl.load(log_sum_exp_ptr + flattened_batch_offsets, mask=output_blank_mask, other=0.0).to(compute_dtype)
    grad_target = tl.load(grad_target_scores_ptr + flattened_batch_offsets, mask=output_target_mask, other=0.0).to(
        compute_dtype
    )
    grad_blank = tl.load(grad_blank_scores_ptr + flattened_batch_offsets, mask=output_blank_mask, other=0.0).to(
        compute_dtype
    )
    sum_grad = grad_target + grad_blank

    # Create block pointers once before the loops
    NUM_HIDDEN_ITERS: tl.constexpr = (hidden_dim + HIDDEN_BLOCK - 1) // HIDDEN_BLOCK
    HIDDEN_RESET: tl.constexpr = NUM_HIDDEN_ITERS * HIDDEN_BLOCK

    joint_hidden_block_ptr = tl.make_block_ptr(
        base=joint_hidden_ptr,
        shape=(flattened_batch_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(flattened_batch_start, 0),
        block_shape=(FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK),
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

    if USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR:
        grad_hidden_acc = tl.zeros([FLATTENED_BATCH_BLOCK, NUM_HIDDEN_BLOCKS, HIDDEN_BLOCK], dtype=compute_dtype)
        hidden_blocks_offsets = tl.arange(0, NUM_HIDDEN_BLOCKS)
    else:
        grad_hidden_block_ptr = tl.make_block_ptr(
            base=grad_joint_hidden_out_ptr,
            shape=(flattened_batch_size, hidden_dim),
            strides=(hidden_dim, 1),
            offsets=(flattened_batch_start, 0),
            block_shape=(FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK),
            order=(1, 0),
        )

    for vocab_start in tl.range(0, vocab_size, VOCAB_BLOCK):
        vocab_offsets = vocab_start + vocab_offsets_in_block
        vocab_mask = vocab_offsets < vocab_size

        bias_chunk = tl.load(bias_block_ptr, boundary_check=(0,)).to(compute_dtype)
        logits_block = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]

        # Inner loop 1: recompute logits
        for _ in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            # hidden_chunk: [FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK]
            hidden_chunk = tl.load(joint_hidden_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
            # weight_chunk: [VOCAB_BLOCK, HIDDEN_BLOCK]
            weight_chunk = tl.load(weight_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

            logits_block += matmul(
                hidden_chunk, weight_chunk.T, USE_FP64=USE_FP64, USE_HIGH_PRECISION=USE_HIGH_PRECISION
            ).to(compute_dtype)

            joint_hidden_block_ptr = tl.advance(joint_hidden_block_ptr, (0, HIDDEN_BLOCK))
            weight_block_ptr = tl.advance(weight_block_ptr, (0, HIDDEN_BLOCK))

        # Reset hidden for joint_hidden only; weight stays at HIDDEN_RESET for reverse Loop 2
        joint_hidden_block_ptr = tl.advance(joint_hidden_block_ptr, (0, -HIDDEN_RESET))

        # Compute grad_logits
        probabilities_block = tl.clamp(tl.exp(logits_block - lse[:, None]), min=0.0, max=1.0)
        grad_logits_block = (
            -(sum_grad[:, None] * probabilities_block)
            + (grad_blank[:, None] * (vocab_offsets == blank_id)[None, :])
            + (grad_target[:, None] * (vocab_offsets[None, :] == targets[:, None]))
        ).to(matmul_dtype)
        grad_logits_block = tl.where(output_blank_mask[:, None] & vocab_mask[None, :], grad_logits_block, 0.0)

        # Inner loop 2: compute grad_hidden (reverse iteration for cache reuse)
        if not USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR:
            grad_hidden_block_ptr = tl.advance(grad_hidden_block_ptr, (0, HIDDEN_RESET))
        for forward_hidden_idx in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            weight_block_ptr = tl.advance(weight_block_ptr, (0, -HIDDEN_BLOCK))
            weight_chunk = tl.load(weight_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

            grad_hidden_delta = matmul(
                grad_logits_block, weight_chunk, USE_FP64=USE_FP64, USE_HIGH_PRECISION=USE_HIGH_PRECISION
            ).to(compute_dtype)

            if USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR:
                reverse_hidden_start = HIDDEN_RESET - HIDDEN_BLOCK - forward_hidden_idx
                grad_hidden_mask = hidden_blocks_offsets == (reverse_hidden_start // HIDDEN_BLOCK)
                grad_hidden_acc += grad_hidden_delta.expand_dims(1) * grad_hidden_mask[None, :, None]
            else:
                grad_hidden_block_ptr = tl.advance(grad_hidden_block_ptr, (0, -HIDDEN_BLOCK))
                old_grad_hidden = tl.load(grad_hidden_block_ptr, boundary_check=(0, 1)).to(compute_dtype)
                tl.store(
                    grad_hidden_block_ptr,
                    (old_grad_hidden + grad_hidden_delta).to(grad_hidden_block_ptr.dtype.element_ty),
                    boundary_check=(0, 1),
                )

        # weight_block_ptr is at hidden 0 after reverse loop; advance vocab only
        weight_block_ptr = tl.advance(weight_block_ptr, (VOCAB_BLOCK, 0))
        bias_block_ptr = tl.advance(bias_block_ptr, (VOCAB_BLOCK,))

    if USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR:
        # Write accumulated grad_hidden to global memory (single write, preserving fp32 precision)
        HIDDEN_DIM_MAX: tl.constexpr = NUM_HIDDEN_BLOCKS * HIDDEN_BLOCK
        hidden_offsets_full = tl.arange(0, HIDDEN_DIM_MAX)
        hidden_mask_full = hidden_offsets_full < hidden_dim
        tl.store(
            grad_joint_hidden_out_ptr + flattened_batch_offsets[:, None] * hidden_dim + hidden_offsets_full[None, :],
            grad_hidden_acc.reshape([FLATTENED_BATCH_BLOCK, HIDDEN_DIM_MAX]),
            mask=flattened_batch_valid_mask[:, None] & hidden_mask_full[None, :],
        )


@triton.jit
def _rnnt_joint_vocab_partial_weight_bias_bwd_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    log_sum_exp_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_weight_out_ptr,
    grad_bias_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    flattened_batch_size: int,
    flattened_batch_split_size: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    FLATTENED_BATCH_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    NUM_HIDDEN_BLOCKS: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_GLOBAL_WEIGHT_GRAD_ACCUMULATOR: tl.constexpr,
    USE_HIGH_PRECISION: tl.constexpr,
):
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    matmul_dtype = (
        compute_dtype
        if USE_HIGH_PRECISION
        else (tl.bfloat16 if weight_ptr.dtype.element_ty == tl.bfloat16 else compute_dtype)
    )

    flattened_batch_split_index = tl.program_id(axis=0)
    vocab_block_index = tl.program_id(axis=1)
    vocab_block_start = vocab_block_index * VOCAB_BLOCK
    vocab_offsets = vocab_block_start + tl.arange(0, VOCAB_BLOCK)
    vocab_mask = vocab_offsets < vocab_size

    split_flattened_batch_start = flattened_batch_split_index * flattened_batch_split_size
    split_flattened_batch_end = tl.minimum(
        split_flattened_batch_start + flattened_batch_split_size, flattened_batch_size
    )

    grad_bias_acc = tl.zeros((VOCAB_BLOCK,), dtype=compute_dtype)
    is_blank_vocab_col = (vocab_offsets == blank_id) & vocab_mask

    if USE_GLOBAL_WEIGHT_GRAD_ACCUMULATOR:
        grad_weight_acc = tl.zeros([VOCAB_BLOCK, NUM_HIDDEN_BLOCKS, HIDDEN_BLOCK], dtype=compute_dtype)
        hidden_blocks_offsets = tl.arange(0, NUM_HIDDEN_BLOCKS)

    # Create block pointers once before the loops
    NUM_HIDDEN_ITERS: tl.constexpr = (hidden_dim + HIDDEN_BLOCK - 1) // HIDDEN_BLOCK
    HIDDEN_RESET: tl.constexpr = NUM_HIDDEN_ITERS * HIDDEN_BLOCK

    joint_hidden_block_ptr = tl.make_block_ptr(
        base=joint_hidden_ptr,
        shape=(flattened_batch_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(split_flattened_batch_start, 0),
        block_shape=(FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK),
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

    for flattened_batch_start in tl.range(
        split_flattened_batch_start, split_flattened_batch_end, FLATTENED_BATCH_BLOCK
    ):
        # iterate over flattened batch
        flattened_batch_offsets = flattened_batch_start + tl.arange(0, FLATTENED_BATCH_BLOCK)
        flattened_batch_mask = flattened_batch_offsets < split_flattened_batch_end

        source_target_block_size = max_src_len * max_tgt_len_plus_1
        batch_index = flattened_batch_offsets // source_target_block_size
        batch_offsets = flattened_batch_offsets - batch_index * source_target_block_size
        source_index = batch_offsets // max_tgt_len_plus_1
        target_index = batch_offsets - source_index * max_tgt_len_plus_1

        source_length = tl.load(src_lengths_ptr + batch_index, mask=flattened_batch_mask, other=0)
        target_length = tl.load(tgt_lengths_ptr + batch_index, mask=flattened_batch_mask, other=0)

        source_mask = source_index < source_length
        target_valid_mask = target_index <= target_length
        target_label_mask = target_index < target_length
        output_blank_mask = flattened_batch_mask & source_mask & target_valid_mask
        output_target_mask = flattened_batch_mask & source_mask & target_label_mask

        targets = tl.load(
            targets_ptr + batch_index * (max_tgt_len_plus_1 - 1) + target_index,
            mask=output_target_mask,
            other=0,
        )

        bias_tile = tl.load(bias_ptr + vocab_offsets, mask=vocab_mask, other=-float("inf")).to(compute_dtype)
        logits_block = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_tile[None, :]

        grad_blank = tl.load(grad_blank_scores_ptr + flattened_batch_offsets, mask=output_blank_mask, other=0.0).to(
            compute_dtype
        )
        grad_target = tl.load(grad_target_scores_ptr + flattened_batch_offsets, mask=output_target_mask, other=0.0).to(
            compute_dtype
        )
        grad_sum = grad_blank + grad_target

        log_sum_exp_scores = tl.load(
            log_sum_exp_ptr + flattened_batch_offsets, mask=flattened_batch_mask, other=0.0
        ).to(compute_dtype)

        # Inner loop 1: recompute logits
        # joint_hidden_tile = tl.zeros([FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK], dtype=matmul_dtype)
        for _ in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            joint_hidden_tile = tl.load(joint_hidden_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
            weight_tile = tl.load(weight_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

            logits_block += matmul(
                joint_hidden_tile, weight_tile.T, USE_FP64=USE_FP64, USE_HIGH_PRECISION=USE_HIGH_PRECISION
            ).to(compute_dtype)

            joint_hidden_block_ptr = tl.advance(joint_hidden_block_ptr, (0, HIDDEN_BLOCK))
            weight_block_ptr = tl.advance(weight_block_ptr, (0, HIDDEN_BLOCK))

        # Reset hidden for weight only; joint_hidden stays at HIDDEN_RESET for reverse Loop 2
        weight_block_ptr = tl.advance(weight_block_ptr, (0, -HIDDEN_RESET))

        probabilities_block = tl.clamp(tl.exp(logits_block - log_sum_exp_scores[:, None]), min=0.0, max=1.0)
        grad_logits_block = (
            -(grad_sum[:, None] * probabilities_block)
            + (grad_blank[:, None] * is_blank_vocab_col[None, :])
            + (grad_target[:, None] * (vocab_offsets[None, :] == targets[:, None]))
        )
        grad_logits_block = tl.where(output_blank_mask[:, None] & vocab_mask[None, :], grad_logits_block, 0.0)

        # compute grad bias addition
        grad_bias_acc += tl.sum(grad_logits_block, axis=0)

        # Inner loop 2: compute grad weight (reverse iteration for cache reuse)
        grad_logits_matmul = grad_logits_block.to(matmul_dtype)
        for forward_hidden_idx in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            reverse_hidden_start = HIDDEN_RESET - HIDDEN_BLOCK - forward_hidden_idx
            hidden_offsets = reverse_hidden_start + tl.arange(0, HIDDEN_BLOCK)
            hidden_mask = hidden_offsets < hidden_dim

            joint_hidden_block_ptr = tl.advance(joint_hidden_block_ptr, (0, -HIDDEN_BLOCK))
            # if forward_hidden_idx > 0:
            joint_hidden_tile = tl.load(joint_hidden_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

            grad_weight_tile = matmul(
                grad_logits_matmul.T, joint_hidden_tile, USE_FP64=USE_FP64, USE_HIGH_PRECISION=USE_HIGH_PRECISION
            ).to(compute_dtype)

            if USE_GLOBAL_WEIGHT_GRAD_ACCUMULATOR:
                grad_weight_mask = hidden_blocks_offsets == (reverse_hidden_start // HIDDEN_BLOCK)
                grad_weight_acc += grad_weight_tile.expand_dims(1) * grad_weight_mask[None, :, None]
            else:
                ptr = grad_weight_out_ptr + flattened_batch_split_index * (hidden_dim * vocab_size)
                old_grad = tl.load(
                    ptr + vocab_offsets[:, None] * hidden_dim + hidden_offsets[None, :],
                    mask=vocab_mask[:, None] & hidden_mask[None, :],
                )
                tl.store(
                    ptr + vocab_offsets[:, None] * hidden_dim + hidden_offsets[None, :],
                    old_grad + grad_weight_tile,
                    mask=vocab_mask[:, None] & hidden_mask[None, :],
                )

        # joint_hidden_block_ptr is at hidden 0 after reverse loop; advance batch only
        joint_hidden_block_ptr = tl.advance(joint_hidden_block_ptr, (FLATTENED_BATCH_BLOCK, 0))

    if USE_GLOBAL_WEIGHT_GRAD_ACCUMULATOR:
        # Atomic add into global grads
        HIDDEN_DIM_MAX: tl.constexpr = NUM_HIDDEN_BLOCKS * HIDDEN_BLOCK
        hidden_offsets_full = tl.arange(0, HIDDEN_DIM_MAX)
        hidden_mask_full = hidden_offsets_full < hidden_dim
        tl.atomic_add(
            grad_weight_out_ptr + vocab_offsets[:, None] * hidden_dim + hidden_offsets_full[None, :],
            grad_weight_acc.reshape([VOCAB_BLOCK, HIDDEN_DIM_MAX]),
            mask=vocab_mask[:, None] & hidden_mask_full[None, :],
        )

    # Atomic add into global grads
    tl.atomic_add(
        grad_bias_out_ptr + vocab_offsets,
        grad_bias_acc,
        mask=vocab_mask,
    )


class RnntJointVocabLogProbs(torch.autograd.Function):
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
        use_high_precision: bool = False,
    ):
        use_fp64 = joint_hidden.dtype == torch.float64
        float_dtype = torch.float64 if use_fp64 else torch.float32

        batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
        flattened_batch_size = batch_size * src_max_length * tgt_max_length_plus_1
        vocab_size = weight.shape[0]
        device = joint_hidden.device

        joint_hidden = joint_hidden.contiguous()
        targets = targets.contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()

        target_logprobs = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1], dtype=float_dtype, device=device
        )
        blank_logprobs = torch.zeros_like(target_logprobs)
        log_sum_exp_scores = torch.empty_like(target_logprobs)

        VOCAB_BLOCK = 64
        HIDDEN_BLOCK = 64
        FLATTENED_BATCH_BLOCK = 128
        flattened_batch_blocks = triton.cdiv(flattened_batch_size, FLATTENED_BATCH_BLOCK)
        forward_num_stages = 1 if use_high_precision else 2
        num_warps = 4

        _rnnt_joint_vocab_fwd_kernel[(flattened_batch_blocks,)](
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            target_logprobs_out_ptr=target_logprobs,
            blank_logprobs_out_ptr=blank_logprobs,
            log_sum_exp_out_ptr=log_sum_exp_scores,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            flattened_batch_size=flattened_batch_size,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            FLATTENED_BATCH_BLOCK=FLATTENED_BATCH_BLOCK,
            HIDDEN_BLOCK=HIDDEN_BLOCK,
            VOCAB_BLOCK=VOCAB_BLOCK,
            USE_FP64=use_fp64,
            USE_HIGH_PRECISION=use_high_precision,
            num_warps=num_warps,
            num_stages=forward_num_stages,
        )

        ctx.save_for_backward(joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, log_sum_exp_scores)
        ctx.blank_id = blank_id
        ctx.use_fp64 = use_fp64
        ctx.use_high_precision = use_high_precision
        return target_logprobs, blank_logprobs

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        (joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, log_sum_exp_scores) = ctx.saved_tensors
        blank_id = ctx.blank_id
        use_fp64 = ctx.use_fp64
        use_high_precision = ctx.use_high_precision
        float_dtype = torch.float64 if use_fp64 else torch.float32

        grad_target_scores = grad_target_scores.contiguous()
        grad_blank_scores = grad_blank_scores.contiguous()

        batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
        flattened_batch_size = batch_size * src_max_length * tgt_max_length_plus_1
        vocab_size = weight.shape[0]
        device = joint_hidden.device

        USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR = False
        FULL_PRECISION_JOINT_GRAD_CALC = use_high_precision  # TODO: make extra param(?)
        hidden_bwd_hidden_block = 64
        num_hidden_blocks = triton.next_power_of_2(triton.cdiv(hidden_dim, hidden_bwd_hidden_block))

        if USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR:
            # Accumulate in fp32 registers, write once — allows native dtype output
            grad_joint_hidden_dtype = joint_hidden.dtype
            hidden_bwd_flattened_batch_block = 16  # reduced to fit 3D register accumulator
        else:
            grad_joint_hidden_dtype = float_dtype if FULL_PRECISION_JOINT_GRAD_CALC else joint_hidden.dtype
            hidden_bwd_flattened_batch_block = 64

        grad_joint_hidden = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim],
            dtype=grad_joint_hidden_dtype,
            device=device,
        )

        if use_high_precision or joint_hidden.dtype != torch.bfloat16:
            hidden_bwd_vocab_block = 64
        else:
            hidden_bwd_vocab_block = 128
        hidden_bwd_num_warps = 4
        hidden_bwd_num_stages = 2

        hidden_bwd_flattened_batch_blocks = triton.cdiv(flattened_batch_size, hidden_bwd_flattened_batch_block)

        _rnnt_joint_vocab_partial_hidden_bwd_kernel[(hidden_bwd_flattened_batch_blocks,)](
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            log_sum_exp_ptr=log_sum_exp_scores,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            grad_joint_hidden_out_ptr=grad_joint_hidden,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            flattened_batch_size=flattened_batch_size,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            FLATTENED_BATCH_BLOCK=hidden_bwd_flattened_batch_block,
            HIDDEN_BLOCK=hidden_bwd_hidden_block,
            NUM_HIDDEN_BLOCKS=num_hidden_blocks,
            VOCAB_BLOCK=hidden_bwd_vocab_block,
            USE_FP64=use_fp64,
            USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR=USE_GLOBAL_HIDDEN_GRAD_ACCUMULATOR,
            USE_HIGH_PRECISION=use_high_precision,
            num_warps=hidden_bwd_num_warps,
            num_stages=hidden_bwd_num_stages,
        )

        USE_GLOBAL_WEIGHT_GRAD_ACCUMULATOR = False  # disable for now
        HIDDEN_BLOCK = 64
        num_hidden_blocks = triton.next_power_of_2(triton.cdiv(hidden_dim, HIDDEN_BLOCK))
        VOCAB_BLOCK = 16 if USE_GLOBAL_WEIGHT_GRAD_ACCUMULATOR else 64
        FLATTENED_BATCH_BLOCK = 128
        vocab_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK)
        FLATTENED_BATCH_SPLITS = 64
        flattened_batch_split_size = triton.cdiv(flattened_batch_size, FLATTENED_BATCH_SPLITS)

        # grad output variables
        # grad_weight = torch.zeros([vocab_size, hidden_dim], dtype=float_dtype, device=device)
        grad_weight = torch.zeros([FLATTENED_BATCH_SPLITS, vocab_size, hidden_dim], dtype=float_dtype, device=device)
        grad_bias = torch.zeros([vocab_size], dtype=float_dtype, device=device)

        weight_bias_num_warps = 4
        weight_bias_num_stages = 2

        _rnnt_joint_vocab_partial_weight_bias_bwd_kernel[(FLATTENED_BATCH_SPLITS, vocab_blocks)](
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            log_sum_exp_ptr=log_sum_exp_scores,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            grad_weight_out_ptr=grad_weight,
            grad_bias_out_ptr=grad_bias,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            flattened_batch_size=flattened_batch_size,
            flattened_batch_split_size=flattened_batch_split_size,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            FLATTENED_BATCH_BLOCK=FLATTENED_BATCH_BLOCK,
            HIDDEN_BLOCK=HIDDEN_BLOCK,
            NUM_HIDDEN_BLOCKS=num_hidden_blocks,
            VOCAB_BLOCK=VOCAB_BLOCK,
            USE_FP64=use_fp64,
            USE_HIGH_PRECISION=use_high_precision,
            USE_GLOBAL_WEIGHT_GRAD_ACCUMULATOR=USE_GLOBAL_WEIGHT_GRAD_ACCUMULATOR,
            num_warps=weight_bias_num_warps,
            num_stages=weight_bias_num_stages,
        )

        # convert grad to desired dtype
        grad_weight = grad_weight.sum(dim=0).to(weight.dtype)
        grad_bias = grad_bias.to(bias.dtype)
        grad_joint_hidden = grad_joint_hidden.to(joint_hidden.dtype)

        return grad_joint_hidden, None, None, None, grad_weight, grad_bias, None, None


def rnnt_joint_vocab_logprobs_triton(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    use_high_precision: bool = False,
):
    target_logprobs, blank_logprobs = RnntJointVocabLogProbs.apply(
        joint_hidden,
        targets,
        tgt_lengths,
        src_lengths,
        weight,
        bias,
        blank_id,
        use_high_precision,
    )
    return target_logprobs, blank_logprobs
