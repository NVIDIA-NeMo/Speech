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

import random

import torch
import triton
import triton.language as tl


from nemo.collections.asr.parts.turbo_transducer.utils_triton import (
    dropout_hash_keep_mask,
    log_add_exp,
    matmul,
    mix_dropout_seed,
)


def _fwd_autotune_configs():
    configs = []
    # (encoder block, predictor block, hidden block, vocab block, num warps, num stages, swizzle group, use .cg)
    # The 8x8x128 case is the original tile. Larger encoder/predictor tiles reduce
    # program count; larger vocab tiles reduce repeated ReLU/dropout work per output.
    candidates = [
        (8, 8, 32, 128, 4, 3, 4),
        (8, 8, 32, 256, 8, 2, 4),
        (4, 8, 32, 512, 8, 2, 4),
        (8, 4, 32, 512, 8, 2, 4),
    ]
    for candidate in candidates:
        (
            encoder_block,
            predictor_block,
            hidden_block,
            vocab_block,
            num_warps,
            num_stages,
            group_size,
        ) = candidate
        configs.append(
            triton.Config(
                {
                    "ENCODER_BLOCK": encoder_block,
                    "PREDICTOR_BLOCK": predictor_block,
                    "HIDDEN_BLOCK": hidden_block,
                    "VOCAB_BLOCK": vocab_block,
                    "GROUP_SIZE": group_size,
                },
                num_warps=num_warps,
                num_stages=num_stages,
            )
        )
    return configs


def _bwd_autotune_configs():
    configs = []
    # (encoder block, predictor block, hidden block, vocab block, num warps, num stages, swizzle group)
    # Backward holds [encoder_block * predictor_block, vocab_block] logits and recomputes hidden tiles.
    # Keep HIDDEN_BLOCK at 32 for lower register pressure, and tune the source-target/vocab tiling separately
    # for small test shapes and the target 1024/8192 vocab regimes.
    candidates = [
        (4, 4, 32, 32, 4, 2, 16),
        (8, 8, 32, 128, 4, 2, 64),
        (8, 8, 32, 256, 8, 2, 64),
        (32, 4, 32, 256, 8, 2, 64),
        (16, 4, 32, 512, 8, 2, 64),
        (16, 16, 32, 128, 8, 2, 64),
    ]
    for candidate in candidates:
        (
            encoder_block,
            predictor_block,
            hidden_block,
            vocab_block,
            num_warps,
            num_stages,
            group_size,
        ) = candidate
        configs.append(
            triton.Config(
                {
                    "ENCODER_BLOCK": encoder_block,
                    "PREDICTOR_BLOCK": predictor_block,
                    "HIDDEN_BLOCK": hidden_block,
                    "VOCAB_BLOCK": vocab_block,
                    "GROUP_SIZE": group_size,
                },
                num_warps=num_warps,
                num_stages=num_stages,
            )
        )
    return configs


@triton.autotune(
    configs=_fwd_autotune_configs(),
    key=["hidden_dim", "vocab_size", "USE_DROPOUT", "USE_HIGH_PRECISION_MM", "MM_FULL_PRECISION"],
    cache_results=True,
)
@triton.jit
def _rnnt_fused_joint_fwd_triton_kernel(
    encoder_output_ptr,
    predictor_output_ptr,
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
    mixed_dropout_seed: int,
    dropout_threshold: int,
    dropout_inv_keep_prob: float,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    ENCODER_BLOCK: tl.constexpr,
    PREDICTOR_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    USE_HIGH_PRECISION_MM: tl.constexpr = False,
    MM_FULL_PRECISION: tl.constexpr = False,
):
    """
    Forward kernel for fused RNN-T Joint + log-softmax.

    Each program handles a tile of [ENCODER_BLOCK, PREDICTOR_BLOCK] positions.
    For each tile:
    1. Loop over vocab chunks (outer), and hidden chunks (inner) to compute logits
    2. Online log-softmax across vocab chunks
    3. Extract target/blank log-probs

    Accumulation uses float32. BF16 projected inputs use BF16 matmul operands unless MM_FULL_PRECISION or
    USE_HIGH_PRECISION_MM is set. FP32 operands use TF32 by default, or IEEE math with USE_HIGH_PRECISION_MM.
    """
    batch_i = tl.program_id(axis=0)
    source_block_i, target_block_i = tl.swizzle2d(
        tl.program_id(axis=1),
        tl.program_id(axis=2),
        tl.num_programs(axis=1),
        tl.num_programs(axis=2),
        GROUP_SIZE,
    )
    source_i_start = source_block_i * ENCODER_BLOCK
    target_i_start = target_block_i * PREDICTOR_BLOCK

    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.load(tgt_lengths_ptr + batch_i)

    if source_i_start >= source_len or target_i_start > target_len:
        return

    compute_dtype = tl.float32
    matmul_dtype = (
        tl.bfloat16
        if (
            (
                encoder_output_ptr.dtype.element_ty == tl.bfloat16
                and predictor_output_ptr.dtype.element_ty == tl.bfloat16
            )
            and (not USE_HIGH_PRECISION_MM)
            and (not MM_FULL_PRECISION)
        )
        else compute_dtype
    )
    indices_dtype = tl.int32  # TODO: make conditional, use int64 if needed

    source_offsets = source_i_start + tl.arange(0, ENCODER_BLOCK)
    target_offsets = target_i_start + tl.arange(0, PREDICTOR_BLOCK)
    source_mask = source_offsets < source_len
    target_valid_mask = target_offsets <= target_len  # blank is valid at u == target_len
    target_label_mask = target_offsets < target_len  # target labels exist at u < target_len
    NUM_TILE_ELEMENTS: tl.constexpr = ENCODER_BLOCK * PREDICTOR_BLOCK
    tile_base_offsets = (batch_i * max_src_len + source_offsets).to(indices_dtype)[
        :, None
    ] * max_tgt_len_plus_1 + target_offsets[None, :].to(indices_dtype)

    vocab_chunk_offsets = tl.arange(0, VOCAB_BLOCK)

    # Initialize log-sum-exp accumulator, blank and target logits
    log_sum_exp_score = tl.full([NUM_TILE_ELEMENTS], value=float("-inf"), dtype=compute_dtype)
    blank_logits = tl.zeros([NUM_TILE_ELEMENTS], dtype=compute_dtype)
    target_logits = tl.zeros([NUM_TILE_ELEMENTS], dtype=compute_dtype)

    # Load target labels with batch offset
    max_tgt_len = max_tgt_len_plus_1 - 1
    targets_predictor = tl.load(targets_ptr + batch_i * max_tgt_len + target_offsets, mask=target_label_mask, other=0)
    targets = targets_predictor[None, :].broadcast_to([ENCODER_BLOCK, PREDICTOR_BLOCK]).reshape([NUM_TILE_ELEMENTS])

    # Create block pointers for encoder, predictor, weight, and bias
    NUM_HIDDEN_ITERS: tl.constexpr = tl.cdiv(hidden_dim, HIDDEN_BLOCK)
    HIDDEN_RESET: tl.constexpr = NUM_HIDDEN_ITERS * HIDDEN_BLOCK

    enc_block_ptr = tl.make_block_ptr(
        base=encoder_output_ptr + batch_i * max_src_len * hidden_dim,
        shape=(source_len, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(source_i_start.to(tl.int32), 0),
        block_shape=(ENCODER_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    pred_block_ptr = tl.make_block_ptr(
        base=predictor_output_ptr + batch_i * max_tgt_len_plus_1 * hidden_dim,
        shape=(target_len + 1, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(target_i_start.to(tl.int32), 0),
        block_shape=(PREDICTOR_BLOCK, HIDDEN_BLOCK),
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
        vocab_offsets = vocab_start + vocab_chunk_offsets
        vocab_mask = vocab_offsets < vocab_size
        contains_blank = vocab_start <= blank_id and blank_id < vocab_start + VOCAB_BLOCK

        bias_chunk = tl.load(bias_block_ptr, boundary_check=(0,), eviction_policy="evict_last").to(compute_dtype)

        block_logits = tl.zeros([NUM_TILE_ELEMENTS, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]

        # Inner loop over hidden dimension chunks
        for hidden_start in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            enc_chunk = tl.load(enc_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
            pred_chunk = tl.load(pred_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

            # hidden = enc + pred -> [ENC, PRED, HIDDEN_BLOCK] -> [TILE, HIDDEN_BLOCK]
            hidden_chunk = enc_chunk[:, None, :] + pred_chunk[None, :, :]
            hidden_chunk = tl.where(hidden_chunk > 0, hidden_chunk, 0)
            if USE_DROPOUT:
                hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
                dropout_offsets = tile_base_offsets[:, :, None] * hidden_dim + hidden_offsets[None, None, :].to(
                    indices_dtype
                )
                dropout_keep_mask = dropout_hash_keep_mask(mixed_dropout_seed, dropout_offsets, dropout_threshold)
                hidden_chunk = tl.where(
                    dropout_keep_mask,
                    hidden_chunk * dropout_inv_keep_prob.to(matmul_dtype),
                    0,
                )
            hidden_chunk = hidden_chunk.reshape([NUM_TILE_ELEMENTS, HIDDEN_BLOCK])

            weight_chunk = tl.load(weight_block_ptr, boundary_check=(0, 1), eviction_policy="evict_last").to(
                matmul_dtype
            )

            block_logits += matmul(hidden_chunk, weight_chunk.T, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM).to(
                compute_dtype
            )

            enc_block_ptr = tl.advance(enc_block_ptr, (0, HIDDEN_BLOCK))
            pred_block_ptr = tl.advance(pred_block_ptr, (0, HIDDEN_BLOCK))
            weight_block_ptr = tl.advance(weight_block_ptr, (0, HIDDEN_BLOCK))

        # Reset hidden dim for enc/pred/weight, advance vocab for weight/bias
        enc_block_ptr = tl.advance(enc_block_ptr, (0, -HIDDEN_RESET))
        pred_block_ptr = tl.advance(pred_block_ptr, (0, -HIDDEN_RESET))
        weight_block_ptr = tl.advance(weight_block_ptr, (VOCAB_BLOCK, -HIDDEN_RESET))
        bias_block_ptr = tl.advance(bias_block_ptr, (VOCAB_BLOCK,))

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

    # Output index in [B, T, U+1] grid
    indices_grid = (batch_i * max_src_len + source_offsets[:, None]) * max_tgt_len_plus_1 + target_offsets[None, :]
    tile_valid_mask = source_mask[:, None] & target_valid_mask[None, :]

    # blank logprobs (valid for all u in [0, target_len])
    tl.store(
        blank_logprobs_out_ptr + indices_grid,
        (blank_logits - log_sum_exp_score).reshape([ENCODER_BLOCK, PREDICTOR_BLOCK]),
        mask=tile_valid_mask,
    )

    # Store target logprobs (valid only for u < target_len)
    output_target_mask = source_mask[:, None] & target_label_mask[None, :]
    tl.store(
        target_logprobs_out_ptr + indices_grid,
        (target_logits - log_sum_exp_score).reshape([ENCODER_BLOCK, PREDICTOR_BLOCK]),
        mask=output_target_mask,
    )
    tl.store(
        log_sum_exp_out_ptr + indices_grid,
        log_sum_exp_score.reshape([ENCODER_BLOCK, PREDICTOR_BLOCK]),
        mask=tile_valid_mask,
    )


@triton.autotune(
    configs=_bwd_autotune_configs(),
    key=["hidden_dim", "vocab_size", "USE_DROPOUT", "USE_HIGH_PRECISION_MM", "MM_FULL_PRECISION"],
    reset_to_zero=["grad_encoder_out_ptr", "grad_predictor_out_ptr", "grad_weight_out_ptr", "grad_bias_out_ptr"],
    cache_results=True,
)
@triton.jit
def _rnnt_fused_joint_bwd_triton_kernel(
    encoder_output_ptr,
    predictor_output_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    log_sum_exp_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_encoder_out_ptr,
    grad_predictor_out_ptr,
    grad_weight_out_ptr,
    grad_bias_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    mixed_dropout_seed: int,
    dropout_threshold: int,
    dropout_p: float,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    ENCODER_BLOCK: tl.constexpr,
    PREDICTOR_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    GROUP_SIZE: tl.constexpr = 4,
    USE_HIGH_PRECISION_MM: tl.constexpr = False,
    MM_FULL_PRECISION: tl.constexpr = False,
):
    compute_dtype = tl.float32
    matmul_dtype = (
        tl.bfloat16
        if (
            (
                encoder_output_ptr.dtype.element_ty == tl.bfloat16
                and predictor_output_ptr.dtype.element_ty == tl.bfloat16
            )
            and (not USE_HIGH_PRECISION_MM)
            and (not MM_FULL_PRECISION)
        )
        else compute_dtype
    )
    indices_dtype = tl.int32  # TODO: make conditional, use int64 if needed
    dropout_inv_keep_prob = 1.0 / (1 - dropout_p) if USE_DROPOUT else 1.0
    dropout_inv_keep_prob = dropout_inv_keep_prob.to(matmul_dtype)

    total_encoder_blocks = tl.cdiv(max_src_len, ENCODER_BLOCK)
    total_predictor_blocks = tl.cdiv(max_tgt_len_plus_1, PREDICTOR_BLOCK)
    source_target_blocks_per_batch = total_encoder_blocks * total_predictor_blocks

    batch_source_target_block, vocab_block_i = tl.swizzle2d(
        tl.program_id(axis=0),
        tl.program_id(axis=1),
        tl.num_programs(axis=0),
        tl.num_programs(axis=1),
        GROUP_SIZE,
    )
    batch_i = batch_source_target_block // source_target_blocks_per_batch
    enc_x_pred_block = batch_source_target_block - batch_i * source_target_blocks_per_batch
    encoder_i = enc_x_pred_block // total_predictor_blocks
    predictor_i = enc_x_pred_block - encoder_i * total_predictor_blocks

    source_len = tl.load(src_lengths_ptr + batch_i)
    target_len = tl.load(tgt_lengths_ptr + batch_i)

    vocab_block_start = vocab_block_i * VOCAB_BLOCK
    vocab_offsets = vocab_block_start + tl.arange(0, VOCAB_BLOCK)
    vocab_mask = vocab_offsets < vocab_size
    is_blank_vocab_col = (vocab_offsets == blank_id) & vocab_mask

    grad_bias_acc = tl.zeros((VOCAB_BLOCK,), dtype=compute_dtype)

    NUM_TILE_ELEMENTS: tl.constexpr = ENCODER_BLOCK * PREDICTOR_BLOCK
    max_tgt_len = max_tgt_len_plus_1 - 1

    NUM_HIDDEN_ITERS: tl.constexpr = (hidden_dim + HIDDEN_BLOCK - 1) // HIDDEN_BLOCK
    HIDDEN_RESET: tl.constexpr = NUM_HIDDEN_ITERS * HIDDEN_BLOCK

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(vocab_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(vocab_block_start, 0),
        block_shape=(VOCAB_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )

    enc_block_ptr = tl.make_block_ptr(
        base=encoder_output_ptr + batch_i * max_src_len * hidden_dim,
        shape=(source_len, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(encoder_i * ENCODER_BLOCK, 0),
        block_shape=(ENCODER_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    pred_block_ptr = tl.make_block_ptr(
        base=predictor_output_ptr + batch_i * max_tgt_len_plus_1 * hidden_dim,
        shape=(target_len + 1, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(predictor_i * PREDICTOR_BLOCK, 0),
        block_shape=(PREDICTOR_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )

    source_offsets = encoder_i * ENCODER_BLOCK + tl.arange(0, ENCODER_BLOCK)
    target_offsets = predictor_i * PREDICTOR_BLOCK + tl.arange(0, PREDICTOR_BLOCK)
    tile_base_offsets = (batch_i * max_src_len + source_offsets).to(indices_dtype)[
        :, None
    ] * max_tgt_len_plus_1 + target_offsets[None, :].to(indices_dtype)
    source_mask = source_offsets < source_len
    target_valid_mask = target_offsets <= target_len
    target_label_mask = target_offsets < target_len

    targets_predictor = tl.load(targets_ptr + batch_i * max_tgt_len + target_offsets, mask=target_label_mask, other=0)
    targets_expanded = (
        targets_predictor[None, :].broadcast_to([ENCODER_BLOCK, PREDICTOR_BLOCK]).reshape([NUM_TILE_ELEMENTS])
    )

    indices_grid = (batch_i * max_src_len + source_offsets[:, None]) * max_tgt_len_plus_1 + target_offsets[None, :]
    tile_valid_mask = source_mask[:, None] & target_valid_mask[None, :]
    target_store_mask = source_mask[:, None] & target_label_mask[None, :]
    tile_flat_mask = tile_valid_mask.reshape([NUM_TILE_ELEMENTS])

    lse = (
        tl.load(log_sum_exp_ptr + indices_grid, mask=tile_valid_mask, other=0.0)
        .reshape([NUM_TILE_ELEMENTS])
        .to(compute_dtype)
    )
    grad_target = (
        tl.load(grad_target_scores_ptr + indices_grid, mask=target_store_mask, other=0.0)
        .reshape([NUM_TILE_ELEMENTS])
        .to(compute_dtype)
    )
    grad_blank = (
        tl.load(grad_blank_scores_ptr + indices_grid, mask=tile_valid_mask, other=0.0)
        .reshape([NUM_TILE_ELEMENTS])
        .to(compute_dtype)
    )
    sum_grad = grad_blank + grad_target

    source_flat_indices = (batch_i * max_src_len + source_offsets).to(indices_dtype)
    target_flat_indices = (batch_i * max_tgt_len_plus_1 + target_offsets).to(indices_dtype)

    bias_tile = tl.load(
        bias_ptr + vocab_offsets, mask=vocab_mask, other=-float("inf"), eviction_policy="evict_last"
    ).to(compute_dtype)
    logits_block = tl.zeros([NUM_TILE_ELEMENTS, VOCAB_BLOCK], dtype=compute_dtype) + bias_tile[None, :]

    # Inner loop 1: recompute logits
    for hidden_start in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        enc_chunk = tl.load(enc_block_ptr, boundary_check=(0, 1), eviction_policy="evict_last").to(matmul_dtype)
        pred_chunk = tl.load(pred_block_ptr, boundary_check=(0, 1), eviction_policy="evict_last").to(matmul_dtype)

        hidden_chunk = enc_chunk[:, None, :] + pred_chunk[None, :, :]
        hidden_chunk = tl.where(hidden_chunk > 0, hidden_chunk * dropout_inv_keep_prob, 0)
        if USE_DROPOUT:
            hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
            dropout_offsets = tile_base_offsets[:, :, None] * hidden_dim + hidden_offsets[None, None, :].to(
                indices_dtype
            )
            dropout_keep_mask = dropout_hash_keep_mask(mixed_dropout_seed, dropout_offsets, dropout_threshold)
            hidden_chunk = tl.where(dropout_keep_mask, hidden_chunk, 0)
        hidden_chunk = hidden_chunk.reshape([NUM_TILE_ELEMENTS, HIDDEN_BLOCK])
        # weight_chunk: [VOCAB_BLOCK, HIDDEN_BLOCK]
        weight_chunk = tl.load(weight_block_ptr, boundary_check=(0, 1), eviction_policy="evict_last").to(matmul_dtype)

        logits_block += matmul(hidden_chunk, weight_chunk.T, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM).to(
            compute_dtype
        )

        enc_block_ptr = tl.advance(enc_block_ptr, (0, HIDDEN_BLOCK))
        pred_block_ptr = tl.advance(pred_block_ptr, (0, HIDDEN_BLOCK))
        weight_block_ptr = tl.advance(weight_block_ptr, (0, HIDDEN_BLOCK))

    # Compute grad_logits
    probabilities_block = tl.exp(logits_block - lse[:, None])
    grad_logits_block = (
        -(sum_grad[:, None] * probabilities_block)
        + (grad_blank[:, None] * is_blank_vocab_col[None, :])
        + (grad_target[:, None] * (vocab_offsets[None, :] == targets_expanded[:, None]))
    )
    grad_logits_block = tl.where(tile_flat_mask[:, None] & vocab_mask[None, :], grad_logits_block, 0.0)
    grad_bias_acc += tl.sum(grad_logits_block, axis=0)

    # Inner loop 2: compute grad_hidden in reverse iteration for cache reuse.
    # Keep this separate from grad_weight to reduce live tensor pressure.
    grad_logits_matmul = grad_logits_block.to(matmul_dtype)
    for forward_hidden_idx in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        reverse_hidden_start = HIDDEN_RESET - HIDDEN_BLOCK - forward_hidden_idx
        hidden_offsets = reverse_hidden_start + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < hidden_dim

        enc_block_ptr = tl.advance(enc_block_ptr, (0, -HIDDEN_BLOCK))
        pred_block_ptr = tl.advance(pred_block_ptr, (0, -HIDDEN_BLOCK))
        weight_block_ptr = tl.advance(weight_block_ptr, (0, -HIDDEN_BLOCK))

        enc_chunk = tl.load(enc_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
        pred_chunk = tl.load(pred_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
        weight_chunk = tl.load(weight_block_ptr, boundary_check=(0, 1), eviction_policy="evict_last").to(matmul_dtype)

        hidden_chunk = enc_chunk[:, None, :] + pred_chunk[None, :, :]
        hidden_chunk = tl.where(hidden_chunk > 0, hidden_chunk * dropout_inv_keep_prob, 0)
        if USE_DROPOUT:
            dropout_offsets = tile_base_offsets[:, :, None] * hidden_dim + hidden_offsets[None, None, :].to(
                indices_dtype
            )
            dropout_keep_mask = dropout_hash_keep_mask(mixed_dropout_seed, dropout_offsets, dropout_threshold)
            dropout_relu_mask = dropout_keep_mask & hidden_mask[None, :] & (hidden_chunk > 0.0)
        else:
            dropout_relu_mask = hidden_mask[None, :] & (hidden_chunk > 0.0)

        dropout_relu_mask = dropout_relu_mask.reshape([NUM_TILE_ELEMENTS, HIDDEN_BLOCK])

        grad_hidden_delta = (
            matmul(grad_logits_matmul, weight_chunk, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM).to(compute_dtype)
            * dropout_inv_keep_prob
        )
        grad_hidden_delta = tl.where(dropout_relu_mask, grad_hidden_delta, 0.0)
        grad_hidden_delta = grad_hidden_delta.reshape([ENCODER_BLOCK, PREDICTOR_BLOCK, HIDDEN_BLOCK])

        grad_encoder_delta = tl.sum(grad_hidden_delta, axis=1)
        tl.atomic_add(
            grad_encoder_out_ptr + source_flat_indices[:, None] * hidden_dim + hidden_offsets[None, :],
            grad_encoder_delta,
            mask=source_mask[:, None] & hidden_mask[None, :],
            sem="relaxed",  # no need to guarantee order of adding - no read inside kernel
        )

        grad_predictor_delta = tl.sum(grad_hidden_delta, axis=0)
        tl.atomic_add(
            grad_predictor_out_ptr + target_flat_indices[:, None] * hidden_dim + hidden_offsets[None, :],
            grad_predictor_delta,
            mask=target_valid_mask[:, None] & hidden_mask[None, :],
            sem="relaxed",  # no need to guarantee order of adding - no read inside kernel
        )

    # Inner loop 3: compute grad_weight. Recompute hidden in a lower-pressure loop.
    for hidden_start in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < hidden_dim

        enc_chunk = tl.load(enc_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
        pred_chunk = tl.load(pred_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

        hidden_chunk = enc_chunk[:, None, :] + pred_chunk[None, :, :]
        hidden_chunk = tl.where(hidden_chunk > 0, hidden_chunk * dropout_inv_keep_prob, 0)
        if USE_DROPOUT:
            dropout_offsets = tile_base_offsets[:, :, None] * hidden_dim + hidden_offsets[None, None, :].to(
                indices_dtype
            )
            dropout_keep_mask = dropout_hash_keep_mask(mixed_dropout_seed, dropout_offsets, dropout_threshold)
            hidden_chunk = tl.where(dropout_keep_mask, hidden_chunk, 0)

        hidden_chunk = hidden_chunk.reshape([NUM_TILE_ELEMENTS, HIDDEN_BLOCK])

        grad_weight_delta = matmul(grad_logits_matmul.T, hidden_chunk, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM).to(
            compute_dtype
        )
        tl.atomic_add(
            grad_weight_out_ptr + vocab_offsets[:, None] * hidden_dim + hidden_offsets[None, :],
            grad_weight_delta,
            mask=vocab_mask[:, None] & hidden_mask[None, :],
            sem="relaxed",  # no need to guarantee order of adding - no read inside kernel
        )

        enc_block_ptr = tl.advance(enc_block_ptr, (0, HIDDEN_BLOCK))
        pred_block_ptr = tl.advance(pred_block_ptr, (0, HIDDEN_BLOCK))

    tl.atomic_add(
        grad_bias_out_ptr + vocab_offsets,
        grad_bias_acc,
        mask=vocab_mask,
        sem="relaxed",  # no need to guarantee order of adding - no read inside kernel
    )


class _RnntFusedJointTritonFunction(torch.autograd.Function):
    """
    Function to calculate log probabilities for target and blank labels for RNN-T, supporting torch.autograd.
    Fuses Joint network (linear + relu + linear) with log-softmax to avoid materializing large intermediate tensors.
    """

    @staticmethod
    def forward(
        ctx,
        encoder_output_projected: torch.Tensor,
        predictor_output_projected: torch.Tensor,
        targets: torch.Tensor,
        tgt_lengths: torch.Tensor,
        src_lengths: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        blank_id: int,
        activation: str = "relu",
        dropout_p: float = 0.0,
        dropout_seed: int | None = None,
        use_high_precision_mm: bool = False,
        input_grad_full_precision: bool = True,
        weight_grad_full_precision: bool = True,
        mm_full_precision: bool = False,
    ):
        if activation != "relu":
            raise NotImplementedError("Only relu activation is supported")

        if dropout_p < 0.0 or dropout_p >= 1.0:
            raise ValueError("dropout_p must be in range [0.0, 1.0)")
        use_dropout = dropout_p > 0.0
        dropout_keep_prob = 1.0 - dropout_p
        dropout_inv_keep_prob = 1.0 / dropout_keep_prob if use_dropout else 1.0
        dropout_threshold = int(dropout_p * torch.iinfo(torch.int32).max) if use_dropout else 0
        if use_dropout:
            if dropout_seed is None:
                dropout_seed = random.randint(0, 2**31 - 1)
            elif not (0 <= dropout_seed < 2**31):
                raise ValueError("dropout_seed must be in range [0, 2**31)")
            mixed_dropout_seed = mix_dropout_seed(dropout_seed)
        else:
            dropout_seed = 0
            mixed_dropout_seed = 0

        float_dtype = torch.float32

        encoder_output_projected = encoder_output_projected.contiguous()
        predictor_output_projected = predictor_output_projected.contiguous()
        targets = targets.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        matmul_weight = (
            weight.to(torch.bfloat16)
            if encoder_output_projected.dtype == predictor_output_projected.dtype == torch.bfloat16
            and not mm_full_precision
            and not use_high_precision_mm
            else weight
        )

        device = encoder_output_projected.device
        batch_size, src_max_length, hidden_dim = encoder_output_projected.shape
        tgt_max_length_plus_1 = predictor_output_projected.shape[1]
        vocab_size = weight.shape[0]

        target_logprobs = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1], dtype=float_dtype, device=device
        )
        blank_logprobs = torch.zeros_like(target_logprobs)
        log_sum_exp_scores = torch.empty_like(target_logprobs)

        fwd_grid = lambda META: (
            batch_size,
            triton.cdiv(src_max_length, META["ENCODER_BLOCK"]),
            triton.cdiv(tgt_max_length_plus_1, META["PREDICTOR_BLOCK"]),
        )

        _rnnt_fused_joint_fwd_triton_kernel[fwd_grid](
            encoder_output_ptr=encoder_output_projected,
            predictor_output_ptr=predictor_output_projected,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=matmul_weight,
            bias_ptr=bias,
            target_logprobs_out_ptr=target_logprobs,
            blank_logprobs_out_ptr=blank_logprobs,
            log_sum_exp_out_ptr=log_sum_exp_scores,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            mixed_dropout_seed=mixed_dropout_seed,
            dropout_threshold=dropout_threshold,
            dropout_inv_keep_prob=dropout_inv_keep_prob,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            USE_DROPOUT=use_dropout,
            USE_HIGH_PRECISION_MM=use_high_precision_mm,
            MM_FULL_PRECISION=mm_full_precision,
        )

        ctx.save_for_backward(
            encoder_output_projected,
            predictor_output_projected,
            weight,
            bias,
            targets,
            src_lengths,
            tgt_lengths,
            log_sum_exp_scores,
            matmul_weight,
        )
        ctx.blank_id = blank_id
        ctx.use_high_precision_mm = use_high_precision_mm
        ctx.mm_full_precision = mm_full_precision
        ctx.use_dropout = use_dropout
        ctx.mixed_dropout_seed = mixed_dropout_seed
        ctx.dropout_threshold = dropout_threshold
        ctx.dropout_p = dropout_p
        ctx.dropout_inv_keep_prob = dropout_inv_keep_prob
        ctx.input_grad_full_precision = input_grad_full_precision
        ctx.weight_grad_full_precision = weight_grad_full_precision
        return target_logprobs, blank_logprobs

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        (
            encoder_output_projected,
            predictor_output_projected,
            weight,
            bias,
            targets,
            src_lengths,
            tgt_lengths,
            log_sum_exp_scores,
            matmul_weight,
        ) = ctx.saved_tensors
        blank_id = ctx.blank_id
        use_high_precision_mm = ctx.use_high_precision_mm
        mm_full_precision = ctx.mm_full_precision
        use_dropout = ctx.use_dropout
        mixed_dropout_seed = ctx.mixed_dropout_seed
        dropout_threshold = ctx.dropout_threshold
        dropout_p = ctx.dropout_p
        input_grad_full_precision = ctx.input_grad_full_precision
        weight_grad_full_precision = ctx.weight_grad_full_precision

        float_dtype = torch.float32

        grad_target_scores = grad_target_scores.contiguous()
        grad_blank_scores = grad_blank_scores.contiguous()

        batch_size, src_max_length, hidden_dim = encoder_output_projected.shape
        tgt_max_length_plus_1 = predictor_output_projected.shape[1]
        vocab_size = weight.shape[0]
        device = encoder_output_projected.device

        if encoder_output_projected.dtype != predictor_output_projected.dtype:
            raise NotImplementedError

        grad_encoder = torch.zeros(
            [batch_size, src_max_length, hidden_dim],
            dtype=float_dtype if input_grad_full_precision else encoder_output_projected.dtype,
            device=device,
        )
        grad_predictor = torch.zeros(
            [batch_size, tgt_max_length_plus_1, hidden_dim],
            dtype=float_dtype if input_grad_full_precision else predictor_output_projected.dtype,
            device=device,
        )

        # grad output variables
        grad_weight = torch.zeros(
            [vocab_size, hidden_dim], dtype=float_dtype if weight_grad_full_precision else weight.dtype, device=device
        )
        grad_bias = torch.zeros(
            [vocab_size], dtype=float_dtype if weight_grad_full_precision else bias.dtype, device=device
        )

        bwd_grid = lambda META: (
            batch_size
            * triton.cdiv(src_max_length, META["ENCODER_BLOCK"])
            * triton.cdiv(tgt_max_length_plus_1, META["PREDICTOR_BLOCK"]),
            triton.cdiv(vocab_size, META["VOCAB_BLOCK"]),
        )

        _rnnt_fused_joint_bwd_triton_kernel[bwd_grid](
            encoder_output_ptr=encoder_output_projected,
            predictor_output_ptr=predictor_output_projected,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=matmul_weight,
            bias_ptr=bias,
            log_sum_exp_ptr=log_sum_exp_scores,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            grad_encoder_out_ptr=grad_encoder,
            grad_predictor_out_ptr=grad_predictor,
            grad_weight_out_ptr=grad_weight,
            grad_bias_out_ptr=grad_bias,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            mixed_dropout_seed=mixed_dropout_seed,
            dropout_threshold=dropout_threshold,
            dropout_p=dropout_p,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            USE_DROPOUT=use_dropout,
            USE_HIGH_PRECISION_MM=use_high_precision_mm,
            MM_FULL_PRECISION=mm_full_precision,
        )

        # convert grad to desired dtype
        grad_encoder = grad_encoder.to(encoder_output_projected.dtype)
        grad_predictor = grad_predictor.to(predictor_output_projected.dtype)
        grad_weight = grad_weight.to(weight.dtype)
        grad_bias = grad_bias.to(bias.dtype)

        return (
            grad_encoder,
            grad_predictor,
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


def rnnt_fused_joint_triton(
    encoder_output_projected: torch.Tensor,
    predictor_output_projected: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    activation: str = "relu",
    dropout_p: float = 0.0,
    dropout_seed: int | None = None,
    use_high_precision_mm: bool = False,
    input_grad_full_precision: bool = True,
    weight_grad_full_precision: bool = True,
    mm_full_precision: bool = False,
):
    target_logprobs, blank_logprobs = _RnntFusedJointTritonFunction.apply(
        encoder_output_projected,
        predictor_output_projected,
        targets,
        tgt_lengths,
        src_lengths,
        weight,
        bias,
        blank_id,
        activation,
        dropout_p,
        dropout_seed,
        use_high_precision_mm,
        input_grad_full_precision,
        weight_grad_full_precision,
        mm_full_precision,
    )
    return target_logprobs, blank_logprobs
