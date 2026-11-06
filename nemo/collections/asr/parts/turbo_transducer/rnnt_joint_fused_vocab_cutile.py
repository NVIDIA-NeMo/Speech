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

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

# Expected active CTAs per SM. cuTile exposes no num_warps/num_stages knob, so this is the main pressure dial.
# Tuned on sm_120: 2 is ~8% faster than the compiler default, 1 is ~33% slower and 4 is catastrophic (5x).
# See tune_rnnt_joint_fused_vocab_cutile.py.
_OCCUPANCY = 2


def _matmul_dtype(array_dtype, use_high_precision_mm: bool):
    """Mirror the Triton kernel's ``matmul_dtype`` rule: keep bfloat16 operands unless high precision is requested."""
    if use_high_precision_mm:
        return ct.float32
    return array_dtype


@ct.kernel(occupancy=_OCCUPANCY)
def _rnnt_joint_fused_vocab_fwd_cutile_kernel(
    joint_hidden_2d,
    targets_flat,
    src_lengths,
    tgt_lengths,
    weight,
    bias,
    target_logprobs_out,
    blank_logprobs_out,
    log_sum_exp_out,
    flattened_batch_size,
    max_src_len,
    max_tgt_len_plus_1,
    max_target_len,
    hidden_dim: ConstInt,
    vocab_size: ConstInt,
    blank_id: ConstInt,
    FLATTENED_BATCH_BLOCK: ConstInt,
    HIDDEN_BLOCK: ConstInt,
    VOCAB_BLOCK: ConstInt,
    USE_HIGH_PRECISION_MM: ConstBool,
):
    matmul_dtype = _matmul_dtype(joint_hidden_2d.dtype, USE_HIGH_PRECISION_MM)

    flattened_batch_block_index = ct.bid(0)
    flattened_batch_start = flattened_batch_block_index * FLATTENED_BATCH_BLOCK
    flattened_batch_offsets = flattened_batch_start + ct.arange(FLATTENED_BATCH_BLOCK, dtype=ct.int32)
    flattened_batch_valid_mask = flattened_batch_offsets < flattened_batch_size

    source_target_block_size = max_src_len * max_tgt_len_plus_1
    batch_index = flattened_batch_offsets // source_target_block_size
    batch_offsets_in_st = flattened_batch_offsets - batch_index * source_target_block_size
    source_index = batch_offsets_in_st // max_tgt_len_plus_1
    target_index = batch_offsets_in_st - source_index * max_tgt_len_plus_1

    source_length = ct.gather(src_lengths, batch_index, mask=flattened_batch_valid_mask, padding_value=0)
    target_length = ct.gather(tgt_lengths, batch_index, mask=flattened_batch_valid_mask, padding_value=0)

    source_mask = source_index < source_length
    target_valid_mask = target_index <= target_length
    target_label_mask = target_index < target_length
    output_blank_mask = flattened_batch_valid_mask & source_mask & target_valid_mask
    output_target_mask = flattened_batch_valid_mask & source_mask & target_label_mask

    target_flat_idx = batch_index * max_target_len + target_index
    targets_row = ct.gather(targets_flat, target_flat_idx, mask=output_target_mask, padding_value=0)

    log_sum_exp_score = ct.full((FLATTENED_BATCH_BLOCK,), float("-inf"), dtype=ct.float32)
    blank_logits = ct.zeros((FLATTENED_BATCH_BLOCK,), dtype=ct.float32)
    target_logits = ct.zeros((FLATTENED_BATCH_BLOCK,), dtype=ct.float32)

    num_vocab_blocks = ct.cdiv(vocab_size, VOCAB_BLOCK)
    num_hidden_blocks = ct.cdiv(hidden_dim, HIDDEN_BLOCK)
    vocab_in_block_offsets = ct.arange(VOCAB_BLOCK, dtype=ct.int32)

    weight_tv = weight.tiled_view((VOCAB_BLOCK, HIDDEN_BLOCK), padding_mode=ct.PaddingMode.ZERO)
    bias_tv = bias.tiled_view((VOCAB_BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    joint_hidden_2d_tv = joint_hidden_2d.tiled_view(
        (FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK), padding_mode=ct.PaddingMode.ZERO
    )

    for vocab_block_index in range(num_vocab_blocks):
        vocab_offsets = vocab_block_index * VOCAB_BLOCK + vocab_in_block_offsets
        vocab_mask = vocab_offsets < vocab_size

        bias_chunk = bias_tv.load((vocab_block_index,))
        bias_chunk_f32 = ct.astype(bias_chunk, ct.float32)
        block_logits = ct.full((FLATTENED_BATCH_BLOCK, VOCAB_BLOCK), 0.0, dtype=ct.float32) + bias_chunk_f32[None, :]

        for hidden_block_index in range(num_hidden_blocks):
            hidden_tile = joint_hidden_2d_tv.load((flattened_batch_block_index, hidden_block_index))
            weight_tile = weight_tv.load((vocab_block_index, hidden_block_index))
            block_logits = ct.mma(
                ct.astype(hidden_tile, matmul_dtype),
                ct.transpose(ct.astype(weight_tile, matmul_dtype)),
                block_logits,
            )

        block_logits = ct.where(vocab_mask[None, :], block_logits, float("-inf"))

        block_logits_max = ct.max(block_logits, axis=-1)
        block_lse = ct.log(ct.sum(ct.exp(block_logits - block_logits_max[:, None]), axis=-1)) + block_logits_max

        max_score = ct.maximum(log_sum_exp_score, block_lse)
        log_sum_exp_score = max_score + ct.log(ct.exp(log_sum_exp_score - max_score) + ct.exp(block_lse - max_score))

        blank_logits += ct.sum(ct.where((vocab_offsets == blank_id)[None, :], block_logits, 0.0), axis=-1)
        target_logits += ct.sum(ct.where(vocab_offsets[None, :] == targets_row[:, None], block_logits, 0.0), axis=-1)

    blank_logprobs_value = blank_logits - log_sum_exp_score
    target_logprobs_value = target_logits - log_sum_exp_score

    ct.store(blank_logprobs_out, index=flattened_batch_block_index, tile=blank_logprobs_value * output_blank_mask)
    ct.store(target_logprobs_out, index=flattened_batch_block_index, tile=target_logprobs_value * output_target_mask)
    ct.store(log_sum_exp_out, index=flattened_batch_block_index, tile=log_sum_exp_score * output_blank_mask)


@ct.kernel(occupancy=_OCCUPANCY)
def _rnnt_joint_fused_vocab_bwd_cutile_kernel(
    joint_hidden_2d,
    targets_flat,
    src_lengths,
    tgt_lengths,
    weight,
    bias,
    log_sum_exp,
    grad_target_scores,
    grad_blank_scores,
    grad_joint_hidden_2d,
    grad_weight,
    grad_bias,
    flattened_batch_size,
    max_src_len,
    max_tgt_len_plus_1,
    max_target_len,
    hidden_dim: ConstInt,
    vocab_size: ConstInt,
    blank_id: ConstInt,
    FLATTENED_BATCH_BLOCK: ConstInt,
    HIDDEN_BLOCK: ConstInt,
    VOCAB_BLOCK: ConstInt,
    USE_HIGH_PRECISION_MM: ConstBool,
):
    matmul_dtype = _matmul_dtype(joint_hidden_2d.dtype, USE_HIGH_PRECISION_MM)

    # 1-D grid, decomposed so that all vocab blocks of one flattened-batch tile run on consecutive CTAs. That keeps
    # the tile's grad_joint_hidden rows resident in L2 across the whole vocab sweep (see the Triton twin, which
    # remaps its 2-D grid to the same order).
    num_vocab_blocks = ct.cdiv(vocab_size, VOCAB_BLOCK)
    linear_block_index = ct.bid(0)
    flattened_batch_block_index = linear_block_index // num_vocab_blocks
    vocab_block_index = linear_block_index - flattened_batch_block_index * num_vocab_blocks

    vocab_offsets = vocab_block_index * VOCAB_BLOCK + ct.arange(VOCAB_BLOCK, dtype=ct.int32)
    vocab_mask = vocab_offsets < vocab_size
    is_blank_vocab_col = (vocab_offsets == blank_id) & vocab_mask

    num_hidden_blocks = ct.cdiv(hidden_dim, HIDDEN_BLOCK)
    source_target_block_size = max_src_len * max_tgt_len_plus_1

    weight_tv = weight.tiled_view((VOCAB_BLOCK, HIDDEN_BLOCK), padding_mode=ct.PaddingMode.ZERO)
    bias_tv = bias.tiled_view((VOCAB_BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    joint_hidden_2d_tv = joint_hidden_2d.tiled_view(
        (FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK), padding_mode=ct.PaddingMode.ZERO
    )
    grad_joint_hidden_2d_tv = grad_joint_hidden_2d.tiled_view((FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK))
    grad_weight_tv = grad_weight.tiled_view((VOCAB_BLOCK, HIDDEN_BLOCK))
    grad_bias_tv = grad_bias.tiled_view((VOCAB_BLOCK,))

    bias_chunk = ct.astype(bias_tv.load((vocab_block_index,)), ct.float32)

    flattened_batch_start = flattened_batch_block_index * FLATTENED_BATCH_BLOCK
    flattened_batch_offsets = flattened_batch_start + ct.arange(FLATTENED_BATCH_BLOCK, dtype=ct.int32)
    flattened_batch_valid_mask = flattened_batch_offsets < flattened_batch_size

    batch_index = flattened_batch_offsets // source_target_block_size
    batch_offsets_in_st = flattened_batch_offsets - batch_index * source_target_block_size
    source_index = batch_offsets_in_st // max_tgt_len_plus_1
    target_index = batch_offsets_in_st - source_index * max_tgt_len_plus_1

    source_length = ct.gather(src_lengths, batch_index, mask=flattened_batch_valid_mask, padding_value=0)
    target_length = ct.gather(tgt_lengths, batch_index, mask=flattened_batch_valid_mask, padding_value=0)

    source_mask = source_index < source_length
    target_valid_mask = target_index <= target_length
    target_label_mask = target_index < target_length
    output_blank_mask = flattened_batch_valid_mask & source_mask & target_valid_mask
    output_target_mask = flattened_batch_valid_mask & source_mask & target_label_mask

    target_flat_idx = batch_index * max_target_len + target_index
    targets_row = ct.gather(targets_flat, target_flat_idx, mask=output_target_mask, padding_value=0)

    lse = ct.astype(
        ct.gather(log_sum_exp, flattened_batch_offsets, mask=output_blank_mask, padding_value=0.0),
        ct.float32,
    )
    grad_target = ct.astype(
        ct.gather(grad_target_scores, flattened_batch_offsets, mask=output_target_mask, padding_value=0.0),
        ct.float32,
    )
    grad_blank = ct.astype(
        ct.gather(grad_blank_scores, flattened_batch_offsets, mask=output_blank_mask, padding_value=0.0),
        ct.float32,
    )
    sum_grad = grad_target + grad_blank

    # Recompute logits (same matmul as forward).
    logits_block = ct.full((FLATTENED_BATCH_BLOCK, VOCAB_BLOCK), 0.0, dtype=ct.float32) + bias_chunk[None, :]
    for hidden_block_index in range(num_hidden_blocks):
        hidden_tile = joint_hidden_2d_tv.load((flattened_batch_block_index, hidden_block_index))
        weight_tile = weight_tv.load((vocab_block_index, hidden_block_index))
        logits_block = ct.mma(
            ct.astype(hidden_tile, matmul_dtype),
            ct.transpose(ct.astype(weight_tile, matmul_dtype)),
            logits_block,
        )

    # Softmax gradient.
    probabilities_block = ct.exp(logits_block - lse[:, None])
    grad_logits_block = (
        -(sum_grad[:, None] * probabilities_block)
        + (grad_blank[:, None] * is_blank_vocab_col[None, :])
        + (grad_target[:, None] * (vocab_offsets[None, :] == targets_row[:, None]))
    )
    grad_logits_block = ct.where(output_blank_mask[:, None] & vocab_mask[None, :], grad_logits_block, 0.0)
    grad_bias_acc = ct.sum(grad_logits_block, axis=0)

    grad_logits_for_mm = ct.astype(grad_logits_block, matmul_dtype)

    # Reverse iteration over the hidden dim so the tail of the logits loop is still cache-hot.
    for reverse_hidden_index in range(num_hidden_blocks):
        hidden_block_index = num_hidden_blocks - reverse_hidden_index - 1
        weight_tile = ct.astype(weight_tv.load((vocab_block_index, hidden_block_index)), matmul_dtype)
        grad_hidden_delta = ct.mma(
            grad_logits_for_mm,
            weight_tile,
            ct.zeros((FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK), dtype=ct.float32),
        )
        # Atomic: every vocab block contributes to the same joint_hidden rows.
        grad_joint_hidden_2d_tv.atomic_store_add(
            (flattened_batch_block_index, hidden_block_index),
            ct.astype(grad_hidden_delta, grad_joint_hidden_2d.dtype),
        )

    grad_logits_for_mm_t = ct.transpose(grad_logits_for_mm)
    for reverse_hidden_index in range(num_hidden_blocks):
        hidden_block_index = num_hidden_blocks - reverse_hidden_index - 1
        hidden_tile = ct.astype(
            joint_hidden_2d_tv.load((flattened_batch_block_index, hidden_block_index)), matmul_dtype
        )
        grad_weight_delta = ct.mma(
            grad_logits_for_mm_t,
            hidden_tile,
            ct.zeros((VOCAB_BLOCK, HIDDEN_BLOCK), dtype=ct.float32),
        )
        # Atomic: every flattened-batch block contributes to the same weight rows.
        grad_weight_tv.atomic_store_add(
            (vocab_block_index, hidden_block_index), ct.astype(grad_weight_delta, grad_weight.dtype)
        )

    grad_bias_tv.atomic_store_add((vocab_block_index,), ct.astype(grad_bias_acc, grad_bias.dtype))


# (FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK, VOCAB_BLOCK), tuned on sm_120 for vocab_size in {1025, 8193};
# both sizes pick the same shape. FLATTENED_BATCH_BLOCK also sets how many atomic accumulations land in each
# grad_weight element, so keep it at 128 to match the Triton backend's gradient accuracy, not just its speed.
_FORWARD_CONFIG = (128, 32, 128)
_BACKWARD_CONFIG = (128, 32, 128)


class _RnntJointFusedVocabCuTileFunction(torch.autograd.Function):
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
        mm_full_precision: bool = False,
    ):
        batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
        flattened_batch_size = batch_size * src_max_length * tgt_max_length_plus_1
        max_target_len = tgt_max_length_plus_1 - 1
        vocab_size = weight.shape[0]
        device = joint_hidden.device
        output_dtype = torch.float64 if joint_hidden.dtype == torch.float64 else torch.float32

        joint_hidden = joint_hidden.contiguous()
        targets = targets.contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        matmul_weight = (
            weight.to(torch.bfloat16)
            if joint_hidden.dtype == torch.bfloat16 and not mm_full_precision and not use_high_precision_mm
            else weight
        )

        joint_hidden_2d = joint_hidden.view(flattened_batch_size, hidden_dim)
        targets_flat = targets.view(-1)

        target_logprobs = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1], dtype=output_dtype, device=device
        )
        blank_logprobs = torch.zeros_like(target_logprobs)
        log_sum_exp_scores = torch.empty_like(target_logprobs)

        flattened_batch_block, hidden_block, vocab_block = _FORWARD_CONFIG
        flattened_batch_blocks = (flattened_batch_size + flattened_batch_block - 1) // flattened_batch_block

        ct.launch(
            torch.cuda.current_stream(),
            (flattened_batch_blocks, 1, 1),
            _rnnt_joint_fused_vocab_fwd_cutile_kernel,
            (
                joint_hidden_2d,
                targets_flat,
                src_lengths,
                tgt_lengths,
                matmul_weight,
                bias,
                target_logprobs.view(-1),
                blank_logprobs.view(-1),
                log_sum_exp_scores.view(-1),
                flattened_batch_size,
                src_max_length,
                tgt_max_length_plus_1,
                max_target_len,
                hidden_dim,
                vocab_size,
                blank_id,
                flattened_batch_block,
                hidden_block,
                vocab_block,
                use_high_precision_mm or mm_full_precision,
            ),
        )

        ctx.save_for_backward(
            joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, log_sum_exp_scores, matmul_weight
        )
        ctx.blank_id = blank_id
        ctx.src_max_length = src_max_length
        ctx.tgt_max_length_plus_1 = tgt_max_length_plus_1
        ctx.use_high_precision_mm = use_high_precision_mm
        ctx.mm_full_precision = mm_full_precision
        ctx.input_grad_full_precision = input_grad_full_precision
        ctx.weight_grad_full_precision = weight_grad_full_precision
        return target_logprobs, blank_logprobs

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        (
            joint_hidden,
            weight,
            bias,
            targets,
            src_lengths,
            tgt_lengths,
            log_sum_exp_scores,
            matmul_weight,
        ) = ctx.saved_tensors
        blank_id = ctx.blank_id
        src_max_length = ctx.src_max_length
        tgt_max_length_plus_1 = ctx.tgt_max_length_plus_1
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

        batch_size, _, _, hidden_dim = joint_hidden.shape
        flattened_batch_size = batch_size * src_max_length * tgt_max_length_plus_1
        max_target_len = tgt_max_length_plus_1 - 1
        vocab_size = weight.shape[0]
        device = joint_hidden.device
        float_dtype = torch.float64 if joint_hidden.dtype == torch.float64 else torch.float32

        grad_joint_hidden = torch.zeros(
            [flattened_batch_size, hidden_dim],
            dtype=float_dtype if input_grad_full_precision else joint_hidden.dtype,
            device=device,
        )
        grad_weight = torch.zeros(
            [vocab_size, hidden_dim],
            dtype=float_dtype if weight_grad_full_precision else weight.dtype,
            device=device,
        )
        grad_bias = torch.zeros(
            [vocab_size], dtype=float_dtype if weight_grad_full_precision else bias.dtype, device=device
        )

        flattened_batch_block, hidden_block, vocab_block = _BACKWARD_CONFIG
        batch_blocks = (flattened_batch_size + flattened_batch_block - 1) // flattened_batch_block
        vocab_blocks = (vocab_size + vocab_block - 1) // vocab_block

        ct.launch(
            torch.cuda.current_stream(),
            (batch_blocks * vocab_blocks, 1, 1),
            _rnnt_joint_fused_vocab_bwd_cutile_kernel,
            (
                joint_hidden.view(flattened_batch_size, hidden_dim),
                targets.view(-1),
                src_lengths,
                tgt_lengths,
                matmul_weight,
                bias,
                log_sum_exp_scores.view(-1),
                grad_target_scores.view(-1),
                grad_blank_scores.view(-1),
                grad_joint_hidden,
                grad_weight,
                grad_bias,
                flattened_batch_size,
                src_max_length,
                tgt_max_length_plus_1,
                max_target_len,
                hidden_dim,
                vocab_size,
                blank_id,
                flattened_batch_block,
                hidden_block,
                vocab_block,
                use_high_precision_mm or mm_full_precision,
            ),
        )

        grad_joint_hidden = grad_joint_hidden.view(batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim)
        grad_joint_hidden = grad_joint_hidden.to(joint_hidden.dtype)
        grad_weight = grad_weight.to(weight.dtype)
        grad_bias = grad_bias.to(bias.dtype)

        return grad_joint_hidden, None, None, None, grad_weight, grad_bias, None, None, None, None, None


def rnnt_joint_fused_vocab_cutile(
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
    mm_full_precision: bool = False,
):
    target_logprobs, blank_logprobs = _RnntJointFusedVocabCuTileFunction.apply(
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
        mm_full_precision,
    )
    return target_logprobs, blank_logprobs
