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

import triton
import triton.language as tl


DROPOUT_HASH_MAX = tl.constexpr(0x7FFFFFFF)
DROPOUT_SEED_MIX_MULTIPLIER = 2654435761


def mix_dropout_seed(dropout_seed: int) -> int:
    """Pre-mix a dropout seed into a well-spread signed int32."""
    mixed_seed = (int(dropout_seed) * DROPOUT_SEED_MIX_MULTIPLIER) & 0xFFFFFFFF
    if mixed_seed >= 0x80000000:
        mixed_seed -= 0x100000000
    return mixed_seed


@triton.jit
def log_add_exp(log_probs_1, log_probs_2):
    max_score = tl.maximum(log_probs_1, log_probs_2)
    return max_score + tl.log(tl.exp(log_probs_1 - max_score) + tl.exp(log_probs_2 - max_score))


@triton.jit
def matmul(a: tl.tensor, b: tl.tensor, USE_FP64: tl.constexpr = False, USE_HIGH_PRECISION: tl.constexpr = False):
    if USE_FP64:
        result = tl.sum(a.T[:, :, None] * b[:, None, :], axis=0)
    elif USE_HIGH_PRECISION:
        result = tl.dot(a, b, input_precision="ieee")
    else:
        result = tl.dot(a, b)
    return result


@triton.jit
def dropout_hash_keep_mask(mixed_dropout_seed, dropout_offsets, dropout_threshold):
    """
    Deterministic dropout mask based on a lightweight per-offset integer hash.

    The offset hash follows the cuTile dropout implementation in TileGym.
    https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/ops/cutile/dropout.py
    The seed is pre-mixed on the Python side before launch.
    """
    combined = dropout_offsets.to(tl.int32) * 1103515245 + mixed_dropout_seed
    hash_value = combined ^ (combined >> 16)
    hash_value = hash_value ^ (hash_value << 8)
    hash_value = hash_value ^ (hash_value >> 4)
    return (hash_value & DROPOUT_HASH_MAX) >= dropout_threshold


@triton.jit
def sum_at_range(x: tl.tensor, y: tl.tensor, start, axis: tl.constexpr):
    """
    Return x[..., 0:start] + (x[..., start:start+y.shape[axis]] + y) + x[..., start+y.shape[axis]:]
    """
    x_offsets = tl.arange(0, x.shape[axis])
    mask = (x_offsets >= start) & (x_offsets < start + y.shape[axis])
    y_indices_safe_to_x = tl.where(mask, x_offsets - start, 0)
    broadcastable_shape: tl.constexpr = [1] * axis + [x.shape[axis]] + ([1] * (len(x.shape) - axis - 1))
    y_to_x_expanded = y.gather(y_indices_safe_to_x.reshape(broadcastable_shape).broadcast_to(x.shape), axis=axis)
    y_to_x_expanded = tl.where(mask.reshape(broadcastable_shape), y_to_x_expanded, 0.0)
    return x + y_to_x_expanded


@triton.jit
def sum_at_block(x: tl.tensor, y: tl.tensor, block_id, axis: tl.constexpr):
    """
    Return x[..., 0:block_id*y.shape[axis]] + (x[..., block_id*y.shape[axis]:(block_id+1)*y.shape[axis] + y) + x[..., (block_id+1)*y.shape[axis]:]
    """
    tl.static_assert(x.shape[axis] % y.shape[axis] == 0)
    num_blocks: tl.constexpr = x.shape[axis] // y.shape[axis]
    mask = tl.arange(0, num_blocks) == block_id
    mask_broadcastable_shape: tl.constexpr = [1] * axis + [num_blocks] + ([1] * (len(x.shape) - axis))
    x_shape_by_blocks: tl.constexpr = x.shape[:axis] + [num_blocks, y.shape[axis]] + x.shape[axis + 1 :]
    return (x.reshape(x_shape_by_blocks) + y.expand_dims(axis) * mask.reshape(mask_broadcastable_shape)).reshape(
        x.shape
    )


@triton.jit
def dropout_scale_mask_kernel(
    mask_out_ptr,
    total_elements: int,
    mixed_dropout_seed: int,
    dropout_threshold: int,
    dropout_inv_keep_prob: float,
    BLOCK: tl.constexpr,
):
    flat_offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    valid_mask = flat_offsets < total_elements
    keep_mask = dropout_hash_keep_mask(mixed_dropout_seed, flat_offsets, dropout_threshold)
    dropout_scale_mask = tl.where(keep_mask, dropout_inv_keep_prob, 0.0)
    tl.store(mask_out_ptr + flat_offsets, dropout_scale_mask, mask=valid_mask)
