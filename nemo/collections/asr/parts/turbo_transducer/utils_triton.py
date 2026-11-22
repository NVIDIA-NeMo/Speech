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

import inspect
from typing import ContextManager

import torch
import triton
import triton.language as tl
from packaging.version import Version

# `tl.atomic_add` supports BF16 values starting from Triton 3.4
BF16_ATOMICS_AVAILABLE = Version(triton.__version__) >= Version("3.4.0")


def cuda_device_guard(tensor: torch.Tensor) -> ContextManager:
    """
    Context manager that makes the CUDA device of the tensor current.

    Triton compiles and launches kernels for the current CUDA device (`torch.cuda.current_device()`),
    which can differ from the device of the tensors (e.g., for a model moved to `cuda:1` manually).
    All kernel launches should be wrapped with this guard.
    """
    return torch.cuda.device(tensor.device)


def autotune(configs: list, key: list[str], reset_to_zero: list[str] | None = None):
    """
    `triton.autotune` with the results of autotuning cached on disk (`cache_results`).
    Triton < 3.4 does not support the cache: the kernels are autotuned on the first use in each process.
    """
    kwargs = {"cache_results": True} if "cache_results" in inspect.signature(triton.autotune).parameters else {}
    return triton.autotune(configs=configs, key=key, reset_to_zero=reset_to_zero, **kwargs)


@triton.jit
def log_add_exp(log_probs_1, log_probs_2):
    """
    Numerically stable log(exp(log_probs_1) + exp(log_probs_2)).
    If both arguments are `-inf` (zero probability), the result is `-inf`. NaN arguments are propagated.
    """
    max_score = tl.maximum(log_probs_1, log_probs_2)
    log_sum = max_score + tl.log(tl.exp(log_probs_1 - max_score) + tl.exp(log_probs_2 - max_score))
    # NB: `-inf - (-inf)` is NaN, the case of two `-inf` arguments is handled separately.
    # Both arguments are checked explicitly (not the maximum): `tl.maximum` ignores NaN, but NaN must be propagated
    both_neg_inf = (log_probs_1 == -float("inf")) & (log_probs_2 == -float("inf"))
    return tl.where(both_neg_inf, -float("inf"), log_sum)


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
def block_offsets_int64(block_start, BLOCK_SIZE: tl.constexpr):
    """
    Offsets [block_start, block_start + BLOCK_SIZE) in int64, e.g., rows of a matrix for `load_rows_block`.

    NB: the offsets are computed in int64 from the scalar start. Conversion of int32 offsets of the block to int64
    makes the kernels 2-5% slower: the differences between the addresses of the rows are not compile-time constants
    anymore (sign extension can't be moved inside the int32 addition), and each address needs its own register.
    """
    return block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)


@triton.jit
def load_rows_block(
    matrix_ptr,
    row_indices,
    row_mask,
    column_start,
    num_columns: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
    TRANSPOSED: tl.constexpr = False,
    CACHE_VOLATILE: tl.constexpr = False,
):
    """
    Load a block [number of rows, COLUMN_BLOCK] of a contiguous row-major matrix with `num_columns` columns:
    columns [column_start, column_start + COLUMN_BLOCK) of the rows `row_indices`.
    The values are zeros for the rows excluded by `row_mask` and for the columns outside the matrix.

    Args:
        row_indices: int64 indices of the rows: `block_offsets_int64` for contiguous rows, or arbitrary indices
        column_start: first column, must be a multiple of `COLUMN_BLOCK`
        TRANSPOSED: return the transposed block [COLUMN_BLOCK, number of rows], loaded directly in the transposed
            layout. For an operand of `tl.dot`, it is not equivalent in speed to the transposition (`.T`) of
            the loaded block: the difference depends on the kernel and the tile sizes (measured for the forward
            kernel of the Joint: from 1% faster to 30% slower): benchmark the kernel when changing the layout
        CACHE_VOLATILE: load with the `.cv` cache modifier

    NB: plain pointers with explicit masks are used in all kernels instead of block pointers (`tl.make_block_ptr`,
    `tl.advance`): block pointers are removed in Triton 3.9.
    """
    column_offsets = block_offsets_int64(column_start, COLUMN_BLOCK)
    # int64: number of elements in the matrix can exceed the int32 range
    row_offsets = row_indices.to(tl.int64) * num_columns
    # mask of the columns is not needed if all blocks are inside the matrix; in both cases the mask is constant
    # along the contiguous dimension (columns), which keeps the loads vectorized
    if TRANSPOSED:
        offsets = column_offsets[:, None] + row_offsets[None, :]
        if num_columns % COLUMN_BLOCK == 0:
            mask = row_mask[None, :]
        else:
            mask = (column_offsets < num_columns)[:, None] & row_mask[None, :]
    else:
        offsets = row_offsets[:, None] + column_offsets[None, :]
        if num_columns % COLUMN_BLOCK == 0:
            mask = row_mask[:, None]
        else:
            mask = row_mask[:, None] & (column_offsets < num_columns)[None, :]
    # NB: the cache modifier is not an argument: string arguments of nested functions fail to compile in Triton < 3.7
    if CACHE_VOLATILE:
        block = tl.load(matrix_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cv")
    else:
        block = tl.load(matrix_ptr + offsets, mask=mask, other=0.0)
    return block
