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

import contextvars
import inspect
from typing import Any, Callable, ContextManager

import torch
import triton
import triton.language as tl
from packaging.version import Version

# `tl.atomic_add` supports BF16 values starting from Triton 3.4
BF16_ATOMICS_AVAILABLE = Version(triton.__version__) >= Version("3.4.0")
# Tensor descriptors (`tl.make_tensor_descriptor`) are available starting from Triton 3.4.
# NB: the kernels must use this name instead of `tl.make_tensor_descriptor`. Triton resolves all attributes of `tl`
# in the source of a kernel when the kernel is created, also in the branches that are not compiled:
# a kernel that refers to an attribute missing in the installed Triton can't be used
make_tensor_descriptor = getattr(tl, "make_tensor_descriptor", None)
TENSOR_DESCRIPTORS_AVAILABLE = make_tensor_descriptor is not None
# Required alignment of the base pointer and of the strides of a tensor descriptor (TMA)
_TENSOR_DESCRIPTOR_ALIGNMENT_BYTES = 16


def cuda_device_guard(tensor: torch.Tensor) -> ContextManager:
    """
    Context manager that makes the CUDA device of the tensor current.

    Triton compiles and launches kernels for the current CUDA device (`torch.cuda.current_device()`),
    which can differ from the device of the tensors (e.g., for a model moved to `cuda:1` manually).
    All kernel launches should be wrapped with this guard.
    """
    return torch.cuda.device(tensor.device)


def can_use_tensor_descriptors(*matrices: torch.Tensor) -> bool:
    """
    Whether blocks of the contiguous row-major matrices `[..., num_columns]` can be loaded / stored by the kernels
    with tensor descriptors instead of pointers with explicit masks (`TENSOR_DESCRIPTORS_ALLOWED` argument of
    the kernels: enables the autotune configs with `TENSOR_DESCRIPTORS`, see `make_tile_configs`).

    Tensor descriptors are allowed only on the GPUs with TMA (compute capability 9.0+: Hopper, Blackwell): a block
    is copied between the global and the shared memory by a single instruction instead of an instruction per
    element. For older GPUs Triton rewrites the descriptors to pointers; the kernels use the explicit loads there
    (also the only option for Triton < 3.4). TMA requires the base pointers and the size of a row to be multiples
    of 16 bytes. No host-device synchronization.
    """
    if not TENSOR_DESCRIPTORS_AVAILABLE:
        return False
    device = matrices[0].device
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 9:
        return False
    return all(
        matrix.data_ptr() % _TENSOR_DESCRIPTOR_ALIGNMENT_BYTES == 0
        and matrix.shape[-1] * matrix.element_size() % _TENSOR_DESCRIPTOR_ALIGNMENT_BYTES == 0
        for matrix in matrices
    )


def launch_kernel_on_device(kernel: Callable, grid: Any, device_tensor: torch.Tensor, **kernel_arguments):
    """
    Launch a Triton kernel `kernel[grid](**kernel_arguments)` for the CUDA device of `device_tensor`.

    A kernel with tensor descriptors (`TENSOR_DESCRIPTORS_ALLOWED`) requires a global scratch memory for
    the descriptors, allocated by Triton at launch with the allocator from a context variable
    (`triton.set_allocator`, not set by default). The allocator is set only for the launch, in a copy of
    the current context: the allocator of the caller (if any) is not replaced, and the launch does not depend
    on the thread that imported this module.
    """
    with cuda_device_guard(device_tensor):
        if kernel_arguments.get("TENSOR_DESCRIPTORS_ALLOWED", False):
            contextvars.copy_context().run(_launch_kernel_with_scratch_allocator, kernel, grid, kernel_arguments)
        else:
            kernel[grid](**kernel_arguments)


def make_tile_configs(
    block_names: list[str],
    tiles: list[tuple[int, ...]],
    tensor_descriptor_tiles: list[tuple[int, ...]] | None = None,
    **meta_parameters,
) -> list:
    """
    Autotune configs (`triton.Config`) from the tuples `(*block sizes, num_warps, num_stages)`.

    Args:
        block_names: names of the meta-parameters of the kernel for the block sizes
        tiles: configs for the loads / stores with pointers and explicit masks (any GPU)
        tensor_descriptor_tiles: configs for the loads / stores with tensor descriptors, for the kernels with
            `TENSOR_DESCRIPTORS` meta-parameter (None for the other kernels). The configs are benchmarked together
            with `tiles` only if the descriptors are allowed for the call (`TENSOR_DESCRIPTORS_ALLOWED`,
            see `can_use_tensor_descriptors`); the fastest config is selected for the GPU.
        **meta_parameters: other meta-parameters of all configs
    """
    variants = [(tiles, False)]
    if tensor_descriptor_tiles and TENSOR_DESCRIPTORS_AVAILABLE:
        variants.append((tensor_descriptor_tiles, True))
    configs = []
    for variant_tiles, use_tensor_descriptors in variants:
        for *block_sizes, num_warps, num_stages in variant_tiles:
            config_meta_parameters = dict(zip(block_names, block_sizes, strict=True), **meta_parameters)
            if tensor_descriptor_tiles is not None:
                config_meta_parameters["TENSOR_DESCRIPTORS"] = use_tensor_descriptors
            configs.append(triton.Config(config_meta_parameters, num_warps=num_warps, num_stages=num_stages))
    return configs


def autotune(configs: list, key: list[str], reset_to_zero: list[str] | None = None):
    """
    `triton.autotune` with the results of autotuning cached on disk (`cache_results`).
    Triton < 3.4 does not support the cache: the kernels are autotuned on the first use in each process.
    The configs with tensor descriptors are skipped unless the descriptors are allowed for the call.
    """
    kwargs = {"cache_results": True} if "cache_results" in inspect.signature(triton.autotune).parameters else {}
    return triton.autotune(
        configs=configs,
        key=key,
        reset_to_zero=reset_to_zero,
        prune_configs_by={"early_config_prune": _prune_tensor_descriptor_configs},
        **kwargs,
    )


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
def matmul(a: tl.tensor, b: tl.tensor, USE_HIGH_PRECISION: tl.constexpr = False):
    """Matrix product `a @ b` with FP32 accumulation; `USE_HIGH_PRECISION` requests IEEE precision for FP32 operands"""
    if USE_HIGH_PRECISION:
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


def _prune_tensor_descriptor_configs(configs: list, named_arguments: dict, **kernel_arguments) -> list:
    """Configs for autotuning of a call: without the configs with tensor descriptors if they are not allowed"""
    if kernel_arguments.get("TENSOR_DESCRIPTORS_ALLOWED", named_arguments.get("TENSOR_DESCRIPTORS_ALLOWED", False)):
        return configs
    return [config for config in configs if not config.kwargs.get("TENSOR_DESCRIPTORS", False)]


def _launch_kernel_with_scratch_allocator(kernel: Callable, grid: Any, kernel_arguments: dict):
    triton.set_allocator(_allocate_global_scratch)
    kernel[grid](**kernel_arguments)


def _allocate_global_scratch(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    """Global scratch memory of a kernel on the current CUDA device (allocations of PyTorch are 512-byte aligned)"""
    return torch.empty(size, dtype=torch.int8, device="cuda")
