# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""CPU checks that codec Triton kernels do not recompile per codec chunk length.

A kernel argument that is a ``tl.constexpr`` (or a specialized integer) produces a new JIT compilation for every
distinct value. Codec chunk lengths vary at serving time (startup chunks, stream tails, incremental input), so any
argument derived from the chunk length must be a plain, non-specialized runtime argument. Otherwise unseen lengths
compile under load and stall every request in the batch. The checks inspect kernel signatures only and need no CUDA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from triton.runtime.jit import JITFunction

pytest.importorskip("triton")

from easymagpie_vllm_omni.codec import kernels  # noqa: E402

# Kernel name -> arguments that take one value per distinct codec chunk length. ``numel`` arguments are left
# specialized on purpose: integer specialization only splits on ``== 1`` and divisibility by 16 (a fixed, small set
# of variants that warmup covers), and the divisibility hint helps vectorized masking.
_CHUNK_DEPENDENT_ARGS = {
    "_gather_packed_state_inputs_kernel": ("sequence_rows",),
}


def _jit_kernels() -> dict[str, JITFunction]:
    """Collect the private Triton JIT kernels defined in the codec kernel module.

    Returns:
        Kernel functions keyed by module attribute name.
    """
    return {
        name: value
        for name, value in vars(kernels).items()
        if name.startswith("_") and name.endswith("_kernel") and hasattr(value, "params")
    }


def test_known_kernels_are_present() -> None:
    """Verify every kernel listed with chunk-dependent arguments exists as a JIT kernel."""
    assert set(_CHUNK_DEPENDENT_ARGS) <= set(_jit_kernels())


@pytest.mark.parametrize("kernel_name", sorted(_CHUNK_DEPENDENT_ARGS))
def test_chunk_dependent_args_are_not_constexpr(kernel_name: str) -> None:
    """Verify chunk-dependent kernel arguments exist and are not ``tl.constexpr``.

    Args:
        kernel_name: Kernel whose chunk-dependent arguments are checked.
    """
    params = {param.name: param for param in _jit_kernels()[kernel_name].params}
    for arg in _CHUNK_DEPENDENT_ARGS[kernel_name]:
        assert arg in params, f"{kernel_name} has no argument {arg}"
        assert not params[arg].is_constexpr, f"{kernel_name}.{arg} is constexpr and recompiles per chunk length"


@pytest.mark.parametrize("kernel_name", sorted(_CHUNK_DEPENDENT_ARGS))
def test_chunk_dependent_args_are_not_specialized(kernel_name: str) -> None:
    """Verify chunk-dependent kernel arguments are excluded from integer specialization.

    Args:
        kernel_name: Kernel whose chunk-dependent arguments are checked.
    """
    params = {param.name: param for param in _jit_kernels()[kernel_name].params}
    for arg in _CHUNK_DEPENDENT_ARGS[kernel_name]:
        assert params[arg].do_not_specialize, f"{kernel_name}.{arg} is specialized and recompiles per value class"
