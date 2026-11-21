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
Tests for the production kernel configs of the `fused_vocab` Joint: Triton autotuning and each autotune config,
the fixed config of the cuTile kernels - with production dimensions (hidden 640, vocab 1025) and precision
(BF16 joint hidden, FP32 master weights, default kernel arguments).

All the other tests compile the kernels with a single small-tile config (see `conftest.py`), which is fast, but
covers neither the autotuning itself (benchmarking of the kernels that accumulate gradients with atomics, selection
of the config) nor the large tiles of the production configs.

SLOW TESTS, NOT FOR CI: the tests compile ~70 kernels with large tiles and benchmark them, which takes several
minutes (the Triton cache is not used, so that the autotuning is always executed). The tests are marked `nightly`
and skipped unless the `--nightly` option is passed. Run them locally after changing the kernels or their configs:

    pytest --nightly tests/collections/asr/turbo_transducer/test_rnnt_joint_fused_vocab_production_configs.py
"""

import pytest
import torch

from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab import (
    rnnt_joint_fused_vocab,
    rnnt_joint_fused_vocab_with_consistency,
)
from nemo.core.utils.optional_libs import CUTILE_AVAILABLE, TRITON_AVAILABLE
from tests.collections.asr.turbo_transducer.utils import (
    assert_close_to_reference,
    assert_pruned_outputs_close_to_reference,
    gather_pruning_ranges,
    make_joint_inputs,
    random_grads_like,
    random_pruning_range_starts,
    reference_outputs,
    reference_pruned_lattice_outputs,
    reference_pruning_windows,
)

if TRITON_AVAILABLE:
    from triton.runtime.errors import OutOfResources

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]
requires_triton = pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
requires_cutile = pytest.mark.skipif(not CUTILE_AVAILABLE, reason="cuTile is not installed")

# (hidden_dim, vocab_size, blank_id)
PRODUCTION_DIMS = (640, 1025, 1024)
# Lattice [4, 40, 21]: 3360 rows, 14-53 row tiles of the production configs
SRC_LENGTHS = (40, 1, 25, 33)
TGT_LENGTHS = (20, 0, 7, 20)
PRUNING_RANGE_SIZE = 8

OUTPUT_NAMES = ["target_logprobs", "blank_logprobs", "consistency_kl"]
GRAD_NAMES = ["joint_hidden grad", "weight grad", "bias grad"]
# Tolerances for BF16 joint hidden, FP32 master weights and default kernel arguments:
# [outputs, joint hidden grad, weight grad, bias grad], see `MIXED_PRECISION_SETTINGS`
# in `test_rnnt_joint_fused_vocab.py`.
# Larger than for the small dimensions: longer sums in FP32 for the outputs, and up to 17 vocab tiles accumulated
# with BF16 atomics for the joint hidden gradient in the atomic backward
TOLERANCES = [5e-5, 6e-2, 1e-2, 1e-5]

# with the consistency loss, backward mode, pruning range, pruned lattice.
# Pruning range without the pruned lattice: backward pruning (dense forward, ranges selected by the gradients);
# pruned lattice: the joint hidden is given only for the pruning ranges, forward and backward are pruned
TRITON_CASES = [
    pytest.param(False, "atomic", None, False, id="rnnt-atomic"),
    pytest.param(False, "chunked", None, False, id="rnnt-chunked"),
    pytest.param(False, "atomic", PRUNING_RANGE_SIZE, False, id="rnnt-atomic-pruned"),
    pytest.param(False, "chunked", PRUNING_RANGE_SIZE, False, id="rnnt-chunked-pruned"),
    pytest.param(False, "atomic", PRUNING_RANGE_SIZE, True, id="rnnt-atomic-pruned-lattice"),
    pytest.param(True, "atomic", None, False, id="consistency-atomic"),
    pytest.param(True, "chunked", None, False, id="consistency-chunked"),
    pytest.param(True, "atomic", PRUNING_RANGE_SIZE, False, id="consistency-atomic-pruned"),
    pytest.param(True, "chunked", PRUNING_RANGE_SIZE, False, id="consistency-chunked-pruned"),
    pytest.param(True, "atomic", PRUNING_RANGE_SIZE, True, id="consistency-atomic-pruned-lattice"),
]
# The chunked backward kernel is the same for backward pruning and for the pruned lattice (it is not autotuned
# again), the rows of the pruning ranges are passed differently
PRUNED_LATTICE_CHUNKED_CASES = [
    pytest.param(False, "chunked", PRUNING_RANGE_SIZE, True, id="rnnt-chunked-pruned-lattice"),
    pytest.param(True, "chunked", PRUNING_RANGE_SIZE, True, id="consistency-chunked-pruned-lattice"),
]


@pytest.fixture(scope="module")
def triton_cache_dir(tmp_path_factory):
    """Empty Triton cache shared by the tests of the module: kernels are compiled once, autotuning is not cached"""
    return tmp_path_factory.mktemp("triton_cache")


@requires_triton
@pytest.mark.parametrize("with_consistency,backward_mode,pruning_range_size,pruned_lattice", TRITON_CASES)
def test_triton_autotuning(
    with_consistency,
    backward_mode,
    pruning_range_size,
    pruned_lattice,
    production_kernel_configs,
    triton_cache_dir,
    monkeypatch,
):
    """
    The call that triggers autotuning returns correct results (the benchmarked kernels accumulate gradients in
    the output tensors, which are reset after benchmarking), as well as the following call with the selected config.
    """
    monkeypatch.setenv("TRITON_CACHE_DIR", str(triton_cache_dir))
    forward_kernel, backward_kernel = _get_triton_kernels(production_kernel_configs, with_consistency, backward_mode)
    assert len(forward_kernel.configs) > 1 and len(backward_kernel.configs) > 1
    num_tuned_backward_keys = len(backward_kernel.cache)
    for step in range(2):
        _run_and_compare_with_reference(
            with_consistency, backward_mode, pruning_range_size, pruned_lattice=pruned_lattice, seed=step
        )
        # backward kernel was autotuned by the first call for the new key (dims, precision, pruning range)
        assert len(backward_kernel.cache) == num_tuned_backward_keys + 1


@requires_triton
@pytest.mark.parametrize(
    "with_consistency,backward_mode,pruning_range_size,pruned_lattice", TRITON_CASES + PRUNED_LATTICE_CHUNKED_CASES
)
def test_each_triton_config(
    with_consistency,
    backward_mode,
    pruning_range_size,
    pruned_lattice,
    production_kernel_configs,
    triton_cache_dir,
    monkeypatch,
):
    """
    Each production config (not only the fastest one, selected by autotuning on the current GPU) either
    computes correct results or is rejected by the compiler due to the resources of the GPU (e.g., shared memory).
    """
    monkeypatch.setenv("TRITON_CACHE_DIR", str(triton_cache_dir))
    forward_kernel, backward_kernel = _get_triton_kernels(production_kernel_configs, with_consistency, backward_mode)
    forward_configs, backward_configs = forward_kernel.configs, backward_kernel.configs
    valid_forward_configs = []
    for forward_config in forward_configs:
        monkeypatch.setattr(forward_kernel, "configs", [forward_config])
        try:
            _run_and_compare_with_reference(
                with_consistency, backward_mode, pruning_range_size, pruned_lattice=pruned_lattice, forward_only=True
            )
        except OutOfResources:
            continue
        valid_forward_configs.append(forward_config)
    assert len(valid_forward_configs) > 0, "No valid config for the forward kernel"

    monkeypatch.setattr(forward_kernel, "configs", valid_forward_configs[:1])
    num_valid_backward_configs = 0
    for backward_config in backward_configs:
        monkeypatch.setattr(backward_kernel, "configs", [backward_config])
        try:
            _run_and_compare_with_reference(
                with_consistency, backward_mode, pruning_range_size, pruned_lattice=pruned_lattice
            )
        except OutOfResources:
            continue
        num_valid_backward_configs += 1
    assert num_valid_backward_configs > 0, "No valid config for the backward kernel"


@requires_cutile
@pytest.mark.parametrize("grad_full_precision", [False, True])
def test_cutile_config(grad_full_precision, production_kernel_configs):
    kernel_kwargs = {}
    if grad_full_precision:
        kernel_kwargs = dict(input_grad_full_precision=True, weight_grad_full_precision=True)
    elif torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("cuTile backend requires compute capability 9.0+ for BF16 atomics")
    _run_and_compare_with_reference(
        with_consistency=False, backward_mode=None, pruning_range_size=None, backend="cutile", **kernel_kwargs
    )


def _get_triton_kernels(autotuned_kernels: dict, with_consistency: bool, backward_mode: str):
    """Autotuned forward and backward kernels used for the Joint with/without consistency and the backward mode"""
    prefix = "_rnnt_joint_fused_vocab_consistency" if with_consistency else "_rnnt_joint_fused_vocab"
    backward_suffix = {"atomic": "bwd", "chunked": "bwd_grad_logits"}[backward_mode]
    return (
        autotuned_kernels[f"{prefix}_fwd_triton_kernel"],
        autotuned_kernels[f"{prefix}_{backward_suffix}_triton_kernel"],
    )


def _run_and_compare_with_reference(
    with_consistency: bool,
    backward_mode: str | None,
    pruning_range_size: int | None,
    pruned_lattice: bool = False,
    forward_only: bool = False,
    seed: int = 0,
    **kernel_kwargs,
):
    """
    Forward and backward of the Joint: compare outputs and gradients with the reference from the logits.
    With `pruned_lattice`, the Joint is evaluated only for random pruning ranges of size `pruning_range_size`;
    otherwise `pruning_range_size` enables backward pruning.
    """
    inputs = make_joint_inputs(
        hidden_dtype=torch.bfloat16,
        dims=PRODUCTION_DIMS,
        src_lengths=SRC_LENGTHS,
        tgt_lengths=TGT_LENGTHS,
        double_batch=with_consistency,
        seed=seed,
    )
    leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
    joint_function = rnnt_joint_fused_vocab_with_consistency if with_consistency else rnnt_joint_fused_vocab
    tensor_kwargs = inputs.kernel_kwargs(leaves)
    reference_kwargs = dict(
        weight_matmul_dtype=torch.bfloat16, consistency_symmetrical=True if with_consistency else None
    )
    if pruned_lattice:
        range_starts = random_pruning_range_starts(inputs, pruning_range_size, paired=with_consistency)
        tensor_kwargs["joint_hidden"] = gather_pruning_ranges(leaves[0], range_starts, pruning_range_size)
        kernel_kwargs["pruning_range_starts"] = range_starts
        expected_outputs, _ = reference_pruned_lattice_outputs(
            inputs, reference_leaves, range_starts, pruning_range_size, **reference_kwargs
        )
        assert_outputs_close = assert_pruned_outputs_close_to_reference
    else:
        kernel_kwargs["backward_pruning_range_size"] = pruning_range_size
        expected_outputs = reference_outputs(inputs, reference_leaves, **reference_kwargs)
        assert_outputs_close = assert_close_to_reference
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = list(joint_function(**tensor_kwargs, backward_mode=backward_mode, **kernel_kwargs))
    assert_outputs_close(outputs, expected_outputs, TOLERANCES[0], OUTPUT_NAMES[: len(outputs)])
    if forward_only:
        return

    if pruned_lattice:
        grads = random_grads_like(outputs, seed=seed + 1)
    else:
        grads = inputs.random_grads(num_outputs=len(outputs), seed=seed + 1)
    reference_grads = [grad.double() for grad in grads]
    if pruning_range_size is not None and not pruned_lattice:
        _, mask = reference_pruning_windows(
            grads[0], grads[1], inputs.src_lengths, inputs.tgt_lengths, pruning_range_size, paired=with_consistency
        )
        masks = [mask, mask, mask[: mask.shape[0] // 2]]
        reference_grads = [grad * grad_mask for grad, grad_mask in zip(reference_grads, masks)]
    torch.autograd.backward(outputs, grads)
    torch.autograd.backward(expected_outputs, reference_grads)
    assert_close_to_reference(
        [leaf.grad for leaf in leaves], [leaf.grad for leaf in reference_leaves], TOLERANCES[1:], GRAD_NAMES
    )
