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
Tests for the `fused_vocab` Joint: target/blank log-probs from the joint hidden without materializing the logits.

All kernels are compiled with a single small-tile config (see `conftest.py`) and share the dimensions and
the precision settings from `utils.py`: a kernel is compiled once for each of them (0.3-0.6 seconds without
the cache), so the tests that compile a new kernel take up to a second, all the other tests - milliseconds.
The first test in the session that uses Triton also initializes its compiler, which takes 1-2 seconds more.
The production autotune configs are covered by `test_rnnt_joint_fused_vocab_production_configs.py`.
"""

import math

import pytest
import torch

from nemo.collections.asr.parts.turbo_transducer.backend import TurboTransducerBackend
from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab import rnnt_joint_fused_vocab
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.decoding.utils import avoid_sync_operations
from tests.collections.asr.turbo_transducer.utils import (
    DIMS_BELOW_TILE_BLANK_MIDDLE,
    DIMS_DEFAULT,
    DIMS_TILE_ALIGNED_BLANK_FIRST,
    FULL_PRECISION,
    FULL_PRECISION_TOLERANCE,
    LATTICES_WITH_DIFFERENT_SPECIALIZATION,
    assert_close_to_reference,
    count_compiled_kernels,
    make_joint_inputs,
    reference_outputs,
    skip_if_bf16_atomics_unsupported,
)

if TRITON_AVAILABLE:
    import triton
    from triton.runtime.autotuner import Autotuner

    from nemo.collections.asr.parts.turbo_transducer import (
        rnnt_joint_fused_vocab_consistency_triton as consistency_triton_module,
    )
    from nemo.collections.asr.parts.turbo_transducer import rnnt_joint_fused_vocab_triton as fused_vocab_triton_module
    from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab_triton import FusedVocabBackwardMode

pytestmark = pytest.mark.unit

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
requires_triton = pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")

# Both backward algorithms are tested on any GPU, independent of the GPU-based default selection
BACKWARD_MODES = ["atomic", "chunked"]

# (hidden_dim, vocab_size, blank_id)
FULL_PRECISION_DIMS = [
    pytest.param(DIMS_DEFAULT, id="default-dims"),
    pytest.param(DIMS_TILE_ALIGNED_BLANK_FIRST, id="tile-aligned-dims-blank-first"),
    pytest.param(DIMS_BELOW_TILE_BLANK_MIDDLE, id="dims-below-tile-blank-middle"),
]

OUTPUT_NAMES = ["target_logprobs", "blank_logprobs"]
GRAD_NAMES = ["joint_hidden grad", "weight grad", "bias grad"]

# Mixed precision settings: BF16 joint hidden (as under autocast) with BF16 parameters or FP32 master weights.
# Tolerances are relative to the largest magnitude of the reference, at least 3x of the errors observed
# for 30 random inputs (accumulation with atomics is not deterministic); errors of wrong kernels are ~1:
# - log-probs: products of BF16 operands are exact in FP32 => exact; TF32 matmul of FP32 operands is not exact
# - joint hidden gradient: always rounded to BF16 (2^-8 relative to the element);
#   more rounding if accumulated in BF16 with atomics
# - weight gradient: exact only with IEEE matmul, otherwise gradients wrt logits are rounded to BF16 / TF32
# - bias gradient: computed in FP32 from the logits; rounded to BF16 for BF16 parameters
# (param dtype, kernel kwargs, dtype of the weight operand of matmul,
#  tolerances: [log-probs, joint hidden grad, weight grad, bias grad], uses BF16 atomics in the atomic backward)
MIXED_PRECISION_SETTINGS = [
    pytest.param(torch.bfloat16, {}, torch.bfloat16, [1e-5, 3e-2, 6e-2, 6e-2], True, id="bf16-params"),
    pytest.param(torch.float32, {}, torch.bfloat16, [1e-5, 3e-2, 1e-2, 1e-5], True, id="fp32-params"),
    pytest.param(
        torch.float32,
        dict(mm_full_precision=True, input_grad_full_precision=True, weight_grad_full_precision=True),
        torch.float32,
        [2e-3, 3e-2, 2e-3, 2e-3],
        False,
        id="fp32-params-mm-full-precision",
    ),
    # Regression: weight gradient was rounded to the dtype of the joint hidden gradient (BF16) in each tile
    pytest.param(
        torch.float32,
        dict(use_high_precision_mm=True, weight_grad_full_precision=True),
        torch.float32,
        [1e-5, 3e-2, 1e-5, 1e-5],
        True,
        id="fp32-params-high-precision-mm",
    ),
]


@pytest.mark.gpu
@requires_cuda
@requires_triton
class TestFusedVocabJoint:
    """
    Forward (independent of the backward mode) and backward are tested separately: each test compiles one kernel,
    if the tests are executed in the order of definition.
    """

    @pytest.mark.parametrize("dims", FULL_PRECISION_DIMS)
    def test_full_precision_forward_matches_reference(self, dims):
        """Exact FP32 computations: log-probs match the FP64 reference from the logits"""
        inputs = make_joint_inputs(dims=dims)
        outputs = rnnt_joint_fused_vocab(**inputs.kernel_kwargs(inputs.leaves()), **FULL_PRECISION)
        expected_outputs = reference_outputs(inputs, inputs.leaves(torch.float64))
        assert_close_to_reference(outputs, expected_outputs, FULL_PRECISION_TOLERANCE, OUTPUT_NAMES)

    @pytest.mark.parametrize("backward_mode", BACKWARD_MODES)
    @pytest.mark.parametrize("dims", FULL_PRECISION_DIMS)
    def test_full_precision_backward_matches_reference(self, backward_mode, dims, monkeypatch):
        """Exact FP32 computations: all gradients match the FP64 reference from the logits"""
        _use_min_chunk_size(monkeypatch)
        inputs = make_joint_inputs(dims=dims)
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        outputs = rnnt_joint_fused_vocab(**inputs.kernel_kwargs(leaves), backward_mode=backward_mode, **FULL_PRECISION)
        grads = inputs.random_grads()
        torch.autograd.backward(outputs, grads)
        torch.autograd.backward(reference_outputs(inputs, reference_leaves), [grad.double() for grad in grads])
        assert_close_to_reference(
            [leaf.grad for leaf in leaves],
            [leaf.grad for leaf in reference_leaves],
            FULL_PRECISION_TOLERANCE,
            GRAD_NAMES,
        )
        assert all(leaf.grad.dtype == torch.float32 for leaf in leaves)

    @pytest.mark.parametrize(
        "param_dtype,kernel_kwargs,weight_matmul_dtype,tolerances,uses_bf16_atomics", MIXED_PRECISION_SETTINGS
    )
    def test_mixed_precision_forward_matches_reference(
        self, param_dtype, kernel_kwargs, weight_matmul_dtype, tolerances, uses_bf16_atomics
    ):
        """
        BF16 joint hidden: FP32 log-probs, matmul operands are BF16 unless FP32 is requested
        (FP32 master weights do not promote the computations).
        """
        inputs = make_joint_inputs(hidden_dtype=torch.bfloat16, param_dtype=param_dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = rnnt_joint_fused_vocab(**inputs.kernel_kwargs(inputs.leaves()), **kernel_kwargs)
        expected_outputs = reference_outputs(
            inputs, inputs.leaves(torch.float64), weight_matmul_dtype=weight_matmul_dtype
        )
        assert all(output.dtype == torch.float32 for output in outputs)
        assert_close_to_reference(outputs, expected_outputs, tolerances[0], OUTPUT_NAMES)

    @pytest.mark.parametrize("backward_mode", BACKWARD_MODES)
    @pytest.mark.parametrize(
        "param_dtype,kernel_kwargs,weight_matmul_dtype,tolerances,uses_bf16_atomics", MIXED_PRECISION_SETTINGS
    )
    def test_mixed_precision_backward_matches_reference(
        self,
        backward_mode,
        param_dtype,
        kernel_kwargs,
        weight_matmul_dtype,
        tolerances,
        uses_bf16_atomics,
        monkeypatch,
    ):
        """BF16 joint hidden: matmul operands as in forward, gradients preserve the dtypes of the inputs"""
        if uses_bf16_atomics:
            skip_if_bf16_atomics_unsupported(backward_mode)
        _use_min_chunk_size(monkeypatch)
        inputs = make_joint_inputs(hidden_dtype=torch.bfloat16, param_dtype=param_dtype)
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        if backward_mode == "chunked":
            # GEMMs of the chunked backward: operands in the matmul dtype, accumulation of the weight grad in FP32
            original_addmm = fused_vocab_triton_module._addmm_full_precision_

            def check_matmul_operands(accumulator, mat1, mat2):
                assert accumulator.dtype == torch.float32
                assert mat1.dtype == mat2.dtype == weight_matmul_dtype
                original_addmm(accumulator, mat1, mat2)

            monkeypatch.setattr(fused_vocab_triton_module, "_addmm_full_precision_", check_matmul_operands)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = rnnt_joint_fused_vocab(
                **inputs.kernel_kwargs(leaves), backward_mode=backward_mode, **kernel_kwargs
            )
        expected_outputs = reference_outputs(inputs, reference_leaves, weight_matmul_dtype=weight_matmul_dtype)
        grads = inputs.random_grads()
        torch.autograd.backward(outputs, grads)
        torch.autograd.backward(expected_outputs, [grad.double() for grad in grads])
        assert_close_to_reference(
            [leaf.grad for leaf in leaves], [leaf.grad for leaf in reference_leaves], tolerances[1:], GRAD_NAMES
        )
        assert [leaf.grad.dtype for leaf in leaves] == [torch.bfloat16, param_dtype, param_dtype]

    @pytest.mark.parametrize("backward_mode", BACKWARD_MODES)
    def test_half_precision_matches_reference(self, backward_mode):
        """FP16 joint hidden (`16-mixed` precision): FP32 matmul operands (TF32 by default), FP16 gradient"""
        inputs = make_joint_inputs(hidden_dtype=torch.float16)
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = rnnt_joint_fused_vocab(**inputs.kernel_kwargs(leaves), backward_mode=backward_mode)
        expected_outputs = reference_outputs(inputs, reference_leaves)
        assert all(output.dtype == torch.float32 for output in outputs)
        # errors of TF32 matmuls and of the FP16 gradient: ~1e-3 for 30 random inputs
        assert_close_to_reference(outputs, expected_outputs, 5e-3, OUTPUT_NAMES)
        grads = inputs.random_grads()
        torch.autograd.backward(outputs, grads)
        torch.autograd.backward(expected_outputs, [grad.double() for grad in grads])
        assert_close_to_reference(
            [leaf.grad for leaf in leaves], [leaf.grad for leaf in reference_leaves], 5e-3, GRAD_NAMES
        )
        assert [leaf.grad.dtype for leaf in leaves] == [torch.float16, torch.float32, torch.float32]

    def test_chunked_backward_inside_autocast(self, monkeypatch):
        """Regression: GEMMs of the chunked backward were autocast to BF16 if backward was called inside autocast"""
        _use_min_chunk_size(monkeypatch)
        kernel_kwargs = dict(backward_mode="chunked", mm_full_precision=True)
        inputs = make_joint_inputs(hidden_dtype=torch.bfloat16)
        results = []
        for backward_inside_autocast in [False, True]:
            leaves = inputs.leaves()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = rnnt_joint_fused_vocab(**inputs.kernel_kwargs(leaves), **kernel_kwargs)
                if backward_inside_autocast:
                    torch.autograd.backward(outputs, inputs.random_grads())
            if not backward_inside_autocast:
                torch.autograd.backward(outputs, inputs.random_grads())
            results.append([leaf.grad for leaf in leaves])
        # the same computations (the bias gradient is accumulated with atomics: not bitwise reproducible);
        # the error of the joint hidden gradient computed with BF16 operands is ~4e-3
        assert_close_to_reference(results[1], results[0], 1e-6, GRAD_NAMES)

    @pytest.mark.parametrize("backward_mode", BACKWARD_MODES)
    @pytest.mark.parametrize(
        "src_lengths,tgt_lengths",
        [
            pytest.param((1,), (1,), id="single-frame"),
            pytest.param((4, 2), (0, 0), id="blank-only"),  # lattice [2, 4, 1], no targets
        ],
    )
    def test_edge_case_lattice(self, backward_mode, src_lengths, tgt_lengths):
        inputs = make_joint_inputs(src_lengths=src_lengths, tgt_lengths=tgt_lengths)
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        outputs = rnnt_joint_fused_vocab(**inputs.kernel_kwargs(leaves), backward_mode=backward_mode, **FULL_PRECISION)
        expected_outputs = reference_outputs(inputs, reference_leaves)
        assert_close_to_reference(outputs, expected_outputs, FULL_PRECISION_TOLERANCE, OUTPUT_NAMES)

        grads = inputs.random_grads()
        torch.autograd.backward(outputs, grads)
        torch.autograd.backward(expected_outputs, [grad.double() for grad in grads])
        assert_close_to_reference(
            [leaf.grad for leaf in leaves],
            [leaf.grad for leaf in reference_leaves],
            FULL_PRECISION_TOLERANCE,
            GRAD_NAMES,
        )

    @pytest.mark.parametrize("backward_mode", BACKWARD_MODES)
    @pytest.mark.parametrize("mixed_precision", [False, True])
    def test_no_cuda_sync_operations(self, backward_mode, mixed_precision):
        if mixed_precision:
            # BF16 joint hidden with FP32 master weights and default kernel arguments, as in training with autocast
            skip_if_bf16_atomics_unsupported(backward_mode)
            inputs, kernel_kwargs = make_joint_inputs(hidden_dtype=torch.bfloat16), {}
        else:
            inputs, kernel_kwargs = make_joint_inputs(), FULL_PRECISION

        def run_forward_backward():
            target_logprobs, blank_logprobs = rnnt_joint_fused_vocab(
                **inputs.kernel_kwargs(inputs.leaves()), backward_mode=backward_mode, **kernel_kwargs
            )
            (target_logprobs.sum() + blank_logprobs.sum()).backward()

        run_forward_backward()  # warmup: compile the kernels
        torch.cuda.synchronize()
        with avoid_sync_operations(torch.device("cuda")):
            run_forward_backward()


@pytest.mark.gpu
@requires_cuda
@requires_triton
def test_default_backend_and_backward_mode(monkeypatch):
    """Without explicit backend and backward mode: Triton backend with the backward mode selected by the GPU"""
    inputs = make_joint_inputs()
    leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
    selected_modes = []
    original_normalize = fused_vocab_triton_module._normalize_backward_options

    def record_backward_mode(*args):
        backward_mode, backward_pruning_range_size = original_normalize(*args)
        selected_modes.append(backward_mode)
        return backward_mode, backward_pruning_range_size

    monkeypatch.setattr(fused_vocab_triton_module, "_normalize_backward_options", record_backward_mode)
    outputs = rnnt_joint_fused_vocab(**inputs.kernel_kwargs(leaves), backend=None, **FULL_PRECISION)
    assert selected_modes == [fused_vocab_triton_module._get_default_backward_mode(torch.device("cuda"))]
    expected_outputs = reference_outputs(inputs, reference_leaves)
    assert_close_to_reference(outputs, expected_outputs, FULL_PRECISION_TOLERANCE, OUTPUT_NAMES)
    torch.autograd.backward(outputs, [torch.ones_like(output) for output in outputs])
    torch.autograd.backward(expected_outputs, [torch.ones_like(output) for output in expected_outputs])
    assert_close_to_reference(
        [leaf.grad for leaf in leaves], [leaf.grad for leaf in reference_leaves], FULL_PRECISION_TOLERANCE, GRAD_NAMES
    )


@pytest.mark.gpu
@requires_cuda
@requires_triton
@pytest.mark.parametrize("backward_mode", ["atomic", "chunked"])
def test_no_recompilation_for_new_lattice_shape(backward_mode, monkeypatch):
    """Regression: kernels were specialized (recompiled) by the lattice sizes equal to 1 or divisible by 16"""
    _use_min_chunk_size(monkeypatch)
    num_compiled_kernels = None
    for src_lengths, tgt_lengths in LATTICES_WITH_DIFFERENT_SPECIALIZATION:
        inputs = make_joint_inputs(src_lengths=src_lengths, tgt_lengths=tgt_lengths)
        target_logprobs, blank_logprobs = rnnt_joint_fused_vocab(
            **inputs.kernel_kwargs(inputs.leaves()), backward_mode=backward_mode, **FULL_PRECISION
        )
        (target_logprobs.sum() + blank_logprobs.sum()).backward()
        if num_compiled_kernels is None:
            num_compiled_kernels = count_compiled_kernels(fused_vocab_triton_module)
        assert count_compiled_kernels(fused_vocab_triton_module) == num_compiled_kernels


@pytest.mark.gpu
@requires_cuda
@requires_triton
def test_small_batches_are_autotuned_separately(monkeypatch):
    """
    Regression: the config selected by autotuning for a small first batch (e.g., a smoke run), which does not load
    the GPU, was cached on disk and used for the batches of any size. The size class is a part of the autotune key.
    """
    kernel = fused_vocab_triton_module._rnnt_joint_fused_vocab_fwd_triton_kernel
    # autotuning requires at least 2 configs (the tests use a single one); small tiles compile fast
    test_config = kernel.configs[0]
    monkeypatch.setattr(kernel, "configs", [test_config, triton.Config(test_config.kwargs, num_warps=2, num_stages=2)])
    monkeypatch.setattr(kernel, "cache", {})
    inputs = make_joint_inputs()
    num_rows = math.prod(inputs.lattice_shape)
    size_class_index = kernel.keys.index("autotune_size_class")
    for min_rows, expected_size_classes in [(num_rows + 1, [0]), (num_rows, [0, 1])]:
        monkeypatch.setattr(fused_vocab_triton_module, "_AUTOTUNE_MIN_ROWS", min_rows)
        with torch.no_grad():
            rnnt_joint_fused_vocab(**inputs.kernel_kwargs(inputs.leaves()), **FULL_PRECISION)
        assert sorted(key[size_class_index] for key in kernel.cache) == expected_size_classes


@pytest.mark.gpu
@requires_cuda
@requires_triton
def test_chunked_backward_is_autotuned_only_for_first_chunk(monkeypatch):
    """
    Regression: autotuning resets the bias gradient accumulated by the kernel, so it must be possible only for
    the first chunk. The last (smaller) chunk had another size class, was autotuned, and the gradient of
    the previous chunks was lost.
    """
    kernel = fused_vocab_triton_module._rnnt_joint_fused_vocab_bwd_grad_logits_triton_kernel
    test_config = kernel.configs[0]
    monkeypatch.setattr(kernel, "configs", [test_config, triton.Config(test_config.kwargs, num_warps=2, num_stages=2)])
    monkeypatch.setattr(kernel, "cache", {})
    _use_min_chunk_size(monkeypatch)  # chunks of 256, 256 and 232 rows; the vocabulary is padded to 64
    monkeypatch.setattr(fused_vocab_triton_module, "_AUTOTUNE_MIN_GRAD_LOGITS", 250 * 64)
    inputs = make_joint_inputs()
    assert math.prod(inputs.lattice_shape) == 744
    leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
    outputs = rnnt_joint_fused_vocab(**inputs.kernel_kwargs(leaves), backward_mode="chunked", **FULL_PRECISION)
    grads = inputs.random_grads()
    torch.autograd.backward(outputs, grads)
    torch.autograd.backward(reference_outputs(inputs, reference_leaves), [grad.double() for grad in grads])
    assert_close_to_reference(
        [leaf.grad for leaf in leaves], [leaf.grad for leaf in reference_leaves], FULL_PRECISION_TOLERANCE, GRAD_NAMES
    )
    assert len(kernel.cache) == 1


@requires_triton
def test_autotuned_kernels_have_size_class():
    """All autotuned kernels: the size class is in the autotune key and does not specialize (recompile) the kernel"""
    kernels = [
        kernel
        for module in (fused_vocab_triton_module, consistency_triton_module)
        for kernel in vars(module).values()
        if isinstance(kernel, Autotuner)
    ]
    assert len(kernels) == 6
    for kernel in kernels:
        assert "autotune_size_class" in kernel.keys
        assert "autotune_size_class" in kernel.fn.do_not_specialize


@requires_triton
@pytest.mark.parametrize(
    "invalid_kwargs,error",
    [
        # targets [B, U+1] instead of [B, U]: labels of all utterances except the first one would be shifted
        (dict(targets=torch.zeros(3, 8, dtype=torch.long)), ValueError),
        (dict(targets=torch.zeros(2, 7, dtype=torch.long)), ValueError),
        (dict(targets=torch.zeros(3, 7)), ValueError),
        (dict(weight=torch.zeros(35, 41)), ValueError),
        (dict(bias=torch.zeros(34)), ValueError),
        (dict(src_lengths=torch.ones(2, dtype=torch.long)), ValueError),
        (dict(tgt_lengths=torch.ones(4, dtype=torch.long)), ValueError),
        (dict(blank_id=35), ValueError),
        (dict(blank_id=-1), ValueError),
        (dict(joint_hidden=torch.zeros(3, 31, 8)), ValueError),
        # FP64 would be computed in FP32 silently
        (dict(joint_hidden=torch.zeros(3, 31, 8, 40, dtype=torch.float64)), TypeError),
    ],
)
def test_rejects_inconsistent_inputs(invalid_kwargs, error):
    """The kernels index the tensors by the shape of the joint hidden: inconsistent inputs are rejected"""
    inputs = make_joint_inputs(device="cpu")
    assert inputs.lattice_shape == (3, 31, 8) and inputs.joint_hidden.shape[-1] == 40 and inputs.blank_id == 34
    kernel_kwargs = dict(inputs.kernel_kwargs(inputs.leaves()), **invalid_kwargs)
    with pytest.raises(error):
        rnnt_joint_fused_vocab(**kernel_kwargs)


@requires_triton
@pytest.mark.parametrize(
    "capability,expected_mode",
    [
        ((7, 5), "chunked"),
        ((8, 0), "chunked"),
        ((8, 6), "chunked"),
        ((8, 9), "chunked"),
        ((9, 0), "atomic"),
        ((10, 0), "atomic"),
        ((12, 0), "atomic"),
    ],
)
@pytest.mark.parametrize("bf16_atomics_available", [True, False])
def test_default_backward_mode(capability, expected_mode, bf16_atomics_available, monkeypatch):
    """Atomic backward requires native BF16 atomics (compute capability 9.0+), supported by Triton 3.4+"""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(fused_vocab_triton_module, "BF16_ATOMICS_AVAILABLE", bf16_atomics_available)
    backward_mode = fused_vocab_triton_module._get_default_backward_mode(torch.device("cuda"))
    assert backward_mode == FusedVocabBackwardMode(expected_mode if bf16_atomics_available else "chunked")


@requires_triton
def test_atomic_backward_requires_bf16_atomics(monkeypatch):
    """Triton < 3.4: a clear error instead of a failed compilation of the kernel with BF16 atomics"""
    monkeypatch.setattr(fused_vocab_triton_module, "BF16_ATOMICS_AVAILABLE", False)
    fp32_accumulator, bf16_accumulator = torch.zeros(1), torch.zeros(1, dtype=torch.bfloat16)
    fused_vocab_triton_module._check_atomic_accumulators(fp32_accumulator, fp32_accumulator)
    with pytest.raises(RuntimeError, match="requires Triton 3.4+"):
        fused_vocab_triton_module._check_atomic_accumulators(fp32_accumulator, bf16_accumulator)


@pytest.mark.parametrize("backend", [TurboTransducerBackend.CUTILE, "cutile"])
def test_cutile_backend_is_not_implemented(backend):
    """cuTile backend is planned: the request is rejected, not replaced silently with Triton"""
    tensor_args = dict.fromkeys(["joint_hidden", "targets", "tgt_lengths", "src_lengths", "weight", "bias"])
    with pytest.raises(NotImplementedError, match="cuTile"):
        rnnt_joint_fused_vocab(**tensor_args, blank_id=0, backend=backend)


@pytest.mark.parametrize("backend", ["torch", "unknown"])
def test_rejects_unsupported_backend(backend):
    tensor_args = dict.fromkeys(["joint_hidden", "targets", "tgt_lengths", "src_lengths", "weight", "bias"])
    with pytest.raises(ValueError):
        rnnt_joint_fused_vocab(**tensor_args, blank_id=0, backend=backend)


def _use_min_chunk_size(monkeypatch):
    """Chunked backward with the minimal chunk (256 rows): several chunks and a partial last chunk"""
    if TRITON_AVAILABLE:
        monkeypatch.setattr(fused_vocab_triton_module, "CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES", 1)
