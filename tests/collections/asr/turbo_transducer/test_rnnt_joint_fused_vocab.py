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
The first test of a backend in the session also initializes its compiler, which takes 1-2 seconds more.
The production autotune configs are covered by `test_rnnt_joint_fused_vocab_production_configs.py`.
"""

import pytest
import torch

from nemo.collections.asr.parts.turbo_transducer.backend import TurboTransducerBackend
from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab import rnnt_joint_fused_vocab
from nemo.core.utils.optional_libs import CUTILE_AVAILABLE, TRITON_AVAILABLE
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
    from nemo.collections.asr.parts.turbo_transducer import rnnt_joint_fused_vocab_triton as fused_vocab_triton_module
    from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab_triton import FusedVocabBackwardMode

pytestmark = pytest.mark.unit

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
requires_triton = pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
requires_cutile = pytest.mark.skipif(not CUTILE_AVAILABLE, reason="cuTile is not installed")

# (backend, backward mode): both backward algorithms of the Triton backend are tested on any GPU,
# independent of the GPU-based default selection
IMPLEMENTATIONS = [
    pytest.param(TurboTransducerBackend.TRITON, "atomic", marks=requires_triton, id="triton-atomic"),
    pytest.param(TurboTransducerBackend.TRITON, "chunked", marks=requires_triton, id="triton-chunked"),
    pytest.param(TurboTransducerBackend.CUTILE, None, marks=requires_cutile, id="cutile"),
]

BACKENDS = [
    pytest.param(TurboTransducerBackend.TRITON, marks=requires_triton, id="triton"),
    pytest.param(TurboTransducerBackend.CUTILE, marks=requires_cutile, id="cutile"),
]

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
class TestFusedVocabJoint:
    """
    Forward (independent of the backward mode) and backward are tested separately: each test compiles one kernel,
    if the tests are executed in the order of definition.
    """

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("dims", FULL_PRECISION_DIMS)
    def test_full_precision_forward_matches_reference(self, backend, dims):
        """Exact FP32 computations: log-probs match the FP64 reference from the logits"""
        inputs = make_joint_inputs(dims=dims)
        outputs = rnnt_joint_fused_vocab(**inputs.kernel_kwargs(inputs.leaves()), backend=backend, **FULL_PRECISION)
        expected_outputs = reference_outputs(inputs, inputs.leaves(torch.float64))
        assert_close_to_reference(outputs, expected_outputs, FULL_PRECISION_TOLERANCE, OUTPUT_NAMES)

    @pytest.mark.parametrize("backend,backward_mode", IMPLEMENTATIONS)
    @pytest.mark.parametrize("dims", FULL_PRECISION_DIMS)
    def test_full_precision_backward_matches_reference(self, backend, backward_mode, dims, monkeypatch):
        """Exact FP32 computations: all gradients match the FP64 reference from the logits"""
        _use_min_chunk_size(monkeypatch)
        inputs = make_joint_inputs(dims=dims)
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        outputs = rnnt_joint_fused_vocab(
            **inputs.kernel_kwargs(leaves), backend=backend, backward_mode=backward_mode, **FULL_PRECISION
        )
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

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize(
        "param_dtype,kernel_kwargs,weight_matmul_dtype,tolerances,uses_bf16_atomics", MIXED_PRECISION_SETTINGS
    )
    def test_mixed_precision_forward_matches_reference(
        self, backend, param_dtype, kernel_kwargs, weight_matmul_dtype, tolerances, uses_bf16_atomics
    ):
        """
        BF16 joint hidden: FP32 log-probs, matmul operands are BF16 unless FP32 is requested
        (FP32 master weights do not promote the computations).
        """
        inputs = make_joint_inputs(hidden_dtype=torch.bfloat16, param_dtype=param_dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = rnnt_joint_fused_vocab(**inputs.kernel_kwargs(inputs.leaves()), backend=backend, **kernel_kwargs)
        expected_outputs = reference_outputs(
            inputs, inputs.leaves(torch.float64), weight_matmul_dtype=weight_matmul_dtype
        )
        assert all(output.dtype == torch.float32 for output in outputs)
        assert_close_to_reference(outputs, expected_outputs, tolerances[0], OUTPUT_NAMES)

    @pytest.mark.parametrize("backend,backward_mode", IMPLEMENTATIONS)
    @pytest.mark.parametrize(
        "param_dtype,kernel_kwargs,weight_matmul_dtype,tolerances,uses_bf16_atomics", MIXED_PRECISION_SETTINGS
    )
    def test_mixed_precision_backward_matches_reference(
        self,
        backend,
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
            _skip_if_bf16_atomics_unsupported(backend, backward_mode)
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
                **inputs.kernel_kwargs(leaves), backend=backend, backward_mode=backward_mode, **kernel_kwargs
            )
        expected_outputs = reference_outputs(inputs, reference_leaves, weight_matmul_dtype=weight_matmul_dtype)
        grads = inputs.random_grads()
        torch.autograd.backward(outputs, grads)
        torch.autograd.backward(expected_outputs, [grad.double() for grad in grads])
        assert_close_to_reference(
            [leaf.grad for leaf in leaves], [leaf.grad for leaf in reference_leaves], tolerances[1:], GRAD_NAMES
        )
        assert [leaf.grad.dtype for leaf in leaves] == [torch.bfloat16, param_dtype, param_dtype]

    @pytest.mark.parametrize("backend,backward_mode", IMPLEMENTATIONS)
    @pytest.mark.parametrize(
        "src_lengths,tgt_lengths",
        [
            pytest.param((1,), (1,), id="single-frame"),
            pytest.param((4, 2), (0, 0), id="blank-only"),  # lattice [2, 4, 1], no targets
        ],
    )
    def test_edge_case_lattice(self, backend, backward_mode, src_lengths, tgt_lengths):
        # NB: cuTile compiles the forward and backward kernels again for some of the new lattice shapes
        # (0.4 seconds per kernel), so the test can take about a second with the cuTile backend
        inputs = make_joint_inputs(src_lengths=src_lengths, tgt_lengths=tgt_lengths)
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        outputs = rnnt_joint_fused_vocab(
            **inputs.kernel_kwargs(leaves), backend=backend, backward_mode=backward_mode, **FULL_PRECISION
        )
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

    @pytest.mark.parametrize("backend,backward_mode", IMPLEMENTATIONS)
    @pytest.mark.parametrize("mixed_precision", [False, True])
    def test_no_cuda_sync_operations(self, backend, backward_mode, mixed_precision):
        if mixed_precision:
            # BF16 joint hidden with FP32 master weights and default kernel arguments, as in training with autocast
            _skip_if_bf16_atomics_unsupported(backend, backward_mode)
            inputs, kernel_kwargs = make_joint_inputs(hidden_dtype=torch.bfloat16), {}
        else:
            inputs, kernel_kwargs = make_joint_inputs(), FULL_PRECISION

        def run_forward_backward():
            target_logprobs, blank_logprobs = rnnt_joint_fused_vocab(
                **inputs.kernel_kwargs(inputs.leaves()), backend=backend, backward_mode=backward_mode, **kernel_kwargs
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


def test_cutile_rejects_triton_only_options():
    tensor_args = dict.fromkeys(["joint_hidden", "targets", "tgt_lengths", "src_lengths", "weight", "bias"])
    with pytest.raises(NotImplementedError, match="Triton"):
        rnnt_joint_fused_vocab(**tensor_args, blank_id=0, backend="cutile", backward_pruning_range_size=2)
    with pytest.raises(ValueError, match="Triton"):
        rnnt_joint_fused_vocab(**tensor_args, blank_id=0, backend="cutile", backward_mode="chunked")


def test_rejects_unknown_backend():
    tensor_args = dict.fromkeys(["joint_hidden", "targets", "tgt_lengths", "src_lengths", "weight", "bias"])
    with pytest.raises(ValueError):
        rnnt_joint_fused_vocab(**tensor_args, blank_id=0, backend="unknown")


def _use_min_chunk_size(monkeypatch):
    """Chunked backward with the minimal chunk (256 rows): several chunks and a partial last chunk"""
    if TRITON_AVAILABLE:
        monkeypatch.setattr(fused_vocab_triton_module, "CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES", 1)


def _skip_if_bf16_atomics_unsupported(backend: TurboTransducerBackend, backward_mode: str | None):
    """
    Accumulation of gradients in BF16 requires BF16 atomics, native only for compute capability 9.0+.
    Triton emulates them on older GPUs with compare-and-swap loops (Triton 3.4+), cuTile does not compile.
    """
    if backend == TurboTransducerBackend.CUTILE and torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("cuTile backend requires compute capability 9.0+ for BF16 atomics")
    if backend == TurboTransducerBackend.TRITON:
        skip_if_bf16_atomics_unsupported(backward_mode)
