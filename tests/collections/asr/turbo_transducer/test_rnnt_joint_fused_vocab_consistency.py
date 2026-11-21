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
Tests for the `fused_vocab` Joint computing the consistency loss inside the Triton kernels.

The kernels are compiled with a single small-tile config and share dimensions and precision settings with the other
tests of the fused Joint, see `conftest.py` and `utils.py`. Backward pruning with the consistency loss is tested
in `test_rnnt_joint_pruning.py`.
"""

import pytest
import torch

from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab import (
    rnnt_joint_fused_vocab,
    rnnt_joint_fused_vocab_with_consistency,
)
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.decoding.utils import avoid_sync_operations
from tests.collections.asr.turbo_transducer.utils import (
    BLANK_ID,
    DIMS_BELOW_TILE_BLANK_MIDDLE,
    DIMS_DEFAULT,
    DIMS_TILE_ALIGNED_BLANK_FIRST,
    FULL_PRECISION,
    FULL_PRECISION_TOLERANCE,
    HIDDEN_DIM,
    LATTICES_WITH_DIFFERENT_SPECIALIZATION,
    PRUNING_RANGE_SIZE,
    assert_close_to_reference,
    count_compiled_kernels,
    make_joint_inputs,
    reference_outputs,
)

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer import (
        rnnt_joint_fused_vocab_consistency_triton as consistency_triton_module,
    )

pytestmark = pytest.mark.unit

requires_triton = pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
requires_triton_cuda = pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()), reason="Triton and CUDA are required"
)

OUTPUT_NAMES = ["target_logprobs", "blank_logprobs", "consistency_kl"]
GRAD_NAMES = ["joint_hidden grad", "weight grad", "bias grad"]


# (hidden_dim, vocab_size, blank_id), symmetrical consistency loss
FULL_PRECISION_CASES = [
    pytest.param(DIMS_DEFAULT, True, id="default-dims-symmetrical"),
    pytest.param(DIMS_DEFAULT, False, id="default-dims"),
    pytest.param(DIMS_TILE_ALIGNED_BLANK_FIRST, False, id="tile-aligned-dims-blank-first"),
    pytest.param(DIMS_BELOW_TILE_BLANK_MIDDLE, True, id="dims-below-tile-blank-middle-symmetrical"),
]

# BF16 joint hidden with default kernel arguments (BF16 matmul operands): param dtype, symmetrical consistency loss,
# tolerances [outputs, joint hidden grad, weight grad, bias grad], see `MIXED_PRECISION_SETTINGS`
# in `test_rnnt_joint_fused_vocab.py`
MIXED_PRECISION_CASES = [
    pytest.param(torch.float32, True, [1e-5, 3e-2, 1e-2, 1e-5], id="fp32-params-symmetrical"),
    pytest.param(torch.bfloat16, False, [1e-5, 3e-2, 6e-2, 6e-2], id="bf16-params"),
]

BACKWARD_MODES = ["atomic", "chunked"]


@pytest.mark.gpu
@requires_triton_cuda
class TestFusedVocabJointWithConsistency:
    """
    Forward (independent of the backward mode) and backward are tested separately: each test compiles one kernel,
    if the tests are executed in the order of definition.
    """

    @pytest.mark.parametrize("dims,symmetrical", FULL_PRECISION_CASES)
    def test_full_precision_forward_matches_reference(self, dims, symmetrical):
        """Exact FP32 computations: log-probs and the consistency loss match the FP64 reference from the logits"""
        inputs = make_joint_inputs(dims=dims, double_batch=True)
        outputs = rnnt_joint_fused_vocab_with_consistency(
            **inputs.kernel_kwargs(inputs.leaves()), consistency_symmetrical=symmetrical, **FULL_PRECISION
        )
        expected_outputs = reference_outputs(inputs, inputs.leaves(torch.float64), consistency_symmetrical=symmetrical)
        assert_close_to_reference(outputs, expected_outputs, FULL_PRECISION_TOLERANCE, OUTPUT_NAMES)

    @pytest.mark.parametrize("backward_mode", BACKWARD_MODES)
    @pytest.mark.parametrize("dims,symmetrical", FULL_PRECISION_CASES)
    def test_full_precision_backward_matches_reference(self, backward_mode, dims, symmetrical, monkeypatch):
        """Exact FP32 computations: gradients for all outputs match the FP64 reference from the logits"""
        _use_min_chunk_size(monkeypatch)
        inputs = make_joint_inputs(dims=dims, double_batch=True)
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        outputs = rnnt_joint_fused_vocab_with_consistency(
            **inputs.kernel_kwargs(leaves),
            consistency_symmetrical=symmetrical,
            backward_mode=backward_mode,
            **FULL_PRECISION,
        )
        expected_outputs = reference_outputs(inputs, reference_leaves, consistency_symmetrical=symmetrical)
        grads = inputs.random_grads(num_outputs=3)
        torch.autograd.backward(tuple(outputs), grads)
        torch.autograd.backward(expected_outputs, [grad.double() for grad in grads])
        assert_close_to_reference(
            [leaf.grad for leaf in leaves],
            [leaf.grad for leaf in reference_leaves],
            FULL_PRECISION_TOLERANCE,
            GRAD_NAMES,
        )

    @pytest.mark.parametrize("param_dtype,symmetrical,tolerances", MIXED_PRECISION_CASES)
    def test_mixed_precision_forward_matches_reference(self, param_dtype, symmetrical, tolerances):
        """BF16 joint hidden with BF16 parameters or FP32 master weights: FP32 outputs from BF16 matmul operands"""
        inputs = make_joint_inputs(hidden_dtype=torch.bfloat16, param_dtype=param_dtype, double_batch=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = rnnt_joint_fused_vocab_with_consistency(
                **inputs.kernel_kwargs(inputs.leaves()), consistency_symmetrical=symmetrical
            )
        expected_outputs = reference_outputs(
            inputs,
            inputs.leaves(torch.float64),
            weight_matmul_dtype=torch.bfloat16,
            consistency_symmetrical=symmetrical,
        )
        assert all(output.dtype == torch.float32 for output in outputs)
        assert_close_to_reference(outputs, expected_outputs, tolerances[0], OUTPUT_NAMES)

    @pytest.mark.parametrize("backward_mode", BACKWARD_MODES)
    @pytest.mark.parametrize("param_dtype,symmetrical,tolerances", MIXED_PRECISION_CASES)
    def test_mixed_precision_backward_matches_reference(
        self, backward_mode, param_dtype, symmetrical, tolerances, monkeypatch
    ):
        """BF16 joint hidden: BF16 matmul operands in backward, gradients preserve the dtypes of the inputs"""
        _use_min_chunk_size(monkeypatch)
        inputs = make_joint_inputs(hidden_dtype=torch.bfloat16, param_dtype=param_dtype, double_batch=True)
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        if backward_mode == "chunked":
            # GEMMs of the chunked backward: BF16 operands, accumulation of the weight grad in FP32
            original_addmm = consistency_triton_module._addmm_full_precision_

            def check_matmul_operands(accumulator, mat1, mat2):
                assert accumulator.dtype == torch.float32
                assert mat1.dtype == mat2.dtype == torch.bfloat16
                original_addmm(accumulator, mat1, mat2)

            monkeypatch.setattr(consistency_triton_module, "_addmm_full_precision_", check_matmul_operands)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = rnnt_joint_fused_vocab_with_consistency(
                **inputs.kernel_kwargs(leaves), consistency_symmetrical=symmetrical, backward_mode=backward_mode
            )
        expected_outputs = reference_outputs(
            inputs, reference_leaves, weight_matmul_dtype=torch.bfloat16, consistency_symmetrical=symmetrical
        )
        grads = inputs.random_grads(num_outputs=3)
        torch.autograd.backward(tuple(outputs), grads)
        torch.autograd.backward(expected_outputs, [grad.double() for grad in grads])
        assert_close_to_reference(
            [leaf.grad for leaf in leaves], [leaf.grad for leaf in reference_leaves], tolerances[1:], GRAD_NAMES
        )
        assert [leaf.grad.dtype for leaf in leaves] == [torch.bfloat16, param_dtype, param_dtype]

    @pytest.mark.parametrize("backward_mode", BACKWARD_MODES)
    def test_matches_kernel_without_consistency(self, backward_mode):
        """Without upstream gradient of the consistency loss: the same log-probs and gradients as without the loss"""
        inputs = make_joint_inputs(double_batch=True)
        results = []
        for with_consistency in [False, True]:
            leaves = inputs.leaves()
            if with_consistency:
                target_logprobs, blank_logprobs, _ = rnnt_joint_fused_vocab_with_consistency(
                    **inputs.kernel_kwargs(leaves), backward_mode=backward_mode, **FULL_PRECISION
                )
            else:
                target_logprobs, blank_logprobs = rnnt_joint_fused_vocab(
                    **inputs.kernel_kwargs(leaves), backward_mode=backward_mode, **FULL_PRECISION
                )
            torch.autograd.backward([target_logprobs, blank_logprobs], inputs.random_grads())
            results.append([target_logprobs, blank_logprobs] + [leaf.grad for leaf in leaves])
        assert_close_to_reference(results[1], results[0], FULL_PRECISION_TOLERANCE, OUTPUT_NAMES[:2] + GRAD_NAMES)

    @pytest.mark.parametrize("backward_mode", BACKWARD_MODES)
    def test_no_cuda_sync_operations(self, backward_mode):
        # BF16 joint hidden with FP32 master weights and default kernel arguments, as in training with autocast
        inputs = make_joint_inputs(hidden_dtype=torch.bfloat16, double_batch=True)

        def run_forward_backward():
            outputs = rnnt_joint_fused_vocab_with_consistency(
                **inputs.kernel_kwargs(inputs.leaves()), backward_mode=backward_mode
            )
            sum(output.sum() for output in outputs).backward()

        run_forward_backward()  # warmup: compile the kernels
        torch.cuda.synchronize()
        with avoid_sync_operations(torch.device("cuda")):
            run_forward_backward()

    @pytest.mark.parametrize("backward_mode", BACKWARD_MODES)
    def test_no_recompilation_for_new_lattice_shape(self, backward_mode, monkeypatch):
        """Regression: kernels were specialized (recompiled) by the lattice sizes equal to 1 or divisible by 16"""
        _use_min_chunk_size(monkeypatch)
        num_compiled_kernels = None
        for src_lengths, tgt_lengths in LATTICES_WITH_DIFFERENT_SPECIALIZATION:
            inputs = make_joint_inputs(src_lengths=src_lengths, tgt_lengths=tgt_lengths, double_batch=True)
            outputs = rnnt_joint_fused_vocab_with_consistency(
                **inputs.kernel_kwargs(inputs.leaves()), backward_mode=backward_mode, **FULL_PRECISION
            )
            sum(output.sum() for output in outputs).backward()
            if num_compiled_kernels is None:
                num_compiled_kernels = count_compiled_kernels(consistency_triton_module)
            assert count_compiled_kernels(consistency_triton_module) == num_compiled_kernels


@pytest.mark.gpu
@requires_triton_cuda
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("pruning_range_size", [None, PRUNING_RANGE_SIZE])
def test_joint_forward_for_loss_matches_materialized_logits_path(symmetrical, pruning_range_size):
    """`RNNTJoint.forward_for_loss` with `fused_vocab` matches the Joint without turbo kernel (logits path)"""
    torch.manual_seed(5)
    device = torch.device("cuda")
    joint_config = dict(
        jointnet={
            "encoder_hidden": 12,
            "pred_hidden": 10,
            "joint_hidden": HIDDEN_DIM,
            "activation": "relu",
            "dropout": 0.0,
        },
        num_classes=BLANK_ID,  # blank is the last label
        log_softmax=False,
    )
    reference_joint = RNNTJoint(**joint_config).to(device).train()
    turbo_joint = RNNTJoint(
        **joint_config,
        turbo_kernel={
            "type": "fused_vocab",
            "backend": "triton",
            **FULL_PRECISION,
            "pruning": None if pruning_range_size is None else {"type": "backward", "range_size": pruning_range_size},
        },
    ).to(device)
    turbo_joint.load_state_dict(reference_joint.state_dict())
    turbo_joint.train()

    half_batch_size, num_frames, num_tokens = 2, 7, 4
    encoder_outputs = torch.randn(half_batch_size, 12, num_frames, device=device).repeat(2, 1, 1)
    encoder_outputs = encoder_outputs + 0.1 * torch.randn_like(encoder_outputs)
    decoder_outputs = torch.randn(half_batch_size, 10, num_tokens + 1, device=device).repeat(2, 1, 1)
    transcripts = torch.randint(0, BLANK_ID, (half_batch_size, num_tokens), device=device).repeat(2, 1)
    encoder_lengths = torch.tensor([num_frames, 5], device=device).repeat(2)
    transcript_lengths = torch.tensor([num_tokens, 2], device=device).repeat(2)

    results = []
    for joint in [reference_joint, turbo_joint]:
        encoder_input = encoder_outputs.clone().requires_grad_()
        decoder_input = decoder_outputs.clone().requires_grad_()
        loss_inputs = joint.forward_for_loss(
            encoder_outputs=encoder_input,
            decoder_outputs=decoder_input,
            encoder_lengths=encoder_lengths,
            transcripts=transcripts,
            transcript_lengths=transcript_lengths,
            compute_consistency=True,
            consistency_symmetrical=symmetrical,
        )
        (
            loss_inputs.target_logprobs.sum() + loss_inputs.blank_logprobs.sum() + loss_inputs.consistency_kl.sum()
        ).backward()
        results.append(
            (list(loss_inputs[:3]), [encoder_input.grad, decoder_input.grad] + [p.grad for p in joint.parameters()])
        )
    for actual, expected in zip(results[1][0], results[0][0]):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    if pruning_range_size is None:
        for actual, expected in zip(results[1][1], results[0][1]):
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@requires_triton
def test_rejects_odd_batch():
    inputs = make_joint_inputs(src_lengths=(3,), tgt_lengths=(2,), device="cpu")
    with pytest.raises(ValueError, match="double batch"):
        rnnt_joint_fused_vocab_with_consistency(**inputs.kernel_kwargs(inputs.leaves()))


def test_cutile_backend_is_not_implemented():
    tensor_args = dict.fromkeys(["joint_hidden", "targets", "tgt_lengths", "src_lengths", "weight", "bias"])
    with pytest.raises(NotImplementedError, match="cuTile"):
        rnnt_joint_fused_vocab_with_consistency(**tensor_args, blank_id=0, backend="cutile")


def _use_min_chunk_size(monkeypatch):
    """Chunked backward with the minimal chunk (256 pairs of rows): several chunks and a partial last chunk"""
    monkeypatch.setattr(consistency_triton_module, "CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES", 1)
