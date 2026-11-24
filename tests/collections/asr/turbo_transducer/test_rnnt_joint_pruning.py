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
Tests for the pruning of the `fused_vocab` Joint (with and without the consistency loss).

- Backward pruning: backward is computed only for a window of target positions per frame, forward is exact.
  The reference is the dense backward with the upstream gradients masked outside the windows.
- Pruned lattice: the joint hidden is given only for the pruning ranges, forward and backward are computed only
  for them. The reference is the dense Joint with the outputs and the upstream gradients masked outside the ranges.

The kernels are compiled with a single small-tile config and share dimensions and precision settings with the other
tests of the fused Joint, see `conftest.py` and `utils.py`; the pruning range is a compile-time constant of
the kernels, so the tests use `PRUNING_RANGE_SIZE` unless the range itself is tested.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency import ConsistencyRNNTLoss
from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab import (
    rnnt_joint_fused_vocab,
    rnnt_joint_fused_vocab_with_consistency,
)
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import rnnt_logprobs_torch
from nemo.collections.asr.parts.turbo_transducer.rnnt_pruning import rnnt_pruning_range_starts
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.decoding.utils import avoid_sync_operations
from tests.collections.asr.turbo_transducer.utils import (
    BLANK_ID,
    DIMS_DEFAULT,
    FULL_PRECISION,
    FULL_PRECISION_TOLERANCE,
    HIDDEN_DIM,
    LATTICES_WITH_DIFFERENT_SPECIALIZATION,
    PRUNING_RANGE_SIZE,
    SRC_LENGTHS,
    TGT_LENGTHS,
    VOCAB_SIZE,
    JointInputs,
    assert_close_to_reference,
    assert_pruned_outputs_close_to_reference,
    count_compiled_kernels,
    gather_pruning_ranges,
    make_joint_inputs,
    random_grads_like,
    random_pruning_range_starts,
    reference_outputs,
    reference_pruned_lattice_outputs,
    reference_pruning_windows,
    skip_if_bf16_atomics_unsupported,
)

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer import (
        rnnt_joint_fused_vocab_consistency_triton as consistency_triton_module,
    )
    from nemo.collections.asr.parts.turbo_transducer import rnnt_joint_fused_vocab_triton as fused_vocab_triton_module
    from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab_triton import FusedVocabBackwardMode
    from nemo.collections.asr.parts.turbo_transducer.rnnt_loss_triton import rnnt_loss_from_logprobs_triton

pytestmark = pytest.mark.unit

requires_triton = pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
requires_triton_cuda = pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()), reason="Triton and CUDA are required"
)

OUTPUT_NAMES = ["target_logprobs", "blank_logprobs", "consistency_kl"]
GRAD_NAMES = ["joint_hidden grad", "weight grad", "bias grad"]
# Symmetry of the consistency loss; None - Joint without the consistency loss
CONSISTENCY_MODES = [
    pytest.param(None, id="rnnt"),
    pytest.param(True, id="consistency"),
]


@pytest.mark.gpu
@requires_triton_cuda
@pytest.mark.parametrize("backward_mode", ["atomic", "chunked"])
class TestBackwardPruning:
    @pytest.mark.parametrize(
        "consistency_symmetrical,range_size",
        [
            pytest.param(None, 1, id="rnnt-range=1"),
            pytest.param(None, PRUNING_RANGE_SIZE, id="rnnt-range=3"),
            pytest.param(None, max(TGT_LENGTHS) + 1, id="rnnt-range=whole-lattice"),  # dense backward
            pytest.param(None, max(TGT_LENGTHS) + 5, id="rnnt-range>lattice"),  # dense backward
            pytest.param(False, 1, id="consistency-range=1"),
            pytest.param(True, PRUNING_RANGE_SIZE, id="consistency-symmetrical-range=3"),
        ],
    )
    def test_full_precision_matches_masked_dense_reference(
        self, backward_mode, consistency_symmetrical, range_size, monkeypatch
    ):
        """Exact FP32 computations: exact outputs, gradients match the FP64 dense reference restricted to windows"""
        _use_min_chunk_size(monkeypatch)
        inputs = make_joint_inputs(double_batch=consistency_symmetrical is not None)
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        outputs = _fused_joint(
            inputs,
            leaves,
            consistency_symmetrical,
            backward_mode=backward_mode,
            backward_pruning_range_size=range_size,
            **FULL_PRECISION,
        )
        expected_outputs = reference_outputs(inputs, reference_leaves, consistency_symmetrical=consistency_symmetrical)
        assert_close_to_reference(outputs, expected_outputs, FULL_PRECISION_TOLERANCE, OUTPUT_NAMES[: len(outputs)])

        grads = inputs.random_grads(num_outputs=len(outputs))
        masked_grads, mask = _mask_grads_outside_windows(inputs, grads, range_size)
        torch.autograd.backward(outputs, grads)
        torch.autograd.backward(expected_outputs, masked_grads)
        assert_close_to_reference(
            [leaf.grad for leaf in leaves],
            [leaf.grad for leaf in reference_leaves],
            FULL_PRECISION_TOLERANCE,
            GRAD_NAMES,
        )
        assert torch.count_nonzero(leaves[0].grad[~mask]) == 0

    # BF16 joint hidden with FP32 master weights; for the tolerances see `MIXED_PRECISION_SETTINGS`
    # in `test_rnnt_joint_fused_vocab.py`: [outputs, joint hidden grad, weight grad, bias grad]
    @pytest.mark.parametrize(
        "consistency_symmetrical,kernel_kwargs,weight_matmul_dtype,tolerances",
        [
            pytest.param(None, {}, torch.bfloat16, [1e-5, 3e-2, 1e-2, 1e-5], id="rnnt"),
            # FP32 matmul operands: BF16 rows of the joint hidden are gathered to FP32 in the chunked backward
            pytest.param(
                None,
                dict(mm_full_precision=True, input_grad_full_precision=True, weight_grad_full_precision=True),
                torch.float32,
                [2e-3, 3e-2, 2e-3, 2e-3],
                id="rnnt-mm-full-precision",
            ),
            pytest.param(True, {}, torch.bfloat16, [1e-5, 3e-2, 1e-2, 1e-5], id="consistency"),
        ],
    )
    def test_mixed_precision_matches_masked_dense_reference(
        self, backward_mode, consistency_symmetrical, kernel_kwargs, weight_matmul_dtype, tolerances, monkeypatch
    ):
        skip_if_bf16_atomics_unsupported(backward_mode, kernel_kwargs)
        _use_min_chunk_size(monkeypatch)
        inputs = make_joint_inputs(hidden_dtype=torch.bfloat16, double_batch=consistency_symmetrical is not None)
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = _fused_joint(
                inputs,
                leaves,
                consistency_symmetrical,
                backward_mode=backward_mode,
                backward_pruning_range_size=PRUNING_RANGE_SIZE,
                **kernel_kwargs,
            )
        expected_outputs = reference_outputs(
            inputs,
            reference_leaves,
            weight_matmul_dtype=weight_matmul_dtype,
            consistency_symmetrical=consistency_symmetrical,
        )
        assert_close_to_reference(outputs, expected_outputs, tolerances[0], OUTPUT_NAMES[: len(outputs)])

        grads = inputs.random_grads(num_outputs=len(outputs))
        masked_grads, mask = _mask_grads_outside_windows(inputs, grads, PRUNING_RANGE_SIZE)
        torch.autograd.backward(outputs, grads)
        torch.autograd.backward(expected_outputs, masked_grads)
        assert_close_to_reference(
            [leaf.grad for leaf in leaves], [leaf.grad for leaf in reference_leaves], tolerances[1:], GRAD_NAMES
        )
        assert [leaf.grad.dtype for leaf in leaves] == [torch.bfloat16, torch.float32, torch.float32]
        assert torch.count_nonzero(leaves[0].grad[~mask]) == 0

    def test_window_boundary_still_emits_target(self, backward_mode):
        """
        Upstream gradient only for the targets inside one window per frame: the target emitted at the last position
        of the window leaves the window, but its gradient is retained - pruned backward is equal to the dense one.
        """
        inputs = make_joint_inputs()
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        grad_target = torch.zeros(inputs.lattice_shape, device="cuda")
        grad_target[:, :, 2 : 2 + PRUNING_RANGE_SIZE] = torch.tensor([1.0, 3.0, 2.0], device="cuda")
        grad_blank = torch.zeros_like(grad_target)
        outputs = rnnt_joint_fused_vocab(
            **inputs.kernel_kwargs(leaves),
            backward_mode=backward_mode,
            backward_pruning_range_size=PRUNING_RANGE_SIZE,
            **FULL_PRECISION,
        )
        torch.autograd.backward(outputs, [grad_target, grad_blank])
        torch.autograd.backward(
            reference_outputs(inputs, reference_leaves), [grad_target.double(), grad_blank.double()]
        )
        assert_close_to_reference(
            [leaf.grad for leaf in leaves],
            [leaf.grad for leaf in reference_leaves],
            FULL_PRECISION_TOLERANCE,
            GRAD_NAMES,
        )
        # the utterance with the longest target: gradient exactly for the window [2, 5)
        nonzero_positions = leaves[0].grad[0].abs().sum(dim=(0, 2)).nonzero().flatten().tolist()
        assert nonzero_positions == [2, 3, 4]

    @pytest.mark.parametrize("fastemit_lambda", [0.0, 0.1])
    def test_full_loss_and_upstream_gradients(self, backward_mode, fastemit_lambda):
        """Pruned Joint with the RNN-T loss: exact loss, windows are selected by the gradients of the loss"""
        torch.manual_seed(123)
        src_lengths = torch.tensor([6, 1, 3], device="cuda")
        tgt_lengths = torch.tensor([4, 2, 0], device="cuda")
        targets = torch.randint(0, BLANK_ID, (3, 4), device="cuda")
        # encoder output, predictor output, weight, bias
        shapes = [(3, 6, HIDDEN_DIM), (3, 5, HIDDEN_DIM), (VOCAB_SIZE, HIDDEN_DIM), (VOCAB_SIZE,)]
        values = [0.1 * torch.randn(*shape, device="cuda") for shape in shapes]
        reference_leaves = [value.detach().clone().requires_grad_() for value in values]
        leaves = [value.detach().clone().requires_grad_() for value in values]
        losses = []
        for (encoder_output, predictor_output, weight, bias), pruned in [(reference_leaves, False), (leaves, True)]:
            torch.manual_seed(9)  # identical dropout realization for the two forwards
            joint_hidden = F.dropout(
                F.relu(encoder_output[:, :, None] + predictor_output[:, None]), p=0.2, training=True
            )
            if pruned:
                target_logprobs, blank_logprobs = rnnt_joint_fused_vocab(
                    joint_hidden,
                    targets,
                    tgt_lengths,
                    src_lengths,
                    weight,
                    bias,
                    BLANK_ID,
                    backward_mode=backward_mode,
                    backward_pruning_range_size=PRUNING_RANGE_SIZE,
                    **FULL_PRECISION,
                )
            else:
                target_logprobs, blank_logprobs = rnnt_logprobs_torch(
                    F.linear(joint_hidden, weight, bias), targets, BLANK_ID, src_lengths, tgt_lengths
                )
            loss = rnnt_loss_from_logprobs_triton(
                target_logprobs, blank_logprobs, src_lengths, tgt_lengths, fastemit_lambda
            ).sum()
            losses.append(loss.detach())
            if pruned:
                loss.backward()
            else:
                grad_target, grad_blank = torch.autograd.grad(
                    loss, (target_logprobs, blank_logprobs), retain_graph=True
                )
                _, mask = reference_pruning_windows(
                    grad_target, grad_blank, src_lengths, tgt_lengths, PRUNING_RANGE_SIZE
                )
                torch.autograd.backward((target_logprobs, blank_logprobs), (grad_target * mask, grad_blank * mask))
        torch.testing.assert_close(losses[0], losses[1])
        for leaf, reference_leaf in zip(leaves, reference_leaves):
            torch.testing.assert_close(leaf.grad, reference_leaf.grad, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("consistency_symmetrical", CONSISTENCY_MODES)
    def test_no_cuda_sync_operations(self, backward_mode, consistency_symmetrical):
        # BF16 joint hidden with FP32 master weights and default kernel arguments, as in training with autocast
        skip_if_bf16_atomics_unsupported(backward_mode)
        inputs = make_joint_inputs(hidden_dtype=torch.bfloat16, double_batch=consistency_symmetrical is not None)

        def run_forward_backward():
            outputs = _fused_joint(
                inputs,
                inputs.leaves(),
                consistency_symmetrical,
                backward_mode=backward_mode,
                backward_pruning_range_size=PRUNING_RANGE_SIZE,
            )
            sum(output.sum() for output in outputs).backward()

        run_forward_backward()  # warmup: compile the kernels
        torch.cuda.synchronize()
        with avoid_sync_operations(torch.device("cuda")):
            run_forward_backward()

    @pytest.mark.parametrize("consistency_symmetrical", CONSISTENCY_MODES)
    def test_no_recompilation_for_new_lattice_shape(self, backward_mode, consistency_symmetrical, monkeypatch):
        """Regression: kernels were specialized (recompiled) by the lattice sizes equal to 1 or divisible by 16"""
        _use_min_chunk_size(monkeypatch)
        modules = (fused_vocab_triton_module, consistency_triton_module)
        num_compiled_kernels = None
        for src_lengths, tgt_lengths in LATTICES_WITH_DIFFERENT_SPECIALIZATION:
            inputs = make_joint_inputs(
                src_lengths=src_lengths, tgt_lengths=tgt_lengths, double_batch=consistency_symmetrical is not None
            )
            outputs = _fused_joint(
                inputs,
                inputs.leaves(),
                consistency_symmetrical,
                backward_mode=backward_mode,
                backward_pruning_range_size=PRUNING_RANGE_SIZE,
                **FULL_PRECISION,
            )
            sum(output.sum() for output in outputs).backward()
            if num_compiled_kernels is None:
                num_compiled_kernels = count_compiled_kernels(*modules)
            assert count_compiled_kernels(*modules) == num_compiled_kernels


@pytest.mark.gpu
@requires_triton_cuda
@pytest.mark.parametrize("backward_mode", ["atomic", "chunked"])
class TestPrunedLattice:
    """Joint evaluated only inside the given pruning ranges: joint hidden [B, T, range_size, hidden_dim]"""

    @pytest.mark.parametrize(
        "consistency_symmetrical",
        [pytest.param(None, id="rnnt"), pytest.param(True, id="consistency-symmetrical")],
    )
    def test_full_precision_forward_matches_masked_dense_reference(self, backward_mode, consistency_symmetrical):
        """Exact FP32 computations: outputs match the FP64 dense reference restricted to the ranges"""
        inputs = make_joint_inputs(double_batch=consistency_symmetrical is not None)
        range_starts = random_pruning_range_starts(
            inputs, PRUNING_RANGE_SIZE, paired=consistency_symmetrical is not None
        )
        with torch.no_grad():
            outputs = _fused_joint(
                inputs,
                inputs.leaves(),
                consistency_symmetrical,
                pruning_range_starts=range_starts,
                backward_mode=backward_mode,
                **FULL_PRECISION,
            )
            expected_outputs, _ = reference_pruned_lattice_outputs(
                inputs,
                inputs.leaves(torch.float64),
                range_starts,
                PRUNING_RANGE_SIZE,
                consistency_symmetrical=consistency_symmetrical,
            )
        assert_pruned_outputs_close_to_reference(
            outputs, expected_outputs, FULL_PRECISION_TOLERANCE, OUTPUT_NAMES[: len(outputs)]
        )

    @pytest.mark.parametrize(
        "consistency_symmetrical",
        [pytest.param(None, id="rnnt"), pytest.param(True, id="consistency-symmetrical")],
    )
    def test_full_precision_matches_masked_dense_reference(self, backward_mode, consistency_symmetrical, monkeypatch):
        """Exact FP32 computations: outputs and gradients match the FP64 dense reference restricted to the ranges"""
        _use_min_chunk_size(monkeypatch)
        inputs = make_joint_inputs(double_batch=consistency_symmetrical is not None)
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        range_starts = random_pruning_range_starts(
            inputs, PRUNING_RANGE_SIZE, paired=consistency_symmetrical is not None
        )
        outputs = _fused_joint(
            inputs,
            leaves,
            consistency_symmetrical,
            pruning_range_starts=range_starts,
            backward_mode=backward_mode,
            **FULL_PRECISION,
        )
        expected_outputs, range_mask = reference_pruned_lattice_outputs(
            inputs, reference_leaves, range_starts, PRUNING_RANGE_SIZE, consistency_symmetrical=consistency_symmetrical
        )
        assert_pruned_outputs_close_to_reference(
            outputs, expected_outputs, FULL_PRECISION_TOLERANCE, OUTPUT_NAMES[: len(outputs)]
        )

        grads = random_grads_like(outputs)
        torch.autograd.backward(outputs, grads)
        torch.autograd.backward(expected_outputs, [grad.double() for grad in grads])
        assert_close_to_reference(
            [leaf.grad for leaf in leaves],
            [leaf.grad for leaf in reference_leaves],
            FULL_PRECISION_TOLERANCE,
            GRAD_NAMES,
        )
        assert torch.count_nonzero(leaves[0].grad[~range_mask]) == 0

    # BF16 joint hidden with FP32 master weights, see `TestBackwardPruning`. The precision of the kernels does not
    # depend on the consistency loss and is tested for it with the dense and the gathered rows of the joint hidden
    def test_mixed_precision_matches_masked_dense_reference(self, backward_mode, monkeypatch):
        skip_if_bf16_atomics_unsupported(backward_mode)
        _use_min_chunk_size(monkeypatch)
        consistency_symmetrical = None
        inputs = make_joint_inputs(hidden_dtype=torch.bfloat16)
        leaves, reference_leaves = inputs.leaves(), inputs.leaves(torch.float64)
        range_starts = random_pruning_range_starts(inputs, PRUNING_RANGE_SIZE, paired=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = _fused_joint(
                inputs, leaves, consistency_symmetrical, pruning_range_starts=range_starts, backward_mode=backward_mode
            )
        expected_outputs, range_mask = reference_pruned_lattice_outputs(
            inputs, reference_leaves, range_starts, PRUNING_RANGE_SIZE, weight_matmul_dtype=torch.bfloat16
        )
        assert_pruned_outputs_close_to_reference(outputs, expected_outputs, 1e-5, OUTPUT_NAMES[: len(outputs)])

        grads = random_grads_like(outputs)
        torch.autograd.backward(outputs, grads)
        torch.autograd.backward(expected_outputs, [grad.double() for grad in grads])
        assert_close_to_reference(
            [leaf.grad for leaf in leaves], [leaf.grad for leaf in reference_leaves], [3e-2, 1e-2, 1e-5], GRAD_NAMES
        )
        assert [leaf.grad.dtype for leaf in leaves] == [torch.bfloat16, torch.float32, torch.float32]
        assert torch.count_nonzero(leaves[0].grad[~range_mask]) == 0

    @pytest.mark.parametrize(
        "consistency_symmetrical,hidden_dtype,kernel_kwargs",
        [
            # BF16 joint hidden with FP32 master weights and default kernel arguments, as in training with autocast
            pytest.param(None, torch.bfloat16, {}, id="rnnt"),
            # synchronization does not depend on the precision: the kernels compiled for the other tests
            pytest.param(True, torch.float32, FULL_PRECISION, id="consistency"),
        ],
    )
    def test_no_cuda_sync_operations(self, backward_mode, consistency_symmetrical, hidden_dtype, kernel_kwargs):
        if hidden_dtype == torch.bfloat16:
            skip_if_bf16_atomics_unsupported(backward_mode, kernel_kwargs)
        inputs = make_joint_inputs(hidden_dtype=hidden_dtype, double_batch=consistency_symmetrical is not None)
        range_starts = random_pruning_range_starts(
            inputs, PRUNING_RANGE_SIZE, paired=consistency_symmetrical is not None
        )

        def run_forward_backward():
            outputs = _fused_joint(
                inputs,
                inputs.leaves(),
                consistency_symmetrical,
                pruning_range_starts=range_starts,
                backward_mode=backward_mode,
                **kernel_kwargs,
            )
            torch.autograd.backward(outputs, [torch.ones_like(output) for output in outputs])

        run_forward_backward()  # warmup: compile the kernels
        torch.cuda.synchronize()
        with avoid_sync_operations(torch.device("cuda")):
            run_forward_backward()

    @pytest.mark.parametrize("consistency_symmetrical", CONSISTENCY_MODES)
    def test_no_recompilation_for_new_lattice_shape(self, backward_mode, consistency_symmetrical, monkeypatch):
        _use_min_chunk_size(monkeypatch)
        modules = (fused_vocab_triton_module, consistency_triton_module)
        num_compiled_kernels = None
        for src_lengths, tgt_lengths in LATTICES_WITH_DIFFERENT_SPECIALIZATION:
            inputs = make_joint_inputs(
                src_lengths=src_lengths, tgt_lengths=tgt_lengths, double_batch=consistency_symmetrical is not None
            )
            outputs = _fused_joint(
                inputs,
                inputs.leaves(),
                consistency_symmetrical,
                pruning_range_starts=random_pruning_range_starts(
                    inputs, PRUNING_RANGE_SIZE, paired=consistency_symmetrical is not None
                ),
                backward_mode=backward_mode,
                **FULL_PRECISION,
            )
            torch.autograd.backward(outputs, [torch.ones_like(output) for output in outputs])
            if num_compiled_kernels is None:
                num_compiled_kernels = count_compiled_kernels(*modules)
            assert count_compiled_kernels(*modules) == num_compiled_kernels


@pytest.mark.gpu
@requires_triton_cuda
class TestJointTwoPassPruning:
    """
    `RNNTJoint` with two-pass pruning. The Joint has the dimensions and the precision settings of the kernel tests
    (and the lattice with U+1 = 5 has the block size of the range selection of the other tests): no new kernels.
    """

    @staticmethod
    def _make_joint(
        pruning: dict | None, dropout: float = 0.0, masking_prob: float = -1.0, full_precision: bool = True
    ) -> RNNTJoint:
        joint = RNNTJoint(
            jointnet={
                "encoder_hidden": 12,
                "pred_hidden": 10,
                "joint_hidden": HIDDEN_DIM,
                "activation": "relu",
                "dropout": dropout,
            },
            num_classes=BLANK_ID,  # blank is the last label
            masking_prob=masking_prob,
            turbo_kernel={
                "type": "fused_vocab",
                "backend": "triton",
                **(FULL_PRECISION if full_precision else {}),
                "pruning": pruning,
            },
        )
        return joint.to("cuda").train()

    @staticmethod
    def _make_inputs(double_batch: bool):
        """
        Encoder output [B, D, T], prediction network output [B, D, U+1], lengths and targets.
        All positions of the second utterance (U+1 = 3) are inside the pruning range
        """
        generator = torch.Generator(device="cuda").manual_seed(5)
        num_frames, num_tokens = 7, 4
        encoder_outputs = torch.randn(2, 12, num_frames, device="cuda", generator=generator)
        decoder_outputs = torch.randn(2, 10, num_tokens + 1, device="cuda", generator=generator)
        targets = torch.randint(0, BLANK_ID, (2, num_tokens), device="cuda", generator=generator)
        encoder_lengths = torch.tensor([num_frames, 5], device="cuda")
        target_lengths = torch.tensor([num_tokens, 2], device="cuda")
        if double_batch:
            # the student half: perturbed encoder output, the same targets and lengths
            student_encoder_outputs = encoder_outputs + 0.1 * torch.randn(
                encoder_outputs.shape, device="cuda", generator=generator
            )
            encoder_outputs = torch.cat([encoder_outputs, student_encoder_outputs])
            decoder_outputs, targets = decoder_outputs.repeat(2, 1, 1), targets.repeat(2, 1)
            encoder_lengths, target_lengths = encoder_lengths.repeat(2), target_lengths.repeat(2)
        return encoder_outputs, decoder_outputs, encoder_lengths, targets, target_lengths

    @pytest.mark.parametrize("compute_consistency", [False, True])
    def test_mixed_precision_with_default_parameters(self, compute_consistency: bool):
        """
        BF16 autocast with the default parameters of the kernel, as in training (both passes run under autocast):
        FP32 outputs close to the FP32 computations, finite gradients
        """
        torch.manual_seed(42)
        pruning = {"type": "two_pass", "range_size": PRUNING_RANGE_SIZE}
        reference_joint = self._make_joint(pruning)
        joint = self._make_joint(pruning, full_precision=False)
        joint.load_state_dict(reference_joint.state_dict(), strict=True)
        inputs = self._make_inputs(double_batch=compute_consistency)
        _, _, src_lengths, _, tgt_lengths = inputs
        # NB: pruning is applied only with gradients enabled
        reference_inputs = reference_joint.forward_for_loss(*inputs, compute_consistency=compute_consistency)

        encoder_outputs = inputs[0].clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_inputs = joint.forward_for_loss(encoder_outputs, *inputs[1:], compute_consistency=compute_consistency)
        outputs = [loss_inputs.target_logprobs, loss_inputs.blank_logprobs]
        if compute_consistency:
            outputs.append(loss_inputs.consistency_kl)
        assert all(output.dtype == torch.float32 for output in outputs)
        # the loss of the full lattice does not depend on the selected ranges: BF16 rounding of the activations
        torch.testing.assert_close(
            loss_inputs.full_lattice_loss, reference_inputs.full_lattice_loss, rtol=3e-2, atol=3e-2
        )
        losses = rnnt_loss_from_logprobs_triton(outputs[0], outputs[1], src_lengths, tgt_lengths)
        assert torch.isfinite(losses).all()
        # the pruned lattice contains a part of the alignments
        assert torch.all(losses >= loss_inputs.full_lattice_loss - 1e-3)
        total_loss = losses.sum() + (outputs[2].sum() if compute_consistency else 0.0)
        total_loss.backward()
        for grad in [encoder_outputs.grad, joint.joint_net[-1].weight.grad, joint.joint_net[-1].bias.grad]:
            assert grad.dtype == torch.float32
            assert torch.isfinite(grad).all()
            assert torch.count_nonzero(grad) > 0

    @pytest.mark.parametrize("compute_consistency", [False, True])
    def test_matches_pruned_lattice_reference(self, compute_consistency: bool):
        """
        The losses and their gradients are equal to the losses computed from the outputs of the full lattice,
        restricted to the pruning ranges (zero probability and no consistency loss outside the ranges).
        """
        torch.manual_seed(42)
        dense_joint = self._make_joint(pruning=None)
        pruned_joint = self._make_joint({"type": "two_pass", "range_size": PRUNING_RANGE_SIZE, "start_step": 5})
        pruned_joint.load_state_dict(dense_joint.state_dict(), strict=True)
        inputs = self._make_inputs(double_batch=compute_consistency)
        _, _, src_lengths, _, tgt_lengths = inputs
        half_batch_size = len(src_lengths) // 2

        def total_loss(loss_inputs, range_mask=None):
            """RNN-T loss + consistency loss; `range_mask` [B, T, U+1] restricts the outputs of the full lattice"""
            target_logprobs, blank_logprobs, consistency_kl = loss_inputs[:3]
            if range_mask is not None:
                target_logprobs = target_logprobs.masked_fill(~range_mask, float("-inf"))
                blank_logprobs = blank_logprobs.masked_fill(~range_mask, float("-inf"))
            loss_value = rnnt_loss_from_logprobs_triton(
                target_logprobs, blank_logprobs, src_lengths, tgt_lengths
            ).sum()
            if not compute_consistency:
                return loss_value
            if range_mask is not None:
                consistency_kl = consistency_kl * range_mask[:half_batch_size]
            # mean over the positions with the consistency loss for each utterance, then over the utterances:
            # `min(U+1, number of rows per frame)` positions for each frame
            num_positions = src_lengths[:half_batch_size] * (tgt_lengths[:half_batch_size] + 1).clamp(
                max=PRUNING_RANGE_SIZE if range_mask is not None else consistency_kl.shape[2]
            )
            return loss_value + (consistency_kl.sum(dim=(1, 2)) / num_positions).mean()

        def forward_for_loss(joint: RNNTJoint, current_step: int):
            return joint.forward_for_loss(*inputs, compute_consistency=compute_consistency, current_step=current_step)

        # reference: the full lattice; ranges are estimated from its log-probabilities
        dense_inputs = forward_for_loss(dense_joint, current_step=10)
        assert dense_inputs.full_lattice_loss is None
        range_starts, full_lattice_loss = rnnt_pruning_range_starts(
            dense_inputs.target_logprobs,
            dense_inputs.blank_logprobs,
            src_lengths,
            tgt_lengths,
            PRUNING_RANGE_SIZE,
            paired=compute_consistency,
        )
        tgt_positions = torch.arange(dense_inputs.blank_logprobs.shape[2], device="cuda")
        starts = range_starts.repeat(2 if compute_consistency else 1, 1).unsqueeze(-1)
        range_mask = (tgt_positions >= starts) & (tgt_positions < starts + PRUNING_RANGE_SIZE)
        dense_loss = total_loss(dense_inputs).detach()
        reference_loss = total_loss(dense_inputs, range_mask=range_mask)
        reference_loss.backward()

        # before `start_step`: the full lattice
        loss_inputs = forward_for_loss(pruned_joint, current_step=4)
        assert loss_inputs.full_lattice_loss is None
        assert torch.equal(total_loss(loss_inputs), dense_loss)

        loss_inputs = forward_for_loss(pruned_joint, current_step=5)
        torch.testing.assert_close(loss_inputs.full_lattice_loss, full_lattice_loss)
        assert not loss_inputs.full_lattice_loss.requires_grad
        # zero probability outside the ranges
        assert torch.isneginf(loss_inputs.blank_logprobs[~range_mask]).all()
        assert torch.isneginf(loss_inputs.target_logprobs[~range_mask]).all()
        if compute_consistency:
            # the consistency loss is computed only for the positions of the ranges
            assert loss_inputs.consistency_kl.shape == (half_batch_size, 7, PRUNING_RANGE_SIZE)
            # `ConsistencyRNNTLoss` normalizes it by the number of these positions
            consistency_loss_value = ConsistencyRNNTLoss(reduction="mean").forward_from_kl(
                loss_inputs.consistency_kl, src_lengths[:half_batch_size], tgt_lengths[:half_batch_size]
            )
            num_positions = src_lengths[:half_batch_size] * (tgt_lengths[:half_batch_size] + 1).clamp(
                max=PRUNING_RANGE_SIZE
            )
            torch.testing.assert_close(
                consistency_loss_value, (loss_inputs.consistency_kl.sum(dim=(1, 2)) / num_positions).mean()
            )
        pruned_loss = total_loss(loss_inputs)
        pruned_loss.backward()

        assert torch.isfinite(pruned_loss) and pruned_loss > dense_loss
        torch.testing.assert_close(pruned_loss, reference_loss, atol=1e-5, rtol=1e-5)
        reference_parameters = dict(dense_joint.named_parameters())
        for parameter_name, parameter in pruned_joint.named_parameters():
            torch.testing.assert_close(
                parameter.grad, reference_parameters[parameter_name].grad, atol=2e-4, rtol=2e-4, msg=parameter_name
            )

    @pytest.mark.parametrize("compute_consistency", [False, True])
    def test_target_length_beyond_predictor_output(self, compute_consistency: bool):
        """
        Regression: for a target length larger than the number of targets (invalid input) the last pruning ranges
        ended after the last target position, and the predictor output was gathered outside the tensor
        (CUDA device-side assert). The ranges are limited by the lattice; the lengths are not changed,
        so the loss of the utterance is NaN. Other utterances are not affected.
        """
        torch.manual_seed(42)
        joint = self._make_joint({"type": "two_pass", "range_size": PRUNING_RANGE_SIZE})
        encoder_outputs, decoder_outputs, src_lengths, targets, tgt_lengths = self._make_inputs(compute_consistency)
        encoder_outputs.requires_grad_()

        def forward(target_lengths: torch.Tensor):
            loss_inputs = joint.forward_for_loss(
                encoder_outputs,
                decoder_outputs,
                src_lengths,
                targets,
                target_lengths,
                compute_consistency=compute_consistency,
            )
            losses = rnnt_loss_from_logprobs_triton(
                loss_inputs.target_logprobs, loss_inputs.blank_logprobs, src_lengths, target_lengths
            )
            return loss_inputs, losses

        _, expected_losses = forward(tgt_lengths)
        # the first utterance (of each half of a double batch): the target length exceeds the number of targets
        invalid_tgt_lengths = tgt_lengths.clone()
        invalid_tgt_lengths[0::2] = targets.shape[1] + 1
        loss_inputs, losses = forward(invalid_tgt_lengths)

        assert torch.isnan(losses[0::2]).all() and torch.isnan(loss_inputs.full_lattice_loss[0::2]).all()
        torch.testing.assert_close(losses[1::2], expected_losses[1::2])
        losses[1::2].sum().backward()
        for grad in [encoder_outputs.grad, joint.joint_net[-1].weight.grad, joint.joint_net[-1].bias.grad]:
            assert torch.isfinite(grad).all()

    def test_selects_ranges_without_dropout_and_gradient(self, monkeypatch):
        """The first pass is deterministic (no dropout and masking of the predictor) and does not require gradient"""
        torch.manual_seed(42)
        joint = self._make_joint({"type": "two_pass", "range_size": PRUNING_RANGE_SIZE}, dropout=0.5, masking_prob=0.5)
        inputs = self._make_inputs(double_batch=False)
        estimation_logprobs = []

        def spy_range_starts(target_logprobs, blank_logprobs, **kwargs):
            estimation_logprobs.append((target_logprobs, blank_logprobs))
            return rnnt_pruning_range_starts(target_logprobs, blank_logprobs, **kwargs)

        monkeypatch.setattr("nemo.collections.asr.modules.rnnt.rnnt_pruning_range_starts", spy_range_starts)
        loss_inputs = joint.forward_for_loss(*inputs)
        assert loss_inputs.target_logprobs.requires_grad
        ((target_logprobs, blank_logprobs),) = estimation_logprobs
        assert not target_logprobs.requires_grad and not blank_logprobs.requires_grad

        joint.eval()  # no pruning, dropout and masking
        with torch.no_grad():
            eval_inputs = joint.forward_for_loss(*inputs)
        assert len(estimation_logprobs) == 1
        assert torch.equal(target_logprobs, eval_inputs.target_logprobs)
        assert torch.equal(blank_logprobs, eval_inputs.blank_logprobs)


@pytest.mark.gpu
@requires_triton_cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Test requires 2 CUDA devices")
@pytest.mark.parametrize("backward_mode", ["atomic", "chunked"])
@pytest.mark.parametrize("consistency_symmetrical", CONSISTENCY_MODES)
@pytest.mark.parametrize("pruning", [None, "backward", "pruned_lattice"])
def test_non_current_device(backward_mode, consistency_symmetrical, pruning, monkeypatch):
    """
    Regression: all kernels of the fused Joint (with and without pruning) were launched for the current CUDA device
    instead of the device of the inputs. The results must be the same on both devices.
    """
    _use_min_chunk_size(monkeypatch)
    current_device = torch.device("cuda", torch.cuda.current_device())
    other_device = torch.device("cuda", (current_device.index + 1) % torch.cuda.device_count())
    # NB: inputs are moved through CPU; copying tensors between CUDA devices directly can hide the issue
    cpu_inputs = make_joint_inputs(double_batch=consistency_symmetrical is not None, device="cpu")
    cpu_range_starts = random_pruning_range_starts(
        cpu_inputs, PRUNING_RANGE_SIZE, paired=consistency_symmetrical is not None
    )
    cpu_grads = None
    results = []
    for device in [current_device, other_device]:
        inputs = JointInputs(
            **{
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in vars(cpu_inputs).items()
                if name != "lattice_shape"
            }
        )
        leaves = inputs.leaves()
        kwargs = dict(backward_mode=backward_mode, **FULL_PRECISION)
        if pruning == "backward":
            kwargs["backward_pruning_range_size"] = PRUNING_RANGE_SIZE
        elif pruning == "pruned_lattice":
            kwargs["pruning_range_starts"] = cpu_range_starts.to(device)
        outputs = _fused_joint(inputs, leaves, consistency_symmetrical, **kwargs)
        assert all(output.device == device for output in outputs)
        if cpu_grads is None:
            cpu_grads = [grad.cpu() for grad in random_grads_like(outputs)]
        torch.autograd.backward(outputs, [grad.to(device) for grad in cpu_grads])
        results.append([tensor.detach().cpu() for tensor in (*outputs, *[leaf.grad for leaf in leaves])])
    num_outputs = len(results[0]) - len(GRAD_NAMES)
    assert_pruned_outputs_close_to_reference(
        results[1], results[0], FULL_PRECISION_TOLERANCE, OUTPUT_NAMES[:num_outputs] + GRAD_NAMES
    )


@requires_triton
def test_rejects_invalid_pruning_range_starts():
    inputs = make_joint_inputs(src_lengths=(4,), tgt_lengths=(5,), double_batch=True, device="cpu")
    kernel_kwargs = inputs.kernel_kwargs(inputs.leaves())
    kernel_kwargs["joint_hidden"] = kernel_kwargs["joint_hidden"][:, :, :PRUNING_RANGE_SIZE]
    starts = torch.zeros(2, 4, dtype=torch.int32)
    for joint_function, valid_starts in [
        (rnnt_joint_fused_vocab, starts),
        (rnnt_joint_fused_vocab_with_consistency, starts[:1]),  # ranges are shared by the halves of the batch
    ]:
        with pytest.raises(ValueError, match="can't be used together"):
            joint_function(**kernel_kwargs, pruning_range_starts=valid_starts, backward_pruning_range_size=2)
        for invalid_starts in [valid_starts.float(), valid_starts[:, :3], torch.zeros(3, 4, dtype=torch.int32)]:
            with pytest.raises(ValueError, match="pruning_range_starts"):
                joint_function(**kernel_kwargs, pruning_range_starts=invalid_starts)
        # more rows of the joint hidden per frame than target positions (U+1 = 6)
        wide_kwargs = dict(kernel_kwargs, joint_hidden=torch.zeros(2, 4, 7, DIMS_DEFAULT[0]))
        with pytest.raises(ValueError, match="exceeds"):
            joint_function(**wide_kwargs, pruning_range_starts=valid_starts)


@pytest.mark.gpu
@requires_triton_cuda
@pytest.mark.parametrize(
    "range_size,paired,dtype",
    [
        (1, False, torch.float32),
        (PRUNING_RANGE_SIZE, False, torch.float32),
        (PRUNING_RANGE_SIZE, False, torch.float64),
        (6, False, torch.float32),  # whole lattice
        (1, True, torch.float32),
        (PRUNING_RANGE_SIZE, True, torch.float32),
        (6, True, torch.float32),  # whole lattice
    ],
)
def test_window_selection(range_size, paired, dtype):
    """Windows maximize the importance, the first one is selected for ties; U+1=6 is not a power of 2"""
    generator = torch.Generator(device="cuda").manual_seed(range_size)
    batch_size = 6 if paired else 3
    grad_target = torch.randn(batch_size, 7, 6, device="cuda", dtype=dtype, generator=generator)
    grad_blank = torch.randn(batch_size, 7, 6, device="cuda", dtype=dtype, generator=generator)
    src_lengths = torch.tensor([7, 1, 4], device="cuda").repeat(2 if paired else 1)
    tgt_lengths = torch.tensor([5, 0, 2], device="cuda").repeat(2 if paired else 1)
    expected_starts, _ = reference_pruning_windows(
        grad_target, grad_blank, src_lengths, tgt_lengths, range_size, paired=paired
    )
    for scale in [1.0, 1024.0, -0.5]:  # selection depends only on the relative magnitudes
        starts = fused_vocab_triton_module._select_pruning_starts(
            grad_target * scale, grad_blank * scale, src_lengths, tgt_lengths, range_size, paired=paired
        )
        assert starts.shape == (3, 7)
        for batch_index, src_length in enumerate(src_lengths[:3].tolist()):
            assert torch.equal(starts[batch_index, :src_length], expected_starts[batch_index, :src_length])
    # ties (zero gradients): the first window
    starts = fused_vocab_triton_module._select_pruning_starts(
        torch.zeros_like(grad_target),
        torch.zeros_like(grad_blank),
        src_lengths,
        tgt_lengths,
        range_size,
        paired=paired,
    )
    assert torch.count_nonzero(starts) == 0


def _invalid_input_cases(inputs: JointInputs, pruned_lattice: bool, range_start: int) -> dict:
    """
    Arguments of `_fused_joint` for the tests with invalid targets or lengths. With `pruned_lattice`, all pruning
    ranges of the first and the last utterance (of each half of a double batch) start at `range_start`.
    """
    if not pruned_lattice:
        return {}
    is_double_batch = inputs.joint_hidden.shape[0] != len(SRC_LENGTHS)
    range_starts = random_pruning_range_starts(inputs, PRUNING_RANGE_SIZE, paired=is_double_batch)
    range_starts[0] = range_start
    range_starts[-1] = range_start
    return dict(pruning_range_starts=range_starts)


def _assert_equal_with_nan(actual: torch.Tensor, expected: torch.Tensor):
    """Bitwise equal tensors, NaN values at the same positions"""
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(actual.nan_to_num(nan=0.0), expected.nan_to_num(nan=0.0))


@pytest.mark.gpu
@requires_triton_cuda
@pytest.mark.parametrize("consistency_symmetrical", CONSISTENCY_MODES)
@pytest.mark.parametrize("pruned_lattice", [False, True])
def test_target_length_beyond_targets_tensor(consistency_symmetrical, pruned_lattice):
    """
    Regression: for a target length larger than the width of the targets tensor (invalid input) the labels were
    read after the end of the tensor. There is no label for the positions after the last target of the tensor:
    their log-probability is NaN (the loss also rejects the length), other utterances are not affected.
    """
    inputs = make_joint_inputs(double_batch=consistency_symmetrical is not None)
    num_utterances, num_targets = len(SRC_LENGTHS), inputs.targets.shape[1]
    last_src_length = SRC_LENGTHS[-1]
    # the ranges of the last utterance contain the last target position (without a label)
    kwargs = _invalid_input_cases(inputs, pruned_lattice, range_start=num_targets + 1 - PRUNING_RANGE_SIZE)
    with torch.no_grad():
        expected_outputs = _fused_joint(inputs, inputs.leaves(), consistency_symmetrical, **kwargs, **FULL_PRECISION)

    # the last utterance (of each half of a double batch): the target length exceeds the number of targets
    tgt_lengths = inputs.tgt_lengths.clone()
    tgt_lengths[num_utterances - 1 :: num_utterances] = num_targets + 1
    outputs_by_label = []
    for label_after_targets in [1, 5]:
        # the targets are the head of a larger buffer: reading after the end of the tensor would change the result
        buffer = torch.cat([inputs.targets.reshape(-1), torch.full_like(inputs.targets[0], label_after_targets)])
        targets = buffer[: inputs.targets.numel()].view(inputs.targets.shape)
        invalid_inputs = dataclasses.replace(inputs, targets=targets, tgt_lengths=tgt_lengths)
        with torch.no_grad():
            outputs_by_label.append(
                _fused_joint(invalid_inputs, inputs.leaves(), consistency_symmetrical, **kwargs, **FULL_PRECISION)
            )
    for outputs, other_outputs in zip(*outputs_by_label):
        _assert_equal_with_nan(outputs, other_outputs)
    target_logprobs = outputs_by_label[0][0]
    for batch_index in range(num_utterances - 1, target_logprobs.shape[0], num_utterances):
        assert torch.isnan(target_logprobs[batch_index, :last_src_length, num_targets]).all()
        # valid target positions of the utterance are not changed
        _assert_equal_with_nan(target_logprobs[batch_index, :, :3], expected_outputs[0][batch_index, :, :3])
    # other utterances are not affected
    is_valid_utterance = torch.ones(target_logprobs.shape[0], dtype=torch.bool, device="cuda")
    is_valid_utterance[num_utterances - 1 :: num_utterances] = False
    for outputs, expected in zip(outputs_by_label[0][:2], expected_outputs[:2]):
        _assert_equal_with_nan(outputs[is_valid_utterance], expected[is_valid_utterance])


@pytest.mark.gpu
@requires_triton_cuda
@pytest.mark.parametrize("consistency_symmetrical", CONSISTENCY_MODES)
@pytest.mark.parametrize("pruned_lattice", [False, True])
@pytest.mark.parametrize("invalid_label", [-1, VOCAB_SIZE])
def test_target_label_outside_vocabulary(consistency_symmetrical, pruned_lattice, invalid_label):
    """
    Regression: a target label outside the vocabulary (invalid input) had a finite log-probability, so the loss
    was finite; with the consistency loss the label was used as an address of the projection parameters.
    The log-probability of such a label is NaN (NaN loss, which can be masked), other positions are not affected.
    The labels after the target length (padding) are ignored.
    """
    inputs = make_joint_inputs(double_batch=consistency_symmetrical is not None)
    num_utterances, num_targets = len(SRC_LENGTHS), inputs.targets.shape[1]
    invalid_target_index = 2
    # the ranges of the first utterance contain the target position with the invalid label
    kwargs = _invalid_input_cases(inputs, pruned_lattice, range_start=invalid_target_index - 1)
    with torch.no_grad():
        expected_outputs = _fused_joint(inputs, inputs.leaves(), consistency_symmetrical, **kwargs, **FULL_PRECISION)

    targets = inputs.targets.clone()
    # arbitrary values of the padding
    is_padding = torch.arange(num_targets, device="cuda")[None, :] >= inputs.tgt_lengths[:, None]
    targets[is_padding] = torch.where(torch.arange(num_targets, device="cuda") % 2 == 0, -100, 10**6).expand_as(
        targets
    )[is_padding]
    assert TGT_LENGTHS[0] > invalid_target_index
    with torch.no_grad():
        outputs = _fused_joint(
            dataclasses.replace(inputs, targets=targets),
            inputs.leaves(),
            consistency_symmetrical,
            **kwargs,
            **FULL_PRECISION,
        )
    for output, expected in zip(outputs, expected_outputs):
        _assert_equal_with_nan(output, expected)  # the padding is ignored

    # the first utterance (of each half of a double batch): a label outside the vocabulary
    targets[::num_utterances, invalid_target_index] = invalid_label
    with torch.no_grad():
        outputs = _fused_joint(
            dataclasses.replace(inputs, targets=targets),
            inputs.leaves(),
            consistency_symmetrical,
            **kwargs,
            **FULL_PRECISION,
        )
    is_invalid = torch.zeros_like(outputs[0], dtype=torch.bool)
    is_invalid[::num_utterances, : SRC_LENGTHS[0], invalid_target_index] = True
    assert torch.isnan(outputs[0][is_invalid]).all()
    _assert_equal_with_nan(outputs[0][~is_invalid], expected_outputs[0][~is_invalid])
    # blank log-probabilities and the consistency loss do not depend on the labels
    for output, expected in zip(outputs[1:], expected_outputs[1:]):
        _assert_equal_with_nan(output, expected)


@pytest.mark.gpu
@requires_triton_cuda
@pytest.mark.parametrize("backward_mode", ["atomic", "chunked"])
@pytest.mark.parametrize("consistency_symmetrical", CONSISTENCY_MODES)
@pytest.mark.parametrize("pruning", [None, "backward", "pruned_lattice"])
def test_utterance_without_upstream_gradient_is_excluded(backward_mode, consistency_symmetrical, pruning, monkeypatch):
    """
    Regression: an utterance with a non-finite loss that is excluded by the loss (`mask_non_finite`: zero upstream
    gradients) made the gradients of all parameters NaN, if its logits recomputed in backward were not finite
    (NaN probabilities multiplied by the zero gradients). The gradients are the gradients for the valid utterances.
    A pair of utterances with the gradient only for the consistency loss is not excluded.
    """
    _use_min_chunk_size(monkeypatch)
    with_consistency = consistency_symmetrical is not None
    inputs = make_joint_inputs(double_batch=with_consistency)
    num_utterances = len(SRC_LENGTHS)
    # the first utterance (of each half of a double batch): finite joint hidden, the vocabulary projection overflows
    joint_hidden = inputs.joint_hidden.clone()
    joint_hidden[::num_utterances] = 3e38 * torch.sign(joint_hidden[::num_utterances])
    overflow_inputs = dataclasses.replace(inputs, joint_hidden=joint_hidden)
    kwargs = dict(backward_mode=backward_mode, **FULL_PRECISION)
    if pruning == "backward":
        kwargs["backward_pruning_range_size"] = PRUNING_RANGE_SIZE
    elif pruning == "pruned_lattice":
        kwargs["pruning_range_starts"] = random_pruning_range_starts(
            inputs, PRUNING_RANGE_SIZE, paired=with_consistency
        )

    grads_by_inputs = []
    for joint_inputs in [overflow_inputs, inputs]:
        leaves = joint_inputs.leaves()
        outputs = _fused_joint(joint_inputs, leaves, consistency_symmetrical, **kwargs)
        if joint_inputs is overflow_inputs:
            assert torch.isnan(outputs[1][0, 0]).any()  # non-finite logits
        upstream_grads = random_grads_like(outputs)
        for upstream_grad in upstream_grads:
            upstream_grad[::num_utterances] = 0.0  # the first utterance is excluded by the losses
        if with_consistency:
            # the last utterance: only the consistency loss
            upstream_grads[0][num_utterances - 1 :: num_utterances] = 0.0
            upstream_grads[1][num_utterances - 1 :: num_utterances] = 0.0
        torch.autograd.backward(outputs, upstream_grads)
        grads_by_inputs.append([leaf.grad for leaf in leaves])
    grads, expected_grads = grads_by_inputs

    assert torch.count_nonzero(grads[0][::num_utterances]) == 0
    assert_close_to_reference(grads, expected_grads, FULL_PRECISION_TOLERANCE, GRAD_NAMES)
    if pruning is None:
        reference_leaves = inputs.leaves(torch.float64)
        reference = reference_outputs(inputs, reference_leaves, consistency_symmetrical=consistency_symmetrical)
        torch.autograd.backward(reference, [upstream_grad.double() for upstream_grad in upstream_grads])
        assert torch.count_nonzero(reference_leaves[0].grad[num_utterances - 1 :: num_utterances]) > 0
        assert_close_to_reference(
            grads, [leaf.grad for leaf in reference_leaves], FULL_PRECISION_TOLERANCE, GRAD_NAMES
        )


@pytest.mark.gpu
@requires_triton_cuda
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_window_selection_with_non_finite_gradients(value):
    """
    Regression: with NaN gradients (e.g., a batch with NaN loss) the selected window was outside the valid
    target positions, and the backward kernels processed rows outside the lattice
    """
    generator = torch.Generator(device="cuda").manual_seed(0)
    grad_target = torch.randn(3, 7, 6, device="cuda", generator=generator)
    grad_blank = torch.randn(3, 7, 6, device="cuda", generator=generator)
    grad_target[0, 2, 1] = value
    grad_blank[2] = value
    src_lengths, tgt_lengths = torch.tensor([7, 1, 4], device="cuda"), torch.tensor([5, 0, 2], device="cuda")
    starts = fused_vocab_triton_module._select_pruning_starts(
        grad_target, grad_blank, src_lengths, tgt_lengths, PRUNING_RANGE_SIZE
    )
    max_starts = (tgt_lengths + 1 - PRUNING_RANGE_SIZE).clamp(min=0)
    assert torch.all(starts >= 0)
    assert torch.all(starts <= max_starts[:, None])


@pytest.mark.gpu
@requires_triton_cuda
@pytest.mark.parametrize("backward_mode", ["atomic", "chunked"])
@pytest.mark.parametrize("consistency_symmetrical", CONSISTENCY_MODES)
def test_pruning_ranges_outside_lattice_are_clamped(backward_mode, consistency_symmetrical, monkeypatch):
    """
    Starts of the pruning ranges outside the lattice: the ranges are moved inside, no access outside the tensors
    (by the kernels and by the scatter of the gradient of the consistency loss)
    """
    _use_min_chunk_size(monkeypatch)
    with_consistency = consistency_symmetrical is not None
    inputs = make_joint_inputs(double_batch=with_consistency)
    max_start = inputs.lattice_shape[2] - PRUNING_RANGE_SIZE
    # starts before (-500) and after (500) the lattice
    invalid_starts = (
        random_pruning_range_starts(inputs, PRUNING_RANGE_SIZE, paired=with_consistency).clamp(max=1) * 1000 - 500
    )
    clamped_starts = invalid_starts.clamp(0, max_start)
    results = []
    for starts in [invalid_starts, clamped_starts]:
        leaves = inputs.leaves()
        kernel_kwargs = inputs.kernel_kwargs(leaves)
        # the same rows of the joint hidden for both variants of the starts
        kernel_kwargs["joint_hidden"] = gather_pruning_ranges(leaves[0], clamped_starts, PRUNING_RANGE_SIZE)
        kwargs = dict(pruning_range_starts=starts, backward_mode=backward_mode, **FULL_PRECISION)
        if with_consistency:
            outputs = rnnt_joint_fused_vocab_with_consistency(
                **kernel_kwargs, consistency_symmetrical=consistency_symmetrical, **kwargs
            )
        else:
            outputs = rnnt_joint_fused_vocab(**kernel_kwargs, **kwargs)
        torch.autograd.backward(outputs, random_grads_like(outputs))
        results.append([*outputs, *[leaf.grad for leaf in leaves]])
    assert_pruned_outputs_close_to_reference(
        results[0], results[1], FULL_PRECISION_TOLERANCE, OUTPUT_NAMES[: len(outputs)] + GRAD_NAMES
    )


@requires_triton
@pytest.mark.parametrize(
    "range_size,expected_range_size",
    [(None, None), (0, None), (-1, None), (1, 1), (3, 3), (4, None), (10, None)],
)
def test_pruning_range_normalization(range_size, expected_range_size):
    """Pruning is disabled for non-positive ranges and for ranges covering the target dimension (U+1=4)"""
    joint_hidden = torch.empty(1, 2, 4, 8)
    backward_mode, normalized_range_size = fused_vocab_triton_module._normalize_backward_options(
        joint_hidden, "atomic", range_size
    )
    assert normalized_range_size == expected_range_size
    assert backward_mode is FusedVocabBackwardMode.ATOMIC


@requires_triton
@pytest.mark.parametrize("range_size", [True, 1.5])
def test_rejects_non_integer_pruning_range(range_size):
    inputs = make_joint_inputs(src_lengths=(2,), tgt_lengths=(2,), double_batch=True, device="cpu")
    for joint_function in (rnnt_joint_fused_vocab, rnnt_joint_fused_vocab_with_consistency):
        with pytest.raises(ValueError, match="integer"):
            joint_function(**inputs.kernel_kwargs(inputs.leaves()), backward_pruning_range_size=range_size)


def test_training_supplies_optimizer_step():
    """Scheduled pruning: `training_step` passes the optimizer step (not the batch index) to the loss computation"""
    encoded, decoder = torch.randn(2, 4, 3), torch.randn(2, 4, 4)
    src_lengths, tgt_lengths = torch.tensor([3, 2]), torch.tensor([3, 2])
    transcript = torch.tensor([[1, 2, 3], [1, 2, 0]])
    model = SimpleNamespace(
        model_guid="pruning-step-test",
        global_step=7,
        _trainer=SimpleNamespace(log_every_n_steps=100, global_step=7),
        is_interctc_enabled=lambda: False,
        forward=Mock(return_value=(encoded, src_lengths)),
        decoder=Mock(return_value=(decoder, tgt_lengths, None)),
        joint=SimpleNamespace(fuse_loss_wer=False, uses_turbo_kernel=True),
        _compute_losses=Mock(side_effect=RuntimeError("stop at loss boundary")),
    )
    with pytest.raises(RuntimeError, match="stop at loss boundary"):
        EncDecRNNTModel.training_step(
            model, (torch.randn(2, 1600), src_lengths, transcript, tgt_lengths), batch_nb=123
        )
    assert model._compute_losses.call_args.kwargs["current_step"] == 7


def _fused_joint(
    inputs: JointInputs,
    leaves: list[torch.Tensor],
    consistency_symmetrical: bool | None,
    pruning_range_starts: torch.Tensor | None = None,
    **kwargs,
) -> list[torch.Tensor]:
    """
    Outputs of the fused Joint without (`consistency_symmetrical` is None) or with the consistency loss.
    With `pruning_range_starts`, the Joint is evaluated for the pruned lattice: the rows of the pruning ranges
    are gathered from the joint hidden, so the gradient for the (dense) leaf is zero outside the ranges.
    """
    kernel_kwargs = inputs.kernel_kwargs(leaves)
    if pruning_range_starts is not None:
        kernel_kwargs["joint_hidden"] = gather_pruning_ranges(leaves[0], pruning_range_starts, PRUNING_RANGE_SIZE)
        kwargs["pruning_range_starts"] = pruning_range_starts
    if consistency_symmetrical is None:
        return list(rnnt_joint_fused_vocab(**kernel_kwargs, **kwargs))
    return list(
        rnnt_joint_fused_vocab_with_consistency(
            **kernel_kwargs, consistency_symmetrical=consistency_symmetrical, **kwargs
        )
    )


def _mask_grads_outside_windows(
    inputs: JointInputs, grads: list[torch.Tensor], range_size: int
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    FP64 upstream gradients (target, blank and optionally consistency loss) zeroed outside the reference pruning
    windows, and the mask of the windows [B, T, U+1]. With the consistency loss, windows are shared by the pairs.
    """
    paired = len(grads) == 3
    _, mask = reference_pruning_windows(
        grads[0], grads[1], inputs.src_lengths, inputs.tgt_lengths, range_size, paired=paired
    )
    masks = [mask, mask, mask[: mask.shape[0] // 2]]
    return [grad.double() * grad_mask for grad, grad_mask in zip(grads, masks)], mask


def _use_min_chunk_size(monkeypatch):
    """Chunked backward with the minimal chunk (256 rows): several chunks and a partial last chunk"""
    monkeypatch.setattr(fused_vocab_triton_module, "CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES", 1)
    monkeypatch.setattr(consistency_triton_module, "CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES", 1)
