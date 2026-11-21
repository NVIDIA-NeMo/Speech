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
Tests for the backward pruning of the `fused_vocab` Joint (with and without the consistency loss): backward is
computed only for a window of target positions per frame, forward is exact.

The reference is the dense backward with the upstream gradients masked outside the windows.
The kernels are compiled with a single small-tile config and share dimensions and precision settings with the other
tests of the fused Joint, see `conftest.py` and `utils.py`; the pruning range is a compile-time constant of
the kernels, so the tests use `PRUNING_RANGE_SIZE` unless the range itself is tested.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab import (
    rnnt_joint_fused_vocab,
    rnnt_joint_fused_vocab_with_consistency,
)
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import rnnt_logprobs_torch
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.decoding.utils import avoid_sync_operations
from tests.collections.asr.turbo_transducer.utils import (
    BLANK_ID,
    FULL_PRECISION,
    FULL_PRECISION_TOLERANCE,
    HIDDEN_DIM,
    LATTICES_WITH_DIFFERENT_SPECIALIZATION,
    PRUNING_RANGE_SIZE,
    TGT_LENGTHS,
    VOCAB_SIZE,
    JointInputs,
    assert_close_to_reference,
    count_compiled_kernels,
    make_joint_inputs,
    reference_outputs,
    reference_pruning_windows,
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
    inputs: JointInputs, leaves: list[torch.Tensor], consistency_symmetrical: bool | None, **kwargs
) -> list[torch.Tensor]:
    """Outputs of the fused Joint without (`consistency_symmetrical` is None) or with the consistency loss"""
    if consistency_symmetrical is None:
        return list(rnnt_joint_fused_vocab(**inputs.kernel_kwargs(leaves), **kwargs))
    return list(
        rnnt_joint_fused_vocab_with_consistency(
            **inputs.kernel_kwargs(leaves), consistency_symmetrical=consistency_symmetrical, **kwargs
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
