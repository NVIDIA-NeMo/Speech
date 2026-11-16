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

import numpy as np
import pytest
import torch

from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.losses.rnnt_pytorch import RNNTLossPytorch
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_numpy import RNNTLoss as RNNTLoss_Numpy
from nemo.collections.asr.parts.turbo_transducer.rnnt_loss import TurboRnntLoss
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

# NB: tests use a small set of vocabulary sizes to reuse compiled Triton kernels (tests should be fast without cache)
VOCAB_SIZE = 5
LARGE_VOCAB_SIZE = 1025  # for memory tests (shared with other Turbo Transducer tests)

requires_two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Test requires 2 CUDA devices")


def compute_loss_and_grad(loss_fn, acts, labels, src_lengths=None, tgt_lengths=None, device="cuda"):
    """Compute per-utterance loss and gradient for logits for the given loss function."""
    acts = torch.as_tensor(acts).to(device).detach().requires_grad_(True)
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    if src_lengths is None:
        src_lengths = [acts.shape[1]] * acts.shape[0]
    if tgt_lengths is None:
        tgt_lengths = [labels.shape[1]] * acts.shape[0]
    src_lengths = torch.as_tensor(src_lengths, dtype=torch.long, device=device)
    tgt_lengths = torch.as_tensor(tgt_lengths, dtype=torch.long, device=device)
    loss = loss_fn(acts, labels, src_lengths, tgt_lengths)
    loss.sum().backward()
    return loss.detach().cpu().double().numpy(), acts.grad.detach().cpu().double().numpy()


def numpy_reference(
    acts: np.ndarray, labels: np.ndarray, src_lengths, tgt_lengths, fastemit_lambda: float = 0.0, blank: int = 0
):
    """Reference loss and gradient computed per utterance (numpy implementation does not support padding)."""
    losses = np.zeros([acts.shape[0]])
    grads = np.zeros_like(acts)
    for i, (src_length, tgt_length) in enumerate(zip(src_lengths, tgt_lengths)):
        loss, grad = compute_loss_and_grad(
            RNNTLoss_Numpy(blank=blank, fastemit_lambda=fastemit_lambda),
            np.ascontiguousarray(acts[i : i + 1, :src_length, : tgt_length + 1]),
            np.ascontiguousarray(labels[i : i + 1, :tgt_length]),
            device="cpu",
        )
        losses[i] = loss.sum()
        grads[i, :src_length, : tgt_length + 1] = grad[0]
    return losses, grads


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is unavailable")
class TestTurboRnntLoss:
    def test_case_small(self):
        acts = np.array(
            [
                [
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.2, 0.8, 0.1]],
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]],
                ]
            ],
            dtype=np.float32,
        )
        expected_grad = np.array(
            [
                [
                    [
                        [-0.13116688, -0.3999269, 0.17703125, 0.17703125, 0.17703125],
                        [-0.18572757, 0.12247056, -0.18168412, 0.12247056, 0.12247056],
                        [-0.32091254, 0.06269141, 0.06928472, 0.12624499, 0.06269141],
                    ],
                    [
                        [0.05456069, -0.21824276, 0.05456069, 0.05456069, 0.05456069],
                        [0.12073959, 0.12073959, -0.48295835, 0.12073959, 0.12073959],
                        [-0.6925882, 0.16871116, 0.18645467, 0.16871116, 0.16871116],
                    ],
                ]
            ]
        )
        loss, grad = compute_loss_and_grad(TurboRnntLoss(blank=0), acts, [[1, 2]])
        assert np.allclose(loss, 4.495666, rtol=1e-6)
        assert np.allclose(grad, expected_grad, atol=1e-6)

    def test_case_small_blank_last(self):
        acts = np.array(
            [
                [
                    [[0.0, 1.0, 3.0], [0.0, 2.0, 3.0], [1.0, 1.0, 3.0], [2.0, 3.0, 2.0]],
                    [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [2.0, 2.0, 0.0]],
                    [[0.0, 2.0, 5.0], [0.0, 3.0, 5.0], [1.0, 2.0, 5.0], [2.0, 4.0, 4.0]],
                    [[0.0, 3.0, 4.0], [0.0, 4.0, 4.0], [1.0, 3.0, 4.0], [2.0, 5.0, 3.0]],
                    [[2.0, 2.0, 1.0], [2.0, 3.0, 1.0], [3.0, 2.0, 1.0], [4.0, 4.0, 0.0]],
                ]
            ],
            dtype=np.float32,
        )
        labels = [[0, 1, 0]]
        loss, grad = compute_loss_and_grad(TurboRnntLoss(blank=acts.shape[-1] - 1), acts, labels)
        expected_loss, expected_grad = numpy_reference(acts, np.array(labels), [5], [3], blank=acts.shape[-1] - 1)
        assert np.allclose(loss, 6.789285182952881, rtol=1e-6)
        assert np.allclose(loss, expected_loss, rtol=1e-6)
        assert np.allclose(grad, expected_grad, atol=1e-6)

    @pytest.mark.parametrize("fastemit_lambda", [0.0, 0.1])
    @pytest.mark.parametrize(
        "src_lengths,tgt_lengths",
        [
            ([1], [1]),  # single frame
            ([4], [2]),
            ([10], [2]),  # T > U + 1
            ([1], [13]),  # T < U + 1
            ([3, 5, 8, 4], [2, 4, 5, 0]),  # variable lengths, empty target
        ],
    )
    def test_matches_numpy_reference(self, src_lengths, tgt_lengths, fastemit_lambda: float):
        rng = np.random.RandomState(0)
        batch_size, src_length_max, tgt_length_max = len(src_lengths), max(src_lengths), max(tgt_lengths)
        acts = rng.randn(batch_size, src_length_max, tgt_length_max + 1, VOCAB_SIZE)
        labels = rng.randint(low=1, high=VOCAB_SIZE, size=[batch_size, tgt_length_max])

        loss, grad = compute_loss_and_grad(
            TurboRnntLoss(blank=0, fastemit_lambda=fastemit_lambda), acts, labels, src_lengths, tgt_lengths
        )
        expected_loss, expected_grad = numpy_reference(acts, labels, src_lengths, tgt_lengths, fastemit_lambda)
        assert np.allclose(loss, expected_loss, rtol=1e-6)
        assert np.allclose(grad, expected_grad, atol=1e-6)

    @pytest.mark.parametrize("src_length,tgt_length", [(130, 64), (65, 129)])
    def test_large_lattice(self, src_length: int, tgt_length: int):
        """Diagonals with >= 128 elements: computation is distributed across warps in the kernels."""
        rng = np.random.RandomState(1)
        batch_size = 2
        acts = rng.randn(batch_size, src_length, tgt_length + 1, VOCAB_SIZE).astype(np.float32)
        labels = rng.randint(low=1, high=VOCAB_SIZE, size=[batch_size, tgt_length])
        src_lengths, tgt_lengths = [src_length, src_length - 3], [tgt_length - 5, tgt_length]

        loss, grad = compute_loss_and_grad(TurboRnntLoss(blank=0), acts, labels, src_lengths, tgt_lengths)
        expected_loss, expected_grad = numpy_reference(acts, labels, src_lengths, tgt_lengths)
        # float32 errors accumulate along the long lattice
        assert np.allclose(loss, expected_loss, rtol=1e-5)
        assert np.allclose(grad, expected_grad, atol=1e-4)

    def test_bfloat16_matches_float32(self):
        """For bfloat16 input, computations are performed in float32, gradient is returned in bfloat16."""
        torch.manual_seed(2)
        acts = torch.randn(2, 6, 4, VOCAB_SIZE, device="cuda").to(torch.bfloat16)
        labels = torch.randint(1, VOCAB_SIZE, (2, 3), device="cuda")
        loss_fn = TurboRnntLoss(blank=0)
        loss_bf16, grad_bf16 = compute_loss_and_grad(loss_fn, acts, labels, [6, 4], [3, 2])
        loss_fp32, grad_fp32 = compute_loss_and_grad(loss_fn, acts.float(), labels, [6, 4], [3, 2])
        assert np.allclose(loss_bf16, loss_fp32, rtol=1e-6)
        assert np.allclose(grad_bf16, grad_fp32, atol=1e-2)

    @pytest.mark.parametrize("extra_target_columns", [-2, 3])
    def test_targets_padding_does_not_change_result(self, extra_target_columns: int):
        """Targets can be narrower or wider than the U dimension of logits (e.g., trimmed by `RNNTLoss`)."""
        torch.manual_seed(3)
        tgt_length_max = 5
        acts = torch.randn(3, 4, tgt_length_max + 1, VOCAB_SIZE, device="cuda")
        labels = torch.randint(1, VOCAB_SIZE, (3, tgt_length_max), device="cuda")
        src_lengths, tgt_lengths = [4, 3, 4], [3, 2, 3]  # max target length < tgt_length_max
        if extra_target_columns > 0:
            padded_labels = torch.cat([labels, labels[:, :extra_target_columns]], dim=1)
        else:
            padded_labels = labels[:, :extra_target_columns].clone()
        loss_fn = TurboRnntLoss(blank=0)
        loss, grad = compute_loss_and_grad(loss_fn, acts, labels, src_lengths, tgt_lengths)
        padded_loss, padded_grad = compute_loss_and_grad(loss_fn, acts, padded_labels, src_lengths, tgt_lengths)
        assert np.array_equal(loss, padded_loss)
        assert np.array_equal(grad, padded_grad)

    def test_empty_source_infinite_loss(self):
        torch.manual_seed(4)
        acts = torch.randn(2, 4, 3, VOCAB_SIZE, device="cuda")
        labels = torch.randint(1, VOCAB_SIZE, (2, 2), device="cuda")
        loss, _ = compute_loss_and_grad(TurboRnntLoss(blank=0), acts, labels, [4, 0], [2, 0])
        assert np.isfinite(loss[0]) and loss[1] == np.inf

        loss, grad = compute_loss_and_grad(TurboRnntLoss(blank=0, mask_non_finite=True), acts, labels, [4, 0], [2, 0])
        assert np.isfinite(loss[0]) and loss[1] == 0.0
        assert np.isfinite(grad).all() and (grad[1] == 0.0).all()

    def test_mask_non_finite(self):
        torch.manual_seed(5)
        acts = torch.randn(2, 4, 3, VOCAB_SIZE, device="cuda")
        labels = torch.randint(1, VOCAB_SIZE, (2, 2), device="cuda")
        clean_loss, _ = compute_loss_and_grad(TurboRnntLoss(blank=0), acts, labels)
        acts[1, 0, 0, 0] = float("nan")

        loss, _ = compute_loss_and_grad(TurboRnntLoss(blank=0), acts, labels)
        assert not np.isfinite(loss[1])

        loss, grad = compute_loss_and_grad(TurboRnntLoss(blank=0, mask_non_finite=True), acts, labels)
        assert np.allclose(loss[0], clean_loss[0], rtol=1e-6) and loss[1] == 0.0
        assert np.isfinite(grad).all() and (grad[1] == 0.0).all()

    @pytest.mark.parametrize(
        "src_lengths,tgt_lengths",
        [
            ([4, 9, 4], [2, 2, 2]),  # source length exceeds T
            ([4, 4, 4], [2, 5, 2]),  # target length exceeds U
            ([4, 4, 4], [2, -1, 2]),  # negative target length
        ],
    )
    def test_invalid_lengths_nan_loss(self, src_lengths, tgt_lengths):
        """Lengths outside the Joint output: NaN loss and zero gradient, other utterances are not affected."""
        torch.manual_seed(8)
        acts = torch.randn(3, 4, 3, VOCAB_SIZE, device="cuda")
        labels = torch.randint(1, VOCAB_SIZE, (3, 2), device="cuda")
        loss_fn = TurboRnntLoss(blank=0)
        expected_loss, expected_grad = compute_loss_and_grad(loss_fn, acts, labels, [4, 4, 4], [2, 2, 2])
        loss, grad = compute_loss_and_grad(loss_fn, acts, labels, src_lengths, tgt_lengths)
        assert np.isnan(loss[1])
        assert (grad[1] == 0.0).all()
        assert np.array_equal(loss[[0, 2]], expected_loss[[0, 2]])
        assert np.array_equal(grad[[0, 2]], expected_grad[[0, 2]])

    def test_targets_shorter_than_lengths_nan_loss(self):
        """Targets tensor is narrower than the target length: NaN loss, other utterances are not affected."""
        torch.manual_seed(9)
        acts = torch.randn(3, 4, 4, VOCAB_SIZE, device="cuda")
        labels = torch.randint(1, VOCAB_SIZE, (3, 3), device="cuda")
        src_lengths, tgt_lengths = [4, 4, 4], [2, 3, 1]
        loss_fn = TurboRnntLoss(blank=0)
        expected_loss, expected_grad = compute_loss_and_grad(loss_fn, acts, labels, src_lengths, tgt_lengths)
        loss, grad = compute_loss_and_grad(loss_fn, acts, labels[:, :2].clone(), src_lengths, tgt_lengths)
        assert np.isnan(loss[1])
        assert np.array_equal(loss[[0, 2]], expected_loss[[0, 2]])
        assert np.array_equal(grad[[0, 2]], expected_grad[[0, 2]])

    @pytest.mark.parametrize("invalid_label", [-1, VOCAB_SIZE])
    def test_out_of_vocabulary_target_nan_loss(self, invalid_label: int):
        """Target label outside the vocabulary: NaN loss, other utterances are not affected."""
        torch.manual_seed(10)
        acts = torch.randn(2, 4, 3, VOCAB_SIZE, device="cuda")
        labels = torch.randint(1, VOCAB_SIZE, (2, 2), device="cuda")
        loss_fn = TurboRnntLoss(blank=0)
        expected_loss, expected_grad = compute_loss_and_grad(loss_fn, acts, labels)
        labels[1, 0] = invalid_label
        loss, grad = compute_loss_and_grad(loss_fn, acts, labels)
        assert np.isnan(loss[1])
        assert loss[0] == expected_loss[0]
        assert np.array_equal(grad[0], expected_grad[0])

        loss, grad = compute_loss_and_grad(TurboRnntLoss(blank=0, mask_non_finite=True), acts, labels)
        assert loss[0] == expected_loss[0] and loss[1] == 0.0
        assert np.array_equal(grad[0], expected_grad[0]) and (grad[1] == 0.0).all()

    def test_impossible_alignment_infinite_loss(self):
        """Target label with zero probability: infinite loss (can be masked), zero gradient for the utterance."""
        torch.manual_seed(11)
        acts = torch.randn(2, 4, 3, VOCAB_SIZE, device="cuda")
        labels = torch.randint(1, VOCAB_SIZE, (2, 2), device="cuda")
        expected_loss, expected_grad = compute_loss_and_grad(TurboRnntLoss(blank=0), acts, labels)
        acts[1, :, :, labels[1, 0]] = float("-inf")

        loss, grad = compute_loss_and_grad(TurboRnntLoss(blank=0), acts, labels)
        assert loss[0] == expected_loss[0] and loss[1] == np.inf
        assert np.array_equal(grad[0], expected_grad[0]) and (grad[1] == 0.0).all()

        loss, grad = compute_loss_and_grad(TurboRnntLoss(blank=0, mask_non_finite=True), acts, labels)
        assert loss[0] == expected_loss[0] and loss[1] == 0.0
        assert np.array_equal(grad[0], expected_grad[0]) and (grad[1] == 0.0).all()

    def test_impossible_alignment_from_logprobs(self):
        """Target transition with `-inf` log-probability: infinite loss, finite utterance in the batch is not affected."""
        log_half = float(np.log(0.5))
        target_logprobs = torch.tensor([[[float("-inf"), 0.0]], [[log_half, 0.0]]], device="cuda", requires_grad=True)
        blank_logprobs = torch.full([2, 1, 2], fill_value=log_half, device="cuda", requires_grad=True)
        src_lengths, tgt_lengths = torch.tensor([1, 1], device="cuda"), torch.tensor([1, 1], device="cuda")

        loss = TurboRnntLoss(blank=0).forward_from_logprobs(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)
        assert loss[0].item() == float("inf")
        assert loss[1].item() == pytest.approx(-2 * log_half, rel=1e-6)

        loss_fn = TurboRnntLoss(blank=0, mask_non_finite=True)
        loss = loss_fn.forward_from_logprobs(target_logprobs, blank_logprobs, src_lengths, tgt_lengths)
        loss.sum().backward()
        assert loss[0].item() == 0.0
        assert loss[1].item() == pytest.approx(-2 * log_half, rel=1e-6)
        assert (target_logprobs.grad[0] == 0.0).all() and (blank_logprobs.grad[0] == 0.0).all()
        assert target_logprobs.grad[1].tolist() == [[-1.0, 0.0]]
        assert blank_logprobs.grad[1].tolist() == [[0.0, -1.0]]

    def test_unreachable_lattice_node(self):
        """Both arcs to a lattice node have zero probability: the loss is finite if a valid alignment exists."""
        torch.manual_seed(12)
        acts = torch.randn(1, 4, 3, VOCAB_SIZE, device="cuda")
        labels = torch.randint(1, VOCAB_SIZE, (1, 2), device="cuda")
        # node (t=1, u=1) is unreachable
        acts[0, 0, 1, 0] = float("-inf")  # blank arc (t=0, u=1) -> (t=1, u=1)
        acts[0, 1, 0, labels[0, 0]] = float("-inf")  # emit arc (t=1, u=0) -> (t=1, u=1)
        lengths = torch.tensor([4], device="cuda"), torch.tensor([2], device="cuda")

        loss, grad = compute_loss_and_grad(TurboRnntLoss(blank=0), acts, labels)
        expected_loss = RNNTLossPytorch(blank=0, reduction=None)(acts, labels, *lengths).cpu().numpy()
        assert np.isfinite(expected_loss).all()
        assert np.allclose(loss, expected_loss, rtol=1e-6)
        assert np.isfinite(grad).all()

    @requires_two_gpus
    def test_non_current_device(self):
        """Inputs are on a CUDA device that is not the current device."""
        torch.manual_seed(13)
        # NB: inputs are created on CPU; copying tensors between CUDA devices can hide the issue
        acts = torch.randn(2, 6, 4, VOCAB_SIZE)
        labels = torch.randint(1, VOCAB_SIZE, (2, 3))
        src_lengths, tgt_lengths = [6, 4], [3, 2]
        other_device = torch.device("cuda", (torch.cuda.current_device() + 1) % torch.cuda.device_count())
        loss_fn = TurboRnntLoss(blank=0)
        expected_loss, expected_grad = compute_loss_and_grad(loss_fn, acts, labels, src_lengths, tgt_lengths)
        loss, grad = compute_loss_and_grad(loss_fn, acts, labels, src_lengths, tgt_lengths, device=other_device)
        assert np.array_equal(loss, expected_loss)
        assert np.array_equal(grad, expected_grad)

    def test_no_cuda_sync_operations(self):
        from tests.collections.asr.decoding.utils import avoid_sync_operations

        torch.manual_seed(6)
        acts = torch.randn(2, 4, 3, VOCAB_SIZE, device="cuda", requires_grad=True)
        labels = torch.randint(1, VOCAB_SIZE, (2, 2), device="cuda")
        lengths = torch.tensor([4, 3], device="cuda")
        label_lengths = torch.tensor([2, 1], device="cuda")
        loss_fn = TurboRnntLoss(blank=0)
        loss_fn(acts, labels, lengths, label_lengths).sum().backward()  # warmup
        with avoid_sync_operations(acts.device):
            loss_fn(acts, labels, lengths, label_lengths).sum().backward()

    @pytest.mark.parametrize("fastemit_lambda", [0.0, 0.1])
    def test_rnnt_loss_wrapper(self, fastemit_lambda: float):
        """`RNNTLoss(loss_name="turbo_rnnt")` resolves to Turbo-RNNT loss and passes kwargs."""
        torch.manual_seed(7)
        acts = torch.randn(3, 5, 4, VOCAB_SIZE, device="cuda")
        labels = torch.randint(0, VOCAB_SIZE - 1, (3, 3), device="cuda")
        src_lengths, tgt_lengths = [5, 2, 4], [3, 1, 0]
        wrapper = RNNTLoss(
            num_classes=VOCAB_SIZE - 1,
            reduction=None,
            loss_name="turbo_rnnt",
            loss_kwargs={"fastemit_lambda": fastemit_lambda},
        )
        assert isinstance(wrapper._loss, TurboRnntLoss)

        def wrapper_fn(acts, labels, src_lengths, tgt_lengths):
            return wrapper(log_probs=acts, targets=labels, input_lengths=src_lengths, target_lengths=tgt_lengths)

        loss, grad = compute_loss_and_grad(wrapper_fn, acts, labels, src_lengths, tgt_lengths)
        reference_fn = TurboRnntLoss(blank=VOCAB_SIZE - 1, fastemit_lambda=fastemit_lambda)
        reference_loss, reference_grad = compute_loss_and_grad(reference_fn, acts, labels, src_lengths, tgt_lengths)
        assert np.array_equal(loss, reference_loss)
        assert np.array_equal(grad, reference_grad)

    def test_rnnt_loss_wrapper_padded_joint_output(self):
        """Joint output padded beyond the maximum lengths is not copied by `RNNTLoss` (lengths are respected)."""
        torch.manual_seed(14)
        batch_size, src_length_max, tgt_length_max, pad = 4, 64, 32, 3
        acts = torch.randn(batch_size, src_length_max + pad, tgt_length_max + 1 + pad, LARGE_VOCAB_SIZE, device="cuda")
        acts.requires_grad_(True)
        labels = torch.randint(0, LARGE_VOCAB_SIZE - 1, (batch_size, tgt_length_max + pad), device="cuda")
        src_lengths = torch.tensor([src_length_max, 40, 64, 1], device="cuda")
        tgt_lengths = torch.tensor([20, tgt_length_max, 0, 7], device="cuda")
        wrapper = RNNTLoss(num_classes=LARGE_VOCAB_SIZE - 1, reduction=None, loss_name="turbo_rnnt")

        def compute_loss(acts):
            return wrapper(log_probs=acts, targets=labels, input_lengths=src_lengths, target_lengths=tgt_lengths)

        loss = compute_loss(acts)
        loss.sum().backward()
        grad, acts.grad = acts.grad, None
        trimmed_acts = acts.detach()[:, :src_length_max, : tgt_length_max + 1].clone().requires_grad_(True)
        trimmed_loss = compute_loss(trimmed_acts)
        trimmed_loss.sum().backward()
        assert torch.equal(loss, trimmed_loss)
        assert torch.equal(grad[:, :src_length_max, : tgt_length_max + 1], trimmed_acts.grad)
        assert (grad[:, src_length_max:] == 0.0).all() and (grad[:, :, tgt_length_max + 1 :] == 0.0).all()

        # no extra full-size tensors except for the gradient
        del loss, grad, trimmed_acts, trimmed_loss
        logits_size_bytes = acts.numel() * acts.element_size()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated()
        compute_loss(acts).sum().backward()
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() - memory_before
        assert peak_memory < 1.1 * logits_size_bytes

    @pytest.mark.parametrize(
        "src_lengths,tgt_lengths",
        [
            ([5, 6, 4], [3, 1, 0]),  # source length exceeds T
            ([5, 2, 4], [3, 4, 0]),  # target length exceeds U (both for Joint output and targets)
        ],
    )
    def test_rnnt_loss_wrapper_invalid_lengths(self, src_lengths, tgt_lengths):
        torch.manual_seed(15)
        acts = torch.randn(3, 5, 4, VOCAB_SIZE, device="cuda")
        labels = torch.randint(0, VOCAB_SIZE - 1, (3, 3), device="cuda")
        wrapper = RNNTLoss(num_classes=VOCAB_SIZE - 1, reduction=None, loss_name="turbo_rnnt")
        with pytest.raises(ValueError, match="exceed"):
            wrapper(
                log_probs=acts,
                targets=labels,
                input_lengths=torch.tensor(src_lengths, device="cuda"),
                target_lengths=torch.tensor(tgt_lengths, device="cuda"),
            )

    def test_rnnt_loss_wrapper_joint_output_shorter_than_targets(self):
        """Joint output with insufficient number of target positions (U+1 < max target length + 1)."""
        torch.manual_seed(16)
        acts = torch.randn(3, 5, 3, VOCAB_SIZE, device="cuda")
        labels = torch.randint(0, VOCAB_SIZE - 1, (3, 3), device="cuda")
        wrapper = RNNTLoss(num_classes=VOCAB_SIZE - 1, reduction=None, loss_name="turbo_rnnt")
        with pytest.raises(ValueError, match="exceed"):
            wrapper(
                log_probs=acts,
                targets=labels,
                input_lengths=torch.tensor([5, 2, 4], device="cuda"),
                target_lengths=torch.tensor([3, 1, 0], device="cuda"),
            )


@pytest.mark.unit
class TestTurboRnntLossInit:
    def test_clamp_not_supported(self):
        with pytest.raises(NotImplementedError):
            TurboRnntLoss(blank=0, clamp=1.0)

    @pytest.mark.parametrize("backend", ["torch", "cutile"])
    def test_unsupported_backend(self, backend: str):
        with pytest.raises(NotImplementedError):
            TurboRnntLoss(blank=0, backend=backend)

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is unavailable")
    def test_cpu_input_not_supported(self):
        with pytest.raises(NotImplementedError):
            TurboRnntLoss(blank=0)(torch.randn(1, 2, 2, 3), torch.tensor([[1]]), torch.tensor([2]), torch.tensor([1]))
