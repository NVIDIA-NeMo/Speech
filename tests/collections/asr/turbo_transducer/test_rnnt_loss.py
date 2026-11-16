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
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_numpy import RNNTLoss as RNNTLoss_Numpy
from nemo.collections.asr.parts.turbo_transducer.rnnt_loss import TurboRnntLoss
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

# NB: tests use a small set of vocabulary sizes to reuse compiled Triton kernels (tests should be fast without cache)
VOCAB_SIZE = 5


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
