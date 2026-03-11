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

import pytest
import torch

from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency import (
    ConsistencyFullRNNTLoss,
    ConsistencyGraphRNNTLoss,
    ConsistencyRNNTLoss,
    consistency_rnnt_kld,
)
from nemo.core.utils.optional_libs import K2_AVAILABLE, TRITON_AVAILABLE


def get_devices_for_testing(use_cpu_always: bool = False) -> list[torch.device]:
    devices = [torch.device("cpu")] if use_cpu_always else []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))

    if torch.mps.is_available():
        devices.append(torch.device("mps"))

    if len(devices) == 0:
        # no fast device for testing, add CPU
        devices.append(torch.device("cpu"))
    return devices


DEVICES = get_devices_for_testing(use_cpu_always=False)
DEVICES_WITH_CPU = get_devices_for_testing(use_cpu_always=True)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_single_frame_single_token(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Edge case: minimal tensor sizes (T=1, U=1)."""
    torch.manual_seed(77)
    teacher_logits = torch.randn(1, 1, 2, 3, device=device)  # [B=1, T=1, U+1=2, V=3]
    student_logits = torch.randn(1, 1, 2, 3, device=device)
    targets = torch.randint(0, 2, (1, 1), device=device)  # blank_id=2

    loss = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=2,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert loss.ndim == 0
    assert loss >= 0
    assert torch.isfinite(loss)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_identity_kl_zero(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """When teacher == student, KL should be 0."""
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 3, 5, device=device)  # [B, T, U+1, V]
    targets = torch.randint(0, 4, (2, 2), device=device)  # [B, U], exclude blank_id=4

    loss = consistency_rnnt_kld(
        teacher_logits=logits,
        student_logits=logits,
        targets=targets,
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_non_negativity(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """KL divergence should always be >= 0."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert loss >= 0


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_symmetrical_swap_invariance(device: torch.device, use_blank: bool, reduction: str):
    """With symmetrical=True, swapping teacher/student gives same loss."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss1 = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=True,
        use_blank=use_blank,
        reduction=reduction,
    )
    loss2 = consistency_rnnt_kld(
        teacher_logits=student_logits,
        student_logits=teacher_logits,
        targets=targets,
        blank_id=4,
        symmetrical=True,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert torch.isclose(loss1, loss2, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_non_symmetrical_different_on_swap(device: torch.device, use_blank: bool, reduction: str):
    """With symmetrical=False, swap gives different loss."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss1 = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=False,
        use_blank=use_blank,
        reduction=reduction,
    )
    loss2 = consistency_rnnt_kld(
        teacher_logits=student_logits,
        student_logits=teacher_logits,
        targets=targets,
        blank_id=4,
        symmetrical=False,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert not torch.isclose(loss1, loss2, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_gradient_flow(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Verify gradients flow to student logits."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device, requires_grad=True)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    loss.backward()

    assert student_logits.grad is not None
    assert not torch.all(student_logits.grad == 0)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES_WITH_CPU)
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_gradient_numerical_check(device: torch.device, use_blank: bool, reduction: str):
    """Numerical gradient verification.

    Note: Only testing symmetrical=False because symmetrical=True uses detach()
    on the student logprobs in the reverse direction, which causes gradcheck to fail
    (detach prevents gradient flow, making numerical and analytical gradients differ).
    """
    if device.type == "mps":
        pytest.skip("MPS does not support float64")
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 3, 2, 4, dtype=torch.float64, device=device)
    student_logits = torch.randn(2, 3, 2, 4, dtype=torch.float64, device=device, requires_grad=True)
    targets = torch.randint(0, 3, (2, 1), device=device)  # exclude blank_id=3

    def loss_fn(s_logits):
        return consistency_rnnt_kld(
            teacher_logits=teacher_logits,
            student_logits=s_logits,
            targets=targets,
            blank_id=3,
            symmetrical=False,
            use_blank=use_blank,
            reduction=reduction,
        )

    assert torch.autograd.gradcheck(loss_fn, (student_logits,), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_variable_lengths(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Test with different src_lengths/tgt_lengths."""
    torch.manual_seed(42)
    batch_size, T, U_plus_1, V = 3, 5, 4, 6
    teacher_logits = torch.randn(batch_size, T, U_plus_1, V, device=device)
    student_logits = torch.randn(batch_size, T, U_plus_1, V, device=device)
    targets = torch.randint(0, V - 1, (batch_size, U_plus_1 - 1), device=device)  # blank_id = V-1
    src_lengths = torch.tensor([5, 3, 4], device=device)
    tgt_lengths = torch.tensor([3, 2, 1], device=device)

    loss = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=V - 1,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert loss.ndim == 0
    assert loss >= 0
    assert torch.isfinite(loss)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_masking_correctness(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Verify padded positions are ignored by checking that loss is same
    regardless of values in padded regions."""
    torch.manual_seed(42)
    batch_size, T, U_plus_1, V = 2, 5, 4, 6
    teacher_logits = torch.randn(batch_size, T, U_plus_1, V, device=device)
    student_logits = torch.randn(batch_size, T, U_plus_1, V, device=device)
    targets = torch.randint(0, V - 1, (batch_size, U_plus_1 - 1), device=device)
    src_lengths = torch.tensor([3, 2], device=device)
    tgt_lengths = torch.tensor([2, 1], device=device)

    loss1 = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=V - 1,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )

    # Modify values in padded regions with deterministic values
    teacher_logits_modified = teacher_logits.clone()
    student_logits_modified = student_logits.clone()
    # Padded time frames for sample 0: t >= 3
    teacher_logits_modified[0, 3:, :, :] = 100.0
    student_logits_modified[0, 3:, :, :] = -100.0
    # Padded time frames for sample 1: t >= 2
    teacher_logits_modified[1, 2:, :, :] = 100.0
    student_logits_modified[1, 2:, :, :] = -100.0

    loss2 = consistency_rnnt_kld(
        teacher_logits=teacher_logits_modified,
        student_logits=student_logits_modified,
        targets=targets,
        blank_id=V - 1,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )

    assert torch.isclose(loss1, loss2, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
def test_reduction_modes_consistency(device: torch.device, symmetrical: bool, use_blank: bool):
    """Both reductions give valid (non-negative, finite) results."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss_mean = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction="mean",
    )
    loss_mean_volume = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction="mean_volume",
    )

    assert loss_mean >= 0
    assert loss_mean_volume >= 0
    assert torch.isfinite(loss_mean)
    assert torch.isfinite(loss_mean_volume)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_batch_size_one(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Edge case: batch_size = 1."""
    torch.manual_seed(42)
    teacher_logits = torch.randn(1, 4, 3, 5, device=device)
    student_logits = torch.randn(1, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (1, 2), device=device)

    loss = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert loss.ndim == 0
    assert loss >= 0
    assert torch.isfinite(loss)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_large_vocab(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Edge case: large vocabulary."""
    torch.manual_seed(42)
    V = 1024
    teacher_logits = torch.randn(2, 4, 3, V, device=device)
    student_logits = torch.randn(2, 4, 3, V, device=device)
    targets = torch.randint(0, V - 1, (2, 2), device=device)  # blank_id = V-1

    loss = consistency_rnnt_kld(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
        blank_id=V - 1,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )
    assert loss.ndim == 0
    assert loss >= 0
    assert torch.isfinite(loss)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("use_blank", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_module_api(device: torch.device, symmetrical: bool, use_blank: bool, reduction: str):
    """Test nn.Module wrapper."""
    torch.manual_seed(42)
    module = ConsistencyRNNTLoss(
        blank_id=4,
        symmetrical=symmetrical,
        use_blank=use_blank,
        reduction=reduction,
    )

    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss = module(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
    )
    assert loss.ndim == 0
    assert loss >= 0
    assert torch.isfinite(loss)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_consistency_full_loss(device: torch.device, symmetrical: bool, reduction: str):
    """Basic test"""
    torch.manual_seed(42)
    module = ConsistencyFullRNNTLoss(
        symmetrical=symmetrical,
        reduction=reduction,
    )

    teacher_logits = torch.randn(2, 4, 3, 5, device=device)
    student_logits = torch.randn(2, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (2, 2), device=device)

    loss = module(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
    )
    assert loss.ndim == 0
    assert loss >= 0
    assert torch.isfinite(loss)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="triton unavailable")
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean", "p_non_blank", "p_non_blank_with_grad"])
def test_consistency_full_loss_use_triton_matches_torch(symmetrical: bool, reduction: str):
    torch.manual_seed(123)
    device = torch.device("cuda")

    B, T, U_plus_1, V = 3, 5, 4, 7
    teacher_logits_base = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
    student_logits_base = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)

    targets = torch.randint(0, V - 1, (B, U_plus_1 - 1), device=device)
    src_lengths = torch.tensor([5, 3, 4], device=device)
    tgt_lengths = torch.tensor([3, 2, 1], device=device)

    def run(use_triton: bool):
        module = ConsistencyFullRNNTLoss(
            symmetrical=symmetrical,
            use_triton=use_triton,
            reduction=reduction,
            blank_id=V - 1,
        )
        teacher_logits = teacher_logits_base.detach().clone().requires_grad_(True)
        student_logits = student_logits_base.detach().clone().requires_grad_(True)
        loss = module(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
        )
        loss.backward()
        teacher_grad = None if teacher_logits.grad is None else teacher_logits.grad.detach().clone()
        student_grad = None if student_logits.grad is None else student_logits.grad.detach().clone()
        return loss.detach(), teacher_grad, student_grad

    torch_loss, torch_teacher_grad, torch_student_grad = run(use_triton=False)
    triton_loss, triton_teacher_grad, triton_student_grad = run(use_triton=True)

    assert torch.allclose(
        triton_loss, torch_loss, atol=3e-5, rtol=3e-4
    ), f"Loss mismatch. triton={triton_loss.item()}, torch={torch_loss.item()}"

    if torch_teacher_grad is None:
        assert triton_teacher_grad is None
    else:
        assert triton_teacher_grad is not None
        assert torch.allclose(
            triton_teacher_grad, torch_teacher_grad, atol=3e-4, rtol=2e-3
        ), f"Teacher grad max diff: {(triton_teacher_grad - torch_teacher_grad).abs().max()}"

    assert triton_student_grad is not None
    assert torch_student_grad is not None
    assert torch.allclose(
        triton_student_grad, torch_student_grad, atol=3e-4, rtol=2e-3
    ), f"Student grad max diff: {(triton_student_grad - torch_student_grad).abs().max()}"


@pytest.mark.unit
@pytest.mark.skipif(not K2_AVAILABLE, reason="k2 unavailable")
@pytest.mark.parametrize("device", DEVICES_WITH_CPU)
@pytest.mark.parametrize("symmetrical", [True, False])
def test_consistency_graph_rnnt_loss(device: torch.device, symmetrical: bool):
    """Basic test"""
    if device.type == "mps":
        pytest.skip(reason="k2 does not support mps")
    torch.manual_seed(42)
    module = ConsistencyGraphRNNTLoss(
        blank_id=4,
        symmetrical=symmetrical,
    )

    teacher_logits = torch.randn(4, 4, 3, 5, device=device)
    student_logits = torch.randn(4, 4, 3, 5, device=device)
    targets = torch.randint(0, 4, (4, 2), device=device)

    loss = module(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        targets=targets,
    )
    assert loss.ndim == 0
    assert loss >= 0
    assert torch.isfinite(loss)


# =============================================================================
# Tests for kl_loss_triton (Triton-based KL divergence loss)
# =============================================================================


def requires_cuda(fn):
    """Decorator to skip tests if CUDA is not available."""
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")(fn)


class TestKLLossTriton:
    """Tests for kl_loss_triton Triton implementation."""

    @pytest.mark.unit
    @requires_cuda
    @pytest.mark.parametrize("symmetrical", [True, False])
    @pytest.mark.parametrize("weighted", ["p_non_blank", "p_non_blank_with_grad"])
    def test_kl_loss_triton_weighted_returns_batch_vector(self, symmetrical: bool, weighted: str):
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 3, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        loss = kl_loss_triton(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            mask=mask,
            symmetrical=symmetrical,
            blank_id=V - 1,
            weighted=weighted,
        )
        assert loss.shape == (B,)
        assert loss.ndim == 1

        loss.mean().backward()
        assert student_logits.grad is not None
        if symmetrical or weighted == "p_non_blank_with_grad":
            assert teacher_logits.grad is not None
        else:
            assert teacher_logits.grad is None

    @pytest.mark.unit
    @requires_cuda
    def test_kl_loss_triton_forward_matches_pytorch(self):
        """Compare Triton forward output to PyTorch F.kl_div."""
        import torch.nn.functional as F

        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        # Triton implementation
        triton_loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=False)

        # PyTorch reference: KL(P||Q) = sum_v P(v) * (log P(v) - log Q(v))
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        # F.kl_div expects (log Q, P) and computes sum P * (log P - log Q)
        pytorch_loss = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)

        assert torch.allclose(
            triton_loss, pytorch_loss, atol=1e-5, rtol=1e-4
        ), f"Max diff: {(triton_loss - pytorch_loss).abs().max()}"

    @pytest.mark.unit
    @requires_cuda
    def test_kl_loss_triton_backward_gradcheck(self):
        """Numerical gradient verification with torch.autograd.gradcheck."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 3, 2, 4

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float64)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float64, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        def loss_fn(s_logits):
            return kl_loss_triton(teacher_logits, s_logits, mask, symmetrical=False).sum()

        assert torch.autograd.gradcheck(loss_fn, (student_logits,), eps=1e-6, atol=1e-4, rtol=1e-3)

    @pytest.mark.unit
    @requires_cuda
    def test_kl_loss_triton_backward_matches_pytorch(self):
        """Compare Triton backward output to PyTorch reference."""
        import torch.nn.functional as F

        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)

        # Triton path
        student_logits_triton = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)
        triton_loss = kl_loss_triton(teacher_logits, student_logits_triton, mask, symmetrical=False).sum()
        triton_loss.backward()
        triton_grad = student_logits_triton.grad.clone()

        # PyTorch reference path
        student_logits_pytorch = student_logits_triton.detach().clone().requires_grad_(True)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits_pytorch, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        pytorch_loss = F.kl_div(student_log_probs, teacher_probs, reduction='sum')
        pytorch_loss.backward()
        pytorch_grad = student_logits_pytorch.grad.clone()

        assert torch.allclose(
            triton_grad, pytorch_grad, atol=1e-5, rtol=1e-4
        ), f"Max grad diff: {(triton_grad - pytorch_grad).abs().max()}"

    @pytest.mark.unit
    @requires_cuda
    def test_kl_loss_triton_masked_positions_zero_grad(self):
        """Verify masked positions have zero gradients."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)

        # Create mask with some positions masked out
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)
        mask[0, 2:, :] = False  # Mask out t >= 2 for sample 0
        mask[1, :, 2:] = False  # Mask out u >= 2 for sample 1

        loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=False).sum()
        loss.backward()

        # Check that masked positions have zero gradient
        grad = student_logits.grad
        assert torch.all(grad[0, 2:, :, :] == 0), "Masked positions should have zero gradient"
        assert torch.all(grad[1, :, 2:, :] == 0), "Masked positions should have zero gradient"

    @pytest.mark.unit
    @requires_cuda
    def test_kl_loss_triton_memory_efficiency(self):
        """Verify peak memory is bounded (should NOT store [B, T, U+1, V] intermediates)."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 16, 129, 65, 2048

        # Calculate expected memory usage
        # Input size: B * T * U * V * 2 bytes (bfloat16) * 2 (teacher + student)
        input_bytes = B * T * U_plus_1 * V * 2 * 2
        # Gradient size same as inputs: B * T * U * V * 2 * 2
        grad_bytes = B * T * U_plus_1 * V * 2 * 2
        # Base memory = inputs + gradients
        base_memory = input_bytes + grad_bytes

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.bfloat16)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=False).sum()
        loss.backward()

        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()

        # The key check: peak memory should not include a full [B, T, U+1, V] intermediate
        # stored during forward for backward. If we stored log-softmax intermediate in float32,
        # it would add: B * T * U * V * 4 * 2 (two tensors for teacher and student) = ~8.5 GB
        # With efficient implementation, peak should be around 2x base (inputs + gradients)
        intermediate_storage_would_add = B * T * U_plus_1 * V * 4 * 2  # ~8.5 GB
        max_allowed_without_intermediate = base_memory + intermediate_storage_would_add

        # Verify we're well under the threshold that would indicate intermediate storage
        # Peak should be around 2x base (inputs + gradients), not 4x+ that intermediate storage would cause
        assert peak_memory < max_allowed_without_intermediate, (
            f"Peak memory {peak_memory / 1e9:.2f} GB suggests intermediate tensor stored. "
            f"Base: {base_memory / 1e9:.2f} GB, intermediate would add: {intermediate_storage_would_add / 1e9:.2f} GB"
        )

    @pytest.mark.unit
    @requires_cuda
    def test_kl_loss_triton_no_sync(self):
        """Verify no CUDA synchronization operations during forward/backward."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton
        from tests.collections.asr.decoding.utils import avoid_sync_operations

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        # Warm up
        _ = kl_loss_triton(teacher_logits, student_logits, mask).sum()

        with avoid_sync_operations(device):
            loss = kl_loss_triton(teacher_logits, student_logits.detach().requires_grad_(True), mask).sum()
            loss.backward()
        # Test passes if no exception is raised

    @pytest.mark.unit
    @requires_cuda
    def test_kl_loss_triton_identity_zero(self):
        """KL(P||P) = 0."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        loss = kl_loss_triton(logits, logits, mask, symmetrical=False)

        assert torch.allclose(
            loss, torch.zeros_like(loss), atol=1e-5
        ), f"KL(P||P) should be 0, got max={loss.abs().max()}"

    @pytest.mark.unit
    @requires_cuda
    def test_kl_loss_triton_non_negativity(self):
        """KL divergence should always be >= 0."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 4, 8, 5, 16

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=False)

        assert torch.all(loss >= -1e-5), f"KL should be >= 0, got min={loss.min()}"

    @pytest.mark.unit
    @requires_cuda
    def test_kl_loss_triton_symmetrical(self):
        """Test symmetric mode: swapping teacher/student gives same loss."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        loss1 = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=True)
        loss2 = kl_loss_triton(student_logits, teacher_logits, mask, symmetrical=True)

        assert torch.allclose(
            loss1, loss2, atol=1e-5
        ), f"Symmetrical loss should be invariant to swap, diff={torch.abs(loss1 - loss2).max()}"

    @pytest.mark.unit
    @requires_cuda
    def test_kl_loss_triton_gradient_flow_to_student(self):
        """Verify gradients flow to student logits but not teacher (teacher detached)."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=False).sum()
        loss.backward()

        # Teacher is detached in kl_loss_triton, so no gradient
        assert teacher_logits.grad is None or torch.all(
            teacher_logits.grad == 0
        ), "Teacher should not receive gradients (detached)"
        # Student should have non-zero gradients
        assert student_logits.grad is not None
        assert not torch.all(student_logits.grad == 0), "Student should receive gradients"

    @pytest.mark.unit
    @requires_cuda
    @pytest.mark.parametrize("V", [4095, 4096])
    def test_kl_loss_triton_large_vocab(self, V):
        """Test with large vocabulary size (power-of-2 and non-power-of-2)."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1 = 2, 4, 3

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=False).sum()
        loss.backward()

        assert torch.isfinite(loss)
        assert student_logits.grad is not None
        assert torch.all(torch.isfinite(student_logits.grad))

    @pytest.mark.unit
    @requires_cuda
    def test_kl_loss_triton_non_power_of_2_vocab(self):
        """Test forward correctness with non-power-of-2 vocab size."""
        import torch.nn.functional as F

        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 1025

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        # Triton implementation
        triton_loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=False)

        # PyTorch reference
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        pytorch_loss = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)

        assert torch.all(torch.isfinite(triton_loss)), "Triton loss contains NaN/Inf"
        assert torch.allclose(
            triton_loss, pytorch_loss, atol=1e-5, rtol=1e-4
        ), f"Max diff: {(triton_loss - pytorch_loss).abs().max()}"

    @pytest.mark.unit
    @requires_cuda
    def test_kl_loss_triton_non_power_of_2_vocab_backward(self):
        """Test backward correctness with non-power-of-2 vocab size."""
        import torch.nn.functional as F

        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 1025

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)

        # Triton path
        student_logits_triton = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)
        triton_loss = kl_loss_triton(teacher_logits, student_logits_triton, mask, symmetrical=False).sum()
        triton_loss.backward()
        triton_grad = student_logits_triton.grad.clone()

        # PyTorch reference path
        student_logits_pytorch = student_logits_triton.detach().clone().requires_grad_(True)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits_pytorch, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        pytorch_loss = F.kl_div(student_log_probs, teacher_probs, reduction='sum')
        pytorch_loss.backward()
        pytorch_grad = student_logits_pytorch.grad.clone()

        assert torch.all(torch.isfinite(triton_grad)), "Triton gradients contain NaN/Inf"
        assert torch.allclose(
            triton_grad, pytorch_grad, atol=1e-5, rtol=1e-4
        ), f"Max grad diff: {(triton_grad - pytorch_grad).abs().max()}"


# =============================================================================
# Tests for FusedKLDivTriton symmetric mode (Fused symmetric kernel)
# =============================================================================


class TestFusedSymmetricKLDivTriton:
    """Tests for the fused symmetric KL divergence Triton implementation."""

    @pytest.mark.unit
    @requires_cuda
    def test_fused_symmetric_forward_matches_two_kernel(self):
        """Verify fused symmetric kernel matches the two-kernel approach."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import FusedKLDivTriton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        # Two-kernel approach (old implementation style)
        kl_ts = FusedKLDivTriton.apply(teacher_logits.detach(), student_logits, mask, False)
        kl_st = FusedKLDivTriton.apply(student_logits.detach(), teacher_logits, mask, False)
        two_kernel_loss = 0.5 * (kl_ts + kl_st)

        # Fused symmetric kernel (symmetric=True)
        fused_loss = FusedKLDivTriton.apply(teacher_logits, student_logits, mask, True)

        assert torch.allclose(
            fused_loss, two_kernel_loss, atol=1e-5, rtol=1e-4
        ), f"Max diff: {(fused_loss - two_kernel_loss).abs().max()}"

    @pytest.mark.unit
    @requires_cuda
    def test_fused_symmetric_forward_matches_pytorch(self):
        """Verify fused symmetric kernel matches PyTorch reference."""
        import torch.nn.functional as F

        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        # Triton implementation
        triton_loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=True)

        # PyTorch reference
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_probs = student_log_probs.exp()

        kl_ts = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
        kl_st = F.kl_div(teacher_log_probs, student_probs, reduction='none').sum(dim=-1)
        pytorch_loss = 0.5 * (kl_ts + kl_st)

        assert torch.allclose(
            triton_loss, pytorch_loss, atol=1e-5, rtol=1e-4
        ), f"Max diff: {(triton_loss - pytorch_loss).abs().max()}"

    @pytest.mark.unit
    @requires_cuda
    def test_fused_symmetric_backward_matches_two_kernel(self):
        """Verify fused symmetric backward matches the original two-kernel approach."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import FusedKLDivTriton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        # Two-kernel approach (original implementation style)
        teacher_logits_2k = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        student_logits_2k = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        kl_ts = FusedKLDivTriton.apply(teacher_logits_2k.detach(), student_logits_2k, mask, False)
        kl_st = FusedKLDivTriton.apply(student_logits_2k.detach(), teacher_logits_2k, mask, False)
        two_kernel_loss = 0.5 * (kl_ts + kl_st)
        two_kernel_loss.sum().backward()
        two_kernel_teacher_grad = teacher_logits_2k.grad.clone()
        two_kernel_student_grad = student_logits_2k.grad.clone()

        # Fused kernel approach (symmetric=True)
        teacher_logits_fused = teacher_logits_2k.detach().clone().requires_grad_(True)
        student_logits_fused = student_logits_2k.detach().clone().requires_grad_(True)

        fused_loss = FusedKLDivTriton.apply(teacher_logits_fused, student_logits_fused, mask, True)
        fused_loss.sum().backward()
        fused_teacher_grad = teacher_logits_fused.grad.clone()
        fused_student_grad = student_logits_fused.grad.clone()

        assert torch.allclose(
            fused_teacher_grad, two_kernel_teacher_grad, atol=1e-5, rtol=1e-4
        ), f"Teacher grad max diff: {(fused_teacher_grad - two_kernel_teacher_grad).abs().max()}"
        assert torch.allclose(
            fused_student_grad, two_kernel_student_grad, atol=1e-5, rtol=1e-4
        ), f"Student grad max diff: {(fused_student_grad - two_kernel_student_grad).abs().max()}"

    @pytest.mark.unit
    @requires_cuda
    def test_fused_symmetric_backward_matches_pytorch(self):
        """Compare fused symmetric backward to PyTorch reference."""
        import torch.nn.functional as F

        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        # Triton path
        teacher_logits_triton = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        student_logits_triton = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        triton_loss = kl_loss_triton(teacher_logits_triton, student_logits_triton, mask, symmetrical=True).sum()
        triton_loss.backward()
        triton_teacher_grad = teacher_logits_triton.grad.clone()
        triton_student_grad = student_logits_triton.grad.clone()

        # PyTorch reference path
        teacher_logits_pytorch = teacher_logits_triton.detach().clone().requires_grad_(True)
        student_logits_pytorch = student_logits_triton.detach().clone().requires_grad_(True)

        teacher_log_probs = F.log_softmax(teacher_logits_pytorch, dim=-1)
        student_log_probs = F.log_softmax(student_logits_pytorch, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_probs = student_log_probs.exp()

        # KL(teacher || student) - gradients flow to student
        kl_ts = F.kl_div(student_log_probs, teacher_probs.detach(), reduction='sum')
        # KL(student || teacher) - gradients flow to teacher
        kl_st = F.kl_div(teacher_log_probs, student_probs.detach(), reduction='sum')
        pytorch_loss = 0.5 * (kl_ts + kl_st)
        pytorch_loss.backward()
        pytorch_teacher_grad = teacher_logits_pytorch.grad.clone()
        pytorch_student_grad = student_logits_pytorch.grad.clone()

        assert torch.allclose(
            triton_teacher_grad, pytorch_teacher_grad, atol=1e-5, rtol=1e-4
        ), f"Teacher grad max diff: {(triton_teacher_grad - pytorch_teacher_grad).abs().max()}"
        assert torch.allclose(
            triton_student_grad, pytorch_student_grad, atol=1e-5, rtol=1e-4
        ), f"Student grad max diff: {(triton_student_grad - pytorch_student_grad).abs().max()}"

    @pytest.mark.unit
    @requires_cuda
    def test_fused_symmetric_both_receive_gradients(self):
        """Verify both teacher and student receive gradients in symmetric mode."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=True).sum()
        loss.backward()

        # Both should receive non-zero gradients
        assert teacher_logits.grad is not None, "Teacher should receive gradients in symmetric mode"
        assert student_logits.grad is not None, "Student should receive gradients in symmetric mode"
        assert not torch.all(teacher_logits.grad == 0), "Teacher gradients should be non-zero"
        assert not torch.all(student_logits.grad == 0), "Student gradients should be non-zero"

    @pytest.mark.unit
    @requires_cuda
    def test_fused_symmetric_gradients_are_negatives(self):
        """Verify teacher and student gradients are negatives of each other."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=True).sum()
        loss.backward()

        # Gradients should be negatives: grad_teacher = -grad_student
        assert torch.allclose(
            teacher_logits.grad, -student_logits.grad, atol=1e-5
        ), f"Gradients should be negatives, max diff: {(teacher_logits.grad + student_logits.grad).abs().max()}"

    @pytest.mark.unit
    @requires_cuda
    def test_fused_symmetric_masked_positions_zero_grad(self):
        """Verify masked positions have zero gradients for both tensors."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)

        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)
        mask[0, 2:, :] = False
        mask[1, :, 2:] = False

        loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=True).sum()
        loss.backward()

        # Check teacher gradients
        assert torch.all(teacher_logits.grad[0, 2:, :, :] == 0), "Masked positions should have zero teacher gradient"
        assert torch.all(teacher_logits.grad[1, :, 2:, :] == 0), "Masked positions should have zero teacher gradient"

        # Check student gradients
        assert torch.all(student_logits.grad[0, 2:, :, :] == 0), "Masked positions should have zero student gradient"
        assert torch.all(student_logits.grad[1, :, 2:, :] == 0), "Masked positions should have zero student gradient"

    @pytest.mark.unit
    @requires_cuda
    def test_fused_symmetric_identity_zero(self):
        """Symmetric KL(P||P) = 0."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        loss = kl_loss_triton(logits, logits, mask, symmetrical=True)

        assert torch.allclose(
            loss, torch.zeros_like(loss), atol=1e-5
        ), f"Symmetric KL(P||P) should be 0, got max={loss.abs().max()}"

    @pytest.mark.unit
    @requires_cuda
    def test_fused_symmetric_non_negativity(self):
        """Symmetric KL divergence should always be >= 0."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 4, 8, 5, 16

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=True)

        assert torch.all(loss >= -1e-5), f"Symmetric KL should be >= 0, got min={loss.min()}"

    @pytest.mark.unit
    @requires_cuda
    @pytest.mark.parametrize("V", [4095, 4096])
    def test_fused_symmetric_large_vocab(self, V):
        """Test fused symmetric kernel with large vocabulary size (power-of-2 and non-power-of-2)."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1 = 2, 4, 3

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=True).sum()
        loss.backward()

        assert torch.isfinite(loss)
        assert teacher_logits.grad is not None
        assert student_logits.grad is not None
        assert torch.all(torch.isfinite(teacher_logits.grad))
        assert torch.all(torch.isfinite(student_logits.grad))

    @pytest.mark.unit
    @requires_cuda
    def test_fused_symmetric_non_power_of_2_vocab(self):
        """Test fused symmetric forward+backward with non-power-of-2 vocab size."""
        import torch.nn.functional as F

        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 1025

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        # Triton implementation
        triton_loss = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=True)

        # PyTorch reference
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_probs = student_log_probs.exp()
        kl_ts = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
        kl_st = F.kl_div(teacher_log_probs, student_probs, reduction='none').sum(dim=-1)
        pytorch_loss = 0.5 * (kl_ts + kl_st)

        assert torch.all(torch.isfinite(triton_loss)), "Triton loss contains NaN/Inf"
        assert torch.allclose(
            triton_loss, pytorch_loss, atol=1e-5, rtol=1e-4
        ), f"Max diff: {(triton_loss - pytorch_loss).abs().max()}"

        # Also verify backward
        triton_loss.sum().backward()
        assert teacher_logits.grad is not None
        assert student_logits.grad is not None
        assert torch.all(torch.isfinite(teacher_logits.grad)), "Teacher gradients contain NaN/Inf"
        assert torch.all(torch.isfinite(student_logits.grad)), "Student gradients contain NaN/Inf"

    @pytest.mark.unit
    @requires_cuda
    def test_fused_symmetric_no_sync(self):
        """Verify no CUDA synchronization operations during fused symmetric forward/backward."""
        from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton
        from tests.collections.asr.decoding.utils import avoid_sync_operations

        torch.manual_seed(42)
        device = torch.device("cuda")
        B, T, U_plus_1, V = 2, 4, 3, 8

        teacher_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        student_logits = torch.randn(B, T, U_plus_1, V, device=device, dtype=torch.float32, requires_grad=True)
        mask = torch.ones(B, T, U_plus_1, dtype=torch.bool, device=device)

        # Warm up
        _ = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=True).sum()

        with avoid_sync_operations(device):
            t = teacher_logits.detach().clone().requires_grad_(True)
            s = student_logits.detach().clone().requires_grad_(True)
            loss = kl_loss_triton(t, s, mask, symmetrical=True).sum()
            loss.backward()
        # Test passes if no exception is raised
