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

import math

import pytest
import torch

from nemo.collections.asr.parts.turbo_transducer import rnnt_consistency
from nemo.collections.asr.parts.turbo_transducer.backend import TurboTransducerBackend
from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency import ConsistencyRNNTLoss
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import get_rnnt_mask
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency_triton import kl_loss_triton

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(pytest.param(torch.device("cuda"), marks=pytest.mark.gpu))

# NB: tests use a small set of shapes and vocabulary sizes to reuse compiled Triton kernels
# (tests should be fast without Triton cache)
SHAPES = [
    # batch_size, num_frames, num_text_units_plus_1, vocab_size (with blank)
    (3, 5, 4, 5),
    (2, 11, 8, 1025),  # vocabulary size is not a power of 2
]


def _random_inputs(
    batch_size: int,
    num_frames: int,
    num_text_units_plus_1: int,
    vocab_size: int,
    device: torch.device,
    float_dtype: torch.dtype = torch.float32,
):
    """Random teacher/student logits with variable lengths (including an empty target)."""
    teacher_logits = torch.randn(batch_size, num_frames, num_text_units_plus_1, vocab_size, device=device)
    student_logits = torch.randn(batch_size, num_frames, num_text_units_plus_1, vocab_size, device=device)
    src_lengths = torch.randint(1, num_frames + 1, [batch_size], device=device)
    tgt_lengths = torch.randint(0, num_text_units_plus_1, [batch_size], device=device)
    src_lengths[0] = num_frames
    tgt_lengths[0] = num_text_units_plus_1 - 1
    tgt_lengths[-1] = 0
    return teacher_logits.to(float_dtype), student_logits.to(float_dtype), src_lengths, tgt_lengths


def _reference_kl(teacher_logits, student_logits, mask, symmetrical: bool) -> torch.Tensor:
    """Straightforward per-position KL divergence (KL(teacher || student))."""
    teacher_logprobs = teacher_logits.log_softmax(dim=-1)
    student_logprobs = student_logits.log_softmax(dim=-1)
    kl = (teacher_logprobs.exp() * (teacher_logprobs - student_logprobs)).sum(dim=-1)
    if symmetrical:
        kl_reverse = (student_logprobs.exp() * (student_logprobs - teacher_logprobs)).sum(dim=-1)
        kl = 0.5 * (kl + kl_reverse)
    return kl * mask


def _get_mask(src_lengths, tgt_lengths, shape):
    batch_size, num_frames, num_text_units_plus_1 = shape[:3]
    _, mask = get_rnnt_mask(
        batch_size=batch_size,
        src_length_max=num_frames,
        tgt_length_max_plus_1=num_text_units_plus_1,
        device=src_lengths.device,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )
    return mask


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_consistency_loss_matches_reference(device: torch.device, symmetrical: bool, reduction: str):
    """The loss (with automatically selected backend) matches the reference formula and reduction."""
    torch.manual_seed(42)
    teacher_logits, student_logits, src_lengths, tgt_lengths = _random_inputs(*SHAPES[0], device=device)
    mask = _get_mask(src_lengths, tgt_lengths, SHAPES[0])
    kl = _reference_kl(teacher_logits, student_logits, mask, symmetrical)
    if reduction == "mean_volume":
        expected = kl.sum() / mask.sum()
    else:
        expected = (kl.sum(dim=(1, 2)) / mask.sum(dim=(1, 2))).mean()

    module = ConsistencyRNNTLoss(symmetrical=symmetrical, reduction=reduction)
    loss = module(teacher_logits, student_logits, src_lengths=src_lengths, tgt_lengths=tgt_lengths)
    assert loss.ndim == 0
    assert torch.allclose(loss, expected, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_consistency_forward_from_kl_matches_masked_reduction(device: torch.device, reduction: str):
    """`forward_from_kl` must reproduce the reduction used by the full `forward` (mask-based denominators)."""
    torch.manual_seed(11)
    module = ConsistencyRNNTLoss(reduction=reduction)

    shape = (3, 5, 4)
    src_lengths = torch.tensor([5, 3, 4], device=device)
    # NB: includes an empty target sequence
    tgt_lengths = torch.tensor([3, 0, 1], device=device)
    mask = _get_mask(src_lengths, tgt_lengths, shape)
    consistency_kl = torch.rand(shape, device=device) * mask

    if reduction == "mean_volume":
        expected = consistency_kl.sum() / mask.sum().clamp(min=1)
    else:
        expected = (consistency_kl.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)).mean()

    actual = module.forward_from_kl(consistency_kl=consistency_kl, src_lengths=src_lengths, tgt_lengths=tgt_lengths)
    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("reduction", ["mean_volume", "mean"])
def test_consistency_loss_low_precision_reduction(reduction: str):
    """Per-position losses in float16 must be reduced in higher precision (the sum exceeds the float16 range)."""
    num_frames = 2048
    teacher_logits = torch.zeros([1, num_frames, 1, 2], dtype=torch.float16, requires_grad=True)
    student_logits = torch.tensor([0.0, -100.0], dtype=torch.float16).repeat(1, num_frames, 1, 1).requires_grad_(True)
    loss_fn = ConsistencyRNNTLoss(backend=TurboTransducerBackend.TORCH, symmetrical=False, reduction=reduction)
    loss = loss_fn(teacher_logits, student_logits)
    loss.backward()
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(50.0 - math.log(2.0), abs=0.05)
    assert torch.isfinite(student_logits.grad).all()
    assert teacher_logits.grad is None


@pytest.mark.unit
class TestConsistencyLossBackend:
    def test_unsupported_backend(self):
        with pytest.raises(NotImplementedError):
            ConsistencyRNNTLoss(backend="cutile")

    def test_triton_backend_without_triton(self, monkeypatch):
        monkeypatch.setattr(rnnt_consistency, "TRITON_AVAILABLE", False)
        with pytest.raises(ImportError):
            ConsistencyRNNTLoss(backend="triton")

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is unavailable")
    def test_triton_backend_cpu_input(self):
        teacher_logits, student_logits, _, _ = _random_inputs(*SHAPES[0], device=torch.device("cpu"))
        with pytest.raises(ValueError, match="CUDA"):
            ConsistencyRNNTLoss(backend="triton")(teacher_logits, student_logits)

    def test_auto_backend(self):
        module = ConsistencyRNNTLoss()
        assert module.get_runtime_backend(torch.zeros(1)) is TurboTransducerBackend.TORCH


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is unavailable")
class TestConsistencyLossTriton:
    @pytest.mark.parametrize("float_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("symmetrical", [True, False])
    @pytest.mark.parametrize("shape", SHAPES)
    def test_triton_matches_torch(self, shape, symmetrical: bool, float_dtype: torch.dtype):
        """Triton and PyTorch backends produce the same loss and gradients."""
        torch.manual_seed(123)
        device = torch.device("cuda")
        teacher_logits, student_logits, src_lengths, tgt_lengths = _random_inputs(
            *shape, device=device, float_dtype=float_dtype
        )
        mask = _get_mask(src_lengths, tgt_lengths, shape)

        def run(backend: TurboTransducerBackend):
            module = ConsistencyRNNTLoss(symmetrical=symmetrical, backend=backend)
            # Triton implementation computes in float32, compare with float32 reference
            dtype = float_dtype if backend is TurboTransducerBackend.TRITON else torch.float32
            teacher = teacher_logits.detach().to(dtype).requires_grad_(True)
            student = student_logits.detach().to(dtype).requires_grad_(True)
            loss = module(teacher, student, src_lengths=src_lengths, tgt_lengths=tgt_lengths)
            loss.backward()
            return loss.detach(), teacher.grad, student.grad

        torch_loss, torch_teacher_grad, torch_student_grad = run(TurboTransducerBackend.TORCH)
        triton_loss, triton_teacher_grad, triton_student_grad = run(TurboTransducerBackend.TRITON)

        assert triton_loss.dtype == torch.float32
        assert torch.allclose(triton_loss, torch_loss, atol=1e-6, rtol=1e-5)
        # gradient is returned in the dtype of the logits
        atol, rtol = (1e-6, 1e-5) if float_dtype == torch.float32 else (1e-5, 1e-2)
        assert torch.allclose(triton_student_grad.float(), torch_student_grad, atol=atol, rtol=rtol)
        assert (triton_student_grad[~mask] == 0.0).all()
        if symmetrical:
            assert torch.allclose(triton_teacher_grad.float(), torch_teacher_grad, atol=atol, rtol=rtol)
            assert (triton_teacher_grad[~mask] == 0.0).all()
        else:
            # teacher is detached for non-symmetrical loss
            assert triton_teacher_grad is None and torch_teacher_grad is None

    @pytest.mark.parametrize("symmetrical", [True, False])
    def test_float64_gradcheck(self, symmetrical: bool):
        torch.manual_seed(1)
        teacher_logits, student_logits, src_lengths, tgt_lengths = _random_inputs(
            *SHAPES[0], device=torch.device("cuda"), float_dtype=torch.float64
        )
        mask = _get_mask(src_lengths, tgt_lengths, SHAPES[0])
        student_logits.requires_grad_(True)
        # NB: the loss uses stop-gradient for the "target" distribution in each KL term, so the gradient for teacher
        # is not the true gradient of the symmetrical loss; check only the student gradient with fixed teacher
        assert torch.autograd.gradcheck(
            lambda x: kl_loss_triton(teacher_logits, x, mask, symmetrical=False), (student_logits,)
        )
        if symmetrical:
            teacher_logits.requires_grad_(True)
            output = kl_loss_triton(teacher_logits, student_logits, mask, symmetrical=True)
            output.sum().backward()
            # teacher gradient: 0.5 * (softmax(teacher) - softmax(student)), negated student gradient
            assert torch.allclose(teacher_logits.grad, -student_logits.grad)

    @pytest.mark.parametrize("symmetrical", [True, False])
    def test_kl_properties(self, symmetrical: bool):
        """KL is zero for identical distributions and non-negative otherwise."""
        torch.manual_seed(2)
        teacher_logits, student_logits, src_lengths, tgt_lengths = _random_inputs(*SHAPES[0], device="cuda")
        mask = _get_mask(src_lengths, tgt_lengths, SHAPES[0])
        assert torch.allclose(
            kl_loss_triton(teacher_logits, teacher_logits.clone(), mask, symmetrical),
            torch.zeros(mask.shape, device=mask.device),
            atol=1e-6,
        )
        assert (kl_loss_triton(teacher_logits, student_logits, mask, symmetrical) >= 0).all()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Test requires 2 CUDA devices")
    @pytest.mark.parametrize("symmetrical", [True, False])
    def test_non_current_device(self, symmetrical: bool):
        """Inputs are on a CUDA device that is not the current device."""
        torch.manual_seed(5)
        # NB: inputs are created on CPU; copying tensors between CUDA devices can hide the issue
        teacher_logits, student_logits, src_lengths, tgt_lengths = _random_inputs(*SHAPES[0], device="cpu")
        other_device = torch.device("cuda", (torch.cuda.current_device() + 1) % torch.cuda.device_count())
        loss_fn = ConsistencyRNNTLoss(symmetrical=symmetrical, backend=TurboTransducerBackend.TRITON)
        results = []
        for device in [torch.device("cuda"), other_device]:
            teacher = teacher_logits.to(device).requires_grad_(True)
            student = student_logits.to(device).requires_grad_(True)
            loss = loss_fn(teacher, student, src_lengths.to(device), tgt_lengths.to(device))
            loss.backward()
            results.append((loss.cpu(), student.grad.cpu(), teacher.grad.cpu() if symmetrical else None))
        (loss, student_grad, teacher_grad), (other_loss, other_student_grad, other_teacher_grad) = results
        assert torch.equal(loss, other_loss)
        assert torch.equal(student_grad, other_student_grad)
        if symmetrical:
            assert torch.equal(teacher_grad, other_teacher_grad)

    def test_no_cuda_sync_operations(self):
        from tests.collections.asr.decoding.utils import avoid_sync_operations

        torch.manual_seed(3)
        teacher_logits, student_logits, src_lengths, tgt_lengths = _random_inputs(*SHAPES[0], device="cuda")
        student_logits.requires_grad_(True)
        module = ConsistencyRNNTLoss(backend="triton")
        module(teacher_logits, student_logits, src_lengths, tgt_lengths).backward()  # warmup
        with avoid_sync_operations(teacher_logits.device):
            module(teacher_logits, student_logits, src_lengths, tgt_lengths).backward()

    @pytest.mark.parametrize("symmetrical", [True, False])
    def test_memory_efficiency(self, symmetrical: bool):
        """No full-size intermediate tensors: forward + backward allocate only the gradient buffers."""
        torch.manual_seed(4)
        shape = (4, 64, 33, SHAPES[1][-1])
        teacher_logits, student_logits, src_lengths, tgt_lengths = _random_inputs(*shape, device="cuda")
        teacher_logits.requires_grad_(symmetrical)
        student_logits.requires_grad_(True)
        mask = _get_mask(src_lengths, tgt_lengths, shape)
        logits_size_bytes = student_logits.numel() * student_logits.element_size()
        num_gradients = 2 if symmetrical else 1

        kl_loss_triton(teacher_logits, student_logits, mask, symmetrical).sum().backward()  # warmup
        teacher_logits.grad, student_logits.grad = None, None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated()
        kl_loss_triton(teacher_logits, student_logits, mask, symmetrical).sum().backward()
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() - memory_before
        assert peak_memory < (num_gradients + 0.1) * logits_size_bytes
