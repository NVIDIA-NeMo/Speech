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
import torch.nn.functional as F

from nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs import rnnt_logprobs_torch
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.decoding.utils import avoid_sync_operations

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.rnnt_triton.rnnt_joint_vocab_logprobs_triton import (
        rnnt_joint_vocab_logprobs_triton,
    )


def _reference_joint_vocab_logprobs(joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, blank_id):
    logits = F.linear(joint_hidden, weight, bias)
    target_logprobs, blank_logprobs = rnnt_logprobs_torch(
        logits=logits,
        targets=targets,
        blank_id=blank_id,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )
    return target_logprobs, blank_logprobs


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
class TestRnntJointVocabLogProbsTriton:
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 2, 1, 4, 3),
            (2, 4, 3, 16, 8),
            (4, 16, 8, 64, 32),
            (5, 17, 65, 28, 35),
            (2, 64, 32, 640, 1024),
        ],
    )
    @pytest.mark.parametrize(
        "float_dtype",
        [torch.float32] + ([torch.bfloat16] if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else []),
    )
    def test_forward_and_backward(self, shape, float_dtype):
        device = torch.device("cuda")
        torch.manual_seed(42)
        use_high_precision = True
        # torch.autograd.set_detect_anomaly(True)
        batch_size, src_length, tgt_length, hidden_dim, vocab_size_no_blank = shape
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        joint_hidden = torch.randn(
            batch_size, src_length, tgt_length + 1, hidden_dim, device=device, dtype=float_dtype
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=float_dtype)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=float_dtype)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.full([batch_size], tgt_length, device=device, dtype=torch.long)

        if float_dtype == torch.bfloat16:
            joint_hidden_ref = joint_hidden.detach().float().clone().requires_grad_(True)
            weight_ref = weight.detach().float().clone().requires_grad_(True)
            bias_ref = bias.detach().float().clone().requires_grad_(True)
        else:
            joint_hidden_ref = joint_hidden.detach().clone().requires_grad_(True)
            weight_ref = weight.detach().clone().requires_grad_(True)
            bias_ref = bias.detach().clone().requires_grad_(True)
        target_ref, blank_ref = _reference_joint_vocab_logprobs(
            joint_hidden=joint_hidden_ref,
            weight=weight_ref,
            bias=bias_ref,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
        )

        joint_hidden_tri = joint_hidden.detach().clone().requires_grad_(True)
        weight_tri = weight.detach().clone().requires_grad_(True)
        bias_tri = bias.detach().clone().requires_grad_(True)
        target_tri, blank_tri = rnnt_joint_vocab_logprobs_triton(
            joint_hidden=joint_hidden_tri,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight_tri,
            bias=bias_tri,
            blank_id=blank_id,
            use_high_precision=use_high_precision,
        )

        forward_atol = 1e-5
        assert torch.allclose(
            target_tri, target_ref, atol=forward_atol
        ), f"target logprobs mismatch: max diff={(target_tri - target_ref).abs().max().item()}"
        assert torch.allclose(
            blank_tri, blank_ref, atol=forward_atol
        ), f"blank logprobs mismatch: max diff={(blank_tri - blank_ref).abs().max().item()}"

        torch.manual_seed(123)
        target_scales = torch.rand_like(target_ref)
        blank_scales = torch.rand_like(blank_ref)

        loss_ref = (target_scales * target_ref + blank_scales * blank_ref).sum()
        loss_tri = (target_scales * target_tri + blank_scales * blank_tri).sum()
        loss_ref.backward()
        loss_tri.backward()

        assert joint_hidden_tri.grad.dtype == float_dtype
        assert weight_tri.grad.dtype == float_dtype
        assert bias_tri.grad.dtype == float_dtype

        grad_atol = 5e-3 if float_dtype == torch.bfloat16 else 1e-4
        grad_rtol = 5e-3 if float_dtype == torch.bfloat16 else 1e-5
        assert torch.allclose(
            joint_hidden_tri.grad.float(), joint_hidden_ref.grad.float(), atol=grad_atol, rtol=grad_rtol
        ), f"joint_hidden grad mismatch: max diff={(joint_hidden_tri.grad.float() - joint_hidden_ref.grad.float()).abs().max().item()}"

        weight_bias_atol = 1e-0 if float_dtype == torch.bfloat16 else 1e-4
        weight_bias_rtol = 1e-3 if float_dtype == torch.bfloat16 else 1e-5
        assert torch.allclose(
            weight_tri.grad.float(), weight_ref.grad.float(), atol=weight_bias_atol, rtol=weight_bias_rtol
        ), f"weight grad mismatch: max diff={(weight_tri.grad.float() - weight_ref.grad.float()).abs().max().item()}"
        assert torch.allclose(
            bias_tri.grad.float(), bias_ref.grad.float(), atol=weight_bias_atol, rtol=weight_bias_rtol
        ), f"bias grad mismatch: max diff={(bias_tri.grad.float() - bias_ref.grad.float()).abs().max().item()}"

    def test_variable_lengths(self):
        device = torch.device("cuda")
        torch.manual_seed(42)
        use_high_precision = True
        batch_size = 4
        src_length = 16
        tgt_length = 8
        hidden_dim = 32
        vocab_size_no_blank = 16
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        joint_hidden = torch.randn(
            batch_size, src_length, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.tensor([src_length, src_length // 2, src_length - 1, 1], device=device, dtype=torch.long)
        tgt_lengths = torch.tensor([tgt_length, tgt_length // 2, tgt_length - 1, 0], device=device, dtype=torch.long)

        joint_hidden_ref = joint_hidden.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        bias_ref = bias.detach().clone().requires_grad_(True)
        target_ref, blank_ref = _reference_joint_vocab_logprobs(
            joint_hidden=joint_hidden_ref,
            weight=weight_ref,
            bias=bias_ref,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
        )

        joint_hidden_tri = joint_hidden.detach().clone().requires_grad_(True)
        weight_tri = weight.detach().clone().requires_grad_(True)
        bias_tri = bias.detach().clone().requires_grad_(True)
        target_tri, blank_tri = rnnt_joint_vocab_logprobs_triton(
            joint_hidden=joint_hidden_tri,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight_tri,
            bias=bias_tri,
            blank_id=blank_id,
            use_high_precision=use_high_precision,
        )

        forward_atol = 1e-5
        assert torch.allclose(target_tri, target_ref, atol=forward_atol)
        assert torch.allclose(blank_tri, blank_ref, atol=forward_atol)

        target_scales = torch.rand_like(target_ref)
        blank_scales = torch.rand_like(blank_ref)
        loss_ref = (target_scales * target_ref + blank_scales * blank_ref).sum()
        loss_tri = (target_scales * target_tri + blank_scales * blank_tri).sum()
        loss_ref.backward()
        loss_tri.backward()

        assert torch.allclose(joint_hidden_tri.grad.float(), joint_hidden_ref.grad.float(), atol=5e-3, rtol=1e-3)
        assert torch.allclose(weight_tri.grad.float(), weight_ref.grad.float(), atol=0.5, rtol=1e-3)
        assert torch.allclose(bias_tri.grad.float(), bias_ref.grad.float(), atol=0.5, rtol=1e-3)

    def test_edge_case_single_frame(self):
        device = torch.device("cuda")
        torch.manual_seed(42)
        batch_size = 1
        src_length = 1
        tgt_length = 1
        hidden_dim = 4
        vocab_size_no_blank = 3
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        joint_hidden = torch.randn(
            batch_size, src_length, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.tensor([src_length], device=device, dtype=torch.long)
        tgt_lengths = torch.tensor([tgt_length], device=device, dtype=torch.long)

        target_ref, blank_ref = _reference_joint_vocab_logprobs(
            joint_hidden=joint_hidden,
            weight=weight,
            bias=bias,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
        )
        target_tri, blank_tri = rnnt_joint_vocab_logprobs_triton(
            joint_hidden=joint_hidden,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight,
            bias=bias,
            blank_id=blank_id,
        )

        assert torch.allclose(target_tri, target_ref, atol=5e-3)
        assert torch.allclose(blank_tri, blank_ref, atol=5e-3)

    def test_edge_case_blank_only(self):
        device = torch.device("cuda")
        torch.manual_seed(42)
        batch_size = 2
        src_length = 4
        tgt_length = 0
        hidden_dim = 8
        vocab_size_no_blank = 5
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        joint_hidden = torch.randn(
            batch_size, src_length, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32)
        targets = torch.randint(
            0, vocab_size_no_blank, (batch_size, max(tgt_length, 1)), device=device, dtype=torch.long
        )
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.zeros([batch_size], device=device, dtype=torch.long)

        target_ref, blank_ref = _reference_joint_vocab_logprobs(
            joint_hidden=joint_hidden,
            weight=weight,
            bias=bias,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
        )
        target_tri, blank_tri = rnnt_joint_vocab_logprobs_triton(
            joint_hidden=joint_hidden,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight,
            bias=bias,
            blank_id=blank_id,
        )

        assert torch.allclose(target_tri, target_ref, atol=5e-3)
        assert torch.allclose(blank_tri, blank_ref, atol=5e-3)

    def test_no_cuda_sync_operations(self):
        device = torch.device("cuda")
        torch.manual_seed(42)

        batch_size = 1
        src_length = 2
        tgt_length = 1
        hidden_dim = 4
        vocab_size_no_blank = 3
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        joint_hidden = torch.randn(
            batch_size, src_length, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32, requires_grad=True
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.full([batch_size], tgt_length, device=device, dtype=torch.long)

        # Warmup
        target_logprobs, blank_logprobs = rnnt_joint_vocab_logprobs_triton(
            joint_hidden=joint_hidden,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight,
            bias=bias,
            blank_id=blank_id,
        )
        warmup_loss = target_logprobs.sum() + blank_logprobs.sum()
        warmup_loss.backward()
        torch.cuda.synchronize()

        joint_hidden_test = joint_hidden.detach().clone().requires_grad_(True)
        weight_test = weight.detach().clone().requires_grad_(True)
        bias_test = bias.detach().clone().requires_grad_(True)

        with avoid_sync_operations(device):
            target_logprobs_test, blank_logprobs_test = rnnt_joint_vocab_logprobs_triton(
                joint_hidden=joint_hidden_test,
                targets=targets,
                tgt_lengths=tgt_lengths,
                src_lengths=src_lengths,
                weight=weight_test,
                bias=bias_test,
                blank_id=blank_id,
            )
            loss = target_logprobs_test.sum() + blank_logprobs_test.sum()
            loss.backward()
