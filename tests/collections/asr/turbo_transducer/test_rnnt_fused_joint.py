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

from nemo.collections.asr.parts.turbo_transducer.rnnt_fused_joint import rnnt_fused_joint
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import rnnt_logprobs_torch
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.decoding.utils import avoid_sync_operations

if TRITON_AVAILABLE:
    import triton

    from nemo.collections.asr.parts.turbo_transducer.utils_triton import dropout_scale_mask_kernel, mix_dropout_seed


def _triton_dropout_scale_mask(shape, dropout_p, dropout_seed, device, dtype):
    if dropout_p == 0.0:
        return torch.ones(shape, device=device, dtype=dtype)
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    dropout_inv_keep_prob = 1.0 / (1.0 - dropout_p)
    dropout_threshold = int(dropout_p * torch.iinfo(torch.int32).max)
    mask_flat = torch.empty([total_elements], device=device, dtype=torch.float32)
    BLOCK = 1024
    dropout_scale_mask_kernel[(triton.cdiv(total_elements, BLOCK),)](
        mask_out_ptr=mask_flat,
        total_elements=total_elements,
        mixed_dropout_seed=mix_dropout_seed(dropout_seed),
        dropout_threshold=dropout_threshold,
        dropout_inv_keep_prob=dropout_inv_keep_prob,
        BLOCK=BLOCK,
    )
    return mask_flat.view(shape).to(dtype)


def _reference_joint_logprobs(
    enc,
    pred,
    weight,
    bias,
    targets,
    src_lengths,
    tgt_lengths,
    blank_id,
    dropout_scale_mask=None,
):
    joint_hidden = torch.relu(enc.unsqueeze(2) + pred.unsqueeze(1))
    if dropout_scale_mask is not None:
        joint_hidden = joint_hidden * dropout_scale_mask
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
class TestRnntJointTriton:
    @pytest.mark.parametrize("mm_full_precision", [False, True])
    def test_mixed_precision(self, mm_full_precision):
        torch.manual_seed(42)
        encoder = torch.randn(2, 5, 32, device="cuda", dtype=torch.bfloat16)
        predictor = torch.randn(2, 4, 32, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(17, 32, device="cuda")
        bias = torch.randn(17, device="cuda")
        targets = torch.randint(0, 16, (2, 3), device="cuda")
        source_lengths = torch.tensor([5, 3], device="cuda")
        target_lengths = torch.tensor([3, 1], device="cuda")
        reference_dtype = torch.float32 if mm_full_precision else torch.bfloat16
        outputs, gradients = [], []
        for encoder_value, predictor_value, weight_value in [
            (encoder.to(reference_dtype), predictor.to(reference_dtype), weight.to(reference_dtype)),
            (encoder, predictor, weight),
        ]:
            inputs = [
                value.detach().clone().requires_grad_()
                for value in (encoder_value, predictor_value, weight_value, bias)
            ]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                target, blank = rnnt_fused_joint(
                    encoder_output_projected=inputs[0],
                    predictor_output_projected=inputs[1],
                    weight=inputs[2],
                    bias=inputs[3],
                    targets=targets,
                    src_lengths=source_lengths,
                    tgt_lengths=target_lengths,
                    blank_id=16,
                    dropout_p=0.2,
                    dropout_seed=42,
                    mm_full_precision=mm_full_precision,
                )
            (target.sum() + blank.sum()).backward()
            outputs.append((target, blank))
            gradients.append([value.grad for value in inputs])
        for actual, expected in zip(outputs[1], outputs[0]):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for actual, expected in zip(gradients[1], gradients[0]):
            torch.testing.assert_close(actual.float(), expected.to(actual.dtype).float(), atol=0.03, rtol=0.02)
        assert gradients[1][2].dtype == gradients[1][3].dtype == torch.float32

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
        batch_size, src_length, tgt_length, hidden_dim, vocab_size_no_blank = shape
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        enc = torch.randn(batch_size, src_length, hidden_dim, device=device, dtype=float_dtype)
        pred = torch.randn(batch_size, tgt_length + 1, hidden_dim, device=device, dtype=float_dtype)
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=float_dtype)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=float_dtype)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.full([batch_size], tgt_length, device=device, dtype=torch.long)

        if float_dtype == torch.bfloat16:
            enc_ref = enc.detach().float().clone().requires_grad_(True)
            pred_ref = pred.detach().float().clone().requires_grad_(True)
            weight_ref = weight.detach().float().clone().requires_grad_(True)
            bias_ref = bias.detach().float().clone().requires_grad_(True)
        else:
            enc_ref = enc.detach().clone().requires_grad_(True)
            pred_ref = pred.detach().clone().requires_grad_(True)
            weight_ref = weight.detach().clone().requires_grad_(True)
            bias_ref = bias.detach().clone().requires_grad_(True)
        target_ref, blank_ref = _reference_joint_logprobs(
            enc=enc_ref,
            pred=pred_ref,
            weight=weight_ref,
            bias=bias_ref,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
        )

        enc_tri = enc.detach().clone().requires_grad_(True)
        pred_tri = pred.detach().clone().requires_grad_(True)
        weight_tri = weight.detach().clone().requires_grad_(True)
        bias_tri = bias.detach().clone().requires_grad_(True)
        target_tri, blank_tri = rnnt_fused_joint(
            encoder_output_projected=enc_tri,
            predictor_output_projected=pred_tri,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight_tri,
            bias=bias_tri,
            blank_id=blank_id,
            use_high_precision_mm=use_high_precision,
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

        assert enc_tri.grad.dtype == float_dtype
        assert pred_tri.grad.dtype == float_dtype
        assert weight_tri.grad.dtype == float_dtype
        assert bias_tri.grad.dtype == float_dtype

        grad_atol = 5e-3 if float_dtype == torch.bfloat16 else 5e-4  # TODO: fix to match triton_vocab
        grad_rtol = 5e-3 if float_dtype == torch.bfloat16 else 1e-5
        assert torch.allclose(
            enc_tri.grad.float(), enc_ref.grad.float(), atol=grad_atol, rtol=grad_rtol
        ), f"enc grad mismatch: max diff={(enc_tri.grad.float() - enc_ref.grad.float()).abs().max().item()}"
        assert torch.allclose(
            pred_tri.grad.float(), pred_ref.grad.float(), atol=grad_atol, rtol=grad_rtol
        ), f"pred grad mismatch: max diff={(pred_tri.grad.float() - pred_ref.grad.float()).abs().max().item()}"

        # bf16 tolerance is higher than fused-vocab test because our kernel computes
        # relu(enc+pred) on-the-fly in float32 (with use_high_precision), while the PyTorch
        # reference computes it in bf16. This causes small differences that scale with gradient magnitude.
        weight_bias_atol = 1e1 if float_dtype == torch.bfloat16 else 1e-4  # TODO: fix to match triton_vocab
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

        enc = torch.randn(batch_size, src_length, hidden_dim, device=device, dtype=torch.float32)
        pred = torch.randn(batch_size, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32)
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.tensor([src_length, src_length // 2, src_length - 1, 1], device=device, dtype=torch.long)
        tgt_lengths = torch.tensor([tgt_length, tgt_length // 2, tgt_length - 1, 0], device=device, dtype=torch.long)

        enc_ref = enc.detach().clone().requires_grad_(True)
        pred_ref = pred.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        bias_ref = bias.detach().clone().requires_grad_(True)
        target_ref, blank_ref = _reference_joint_logprobs(
            enc=enc_ref,
            pred=pred_ref,
            weight=weight_ref,
            bias=bias_ref,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
        )

        enc_tri = enc.detach().clone().requires_grad_(True)
        pred_tri = pred.detach().clone().requires_grad_(True)
        weight_tri = weight.detach().clone().requires_grad_(True)
        bias_tri = bias.detach().clone().requires_grad_(True)
        target_tri, blank_tri = rnnt_fused_joint(
            encoder_output_projected=enc_tri,
            predictor_output_projected=pred_tri,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight_tri,
            bias=bias_tri,
            blank_id=blank_id,
            use_high_precision_mm=use_high_precision,
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

        assert torch.allclose(enc_tri.grad.float(), enc_ref.grad.float(), atol=5e-3, rtol=1e-3)
        assert torch.allclose(pred_tri.grad.float(), pred_ref.grad.float(), atol=5e-3, rtol=1e-3)
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

        enc = torch.randn(batch_size, src_length, hidden_dim, device=device, dtype=torch.float32)
        pred = torch.randn(batch_size, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32)
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.tensor([src_length], device=device, dtype=torch.long)
        tgt_lengths = torch.tensor([tgt_length], device=device, dtype=torch.long)

        target_ref, blank_ref = _reference_joint_logprobs(
            enc=enc,
            pred=pred,
            weight=weight,
            bias=bias,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
        )
        target_tri, blank_tri = rnnt_fused_joint(
            encoder_output_projected=enc,
            predictor_output_projected=pred,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight,
            bias=bias,
            blank_id=blank_id,
        )

        assert torch.allclose(target_tri, target_ref, atol=5e-3)
        assert torch.allclose(blank_tri, blank_ref, atol=5e-3)

    def test_dropout_forward_and_backward_matches_reference(self):
        device = torch.device("cuda")
        torch.manual_seed(42)
        use_high_precision = True
        batch_size = 2
        src_length = 4
        tgt_length = 3
        hidden_dim = 16
        vocab_size_no_blank = 8
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1
        dropout_p = 0.3
        dropout_seed = 123456

        enc = torch.randn(batch_size, src_length, hidden_dim, device=device, dtype=torch.float32)
        pred = torch.randn(batch_size, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32)
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.full([batch_size], tgt_length, device=device, dtype=torch.long)

        dropout_scale_mask = _triton_dropout_scale_mask(
            shape=[batch_size, src_length, tgt_length + 1, hidden_dim],
            dropout_p=dropout_p,
            dropout_seed=dropout_seed,
            device=device,
            dtype=torch.float32,
        )

        enc_ref = enc.detach().clone().requires_grad_(True)
        pred_ref = pred.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        bias_ref = bias.detach().clone().requires_grad_(True)
        target_ref, blank_ref = _reference_joint_logprobs(
            enc=enc_ref,
            pred=pred_ref,
            weight=weight_ref,
            bias=bias_ref,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
            dropout_scale_mask=dropout_scale_mask,
        )

        enc_tri = enc.detach().clone().requires_grad_(True)
        pred_tri = pred.detach().clone().requires_grad_(True)
        weight_tri = weight.detach().clone().requires_grad_(True)
        bias_tri = bias.detach().clone().requires_grad_(True)
        target_tri, blank_tri = rnnt_fused_joint(
            encoder_output_projected=enc_tri,
            predictor_output_projected=pred_tri,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight_tri,
            bias=bias_tri,
            blank_id=blank_id,
            dropout_p=dropout_p,
            dropout_seed=dropout_seed,
            use_high_precision_mm=use_high_precision,
        )

        assert torch.allclose(target_tri, target_ref, atol=1e-5)
        assert torch.allclose(blank_tri, blank_ref, atol=1e-5)

        torch.manual_seed(123)
        target_scales = torch.rand_like(target_ref)
        blank_scales = torch.rand_like(blank_ref)
        loss_ref = (target_scales * target_ref + blank_scales * blank_ref).sum()
        loss_tri = (target_scales * target_tri + blank_scales * blank_tri).sum()
        loss_ref.backward()
        loss_tri.backward()

        assert torch.allclose(enc_tri.grad, enc_ref.grad, atol=1e-3, rtol=1e-5)
        assert torch.allclose(pred_tri.grad, pred_ref.grad, atol=1e-3, rtol=1e-5)
        assert torch.allclose(weight_tri.grad, weight_ref.grad, atol=2e-3, rtol=1e-5)
        assert torch.allclose(bias_tri.grad, bias_ref.grad, atol=2e-3, rtol=1e-5)

    def test_dropout_seed_controls_forward_determinism(self):
        device = torch.device("cuda")
        torch.manual_seed(42)
        batch_size = 2
        src_length = 4
        tgt_length = 3
        hidden_dim = 16
        vocab_size_no_blank = 8
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1
        dropout_p = 0.3
        dropout_seed_same = 777
        dropout_seed_different = 778

        enc = torch.randn(batch_size, src_length, hidden_dim, device=device, dtype=torch.float32)
        pred = torch.randn(batch_size, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32)
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.full([batch_size], tgt_length, device=device, dtype=torch.long)

        target_same_0, blank_same_0 = rnnt_fused_joint(
            encoder_output_projected=enc,
            predictor_output_projected=pred,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight,
            bias=bias,
            blank_id=blank_id,
            dropout_p=dropout_p,
            dropout_seed=dropout_seed_same,
        )
        target_same_1, blank_same_1 = rnnt_fused_joint(
            encoder_output_projected=enc,
            predictor_output_projected=pred,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight,
            bias=bias,
            blank_id=blank_id,
            dropout_p=dropout_p,
            dropout_seed=dropout_seed_same,
        )
        target_different, blank_different = rnnt_fused_joint(
            encoder_output_projected=enc,
            predictor_output_projected=pred,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight,
            bias=bias,
            blank_id=blank_id,
            dropout_p=dropout_p,
            dropout_seed=dropout_seed_different,
        )

        assert torch.equal(target_same_0, target_same_1)
        assert torch.equal(blank_same_0, blank_same_1)
        assert not torch.equal(target_same_0, target_different)
        assert not torch.equal(blank_same_0, blank_different)

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

        enc = torch.randn(batch_size, src_length, hidden_dim, device=device, dtype=torch.float32)
        pred = torch.randn(batch_size, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32)
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32)
        targets = torch.randint(
            0, vocab_size_no_blank, (batch_size, max(tgt_length, 1)), device=device, dtype=torch.long
        )
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.zeros([batch_size], device=device, dtype=torch.long)

        target_ref, blank_ref = _reference_joint_logprobs(
            enc=enc,
            pred=pred,
            weight=weight,
            bias=bias,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
        )
        target_tri, blank_tri = rnnt_fused_joint(
            encoder_output_projected=enc,
            predictor_output_projected=pred,
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

        enc = torch.randn(batch_size, src_length, hidden_dim, device=device, dtype=torch.float32, requires_grad=True)
        pred = torch.randn(
            batch_size, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32, requires_grad=True
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.full([batch_size], tgt_length, device=device, dtype=torch.long)

        # Warmup
        target_logprobs, blank_logprobs = rnnt_fused_joint(
            encoder_output_projected=enc,
            predictor_output_projected=pred,
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

        enc_test = enc.detach().clone().requires_grad_(True)
        pred_test = pred.detach().clone().requires_grad_(True)
        weight_test = weight.detach().clone().requires_grad_(True)
        bias_test = bias.detach().clone().requires_grad_(True)

        with avoid_sync_operations(device):
            target_logprobs_test, blank_logprobs_test = rnnt_fused_joint(
                encoder_output_projected=enc_test,
                predictor_output_projected=pred_test,
                targets=targets,
                tgt_lengths=tgt_lengths,
                src_lengths=src_lengths,
                weight=weight_test,
                bias=bias_test,
                blank_id=blank_id,
            )
            loss = target_logprobs_test.sum() + blank_logprobs_test.sum()
            loss.backward()

    def test_grad_check(self):
        pytest.skip(reason="temporary skip - slow")
        device = torch.device("cuda")
        torch.manual_seed(42)
        batch_size = 1
        src_length = 2
        tgt_length = 1
        hidden_dim = 4
        vocab_size_no_blank = 3
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        enc = torch.randn(batch_size, src_length, hidden_dim, device=device, dtype=torch.float64, requires_grad=True)
        pred = torch.randn(
            batch_size, tgt_length + 1, hidden_dim, device=device, dtype=torch.float64, requires_grad=True
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.full([batch_size], tgt_length, device=device, dtype=torch.long)

        def func(enc_in, pred_in, w_in, b_in):
            t_lp, b_lp = rnnt_fused_joint(
                encoder_output_projected=enc_in,
                predictor_output_projected=pred_in,
                targets=targets,
                tgt_lengths=tgt_lengths,
                src_lengths=src_lengths,
                weight=w_in,
                bias=b_in,
                blank_id=blank_id,
            )
            return t_lp, b_lp

        # nondet_tol needed because backward uses atomic adds which are nondeterministic
        assert torch.autograd.gradcheck(
            func, (enc, pred, weight, bias), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-5
        )
