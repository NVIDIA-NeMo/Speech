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

"""Tests for the `fused_vocab` Joint computing the consistency loss inside the Triton kernels."""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency import get_rnnt_mask, kl_loss_torch
from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab import rnnt_joint_fused_vocab_with_consistency
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import rnnt_logprobs_torch
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.decoding.utils import avoid_sync_operations

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer import (
        rnnt_joint_fused_vocab_consistency_triton as consistency_kernel_module,
    )
    from nemo.collections.asr.parts.turbo_transducer import rnnt_joint_fused_vocab_triton as kernel_module

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA and Triton required"
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _double_batch_inputs(
    half_batch_size: int,
    max_src_len: int,
    max_tgt_len: int,
    hidden_dim: int,
    vocab_size: int,
    blank_id: int,
    src_lengths: list[int],
    tgt_lengths: list[int],
    dtype: torch.dtype,
    seed: int = 0,
):
    """Teacher and student halves with correlated hidden states, identical targets and lengths"""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    teacher_hidden = torch.randn(
        half_batch_size, max_src_len, max_tgt_len + 1, hidden_dim, device=DEVICE, generator=generator
    )
    student_hidden = teacher_hidden + 0.3 * torch.randn(teacher_hidden.shape, device=DEVICE, generator=generator)
    joint_hidden = (torch.cat([teacher_hidden, student_hidden]) * 0.2).to(dtype)
    weight = torch.randn(vocab_size, hidden_dim, device=DEVICE, generator=generator) * 0.2
    bias = torch.randn(vocab_size, device=DEVICE, generator=generator) * 0.1
    non_blank_labels = torch.tensor([label for label in range(vocab_size) if label != blank_id], device=DEVICE)
    targets = non_blank_labels[
        torch.randint(0, vocab_size - 1, (half_batch_size, max_tgt_len), device=DEVICE, generator=generator)
    ].repeat(2, 1)
    src_lengths = torch.tensor(src_lengths, device=DEVICE).repeat(2)
    tgt_lengths = torch.tensor(tgt_lengths, device=DEVICE).repeat(2)
    return joint_hidden, weight, bias, targets, src_lengths, tgt_lengths


def _reference(joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, blank_id, symmetrical, dtype):
    """Full logits in FP32 (BF16-rounded weight for BF16 operands) + Torch log-probs and KL"""
    weight = weight + (weight.to(dtype).float() - weight).detach()
    logits = F.linear(joint_hidden.float(), weight, bias)
    target_logprobs, blank_logprobs = rnnt_logprobs_torch(logits, targets, blank_id, src_lengths, tgt_lengths)
    half_batch_size, max_src_len, max_tgt_len_plus_1 = logits.shape[0] // 2, logits.shape[1], logits.shape[2]
    _, consistency_mask = get_rnnt_mask(
        batch_size=half_batch_size,
        src_length_max=max_src_len,
        tgt_length_max_plus_1=max_tgt_len_plus_1,
        device=logits.device,
        src_lengths=src_lengths[:half_batch_size],
        tgt_lengths=tgt_lengths[:half_batch_size],
    )
    consistency_kl = kl_loss_torch(
        teacher_logprobs=F.log_softmax(logits[:half_batch_size], dim=-1),
        student_logprobs=F.log_softmax(logits[half_batch_size:], dim=-1),
        mask=consistency_mask,
        symmetrical=symmetrical,
    )
    return target_logprobs, blank_logprobs, consistency_kl


def _reference_paired_windows(grad_target, grad_blank, src_lengths, tgt_lengths, width):
    """Windows maximizing summed teacher + student importance; mask for the double batch [2B, T, U+1]"""
    half_batch_size, max_src_len, max_tgt_len_plus_1 = grad_blank.shape[0] // 2, *grad_blank.shape[1:]
    target_positions = torch.arange(max_tgt_len_plus_1, device=grad_blank.device)
    target_valid = target_positions[None, :] < tgt_lengths[:, None]
    blank_valid = target_positions[None, :] <= tgt_lengths[:, None]
    importance = grad_target.abs() * target_valid[:, None, :] + grad_blank.abs() * blank_valid[:, None, :]
    importance = importance[:half_batch_size] + importance[half_batch_size:]
    starts = torch.zeros((half_batch_size, max_src_len), dtype=torch.int32, device=grad_blank.device)
    mask = torch.zeros((half_batch_size, max_src_len, max_tgt_len_plus_1), dtype=torch.bool, device=grad_blank.device)
    for batch_index in range(half_batch_size):
        src_length = int(src_lengths[batch_index])
        tgt_length = int(tgt_lengths[batch_index])
        effective_width = min(width, tgt_length + 1)
        for src_index in range(src_length):
            windows = importance[batch_index, src_index, : tgt_length + 1].unfold(0, effective_width, 1)
            start = int(windows.sum(-1).argmax())
            starts[batch_index, src_index] = start
            mask[batch_index, src_index, start : start + effective_width] = True
    return starts, mask


def _run_and_compare(
    monkeypatch, mode, dtype, symmetrical, vocab_size, blank_id, pruning_range_size, seed, small_chunks=True
):
    if small_chunks:
        # Force multiple chunks and a partial last chunk in the chunked backward.
        monkeypatch.setattr(
            consistency_kernel_module, "CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES", 2 * 256 * 128 * dtype.itemsize
        )
    joint_hidden, weight, bias, targets, src_lengths, tgt_lengths = _double_batch_inputs(
        half_batch_size=3,
        max_src_len=21,
        max_tgt_len=9,
        hidden_dim=48,
        vocab_size=vocab_size,
        blank_id=blank_id,
        src_lengths=[21, 1, 13],
        tgt_lengths=[9, 0, 3],
        dtype=dtype,
        seed=seed,
    )
    inputs = [value.detach().clone().requires_grad_() for value in (joint_hidden, weight, bias)]
    reference_inputs = [value.detach().float().requires_grad_() for value in (joint_hidden, weight, bias)]
    outputs = rnnt_joint_fused_vocab_with_consistency(
        *inputs[:1],
        targets=targets,
        tgt_lengths=tgt_lengths,
        src_lengths=src_lengths,
        weight=inputs[1],
        bias=inputs[2],
        blank_id=blank_id,
        consistency_symmetrical=symmetrical,
        use_high_precision_mm=dtype == torch.float32,
        input_grad_full_precision=True,
        weight_grad_full_precision=True,
        backward_mode=mode,
        backward_pruning_range_size=pruning_range_size,
    )
    reference_outputs = _reference(
        *reference_inputs[:1],
        reference_inputs[1],
        reference_inputs[2],
        targets,
        src_lengths,
        tgt_lengths,
        blank_id,
        symmetrical,
        dtype,
    )
    tolerance = 1e-5 if dtype == torch.float32 else 0.02
    for actual, expected in zip(outputs, reference_outputs):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    upstream_grads = [torch.randn_like(output) for output in reference_outputs]
    reference_upstream_grads = upstream_grads
    pruning_mask = None
    if pruning_range_size is not None:
        _, pruning_mask = _reference_paired_windows(
            upstream_grads[0], upstream_grads[1], src_lengths, tgt_lengths, pruning_range_size
        )
        double_batch_mask = pruning_mask.repeat(2, 1, 1)
        reference_upstream_grads = [
            upstream_grads[0] * double_batch_mask,
            upstream_grads[1] * double_batch_mask,
            upstream_grads[2] * pruning_mask,
        ]
    torch.autograd.backward(reference_outputs, reference_upstream_grads)
    torch.autograd.backward(tuple(outputs), upstream_grads)
    for actual, expected in zip(inputs, reference_inputs):
        torch.testing.assert_close(actual.grad.float(), expected.grad, atol=tolerance * 4, rtol=tolerance * 2)
    if pruning_mask is not None:
        assert torch.count_nonzero(inputs[0].grad[~pruning_mask.repeat(2, 1, 1)]) == 0


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["atomic", "chunked"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("vocab_size,blank_id", [(33, 32), (130, 0), (257, 256)])
def test_matches_logits_reference(monkeypatch, mode, dtype, symmetrical, vocab_size, blank_id):
    _run_and_compare(
        monkeypatch,
        mode=mode,
        dtype=dtype,
        symmetrical=symmetrical,
        vocab_size=vocab_size,
        blank_id=blank_id,
        pruning_range_size=None,
        seed=vocab_size,
    )


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["atomic", "chunked"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("width", [1, 3, 4, 9])
def test_pruned_backward_matches_masked_dense_reference(monkeypatch, mode, dtype, symmetrical, width):
    """Pruned backward: shared teacher/student windows, consistency gradient restricted to the windows too"""
    _run_and_compare(
        monkeypatch,
        mode=mode,
        dtype=dtype,
        symmetrical=symmetrical,
        vocab_size=67,
        blank_id=66,
        pruning_range_size=width,
        seed=width,
    )


@pytest.mark.unit
@pytest.mark.parametrize("width", [1, 3, 5])
def test_paired_window_selection(width):
    torch.manual_seed(width)
    grad_target = torch.randn(6, 7, 9, device=DEVICE)
    grad_blank = torch.randn(6, 7, 9, device=DEVICE)
    src_lengths = torch.tensor([7, 1, 4], device=DEVICE).repeat(2)
    tgt_lengths = torch.tensor([8, 0, 3], device=DEVICE).repeat(2)
    starts = kernel_module._select_pruning_starts(
        grad_target, grad_blank, src_lengths, tgt_lengths, width, paired=True
    )
    expected_starts, _ = _reference_paired_windows(grad_target, grad_blank, src_lengths, tgt_lengths, width)
    assert starts.shape == (3, 7)
    for batch_index, src_length in enumerate(src_lengths[:3].tolist()):
        assert torch.equal(starts[batch_index, :src_length], expected_starts[batch_index, :src_length])


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["atomic", "chunked"])
def test_consistency_values_without_upstream_grad_match_rnnt_only_kernel(mode):
    """Target/blank log-probs and their gradients are identical to the kernel without consistency"""
    joint_hidden, weight, bias, targets, src_lengths, tgt_lengths = _double_batch_inputs(
        2, 11, 5, 32, 65, 64, [11, 6], [5, 2], torch.float32, seed=3
    )
    results = []
    for with_consistency in [False, True]:
        inputs = [value.detach().clone().requires_grad_() for value in (joint_hidden, weight, bias)]
        kwargs = dict(use_high_precision_mm=True, backward_mode=mode, weight_grad_full_precision=True)
        if with_consistency:
            target, blank, _ = consistency_kernel_module.rnnt_joint_fused_vocab_consistency_triton(
                inputs[0], targets, tgt_lengths, src_lengths, inputs[1], inputs[2], 64, **kwargs
            )
        else:
            target, blank = kernel_module.rnnt_joint_fused_vocab_triton(
                inputs[0], targets, tgt_lengths, src_lengths, inputs[1], inputs[2], 64, **kwargs
            )
        (target.sum() + blank.sum()).backward()
        results.append([target, blank] + [value.grad for value in inputs])
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["atomic", "chunked"])
@pytest.mark.parametrize("pruning_range_size", [None, 2])
def test_no_cuda_synchronization(mode, pruning_range_size):
    joint_hidden, weight, bias, targets, src_lengths, tgt_lengths = _double_batch_inputs(
        2, 5, 4, 32, 17, 16, [5, 3], [4, 0], torch.float32
    )
    joint_hidden.requires_grad_()
    weight.requires_grad_()
    bias.requires_grad_()
    for warmup in [True, False]:
        with contextlib.nullcontext() if warmup else avoid_sync_operations(DEVICE):
            outputs = rnnt_joint_fused_vocab_with_consistency(
                joint_hidden,
                targets,
                tgt_lengths,
                src_lengths,
                weight,
                bias,
                16,
                backward_mode=mode,
                backward_pruning_range_size=pruning_range_size,
            )
            sum(output.sum() for output in outputs).backward()


@pytest.mark.unit
def test_rejects_odd_batch():
    joint_hidden, weight, bias, targets, src_lengths, tgt_lengths = _double_batch_inputs(
        1, 3, 2, 16, 9, 8, [3], [2], torch.float32
    )
    with pytest.raises(ValueError, match="double batch"):
        rnnt_joint_fused_vocab_with_consistency(
            joint_hidden[:1], targets[:1], tgt_lengths[:1], src_lengths[:1], weight, bias, 8
        )


@pytest.mark.unit
def test_cutile_backend_is_not_implemented():
    joint_hidden, weight, bias, targets, src_lengths, tgt_lengths = _double_batch_inputs(
        1, 3, 2, 16, 9, 8, [3], [2], torch.float32
    )
    with pytest.raises(NotImplementedError, match="cuTile"):
        rnnt_joint_fused_vocab_with_consistency(
            joint_hidden, targets, tgt_lengths, src_lengths, weight, bias, 8, backend="cutile"
        )


@pytest.mark.unit
@pytest.mark.parametrize("symmetrical", [True, False])
@pytest.mark.parametrize("pruning_range_size", [None, 2])
def test_joint_forward_for_loss_matches_materialized_logits_path(symmetrical, pruning_range_size):
    """`RNNTJoint.forward_for_loss` with `fused_vocab` matches the Joint without turbo kernel (logits path)"""
    torch.manual_seed(5)
    joint_config = dict(
        jointnet={"encoder_hidden": 12, "pred_hidden": 10, "joint_hidden": 32, "activation": "relu", "dropout": 0.0},
        num_classes=20,
        log_softmax=False,
    )
    reference_joint = RNNTJoint(**joint_config).to(DEVICE).train()
    turbo_joint = RNNTJoint(
        **joint_config,
        turbo_kernel={
            "type": "fused_vocab",
            "kwargs": {
                "backend": "triton",
                "use_high_precision_mm": True,
                "input_grad_full_precision": True,
                "weight_grad_full_precision": True,
            },
            "backward_pruning_range_size": pruning_range_size,
        },
    ).to(DEVICE)
    turbo_joint.load_state_dict(reference_joint.state_dict())
    turbo_joint.train()

    half_batch_size, num_frames, num_tokens = 2, 7, 4
    encoder_outputs = torch.randn(half_batch_size, 12, num_frames, device=DEVICE).repeat(2, 1, 1)
    encoder_outputs = encoder_outputs + 0.1 * torch.randn_like(encoder_outputs)
    decoder_outputs = torch.randn(half_batch_size, 10, num_tokens + 1, device=DEVICE).repeat(2, 1, 1)
    transcripts = torch.randint(0, 20, (half_batch_size, num_tokens), device=DEVICE).repeat(2, 1)
    encoder_lengths = torch.tensor([num_frames, 5], device=DEVICE).repeat(2)
    transcript_lengths = torch.tensor([num_tokens, 2], device=DEVICE).repeat(2)

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
            (list(loss_inputs), [encoder_input.grad, decoder_input.grad] + [p.grad for p in joint.parameters()])
        )
    for actual, expected in zip(results[1][0], results[0][0]):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    if pruning_range_size is None:
        for actual, expected in zip(results[1][1], results[0][1]):
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
