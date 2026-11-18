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

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab import rnnt_joint_fused_vocab
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import rnnt_logprobs_torch
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.decoding.utils import avoid_sync_operations

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer import rnnt_joint_fused_vocab_triton as kernel_module
    from nemo.collections.asr.parts.turbo_transducer.rnnt_loss_triton import rnnt_loss_from_logprobs_triton


@pytest.mark.unit
def test_training_supplies_optimizer_step():
    encoded, decoder = torch.randn(2, 4, 3), torch.randn(2, 4, 4)
    src_lengths, tgt_lengths = torch.tensor([3, 2]), torch.tensor([3, 2])
    transcript = torch.tensor([[1, 2, 3], [1, 2, 0]])
    model = SimpleNamespace(
        model_guid='pruning-step-test',
        use_double_batch=False,
        consistency_loss=None,
        global_step=7,
        _trainer=SimpleNamespace(log_every_n_steps=100, global_step=7),
        is_interctc_enabled=lambda: False,
        forward=Mock(return_value=(encoded, src_lengths)),
        decoder=Mock(return_value=(decoder, tgt_lengths, None)),
        joint=SimpleNamespace(fuse_loss_wer=False, uses_turbo_kernel=True),
        _compute_losses=Mock(side_effect=RuntimeError('stop at loss boundary')),
    )
    with pytest.raises(RuntimeError, match='stop at loss boundary'):
        EncDecRNNTModel.training_step(
            model, (torch.randn(2, 1600), src_lengths, transcript, tgt_lengths), batch_nb=123
        )
    assert model._compute_losses.call_args.kwargs['current_step'] == 7


@pytest.mark.unit
@pytest.mark.skipif(not TRITON_AVAILABLE, reason='Triton is unavailable')
@pytest.mark.parametrize('width', [True, 1.5])
def test_invalid_low_level_pruning_width(width):
    with pytest.raises(ValueError, match='integer'):
        kernel_module.rnnt_joint_fused_vocab_triton(
            torch.empty(1, 2, 3, 4),
            torch.empty(1, 2),
            torch.tensor([2]),
            torch.tensor([2]),
            torch.empty(5, 4),
            torch.empty(5),
            4,
            backward_pruning_range_size=width,
        )


@pytest.mark.unit
def test_cutile_rejects_pruning():
    with pytest.raises(NotImplementedError, match='Triton'):
        rnnt_joint_fused_vocab(
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            backend='cutile',
            backward_pruning_range_size=2,
        )


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason='CUDA and Triton required')
class TestPrunedFusedVocab:
    @pytest.mark.parametrize('mode', ['atomic', 'chunked'])
    @pytest.mark.parametrize('mm_full_precision', [False, True])
    def test_default_grad_precision_matches_masked_dense_reference(self, mode, mm_full_precision):
        torch.manual_seed(29)
        hidden = torch.randn(3, 11, 7, 35, device='cuda', dtype=torch.bfloat16) * 0.1
        weight = torch.randn(65, 35, device='cuda') * 0.1
        bias = torch.randn(65, device='cuda') * 0.1
        targets = torch.randint(0, 64, (3, 6), device='cuda')
        src_lengths, tgt_lengths = torch.tensor([11, 1, 7], device='cuda'), torch.tensor([6, 0, 2], device='cuda')
        grad_target, grad_blank = torch.randn(3, 11, 7, device='cuda'), torch.randn(3, 11, 7, device='cuda')
        inputs = [value.detach().clone().requires_grad_() for value in (hidden, weight, bias)]
        reference = [value.detach().float().requires_grad_() for value in (hidden, weight, bias)]
        ref_weight = reference[1]
        if not mm_full_precision:
            ref_weight = ref_weight + (ref_weight.bfloat16().float() - ref_weight).detach()
        target_ref, blank_ref = rnnt_logprobs_torch(
            F.linear(reference[0], ref_weight, reference[2]), targets, 64, src_lengths, tgt_lengths
        )
        target, blank = kernel_module.rnnt_joint_fused_vocab_triton(
            inputs[0],
            targets,
            tgt_lengths,
            src_lengths,
            inputs[1],
            inputs[2],
            64,
            backward_mode=mode,
            mm_full_precision=mm_full_precision,
            backward_pruning_range_size=3,
        )
        _, mask = _reference_windows(grad_target, grad_blank, src_lengths, tgt_lengths, 3)
        torch.autograd.backward((target_ref, blank_ref), (grad_target * mask, grad_blank * mask))
        torch.autograd.backward((target, blank), (grad_target, grad_blank))
        for actual, expected in zip((target, blank), (target_ref, blank_ref)):
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
        for actual, expected in zip(inputs, reference):
            torch.testing.assert_close(actual.grad.float(), expected.grad, atol=0.02, rtol=0.03)

    @pytest.mark.parametrize('mode', ['atomic', 'chunked'])
    def test_window_boundary_still_emits_target(self, mode):
        torch.manual_seed(11)
        hidden = torch.randn(1, 3, 6, 32, device='cuda') * 0.1
        weight = torch.randn(17, 32, device='cuda') * 0.1
        bias = torch.randn(17, device='cuda') * 0.1
        targets = torch.tensor([[1, 2, 3, 4, 5]], device='cuda')
        src_lengths, tgt_lengths = torch.tensor([3], device='cuda'), torch.tensor([5], device='cuda')
        grad_target = torch.zeros(1, 3, 6, device='cuda')
        grad_target[:, :, 2:4] = torch.tensor([1.0, 3.0], device='cuda')
        gradients = []
        for pruned in [False, True]:
            inputs = [value.detach().clone().requires_grad_() for value in (hidden, weight, bias)]
            if pruned:
                target, blank = kernel_module.rnnt_joint_fused_vocab_triton(
                    inputs[0],
                    targets,
                    tgt_lengths,
                    src_lengths,
                    inputs[1],
                    inputs[2],
                    16,
                    use_high_precision_mm=True,
                    input_grad_full_precision=True,
                    weight_grad_full_precision=True,
                    backward_mode=mode,
                    backward_pruning_range_size=2,
                )
            else:
                target, blank = rnnt_logprobs_torch(
                    F.linear(inputs[0], inputs[1], inputs[2]), targets, 16, src_lengths, tgt_lengths
                )
            torch.autograd.backward((target, blank), (grad_target, torch.zeros_like(grad_target)))
            gradients.append([value.grad for value in inputs])
        for actual, expected in zip(gradients[1], gradients[0]):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize('width', [1, 4, 18])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
    def test_window_selection(self, width, dtype):
        torch.manual_seed(123)
        grad_target = torch.randn(3, 7, 18, device='cuda', dtype=dtype)
        grad_blank = torch.randn_like(grad_target)
        src_lengths = torch.tensor([7, 1, 4], device='cuda')
        tgt_lengths = torch.tensor([17, 0, 2], device='cuda')
        expected_starts, _ = _reference_windows(grad_target, grad_blank, src_lengths, tgt_lengths, width)
        for scale in [1.0, 1024.0, -0.5]:
            starts = kernel_module._select_pruning_starts(
                grad_target * scale, grad_blank * scale, src_lengths, tgt_lengths, width
            )
            torch.testing.assert_close(starts, expected_starts)
        starts = kernel_module._select_pruning_starts(
            torch.zeros_like(grad_target), torch.zeros_like(grad_blank), src_lengths, tgt_lengths, width
        )
        assert torch.count_nonzero(starts) == 0

    @pytest.mark.parametrize('mode', ['atomic', 'chunked'])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize('width', [1, 3, 4, 18, 30])
    def test_backward_matches_masked_dense_reference(self, monkeypatch, mode, dtype, width):
        torch.manual_seed(17)
        hidden = torch.randn(3, 33, 18, 64, device='cuda', dtype=dtype) * 0.15
        weight = torch.randn(67, 64, device='cuda') * 0.15
        bias = torch.randn(67, device='cuda') * 0.1
        targets = torch.randint(1, 67, (3, 17), device='cuda')  # blank_id=0
        src_lengths = torch.tensor([33, 1, 20], device='cuda')
        tgt_lengths = torch.tensor([17, 0, 2], device='cuda')
        # Force multiple chunks and a partial last chunk for pruned as well as dense backward.
        monkeypatch.setattr(kernel_module, 'CHUNKED_BWD_GRAD_LOGITS_BUFFER_BYTES', 256 * 128 * dtype.itemsize)
        inputs = [value.detach().clone().requires_grad_() for value in (hidden, weight, bias)]
        reference = [value.detach().float().requires_grad_() for value in (hidden, weight, bias)]
        # Match BF16 forward operands while retaining FP32 master-parameter gradient accumulation.
        ref_weight = reference[1] + (reference[1].to(dtype).float() - reference[1]).detach()
        target_ref, blank_ref = rnnt_logprobs_torch(
            F.linear(reference[0], ref_weight, reference[2]), targets, 0, src_lengths, tgt_lengths
        )
        target, blank = kernel_module.rnnt_joint_fused_vocab_triton(
            inputs[0],
            targets,
            tgt_lengths,
            src_lengths,
            inputs[1],
            inputs[2],
            0,
            use_high_precision_mm=dtype == torch.float32,
            input_grad_full_precision=True,
            weight_grad_full_precision=True,
            backward_mode=mode,
            backward_pruning_range_size=width,
        )
        tolerance = 1e-5 if dtype == torch.float32 else 0.025
        torch.testing.assert_close(target, target_ref, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(blank, blank_ref, atol=tolerance, rtol=tolerance)
        grad_target, grad_blank = torch.randn_like(target), torch.randn_like(blank)
        _, mask = _reference_windows(grad_target, grad_blank, src_lengths, tgt_lengths, width)
        torch.autograd.backward((target_ref, blank_ref), (grad_target * mask, grad_blank * mask))
        torch.autograd.backward((target, blank), (grad_target, grad_blank))
        for actual, expected in zip(inputs, reference):
            torch.testing.assert_close(actual.grad.float(), expected.grad, atol=tolerance * 4, rtol=tolerance * 2)
        assert torch.count_nonzero(inputs[0].grad[~mask]) == 0

    @pytest.mark.parametrize('mode', ['atomic', 'chunked'])
    @pytest.mark.parametrize('fastemit_lambda', [0.0, 0.1])
    def test_full_loss_and_upstream_gradients(self, mode, fastemit_lambda):
        torch.manual_seed(123)
        src_lengths = torch.tensor([6, 1, 3], device='cuda')
        tgt_lengths = torch.tensor([4, 2, 0], device='cuda')
        targets = torch.randint(0, 16, (3, 4), device='cuda')
        values = [torch.randn(*shape, device='cuda') * 0.1 for shape in [(3, 6, 32), (3, 5, 32), (17, 32), (17,)]]
        reference = [value.detach().clone().requires_grad_() for value in values]
        actual = [value.detach().clone().requires_grad_() for value in values]
        losses = []
        for inputs, pruned in [(reference, False), (actual, True)]:
            torch.manual_seed(9)  # Identical dropout realization for the two forwards.
            hidden = F.dropout(F.relu(inputs[0][:, :, None] + inputs[1][:, None]), p=0.2, training=True)
            if pruned:
                target, blank = kernel_module.rnnt_joint_fused_vocab_triton(
                    hidden,
                    targets,
                    tgt_lengths,
                    src_lengths,
                    inputs[2],
                    inputs[3],
                    16,
                    use_high_precision_mm=True,
                    input_grad_full_precision=True,
                    weight_grad_full_precision=True,
                    backward_mode=mode,
                    backward_pruning_range_size=3,
                )
            else:
                target, blank = rnnt_logprobs_torch(
                    F.linear(hidden, inputs[2], inputs[3]), targets, 16, src_lengths, tgt_lengths
                )
            loss = rnnt_loss_from_logprobs_triton(target, blank, src_lengths, tgt_lengths, fastemit_lambda).sum()
            losses.append(loss.detach())
            if pruned:
                loss.backward()
            else:
                grad_target, grad_blank = torch.autograd.grad(loss, (target, blank), retain_graph=True)
                _, mask = _reference_windows(grad_target, grad_blank, src_lengths, tgt_lengths, 3)
                torch.autograd.backward((target, blank), (grad_target * mask, grad_blank * mask))
        torch.testing.assert_close(losses[0], losses[1])
        for actual_input, reference_input in zip(actual, reference):
            torch.testing.assert_close(actual_input.grad, reference_input.grad, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize('mode', ['atomic', 'chunked'])
    def test_no_cuda_synchronization(self, mode):
        hidden = torch.randn(2, 5, 6, 32, device='cuda', requires_grad=True)
        weight = torch.randn(17, 32, device='cuda', requires_grad=True)
        bias = torch.randn(17, device='cuda', requires_grad=True)
        targets = torch.randint(0, 16, (2, 5), device='cuda')
        src_lengths = torch.tensor([5, 3], device='cuda')
        tgt_lengths = torch.tensor([5, 0], device='cuda')
        for warmup in [True, False]:
            with contextlib.nullcontext() if warmup else avoid_sync_operations(torch.device('cuda')):
                target, blank = kernel_module.rnnt_joint_fused_vocab_triton(
                    hidden,
                    targets,
                    tgt_lengths,
                    src_lengths,
                    weight,
                    bias,
                    16,
                    backward_mode=mode,
                    backward_pruning_range_size=2,
                )
                (target.sum() + blank.sum()).backward()


def _reference_windows(grad_target, grad_blank, src_lengths, tgt_lengths, width):
    batch_size, max_src_len, max_tgt_len_plus_1 = grad_blank.shape
    width = min(width, max_tgt_len_plus_1)
    importance = grad_target.abs() + grad_blank.abs()
    starts = torch.zeros((batch_size, max_src_len), dtype=torch.int32, device=grad_blank.device)
    mask = torch.zeros_like(grad_blank, dtype=torch.bool)
    for batch_index in range(batch_size):
        src_length = int(src_lengths[batch_index])
        tgt_length = int(tgt_lengths[batch_index])
        importance[batch_index, :, tgt_length] = grad_blank[batch_index, :, tgt_length].abs()
        importance[batch_index, :, tgt_length + 1 :] = 0
        effective_width = min(width, tgt_length + 1)
        for src_index in range(src_length):
            windows = importance[batch_index, src_index, : tgt_length + 1].unfold(0, effective_width, 1)
            start = int(windows.sum(-1).argmax())
            starts[batch_index, src_index] = start
            mask[batch_index, src_index, start : start + effective_width] = True
    return starts, mask
