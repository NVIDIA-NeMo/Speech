# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from nemo.collections.asr.losses import ctc as ctc_module
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.parts.packed_sequence import pack_encoder_output
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

CUDA_TRITON_AVAILABLE = TRITON_AVAILABLE and torch.cuda.is_available()


class TestPackedCTCLoss:
    """`forward_packed` against `forward`: loss and gradient parity, edge cases and input checks."""

    @pytest.mark.unit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum", "mean_batch", "mean_volume"])
    def test_loss_and_gradient_match_padded(self, reduction):
        torch.manual_seed(7)
        lengths = torch.tensor([8, 5, 2])
        target_lengths = torch.tensor([3, 1, 0])
        targets = torch.tensor([[1, 1, 2], [2, 0, 0], [0, 0, 0]])
        dense_logits = torch.randn(3, 11, 5, dtype=torch.float64, requires_grad=True)
        packed_logits = dense_logits.detach().clone().requires_grad_()
        criterion = CTCLoss(num_classes=4, reduction=reduction)
        expected = criterion(
            log_probs=dense_logits.log_softmax(-1),
            targets=targets,
            input_lengths=lengths,
            target_lengths=target_lengths,
        )
        packed = pack_encoder_output(packed_logits.log_softmax(-1), lengths)
        actual = criterion.forward_packed(packed, targets, target_lengths)
        torch.testing.assert_close(actual, expected)
        expected.sum().backward()
        actual.sum().backward()
        torch.testing.assert_close(packed_logits.grad, dense_logits.grad)

    @pytest.mark.unit
    @pytest.mark.parametrize("zero_infinity", [False, True])
    def test_impossible_alignment(self, zero_infinity):
        logits = torch.tensor([[[1.0, 0.5, -0.5]]], requires_grad=True)
        lengths = torch.tensor([1])
        targets = torch.tensor([[0, 0]])
        target_lengths = torch.tensor([2])
        criterion = CTCLoss(num_classes=2, zero_infinity=zero_infinity)
        packed = pack_encoder_output(logits.log_softmax(-1), lengths)
        actual = criterion.forward_packed(packed, targets, target_lengths)
        if zero_infinity:
            assert actual.item() == 0.0
            actual.backward()
            torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        else:
            assert torch.isposinf(actual)

    @pytest.mark.unit
    def test_rejects_padded_log_probs(self):
        criterion = CTCLoss(num_classes=2)
        with pytest.raises(TypeError, match="PackedEncoderActivations"):
            criterion.forward_packed(torch.zeros(1, 3, 3), torch.zeros(1, 1, dtype=torch.long), torch.tensor([1]))


@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="the packed CTC loss kernel needs CUDA and Triton")
class TestPackedCTCLossKernel:
    """On CUDA `forward_packed` runs the Triton kernel over the token-flat log probabilities:
    loss and gradient parity with the padded `forward`, the edge cases of the recursion, dtypes."""

    @staticmethod
    def _compare(criterion, logits, lengths, targets, target_lengths, *, atol=1e-5, rtol=1e-5, autocast=False):
        """Loss and gradient of `forward_packed` (the kernel) against `forward` (padded) on the same logits."""
        dense = logits.detach().clone().requires_grad_()
        flat = logits.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            expected = criterion(
                log_probs=dense.log_softmax(-1), targets=targets, input_lengths=lengths, target_lengths=target_lengths
            )
            packed = pack_encoder_output(flat.log_softmax(-1), lengths)
            actual = criterion.forward_packed(packed, targets, target_lengths)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        expected.sum().backward()
        actual.sum().backward()
        torch.testing.assert_close(flat.grad, dense.grad, atol=atol, rtol=rtol)
        return actual

    @staticmethod
    def _small_case(device):
        torch.manual_seed(7)
        lengths = torch.tensor([8, 5, 2], device=device)
        target_lengths = torch.tensor([3, 1, 0], device=device)
        targets = torch.tensor([[1, 1, 2], [2, 0, 0], [0, 0, 0]], device=device)
        logits = torch.randn(3, 11, 5, device=device)
        return logits, lengths, targets, target_lengths

    @pytest.mark.unit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum", "mean_batch", "mean_volume"])
    def test_loss_and_gradient_match_padded(self, reduction):
        logits, lengths, targets, target_lengths = self._small_case("cuda")
        criterion = CTCLoss(num_classes=4, reduction=reduction)
        self._compare(criterion, logits, lengths, targets, target_lengths)

    @pytest.mark.unit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum", "mean_batch", "mean_volume"])
    @pytest.mark.parametrize("on_cpu", ["targets", "target_lengths", "both"])
    def test_cpu_target_metadata(self, reduction, on_cpu):
        # the targets and their lengths may stay on the CPU, as forward accepts them
        logits, lengths, targets, target_lengths = self._small_case("cuda")
        if on_cpu != "target_lengths":
            targets = targets.cpu()
        if on_cpu != "targets":
            target_lengths = target_lengths.cpu()
        self._compare(CTCLoss(num_classes=4, reduction=reduction), logits, lengths, targets, target_lengths)

    @pytest.mark.unit
    def test_kernel_path_is_taken(self, monkeypatch):
        def no_unpack(*args, **kwargs):
            raise AssertionError("forward_packed formed the padded log probabilities on CUDA")

        monkeypatch.setattr(ctc_module, "unpack_encoder_output", no_unpack)
        logits, lengths, targets, target_lengths = self._small_case("cuda")
        self._compare(CTCLoss(num_classes=4), logits, lengths, targets, target_lengths)

    @pytest.mark.unit
    def test_packed_ctc_loss_function(self):
        from nemo.collections.asr.parts.triton.ctc_loss import packed_ctc_loss

        logits, lengths, targets, target_lengths = self._small_case("cuda")
        log_probs = logits.log_softmax(-1)
        packed = pack_encoder_output(log_probs, lengths)
        losses = packed_ctc_loss(
            packed.data, packed.cu_seqlens, packed.lengths, targets, target_lengths, blank=4, zero_infinity=False
        )
        expected = torch.nn.functional.ctc_loss(
            log_probs.transpose(0, 1), targets, lengths, target_lengths, blank=4, reduction="none"
        )
        assert losses.shape == (3,) and losses.dtype == torch.float32
        torch.testing.assert_close(losses, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.unit
    def test_repeated_and_blank_labels(self):
        # a repeated label needs the blank between its two copies, and a blank id inside the targets is a label
        # that may not skip, as in the padded kernel
        torch.manual_seed(3)
        lengths = torch.tensor([9, 9, 7], device="cuda")
        targets = torch.tensor([[1, 1, 2, 2], [4, 4, 1, 0], [2, 4, 2, 0]], device="cuda")
        target_lengths = torch.tensor([4, 3, 3], device="cuda")
        logits = torch.randn(3, 9, 5, device="cuda")
        self._compare(CTCLoss(num_classes=4, reduction="none"), logits, lengths, targets, target_lengths)

    @pytest.mark.unit
    def test_empty_targets(self):
        torch.manual_seed(5)
        lengths = torch.tensor([6, 3], device="cuda")
        targets = torch.zeros(2, 0, dtype=torch.long, device="cuda")
        target_lengths = torch.tensor([0, 0], device="cuda")
        logits = torch.randn(2, 6, 3, device="cuda")
        self._compare(CTCLoss(num_classes=2, reduction="none"), logits, lengths, targets, target_lengths)

    @pytest.mark.unit
    @pytest.mark.parametrize("zero_infinity", [False, True])
    def test_zero_length_utterance(self, zero_infinity):
        # an utterance without frames: loss zero without targets, infinite (or zero) with targets, no gradient
        torch.manual_seed(11)
        lengths = torch.tensor([5, 0, 3], device="cuda")
        targets = torch.tensor([[1, 2], [1, 0], [2, 0]], device="cuda")
        target_lengths = torch.tensor([2, 1, 1], device="cuda")
        logits = torch.randn(3, 5, 4, device="cuda")
        criterion = CTCLoss(num_classes=3, reduction="none", zero_infinity=zero_infinity)
        losses = self._compare(criterion, logits, lengths, targets, target_lengths)
        assert losses[1].item() == (0.0 if zero_infinity else float("inf"))
        target_lengths[1] = 0
        losses = self._compare(criterion, logits, lengths, targets, target_lengths)
        assert losses[1].item() == 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize("zero_infinity", [False, True])
    def test_impossible_alignment(self, zero_infinity):
        logits = torch.tensor([[[1.0, 0.5, -0.5]], [[0.2, 0.1, 0.0]]], device="cuda", requires_grad=True)
        lengths = torch.tensor([1, 1], device="cuda")
        targets = torch.tensor([[0, 0], [1, 0]], device="cuda")
        target_lengths = torch.tensor([2, 1], device="cuda")
        criterion = CTCLoss(num_classes=2, reduction="none", zero_infinity=zero_infinity)
        packed = pack_encoder_output(logits.log_softmax(-1), lengths)
        actual = criterion.forward_packed(packed, targets, target_lengths)
        if zero_infinity:
            assert actual[0].item() == 0.0
            actual.sum().backward()
            torch.testing.assert_close(logits.grad[0], torch.zeros_like(logits[0]))
            assert logits.grad[1].abs().sum() > 0
        else:
            assert torch.isposinf(actual[0])
            assert torch.isfinite(actual[1])

    @pytest.mark.unit
    def test_production_shape(self):
        # long utterances of different lengths, a vocabulary of 1,027 classes, one target per six frames
        torch.manual_seed(13)
        lengths = torch.tensor([667, 401, 123, 50, 599, 300], device="cuda")
        target_lengths = lengths // 6
        targets = torch.randint(0, 1026, (6, int(target_lengths.max())), device="cuda")
        logits = torch.randn(6, 667, 1027, device="cuda")
        # both kernels form the posteriors from alpha + beta + nll in float32, which at magnitude 1e3 carries 1e-4
        criterion = CTCLoss(num_classes=1026, reduction="mean_batch")
        self._compare(criterion, logits, lengths, targets, target_lengths, atol=1e-4, rtol=1e-4)

    @pytest.mark.unit
    def test_float32_under_autocast(self):
        logits, lengths, targets, target_lengths = self._small_case("cuda")
        self._compare(CTCLoss(num_classes=4, reduction="sum"), logits, lengths, targets, target_lengths, autocast=True)

    @pytest.mark.unit
    def test_bfloat16_log_probs(self):
        # bfloat16 log probabilities are read as they are; the padded kernel takes them through autocast's float32 cast
        logits, lengths, targets, target_lengths = self._small_case("cuda")
        dense = logits.to(torch.bfloat16).requires_grad_()
        flat = dense.detach().clone().requires_grad_()
        criterion = CTCLoss(num_classes=4, reduction="sum")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = criterion(
                log_probs=dense.log_softmax(-1), targets=targets, input_lengths=lengths, target_lengths=target_lengths
            )
        packed = pack_encoder_output(flat.log_softmax(-1), lengths)
        assert packed.data.dtype == torch.bfloat16
        actual = criterion.forward_packed(packed, targets, target_lengths)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        expected.backward()
        actual.backward()
        assert flat.grad.dtype == torch.bfloat16
        torch.testing.assert_close(flat.grad, dense.grad)

    @pytest.mark.unit
    def test_float64_takes_padded_path(self, monkeypatch):
        # the kernel computes in float32; float64 log probabilities keep the padded kernel and its precision
        def no_kernel(*args, **kwargs):
            raise AssertionError("forward_packed ran the kernel on float64 log probabilities")

        monkeypatch.setattr(ctc_module, "packed_ctc_loss", no_kernel)
        logits, lengths, targets, target_lengths = self._small_case("cuda")
        criterion = CTCLoss(num_classes=4, reduction="none")
        losses = self._compare(criterion, logits.double(), lengths, targets, target_lengths, atol=1e-10, rtol=1e-10)
        assert losses.dtype == torch.float64

    @pytest.mark.unit
    def test_kernel_rejects_float64(self):
        from nemo.collections.asr.parts.triton.ctc_loss import packed_ctc_loss

        logits, lengths, targets, target_lengths = self._small_case("cuda")
        packed = pack_encoder_output(logits.double().log_softmax(-1), lengths)
        with pytest.raises(TypeError, match="float32"):
            packed_ctc_loss(packed.data, packed.cu_seqlens, packed.lengths, targets, target_lengths, blank=4)

    @pytest.mark.unit
    def test_over_allocated_targets_and_filler_utterances(self):
        # the static-shape regime of a captured step: the targets buffer is far wider than the longest target
        # and holds junk beyond each length, and the batch is padded with zero-length utterances at the end;
        # loss and gradient must not depend on any of it
        from nemo.collections.asr.parts.triton.ctc_loss import packed_ctc_loss

        torch.manual_seed(17)
        lengths = torch.tensor([9, 7, 4], device="cuda")
        target_lengths = torch.tensor([4, 3, 2], device="cuda")
        targets = torch.randint(0, 5, (3, 4), device="cuda")
        log_probs = torch.randn(20, 6, device="cuda").log_softmax(-1)
        cu_seqlens = torch.tensor([0, 9, 16, 20], dtype=torch.int32, device="cuda")
        tight = log_probs.detach().clone().requires_grad_()
        loss_tight = packed_ctc_loss(tight, cu_seqlens, lengths, targets, target_lengths, blank=5)
        loss_tight.sum().backward()

        wide_targets = torch.randint(0, 5, (5, 32), device="cuda")
        wide_targets[:3, :4] = targets
        wide_lengths = torch.cat([lengths, lengths.new_zeros(2)])
        wide_target_lengths = torch.cat([target_lengths, target_lengths.new_zeros(2)])
        wide_cu_seqlens = torch.tensor([0, 9, 16, 20, 20, 20], dtype=torch.int32, device="cuda")
        wide = log_probs.detach().clone().requires_grad_()
        loss_wide = packed_ctc_loss(wide, wide_cu_seqlens, wide_lengths, wide_targets, wide_target_lengths, blank=5)
        loss_wide.sum().backward()
        torch.testing.assert_close(loss_wide[:3], loss_tight)
        assert loss_wide[3:].abs().sum().item() == 0.0
        torch.testing.assert_close(wide.grad, tight.grad)

    @pytest.mark.unit
    def test_rejects_target_lengths_beyond_width(self):
        logits, lengths, targets, target_lengths = self._small_case("cuda")
        packed = pack_encoder_output(logits.log_softmax(-1), lengths)
        with pytest.raises(ValueError, match="targets width"):
            CTCLoss(num_classes=4).forward_packed(packed, targets, target_lengths + 1)

    @pytest.mark.unit
    @pytest.mark.parametrize("zero_infinity", [False, True])
    def test_rejects_negative_target_lengths(self, zero_infinity):
        # a negative length is malformed input, not an impossible alignment that zero_infinity may zero
        logits, lengths, targets, target_lengths = self._small_case("cuda")
        packed = pack_encoder_output(logits.log_softmax(-1), lengths)
        target_lengths[0] = -1
        with pytest.raises(ValueError, match="negative"):
            CTCLoss(num_classes=4, zero_infinity=zero_infinity).forward_packed(packed, targets, target_lengths)
