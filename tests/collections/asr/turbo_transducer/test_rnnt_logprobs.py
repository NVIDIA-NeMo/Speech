# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn.functional as F

from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency import kl_loss_torch, rnnt_logprobs_with_consistency
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import get_rnnt_mask, rnnt_logprobs, rnnt_logprobs_torch
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.decoding.utils import avoid_sync_operations

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency_triton import kl_loss_triton
    from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs_triton import rnnt_logprobs_triton

# NB: tests use a small set of shapes and vocabulary sizes to reuse compiled Triton kernels
# (tests should be fast without Triton cache). Vocabulary size here does not include blank.
SHAPES = [
    # batch_size, num_frames, num_text_units, vocab_size
    (2, 4, 3, 4),
    (3, 11, 7, 1024),  # vocabulary size (with blank) is not a power of 2
]
FLOAT_DTYPES = [torch.float32, torch.bfloat16]

pytestmark = pytest.mark.unit

TRITON_MARKS = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is unavailable"),
]


def _random_batch(
    batch_size: int,
    num_frames: int,
    num_text_units: int,
    vocab_size: int,
    float_dtype: torch.dtype = torch.float32,
    double_batch: bool = False,
):
    """
    Generate random logits, targets and lengths; the maximum lengths are reached (tensors are not over-padded).
    If `double_batch` is set, the batch is duplicated (teacher first half, student second half) with
    the same targets and lengths, but different logits.
    """
    device = torch.device("cuda")
    targets = torch.randint(0, vocab_size, [batch_size, num_text_units], device=device)
    src_lengths = torch.randint(1, num_frames + 1, [batch_size], device=device)
    tgt_lengths = torch.randint(0, num_text_units + 1, [batch_size], device=device)
    src_lengths[0] = num_frames
    tgt_lengths[0] = num_text_units
    if double_batch:
        targets, src_lengths, tgt_lengths = targets.repeat(2, 1), src_lengths.repeat(2), tgt_lengths.repeat(2)
        batch_size *= 2
    logits = torch.randn([batch_size, num_frames, num_text_units + 1, vocab_size + 1], device=device).to(float_dtype)
    return logits, targets, src_lengths, tgt_lengths


class TestRnntLogProbs:
    pytestmark = TRITON_MARKS

    @pytest.mark.parametrize("float_dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("batch_size,num_frames,num_text_units,vocab_size", SHAPES)
    def test_matches_torch_implementation(
        self, batch_size: int, num_frames: int, num_text_units: int, vocab_size: int, float_dtype: torch.dtype
    ):
        torch.manual_seed(777)
        logits, targets, src_lengths, tgt_lengths = _random_batch(
            batch_size, num_frames, num_text_units, vocab_size, float_dtype
        )
        # Triton implementation computes in float32, compare with float32 reference
        reference_logits = logits.detach().float().clone().requires_grad_(True)
        reference_target_scores, reference_blank_scores = rnnt_logprobs_torch(
            logits=reference_logits,
            targets=targets,
            blank_id=vocab_size,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
        )
        logits.requires_grad_(True)
        target_scores, blank_scores = rnnt_logprobs_triton(
            logits=logits, targets=targets, blank_id=vocab_size, src_lengths=src_lengths, tgt_lengths=tgt_lengths
        )
        assert target_scores.dtype == torch.float32 and blank_scores.dtype == torch.float32
        assert torch.allclose(blank_scores, reference_blank_scores, atol=1e-5)
        assert torch.allclose(target_scores, reference_target_scores, atol=1e-5)

        target_scales = torch.rand_like(target_scores)
        blank_scales = torch.rand_like(blank_scores)
        (target_scales * reference_target_scores + blank_scales * reference_blank_scores).sum().backward()
        (target_scales * target_scores + blank_scales * blank_scores).sum().backward()
        assert logits.grad.dtype == float_dtype
        atol = 1e-5 if float_dtype == torch.float32 else 1e-2
        assert torch.allclose(logits.grad.float(), reference_logits.grad, atol=atol)
        # positions outside lengths have exactly zero gradient
        _, mask = get_rnnt_mask(batch_size, num_frames, num_text_units + 1, logits.device, src_lengths, tgt_lengths)
        assert (logits.grad[~mask] == 0.0).all()

    @pytest.mark.parametrize("offset", [1e6, -1e6])
    def test_large_logits_offset(self, offset: float):
        """Log-probabilities are computed relative to the maximum logit: no precision loss for large logits."""
        torch.manual_seed(778)
        batch_size, num_frames, num_text_units, vocab_size = SHAPES[0]
        _, targets, src_lengths, tgt_lengths = _random_batch(*SHAPES[0])
        # logits are multiples of 0.25: exactly representable in float32 with the offset
        logits = torch.randint(-20, 21, [batch_size, num_frames, num_text_units + 1, vocab_size + 1], device="cuda")
        logits = (logits * 0.25 + offset).requires_grad_(True)
        reference_logits = logits.detach().clone().requires_grad_(True)
        reference_target_scores, reference_blank_scores = rnnt_logprobs_torch(
            reference_logits, targets, vocab_size, src_lengths, tgt_lengths
        )
        target_scores, blank_scores = rnnt_logprobs_triton(logits, targets, vocab_size, src_lengths, tgt_lengths)
        assert torch.allclose(blank_scores, reference_blank_scores, atol=1e-5)
        assert torch.allclose(target_scores, reference_target_scores, atol=1e-5)

        target_scales = torch.rand_like(target_scores)
        blank_scales = torch.rand_like(blank_scores)
        (target_scales * reference_target_scores + blank_scales * reference_blank_scores).sum().backward()
        (target_scales * target_scores + blank_scales * blank_scores).sum().backward()
        assert torch.allclose(logits.grad, reference_logits.grad, atol=1e-5)

    def test_constant_large_logits(self):
        """Uniform distribution for constant logits of any magnitude."""
        vocab_size_with_blank = SHAPES[0][-1] + 1
        logits = torch.full([1, 1, 2, vocab_size_with_blank], fill_value=1e8, device="cuda")
        targets = torch.tensor([[1]], device="cuda")
        target_scores, blank_scores = rnnt_logprobs_triton(logits, targets, blank_id=vocab_size_with_blank - 1)
        expected_score = -math.log(vocab_size_with_blank)
        assert torch.allclose(blank_scores, torch.full_like(blank_scores, expected_score), atol=1e-6)
        assert target_scores[0, 0].tolist() == pytest.approx([expected_score, 0.0], abs=1e-6)

    def test_without_lengths(self):
        torch.manual_seed(1)
        logits, targets, _, _ = _random_batch(*SHAPES[0])
        target_scores, blank_scores = rnnt_logprobs_triton(logits=logits, targets=targets, blank_id=SHAPES[0][-1])
        reference_target_scores, reference_blank_scores = rnnt_logprobs_torch(
            logits=logits, targets=targets, blank_id=SHAPES[0][-1]
        )
        assert torch.allclose(blank_scores, reference_blank_scores, atol=1e-5)
        assert torch.allclose(target_scores, reference_target_scores, atol=1e-5)

    def test_non_contiguous_upstream_gradient(self):
        """Upstream gradients can be non-contiguous, e.g., expanded after `sum()`."""
        torch.manual_seed(2)
        logits, targets, src_lengths, tgt_lengths = _random_batch(*SHAPES[0])
        blank_id = SHAPES[0][-1]
        reference_logits = logits.clone().requires_grad_(True)
        logits.requires_grad_(True)
        target_scores, blank_scores = rnnt_logprobs_triton(logits, targets, blank_id, src_lengths, tgt_lengths)
        (target_scores.sum() + blank_scores.sum()).backward()
        reference_target_scores, reference_blank_scores = rnnt_logprobs_torch(
            reference_logits, targets, blank_id, src_lengths, tgt_lengths
        )
        (reference_target_scores.sum() + reference_blank_scores.sum()).backward()
        assert torch.allclose(logits.grad, reference_logits.grad, atol=1e-5)

    @pytest.mark.parametrize("extra_target_columns", [-2, 3])
    def test_targets_padding(self, extra_target_columns: int):
        """Targets can be narrower or wider than the U dimension of logits."""
        torch.manual_seed(3)
        logits, targets, src_lengths, tgt_lengths = _random_batch(*SHAPES[0])
        tgt_lengths.clamp_(max=targets.shape[1] - 2)
        blank_id = SHAPES[0][-1]
        if extra_target_columns > 0:
            padded_targets = torch.cat([targets, targets[:, :extra_target_columns]], dim=1)
        else:
            padded_targets = targets[:, :extra_target_columns].clone()
        expected = rnnt_logprobs_triton(logits, targets, blank_id, src_lengths, tgt_lengths)
        actual = rnnt_logprobs_triton(logits, padded_targets, blank_id, src_lengths, tgt_lengths)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])

    def test_float64_gradcheck(self):
        torch.manual_seed(4)
        logits, targets, src_lengths, tgt_lengths = _random_batch(*SHAPES[0], float_dtype=torch.float64)
        logits.requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda x: rnnt_logprobs_triton(x, targets, SHAPES[0][-1], src_lengths, tgt_lengths), (logits,)
        )

    def test_invalid_inputs(self):
        logits, targets, _, _ = _random_batch(*SHAPES[0])
        with pytest.raises(ValueError, match="blank_id"):
            rnnt_logprobs_triton(logits=logits, targets=targets, blank_id=logits.shape[-1])
        with pytest.raises(ValueError, match="contiguous"):
            rnnt_logprobs_triton(logits=logits.transpose(1, 2), targets=targets, blank_id=0)


class TestRnntLogProbsTorch:
    """Pure PyTorch implementation (fallback for CPU and for the systems without Triton)"""

    def test_targets_narrower_than_lattice(self):
        """Targets can be narrower than the U dimension of logits (as for the Triton implementation)."""
        vocab_size_with_blank = 3
        logits = torch.zeros([1, 2, 4, vocab_size_with_blank], requires_grad=True)
        targets = torch.tensor([[1]])
        src_lengths, tgt_lengths = torch.tensor([2]), torch.tensor([1])
        target_scores, blank_scores = rnnt_logprobs(
            logits, targets, vocab_size_with_blank - 1, src_lengths, tgt_lengths
        )
        expected_score = -math.log(vocab_size_with_blank)
        expected_target_scores = torch.zeros([1, 2, 4])
        expected_target_scores[:, :, 0] = expected_score
        expected_blank_scores = torch.zeros([1, 2, 4])
        expected_blank_scores[:, :, :2] = expected_score
        assert torch.allclose(target_scores, expected_target_scores)
        assert torch.allclose(blank_scores, expected_blank_scores)

        # same result and gradient with explicitly padded targets
        padded_logits = logits.detach().clone().requires_grad_(True)
        padded_targets = torch.tensor([[1, 0, 0]])
        padded_target_scores, padded_blank_scores = rnnt_logprobs(
            padded_logits, padded_targets, vocab_size_with_blank - 1, src_lengths, tgt_lengths
        )
        assert torch.equal(target_scores, padded_target_scores) and torch.equal(blank_scores, padded_blank_scores)
        (target_scores.sum() + 2 * blank_scores.sum()).backward()
        (padded_target_scores.sum() + 2 * padded_blank_scores.sum()).backward()
        assert torch.equal(logits.grad, padded_logits.grad)

    @pytest.mark.parametrize("num_targets", [0, 2, 3, 5])
    def test_targets_width(self, num_targets: int):
        """Targets narrower/wider than the lattice produce the same result as targets of the matching width."""
        torch.manual_seed(779)
        batch_size, num_frames, num_text_units, vocab_size = 2, 4, 3, 4
        logits = torch.randn([batch_size, num_frames, num_text_units + 1, vocab_size + 1])
        targets = torch.randint(0, vocab_size, [batch_size, num_text_units])
        src_lengths = torch.tensor([4, 3])
        tgt_lengths = torch.tensor([min(2, num_targets), 0])
        expected = rnnt_logprobs_torch(logits, targets, vocab_size, src_lengths, tgt_lengths)
        if num_targets <= num_text_units:
            resized_targets = targets[:, :num_targets]
        else:
            resized_targets = torch.cat([targets, targets[:, : num_targets - num_text_units]], dim=1)
        actual = rnnt_logprobs_torch(logits, resized_targets, vocab_size, src_lengths, tgt_lengths)
        assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])

    @pytest.mark.parametrize("use_lengths", [True, False])
    def test_missing_targets_raise(self, use_lengths: bool):
        """Targets required by the lengths are absent: error instead of using arbitrary labels."""
        logits = torch.zeros([1, 2, 4, 3])
        targets = torch.tensor([[1]])
        # without lengths: all `U` targets are required
        tgt_lengths = torch.tensor([2]) if use_lengths else None
        with pytest.raises(ValueError, match="targets"):
            rnnt_logprobs_torch(logits, targets, blank_id=2, tgt_lengths=tgt_lengths)

    def test_consistency_targets_narrower_than_lattice(self):
        """PyTorch fallback of the fused function with targets narrower than the lattice."""
        torch.manual_seed(780)
        vocab_size = 4
        logits = torch.randn([4, 3, 4, vocab_size + 1])
        targets = torch.randint(0, vocab_size, [2, 3]).repeat(2, 1)
        src_lengths, tgt_lengths = torch.tensor([3, 2]).repeat(2), torch.tensor([2, 1]).repeat(2)
        expected = rnnt_logprobs_with_consistency(
            logits, targets, vocab_size, src_lengths, tgt_lengths, compute_consistency=True
        )
        actual = rnnt_logprobs_with_consistency(
            logits, targets[:, :2], vocab_size, src_lengths, tgt_lengths, compute_consistency=True
        )
        for expected_tensor, actual_tensor in zip(expected, actual):
            assert torch.equal(expected_tensor, actual_tensor)


class TestRnntLogProbsWithConsistency:
    """
    Tests for the fused extraction of the RNN-T log-probs and the consistency loss from the Joint output.
    Both are computed from the same logits tensor and share a single gradient buffer.
    """

    pytestmark = TRITON_MARKS

    @pytest.mark.parametrize("symmetrical", [True, False])
    @pytest.mark.parametrize("float_dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("half_batch_size,num_frames,num_text_units,vocab_size", SHAPES)
    def test_matches_torch_implementation(
        self,
        half_batch_size: int,
        num_frames: int,
        num_text_units: int,
        vocab_size: int,
        float_dtype: torch.dtype,
        symmetrical: bool,
    ):
        """Triton implementation must match the pure PyTorch implementation (forward and backward)."""
        torch.manual_seed(777)
        blank_id = vocab_size
        logits, targets, src_lengths, tgt_lengths = _random_batch(
            half_batch_size, num_frames, num_text_units, vocab_size, float_dtype, double_batch=True
        )

        # Triton implementation computes in float32, compare with float32 reference
        reference_logits = logits.detach().float().clone().requires_grad_(True)
        target_logprobs, blank_logprobs = rnnt_logprobs_torch(
            logits=reference_logits,
            targets=targets,
            blank_id=blank_id,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
        )
        _, mask = get_rnnt_mask(
            batch_size=half_batch_size,
            src_length_max=num_frames,
            tgt_length_max_plus_1=num_text_units + 1,
            device=logits.device,
            src_lengths=src_lengths[:half_batch_size],
            tgt_lengths=tgt_lengths[:half_batch_size],
        )
        consistency_kl = kl_loss_torch(
            teacher_logprobs=F.log_softmax(reference_logits[:half_batch_size], dim=-1),
            student_logprobs=F.log_softmax(reference_logits[half_batch_size:], dim=-1),
            mask=mask,
            symmetrical=symmetrical,
        )

        logits.requires_grad_(True)
        output = rnnt_logprobs_with_consistency(
            logits=logits,
            targets=targets,
            blank_id=blank_id,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            compute_consistency=True,
            consistency_symmetrical=symmetrical,
        )
        assert output.consistency_kl.dtype == torch.float32
        assert torch.allclose(output.target_logprobs, target_logprobs, atol=1e-5)
        assert torch.allclose(output.blank_logprobs, blank_logprobs, atol=1e-5)
        assert torch.allclose(output.consistency_kl, consistency_kl, atol=1e-5, rtol=1e-4)
        assert (output.consistency_kl[~mask] == 0.0).all()

        target_scales = torch.rand_like(target_logprobs)
        blank_scales = torch.rand_like(blank_logprobs)
        consistency_scales = torch.rand_like(consistency_kl)

        def weighted_sum(loss_target_logprobs, loss_blank_logprobs, loss_consistency_kl):
            return (
                (target_scales * loss_target_logprobs).sum()
                + (blank_scales * loss_blank_logprobs).sum()
                + (consistency_scales * loss_consistency_kl).sum()
            )

        weighted_sum(*output).backward()
        weighted_sum(target_logprobs, blank_logprobs, consistency_kl).backward()
        atol = 1e-5 if float_dtype == torch.float32 else 1e-2
        assert torch.allclose(logits.grad.float(), reference_logits.grad, atol=atol)

    def test_masked_positions_are_zero(self):
        """Positions outside the source/target lengths must not contribute to the consistency loss."""
        torch.manual_seed(3)
        half_batch_size, num_frames, num_text_units, vocab_size = SHAPES[0]
        logits, targets, _, _ = _random_batch(*SHAPES[0], double_batch=True)
        logits.requires_grad_(True)
        src_lengths = torch.tensor([num_frames, 2], device=logits.device).repeat(2)
        tgt_lengths = torch.tensor([num_text_units, 1], device=logits.device).repeat(2)

        output = rnnt_logprobs_with_consistency(
            logits=logits,
            targets=targets,
            blank_id=vocab_size,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            compute_consistency=True,
        )
        assert (output.consistency_kl[1, 2:] == 0.0).all()
        assert (output.consistency_kl[1, :, 3:] == 0.0).all()

        output.consistency_kl.sum().backward()
        # no consistency gradient outside the mask (the RNN-T part is not used here)
        assert (logits.grad[1, 2:] == 0.0).all()
        assert (logits.grad[half_batch_size + 1, 2:] == 0.0).all()

    def test_without_consistency_matches_plain_logprobs(self):
        torch.manual_seed(4)
        logits, targets, src_lengths, tgt_lengths = _random_batch(*SHAPES[0], double_batch=True)
        vocab_size = SHAPES[0][-1]
        output = rnnt_logprobs_with_consistency(
            logits=logits,
            targets=targets,
            blank_id=vocab_size,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            compute_consistency=False,
        )
        target_logprobs, blank_logprobs = rnnt_logprobs_triton(
            logits=logits, targets=targets, blank_id=vocab_size, src_lengths=src_lengths, tgt_lengths=tgt_lengths
        )
        assert output.consistency_kl is None
        assert torch.equal(output.target_logprobs, target_logprobs)
        assert torch.equal(output.blank_logprobs, blank_logprobs)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Test requires 2 CUDA devices")
    def test_non_current_device(self):
        """Inputs are on a CUDA device that is not the current device."""
        torch.manual_seed(7)
        vocab_size = SHAPES[0][-1]
        # NB: inputs are moved through CPU; copying tensors between CUDA devices directly can hide the issue
        batch = [tensor.cpu() for tensor in _random_batch(*SHAPES[0], double_batch=True)]
        other_device = torch.device("cuda", (torch.cuda.current_device() + 1) % torch.cuda.device_count())
        results = []
        for device in [torch.device("cuda"), other_device]:
            logits, targets, src_lengths, tgt_lengths = [tensor.to(device) for tensor in batch]
            logits.requires_grad_(True)
            output = rnnt_logprobs_with_consistency(
                logits=logits,
                targets=targets,
                blank_id=vocab_size,
                src_lengths=src_lengths,
                tgt_lengths=tgt_lengths,
                compute_consistency=True,
            )
            (output.target_logprobs.sum() + output.blank_logprobs.sum() + output.consistency_kl.sum()).backward()
            results.append([tensor.detach().cpu() for tensor in (*output, logits.grad)])
        for reference, actual in zip(*results):
            assert torch.equal(reference, actual)

    def test_odd_batch_size_raises(self):
        logits, targets, _, _ = _random_batch(3, 4, 2, 4)
        with pytest.raises(ValueError, match="double batch"):
            rnnt_logprobs_with_consistency(logits=logits, targets=targets, blank_id=4, compute_consistency=True)

    def test_no_cuda_sync_operations(self):
        torch.manual_seed(5)
        logits, targets, src_lengths, tgt_lengths = _random_batch(*SHAPES[0], double_batch=True)
        logits.requires_grad_(True)

        def run():
            output = rnnt_logprobs_with_consistency(
                logits=logits,
                targets=targets,
                blank_id=SHAPES[0][-1],
                src_lengths=src_lengths,
                tgt_lengths=tgt_lengths,
                compute_consistency=True,
            )
            (output.target_logprobs.sum() + output.blank_logprobs.sum() + output.consistency_kl.sum()).backward()

        run()  # warmup (Triton compilation)
        with avoid_sync_operations(logits.device):
            run()

    def test_backward_memory_is_lower_than_separate_losses(self):
        """The fused implementation must allocate a single gradient buffer for the logits."""
        torch.manual_seed(6)
        half_batch_size, num_frames, num_text_units, vocab_size = 4, 64, 32, SHAPES[1][-1]
        logits, targets, src_lengths, tgt_lengths = _random_batch(
            half_batch_size, num_frames, num_text_units, vocab_size, double_batch=True
        )
        logits_size_bytes = logits.numel() * logits.element_size()
        _, mask = get_rnnt_mask(
            batch_size=half_batch_size,
            src_length_max=num_frames,
            tgt_length_max_plus_1=num_text_units + 1,
            device=logits.device,
            src_lengths=src_lengths[:half_batch_size],
            tgt_lengths=tgt_lengths[:half_batch_size],
        )

        def measure_backward_peak(fused: bool) -> int:
            current_logits = logits.clone().requires_grad_(True)
            if fused:
                output = rnnt_logprobs_with_consistency(
                    logits=current_logits,
                    targets=targets,
                    blank_id=vocab_size,
                    src_lengths=src_lengths,
                    tgt_lengths=tgt_lengths,
                    compute_consistency=True,
                )
                loss = output.target_logprobs.sum() + output.blank_logprobs.sum() + output.consistency_kl.sum()
            else:
                target_logprobs, blank_logprobs = rnnt_logprobs_triton(
                    logits=current_logits,
                    targets=targets,
                    blank_id=vocab_size,
                    src_lengths=src_lengths,
                    tgt_lengths=tgt_lengths,
                )
                consistency_kl = kl_loss_triton(
                    teacher_logits=current_logits[:half_batch_size],
                    student_logits=current_logits[half_batch_size:],
                    mask=mask,
                    symmetrical=True,
                )
                loss = target_logprobs.sum() + blank_logprobs.sum() + consistency_kl.sum()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated()
            loss.backward()
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() - memory_before
            del current_logits, loss
            return peak_memory

        fused_peak = measure_backward_peak(fused=True)
        separate_peak = measure_backward_peak(fused=False)

        # single gradient buffer (plus small tensors) for the fused implementation
        assert fused_peak < 1.1 * logits_size_bytes
        # the separate implementation additionally materializes the KL gradients and the slice buffers
        assert separate_peak > 2.0 * logits_size_bytes
