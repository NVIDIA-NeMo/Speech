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

import random

import pytest
import torch

from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency import kl_loss_torch, rnnt_logprobs_with_consistency
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import get_rnnt_mask, rnnt_logprobs_torch
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from tests.collections.asr.decoding.utils import avoid_sync_operations

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency_triton import kl_loss_triton
    from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs_triton import rnnt_logprobs_triton


def _random_consistency_batch(
    half_batch_size: int,
    num_frames: int,
    num_text_units: int,
    vocab_size: int,
    float_dtype: torch.dtype,
    device: torch.device,
):
    """Generate a random double batch (teacher first half, student second half) with matching lengths."""
    targets = torch.randint(0, vocab_size, [half_batch_size, num_text_units], device=device).repeat(2, 1)
    src_lengths = torch.randint(1, num_frames + 1, [half_batch_size], device=device).repeat(2)
    tgt_lengths = torch.randint(0, num_text_units + 1, [half_batch_size], device=device).repeat(2)
    # make sure the maximum lengths are reached (tensors are not over-padded)
    src_lengths[0] = num_frames
    src_lengths[half_batch_size] = num_frames
    tgt_lengths[0] = num_text_units
    tgt_lengths[half_batch_size] = num_text_units
    logits = torch.randn(
        [2 * half_batch_size, num_frames, num_text_units + 1, vocab_size + 1], dtype=float_dtype, device=device
    )
    return logits, targets, src_lengths, tgt_lengths


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed, skipping RNNT Log Probs tests")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
class TestRnntLogProbs:
    @pytest.mark.parametrize(
        "batch_size,num_frames,num_text_units,vocab_size",
        [
            (1, 4, 2, 4),
            (2, 3, 2, 5),
            (2, 16, 31, 17),
            (16, 129, 65, 2048),
        ],
    )
    @pytest.mark.parametrize(
        "float_dtype",
        [torch.float32] + ([torch.bfloat16] if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else []),
    )
    def test_rnnt_logprobs_random(
        self, batch_size: int, num_frames: int, num_text_units: int, vocab_size: int, float_dtype: torch.dtype
    ):
        """
        Test Triton-based implementation using etalon Torch-based implementation for RNN-T log-probs.
        """
        device = torch.device("cuda")
        torch.manual_seed(777)

        targets = torch.tensor(
            [[random.randrange(0, vocab_size - 1) for i in range(num_text_units)] for j in range(batch_size)],
            device=device,
            dtype=torch.long,
        )

        logits = torch.rand(
            [batch_size, num_frames, num_text_units + 1, vocab_size + 1],
            dtype=float_dtype,
            device=device,
            requires_grad=True,
        )

        # Triton-based implementation works in float32 precision for accuracy purposes, should compare with float32
        target_scores_etalon, blank_scores_etalon = rnnt_logprobs_torch(
            logits=logits.to(torch.float32), targets=targets, blank_id=vocab_size
        )
        logits2 = logits.clone().detach()
        logits2.requires_grad_(True)
        target_scores, blank_scores = rnnt_logprobs_triton(logits=logits2, targets=targets, blank_id=vocab_size)
        target_scores[..., -1:] = 0.0
        target_scores_etalon[..., -1:] = 0.0
        assert torch.allclose(blank_scores, blank_scores_etalon, atol=1e-5)
        assert torch.allclose(target_scores, target_scores_etalon, atol=1e-5)

        # test backward
        target_scales = torch.rand_like(target_scores, requires_grad=False)
        blank_scales = torch.rand_like(blank_scores, requires_grad=False)
        loss_etalon = (target_scales * target_scores_etalon + blank_scales * blank_scores_etalon).sum()
        loss = (target_scales * target_scores + blank_scales * blank_scores).sum()
        loss_etalon.backward()
        loss.backward()
        assert torch.allclose(logits.grad, logits2.grad, atol=1e-5)


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed, skipping RNNT Log Probs tests")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
class TestRnntLogProbsWithConsistency:
    """
    Tests for the fused extraction of the RNN-T log-probs and the consistency loss from the Joint output.
    Both are computed from the same logits tensor and share a single gradient buffer.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "half_batch_size,num_frames,num_text_units,vocab_size",
        [
            (1, 4, 2, 4),
            (2, 3, 2, 5),
            (2, 16, 31, 17),
            (3, 11, 7, 128),
        ],
    )
    @pytest.mark.parametrize(
        "float_dtype",
        [torch.float32] + ([torch.bfloat16] if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else []),
    )
    @pytest.mark.parametrize("symmetrical", [True, False])
    def test_matches_separate_computation(
        self,
        half_batch_size: int,
        num_frames: int,
        num_text_units: int,
        vocab_size: int,
        float_dtype: torch.dtype,
        symmetrical: bool,
    ):
        """Fused forward/backward must match separate `rnnt_logprobs` and `kl_loss` computations."""
        device = torch.device("cuda")
        torch.manual_seed(777)
        blank_id = vocab_size
        logits, targets, src_lengths, tgt_lengths = _random_consistency_batch(
            half_batch_size, num_frames, num_text_units, vocab_size, float_dtype, device
        )

        fused_logits = logits.clone().requires_grad_(True)
        fused = rnnt_logprobs_with_consistency(
            logits=fused_logits,
            targets=targets,
            blank_id=blank_id,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            compute_consistency=True,
            consistency_symmetrical=symmetrical,
        )

        reference_logits = logits.clone().requires_grad_(True)
        target_logprobs, blank_logprobs = rnnt_logprobs_triton(
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
            device=device,
            src_lengths=src_lengths[:half_batch_size],
            tgt_lengths=tgt_lengths[:half_batch_size],
        )
        consistency_kl = kl_loss_triton(
            teacher_logits=reference_logits[:half_batch_size],
            student_logits=reference_logits[half_batch_size:],
            mask=mask,
            symmetrical=symmetrical,
        )

        assert torch.allclose(fused.target_logprobs, target_logprobs, atol=1e-5)
        assert torch.allclose(fused.blank_logprobs, blank_logprobs, atol=1e-5)
        # NB: the fused implementation keeps the consistency loss in float32, while the standalone
        # `kl_loss_triton` returns it in the dtype of the logits, hence the larger tolerance for bfloat16
        consistency_atol = 1e-5 if float_dtype == torch.float32 else 1e-2
        assert torch.allclose(
            fused.consistency_kl,
            consistency_kl.to(fused.consistency_kl.dtype),
            atol=consistency_atol,
            rtol=consistency_atol,
        )

        target_scales = torch.rand_like(target_logprobs)
        blank_scales = torch.rand_like(blank_logprobs)
        consistency_scales = torch.rand_like(consistency_kl)

        def weighted_sum(loss_target_logprobs, loss_blank_logprobs, loss_consistency_kl):
            return (
                (target_scales * loss_target_logprobs).sum()
                + (blank_scales * loss_blank_logprobs).sum()
                + (consistency_scales.to(loss_consistency_kl.dtype) * loss_consistency_kl).sum()
            )

        weighted_sum(fused.target_logprobs, fused.blank_logprobs, fused.consistency_kl).backward()
        weighted_sum(target_logprobs, blank_logprobs, consistency_kl).backward()
        atol = 1e-5 if float_dtype == torch.float32 else 1e-2
        assert torch.allclose(fused_logits.grad, reference_logits.grad, atol=atol)

    @pytest.mark.unit
    @pytest.mark.parametrize("symmetrical", [True, False])
    def test_matches_torch_implementation(self, symmetrical: bool):
        """Triton implementation must match the pure PyTorch fallback."""
        device = torch.device("cuda")
        torch.manual_seed(15)
        half_batch_size, num_frames, num_text_units, vocab_size = 2, 6, 4, 11
        blank_id = vocab_size
        logits, targets, src_lengths, tgt_lengths = _random_consistency_batch(
            half_batch_size, num_frames, num_text_units, vocab_size, torch.float32, device
        )

        triton_logits = logits.clone().requires_grad_(True)
        triton_output = rnnt_logprobs_with_consistency(
            logits=triton_logits,
            targets=targets,
            blank_id=blank_id,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            compute_consistency=True,
            consistency_symmetrical=symmetrical,
        )

        torch_logits = logits.clone().requires_grad_(True)
        target_logprobs, blank_logprobs = rnnt_logprobs_torch(
            logits=torch_logits,
            targets=targets,
            blank_id=blank_id,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
        )
        _, mask = get_rnnt_mask(
            batch_size=half_batch_size,
            src_length_max=num_frames,
            tgt_length_max_plus_1=num_text_units + 1,
            device=device,
            src_lengths=src_lengths[:half_batch_size],
            tgt_lengths=tgt_lengths[:half_batch_size],
        )
        consistency_kl = kl_loss_torch(
            teacher_logprobs=torch.nn.functional.log_softmax(torch_logits[:half_batch_size], dim=-1),
            student_logprobs=torch.nn.functional.log_softmax(torch_logits[half_batch_size:], dim=-1),
            mask=mask,
            symmetrical=symmetrical,
        )

        assert torch.allclose(triton_output.target_logprobs, target_logprobs, atol=1e-5)
        assert torch.allclose(triton_output.blank_logprobs, blank_logprobs, atol=1e-5)
        assert torch.allclose(triton_output.consistency_kl, consistency_kl, atol=1e-5)

        (triton_output.target_logprobs.sum() + triton_output.consistency_kl.sum()).backward()
        (target_logprobs.sum() + consistency_kl.sum()).backward()
        assert torch.allclose(triton_logits.grad, torch_logits.grad, atol=1e-5)

    @pytest.mark.unit
    def test_masked_positions_are_zero(self):
        """Positions outside the source/target lengths must not contribute to the consistency loss."""
        device = torch.device("cuda")
        torch.manual_seed(3)
        half_batch_size, num_frames, num_text_units, vocab_size = 2, 5, 4, 7
        logits = torch.randn(
            [2 * half_batch_size, num_frames, num_text_units + 1, vocab_size + 1], device=device, requires_grad=True
        )
        targets = torch.randint(0, vocab_size, [half_batch_size, num_text_units], device=device).repeat(2, 1)
        src_lengths = torch.tensor([num_frames, 2], device=device).repeat(2)
        tgt_lengths = torch.tensor([num_text_units, 1], device=device).repeat(2)

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

    @pytest.mark.unit
    def test_without_consistency_matches_plain_logprobs(self):
        device = torch.device("cuda")
        torch.manual_seed(4)
        half_batch_size, num_frames, num_text_units, vocab_size = 2, 5, 4, 7
        logits, targets, src_lengths, tgt_lengths = _random_consistency_batch(
            half_batch_size, num_frames, num_text_units, vocab_size, torch.float32, device
        )
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

    @pytest.mark.unit
    def test_odd_batch_size_raises(self):
        device = torch.device("cuda")
        logits = torch.randn([3, 4, 3, 5], device=device)
        targets = torch.randint(0, 4, [3, 2], device=device)
        with pytest.raises(ValueError, match="double batch"):
            rnnt_logprobs_with_consistency(
                logits=logits, targets=targets, blank_id=4, compute_consistency=True  # noqa: E501
            )

    @pytest.mark.unit
    def test_no_cuda_sync_operations(self):
        device = torch.device("cuda")
        torch.manual_seed(5)
        half_batch_size, num_frames, num_text_units, vocab_size = 2, 5, 4, 7
        logits, targets, src_lengths, tgt_lengths = _random_consistency_batch(
            half_batch_size, num_frames, num_text_units, vocab_size, torch.float32, device
        )
        logits.requires_grad_(True)

        def run():
            output = rnnt_logprobs_with_consistency(
                logits=logits,
                targets=targets,
                blank_id=vocab_size,
                src_lengths=src_lengths,
                tgt_lengths=tgt_lengths,
                compute_consistency=True,
            )
            (output.target_logprobs.sum() + output.blank_logprobs.sum() + output.consistency_kl.sum()).backward()

        run()  # warmup (Triton compilation)
        with avoid_sync_operations(device):
            run()

    @pytest.mark.unit
    def test_backward_memory_is_lower_than_separate_losses(self):
        """The fused implementation must allocate a single gradient buffer for the logits."""
        device = torch.device("cuda")
        torch.manual_seed(6)
        half_batch_size, num_frames, num_text_units, vocab_size = 4, 65, 33, 511
        logits, targets, src_lengths, tgt_lengths = _random_consistency_batch(
            half_batch_size, num_frames, num_text_units, vocab_size, torch.float32, device
        )
        logits_size_bytes = logits.numel() * logits.element_size()
        _, mask = get_rnnt_mask(
            batch_size=half_batch_size,
            src_length_max=num_frames,
            tgt_length_max_plus_1=num_text_units + 1,
            device=device,
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

        measure_backward_peak(fused=True)  # warmup
        fused_peak = measure_backward_peak(fused=True)
        separate_peak = measure_backward_peak(fused=False)

        # single gradient buffer (plus small tensors) for the fused implementation
        assert fused_peak < 1.1 * logits_size_bytes
        # the separate implementation additionally materializes the KL gradients and the slice buffers
        assert separate_peak > 2.0 * logits_size_bytes
