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

"""
Tests for the established import paths of the RNN-T log-probs helpers: `nemo.collections.asr.parts.k2.rnnt_logprobs`
and `nemo.collections.asr.parts.k2.rnnt_logprobs_triton` (k2 is not required for these modules).
The helpers are implemented in `nemo.collections.asr.parts.turbo_transducer`,
see `tests/collections/asr/turbo_transducer/test_rnnt_logprobs.py` for the tests of the implementation.
"""

import math

import pytest
import torch

from nemo.core.utils.optional_libs import TRITON_AVAILABLE


@pytest.mark.unit
class TestRnntLogProbsTorchImportPath:
    def test_rnnt_logprobs_torch(self):
        from nemo.collections.asr.parts.k2.rnnt_logprobs import rnnt_logprobs_torch

        vocab_size, blank_id = 5, 4
        logits = torch.zeros(2, 3, 2, vocab_size, requires_grad=True)
        targets = torch.tensor([[1], [2]])
        target_scores, blank_scores = rnnt_logprobs_torch(logits=logits, targets=targets, blank_id=blank_id)
        uniform_score = -math.log(vocab_size)
        assert torch.allclose(blank_scores, torch.full([2, 3, 2], uniform_score))
        assert torch.allclose(target_scores[:, :, 0], torch.full([2, 3], uniform_score))
        # scores for the last (non-existent) target are zero
        assert (target_scores[:, :, 1] == 0.0).all()

        # gradient: d(scale * log_softmax(logits)[label]) / d(logits) = scale * (one_hot(label) - softmax(logits))
        target_scale, blank_scale = 2.0, 3.0
        (target_scale * target_scores.sum() + blank_scale * blank_scores.sum()).backward()
        expected_grad = torch.full([2, 3, 2, vocab_size], -blank_scale / vocab_size)
        expected_grad[..., blank_id] += blank_scale
        expected_grad[:, :, 0] -= target_scale / vocab_size
        expected_grad[0, :, 0, 1] += target_scale
        expected_grad[1, :, 0, 2] += target_scale
        assert torch.allclose(logits.grad, expected_grad, atol=1e-6)


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is unavailable")
class TestRnntLogProbsTritonImportPath:
    @staticmethod
    def _random_batch():
        torch.manual_seed(0)
        vocab_size = 5
        logits = torch.randn(3, 6, 4, vocab_size, device="cuda")
        targets = torch.randint(0, vocab_size - 1, (3, 3), device="cuda")
        source_lengths = torch.tensor([6, 4, 5], device="cuda")
        target_lengths = torch.tensor([3, 1, 0], device="cuda")
        return logits, targets, vocab_size - 1, source_lengths, target_lengths

    @staticmethod
    def _backward(logits: torch.Tensor, target_scores: torch.Tensor, blank_scores: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(1)
        target_scales, blank_scales = torch.rand_like(target_scores), torch.rand_like(blank_scores)
        (target_scales * target_scores + blank_scales * blank_scores).sum().backward()
        return logits.grad

    @pytest.mark.parametrize("use_lengths", [True, False])
    def test_rnnt_logprobs_triton(self, use_lengths: bool):
        """The function with the original keyword arguments (`source_lengths`, `target_lengths`)."""
        from nemo.collections.asr.parts.k2.rnnt_logprobs_triton import rnnt_logprobs_triton
        from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs_triton import (
            rnnt_logprobs_triton as rnnt_logprobs_triton_impl,
        )

        logits, targets, blank_id, source_lengths, target_lengths = self._random_batch()
        if not use_lengths:
            source_lengths, target_lengths = None, None

        logits_reference = logits.clone().requires_grad_(True)
        target_scores_reference, blank_scores_reference = rnnt_logprobs_triton_impl(
            logits=logits_reference,
            targets=targets,
            blank_id=blank_id,
            src_lengths=source_lengths,
            tgt_lengths=target_lengths,
        )
        logits.requires_grad_(True)
        if use_lengths:
            target_scores, blank_scores = rnnt_logprobs_triton(
                logits=logits,
                targets=targets,
                blank_id=blank_id,
                source_lengths=source_lengths,
                target_lengths=target_lengths,
            )
        else:
            target_scores, blank_scores = rnnt_logprobs_triton(logits=logits, targets=targets, blank_id=blank_id)
        assert torch.equal(target_scores, target_scores_reference)
        assert torch.equal(blank_scores, blank_scores_reference)
        if use_lengths:
            # positions beyond the lengths are masked
            assert (blank_scores[1, 4:] == 0.0).all() and (blank_scores[1, :, 2:] == 0.0).all()
            assert (blank_scores[1, :4, :2] != 0.0).all()
        grad = self._backward(logits, target_scores, blank_scores)
        grad_reference = self._backward(logits_reference, target_scores_reference, blank_scores_reference)
        assert torch.equal(grad, grad_reference)

    def test_rnnt_logprobs_function(self):
        """`RnntLogProbs` autograd function with the original order of positional arguments."""
        from nemo.collections.asr.parts.k2.rnnt_logprobs_triton import RnntLogProbs, rnnt_logprobs_triton

        logits, targets, blank_id, source_lengths, target_lengths = self._random_batch()
        logits_reference = logits.clone().requires_grad_(True)
        target_scores_reference, blank_scores_reference = rnnt_logprobs_triton(
            logits_reference, targets, blank_id, source_lengths, target_lengths
        )
        logits.requires_grad_(True)
        target_scores, blank_scores = RnntLogProbs.apply(logits, targets, blank_id, source_lengths, target_lengths)
        assert torch.equal(target_scores, target_scores_reference)
        assert torch.equal(blank_scores, blank_scores_reference)
        grad = self._backward(logits, target_scores, blank_scores)
        grad_reference = self._backward(logits_reference, target_scores_reference, blank_scores_reference)
        assert torch.equal(grad, grad_reference)
