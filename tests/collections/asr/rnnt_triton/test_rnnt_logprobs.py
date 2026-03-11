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

from nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs import rnnt_logprobs_torch
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs_triton import rnnt_logprobs_triton

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
