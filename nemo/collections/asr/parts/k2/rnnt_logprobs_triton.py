# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
RNN-T log-probs helpers (Triton).

The implementation is in `nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs_triton`;
this module preserves the established import path and the argument names of `rnnt_logprobs_triton`.
"""

import torch

from nemo.collections.asr.parts.turbo_transducer import rnnt_logprobs_triton as _impl
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs_triton import RnntLogProbs

__all__ = ["RnntLogProbs", "rnnt_logprobs_triton"]


def rnnt_logprobs_triton(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    source_lengths: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given logits, calculate log probabilities for blank and target labels needed for transducer loss calculation.
    Optimized implementation in Triton.

    Args:
        logits: Joint tensor of size [B, T, U+1, D]
        targets: Targets of size [B, U]
        blank_id: id of the blank output
        source_lengths: optional tensor with lengths for source utterances
        target_lengths: optional tensor with lengths for targets

    Returns:
        Tuple of tensors with log probabilities for targets and blank labels, both of size [B, T, U+1].
        For the non-existent targets (U+1 or beyond target_lengths) output is zero.
    """
    return _impl.rnnt_logprobs_triton(
        logits=logits, targets=targets, blank_id=blank_id, src_lengths=source_lengths, tgt_lengths=target_lengths
    )
