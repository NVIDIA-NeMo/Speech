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

import torch
import torch.nn.functional as F


def rnnt_logprobs_torch(
    logits: torch.Tensor, targets: torch.Tensor, blank_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given logits, calculate log probabilities for blank and target labels needed for transducer loss calculation.
    Naive implementation in PyTorch, for testing and prototyping purposes.

    Args:
        logits: Joint tensor of size [B, T, U+1, D]
        targets: Targets of size [B, U]
        blank_id: id of the blank output

    Returns:
        Tuple of tensors with log probabilities for targets and blank labels, both of size [B, T, U+1].
        For the last non-existent target (U+1) output is zero.
    """
    device = logits.device
    batch_size = logits.shape[0]
    log_probs = F.log_softmax(logits, dim=-1)
    blank_scores = log_probs[..., blank_id]
    targets = torch.cat((targets, torch.zeros([batch_size], dtype=targets.dtype, device=device).unsqueeze(1)), dim=-1)
    target_scores = torch.gather(
        log_probs, dim=-1, index=targets.unsqueeze(1).expand(log_probs.shape[:-1]).unsqueeze(-1)
    ).squeeze(-1)
    target_scores[:, :, -1] = 0.0
    return target_scores, blank_scores
