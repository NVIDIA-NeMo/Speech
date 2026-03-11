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

from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from nemo.utils import logging
from nemo.utils.nemo_logging import LogMode

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs_triton import rnnt_logprobs_triton

def get_rnnt_mask(
        batch_size: int,
        src_length_max: int,
        tgt_length_max_plus_1: int,
        device: torch.device,
        src_lengths: torch.Tensor | None = None,
        tgt_lengths: torch.Tensor | None = None):
    if src_lengths is not None:
        mask_src = torch.arange(src_length_max, device=device)[None, :] < src_lengths[:, None]
    else:
        mask_src = torch.ones([batch_size, src_length_max], dtype=torch.bool, device=device)
    if tgt_lengths is not None:
        mask_tgt_nb = torch.arange(tgt_length_max_plus_1, device=device)[None, :] < tgt_lengths[:, None]
        mask_tgt_blank = (
                torch.arange(tgt_length_max_plus_1, device=device)[None, :] < (tgt_lengths[:, None] + 1)
        )
    else:
        mask_tgt_nb = torch.ones([batch_size, tgt_length_max_plus_1], dtype=torch.bool, device=device)
        mask_tgt_nb[:, -1] = False
        mask_tgt_blank = torch.ones([batch_size, tgt_length_max_plus_1], dtype=torch.bool, device=device)
    mask_nb = mask_src[..., None] * mask_tgt_nb[:, None, :]
    mask_blank = mask_src[..., None] * mask_tgt_blank[:, None, :]
    return mask_nb, mask_blank


def rnnt_logprobs_torch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    src_lengths: torch.Tensor | None = None,
    tgt_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given logits, calculate log probabilities for blank and target labels needed for transducer loss calculation.
    Naive implementation in PyTorch, for testing and prototyping purposes.

    Args:
        logits: Joint tensor of size [B, T, U+1, D]
        targets: Targets of size [B, U]
        blank_id: id of the blank output
        src_lengths: optional tensor with lengths for source utterances
        tgt_lengths: optional tensor with lengths for targets

    Returns:
        Tuple of tensors with log probabilities for targets and blank labels, both of size [B, T, U+1].
        For the last non-existent target (U+1) output is zero.
    """
    device = logits.device
    batch_size = logits.shape[0]
    tgt_length_max = logits.shape[2] - 1
    log_probs = F.log_softmax(logits, dim=-1)
    blank_scores = log_probs[..., blank_id]
    # Truncate targets to match the U dimension of logits (handles padded targets when tgt_length=0)
    targets = targets[:, :tgt_length_max]
    targets = torch.cat((targets, torch.zeros([batch_size, 1], dtype=targets.dtype, device=device)), dim=-1)
    target_scores = torch.gather(
        log_probs, dim=-1, index=targets.unsqueeze(1).expand(log_probs.shape[:-1]).unsqueeze(-1)
    ).squeeze(-1)
    target_scores[:, :, -1] = 0.0
    if src_lengths is not None or tgt_lengths is not None:
        batch_size, src_length_max, tgt_length_max_plus_1, _ = logits.shape
        mask_nb, mask_blank = get_rnnt_mask(
            batch_size=batch_size,
            src_length_max=src_length_max,
            tgt_length_max_plus_1=tgt_length_max_plus_1,
            device=device,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
        )
        target_scores = target_scores * mask_nb
        blank_scores = blank_scores * mask_blank
    return target_scores, blank_scores


def rnnt_logprobs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    src_lengths: torch.Tensor | None = None,
    tgt_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given logits, calculate log probabilities for blank and target labels needed for transducer loss calculation.

    Args:
        logits: Joint tensor of size [B, T, U+1, D]
        targets: Targets of size [B, U]
        blank_id: id of the blank output
        src_lengths: optional tensor with lengths for source utterances
        tgt_lengths: optional tensor with lengths for targets

    Returns:
        Tuple of tensors with log probabilities for targets and blank labels, both of size [B, T, U+1].
        For the last non-existent target (U+1) output is zero.
    """
    device: torch.device = logits.device
    if TRITON_AVAILABLE and device.type == "cuda":
        return rnnt_logprobs_triton(
            logits=logits,
            targets=targets,
            blank_id=blank_id,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
        )
    logging.warning(
        "Triton is unavailable, pure PyTorch implementation of `rnnt_logprobs` can use a lot of extra memory."
        " Install triton for using memory-efficient implementation",
        mode=LogMode.ONCE,
    )
    return rnnt_logprobs_torch(
        logits=logits,
        targets=targets,
        blank_id=blank_id,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )
