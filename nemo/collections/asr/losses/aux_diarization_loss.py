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

import torch
import torch.nn.functional as F


def activity_loss(
    activity_logits: torch.Tensor,
    targets: torch.Tensor,
    target_lens: torch.Tensor,
) -> torch.Tensor:
    """Compute three-class activity loss over valid frames.

    Speaker targets are converted to mutually exclusive silence, single-speaker,
    and overlap classes without applying PIL or ATS speaker alignment. Padded
    frames are excluded, and cross-entropy is evaluated in FP32.

    Args:
        activity_logits: Raw activity logits with shape ``(B, T, 3)``.
        targets: Unpermuted speaker targets with shape ``(B, T, S)``.
        target_lens: Number of valid frames per batch item with shape ``(B,)``.

    Returns:
        Mean cross-entropy over valid frames as an FP32 scalar.
    """
    activity_targets = (targets > 0.5).sum(dim=-1).clamp(max=2).long()
    valid_frames = torch.arange(activity_logits.shape[1], device=activity_logits.device).unsqueeze(0) < target_lens.to(
        activity_logits.device
    ).unsqueeze(1)
    with torch.autocast(device_type=activity_logits.device.type, enabled=False):
        frame_losses = F.cross_entropy(
            activity_logits.float().transpose(1, 2),
            activity_targets,
            reduction="none",
        )
        loss = (frame_losses * valid_frames).sum() / valid_frames.sum().clamp_min(1).float()
    return loss


def phantom_loss(
    logits: torch.Tensor,
    phantom_targets: torch.Tensor,
    target_lens: torch.Tensor,
    threshold: float,
    temperature: float,
) -> torch.Tensor:
    """Penalize ``phantom`` speakers predicted in channels with no target speaker.

    Here a phantom speaker means high-confidence speaker activity assigned to an
    output channel that is never associated with a target speaker during the valid
    segment. Within such channels, frames whose detached probability exceeds
    ``threshold`` contribute negative-label BCE from the raw logits. Those frame
    losses are reduced per channel with temperature-scaled normalized log-mean-exp,
    then summed over channels, divided by the fixed speaker-slot count, and averaged
    across the batch. Padded and unselected frames have no gradient.

    Args:
        logits: Raw speaker logits with shape ``(B, T, S)``.
        phantom_targets: Aligned targets used to detect empty channels, with shape
            ``(B, T, S)``.
        target_lens: Number of valid frames per batch item with shape ``(B,)``.
        threshold: Speaker probability above which eligible frames are selected.
        temperature: Temperature used by the normalized log-mean-exp reduction.

    Returns:
        The FP32 batch-mean phantom loss.
    """
    num_frames, num_spks = logits.shape[1], logits.shape[2]
    if num_frames == 0:
        return logits.float().sum() * 0.0

    valid_frames = torch.arange(num_frames, device=logits.device).unsqueeze(0) < target_lens.to(
        logits.device
    ).unsqueeze(1)
    valid_frame_mask = valid_frames.unsqueeze(-1)

    # A channel is phantom-eligible only if no valid target frame activates it.
    target_empty_channels = ~((phantom_targets > 0.5) & valid_frame_mask).any(dim=1)

    # Detach threshold selection while preserving gradients through the selected logits below.
    detached_probs = torch.sigmoid(logits.detach())
    selected_phantom_frames = valid_frame_mask & target_empty_channels.unsqueeze(1) & (detached_probs > threshold)

    with torch.autocast(device_type=logits.device.type, enabled=False):
        # softplus(z) is BCEWithLogits(z, target=0) for each frame and speaker channel.
        negative_bce = F.softplus(logits.float())
        selected_frame_count = selected_phantom_frames.sum(dim=1)
        scaled_negative_bce = negative_bce / temperature

        # In log space, -inf removes unselected frames from logsumexp exactly.
        selected_scaled_bce = scaled_negative_bce.masked_fill(~selected_phantom_frames, float('-inf'))
        has_selected_frames = selected_frame_count > 0

        # Give empty selections one finite dummy value so logsumexp and its gradients stay finite.
        safe_first_frame = torch.where(
            has_selected_frames.unsqueeze(1),
            selected_scaled_bce[:, :1, :],
            torch.zeros_like(selected_scaled_bce[:, :1, :]),
        )
        selected_scaled_bce = torch.cat((safe_first_frame, selected_scaled_bce[:, 1:, :]), dim=1)

        # Normalize over selected frames, then make channels without selections exact zeros.
        per_channel_loss = temperature * (
            torch.logsumexp(selected_scaled_bce, dim=1)
            - torch.log(selected_frame_count.clamp_min(1).to(negative_bce.dtype))
        )
        per_channel_loss = torch.where(
            has_selected_frames,
            per_channel_loss,
            torch.zeros_like(per_channel_loss),
        )
        phantom_loss = (per_channel_loss.sum(dim=1) / num_spks).mean()
    return phantom_loss
