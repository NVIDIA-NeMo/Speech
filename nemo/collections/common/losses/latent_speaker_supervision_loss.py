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

from typing import List, Optional

import torch

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LogprobsType, LossType, MaskType, NeuralType

__all__ = ["LatentSpeakerSupervisionLoss"]


class LatentSpeakerSupervisionLoss(Loss):
    """
    Combined loss for SOT (Serialized Output Training) multi-speaker ASR that provides:

        1. Standard smoothed cross-entropy loss on **segment-level** SOT labels
        2. Auxiliary per-word speaker classification loss (latent speaker supervision)

    Motivation:
        In SOT-based multi-speaker ASR, there are two labeling strategies:

        - **Word-level SOT**: A speaker token precedes every word.
          e.g. ``<spk:0> hi <spk:0> how <spk:0> are <spk:0> you <spk:1> I <spk:1> am ...``
          (+) More speaker-label gradients  (-) Very token-inefficient

        - **Segment-level SOT**: Speaker tokens appear only at speaker changes.
          e.g. ``<spk:0> hi how are you <spk:1> I am good thanks``
          (+) Token-efficient  (-) Fewer speaker-label gradients

        This loss gives the **best of both worlds**: the labels remain segment-level
        (efficient token throughput), but an auxiliary loss provides speaker supervision
        at every word position (strong speaker gradients), without generating extra
        speaker tokens at inference time.

    How it works:
        For each word position in the label sequence, the "active speaker" is determined
        by forward-filling the last observed speaker token. The model's log-probabilities
        for the speaker token vocabulary are extracted, renormalized, and a cross-entropy
        loss is computed against the active speaker. This is combined with the standard
        CE loss via a configurable weight.

    Args:
        speaker_token_ids: List of token IDs for speaker tokens
            (e.g., IDs for ``<spk:0>``, ``<spk:1>``, ``<spk:2>``, ``<spk:3>``).
        speaker_loss_weight: Weight for the auxiliary speaker supervision loss.
        include_ce_loss: If True (default), the forward pass returns standard CE + speaker loss.
            If False, returns **only** the auxiliary speaker loss (useful when a separate
            CE loss module is already applied, e.g. ``self.loss`` in the model).
        pad_id: Padding token ID (used to create output_mask if not provided).
        label_smoothing: Label smoothing coefficient for the standard CE loss (only used when
            ``include_ce_loss=True``).
        speaker_label_smoothing: Label smoothing coefficient for the speaker CE loss.
        eps: Small constant to avoid division by zero.
        per_token_reduction: If True, reduces loss per token; if False, returns per-token losses.
        per_speaker_normalization: If True, the speaker supervision loss is computed as the
            mean of per-speaker average losses, giving equal importance to each speaker
            regardless of how many words they have. For example, if speaker 0 has 2 words
            and speaker 1 has 20 words, each speaker still contributes 50% of the loss.
            Default: True.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {
            "log_probs": NeuralType(("B", "T", "D"), LogprobsType()),
            "labels": NeuralType(("B", "T"), LabelsType()),
            "output_mask": NeuralType(("B", "T"), MaskType(), optional=True),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        speaker_token_ids: List[int],
        speaker_loss_weight: float = 1.0,
        include_ce_loss: bool = False,
        pad_id: Optional[int] = None,
        label_smoothing: float = 0.0,
        speaker_label_smoothing: float = 0.0,
        eps: float = 1e-6,
        per_token_reduction: bool = True,
        per_speaker_normalization: bool = True,
    ):
        super().__init__()
        self.register_buffer("speaker_token_ids", torch.tensor(speaker_token_ids, dtype=torch.long))
        self.num_speakers = len(speaker_token_ids)
        self.speaker_loss_weight = speaker_loss_weight
        self.include_ce_loss = include_ce_loss
        self._pad_id = pad_id
        self._label_smoothing = label_smoothing
        self._speaker_label_smoothing = speaker_label_smoothing
        self._eps = eps
        self._per_token_reduction = per_token_reduction
        self._per_speaker_normalization = per_speaker_normalization

    def _compute_standard_ce(self, log_probs, labels, output_mask):
        """
        Standard smoothed cross-entropy loss, identical to SmoothedCrossEntropyLoss.

        Args:
            log_probs: (B, T, V) log-probabilities over vocabulary.
            labels: (B, T) target token IDs.
            output_mask: (B, T) binary mask.

        Returns:
            Scalar loss (if per_token_reduction) or (B, T) per-token losses.
        """
        vocab_size = log_probs.size(2)
        smoothing = vocab_size * self._label_smoothing / (vocab_size - 1)
        target_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)
        smoothing_log_probs = log_probs.mean(dim=-1)
        neg_log_likelihood = (1.0 - smoothing) * target_log_probs + smoothing * smoothing_log_probs

        if self._per_token_reduction:
            neg_log_likelihood = -torch.sum(neg_log_likelihood * output_mask)
            neg_log_likelihood = neg_log_likelihood / (output_mask.sum() + self._eps)
        else:
            neg_log_likelihood = -(neg_log_likelihood * output_mask)

        return neg_log_likelihood

    def _get_active_speaker_per_position(self, labels):
        """
        For each position in the label sequence, determine the active speaker
        by forward-filling the last observed speaker token.

        Uses a vectorized cummax approach for efficiency (no Python loops over T).

        Args:
            labels: (B, T) target token IDs.

        Returns:
            active_speaker: (B, T) tensor with 0-based speaker index at each position.
                            -1 at positions before the first speaker token.
            speaker_mask:   (B, T) boolean tensor, True at speaker-token positions.
            word_mask:      (B, T) boolean tensor, True at non-speaker-token positions
                            that have a valid active speaker.
        """
        batch_size, seq_len = labels.shape
        device = labels.device

        # Step 1: Identify speaker token positions and their speaker indices.
        # speaker_idx_plus1 stores (speaker_index + 1) at speaker positions, 0 elsewhere.
        # The +1 offset distinguishes "speaker 0" from "no speaker".
        speaker_idx_plus1 = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
        speaker_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)

        for idx in range(self.num_speakers):
            match = labels == self.speaker_token_ids[idx]
            speaker_mask |= match
            speaker_idx_plus1[match] = idx + 1

        # Step 2: Forward-fill using cummax on position indices.
        # At speaker positions, store the time-step index; elsewhere, store -1.
        # cummax propagates the largest seen position forward, which is always the
        # most recent speaker-token position.
        positions = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        speaker_positions = torch.where(speaker_mask, positions, torch.tensor(-1, device=device, dtype=torch.long))
        last_speaker_pos, _ = speaker_positions.cummax(dim=1)  # (B, T)

        # Step 3: Gather the speaker index at the last speaker position.
        last_speaker_pos_clamped = last_speaker_pos.clamp(min=0)
        active_speaker = speaker_idx_plus1.gather(1, last_speaker_pos_clamped) - 1  # back to 0-based

        # Positions before the first speaker token have last_speaker_pos == -1.
        active_speaker[last_speaker_pos < 0] = -1

        # Word mask: non-speaker positions with a valid active speaker.
        word_mask = ~speaker_mask & (active_speaker >= 0)

        return active_speaker, speaker_mask, word_mask

    def _compute_speaker_loss(self, log_probs, active_speaker, word_mask):
        """
        Compute auxiliary speaker classification loss at word positions.

        At each word position, the log-probabilities for speaker tokens are extracted
        from the full vocabulary logits, renormalized (conditional probability among
        speakers only), and cross-entropy is computed against the active speaker.

        When ``per_speaker_normalization`` is enabled, the loss for each speaker is
        averaged independently, then the per-speaker losses are averaged together.
        This prevents speakers with many words from dominating the gradient.

        Args:
            log_probs: (B, T, V) log-probabilities over vocabulary.
            active_speaker: (B, T) 0-based speaker index at each position.
            word_mask: (B, T) boolean mask for positions where speaker loss applies.

        Returns:
            Scalar speaker loss.
        """
        # Extract log-probs for speaker tokens only: (B, T, num_speakers)
        speaker_log_probs = log_probs[:, :, self.speaker_token_ids]

        # Renormalize over speaker tokens (conditional probability given speaker-token subset).
        speaker_log_probs = speaker_log_probs - torch.logsumexp(speaker_log_probs, dim=-1, keepdim=True)

        # Clamp active_speaker to valid range for gather (invalid positions are masked out anyway).
        active_speaker_clamped = active_speaker.clamp(min=0).unsqueeze(2)  # (B, T, 1)

        if self._speaker_label_smoothing > 0:
            smoothing = self.num_speakers * self._speaker_label_smoothing / (self.num_speakers - 1)
            target_log_probs = speaker_log_probs.gather(2, active_speaker_clamped).squeeze(2)
            smoothing_log_probs = speaker_log_probs.mean(dim=-1)
            speaker_nll = (1.0 - smoothing) * target_log_probs + smoothing * smoothing_log_probs
        else:
            speaker_nll = speaker_log_probs.gather(2, active_speaker_clamped).squeeze(2)

        word_mask_float = word_mask.to(log_probs.dtype)

        if self._per_speaker_normalization:
            # Average loss per speaker, then average across speakers.
            # This gives equal weight to each speaker regardless of word count.
            per_speaker_losses = []
            speakers_present = active_speaker[word_mask].unique()
            for spk_id in speakers_present:
                spk_mask = word_mask & (active_speaker == spk_id)
                spk_mask_float = spk_mask.to(log_probs.dtype)
                spk_count = spk_mask_float.sum()
                if spk_count > 0:
                    spk_loss = -torch.sum(speaker_nll * spk_mask_float) / (spk_count + self._eps)
                    per_speaker_losses.append(spk_loss)
            if per_speaker_losses:
                speaker_loss = torch.stack(per_speaker_losses).mean()
            else:
                speaker_loss = torch.tensor(0.0, device=log_probs.device, dtype=log_probs.dtype)
        else:
            speaker_loss = -torch.sum(speaker_nll * word_mask_float) / (word_mask_float.sum() + self._eps)

        return speaker_loss

    @typecheck()
    def forward(
        self,
        log_probs,
        labels=None,
        output_mask=None,
    ):
        """
        Compute the combined loss: standard CE + weighted latent speaker supervision.

        Args:
            log_probs: (B, T, V) log-probabilities over vocabulary.
            labels: (B, T) target token IDs in segment-level SOT format.
            output_mask: (B, T) binary mask (optional; computed from pad_id if None).

        Returns:
            Scalar combined loss.
        """
        if output_mask is None and self._pad_id is None:
            raise ValueError("Both output_mask and pad_id are None")
        if output_mask is None and self._pad_id is not None:
            output_mask = (labels != self._pad_id).to(log_probs.dtype)
        if output_mask.dtype is not log_probs.dtype:
            output_mask = output_mask.to(log_probs.dtype)

        # 1. Standard cross-entropy loss (on the full segment-level label sequence).
        if self.include_ce_loss:
            ce_loss = self._compute_standard_ce(log_probs, labels, output_mask)
        else:
            ce_loss = None

        # 2. Latent speaker supervision loss (auxiliary, at word positions only).
        active_speaker, _speaker_mask, word_mask = self._get_active_speaker_per_position(labels)

        # Intersect word_mask with the output_mask so we don't compute speaker loss on padding.
        word_mask = word_mask & (output_mask > 0)

        if word_mask.any():
            speaker_loss = self._compute_speaker_loss(log_probs, active_speaker, word_mask)
        else:
            speaker_loss = torch.tensor(0.0, device=log_probs.device, dtype=log_probs.dtype)

        # 3. Combine.
        if ce_loss is not None:
            total_loss = ce_loss + self.speaker_loss_weight * speaker_loss
        else:
            total_loss = self.speaker_loss_weight * speaker_loss

        return total_loss
