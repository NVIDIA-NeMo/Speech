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

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.asr.parts.turbo_transducer.backend import TurboTransducerBackend
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import get_rnnt_mask, rnnt_logprobs, rnnt_logprobs_torch
from nemo.collections.asr.parts.turbo_transducer.types import TransducerLossInputs
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum
from nemo.utils.nemo_logging import LogMode

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency_triton import kl_loss_triton
    from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs_triton import rnnt_logprobs_with_consistency_triton


class ConsistencyRNNTReductionType(PrettyStrEnum):
    MEAN = "mean"
    MEAN_VOLUME = "mean_volume"


def kl_loss_torch(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    mask: torch.Tensor,
    symmetrical: bool,
) -> torch.Tensor:
    """
    Compute masked KL divergence loss over probability distributions.

    Args:
        teacher_logprobs: Teacher log probabilities [B, T, U+1, V] where V is vocab size
        student_logprobs: Student log probabilities [B, T, U+1, V]
        mask: Mask tensor [B, T, U+1]
        symmetrical: Whether to use symmetric KL (average of both directions)

    Returns:
        KL loss tensor [B, T, U+1] with mask applied
    """
    # F.kl_div computes sum over last dim when reduction='none', but we want per-position
    # So we sum over the vocab dimension (K) ourselves
    kl_s_to_t = F.kl_div(
        input=student_logprobs,
        target=teacher_logprobs.detach(),
        reduction='none',
        log_target=True,
    ).sum(
        dim=-1
    )  # Sum over vocab dimension

    if symmetrical:
        kl_t_to_s = F.kl_div(
            input=teacher_logprobs,
            target=student_logprobs.detach(),
            reduction='none',
            log_target=True,
        ).sum(dim=-1)
        return 0.5 * (kl_s_to_t + kl_t_to_s) * mask
    return kl_s_to_t * mask


def rnnt_logprobs_with_consistency(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    src_lengths: torch.Tensor | None = None,
    tgt_lengths: torch.Tensor | None = None,
    compute_consistency: bool = False,
    consistency_symmetrical: bool = True,
) -> TransducerLossInputs:
    """
    Given logits, calculate the log probabilities for blank and target labels needed for the transducer loss
    and, optionally, the per-position consistency (KL) loss.

    When `compute_consistency` is set, the logits are expected to represent a double batch: rows `[0, B)`
    are the teacher pass and rows `[B, 2B)` are the student pass of the same utterances. Both halves must
    have identical `src_lengths` / `tgt_lengths` (this is not checked to avoid host-device synchronization).
    Fusing both losses into a single autograd function allows using one gradient buffer for the logits.

    Args:
        logits: Joint tensor of size [B, T, U+1, V], raw logits (not after log-softmax)
        targets: Targets of size [B, U]
        blank_id: id of the blank output
        src_lengths: optional tensor with lengths for source utterances
        tgt_lengths: optional tensor with lengths for targets
        compute_consistency: whether to compute the consistency loss between the halves of the batch
        consistency_symmetrical: if True, compute symmetric KL divergence for the consistency loss

    Returns:
        `TransducerLossInputs` with target/blank log probabilities of size [B, T, U+1] and, if requested,
        the consistency loss of size [B // 2, T, U+1].
    """
    device = logits.device
    batch_size, src_length_max, tgt_length_max_plus_1, _ = logits.shape
    if src_lengths is None:
        src_lengths = torch.full([batch_size], fill_value=src_length_max, dtype=torch.long, device=device)
    if tgt_lengths is None:
        tgt_lengths = torch.full([batch_size], fill_value=tgt_length_max_plus_1 - 1, dtype=torch.long, device=device)

    if not compute_consistency:
        target_logprobs, blank_logprobs = rnnt_logprobs(
            logits=logits, targets=targets, blank_id=blank_id, src_lengths=src_lengths, tgt_lengths=tgt_lengths
        )
        return TransducerLossInputs(target_logprobs=target_logprobs, blank_logprobs=blank_logprobs)

    if batch_size % 2 != 0:
        raise ValueError(f"Consistency loss requires a double batch, got odd batch size {batch_size}")
    half_batch_size = batch_size // 2
    _, consistency_mask = get_rnnt_mask(
        batch_size=half_batch_size,
        src_length_max=src_length_max,
        tgt_length_max_plus_1=tgt_length_max_plus_1,
        device=device,
        src_lengths=src_lengths[:half_batch_size],
        tgt_lengths=tgt_lengths[:half_batch_size],
    )

    if TRITON_AVAILABLE and device.type == "cuda":
        return rnnt_logprobs_with_consistency_triton(
            logits=logits,
            targets=targets,
            blank_id=blank_id,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            consistency_mask=consistency_mask,
            consistency_symmetrical=consistency_symmetrical,
        )

    logging.warning(
        "Triton is unavailable, pure PyTorch implementation of `rnnt_logprobs_with_consistency` can use a lot"
        " of extra memory. Install triton for using memory-efficient implementation",
        mode=LogMode.ONCE,
    )
    target_logprobs, blank_logprobs = rnnt_logprobs_torch(
        logits=logits, targets=targets, blank_id=blank_id, src_lengths=src_lengths, tgt_lengths=tgt_lengths
    )
    consistency_kl = kl_loss_torch(
        teacher_logprobs=F.log_softmax(logits[:half_batch_size], dim=-1),
        student_logprobs=F.log_softmax(logits[half_batch_size:], dim=-1),
        mask=consistency_mask,
        symmetrical=consistency_symmetrical,
    )
    return TransducerLossInputs(
        target_logprobs=target_logprobs, blank_logprobs=blank_logprobs, consistency_kl=consistency_kl
    )


class ConsistencyRNNTLoss(nn.Module):
    def __init__(
        self,
        symmetrical: bool = True,
        reduction: str | ConsistencyRNNTReductionType = 'mean_volume',
        backend: str | TurboTransducerBackend | None = None,
    ):
        super().__init__()
        self.reduction = ConsistencyRNNTReductionType(reduction)
        self.symmetrical = symmetrical
        self._backend = TurboTransducerBackend(backend) if backend is not None else None  # None means "auto"

    def get_runtime_backend(self, input_: torch.Tensor) -> TurboTransducerBackend:
        if self._backend is not None:
            # backend if forced
            return self._backend
        if input_.device.type != "cuda":
            return TurboTransducerBackend.TORCH
        return TurboTransducerBackend.TRITON if TRITON_AVAILABLE else TurboTransducerBackend.TORCH

    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        src_lengths: torch.Tensor | None = None,
        tgt_lengths: torch.Tensor | None = None,
    ):
        device = teacher_logits.device
        batch_size, src_length_max, tgt_length_max_plus_1, _ = teacher_logits.shape
        tgt_length_max = tgt_length_max_plus_1 - 1

        if src_lengths is None:
            src_lengths = torch.full([batch_size], fill_value=src_length_max, dtype=torch.long, device=device)
        if tgt_lengths is None:
            tgt_lengths = torch.full([batch_size], fill_value=tgt_length_max, dtype=torch.long, device=device)

        _, mask_blank = get_rnnt_mask(
            batch_size=batch_size,
            src_length_max=src_length_max,
            tgt_length_max_plus_1=tgt_length_max_plus_1,
            device=device,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
        )
        mask = mask_blank

        runtime_backend = self.get_runtime_backend(teacher_logits)
        if runtime_backend is TurboTransducerBackend.TRITON:
            kl_loss = kl_loss_triton(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                mask=mask,
                symmetrical=self.symmetrical,
            )
        elif runtime_backend is TurboTransducerBackend.TORCH:
            kl_loss = kl_loss_torch(
                teacher_logprobs=F.log_softmax(teacher_logits, dim=-1),
                student_logprobs=F.log_softmax(student_logits, dim=-1),
                mask=mask,
                symmetrical=self.symmetrical,
            )
        else:
            raise NotImplementedError(f"Unsupported backend {runtime_backend}")

        return self.forward_from_kl(consistency_kl=kl_loss, src_lengths=src_lengths, tgt_lengths=tgt_lengths)

    def forward_from_kl(
        self,
        consistency_kl: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reduce the per-position consistency loss to a scalar.

        Args:
            consistency_kl: per-position consistency loss of size [B, T, U+1], zero outside valid positions
            src_lengths: source lengths of size [B]
            tgt_lengths: target lengths of size [B]

        Returns:
            scalar consistency loss
        """
        _, src_length_max, tgt_length_max_plus_1 = consistency_kl.shape
        # number of valid (non-masked) positions per utterance; equals `mask_blank.sum(dim=(1, 2))`
        num_positions = src_lengths.clamp(max=src_length_max) * (tgt_lengths + 1).clamp(max=tgt_length_max_plus_1)
        match self.reduction:
            case ConsistencyRNNTReductionType.MEAN:
                return consistency_kl.sum() / num_positions.sum().clamp(min=1)
            case ConsistencyRNNTReductionType.MEAN_VOLUME:
                return (consistency_kl.sum(dim=(1, 2)) / num_positions.clamp(min=1)).mean()
            case _:
                raise NotImplementedError(f"Unsupported reduction {self.reduction}")
