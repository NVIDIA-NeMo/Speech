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

from nemo.collections.asr.parts.rnnt_triton.rnnt_logprobs import get_rnnt_mask, rnnt_logprobs
from nemo.core.utils.optional_libs import K2_AVAILABLE, TRITON_AVAILABLE
from nemo.utils.enum import PrettyStrEnum

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.rnnt_triton.rnnt_consistency_triton import kl_loss_triton

EPS = 1e-5


class ConsistencyRNNTReductionType(PrettyStrEnum):
    MEAN = "mean"
    MEAN_VOLUME = "mean_volume"
    P_NON_BLANK = "p_non_blank"
    P_NON_BLANK_WITH_GRAD = "p_non_blank_with_grad"


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Compute log(1 - exp(x)) in a numerically stable way.

    Uses log1p(-exp(x)) which is more stable than log(1 - exp(x))
    when x is close to 0 (i.e., when exp(x) is close to 1).
    """
    return torch.log1p(-torch.exp(x))


def _build_log_distribution(
    target_logprobs: torch.Tensor,
    blank_logprobs: torch.Tensor | None = None,
    complete_distribution: bool = True,
    min_log_prob: float = -1e3,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Build log probability distribution staying in log space for numerical stability.

    Args:
        target_logprobs: Log probabilities for target tokens [B, T, U+1]
        blank_logprobs: Log probabilities for blank token [B, T, U+1], or None

    Returns:
        Log probabilities of shape [B, T, U+1, 2] if blank_logprobs is None,
        or [B, T, U+1, 3] if blank_logprobs is provided.
        Distribution: [target, rest] or [target, blank, rest]
    """
    if not complete_distribution:
        if blank_logprobs is None:
            return target_logprobs[..., None]
        return torch.stack([target_logprobs, blank_logprobs], dim=-1)
    if blank_logprobs is not None:
        # rest_logprob = log(1 - exp(target) - exp(blank))
        #              = log(1 - exp(logsumexp([target, blank])))
        #              = log1mexp(logsumexp([target, blank]))
        sum_logprobs = torch.logsumexp(torch.stack([target_logprobs, blank_logprobs], dim=-1), dim=-1)
        rest_logprobs = _log1mexp(sum_logprobs).clamp(min=min_log_prob, max=-eps)
        log_dist = torch.stack([target_logprobs, blank_logprobs, rest_logprobs], dim=-1)
    else:
        # rest_logprob = log(1 - exp(target)) = log1mexp(target)
        rest_logprobs = _log1mexp(target_logprobs).clamp(min=min_log_prob, max=-eps)
        log_dist = torch.stack([target_logprobs, rest_logprobs], dim=-1)

    return log_dist


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


def consistency_rnnt_kld(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    src_lengths: torch.Tensor | None = None,
    tgt_lengths: torch.Tensor | None = None,
    symmetrical: bool = True,
    use_blank: bool = False,
    complete_distribution: bool = True,
    min_log_prob: float = -1e3,
    eps: float = 1e-5,
    reduction: str | ConsistencyRNNTReductionType = 'mean_volume',  # 'mean' or 'mean_volume'
) -> torch.Tensor:
    """
    Compute Consistency-Regularized RNN-T KL Divergence loss using targets and (optional) blank probabilities.

    Args:
        teacher_logits: Logits from teacher (offline) mode [B, T, U+1, V]
        student_logits: Logits from student (streaming) mode [B, T, U+1, V]
        targets: Target token indices [B, U]
        blank_id: Index of the blank token in the vocabulary
        src_lengths: Optional source (encoder) sequence lengths [B]
        tgt_lengths: Optional target sequence lengths [B]
        symmetrical: If True, compute symmetric KL (average of both directions)
        use_blank: If True, include blank probabilities in the distribution
        complete_distribution: If True, build complete probability distributions that sum to 1
        min_log_prob: Minimum log probability value for numerical stability
        eps: Small epsilon for numerical stability in clamping
        reduction: 'mean' (normalize by frames) or 'mean_volume' (normalize per-sample then average)

    Returns:
        Scalar KL divergence loss
    """
    reduction = ConsistencyRNNTReductionType(reduction)
    assert teacher_logits.shape == student_logits.shape
    batch_size, src_length_max, tgt_length_max_plus_1, _ = teacher_logits.shape
    device = teacher_logits.device
    teacher_target_logprobs, teacher_blank_logprobs = rnnt_logprobs(
        logits=teacher_logits,
        targets=targets,
        blank_id=blank_id,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )
    student_target_logprobs, student_blank_logprobs = rnnt_logprobs(
        logits=student_logits,
        targets=targets,
        blank_id=blank_id,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )
    mask_nb, mask_blank = get_rnnt_mask(
        batch_size=batch_size,
        src_length_max=src_length_max,
        tgt_length_max_plus_1=tgt_length_max_plus_1,
        device=device,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )

    # clean up logprobs
    teacher_target_logprobs = torch.where(mask_nb, teacher_target_logprobs, min_log_prob)
    teacher_blank_logprobs = torch.where(mask_blank, teacher_blank_logprobs, min_log_prob)
    student_target_logprobs = torch.where(mask_nb, student_target_logprobs, min_log_prob)
    student_blank_logprobs = torch.where(mask_blank, student_blank_logprobs, min_log_prob)

    primary_mask = mask_blank if use_blank else mask_nb

    # Build proper probability distributions that sum to 1
    # This ensures KL divergence is always non-negative
    teacher_log_dist = _build_log_distribution(
        target_logprobs=teacher_target_logprobs,
        blank_logprobs=teacher_blank_logprobs if use_blank else None,
        complete_distribution=complete_distribution,
        min_log_prob=min_log_prob,
        eps=eps,
    )
    student_log_dist = _build_log_distribution(
        target_logprobs=student_target_logprobs,
        blank_logprobs=student_blank_logprobs if use_blank else None,
        complete_distribution=complete_distribution,
        min_log_prob=min_log_prob,
        eps=eps,
    )

    kl_loss = kl_loss_torch(
        teacher_logprobs=teacher_log_dist,
        student_logprobs=student_log_dist,
        mask=primary_mask,
        symmetrical=symmetrical,
    )

    match reduction:
        case ConsistencyRNNTReductionType.MEAN:
            kl_loss_value = kl_loss.sum() / primary_mask.sum().clamp(min=1)
        case ConsistencyRNNTReductionType.MEAN_VOLUME:
            kl_loss_value = (kl_loss.sum(dim=(1, 2)) / primary_mask.sum(dim=(1, 2)).clamp(min=1)).mean()
        case _:
            raise NotImplementedError(f"Unsupported reduction {reduction}")

    return kl_loss_value


class ConsistencyRNNTLoss(nn.Module):
    def __init__(
        self,
        blank_id: int,
        symmetrical: bool = True,
        use_blank: bool = False,
        reduction: str | ConsistencyRNNTReductionType = 'mean_volume',
        complete_distribution: bool = True,
        min_log_prob: float = -1e3,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.reduction = ConsistencyRNNTReductionType(reduction)
        self.use_blank = use_blank
        self.blank_id = blank_id
        self.symmetrical = symmetrical
        self.complete_distribution = complete_distribution
        self.min_log_prob = min_log_prob
        self.eps = eps

    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        src_lengths: torch.Tensor | None = None,
        tgt_lengths: torch.Tensor | None = None,
    ):
        return consistency_rnnt_kld(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            targets=targets,
            blank_id=self.blank_id,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            symmetrical=self.symmetrical,
            use_blank=self.use_blank,
            complete_distribution=self.complete_distribution,
            min_log_prob=self.min_log_prob,
            eps=self.eps,
            reduction=self.reduction,
        )


class ConsistencyFullRNNTLoss(nn.Module):
    def __init__(
        self,
        symmetrical: bool = True,
        use_triton: bool | None = None,  # None -> auto
        reduction: str | ConsistencyRNNTReductionType = 'mean_volume',
        blank_id: int | None = None,
    ):
        super().__init__()
        self.reduction = ConsistencyRNNTReductionType(reduction)
        self.symmetrical = symmetrical
        self.use_triton = TRITON_AVAILABLE if use_triton is None else use_triton
        self.blank_id = blank_id

    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        targets: torch.Tensor | None = None,  # not needed, but keep consistent for now with `ConsistencyRNNTLoss`
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

        if self.use_triton and device.type == "cuda":
            kl_loss = kl_loss_triton(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                mask=mask,
                symmetrical=self.symmetrical,
                blank_id=self.blank_id,
                weighted=(
                    str(self.reduction)
                    if self.reduction
                    in {ConsistencyRNNTReductionType.P_NON_BLANK, ConsistencyRNNTReductionType.P_NON_BLANK_WITH_GRAD}
                    else None
                ),
            )

            match self.reduction:
                case ConsistencyRNNTReductionType.MEAN:
                    kl_loss_value = kl_loss.sum() / mask.sum().clamp(min=1)
                case ConsistencyRNNTReductionType.MEAN_VOLUME:
                    kl_loss_value = (kl_loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)).mean()
                case ConsistencyRNNTReductionType.P_NON_BLANK:
                    assert kl_loss.dim() == 1  # already reduced
                    kl_loss_value = kl_loss.mean()
                case ConsistencyRNNTReductionType.P_NON_BLANK_WITH_GRAD:
                    assert kl_loss.dim() == 1  # already reduced
                    kl_loss_value = kl_loss.mean()
                case _:
                    raise NotImplementedError(f"Unsupported reduction {self.reduction}")
        else:
            teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
            student_logprobs = F.log_softmax(student_logits, dim=-1)

            kl_loss = kl_loss_torch(
                teacher_logprobs=teacher_logprobs,
                student_logprobs=student_logprobs,
                mask=mask,
                symmetrical=self.symmetrical,
            )

            def get_non_blank_weights():
                if self.symmetrical:
                    blank_probs = (
                        torch.exp(teacher_logprobs[..., self.blank_id])
                        + torch.exp(student_logprobs[..., self.blank_id])
                    ) / 2
                else:
                    blank_probs = torch.exp(teacher_logprobs[..., self.blank_id])
                weights = (1 - blank_probs) * mask
                return weights

            match self.reduction:
                case ConsistencyRNNTReductionType.MEAN:
                    kl_loss_value = kl_loss.sum() / mask.sum().clamp(min=1)
                case ConsistencyRNNTReductionType.MEAN_VOLUME:
                    kl_loss_value = (kl_loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)).mean()
                case ConsistencyRNNTReductionType.P_NON_BLANK:
                    weights = get_non_blank_weights().detach()
                    kl_loss_value = ((kl_loss * weights).sum(dim=(1, 2)) / (weights.sum(dim=(1, 2)) + EPS)).mean()
                case ConsistencyRNNTReductionType.P_NON_BLANK_WITH_GRAD:
                    weights = get_non_blank_weights()
                    kl_loss_value = ((kl_loss * weights).sum(dim=(1, 2)) / (weights.sum(dim=(1, 2)) + EPS)).mean()
                case _:
                    raise NotImplementedError(f"Unsupported reduction {self.reduction}")
        return kl_loss_value


class ConsistencyGraphRNNTLoss(nn.Module):
    def __init__(
        self,
        blank_id: int,
        symmetrical: bool = True,
    ):
        super().__init__()
        # TODO: move this loss to k2 library after experiments
        self.symmetrical = symmetrical
        self.blank_id = blank_id
        if K2_AVAILABLE:
            from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss

            self.graph_rnnt = GraphRnntLoss(blank=blank_id)
        else:
            self.graph_rnnt = None

    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        src_lengths: torch.Tensor | None = None,
        tgt_lengths: torch.Tensor | None = None,
    ):
        if self.graph_rnnt is None:
            raise RuntimeError("K2 is not available, cannot compute loss")

        batch_size, src_length_max, tgt_length_max_plus_1, _ = teacher_logits.shape
        device = teacher_logits.device
        tgt_length_max = tgt_length_max_plus_1 - 1

        if src_lengths is None:
            src_lengths = torch.full([batch_size], fill_value=src_length_max, dtype=torch.long, device=device)
        if tgt_lengths is None:
            tgt_lengths = torch.full([batch_size], fill_value=tgt_length_max, dtype=torch.long, device=device)

        teacher_graphs = self.graph_rnnt.get_weighted_graphs(
            logits=teacher_logits, targets=targets, source_lengths=src_lengths, target_lengths=tgt_lengths
        )

        student_graphs = self.graph_rnnt.get_weighted_graphs(
            logits=student_logits, targets=targets, source_lengths=src_lengths, target_lengths=tgt_lengths
        )

        kl_loss_value = self.graph_rnnt.consistency_loss(
            teacher_fsas_vec=teacher_graphs,
            student_fsas_vec=student_graphs,
            source_lengths=src_lengths,
            target_lengths=tgt_lengths,
            symmetrical=self.symmetrical,
        )

        return kl_loss_value
