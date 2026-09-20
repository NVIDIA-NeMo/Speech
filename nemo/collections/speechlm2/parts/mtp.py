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
"""Reusable Multi-Token Prediction helpers for SpeechLM models."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from nemo.utils import logging, logging_mode


def build_mtp_loss_fn() -> torch.nn.Module:
    """Select the memory-efficient MTP loss with an optional-dependency fallback."""
    from nemo_automodel.components.loss import linear_ce

    if linear_ce.HAVE_CUT_CROSS_ENTROPY:
        # Fuse the shared LM projection with CE so each MTP depth does not materialize
        # a full [tokens, vocab] logits tensor. reduction="sum" lets the helper
        # normalize by the global labeled-token count.
        return linear_ce.FusedLinearCrossEntropy(reduction="sum")

    # cut-cross-entropy is optional in Automodel and may be absent from NeMo Speech
    # containers. Retain the unfused path so enabling MTP does not introduce a new
    # undeclared runtime requirement.
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

    logging.warning(
        "cut_cross_entropy is unavailable; falling back to the unfused MTP loss. "
        "Install cut-cross-entropy to reduce peak MTP loss memory."
    )
    return MaskedCrossEntropy(reduction="sum", fp32_upcast=False)


def calculate_mtp_loss_with_per_depth(*args: Any, **kwargs: Any) -> Any:
    """Call Automodel's per-depth MTP loss API only when MTP training runs."""
    try:
        from nemo_automodel.components.loss.mtp import MTPLossOutput, calculate_mtp_loss
    except ImportError as error:
        raise RuntimeError("MTP training requires an Automodel version with per-depth MTP loss support") from error

    output = calculate_mtp_loss(*args, **kwargs)
    if not isinstance(output, MTPLossOutput):
        raise TypeError("Automodel did not return the requested per-depth MTP loss output")
    return output


def resolve_mtp_seq_idx(
    labels: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Resolve packed-sequence IDs and align them with the label layout."""
    if seq_idx is None and cu_seqlens is not None:
        cs = cu_seqlens
        if cs.dim() == 2:
            if cs.shape[0] != 1:
                raise ValueError(f"MTP cu_seqlens must have shape [N+1] or [1, N+1], got {tuple(cs.shape)}")
            cs = cs.squeeze(0)
        if cs.dim() != 1:
            raise ValueError(f"MTP cu_seqlens must have shape [N+1] or [1, N+1], got {tuple(cs.shape)}")
        positions = torch.arange(labels.shape[-1], device=labels.device)
        seq_idx = torch.searchsorted(cs[1:].contiguous(), positions, right=True)

    if seq_idx is None:
        return None
    if seq_idx.dim() == 1 and labels.dim() == 2:
        seq_idx = seq_idx.unsqueeze(0).expand(labels.shape[0], -1)
    elif seq_idx.dim() == 2 and labels.dim() == 1 and seq_idx.shape[0] == 1:
        seq_idx = seq_idx.squeeze(0)
    if seq_idx.shape != labels.shape:
        raise ValueError(f"MTP seq_idx shape {tuple(seq_idx.shape)} does not match labels shape {tuple(labels.shape)}")
    return seq_idx


def iter_mtp_depth_targets(
    labels: torch.Tensor,
    num_depths: int,
    *,
    ignore_index: int = -100,
    cu_seqlens: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
) -> Iterator[torch.Tensor]:
    """Yield shifted labels with trailing and packed-boundary positions masked."""
    from nemo_automodel.components.models.common.mtp import roll_tensor

    seq_idx = resolve_mtp_seq_idx(labels, cu_seqlens=cu_seqlens, seq_idx=seq_idx)
    cur_labels = labels
    for depth in range(1, num_depths + 1):
        cur_labels = roll_tensor(cur_labels, shifts=-1, dim=-1)
        masked = cur_labels.clone()
        n_invalid = min(depth, masked.shape[-1])
        masked[..., -n_invalid:] = ignore_index

        if seq_idx is not None:
            rolled_seq_idx = roll_tensor(seq_idx, shifts=-depth, dim=-1)
            masked = torch.where(rolled_seq_idx != seq_idx, torch.full_like(masked, ignore_index), masked)

        yield masked


def vocab_parallel_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Return global vocabulary argmax IDs without gathering full logits.

    PyTorch's DTensor argmax handler reduces sharded maxima and their global
    indices across the vocabulary mesh. Materialize only the resulting token-ID
    tensor, whose vocabulary dimension has already been removed.
    """
    predictions = logits.argmax(dim=-1)
    if isinstance(predictions, DTensor):
        predictions = predictions.full_tensor()
    return predictions


def calculate_mtp_teacher_forced_agreement(
    *,
    mtp_per_depth_h: list[torch.Tensor],
    labels: torch.Tensor,
    model: torch.nn.Module,
    verifier_predictions: torch.Tensor,
    ignore_index: int = -100,
    cu_seqlens: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-head teacher-forced MTP/verifier agreement counts for validation.

    For each MTP depth ``k`` the head's argmax prediction is compared with the
    verifier's prediction for the same future position from the validation
    forward conditioned on ground-truth tokens. Counts are prefix-based: depth
    ``k`` agrees only when every draft through ``k`` matches. This must not be
    reported as speculative-decoding acceptance, which requires verifier logits
    conditioned on the proposed draft prefix. The same rolled/masked labels as
    the MTP loss define eligible positions, including packed-THD boundaries.
    """
    from nemo_automodel.components.loss.utils import _get_lm_head_module
    from nemo_automodel.components.models.common.mtp import roll_tensor

    mtp_outputs = mtp_per_depth_h
    if labels.dim() == 1:
        mtp_outputs = [h.squeeze(0) if (h.dim() == 3 and h.shape[0] == 1) else h for h in mtp_outputs]
        if verifier_predictions.dim() == 2 and verifier_predictions.shape[0] == 1:
            verifier_predictions = verifier_predictions.squeeze(0)
    if verifier_predictions.shape != labels.shape:
        raise ValueError(
            f"verifier_predictions.shape={tuple(verifier_predictions.shape)} does not "
            f"match labels.shape={tuple(labels.shape)}"
        )

    lm_head = _get_lm_head_module(model)
    if lm_head is None:
        raise ValueError("lm_head module not found in model")

    prefix_matches = torch.ones_like(labels, dtype=torch.bool)
    prefix_valid = torch.ones_like(labels, dtype=torch.bool)
    correct_by_head = []
    valid_by_head = []
    depth_targets = iter_mtp_depth_targets(
        labels,
        len(mtp_outputs),
        ignore_index=ignore_index,
        cu_seqlens=cu_seqlens,
        seq_idx=seq_idx,
    )
    for k, (mtp_output, masked) in enumerate(zip(mtp_outputs, depth_targets)):
        logits = lm_head(mtp_output)
        preds = vocab_parallel_argmax(logits)
        valid = masked != ignore_index
        verifier_for_depth = roll_tensor(verifier_predictions, shifts=-(k + 1), dim=-1)
        prefix_valid = prefix_valid & valid
        prefix_matches = prefix_matches & preds.eq(verifier_for_depth)
        correct_by_head.append((prefix_matches & prefix_valid).sum())
        valid_by_head.append(prefix_valid.sum())

    return correct_by_head, valid_by_head


@contextmanager
def mtp_validation_forward(llm: torch.nn.Module, *, enabled: bool):
    """Run MTP during one eval forward without changing child-module eval state."""
    if not enabled:
        yield
        return

    previous = getattr(llm, "compute_mtp_in_eval", None)
    if previous is None:
        logging.warning(
            f"{type(llm).__name__} does not expose compute_mtp_in_eval; skipping the MTP validation forward.",
            mode=logging_mode.ONCE,
        )
        yield
        return

    llm.compute_mtp_in_eval = True
    try:
        yield
    finally:
        llm.compute_mtp_in_eval = previous


def compute_mtp_agreement_lengths(
    correct_counts: Sequence[torch.Tensor],
    valid_counts: Sequence[torch.Tensor],
    *,
    reduce_sums: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate prefix-agreement counters into per-depth rates and mean length.

    ``correct_counts`` and ``valid_counts`` contain one integer counter vector
    per validation step. Their depth dimension must match. ``reduce_sums`` can
    optionally reduce the concatenated integer counters across data-parallel
    ranks before conversion to floating point.
    """
    if not correct_counts or not valid_counts:
        raise ValueError("MTP agreement counters must not be empty")

    correct = torch.stack(tuple(correct_counts)).sum(dim=0)
    valid = torch.stack(tuple(valid_counts)).sum(dim=0)
    if correct.shape != valid.shape:
        raise ValueError(
            f"MTP correct-count shape {tuple(correct.shape)} does not match valid-count shape {tuple(valid.shape)}"
        )

    num_depths = correct.numel()
    reduced = torch.cat((correct, valid))
    if reduce_sums is not None:
        reduced = reduce_sums(reduced)

    per_depth = reduced[:num_depths].float() / reduced[num_depths:].clamp(min=1).float()
    mean_prefix_length = per_depth.new_tensor(1.0) + per_depth.sum()
    return per_depth, mean_prefix_length
