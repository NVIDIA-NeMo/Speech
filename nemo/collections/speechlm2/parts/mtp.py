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
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from nemo.utils import logging, logging_mode


_LK_MAX_CHUNK_ELEMENTS = 4 * 1024 * 1024


@dataclass(frozen=True)
class MTPLKLossOutput:
    """Aggregate LK loss and its unscaled per-depth components."""

    loss: torch.Tensor
    per_depth_losses: list[torch.Tensor]
    per_depth_kl: list[torch.Tensor]
    per_depth_tv: list[torch.Tensor]


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


def calculate_mtp_lk_loss(
    *,
    mtp_per_depth_h: list[torch.Tensor],
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    scaling_factor: float,
    lk_lambda: float,
    num_label_tokens: int | torch.Tensor | None = None,
    ignore_index: int = -100,
    cu_seqlens: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
    mtp_per_depth_targets: Sequence[torch.Tensor] | None = None,
    projection_sync_group: dist.ProcessGroup | None = None,
    context_parallel_group: dist.ProcessGroup | None = None,
) -> MTPLKLossOutput:
    """Distill MTP heads from the frozen backbone with the hybrid LK objective.

    For each MTP depth, the backbone distribution is shifted to the same future
    position as that depth's ordinary token target. The objective is
    ``lambda * KL(p || q) + (1 - lambda) * TV(p, q)``, where ``p`` is detached
    from autograd and ``q`` is the MTP distribution. Packed-sequence validity is
    inherited from the same targets used by the cross-entropy MTP objective.

    When context parallelism is active, rank-local logits are exchanged so each
    rank receives only the future teacher rows needed by its local MTP outputs.
    This avoids gathering the full sequence-by-vocabulary tensor on every rank.
    """
    if not 0.0 <= lk_lambda <= 1.0:
        raise ValueError(f"lk_lambda must be in [0, 1], got {lk_lambda}")
    if not mtp_per_depth_h:
        raise ValueError("mtp_per_depth_h must contain at least one prediction depth")

    from nemo_automodel.components.loss.utils import _get_lm_head_module

    mtp_outputs = mtp_per_depth_h
    if labels.dim() == 1:
        mtp_outputs = [h.squeeze(0) if h.dim() == 3 and h.shape[0] == 1 else h for h in mtp_outputs]
        if teacher_logits.dim() == 3 and teacher_logits.shape[0] == 1:
            teacher_logits = teacher_logits.squeeze(0)
    if teacher_logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"teacher_logits shape {tuple(teacher_logits.shape)} is incompatible with labels shape {tuple(labels.shape)}"
        )

    num_depths = len(mtp_outputs)
    if mtp_per_depth_targets is None:
        depth_targets = tuple(
            iter_mtp_depth_targets(
                labels,
                num_depths,
                ignore_index=ignore_index,
                cu_seqlens=cu_seqlens,
                seq_idx=seq_idx,
            )
        )
    else:
        if seq_idx is not None:
            raise ValueError("mtp_per_depth_targets cannot be combined with seq_idx")
        if len(mtp_per_depth_targets) != num_depths:
            raise ValueError(f"Expected {num_depths} mtp_per_depth_targets, got {len(mtp_per_depth_targets)}")
        depth_targets = tuple(mtp_per_depth_targets)
        for depth, targets in enumerate(depth_targets, start=1):
            if targets.shape != labels.shape:
                raise ValueError(
                    f"MTP depth {depth} target shape {tuple(targets.shape)} does not match "
                    f"labels shape {tuple(labels.shape)}"
                )

    lm_head = _get_lm_head_module(model)
    if lm_head is None:
        raise ValueError("lm_head module not found in model")

    if num_label_tokens is None:
        normalizer = (labels != ignore_index).sum().clamp(min=1)
    elif torch.is_tensor(num_label_tokens):
        normalizer = num_label_tokens.to(device=teacher_logits.device).clamp(min=1)
    else:
        normalizer = teacher_logits.new_tensor(max(num_label_tokens, 1))

    teacher_logits = teacher_logits.detach()
    teacher_rows = torch.arange(labels.numel(), device=labels.device).reshape(labels.shape)
    use_context_parallel = context_parallel_group is not None and dist.get_world_size(context_parallel_group) > 1
    if use_context_parallel and labels.dim() != 1:
        raise ValueError("Context-parallel LK loss requires flattened THD labels")
    if use_context_parallel and cu_seqlens is None:
        raise ValueError("Context-parallel LK loss requires global cu_seqlens")

    total = mtp_outputs[0].new_zeros(())
    per_depth_losses = []
    per_depth_kl = []
    per_depth_tv = []
    for depth, (mtp_output, targets) in enumerate(zip(mtp_outputs, depth_targets), start=1):
        if mtp_output.shape[:-1] != labels.shape:
            raise ValueError(
                f"MTP depth {depth} hidden-state shape {tuple(mtp_output.shape)} is incompatible with "
                f"labels shape {tuple(labels.shape)}"
            )
        valid = targets != ignore_index
        if use_context_parallel:
            aligned_teacher_logits = _exchange_context_parallel_teacher_logits(
                teacher_logits,
                valid=valid,
                depth=depth,
                cu_seqlens=cu_seqlens,
                group=context_parallel_group,
            )
            source_rows = teacher_rows
        else:
            aligned_teacher_logits = teacher_logits
            source_rows = torch.roll(teacher_rows, shifts=-depth, dims=-1)

        kl_sum, tv_sum = _calculate_lk_sums(
            mtp_output,
            aligned_teacher_logits,
            lm_head=lm_head,
            projection_sync_group=projection_sync_group,
            valid=valid,
            teacher_rows=source_rows,
        )
        kl_loss = kl_sum / normalizer
        tv_loss = tv_sum / normalizer
        depth_loss = lk_lambda * kl_loss + (1.0 - lk_lambda) * tv_loss
        per_depth_kl.append(kl_loss)
        per_depth_tv.append(tv_loss)
        per_depth_losses.append(depth_loss)
        total = total + depth_loss

    return MTPLKLossOutput(
        loss=total * (scaling_factor / num_depths),
        per_depth_losses=per_depth_losses,
        per_depth_kl=per_depth_kl,
        per_depth_tv=per_depth_tv,
    )


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


def _calculate_lk_sums(
    draft_hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    lm_head: torch.nn.Module,
    projection_sync_group: dist.ProcessGroup | None,
    valid: torch.Tensor,
    teacher_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed forward-KL and TV with rank-symmetric projection work.

    FSDP/EP ranks may have different numbers of labeled tokens. DeepEP's
    backward collectives require those ranks to reach each rendezvous in the
    same order and within its fixed timeout, so synchronizing only the number
    of LM-head calls is insufficient: a rank with empty tail chunks can finish
    much earlier than a rank projecting full chunks. Pad every call to the same
    token count and mask the dummy rows out of the objective. Checkpoint
    recomputation then performs identical-shape projection work on every rank.
    """
    vocab_size = teacher_logits.shape[-1]
    draft_flat = draft_hidden.reshape(-1, draft_hidden.shape[-1])
    teacher_flat = teacher_logits.reshape(-1, vocab_size)
    valid_rows = valid.reshape(-1).nonzero(as_tuple=True)[0]
    source_rows = teacher_rows.reshape(-1).index_select(0, valid_rows)
    chunk_tokens = max(1, _LK_MAX_CHUNK_ELEMENTS // vocab_size)
    num_chunks = (valid_rows.numel() + chunk_tokens - 1) // chunk_tokens
    if projection_sync_group is not None and dist.is_available() and dist.is_initialized():
        synchronized_chunks = torch.tensor(num_chunks, dtype=torch.long, device=draft_hidden.device)
        dist.all_reduce(synchronized_chunks, op=dist.ReduceOp.MAX, group=projection_sync_group)
        num_chunks = int(synchronized_chunks.item())

    # Even a globally empty supervised batch must retain the MTP/DeepEP
    # autograd path. One fully masked projection gives every rank the same
    # differentiable graph and produces exactly zero loss/gradient.
    num_chunks = max(1, num_chunks)

    chunk_fn = partial(_calculate_lk_chunk, lm_head=lm_head)
    kl_sum = draft_hidden.new_zeros((), dtype=torch.float32)
    tv_sum = draft_hidden.new_zeros((), dtype=torch.float32)
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_tokens
        draft_indices = valid_rows[start : start + chunk_tokens]
        teacher_indices = source_rows[start : start + chunk_tokens]
        num_valid_rows = draft_indices.numel()
        num_padding_rows = chunk_tokens - num_valid_rows
        if num_padding_rows:
            padding = draft_indices.new_zeros(num_padding_rows)
            draft_indices = torch.cat((draft_indices, padding))
            teacher_indices = torch.cat((teacher_indices, padding))
        valid_chunk_rows = torch.arange(chunk_tokens, device=draft_indices.device) < num_valid_rows
        if torch.is_grad_enabled() and draft_hidden.requires_grad:
            chunk_kl, chunk_tv = checkpoint(
                chunk_fn,
                draft_flat,
                teacher_flat,
                draft_indices,
                teacher_indices,
                valid_chunk_rows,
                use_reentrant=False,
            )
        else:
            chunk_kl, chunk_tv = chunk_fn(
                draft_flat,
                teacher_flat,
                draft_indices,
                teacher_indices,
                valid_chunk_rows,
            )
        kl_sum = kl_sum + chunk_kl
        tv_sum = tv_sum + chunk_tv

    return kl_sum, tv_sum


def _calculate_lk_chunk(
    draft_hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    draft_rows: torch.Tensor,
    teacher_rows: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    lm_head: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project one fixed-size chunk and exclude its padded rows from LK."""
    if draft_rows.shape != teacher_rows.shape or draft_rows.shape != valid_rows.shape:
        raise ValueError(
            "LK chunk row tensors must have matching shapes, got "
            f"{tuple(draft_rows.shape)}, {tuple(teacher_rows.shape)}, and {tuple(valid_rows.shape)}"
        )
    draft_logits = lm_head(draft_hidden.index_select(0, draft_rows))
    selected_teacher_logits = teacher_logits.index_select(0, teacher_rows)
    if draft_logits.shape != selected_teacher_logits.shape:
        raise ValueError(
            f"MTP draft logits shape {tuple(draft_logits.shape)} does not match "
            f"teacher logits shape {tuple(selected_teacher_logits.shape)}"
        )

    draft_log_probs = torch.log_softmax(draft_logits.float(), dim=-1)
    teacher_log_probs = torch.log_softmax(selected_teacher_logits.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    draft_probs = draft_log_probs.exp()
    row_weights = valid_rows.to(dtype=draft_log_probs.dtype)
    kl_sum = ((teacher_probs * (teacher_log_probs - draft_log_probs)).sum(dim=-1) * row_weights).sum()
    tv_sum = (0.5 * (teacher_probs - draft_probs).abs().sum(dim=-1) * row_weights).sum()
    return kl_sum, tv_sum


@torch.no_grad()
def _exchange_context_parallel_teacher_logits(
    teacher_logits: torch.Tensor,
    *,
    valid: torch.Tensor,
    depth: int,
    cu_seqlens: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Exchange only the future teacher rows required by this CP rank."""
    import transformer_engine_torch as tex

    cp_size = dist.get_world_size(group)
    cp_rank = dist.get_rank(group)
    cs = cu_seqlens.squeeze(0) if cu_seqlens.dim() == 2 and cu_seqlens.shape[0] == 1 else cu_seqlens
    if cs.dim() != 1:
        raise ValueError(f"cu_seqlens must have shape [N+1] or [1, N+1], got {tuple(cu_seqlens.shape)}")
    total_tokens = int(cs[-1].item())

    partitions = [
        tex.thd_get_partitioned_indices(cs, total_tokens, cp_size, rank).to(
            device=teacher_logits.device, dtype=torch.long
        )
        for rank in range(cp_size)
    ]
    local_global_rows = partitions[cp_rank]
    if local_global_rows.numel() != teacher_logits.shape[0]:
        raise ValueError(
            f"CP partition has {local_global_rows.numel()} rows but teacher_logits has {teacher_logits.shape[0]}"
        )

    global_owner = torch.empty(total_tokens, dtype=torch.long, device=teacher_logits.device)
    global_local_row = torch.empty_like(global_owner)
    for rank, partition in enumerate(partitions):
        global_owner.index_fill_(0, partition, rank)
        global_local_row.index_copy_(
            0,
            partition,
            torch.arange(partition.numel(), dtype=torch.long, device=teacher_logits.device),
        )

    destination_rows = valid.reshape(-1).nonzero(as_tuple=True)[0]
    source_global_rows = local_global_rows.index_select(0, destination_rows) + depth
    if source_global_rows.numel() and int(source_global_rows.max().item()) >= total_tokens:
        raise ValueError("A valid MTP target points beyond the global packed sequence")

    source_owners = global_owner.index_select(0, source_global_rows)
    request_chunks = []
    destination_chunks = []
    send_counts = torch.zeros(cp_size, dtype=torch.long, device=teacher_logits.device)
    for owner in range(cp_size):
        owned = source_owners == owner
        requested_global = source_global_rows[owned]
        request_chunks.append(global_local_row.index_select(0, requested_global))
        destination_chunks.append(destination_rows[owned])
        send_counts[owner] = requested_global.numel()

    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    send_requests = (
        torch.cat(request_chunks) if request_chunks else torch.empty(0, dtype=torch.long, device=teacher_logits.device)
    )
    recv_requests = torch.empty(int(recv_counts.sum().item()), dtype=torch.long, device=teacher_logits.device)
    send_splits = [int(value) for value in send_counts.tolist()]
    recv_splits = [int(value) for value in recv_counts.tolist()]
    dist.all_to_all_single(
        recv_requests,
        send_requests,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=group,
    )

    response_rows = teacher_logits.index_select(0, recv_requests).contiguous()
    received_rows = torch.empty(
        (int(send_counts.sum().item()), teacher_logits.shape[-1]),
        dtype=teacher_logits.dtype,
        device=teacher_logits.device,
    )
    dist.all_to_all_single(
        received_rows,
        response_rows,
        output_split_sizes=send_splits,
        input_split_sizes=recv_splits,
        group=group,
    )

    aligned = torch.zeros_like(teacher_logits)
    if destination_rows.numel():
        aligned.index_copy_(0, torch.cat(destination_chunks), received_rows)
    return aligned
