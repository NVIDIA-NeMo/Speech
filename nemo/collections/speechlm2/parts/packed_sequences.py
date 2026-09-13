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
"""Packed-sequence (THD-format) helpers for SALMAutomodel training.

The Nemotron-V3 LLM in `nemo_automodel` already accepts THD batches — TE
attention switches to varlen FlashAttention via `cu_seqlens`, and the Mamba
mixer derives `seq_idx` from the same `cu_seqlens` so SSM state resets at
document boundaries. The functions in this module concatenate a SALM
multi-utterance minibatch into a single packed sequence so the LLM is fed
`inputs_embeds` of shape ``[1, T_total, H]`` plus the THD metadata it needs.

All tensor logic is kept here (no `SALMAutomodel` knowledge) so it is unit
testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

from nemo.collections.speechlm2.parts.input_utils import _unpad_inputs


def pack_audio_into_text_embeds(
    input_ids: Tensor,
    embeds: Tensor,
    target_ids: Tensor,
    replacements: list[Tensor],
    padding_id: int,
    placeholder_id: int,
    cp_size: int = 1,
    tp_size: int = 1,
    ignore_index: int = -100,
) -> dict[str, Tensor]:
    """Splice audio frames into per-utterance text embeddings and pack into THD.

    Mirrors :func:`replace_placeholders_and_build_targets` but emits a single
    flat THD batch instead of a right-padded BSHD one. Labels are next-token
    shifted *per utterance* before cross-utterance concatenation, so the LLM
    can be called without any further shift.

    Args:
        input_ids:       ``[B, S]`` int64; left-padded.
        embeds:          ``[B, S, H]`` text-token embeddings (placeholder slots
                         are pre-zeroed by the caller; they get overwritten).
        target_ids:      ``[B, S]`` int64; ``-100`` outside assistant spans.
        replacements:    list of ``[L_i, H]`` audio-frame embeddings, one per
                         placeholder occurrence in row-major order.
        padding_id:      pad-token id in ``input_ids`` (used to strip left-pad
                         and to mark padding positions as ``ignore_index`` in
                         labels).
        placeholder_id:  the ``<|audio|>`` token id.
        cp_size:         per-utterance flat lengths are rounded up to a
                         multiple of ``2 * cp_size`` (TE-CP requirement).
        tp_size:         the last utterance's padded length is bumped so that
                         ``T_total % tp_size == 0`` (sequence-parallel). When
                         CP is active, the bump also preserves the
                         ``2 * cp_size`` per-utterance alignment.
        ignore_index:    label fill for audio-frame slots, padding slots, and
                         the last position of every utterance.

    Returns a dict with:

    - ``inputs_embeds``    ``[T_total, H]`` (2D, mirrors Automodel's
                           ``process_input_for_thd`` / ``_shard_thd_chunk_for_te``
                           contract — no leading batch dim)
    - ``labels``           ``[T_total]`` int64, already shifted
    - ``position_ids``     ``[T_total]`` int64, resets to 0 per utt
    - ``seq_lens``         ``[B, 1]`` int64, real per-utt flat lengths
    - ``seq_lens_padded``  ``[B, 1]`` int64, post-rounding lengths
    - ``cu_seqlens``       ``[B+1]`` int32, ``cumsum`` of ``seq_lens_padded``
    - ``max_seqlen``       int32 scalar, ``max(seq_lens_padded)``
    - ``qkv_format``       ``"thd"``
    """
    B = input_ids.shape[0]
    H = embeds.shape[-1]
    device = embeds.device
    dtype = embeds.dtype

    # Strip left-padding so per-utt sequences are tight before splicing.
    ids_unpad, embs_unpad, tgts_unpad = _unpad_inputs(input_ids, embeds, target_ids, padding_id)

    seq_embs: list[Tensor] = []
    seq_labs: list[Tensor] = []
    real_lens: list[int] = []
    rep_idx = 0

    for i in range(B):
        ids_i = ids_unpad[i]
        emb_i = embs_unpad[i]
        tgt_i = tgts_unpad[i]
        placeholders = (ids_i == placeholder_id).nonzero(as_tuple=True)[0].tolist()

        emb_segments: list[Tensor] = []
        lab_segments: list[Tensor] = []
        prev = 0
        for p in placeholders:
            if p > prev:
                emb_segments.append(emb_i[prev:p])
                seg_lab = tgt_i[prev:p].clone()
                seg_lab[ids_i[prev:p] == padding_id] = ignore_index
                lab_segments.append(seg_lab)
            rep = replacements[rep_idx]
            rep_idx += 1
            emb_segments.append(rep)
            lab_segments.append(torch.full((rep.shape[0],), ignore_index, dtype=torch.long, device=device))
            prev = p + 1
        if prev < ids_i.numel():
            emb_segments.append(emb_i[prev:])
            seg_lab = tgt_i[prev:].clone()
            seg_lab[ids_i[prev:] == padding_id] = ignore_index
            lab_segments.append(seg_lab)

        emb_cat = torch.cat(emb_segments, dim=0)  # [L_i, H]
        lab_cat = torch.cat(lab_segments, dim=0)  # [L_i]
        L = emb_cat.shape[0]
        # Per-utterance next-token shift: labels[t] = orig[t+1], last slot is ignored.
        lab_shift = torch.cat(
            [lab_cat[1:], torch.full((1,), ignore_index, dtype=torch.long, device=device)],
            dim=0,
        )
        seq_embs.append(emb_cat)
        seq_labs.append(lab_shift)
        real_lens.append(L)

    if rep_idx != len(replacements):
        raise ValueError(
            f"Used {rep_idx} of {len(replacements)} audio replacements — "
            f"placeholder occurrences in input_ids do not match replacements length."
        )

    # Round each utterance's length up to a multiple of 2*cp_size (TE-CP
    # interleaves 2 chunks per rank); skip rounding when cp_size == 1. Then
    # bump the last so the total is divisible by tp_size for sequence
    # parallelism, preserving the CP alignment when CP is active.
    if cp_size > 1:
        cp_mult = 2 * cp_size
        padded_lens = [((L + cp_mult - 1) // cp_mult) * cp_mult for L in real_lens]
    else:
        padded_lens = list(real_lens)
    if tp_size > 1:
        total_len = sum(padded_lens)
        if cp_size > 1:
            cp_mult = 2 * cp_size
            tp_bump = 0
            while (total_len + tp_bump) % tp_size != 0:
                tp_bump += cp_mult
            padded_lens[-1] += tp_bump
        else:
            rem = total_len % tp_size
            if rem != 0:
                padded_lens[-1] += tp_size - rem

    # Materialize the flat THD batch.
    flat_emb_segs: list[Tensor] = []
    flat_lab_segs: list[Tensor] = []
    flat_pos_segs: list[Tensor] = []
    for emb, lab, l_real, l_pad in zip(seq_embs, seq_labs, real_lens, padded_lens):
        flat_emb_segs.append(emb)
        flat_lab_segs.append(lab)
        flat_pos_segs.append(torch.arange(l_real, dtype=torch.long, device=device))
        pad_n = l_pad - l_real
        if pad_n > 0:
            flat_emb_segs.append(torch.zeros(pad_n, H, dtype=dtype, device=device))
            flat_lab_segs.append(torch.full((pad_n,), ignore_index, dtype=torch.long, device=device))
            flat_pos_segs.append(torch.arange(l_real, l_pad, dtype=torch.long, device=device))

    inputs_embeds = torch.cat(flat_emb_segs, dim=0)  # [T_total, H]
    labels = torch.cat(flat_lab_segs, dim=0)  # [T_total]
    position_ids = torch.cat(flat_pos_segs, dim=0)  # [T_total]

    seq_lens = torch.tensor(real_lens, dtype=torch.long, device=device).unsqueeze(-1)
    seq_lens_padded = torch.tensor(padded_lens, dtype=torch.long, device=device).unsqueeze(-1)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.tensor(padded_lens, dtype=torch.int32, device=device).cumsum(0).to(torch.int32),
        ]
    )
    max_seqlen = torch.tensor(max(padded_lens), dtype=torch.int32, device=device)

    return {
        "inputs_embeds": inputs_embeds,
        "labels": labels,
        "position_ids": position_ids,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
        "qkv_format": "thd",
    }


def _shard_packed_for_cp(
    packed: dict[str, Tensor],
    cp_mesh,
    mtp_inputs: _PackedMTPInputs | None = None,
) -> tuple[dict[str, Tensor], _PackedMTPInputs | None]:
    """Partition a packed THD batch across CP ranks (TE's interleaved scheme).

    Mirrors ``nemo_automodel.components.distributed.cp_utils._shard_thd_chunk_for_te``
    but preserves the float dtype of ``inputs_embeds`` (the upstream helper
    casts everything to int64, which would silently corrupt embeddings).

    Args:
        packed: Global packed tensors. ``inputs_embeds`` has shape [tokens,
            hidden], while ``labels`` and ``position_ids`` have shape [tokens].
        cp_mesh: One-dimensional context-parallel device mesh.
        mtp_inputs: Optional globally shifted and boundary-masked MTP tensors.
            Embeddings have shape [tokens, hidden]; position IDs and targets
            have shape [tokens].

    Returns:
        The rank-local packed mapping and optional rank-local MTP tensors. All
        token tensors use TE's identical interleaved CP partition and have
        local shape [tokens / cp_size, ...].
    """
    import transformer_engine_torch as tex  # local import — only needed when CP > 1

    cp_size = cp_mesh.size()
    cp_rank = torch.distributed.get_rank(group=cp_mesh.get_group())

    cu_seqlens = packed["cu_seqlens"]
    inputs_embeds = packed["inputs_embeds"]  # [T, H]
    labels = packed["labels"]  # [T]
    position_ids = packed["position_ids"]  # [T]

    index = tex.thd_get_partitioned_indices(cu_seqlens, inputs_embeds.shape[0], cp_size, cp_rank)
    inputs_embeds = inputs_embeds.index_select(0, index)
    labels = labels.index_select(0, index)
    position_ids = position_ids.index_select(0, index)
    if mtp_inputs is not None:
        mtp_inputs = _PackedMTPInputs(
            embed_inputs=tuple(embeds.index_select(0, index).contiguous() for embeds in mtp_inputs.embed_inputs),
            position_ids=tuple(
                position_ids.index_select(0, index).to(torch.int64).contiguous()
                for position_ids in mtp_inputs.position_ids
            ),
            targets=tuple(
                targets.index_select(0, index).to(torch.int64).contiguous() for targets in mtp_inputs.targets
            ),
        )

    return (
        {
            "inputs_embeds": inputs_embeds.contiguous(),
            "labels": labels.to(torch.int64).contiguous(),
            "position_ids": position_ids.to(torch.int64).contiguous(),
            "cu_seqlens": cu_seqlens.to(torch.int32).contiguous(),
            "max_seqlen": packed["max_seqlen"],
            "qkv_format": "thd",
        },
        mtp_inputs,
    )


def prepare_packed_llm_inputs(
    input_ids: Tensor,
    text_embs: Tensor,
    audio_embs: list[Tensor],
    target_ids: Tensor,
    padding_id: int,
    placeholder_id: int,
    device_mesh: Optional[Any] = None,
    mtp_num_depths: int = 0,
) -> dict[str, Any]:
    """Pack a SALM minibatch and (optionally) shard it across CP ranks.

    Args:
        input_ids: Token IDs of shape [batch, sequence].
        text_embs: Text embeddings of shape [batch, sequence, hidden].
        audio_embs: Audio replacement tensors, each of shape [audio_frames, hidden].
        target_ids: Unshifted labels of shape [batch, sequence].
        padding_id: Token ID used for left padding.
        placeholder_id: Token ID replaced by audio frames.
        device_mesh: Optional device mesh containing CP and TP axes.
        mtp_num_depths: Number of future-token MTP input/target tensors to
            prepare. These are emitted only when CP is active because
            rank-local rolling is otherwise incorrect.

    Returns:
        Mapping containing rank-local THD model inputs. ``input_embeds`` has
        shape [local_tokens, hidden], ``target_ids`` and ``position_ids`` have
        shape [local_tokens], and ``cu_seqlens`` describes the global packed
        stream. Under CP with MTP, ``llm_kwargs`` also contains future
        embeddings [local_tokens, hidden] and position IDs [local_tokens], and
        ``mtp_per_depth_targets`` contains one [local_tokens] target tensor per
        depth. All are shifted in global order before TE partitioning.
    """
    from nemo.collections.speechlm2.parts.cp_helpers import get_cp_mesh

    cp_mesh, cp_size, _ = get_cp_mesh(device_mesh)
    tp_size = 1
    if device_mesh is not None:
        names = device_mesh.mesh_dim_names or ()
        if "tp" in names and device_mesh["tp"].size() > 1:
            tp_size = device_mesh["tp"].size()

    packed = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=text_embs,
        target_ids=target_ids,
        replacements=audio_embs,
        padding_id=padding_id,
        placeholder_id=placeholder_id,
        cp_size=cp_size,
        tp_size=tp_size,
    )
    num_tokens = packed["seq_lens"].sum()
    num_examples = torch.tensor(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    mtp_inputs = None
    if cp_mesh is not None and mtp_num_depths > 0:
        mtp_inputs = _build_packed_mtp_inputs(packed, mtp_num_depths)
    if cp_mesh is not None:
        packed, mtp_inputs = _shard_packed_for_cp(packed, cp_mesh, mtp_inputs)

    result = {
        "input_embeds": packed["inputs_embeds"],
        "attention_mask": None,
        "target_ids": packed["labels"],
        "num_tokens": num_tokens,
        "num_examples": num_examples,
        "llm_kwargs": {
            "qkv_format": "thd",
            # Match Automodel's standard THD contract (``thd_utils.process_input_for_thd``
            # and ``cp_utils._shard_thd_chunk_for_te``): emit only ``cu_seqlens`` (the
            # padded cumsum) and a single ``max_seqlen``. Passing ``cu_seqlens_padded``
            # too would activate the ``pad_between_seqs=True`` branch in
            # ``nemo_automodel/.../attention/utils.py``, which routes TE down a different
            # attention path. Passing pre-split ``max_seqlen_q`` / ``max_seqlen_kv``
            # gets them silently dropped by the preprocessor.
            "cu_seqlens": packed["cu_seqlens"],
            "position_ids": packed["position_ids"],
            "max_seqlen": packed["max_seqlen"],
        },
    }
    if mtp_inputs is not None:
        result["llm_kwargs"]["mtp_embed_inputs"] = mtp_inputs.embed_inputs
        result["llm_kwargs"]["mtp_per_depth_position_ids"] = mtp_inputs.position_ids
        result["mtp_per_depth_targets"] = mtp_inputs.targets
    return result


@dataclass(frozen=True)
class _PackedMTPInputs:
    """MTP tensors prepared in global packed order and optionally CP-sharded.

    Every tuple contains one tensor per MTP depth. ``embed_inputs`` tensors
    have shape [tokens, hidden]; ``position_ids`` and ``targets`` tensors have
    shape [tokens]. Invalid future positions at packed-sequence boundaries are
    zero-filled for model inputs and use ``ignore_index`` for targets.
    """

    embed_inputs: tuple[Tensor, ...]
    position_ids: tuple[Tensor, ...]
    targets: tuple[Tensor, ...]


def _build_packed_mtp_shift_masks(seq_idx: Tensor, num_depths: int) -> tuple[Tensor, ...]:
    """Build one packed-boundary validity mask per MTP depth."""
    if num_depths < 1:
        raise ValueError(f"num_depths must be positive, got {num_depths}")

    token_idx = torch.arange(seq_idx.numel(), device=seq_idx.device)
    return tuple(
        (token_idx + depth < seq_idx.numel()) & (torch.roll(seq_idx, shifts=-depth, dims=0) == seq_idx)
        for depth in range(1, num_depths + 1)
    )


def _shift_packed_tensor_for_mtp(tensor: Tensor, depth: int, valid_mask: Tensor) -> Tensor:
    """Shift a packed tensor left without crossing document boundaries.

    Args:
        tensor: Global packed tensor of shape [tokens, ...].
        depth: Number of future-token positions to shift.
        valid_mask: Precomputed packed-boundary validity mask of shape [tokens].

    Returns:
        A tensor with the same shape and dtype as ``tensor``. Positions whose
        future token is outside their current document are zero.
    """
    if depth < 1:
        raise ValueError(f"MTP depth must be positive, got {depth}")
    if tensor.shape[0] != valid_mask.shape[0]:
        raise ValueError(
            f"Packed tensor token count {tensor.shape[0]} does not match validity mask token count {valid_mask.shape[0]}"
        )

    shifted = torch.roll(tensor, shifts=-depth, dims=0)
    valid = valid_mask
    while valid.ndim < shifted.ndim:
        valid = valid.unsqueeze(-1)
    return torch.where(valid, shifted, torch.zeros((), dtype=tensor.dtype, device=tensor.device))


def _build_packed_mtp_inputs(packed: dict[str, Tensor], num_depths: int) -> _PackedMTPInputs:
    """Prepare globally shifted, boundary-safe MTP tensors before CP sharding.

    ``packed`` contains embeddings [tokens, hidden], position IDs [tokens],
    labels [tokens], and global ``cu_seqlens``. The returned tensors retain
    those layouts and global token order.
    """
    from nemo.collections.speechlm2.parts.mtp import iter_mtp_depth_targets, resolve_mtp_seq_idx

    if num_depths < 1:
        raise ValueError(f"num_depths must be positive, got {num_depths}")
    seq_idx = resolve_mtp_seq_idx(packed["position_ids"], cu_seqlens=packed["cu_seqlens"])
    if seq_idx is None:
        raise RuntimeError("Could not derive packed sequence IDs from cu_seqlens")

    depths = range(1, num_depths + 1)
    valid_masks = _build_packed_mtp_shift_masks(seq_idx, num_depths)
    return _PackedMTPInputs(
        embed_inputs=tuple(
            _shift_packed_tensor_for_mtp(packed["inputs_embeds"], depth, valid_mask)
            for depth, valid_mask in zip(depths, valid_masks)
        ),
        position_ids=tuple(
            _shift_packed_tensor_for_mtp(packed["position_ids"], depth, valid_mask)
            for depth, valid_mask in zip(depths, valid_masks)
        ),
        targets=tuple(iter_mtp_depth_targets(packed["labels"], num_depths, seq_idx=seq_idx)),
    )
