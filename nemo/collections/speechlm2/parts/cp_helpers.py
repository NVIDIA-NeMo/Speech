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
"""Context-Parallelism (CP) helpers for SALMAutomodel.

These helpers consolidate the CP-shape work needed to feed both BSHD and THD
batches into a Nemotron-V3 LLM whose attention/Mamba layers were CP-wired by
the Automodel parallelizer (`set_context_parallel_group()` / `mixer.cp =
MambaContextParallel(...)`). Three concerns:

1. ``get_cp_mesh`` — read the CP submesh out of a device mesh, returning
   ``(None, 1, 0)`` when CP is inactive so callers can short-circuit.
2. ``shard_bshd_for_cp`` — pad and partition a BSHD batch along the seq dim
   using TE's DualChunkSwap pattern (matches Automodel's Config 1 reference
   test in ``run_hybrid_nemotron_v3_cp.py``).
3. ``encode_audio_with_cp_distribution`` — distribute the audio encoder
   forward across CP ranks so it isn't recomputed cp_size times. Pads to a
   multiple of cp_size with dummy zero-audios so every rank participates in
   FSDP all-gather; dummies are dropped after the post-encoder all-gather.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from nemo.collections.speechlm2.parts.encoder_chunking import encode_audio_with_optional_chunking


def get_cp_mesh(device_mesh) -> tuple[Optional[object], int, int]:
    """Return ``(cp_mesh, cp_size, cp_rank)`` or ``(None, 1, 0)`` when CP is inactive."""
    if device_mesh is None:
        return None, 1, 0
    names = device_mesh.mesh_dim_names or ()
    if "cp" not in names or device_mesh["cp"].size() <= 1:
        return None, 1, 0
    cp_mesh = device_mesh["cp"]
    cp_rank = dist.get_rank(group=cp_mesh.get_group())
    return cp_mesh, cp_mesh.size(), cp_rank


def shard_bshd_for_cp(
    input_embs: Tensor,
    attention_mask: Tensor,
    target_ids: Tensor,
    cp_mesh,
    tp_size: int = 1,
) -> dict[str, Tensor]:
    """Pre-shard a BSHD batch across CP ranks via TE's DualChunkSwap pattern.

    Right-pads the seq dim to a multiple of ``2 * cp_size * tp_size`` (TE-CP
    requires ``2 * cp_size``; SP requires per-rank len divisible by ``tp_size``)
    and partitions along the seq dim using
    ``transformer_engine_torch.thd_get_partitioned_indices``.

    Args:
        input_embs:     ``[B, T, H]`` float.
        attention_mask: ``[B, T]`` bool/long; pad slots become 0.
        target_ids:     ``[B, T]`` int64; pad slots become ``-100``.
        cp_mesh:        the CP submesh of size ``cp_size > 1``.
        tp_size:        tensor-parallel world size (1 if TP is inactive).

    Returns dict with keys ``input_embs``, ``attention_mask``, ``target_ids``,
    each shape ``[B, T_padded // cp_size, ...]``.
    """
    import transformer_engine_torch as tex

    cp_size = cp_mesh.size()
    cp_rank = dist.get_rank(group=cp_mesh.get_group())
    device = input_embs.device

    B, T, H = input_embs.shape
    mult = 2 * cp_size * max(1, tp_size)
    T_padded = ((T + mult - 1) // mult) * mult
    pad_n = T_padded - T
    if pad_n > 0:
        input_embs = F.pad(input_embs, (0, 0, 0, pad_n), value=0.0)
        attention_mask = F.pad(attention_mask.to(torch.long), (0, pad_n), value=0).to(torch.bool)
        target_ids = F.pad(target_ids, (0, pad_n), value=-100)

    cu_seqlens = torch.tensor([0, T_padded], dtype=torch.int32, device=device)
    indices = tex.thd_get_partitioned_indices(cu_seqlens, T_padded, cp_size, cp_rank)

    return {
        "input_embs": input_embs.index_select(1, indices).contiguous(),
        "attention_mask": attention_mask.index_select(1, indices).contiguous(),
        "target_ids": target_ids.index_select(1, indices).contiguous(),
    }


def encode_audio_with_cp_distribution(
    perception,
    audios: Tensor,
    audio_lens: Tensor,
    *,
    chunk_size_seconds: Optional[float],
    sampling_rate: int,
    cp_mesh=None,
) -> list[Tensor]:
    """Distribute the audio encoder forward across CP ranks.

    Falls back to :func:`encode_audio_with_optional_chunking` when ``cp_mesh is
    None`` or there are no audios in the batch.

    With CP active, each rank encodes a contiguous slice of the audio batch
    (rank ``r`` gets ``audios[r*per_rank : (r+1)*per_rank]`` where
    ``per_rank = ceil(B_aud / cp_size)``). When ``B_aud`` is not a multiple of
    ``cp_size`` the audio batch is right-padded with zero-audio dummies; every
    rank still calls ``perception`` so FSDP all-gather and activation
    checkpointing fire uniformly. The dummy length is set to the smallest real
    audio length in the batch (guaranteed to satisfy the encoder's minimum-
    length constraints since at least one real sample of that length already
    does).

    After local encoding, each rank's variable-length embedding tensors are
    zero-padded to a globally-consistent ``max_L`` and ``all_gather``ed across
    the CP group. The full ordered list is reconstructed and dummies are
    dropped, so the return value is identical on every CP rank.
    """
    B_aud = int(audios.shape[0])
    if cp_mesh is None or B_aud == 0:
        return encode_audio_with_optional_chunking(
            perception, audios, audio_lens,
            chunk_size_seconds=chunk_size_seconds, sampling_rate=sampling_rate,
        )

    cp_size = cp_mesh.size()
    cp_rank = dist.get_rank(group=cp_mesh.get_group())
    device = audios.device

    per_rank = (B_aud + cp_size - 1) // cp_size
    B_padded = per_rank * cp_size
    pad_n = B_padded - B_aud

    if pad_n > 0:
        dummy_len = int(audio_lens.min().item())
        T_samp = audios.shape[1]
        dummy_audios = torch.zeros(pad_n, T_samp, dtype=audios.dtype, device=device)
        dummy_lens = torch.full((pad_n,), dummy_len, dtype=audio_lens.dtype, device=device)
        audios = torch.cat([audios, dummy_audios], dim=0)
        audio_lens = torch.cat([audio_lens, dummy_lens], dim=0)

    start = cp_rank * per_rank
    end = start + per_rank
    local_audios = audios[start:end]
    local_audio_lens = audio_lens[start:end]

    local_embs = encode_audio_with_optional_chunking(
        perception, local_audios, local_audio_lens,
        chunk_size_seconds=chunk_size_seconds, sampling_rate=sampling_rate,
    )

    # All-gather across CP. Variable-length: pad to a common max-L first.
    H = local_embs[0].shape[-1]
    local_max_L = max(e.shape[0] for e in local_embs)
    max_L_t = torch.tensor(local_max_L, dtype=torch.long, device=device)
    dist.all_reduce(max_L_t, op=dist.ReduceOp.MAX, group=cp_mesh.get_group())
    max_L = int(max_L_t.item())

    local_stack = torch.zeros(per_rank, max_L, H, device=device, dtype=local_embs[0].dtype)
    local_lens = torch.zeros(per_rank, dtype=torch.long, device=device)
    for i, e in enumerate(local_embs):
        local_stack[i, : e.shape[0]] = e
        local_lens[i] = e.shape[0]

    gathered_stack = [torch.zeros_like(local_stack) for _ in range(cp_size)]
    gathered_lens = [torch.zeros_like(local_lens) for _ in range(cp_size)]
    dist.all_gather(gathered_stack, local_stack, group=cp_mesh.get_group())
    dist.all_gather(gathered_lens, local_lens, group=cp_mesh.get_group())

    full_embs: list[Tensor] = []
    for r in range(cp_size):
        for i in range(per_rank):
            full_idx = r * per_rank + i
            if full_idx >= B_aud:
                break  # dummy slot
            L = int(gathered_lens[r][i].item())
            full_embs.append(gathered_stack[r][i, :L])

    return full_embs
