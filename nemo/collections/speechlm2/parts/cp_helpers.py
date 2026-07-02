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

These helpers consolidate the CP-shape work needed to feed THD packed
batches into a Nemotron-V3 LLM whose attention/Mamba layers were CP-wired
by the Automodel parallelizer (``set_context_parallel_group()`` /
``mixer.cp = MambaContextParallel(...)``). Two concerns:

1. ``get_cp_mesh`` — read the CP submesh out of a device mesh, returning
   ``(None, 1, 0)`` when CP is inactive so callers can short-circuit.
2. ``encode_audio_with_cp_distribution`` — distribute the audio encoder
   forward across CP ranks so it isn't recomputed cp_size times. Pads to a
   multiple of cp_size with dummy zero-audios so every rank participates in
   FSDP all-gather; dummies are dropped after the post-encoder all-gather.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

from nemo.collections.speechlm2.parts.encoder_chunking import (
    _get_min_chunk_size_samples,
    encode_audio_with_optional_chunking,
)


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


def get_perception_fsdp_group(device_mesh):
    """Return the process group used to FSDP-shard the perception module, if any."""
    if device_mesh is None:
        return None
    dim_names = device_mesh.mesh_dim_names or ()
    if "dp_replicate" in dim_names and "dp_shard_cp" in dim_names:
        fsdp_mesh = device_mesh["dp_replicate", "dp_shard_cp"]
    elif "dp_shard_cp" in dim_names:
        fsdp_mesh = device_mesh["dp_shard_cp"]
    else:
        fsdp_mesh = device_mesh["dp"]
    if fsdp_mesh.size() <= 1:
        return None
    return fsdp_mesh.get_group()


def encode_audio_with_cp_distribution(
    perception,
    audios: Tensor,
    audio_lens: Tensor,
    *,
    chunk_size_seconds: Optional[float],
    sampling_rate: int,
    cp_mesh=None,
    spk_targets: Tensor | None = None,
    fsdp_sync_group=None,
    return_dummy_loss: bool = False,
) -> list[Tensor] | tuple[list[Tensor], Tensor | None]:
    """Distribute the audio encoder forward across CP ranks.

    Falls back to :func:`encode_audio_with_optional_chunking` when ``cp_mesh is
    None``.

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

    When ``fsdp_sync_group`` is provided and this rank has a text-only batch
    while another rank in the perception FSDP group has audio, this function
    runs a single dummy audio row through ``perception`` and returns a zero-valued
    loss term. Adding that term to the training loss preserves the autograd edge
    so FSDP forward/backward hooks fire on the text-only rank without affecting
    gradients numerically.
    """
    B_aud = int(audios.shape[0])
    fsdp_group_has_audio = _fsdp_group_has_audio(B_aud, audios.device, fsdp_sync_group)
    if B_aud == 0:
        dummy_loss = (
            _dummy_audio_loss_for_fsdp_sync(
                perception,
                audios,
                audio_lens,
                chunk_size_seconds=chunk_size_seconds,
                sampling_rate=sampling_rate,
            )
            if fsdp_group_has_audio
            else None
        )
        ans = []
        return (ans, dummy_loss) if return_dummy_loss else ans

    if cp_mesh is None:
        ans = encode_audio_with_optional_chunking(
            perception,
            audios,
            audio_lens,
            chunk_size_seconds=chunk_size_seconds,
            sampling_rate=sampling_rate,
            spk_targets=spk_targets,
        )
        return (ans, None) if return_dummy_loss else ans

    cp_size = cp_mesh.size()
    cp_group = cp_mesh.get_group()
    cp_rank = dist.get_rank(group=cp_group)
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
        if spk_targets is not None:
            dummy_targets = torch.zeros(
                pad_n,
                spk_targets.shape[1],
                spk_targets.shape[2],
                dtype=spk_targets.dtype,
                device=spk_targets.device,
            )
            spk_targets = torch.cat([spk_targets, dummy_targets], dim=0)

    start = cp_rank * per_rank
    end = start + per_rank
    local_audios = audios[start:end]
    local_audio_lens = audio_lens[start:end]
    local_spk_targets = spk_targets[start:end] if spk_targets is not None else None

    local_embs = encode_audio_with_optional_chunking(
        perception,
        local_audios,
        local_audio_lens,
        chunk_size_seconds=chunk_size_seconds,
        sampling_rate=sampling_rate,
        spk_targets=local_spk_targets,
    )

    # All-gather across CP. Variable-length: pad to a common max-L first.
    H = local_embs[0].shape[-1]
    local_max_L = max(e.shape[0] for e in local_embs)
    max_L_t = torch.tensor(local_max_L, dtype=torch.long, device=device)
    dist.all_reduce(max_L_t, op=dist.ReduceOp.MAX, group=cp_group)
    max_L = int(max_L_t.item())

    local_stack = torch.zeros(per_rank, max_L, H, device=device, dtype=local_embs[0].dtype)
    local_lens = torch.zeros(per_rank, dtype=torch.long, device=device)
    for i, e in enumerate(local_embs):
        local_stack[i, : e.shape[0]] = e
        local_lens[i] = e.shape[0]

    gathered_lens = [torch.zeros_like(local_lens) for _ in range(cp_size)]
    gathered_stack = differentiable_all_gather(local_stack, group=cp_group)
    dist.all_gather(gathered_lens, local_lens, group=cp_group)

    full_embs: list[Tensor] = []
    for r in range(cp_size):
        for i in range(per_rank):
            full_idx = r * per_rank + i
            if full_idx >= B_aud:
                break  # dummy slot
            L = int(gathered_lens[r][i].item())
            full_embs.append(gathered_stack[r][i, :L])

    return (full_embs, None) if return_dummy_loss else full_embs


def _fsdp_group_has_audio(B_aud: int, device: torch.device, fsdp_sync_group=None) -> bool:
    if fsdp_sync_group is None or not (dist.is_available() and dist.is_initialized()):
        return False
    local_has_audio = torch.tensor(1 if B_aud > 0 else 0, dtype=torch.int32, device=device)
    dist.all_reduce(local_has_audio, op=dist.ReduceOp.MAX, group=fsdp_sync_group)
    return bool(int(local_has_audio.item()))


def _dummy_audio_loss_for_fsdp_sync(
    perception,
    audios: Tensor,
    audio_lens: Tensor,
    *,
    chunk_size_seconds: Optional[float],
    sampling_rate: int,
) -> Tensor | None:
    # The preprocessor minimum alone can be too short after Conformer
    # subsampling, leaving BatchNorm with a single value per channel.
    dummy_len = max(_get_min_chunk_size_samples(perception), int(sampling_rate))
    dummy_audio = torch.zeros(1, dummy_len, dtype=audios.dtype, device=audios.device)
    dummy_lens = torch.full((1,), dummy_len, dtype=audio_lens.dtype, device=audio_lens.device)
    dummy_embs = encode_audio_with_optional_chunking(
        perception,
        dummy_audio,
        dummy_lens,
        chunk_size_seconds=chunk_size_seconds,
        sampling_rate=sampling_rate,
    )
    dummy_loss = sum(emb.float().sum() for emb in dummy_embs)
    return dummy_loss * 0.0
