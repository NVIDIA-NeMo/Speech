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

"""Grouped-GEMM building blocks for Transformer encoder execution.

This module centralizes reusable code for combining compatible matrix
multiplications across named Transformer encoders and MoE feed-forward experts:

* :func:`grouped_ffn_compute` executes stacked FFN weights with a batched GEMM
  backend and provides a per-group loop as a reference implementation.
* :class:`GroupedFeedForward` packages identically shaped FFNs in the stacked
  parameter layout consumed by :func:`grouped_ffn_compute`.
* :func:`bucket_ffns_by_shape` and :func:`pad_feedforward` reconcile
  heterogeneous FFN widths through shape bucketing or structural zero padding.
* :class:`GGEMMTransformerEncoder` exposes serial, grouped-FFN, and packed-head
  execution paths for a mapping of Transformer encoders. Compatible FFN units
  are combined without changing routing decisions or encoder outputs.

``baddbmm`` is the portable grouped-GEMM implementation used here. The public
layout and backend seam allow a fused CUTLASS or Triton implementation to be
added without coupling this module to a particular model architecture or task.
"""

import contextlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import and_masks, create_block_mask

# Prefer FlashAttention (including FA4 when registered), then efficient attention,
# with a math fallback for unsupported inputs.
_SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

from nemo.collections.asr.modules.moe_transformer_encoder import MoEFeedForward
from nemo.collections.asr.modules.transformer_encoder import FeedForward, TransformerEncoderConfig
from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking

__all__ = [
    'GroupedFeedForward',
    'GGEMMTransformerEncoder',
    'pad_feedforward',
    'bucket_ffns_by_shape',
    'grouped_ffn_compute',
    'GROUPED_GEMM_BACKENDS',
]

# Available grouped-GEMM backends for the position-wise FFN of a shape bucket.
#   'baddbmm' : one batched GEMM over the stacked experts (default; the portable
#               stand-in for a fused CUTLASS/Triton grouped-GEMM).
#   'loop'    : per-expert ``addmm`` reference; same math, slower, used to validate
#               the batched path and as a fallback where bmm is unavailable.
# A future 'triton'/'cutlass' backend (ragged per-group offsets) plugs in here
# without changing the parameter layout or any call site.
GROUPED_GEMM_BACKENDS = ('baddbmm', 'loop')


def _autocast_compute_dtype(x: torch.Tensor) -> torch.dtype:
    """Return the dtype GEMMs will use for ``x`` under the active autocast policy.

    Args:
        x (torch.Tensor): Reference tensor whose device determines the autocast policy.
            Shape: arbitrary

    Returns:
        dtype (torch.dtype): Active autocast dtype when autocast is enabled, else ``x.dtype``.
    """
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return x.dtype


def grouped_ffn_compute(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    drop_rate: float = 0.0,
    training: bool = False,
    backend: str = 'baddbmm',
) -> torch.Tensor:
    """Batched position-wise FFN over ``E`` stacked experts.

    Computes, for every expert ``e``::

        h_e   = gelu(x_e @ w1_e + b1_e)
        out_e = dropout(h_e) @ w2_e + b2_e

    matching the per-expert ``FeedForward`` (``Linear -> GELU -> Dropout ->
    Linear -> Dropout``) exactly, but for all experts at once.

    Args:
        x (torch.Tensor): Per-expert token batches (same ``T`` per expert).
            Shape: (E, T, d_model)
        w1 (torch.Tensor): Stacked input projection weights.
            Shape: (E, d_model, d_hidden)
        b1 (torch.Tensor): Stacked input projection biases.
            Shape: (E, 1, d_hidden)
        w2 (torch.Tensor): Stacked output projection weights.
            Shape: (E, d_hidden, d_model)
        b2 (torch.Tensor): Stacked output projection biases.
            Shape: (E, 1, d_model)
        drop_rate (float): Dropout probability (applied after GELU and after ``w2``).
        training (bool): Whether dropout is active (pass ``module.training``).
        backend (str): One of :data:`GROUPED_GEMM_BACKENDS`.

    Returns:
        out (torch.Tensor): Stacked expert outputs.
            Shape: (E, T, d_model)
    """
    # Match cached weights to autocast's GEMM dtype; unlike parameters, cached
    # tensors are not automatically recast on each forward.
    compute_dtype = _autocast_compute_dtype(x)
    x = x.to(compute_dtype)
    w1, b1 = w1.to(compute_dtype), b1.to(compute_dtype)
    w2, b2 = w2.to(compute_dtype), b2.to(compute_dtype)

    if backend == 'baddbmm':
        hidden = torch.baddbmm(b1, x, w1)  # (E, T, d_hidden)
        hidden = F.gelu(hidden)
        hidden = F.dropout(hidden, p=drop_rate, training=training)
        out = torch.baddbmm(b2, hidden, w2)  # (E, T, d_model)
    elif backend == 'loop':
        outs = []
        for e in range(x.shape[0]):
            h = torch.addmm(b1[e], x[e], w1[e])  # (T, d_hidden)
            h = F.gelu(h)
            h = F.dropout(h, p=drop_rate, training=training)
            outs.append(torch.addmm(b2[e], h, w2[e]))  # (T, d_model)
        out = torch.stack(outs, dim=0)
    else:
        raise ValueError(f"Unknown grouped-GEMM backend '{backend}'; expected one of {GROUPED_GEMM_BACKENDS}.")

    return F.dropout(out, p=drop_rate, training=training)


class GroupedFeedForward(nn.Module):
    """Batched, numerically-exact replacement for ``E`` same-shape ``FeedForward``s.

    Each of the ``E`` experts is the standard NeMo Transformer FFN
    ``Linear(d_model, d_hidden) -> GELU -> Dropout -> Linear(d_hidden, d_model)
    -> Dropout``. Instead of holding ``E`` separate :class:`FeedForward` modules
    and looping over them (``E`` kernel launches per projection), this module
    stacks the expert weights and evaluates all experts with two batched matmuls
    (``torch.baddbmm``), which lowers to a single batched/grouped GEMM on GPU.

    This is the portable stand-in for a fused grouped-GEMM kernel: same math,
    one launch. Swapping in a CUTLASS/Triton grouped-GEMM later only changes the
    two ``baddbmm`` calls, not the parameter layout or the public API.

    Parameter layout (registered as ``nn.Parameter``):

    - ``w1``: ``(E, d_model, d_hidden)`` -- input projections (transposed from the
      ``nn.Linear`` ``(out, in)`` convention so we can do ``x @ w1``).
    - ``b1``: ``(E, 1, d_hidden)``
    - ``w2``: ``(E, d_hidden, d_model)``
    - ``b2``: ``(E, 1, d_model)``

    Args:
        num_experts (int): Number of expert FFNs ``E`` batched together.
        d_model (int): Input/output width shared by every expert in this group.
        d_hidden (int): FFN inner width shared by every expert in this group.
        drop_rate (float): Dropout probability (matches ``FeedForward``). Defaults to 0.0.
        backend (str): Grouped-GEMM backend, one of :data:`GROUPED_GEMM_BACKENDS`.
            Defaults to ``'baddbmm'``.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_hidden: int,
        drop_rate: float = 0.0,
        backend: str = 'baddbmm',
    ):
        """Initialize stacked expert FFN parameters.

        Args:
            num_experts (int): Number of expert FFNs ``E`` batched together.
            d_model (int): Input/output width shared by every expert in this group.
            d_hidden (int): FFN inner width shared by every expert in this group.
            drop_rate (float): Dropout probability (matches ``FeedForward``).
            backend (str): Grouped-GEMM backend, one of :data:`GROUPED_GEMM_BACKENDS`.
        """
        super().__init__()
        if backend not in GROUPED_GEMM_BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'; expected one of {GROUPED_GEMM_BACKENDS}.")
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.drop_rate = drop_rate
        self.backend = backend

        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, d_hidden))
        self.b1 = nn.Parameter(torch.zeros(num_experts, 1, d_hidden))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_hidden, d_model))
        self.b2 = nn.Parameter(torch.zeros(num_experts, 1, d_model))

        # Match nn.Linear's default Kaiming-uniform init per expert so a freshly
        # constructed GroupedFeedForward behaves like a stack of fresh FeedForwards.
        for e in range(num_experts):
            nn.init.kaiming_uniform_(self.w1[e].t(), a=5**0.5)
            nn.init.kaiming_uniform_(self.w2[e].t(), a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate all grouped FFNs on their corresponding token batches.

        Args:
            x (torch.Tensor): Per-expert token batches; the ``e``-th slice is the
                dense token batch for expert ``e``. Every expert sees the same
                number of tokens ``T``.
                Shape: (E, T, d_model)

        Returns:
            out (torch.Tensor): Stacked expert outputs.
                Shape: (E, T, d_model)
        """
        if x.dim() != 3 or x.shape[0] != self.num_experts or x.shape[2] != self.d_model:
            raise ValueError(
                f"GroupedFeedForward expects input of shape (E={self.num_experts}, T, "
                f"d_model={self.d_model}), got {tuple(x.shape)}."
            )
        return grouped_ffn_compute(
            x,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            drop_rate=self.drop_rate,
            training=self.training,
            backend=self.backend,
        )

    @classmethod
    def from_feedforwards(
        cls, ffns: Sequence[FeedForward], drop_rate: float = 0.0, backend: str = 'baddbmm'
    ) -> "GroupedFeedForward":
        """Build a grouped FFN from a list of identically-shaped ``FeedForward``s.

        The resulting module is numerically equivalent (up to float reduction
        order) to running each input ``FeedForward`` on its own token batch.

        Args:
            ffns (Sequence): Sequence of ``FeedForward`` modules sharing the same
                ``(d_model, d_hidden)`` shape.
            drop_rate (float): Dropout probability for the grouped module.
            backend (str): Grouped-GEMM backend, one of :data:`GROUPED_GEMM_BACKENDS`.

        Returns:
            grouped (GroupedFeedForward): Stacked FFN module with weights copied
                from ``ffns``.
        """
        if len(ffns) == 0:
            raise ValueError("from_feedforwards requires at least one FeedForward.")
        w1_0 = ffns[0].net[0]
        d_model = w1_0.in_features
        d_hidden = w1_0.out_features
        for i, ff in enumerate(ffns):
            l1, l2 = ff.net[0], ff.net[3]
            if (l1.in_features, l1.out_features, l2.in_features, l2.out_features) != (
                d_model,
                d_hidden,
                d_hidden,
                d_model,
            ):
                raise ValueError(
                    f"FeedForward {i} has shape mismatch; all experts in a group must share "
                    f"(d_model={d_model}, d_hidden={d_hidden})."
                )
        grouped = cls(num_experts=len(ffns), d_model=d_model, d_hidden=d_hidden, drop_rate=drop_rate, backend=backend)
        with torch.no_grad():
            for e, ff in enumerate(ffns):
                grouped.w1[e].copy_(ff.net[0].weight.t())
                grouped.b1[e, 0].copy_(ff.net[0].bias)
                grouped.w2[e].copy_(ff.net[3].weight.t())
                grouped.b2[e, 0].copy_(ff.net[3].bias)
        return grouped


def pad_feedforward(ffn: FeedForward, target_d_model: int) -> FeedForward:
    """Zero-pad a narrow ``FeedForward`` up to ``target_d_model`` (option A).

    Returns a NEW ``FeedForward`` whose input/output width is ``target_d_model``
    while the FFN hidden width is unchanged. The original weights occupy the top
    ``d_model`` rows/columns; the rest are zeros. Running this padded FFN on
    ``[x; 0]`` (the original activation padded with zeros) and slicing the top
    ``d_model`` outputs reproduces the original FFN exactly, so it can join a
    uniform ``target_d_model`` grouped-GEMM bucket without changing the result.

    Args:
        ffn (FeedForward): Source feed-forward with ``in_features = out_features = d_model``.
        target_d_model (int): Wider model width to pad up to (>= the source ``d_model``).

    Returns:
        padded (FeedForward): New feed-forward of width ``target_d_model`` with
            the same hidden width as ``ffn``.
    """
    l1, l2 = ffn.net[0], ffn.net[3]
    d_model = l1.in_features
    d_hidden = l1.out_features
    if target_d_model < d_model:
        raise ValueError(f"target_d_model ({target_d_model}) must be >= source d_model ({d_model}).")

    cfg = TransformerEncoderConfig(
        d_model=target_d_model,
        ff_expansion=d_hidden / target_d_model,
        drop_rate=ffn.net[2].p,
    )
    padded = FeedForward(cfg)
    # FeedForward derives d_hidden via int(ff_expansion * d_model); guard against
    # rounding so the padded hidden width matches the source exactly.
    if padded.net[0].out_features != d_hidden:
        raise ValueError(
            f"Rounding produced hidden={padded.net[0].out_features}, expected {d_hidden}; "
            "pass a target_d_model that divides evenly."
        )
    with torch.no_grad():
        padded.net[0].weight.zero_()
        padded.net[0].weight[:, :d_model].copy_(l1.weight)
        padded.net[0].bias.copy_(l1.bias)  # hidden width unchanged
        padded.net[3].weight.zero_()
        padded.net[3].weight[:d_model, :].copy_(l2.weight)
        padded.net[3].bias.zero_()
        padded.net[3].bias[:d_model].copy_(l2.bias)
    return padded


def bucket_ffns_by_shape(
    ffns: Sequence[FeedForward],
) -> Dict[Tuple[int, int], List[int]]:
    """Group FFN indices by their ``(d_model, d_hidden)`` shape.

    Each returned bucket can be fused into one :class:`GroupedFeedForward` /
    one grouped-GEMM call. Heterogeneous FFNs land in separate buckets, avoiding
    the extra computation introduced by padding them to a common width.

    Args:
        ffns (Sequence): Sequence of feed-forward modules to bucket by shape.

    Returns:
        buckets (Dict[Tuple[int, int], List[int]]): Mapping
            ``(d_model, d_hidden) -> [indices into ffns]``.
    """
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i, ff in enumerate(ffns):
        key = (ff.net[0].in_features, ff.net[0].out_features)
        buckets.setdefault(key, []).append(i)
    return buckets


def _padding_mask_mod(lengths):
    """Build a FlexAttention mask mod that masks padded key positions.

    Args:
        lengths (torch.Tensor): Valid frame counts per batch sample.
            Shape: (B,)

    Returns:
        pad_mask (Callable): FlexAttention mask mod; ``True`` when ``kv_idx`` is
            within the valid prefix for batch index ``b``.
    """

    def pad_mask(b, h, q_idx, kv_idx):
        """Return whether key position ``kv_idx`` is valid for batch ``b``.

        Args:
            b (int): Batch index.
            h (int): Head index (unused).
            q_idx (int): Query position index (unused).
            kv_idx (int): Key position index.

        Returns:
            mask_value (torch.Tensor): Whether the key position is valid.
                Shape: scalar
        """
        return kv_idx < lengths[b]

    return pad_mask


def _causal_mask_mod():
    """Build a FlexAttention mask mod for causal (lower-triangular) attention.

    Returns:
        causal (Callable): FlexAttention mask mod; ``True`` when
            ``q_idx >= kv_idx``.
    """

    def causal(b, h, q_idx, kv_idx):
        """Return whether query position ``q_idx`` may attend to key ``kv_idx``.

        Args:
            b (int): Batch index (unused).
            h (int): Head index (unused).
            q_idx (int): Query position index.
            kv_idx (int): Key position index.

        Returns:
            mask_value (torch.Tensor): Whether the query may attend to the key.
                Shape: scalar
        """
        return q_idx >= kv_idx

    return causal


class GGEMMTransformerEncoder(nn.Module):
    """Grouped-GEMM execution container for named Transformer encoders.

    Registers encoders and runs them via :meth:`forward`, :meth:`forward_all`,
    :meth:`forward_grouped` (FFN GEMMs), or :meth:`forward_packed` (heads + FFNs).
    Grouping changes launch strategy only; widths/pos-enc may differ, with shared
    layer count and frame rate required. Heads, fusion, and checkpoints stay owner-side.

    Args:
        experts (dict): Mapping from a stable name to an already-constructed
            ``TransformerEncoder``-family module.
    """

    def __init__(self, experts: Dict[str, nn.Module]):
        """Register named Transformer encoders for grouped execution.

        Args:
            experts (dict): Mapping from a stable name to an already-constructed
                ``TransformerEncoder``-family module.
        """
        super().__init__()
        if not experts:
            raise ValueError("GGEMMTransformerEncoder requires at least one expert encoder.")
        self.experts = nn.ModuleDict(experts)
        self.expert_names: List[str] = list(experts.keys())

        # Eval-only grouped-FFN weights; clear on parameter or device changes.
        self._use_packed_grouped: bool = True
        self._packed_cache: Dict[Tuple[object, ...], Tuple[torch.Tensor, ...]] = {}
        # Runtime-dtype RoPE buffers avoid repeated per-layer casts.
        self._rope_cache: Dict[Tuple[object, ...], Tuple[torch.Tensor, torch.Tensor]] = {}

        # Enable fused SDPA backends for unpadded, non-causal inputs.
        self.sdpa_fastpath: bool = True

    def train(self, mode: bool = True):
        """Set training mode and invalidate eval-time weight caches.

        Args:
            mode (bool): Whether to enable training mode.

        Returns:
            GGEMMTransformerEncoder: This module.
        """
        # Packed weights are an eval-time optimization; invalidate when (re-)entering
        # training so we never read stale, pre-update weights.
        self._packed_cache.clear()
        self._rope_cache.clear()
        return super().train(mode)

    def _apply(self, fn, recurse: bool = True):
        """Apply ``fn`` to parameters/buffers and clear non-registered caches.

        Args:
            fn (Callable): Callable applied to each parameter and buffer (e.g. ``.to``).
            recurse (bool): Whether to recurse into child modules.

        Returns:
            GGEMMTransformerEncoder: This module.
        """
        # These caches are plain tensors rather than registered buffers, so Module.to()
        # cannot migrate them safely. Rebuild lazily after any device/dtype transform.
        self._packed_cache.clear()
        self._rope_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def clear_packed_weights(self) -> None:
        """Drop eval-time grouped-FFN and RoPE caches.

        Call after mutating expert parameters outside ``load_state_dict`` so the
        next grouped/packed forward rebuilds its runtime-dtype tensors.
        """
        self._packed_cache.clear()
        self._rope_cache.clear()

    @property
    def expert_d_models(self) -> Dict[str, int]:
        """Return ``name -> d_model`` for encoders that expose ``d_model``."""
        return {name: m.d_model for name, m in self.experts.items() if hasattr(m, 'd_model')}

    def get_expert(self, expert_name: str) -> nn.Module:
        """Return the encoder registered as ``expert_name``.

        Args:
            expert_name (str): Name of the registered expert encoder.

        Returns:
            expert (nn.Module): The encoder module for ``expert_name``.
        """
        if expert_name not in self.experts:
            raise KeyError(f"Unknown expert '{expert_name}'. Available: {self.expert_names}.")
        return self.experts[expert_name]

    def forward(self, expert_name: str, audio_signal, length, bypass_pre_encode: bool = False):
        """Run a single expert along its own (unmodified) inference path.

        Args:
            expert_name (str): Which expert to run; must be one of ``self.expert_names``.
            audio_signal (torch.Tensor): Input features forwarded as-is to the expert.
                Shape: (B, C, T) mel/features, or (B, T, D) if ``bypass_pre_encode``.
            length (torch.Tensor): Valid frame counts per sample.
                Shape: (B,)
            bypass_pre_encode (bool): Passed through to the expert encoder.

        Returns:
            encoded (object): Expert-specific encoded output (typically ``(B, D, T')`` for NeMo
                Transformer encoders) and output lengths.
        """
        if expert_name not in self.experts:
            raise KeyError(f"Unknown expert '{expert_name}'. Available experts: {self.expert_names}.")
        return self.experts[expert_name](audio_signal, length, bypass_pre_encode=bypass_pre_encode)

    def forward_all(self, audio_signal, length, bypass_pre_encode: bool = False) -> Dict[str, object]:
        """Run every encoder serially and return a ``name -> output`` mapping.

        This is the non-grouped reference used to validate the grouped execution
        paths. Each encoder receives the same input and runs its native forward.

        Args:
            audio_signal (torch.Tensor): Input features shared by every encoder.
                Shape: (B, C, T) mel/features, or (B, T, D) if ``bypass_pre_encode``.
            length (torch.Tensor): Valid frame counts per sample.
                Shape: (B,)
            bypass_pre_encode (bool): Passed through to each expert encoder.

        Returns:
            outputs (Dict[str, object]): Mapping from expert name to that encoder's
                native forward output.
        """
        return {
            name: self.experts[name](audio_signal, length, bypass_pre_encode=bypass_pre_encode)
            for name in self.expert_names
        }

    @staticmethod
    def _prepend_prefix(
        x: torch.Tensor,
        length: torch.Tensor,
        prefix: Optional[torch.Tensor],
        mode: str = 'extend',
    ):
        """Splice a streaming cache ``prefix`` onto ``x``; returns ``(x, length, chunk)``.

        ``prefix`` is ``(B, P, d_model)`` of already-projected embeddings (the same
        representation ``x`` carries at this point: post-projection, pre-norm), so the
        concatenation is normalized as one sequence -- matching how Sortformer's
        ``forward_streaming_step`` feeds ``[spkcache | fifo | chunk]`` through the
        encoder body in a single pass.

        Two ways to splice, differing in what happens to the *other* experts:

        ``'extend'``
            ``[prefix | x]``, so the prefixed expert grows to ``P + T`` while every
            other expert stays at ``T`` and gets right-padded with zeros to match.
            Those pad frames are masked out but still add attention and FFN work.

        ``'replace'``
            ``[prefix | x[:, P:]]``, i.e. the cache *substitutes* for the expert's own
            leading ``P`` frames rather than extending past them. ``T`` is unchanged,
            so every expert in the group already agrees on ``T`` and nothing is padded.


        Args:
            x (torch.Tensor): Projected token embeddings before prefix splice.
                Shape: (B, T, d_model)
            length (torch.Tensor): Valid frame counts per sample.
                Shape: (B,)
            prefix (torch.Tensor, optional): Streaming-cache embeddings to prepend.
                Shape: (B, P, d_model)
            mode (str): Splice mode, either ``'extend'`` or ``'replace'``.

        Returns:
            x (torch.Tensor): Spliced embeddings (length may grow under ``'extend'``).
                Shape: (B, T', d_model)
            length (torch.Tensor): Updated valid frame counts.
                Shape: (B,)
            chunk (torch.Tensor): Current chunk embeddings for cache update.
                Shape: (B, T_chunk, d_model)
        """
        if prefix is None:
            return x, length, x
        if mode not in ('extend', 'replace'):
            raise ValueError(f"prefix mode must be 'extend' or 'replace', got {mode!r}.")
        if prefix.dim() != 3 or prefix.shape[0] != x.shape[0] or prefix.shape[-1] != x.shape[-1]:
            raise ValueError(f"prefix must be (B={x.shape[0]}, P, d_model={x.shape[-1]}), got {tuple(prefix.shape)}.")
        p = prefix.shape[1]
        prefix = prefix.to(dtype=x.dtype, device=x.device)
        if mode == 'extend':
            return torch.cat([prefix, x], dim=1), length + p, x
        if p > x.shape[1]:
            raise ValueError(
                f"prefix of {p} frames cannot replace the leading frames of a {x.shape[1]}-frame "
                "input; feed a window extended at least P frames further back, or use "
                "mode='extend'."
            )
        chunk = x[:, p:]
        # The prefix frames are cache and always valid, so a sample whose audio ended
        # inside them still has P valid frames.
        return torch.cat([prefix, chunk], dim=1), torch.clamp(length, min=p), chunk

    @staticmethod
    def _right_pad_to(x: torch.Tensor, t_max: int) -> torch.Tensor:
        """Right-pad ``x`` with zeros to ``T = t_max``.

        Padding on the RIGHT is load-bearing: every mask builder here assumes the
        valid frames of a sample are the prefix ``[0, length)``, so appending keeps
        ``_padding_additive_mask`` / ``_no_padding`` correct with ``length`` left at
        the true valid count.

        Args:
            x (torch.Tensor): Token embeddings to pad.
                Shape: (B, T, D)
            t_max (int): Target sequence length after right padding.

        Returns:
            x (torch.Tensor): Padded embeddings (unchanged when ``T >= t_max``).
                Shape: (B, t_max, D)
        """
        pad = t_max - x.shape[1]
        if pad <= 0:
            return x
        return F.pad(x, (0, 0, 0, pad))

    def _expert_pre(
        self,
        expert: nn.Module,
        audio_signal,
        length,
        bypass_pre_encode: bool,
        build_block_mask: bool = True,
        prefix: Optional[torch.Tensor] = None,
        t_max: Optional[int] = None,
        return_pre_encode: bool = False,
        prefix_mode: str = 'extend',
    ):
        """Run an expert's pre-layer stack (mirrors ``forward`` + pre-loop of
        ``forward_internal``).

        With ``prefix`` / ``t_max`` set, the streaming cache is prepended to the
        projected embeddings before the norm and the result is right-padded to a
        common ``t_max`` (see :meth:`_prepend_prefix` / :meth:`_right_pad_to`).
        The final return value contains the projected chunk embeddings (pre-prefix,
        pre-norm) when ``return_pre_encode`` is true, so a caller can update its
        cache without recomputing the projection; otherwise it is ``None``.

        Args:
            expert (nn.Module): Transformer encoder module to prepare.
            audio_signal (torch.Tensor): Raw input features.
                Shape: (B, C, T) mel/features, or (B, T, D) if ``bypass_pre_encode``.
            length (torch.Tensor): Valid frame counts per sample.
                Shape: (B,)
            bypass_pre_encode (bool): Skip the expert's pre-encode stack when True.
            build_block_mask (bool): Whether to build a FlexAttention block mask.
            prefix (torch.Tensor, optional): Streaming-cache embeddings.
                Shape: (B, P, d_model)
            t_max (int, optional): Optional common sequence length for right-padding.
            return_pre_encode (bool): Also return projected chunk embeddings for caching.
            prefix_mode (str): How ``prefix`` is spliced (``'extend'`` or ``'replace'``).

        Returns:
            x (torch.Tensor): Pre-layer embeddings ready for the encoder body.
                Shape: (B, T, d_model)
            layer_pos_emb (torch.Tensor, optional): Relative positional embeddings,
                or ``None`` for RoPE / no positional encoding.
            block_mask (object, optional): FlexAttention block mask, or ``None`` when not built.
            length (torch.Tensor): Updated valid frame counts.
                Shape: (B,)
            pre_encode_out (tuple, optional): ``(x_proj, proj_length)`` when
                ``return_pre_encode`` is True; otherwise ``None``.
        """
        if not bypass_pre_encode and audio_signal.shape[-2] != expert._feat_in:
            raise ValueError(f"Expert expects feat_in={expert._feat_in} on dim -2, got {audio_signal.shape[-2]}.")
        if bypass_pre_encode and audio_signal.shape[-1] != expert.d_model:
            raise ValueError(
                f"Expert expects d_model={expert.d_model} on dim -1 when bypassing pre-encode, "
                f"got {audio_signal.shape[-1]}."
            )
        if bypass_pre_encode:
            expert.update_max_seq_length(seq_length=audio_signal.size(1), device=audio_signal.device)
        else:
            expert.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)

        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),),
                audio_signal.size(1) if bypass_pre_encode else audio_signal.size(-1),
                dtype=torch.int64,
                device=audio_signal.device,
            )

        if not bypass_pre_encode:
            if isinstance(expert.pre_encode, FeatureStacking):
                x, length = expert.pre_encode(audio_signal, length)
            else:
                x = torch.transpose(audio_signal, 1, 2)
            if isinstance(expert.pre_encode, nn.Linear):
                x = expert.pre_encode(x)
            elif not isinstance(expert.pre_encode, FeatureStacking):
                x, length = expert.pre_encode(x=x, lengths=length)
            length = length.to(torch.int64)
        else:
            x = audio_signal
            length = length.to(torch.int64)

        # Projected chunk embeddings, before the norm -- what a streaming caller
        # pushes into its cache. Under prefix_mode='replace' this is the post-splice
        # chunk, i.e. the leading P frames consumed by the cache are excluded, so its
        # length has to drop by the same amount to stay in step.
        pre_len = length
        n_before = x.shape[1]
        x, length, x_proj = self._prepend_prefix(x, length, prefix, mode=prefix_mode)
        proj_length = (pre_len - (n_before - x_proj.shape[1])).clamp(min=0)

        if expert.self_attention_model == "rope":
            if expert.xscale:
                x = x * expert.xscale
            x = expert.dropout_pre_encoder(x)
            pos_emb = None
        elif expert.pos_enc is not None:
            x, pos_emb = expert.pos_enc(x=x)
        else:
            pos_emb = None
        x = expert.embed_norm(x)
        if t_max is not None:
            x = self._right_pad_to(x, t_max)  # `length` stays at the true valid count

        block_mask = None
        if build_block_mask:
            B, T, _ = x.shape
            if expert.attn_mode == "causal":
                mask_mod = and_masks(_causal_mask_mod(), _padding_mask_mod(length))
            else:
                mask_mod = _padding_mask_mod(length)
            block_mask = create_block_mask(mask_mod, B=B, H=1, Q_LEN=T, KV_LEN=T, device=x.device)
        layer_pos_emb = pos_emb if expert.self_attention_model == "rel_pos" else None
        pre_encode_out = (x_proj, proj_length) if return_pre_encode else None
        return x, layer_pos_emb, block_mask, length, pre_encode_out

    def _experts_pre(
        self,
        encs: Dict[str, nn.Module],
        audio_signal: torch.Tensor,
        length: Optional[torch.Tensor],
        bypass_pre_encode: bool,
        build_block_mask: bool,
        prefix: Optional[Dict[str, torch.Tensor]] = None,
        return_pre_encode: bool = False,
        prefix_mode: str = 'extend',
    ) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor], object, torch.Tensor]]:
        """Prepare every expert, batching compatible ``FeatureStacking`` projections.

        Compatible projections share a stacking factor and input width but may
        have different output widths. In eval, this method stacks and pads their
        weights once, then evaluates one ``bmm`` over an expanded view of the
        common stacked features. The padded output uses additional workspace to
        reduce the number of projection launches.

        Unsupported/training/bypass cases retain the original per-expert path.
        FlexAttention masks are also shared by experts with the same attention mode;
        SDPA callers request no block mask.

        Args:
            encs (dict): Mapping of expert name to encoder module.
            audio_signal (torch.Tensor): Shared input features.
                Shape: (B, C, T) mel/features, or (B, T, D) if ``bypass_pre_encode``.
            length (torch.Tensor, optional): Valid frame counts per sample.
                Shape: (B,)
            bypass_pre_encode (bool): Skip pre-encode stacks when True.
            build_block_mask (bool): Whether to build FlexAttention block masks.
            prefix (Dict[str, torch.Tensor], optional): Per-expert streaming caches.
                Each value has shape (B, P, d_model).
            return_pre_encode (bool): Also return projected chunk embeddings per expert.
            prefix_mode (str): How prefixes are spliced (``'extend'`` or ``'replace'``).

        Returns:
            prepared (Dict): Mapping ``name -> (x, layer_pos_emb, block_mask, length)``,
                or ``(prepared, pre_encode_out)`` when ``return_pre_encode`` is True.
        """
        names = list(encs)
        prefix = prefix or {}
        for name in prefix:
            if name not in encs:
                raise KeyError(f"prefix references unknown expert '{name}'. Available: {names}.")
        if prefix and build_block_mask:
            # A prefix makes experts ragged, so they get right-padded to a common T
            # -- a FlexAttention BlockMask built for the unpadded T would no longer
            # describe the tensor. Streaming runs through the SDPA paths, which build
            # their mask from `length` per layer, so this combination never arises.
            raise ValueError(
                "prefix is only supported on the SDPA paths (build_block_mask=False); "
                "forward_grouped's FlexAttention masks cannot describe the padded T."
            )
        feature_stackers = [encs[name].pre_encode for name in names]
        can_batch = (
            not self.training
            and not bypass_pre_encode
            and feature_stackers
            and all(isinstance(pre, FeatureStacking) for pre in feature_stackers)
            and len({pre.subsampling_factor for pre in feature_stackers}) == 1
            and len({pre.proj.in_features for pre in feature_stackers}) == 1
            and len({encs[name]._feat_in for name in names}) == 1
        )
        if not can_batch:
            # Pad to the observed max rather than a precomputed one: the padding in
            # `_expert_pre` is applied after the norm, so appending it here instead
            # is the same tensor and works for any pre_encode kind.
            out = {
                name: self._expert_pre(
                    encs[name],
                    audio_signal,
                    length,
                    bypass_pre_encode,
                    build_block_mask=build_block_mask,
                    prefix=prefix.get(name),
                    return_pre_encode=return_pre_encode,
                    prefix_mode=prefix_mode,
                )
                for name in names
            }
            if prefix:
                t_max = max(v[0].shape[1] for v in out.values())
                out = {n: (self._right_pad_to(v[0], t_max),) + tuple(v[1:]) for n, v in out.items()}
            if return_pre_encode:
                return ({n: v[:4] for n, v in out.items()}, {n: v[4] for n, v in out.items()})
            return {n: v[:4] for n, v in out.items()}

        feat_in = encs[names[0]]._feat_in
        if audio_signal.shape[-2] != feat_in:
            raise ValueError(f"Experts expect feat_in={feat_in} on dim -2, got {audio_signal.shape[-2]}.")
        for expert in encs.values():
            expert.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)

        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),),
                audio_signal.size(-1),
                dtype=torch.int64,
                device=audio_signal.device,
            )
        else:
            length = length.to(torch.int64)

        factor = feature_stackers[0].subsampling_factor
        stacked = audio_signal.transpose(1, 2)
        B, T, C = stacked.shape
        pad_size = (factor - (T % factor)) % factor
        if pad_size:
            stacked = F.pad(stacked, (0, 0, 0, pad_size))
        T_out = (T + pad_size) // factor
        stacked = stacked.reshape(B * T_out, C * factor)
        out_length = feature_stackers[0].compute_num_out_frames(length)

        target_d = max(pre.proj.out_features for pre in feature_stackers)
        compute_dtype = _autocast_compute_dtype(stacked)
        key = ('pre_encode', tuple(names), target_d, compute_dtype)
        packed = self._packed_cache.get(key)
        if packed is None:
            with torch.no_grad():
                weights = []
                for pre in feature_stackers:
                    weight = pre.proj.weight.t()
                    weights.append(F.pad(weight, (0, target_d - weight.shape[1])))
                packed = (torch.stack(weights).to(compute_dtype).contiguous(),)
            self._packed_cache[key] = packed
        weights = packed[0]
        shared_input = stacked.to(compute_dtype).unsqueeze(0).expand(len(names), -1, -1)
        projected = torch.bmm(shared_input, weights)

        # Under 'extend' a prefixed expert grows past the others and they get padded up
        # to it. Under 'replace' the prefix consumes that expert's own leading frames,
        # so every expert already shares T_out and nothing is padded.
        t_max = T_out
        if prefix_mode == 'extend':
            t_max += max((p.shape[1] for p in prefix.values()), default=0)

        mask_cache = {}
        result = {}
        pre_encode_out = {}
        for slot, name in enumerate(names):
            expert = encs[name]
            x = projected[slot, :, : expert.d_model].reshape(B, T_out, expert.d_model)
            n_before = x.shape[1]
            x, length_n, chunk = self._prepend_prefix(x, out_length, prefix.get(name), mode=prefix_mode)
            # Pre-norm chunk embeddings a streaming caller pushes into its cache.
            if return_pre_encode:
                n_dropped = n_before - chunk.shape[1]
                pre_encode_out[name] = (chunk, (out_length - n_dropped).clamp(min=0))

            if expert.self_attention_model == "rope":
                if expert.xscale:
                    x = x * expert.xscale
                x = expert.dropout_pre_encoder(x)
                pos_emb = None
            elif expert.pos_enc is not None:
                x, pos_emb = expert.pos_enc(x=x)
            else:
                pos_emb = None
            x = expert.embed_norm(x)
            if prefix:
                x = self._right_pad_to(x, t_max)  # `length_n` stays the true count

            block_mask = None
            if build_block_mask:
                mask_key = expert.attn_mode
                block_mask = mask_cache.get(mask_key)
                if block_mask is None:
                    if expert.attn_mode == "causal":
                        mask_mod = and_masks(_causal_mask_mod(), _padding_mask_mod(out_length))
                    else:
                        mask_mod = _padding_mask_mod(out_length)
                    block_mask = create_block_mask(mask_mod, B=B, H=1, Q_LEN=T_out, KV_LEN=T_out, device=x.device)
                    mask_cache[mask_key] = block_mask
            layer_pos_emb = pos_emb if expert.self_attention_model == "rel_pos" else None
            result[name] = (x, layer_pos_emb, block_mask, length_n)
        if return_pre_encode:
            return result, pre_encode_out
        return result

    def _expert_post(self, expert: nn.Module, x, length):
        """Run an expert's post-layer stack (mirrors post-loop of ``forward_internal``).

        Args:
            expert (nn.Module): Transformer encoder module.
            x (torch.Tensor): Final-layer hidden states.
                Shape: (B, T, D)
            length (torch.Tensor): Valid frame counts per sample.
                Shape: (B,)

        Returns:
            x (torch.Tensor): Encoded output in NeMo layout.
                Shape: (B, D, T)
            length (torch.Tensor): Output lengths as int64.
                Shape: (B,)
        """
        x = expert.final_norm(x)
        if expert.out_proj is not None:
            x = expert.out_proj(x)
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        return x, length.to(dtype=torch.int64)

    def _unified_weights(self, group, target_d: int, d_hidden: int, layer_idx: int, dtype: torch.dtype):
        """Stack the FFN weights of every unit in ``group`` into the grouped-bmm
        layout, zero-padding any unit whose ``d_model < target_d`` (option A; the
        pad rows/cols are structurally zero so the unit's output is unchanged).

        Cached per ``(layer_idx, d_hidden, target_d, names, dtype)`` in eval so
        repeated calls skip the ``stack`` + ``.to(dtype)`` that would otherwise
        dominate the FFN cost; rebuilt live (grad-preserving) under training.

        Args:
            group (list): List of FFN plan dicts, each with ``'units'`` feed-forward modules.
            target_d (int): Padded model width for the grouped bucket.
            d_hidden (int): Shared FFN inner width for the bucket.
            layer_idx (int): Transformer layer index (used for cache keying).
            dtype (torch.dtype): Runtime dtype for cached weight tensors.

        Returns:
            w1 (torch.Tensor): Stacked input weights.
                Shape: (E_total, target_d, d_hidden)
            b1 (torch.Tensor): Stacked input biases.
                Shape: (E_total, 1, d_hidden)
            w2 (torch.Tensor): Stacked output weights.
                Shape: (E_total, d_hidden, target_d)
            b2 (torch.Tensor): Stacked output biases.
                Shape: (E_total, 1, target_d)
        """
        units, srcs = [], []
        for p in group:
            for ff in p['units']:
                units.append(ff)
                srcs.append(ff.net[0].in_features)

        def _stack():
            """Stack and pad the grouped FFN weights."""
            w1s, b1s, w2s, b2s = [], [], [], []
            for ff, src_d in zip(units, srcs):
                w1 = ff.net[0].weight.t()  # (src_d, d_hidden)
                w2 = ff.net[3].weight.t()  # (d_hidden, src_d)
                b1 = ff.net[0].bias.unsqueeze(0)  # (1, d_hidden) -- hidden never padded
                b2 = ff.net[3].bias.unsqueeze(0)  # (1, src_d)
                if src_d != target_d:
                    pad = target_d - src_d
                    w1 = F.pad(w1, (0, 0, 0, pad))  # pad input rows -> (target_d, d_hidden)
                    w2 = F.pad(w2, (0, pad))  # pad output cols -> (d_hidden, target_d)
                    b2 = F.pad(b2, (0, pad))  # -> (1, target_d)
                w1s.append(w1)
                b1s.append(b1)
                w2s.append(w2)
                b2s.append(b2)
            return (torch.stack(w1s, 0), torch.stack(b1s, 0), torch.stack(w2s, 0), torch.stack(b2s, 0))

        if self.training or not self._use_packed_grouped:
            return _stack()
        key = (layer_idx, 'unified', d_hidden, target_d, tuple(p['name'] for p in group), dtype)
        cached = self._packed_cache.get(key)
        if cached is None:
            with torch.no_grad():
                w1, b1, w2, b2 = _stack()
                cached = (
                    w1.to(dtype).contiguous(),
                    b1.to(dtype).contiguous(),
                    w2.to(dtype).contiguous(),
                    b2.to(dtype).contiguous(),
                )
            self._packed_cache[key] = cached
        return cached

    def _unified_ffn_step(self, encs, state, layer_idx: int, backend: str, moe_mode: str = 'dense') -> None:
        """FFN sub-block for layer ``layer_idx`` fusing *all* experts' FFNs.

        Builds one grouped GEMM per inner-width (``d_hidden``) bucket over every
        supported FFN unit.

        - ``'dense'`` (default): the MoE's ``num_experts`` experts join the shared
          bucket and run on *all* tokens, then are recombined with the router's
          renormalized top-k weights. Exact, one big batched GEMM, but ~``num_experts
          / top_k`` redundant FFN FLOPs.
        - ``'topk'``: only the routed top-k expert/token pairs are computed, in a
          separate capacity-padded batched GEMM (:meth:`_moe_topk_ffn_step`).
          Exact (capacity = max expert load, no drops); far less compute/memory
          when compute-bound, at the cost of gather/scatter + a second launch.

        Residual updates are applied in place.

        Args:
            encs (dict): Mapping of expert name to encoder module.
            state (dict): Per-expert mutable state dicts (``'x'``, ``'length'``, etc.).
            layer_idx (int): Transformer layer index to execute.
            backend (str): Grouped-GEMM backend (:data:`GROUPED_GEMM_BACKENDS`).
            moe_mode (str): MoE strategy, either ``'dense'`` or ``'topk'``.
        """
        if moe_mode not in ('dense', 'topk'):
            raise ValueError(f"moe_mode must be 'dense' or 'topk', got {moe_mode!r}.")
        plans = []
        for n in self.expert_names:
            ffn = encs[n].layers[layer_idx].ffn
            if isinstance(ffn, MoEFeedForward):
                plans.append({'name': n, 'kind': 'moe', 'units': list(ffn.experts), 'moe': ffn})
            elif isinstance(ffn, FeedForward):
                plans.append({'name': n, 'kind': 'dense', 'units': [ffn], 'moe': None})
            else:
                # Anything we cannot express as FeedForward units runs its own path.
                layer = encs[n].layers[layer_idx]
                state[n]['x'] = state[n]['x'] + layer.drop(ffn(layer.norm2(state[n]['x'])))

        if moe_mode == 'topk':
            # Sparse MoE: compute only routed top-k pairs in a separate grouped
            # call; dense units stay in their shared bucket.
            for p in plans:
                if p['kind'] == 'moe':
                    self._moe_topk_ffn_step(encs, state, layer_idx, p, backend)
            grouped = [p for p in plans if p['kind'] == 'dense']
        else:
            grouped = [p for p in plans if p['kind'] in ('dense', 'moe')]
        if not grouped:
            return

        by_hidden: Dict[int, List[dict]] = {}
        for p in grouped:
            by_hidden.setdefault(p['units'][0].net[0].out_features, []).append(p)

        for d_hidden, group in by_hidden.items():
            target_d = max(p['units'][0].net[0].in_features for p in group)
            rows, layout, slot = [], [], 0
            for p in group:
                n = p['name']
                layer = encs[n].layers[layer_idx]
                h = layer.norm2(state[n]['x'])  # (B, T, src_d)
                B, T, src_d = h.shape
                hf = h.reshape(B * T, src_d)
                hf_p = hf if src_d == target_d else F.pad(hf, (0, target_d - src_d))
                n_units = len(p['units'])
                entry = {
                    'name': n,
                    'kind': p['kind'],
                    'src_d': src_d,
                    'slot': slot,
                    'n_units': n_units,
                    'B': B,
                    'T': T,
                    'W': None,
                }
                if p['kind'] == 'moe':
                    moe = p['moe']
                    gate = moe.router(hf)  # (N, num_experts) softmax probs
                    topv, topi = torch.topk(gate, moe.top_k, dim=-1)
                    if moe.top_k > 1:
                        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
                    Wmat = torch.zeros(B * T, moe.num_experts, dtype=gate.dtype, device=gate.device)
                    entry['W'] = Wmat.scatter_(1, topi, topv)
                    rows.extend([hf_p] * n_units)  # dense MoE: every expert sees all tokens
                else:
                    rows.append(hf_p)
                layout.append(entry)
                slot += n_units

            H = torch.stack(rows, dim=0)  # (E_total, N, target_d)
            compute_dtype = _autocast_compute_dtype(H)
            w1, b1, w2, b2 = self._unified_weights(group, target_d, d_hidden, layer_idx, compute_dtype)
            drop_rate = group[0]['units'][0].net[2].p
            out = grouped_ffn_compute(H, w1, b1, w2, b2, drop_rate=drop_rate, training=self.training, backend=backend)

            for entry in layout:
                n, slot, src_d = entry['name'], entry['slot'], entry['src_d']
                B, T = entry['B'], entry['T']
                layer = encs[n].layers[layer_idx]
                if entry['kind'] == 'dense':
                    o = out[slot][:, :src_d].reshape(B, T, src_d)
                else:
                    ne = entry['n_units']
                    o_slots = out[slot : slot + ne][:, :, :src_d]  # (ne, N, src_d)
                    # fp32 recombine to mirror the MoE's fp32 index_add accumulation.
                    Wt = entry['W'].t().unsqueeze(-1).float()  # (ne, N, 1)
                    o = (o_slots.float() * Wt).sum(0).to(state[n]['x'].dtype).reshape(B, T, src_d)
                state[n]['x'] = state[n]['x'] + layer.drop(o)

    def _moe_weights(self, name: str, moe, layer_idx: int, dtype: torch.dtype):
        """Stack the MoE's ``num_experts`` expert FFNs into grouped-bmm layout; cached in eval.

        Args:
            name (str): Expert encoder name (used for cache keying).
            moe (MoEFeedForward): ``MoEFeedForward`` module whose expert weights are stacked.
            layer_idx (int): Transformer layer index (used for cache keying).
            dtype (torch.dtype): Runtime dtype for cached weight tensors.

        Returns:
            w1 (torch.Tensor): Stacked input weights.
                Shape: (ne, d_model, d_hidden)
            b1 (torch.Tensor): Stacked input biases.
                Shape: (ne, 1, d_hidden)
            w2 (torch.Tensor): Stacked output weights.
                Shape: (ne, d_hidden, d_model)
            b2 (torch.Tensor): Stacked output biases.
                Shape: (ne, 1, d_model)
        """

        def _stack():
            """Stack the MoE expert weights."""
            ffns = list(moe.experts)
            return (
                torch.stack([f.net[0].weight.t() for f in ffns], 0),
                torch.stack([f.net[0].bias.unsqueeze(0) for f in ffns], 0),
                torch.stack([f.net[3].weight.t() for f in ffns], 0),
                torch.stack([f.net[3].bias.unsqueeze(0) for f in ffns], 0),
            )

        if self.training or not self._use_packed_grouped:
            return _stack()
        key = (layer_idx, 'moe', name, dtype)
        cached = self._packed_cache.get(key)
        if cached is None:
            with torch.no_grad():
                w1, b1, w2, b2 = _stack()
                cached = (
                    w1.to(dtype).contiguous(),
                    b1.to(dtype).contiguous(),
                    w2.to(dtype).contiguous(),
                    b2.to(dtype).contiguous(),
                )
            self._packed_cache[key] = cached
        return cached

    def _moe_topk_ffn_step(self, encs, state, layer_idx: int, p: dict, backend: str) -> None:
        """Sparse MoE FFN: compute ONLY the routed top-k expert/token pairs.

        Mirrors :meth:`MoEFeedForward.forward` but batches the per-expert matmuls
        into a single grouped GEMM.

        Args:
            encs (dict): Mapping of expert name to encoder module.
            state (dict): Per-expert mutable state dicts updated in place.
            layer_idx (int): Transformer layer index to execute.
            p (dict): MoE FFN plan dict with ``'name'``, ``'moe'``, and related fields.
            backend (str): Grouped-GEMM backend (:data:`GROUPED_GEMM_BACKENDS`).
        """
        n = p['name']
        moe = p['moe']
        layer = encs[n].layers[layer_idx]
        x = state[n]['x']
        h = layer.norm2(x)
        B, T, d = h.shape
        N = B * T
        x_flat = h.reshape(N, d)
        ne, top_k = moe.num_experts, moe.top_k

        gate = moe.router(x_flat)  # (N, ne) softmax probs
        topv, topi = torch.topk(gate, top_k, dim=-1)
        if top_k > 1:
            topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)

        M = N * top_k
        flat_expert = topi.reshape(-1)  # (M,)
        flat_weight = topv.reshape(-1)  # (M,)
        flat_token = torch.arange(N, device=x.device).unsqueeze(1).expand(N, top_k).reshape(-1)  # (M,)
        counts = torch.bincount(flat_expert, minlength=ne)  # (ne,)
        capacity = int(counts.max().item()) if M > 0 else 0
        if capacity == 0:
            return  # no tokens routed (degenerate); nothing to add

        # Sort dispatch rows by expert so each expert's tokens are contiguous, then
        # compute each row's slot within its expert segment (0..count_e-1).
        order = torch.sort(flat_expert, stable=True).indices
        s_expert = flat_expert[order]
        s_token = flat_token[order]
        s_weight = flat_weight[order]
        seg_start = torch.zeros(ne, dtype=torch.long, device=x.device)
        if ne > 1:
            seg_start[1:] = torch.cumsum(counts, 0)[:-1]
        within = torch.arange(M, device=x.device) - seg_start[s_expert]  # (M,)

        # Capacity-padded per-expert token buffer (unused slots stay zero).
        buf = x_flat.new_zeros(ne, capacity, d)
        buf[s_expert, within] = x_flat.index_select(0, s_token)

        compute_dtype = _autocast_compute_dtype(buf)
        w1, b1, w2, b2 = self._moe_weights(n, moe, layer_idx, compute_dtype)
        out = grouped_ffn_compute(
            buf,
            w1,
            b1,
            w2,
            b2,
            drop_rate=moe.experts[0].net[2].p,
            training=self.training,
            backend=backend,
        )  # (ne, capacity, d)

        disp_out = out[s_expert, within]  # (M, d) -- one row per (token, expert) pair
        acc = torch.zeros(N, d, dtype=torch.float32, device=x.device)
        acc.index_add_(0, s_token, disp_out.float() * s_weight.float().unsqueeze(-1))
        o = acc.to(x.dtype).reshape(B, T, d)
        state[n]['x'] = x + layer.drop(o)

    def forward_grouped(
        self,
        audio_signal,
        length,
        bypass_pre_encode: bool = False,
        backend: str = 'baddbmm',
        moe_mode: str = 'dense',
    ) -> Dict[str, object]:
        """Run encoders in lockstep while grouping compatible FFN units.

        Each layer first runs the encoders' attention blocks independently, then
        evaluates supported FFN units in one grouped GEMM per ``d_hidden`` bucket.
        Dense FFNs contribute one unit, MoE FFNs contribute their expert units,
        and narrower units are structurally zero-padded to the bucket width.

        Args:
            audio_signal (torch.Tensor): Input features shared by every encoder.
                Shape: (B, C, T) mel/features, or (B, T, D) if ``bypass_pre_encode``.
            length (torch.Tensor): Valid frame counts per sample.
                Shape: (B,)
            bypass_pre_encode (bool): Passed through to each expert encoder.
            backend (str): Grouped-GEMM backend (:data:`GROUPED_GEMM_BACKENDS`).
            moe_mode (str): MoE strategy for fused FFN steps (``'dense'`` or ``'topk'``).

        Returns:
            outputs (Dict[str, object]): Mapping ``name -> (encoded, length)``,
                numerically equivalent to :meth:`forward_all` in eval mode.
        """
        if backend not in GROUPED_GEMM_BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'; expected one of {GROUPED_GEMM_BACKENDS}.")
        encs = {name: self.experts[name] for name in self.expert_names}
        for name, e in encs.items():
            if not hasattr(e, 'layers') or not hasattr(e, 'n_layers'):
                raise TypeError(
                    f"Expert '{name}' is not a flex TransformerEncoder-family module; "
                    f"forward_grouped requires per-layer access."
                )
        n_layers_set = {e.n_layers for e in encs.values()}
        if len(n_layers_set) != 1:
            raise ValueError(f"forward_grouped requires equal n_layers across experts, got {n_layers_set}.")
        n_layers = n_layers_set.pop()

        state: Dict[str, dict] = {}
        prepared = self._experts_pre(encs, audio_signal, length, bypass_pre_encode, build_block_mask=True)
        for name, e in encs.items():
            x, pos_emb, block_mask, ln = prepared[name]
            state[name] = {'x': x, 'pos_emb': pos_emb, 'block_mask': block_mask, 'length': ln}

        for i in range(n_layers):
            # Run each encoder's attention sub-block first.
            for name, e in encs.items():
                layer = e.layers[i]
                s = state[name]
                s['x'] = s['x'] + layer.drop(
                    layer.attn(layer.norm1(s['x']), block_mask=s['block_mask'], pos_emb=s['pos_emb'])
                )
            # Then fuse dense and MoE FFN units into grouped GEMMs by d_hidden,
            # padding narrower units within each bucket.
            self._unified_ffn_step(encs, state, i, backend, moe_mode=moe_mode)

        return {
            name: self._expert_post(encs[name], state[name]['x'], state[name]['length']) for name in self.expert_names
        }

    @staticmethod
    def _padding_additive_mask(length: torch.Tensor, T: int, dtype: torch.dtype) -> torch.Tensor:
        """Additive key-padding mask: 0 for valid keys, -inf for pads.

        Args:
            length (torch.Tensor): Valid frame counts per sample.
                Shape: (B,)
            T (int): Sequence length (number of key positions).
            dtype (torch.dtype): Floating dtype for the additive mask values.

        Returns:
            mask (torch.Tensor): Additive key-padding mask broadcastable over heads
                and queries.
                Shape: (B, 1, 1, T)
        """
        device = length.device
        valid = torch.arange(T, device=device)[None, :] < length[:, None]  # (B, T)
        mask = torch.zeros(length.shape[0], 1, 1, T, dtype=dtype, device=device)
        return mask.masked_fill(~valid[:, None, None, :], torch.finfo(dtype).min)

    @staticmethod
    def _no_padding(length: torch.Tensor, T: int) -> bool:
        """True iff every sequence fills all ``T`` frames (no key padding).

        Synchronizes once (host read) -- callers cache the result for the whole
        forward (length is layer-invariant) so the 32-layer attention loop stays
        sync-free. When True the SDPA padding mask can be dropped entirely, which
        lets the dispatcher pick the FlashAttention-2 kernel (it requires
        ``attn_mask=None``).

        Args:
            length (torch.Tensor): Valid frame counts per sample.
                Shape: (B,)
            T (int): Sequence length to test against.

        Returns:
            no_pad (bool): ``True`` when every sample has ``length >= T``.
        """
        return bool(torch.all(length >= T))

    def _opt_additive_mask(self, length: torch.Tensor, T: int, dtype: torch.dtype, device, no_pad: bool, causal: bool):
        """Build the additive attention mask, or ``None`` when none is needed.

        Returns ``None`` when there is neither padding nor causal masking (so SDPA
        gets ``attn_mask=None`` and can dispatch FlashAttention-2). Otherwise
        returns a float bias broadcastable to ``(B, H, T, T)`` combining the
        key-padding mask (skipped when ``no_pad``) and the causal mask.

        Args:
            length (torch.Tensor): Valid frame counts per sample.
                Shape: (B,)
            T (int): Sequence length (query and key positions).
            dtype (torch.dtype): Floating dtype for mask values.
            device (torch.device): Device for mask tensor allocation.
            no_pad (bool): Skip key-padding mask when every sequence is full.
            causal (bool): Include upper-triangular causal masking.

        Returns:
            base (torch.Tensor, optional): Additive attention mask, or ``None``
                when no masking is required. When present, broadcastable to
                ``(B, H, T, T)`` or ``(B, 1, 1, T)``.
        """
        base = None
        if not no_pad:
            base = self._padding_additive_mask(length, T, dtype)  # (B, 1, 1, T)
        if causal:
            causal_add = torch.zeros(T, T, dtype=dtype, device=device).masked_fill(
                torch.ones(T, T, dtype=torch.bool, device=device).triu(1), torch.finfo(dtype).min
            )  # (T, T) -> broadcasts to (B, 1, T, T)
            base = causal_add if base is None else base + causal_add
        return base

    def _group_additive_mask(
        self,
        gkey,
        gnames: List[str],
        head_counts: List[int],
        state: dict,
        T: int,
        dtype: torch.dtype,
        device,
        causal: bool,
    ):
        """Additive attention mask for one packed-head group, honouring per-expert lengths.

        When every encoder in the group shares a length this is the usual
        ``(B, 1, 1, T)`` key-padding mask (or ``None`` when nothing needs masking, so
        SDPA can dispatch FlashAttention-2). When lengths differ -- the streaming case,
        where one encoder carries a cache prefix and the others are right-padded to
        match -- each encoder's ``(B, 1, 1, T)`` mask is expanded over its own head count
        and concatenated on the head axis into ``(B, Hg, 1, T)``, which SDPA broadcasts
        over queries. This keeps storage linear in sequence length instead of creating
        a full query-by-key mask for every packed head.

        The mask is cached on ``state`` because lengths do not change across layers.

        Args:
            gkey (tuple): Packed-head group key ``(head_dim, pos_scheme, attn_mode)``.
            gnames (list): Expert names in this packed-head group.
            head_counts (list): Per-expert head counts aligned with ``gnames``.
            state (dict): Shared forward state dict (holds per-expert ``length``, ``no_pad``).
            T (int): Sequence length for mask construction.
            dtype (torch.dtype): Floating dtype for mask values.
            device (torch.device): Device for mask tensor allocation.
            causal (bool): Include upper-triangular causal masking.

        Returns:
            base (torch.Tensor, optional): Additive attention mask for the group,
                or ``None`` when no masking is required. When present, shape is
                ``(B, 1, 1, T)`` or ``(B, Hg, 1, T)``.
        """
        cache = state.setdefault('_mask_cache', {})
        ckey = (gkey, causal, dtype)
        if ckey in cache:
            return cache[ckey]

        # The group can skip its padding mask only if EVERY member is unpadded.
        no_pad = all(state[n]['no_pad'] for n in gnames) and self.sdpa_fastpath
        lengths = [state[n]['length'] for n in gnames]
        uniform = no_pad or all(torch.equal(lengths[0], ln) for ln in lengths[1:])

        if uniform:
            base = self._opt_additive_mask(lengths[0], T, dtype, device, no_pad, causal)
        else:
            # (B, Hg, 1, T): each expert's key-padding mask over its own heads.
            parts = [
                self._padding_additive_mask(ln, T, dtype).expand(-1, hn, -1, -1)
                for ln, hn in zip(lengths, head_counts)
            ]
            base = torch.cat(parts, dim=1)
            if causal:
                causal_add = torch.zeros(T, T, dtype=dtype, device=device).masked_fill(
                    torch.ones(T, T, dtype=torch.bool, device=device).triu(1), torch.finfo(dtype).min
                )
                base = base + causal_add  # (B, Hg, 1, T) + (T, T) -> (B, Hg, T, T)
        cache[ckey] = base
        return base

    def _apply_rope_cached(self, rope, q: torch.Tensor, k: torch.Tensor):
        """Apply RoPE using cos/sin buffers cached in the q/k runtime dtype.

        Args:
            rope (object): RoPE module providing ``cos``, ``sin``, and ``_apply_rotary``.
            q (torch.Tensor): Query tensor before or after head split.
                Shape: (B, H, T_q, D)
            k (torch.Tensor): Key tensor.
                Shape: (B, H, T_k, D)

        Returns:
            q (torch.Tensor): Rotary-embedded queries.
                Shape: (B, H, T_q, D)
            k (torch.Tensor): Rotary-embedded keys.
                Shape: (B, H, T_k, D)
        """
        cos = rope.cos
        sin = rope.sin
        key = (
            id(rope),
            cos.data_ptr(),
            sin.data_ptr(),
            cos.device,
            cos.dtype,
            q.dtype,
        )
        runtime = self._rope_cache.get(key)
        if runtime is None:
            runtime = (cos.to(q.dtype), sin.to(q.dtype))
            self._rope_cache[key] = runtime
        cos, sin = runtime

        t_q = q.size(2)
        t_k = k.size(2)
        cache_len = t_k - t_q
        cos_k = cos[:t_k].view(1, 1, t_k, rope.d_k_rot)
        sin_k = sin[:t_k].view(1, 1, t_k, rope.d_k_rot)
        cos_q = cos[cache_len:t_k].view(1, 1, t_q, rope.d_k_rot)
        sin_q = sin[cache_len:t_k].view(1, 1, t_q, rope.d_k_rot)
        return rope._apply_rotary(q, cos_q, sin_q), rope._apply_rotary(k, cos_k, sin_k)

    def _expert_qkv(self, attn, x: torch.Tensor, pos_emb):
        """Project ``x`` to per-head q/k/v for one expert and return an optional
        additive score bias, reproducing the flex-attention math in
        :class:`MultiHeadAttention` (rope rotation / Transformer-XL rel-pos) so the
        batched-SDPA result matches the per-expert flex path to ULP tolerance.

        Args:
            attn (object): Multi-head attention module for one expert.
            x (torch.Tensor): Pre-attention hidden states.
                Shape: (B, T, d_model)
            pos_emb (torch.Tensor, optional): Relative positional embeddings for
                rel-pos attention.
                Shape: (B, 2T-1, d_model) when used; ``None`` for RoPE / no_pos.

        Returns:
            q (torch.Tensor): Query tensor.
                Shape: (B, H, T, head_dim)
            k (torch.Tensor): Key tensor.
                Shape: (B, H, T, head_dim)
            v (torch.Tensor): Value tensor.
                Shape: (B, H, T, head_dim)
            bias (torch.Tensor, optional): Additive relative-position bias, or
                ``None`` for RoPE / no positional encoding.
                Shape: (B, H, T, T) when present.
        """
        B, T, _ = x.shape
        H, D = attn.n_heads, attn.head_dim
        qkv = attn.w_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, T, D)
        if attn.qk_norm:
            q = attn.q_norm(q).to(v.dtype)
            k = attn.k_norm(k).to(v.dtype)
        if attn._uses_rope:
            q, k = self._apply_rope_cached(attn.rope, q, k)
            return q, k, v, None
        if attn._uses_rel_pos:
            # Mirror MultiHeadAttention._build_rel_pos_score_mod, but return the bias
            # tensor (SDPA additive mask) instead of a flex score_mod closure.
            p = attn.linear_pos(pos_emb).view(pos_emb.size(0), -1, H, D).transpose(1, 2)
            bias_u = attn.pos_bias_u.view(1, H, 1, D).to(q.dtype)
            bias_v = attn.pos_bias_v.view(1, H, 1, D).to(q.dtype)
            matrix_bd = torch.matmul(q + bias_v, p.transpose(-2, -1))  # (B, H, T, 2T-1)
            rel_pos_bias = attn._rel_shift(matrix_bd)[..., :T] * (D**-0.5)  # (B, H, T, T)
            return q + bias_u, k, v, rel_pos_bias
        return q, k, v, None  # no_pos

    def _packed_attention_step(self, encs, state, layer_idx: int) -> None:
        """Run every expert's attention for layer ``layer_idx`` as batched SDPA over
        packed heads (one call per (head_dim, pos-scheme, attn_mode) group), then
        apply each expert's out-projection + residual in place on ``state``.

        Args:
            encs (dict): Mapping of expert name to encoder module.
            state (dict): Per-expert mutable state dicts updated in place.
            layer_idx (int): Transformer layer index to execute.
        """
        names = self.expert_names
        # Compute per-expert q/k/v (+bias) from the pre-attn LayerNorm of each expert.
        qkv = {}
        groups: Dict[Tuple[int, str, str], List[str]] = {}
        for n in names:
            e = encs[n]
            layer = e.layers[layer_idx]
            attn = layer.attn
            h = layer.norm1(state[n]['x'])
            q, k, v, bias = self._expert_qkv(attn, h, state[n]['pos_emb'])
            qkv[n] = (q, k, v, bias)
            key = (attn.head_dim, attn.self_attention_model if attn._uses_rel_pos else 'nobias', e.attn_mode)
            groups.setdefault(key, []).append(n)

        for gkey, gnames in groups.items():
            B, _, T, D = qkv[gnames[0]][0].shape
            dtype = qkv[gnames[0]][0].dtype
            device = qkv[gnames[0]][0].device
            causal = encs[gnames[0]].attn_mode == 'causal'
            Q = torch.cat([qkv[n][0] for n in gnames], dim=1)  # (B, Hg, T, D)
            K = torch.cat([qkv[n][1] for n in gnames], dim=1)
            V = torch.cat([qkv[n][2] for n in gnames], dim=1)
            has_bias = any(qkv[n][3] is not None for n in gnames)
            # Additive padding(+causal) mask; None when fully packed (no padding,
            # non-causal) so the no-bias path can hit FlashAttention-2.
            base = self._group_additive_mask(
                gkey, gnames, [qkv[n][0].shape[1] for n in gnames], state, T, dtype, device, causal
            )
            with sdpa_kernel(_SDPA_BACKENDS) if self.sdpa_fastpath else contextlib.nullcontext():
                if has_bias:
                    # (B, Hg, T, T) additive mask = per-expert rel-pos bias (or 0) [+ pad/causal].
                    # When `base` is per-expert (B, Hg, 1, T) it must be sliced to each
                    # expert's own head span rather than added whole.
                    per_head_base = base is not None and base.shape[1] > 1
                    parts, hoff = [], 0
                    for n in gnames:
                        Hn = qkv[n][0].shape[1]
                        b = (
                            qkv[n][3]
                            if qkv[n][3] is not None
                            else torch.zeros(B, Hn, T, T, dtype=dtype, device=device)
                        )
                        if base is not None:
                            b = b + (base[:, hoff : hoff + Hn] if per_head_base else base)
                        hoff += Hn
                        parts.append(b)
                    attn_mask = torch.cat(parts, dim=1)
                    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
                elif base is None:
                    # Fully packed, no mask -> FlashAttention-2 eligible (is_causal
                    # handled here only because base is None implies non-causal).
                    out = F.scaled_dot_product_attention(Q, K, V)
                else:
                    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=base)
            # Split heads back per expert, out-project, residual.
            off = 0
            for n in gnames:
                layer = encs[n].layers[layer_idx]
                Hn = qkv[n][0].shape[1]
                o = out[:, off : off + Hn]  # (B, Hn, T, D)
                off += Hn
                o = o.transpose(1, 2).contiguous().view(B, T, Hn * D)
                o = layer.attn.out_proj(o)
                state[n]['x'] = state[n]['x'] + layer.drop(o)

    def forward_packed(
        self,
        audio_signal,
        length,
        bypass_pre_encode: bool = False,
        backend: str = 'baddbmm',
        moe_mode: str = 'dense',
        prefix: Optional[Dict[str, torch.Tensor]] = None,
        return_pre_encode: bool = False,
        prefix_mode: str = 'extend',
    ) -> Dict[str, object]:
        """Run encoders with packed-head SDPA and grouped FFNs.

        Per layer, every encoder's attention is computed as one batched
        ``scaled_dot_product_attention`` per (head_dim, pos-scheme, attn_mode)
        group over concatenated heads. Heads, rather than ``d_model``, are the
        packing unit, so compatible encoders do not require padded head slots.
        The FFN sub-block then uses one grouped GEMM per ``d_hidden`` bucket; see
        :meth:`_unified_ffn_step`. The output is numerically equivalent to
        :meth:`forward_all` in eval mode. Because attention switches from FlexAttention
        to SDPA, equality is tolerance-based rather than bitwise.

        Args:
            audio_signal (torch.Tensor): Input features shared by every encoder.
                Shape: (B, C, T) mel/features, or (B, T, D) if ``bypass_pre_encode``.
            length (torch.Tensor): Valid frame counts per sample.
                Shape: (B,)
            bypass_pre_encode (bool): Passed through to each expert encoder.
            backend (str): Grouped-GEMM backend (:data:`GROUPED_GEMM_BACKENDS`).
            moe_mode (str): MoE strategy for fused FFN steps (``'dense'`` or ``'topk'``).
            prefix (Dict[str, torch.Tensor], optional): Per-expert streaming caches.
                Each value has shape (B, P, d_model).
            return_pre_encode (bool): Also return projected chunk embeddings per expert.
            prefix_mode (str): How prefixes are spliced (see :meth:`_prepend_prefix`).

        Returns:
            outputs (Dict[str, object]): ``{name: (out, length)}``, or
                ``({name: (out, length)}, {name: (x_proj, out_length)})`` when
                ``return_pre_encode`` is True.
        """
        encs = {name: self.experts[name] for name in self.expert_names}
        for name, e in encs.items():
            if not hasattr(e, 'layers') or not hasattr(e, 'n_layers'):
                raise TypeError(f"Expert '{name}' is not a flex TransformerEncoder-family module.")
        n_layers_set = {e.n_layers for e in encs.values()}
        if len(n_layers_set) != 1:
            raise ValueError(f"forward_packed requires equal n_layers, got {n_layers_set}.")
        n_layers = n_layers_set.pop()

        state: Dict[str, dict] = {}
        prepared = self._experts_pre(
            encs,
            audio_signal,
            length,
            bypass_pre_encode,
            build_block_mask=False,
            prefix=prefix,
            return_pre_encode=return_pre_encode,
            prefix_mode=prefix_mode,
        )
        if return_pre_encode:
            prepared, pre_encode_out = prepared
        for name, e in encs.items():
            x, pos_emb, block_mask, ln = prepared[name]
            # Cache the no-padding flag once (length is layer-invariant) so the
            # per-layer attention loop never re-syncs to decide whether the SDPA
            # mask can be dropped (FlashAttention-2 path).
            state[name] = {
                'x': x,
                'pos_emb': pos_emb,
                'block_mask': block_mask,
                'length': ln,
                'no_pad': self._no_padding(ln, x.shape[1]),
            }

        for i in range(n_layers):
            self._packed_attention_step(encs, state, i)
            # FFN: fuse dense and MoE units into grouped GEMMs by d_hidden,
            # structurally padding narrower units within each bucket.
            self._unified_ffn_step(encs, state, i, backend, moe_mode=moe_mode)

        out = {
            name: self._expert_post(encs[name], state[name]['x'], state[name]['length']) for name in self.expert_names
        }
        if return_pre_encode:
            return out, pre_encode_out
        return out

    def _sdpa_attention_single(self, e, layer_idx: int, s: dict) -> None:
        """Run one encoder's attention through SDPA without packing its heads.

        Uses the same SDPA backend policy as :meth:`forward_packed`, then applies
        the output projection and residual in place on ``s``.

        This is the per-expert analogue of :meth:`_packed_attention_step`; the two
        produce numerically equivalent attention. It provides the serial-SDPA
        reference used to isolate head packing from the attention backend change;
        see :meth:`forward_serial_sdpa`.

        Args:
            e (nn.Module): Transformer encoder module.
            layer_idx (int): Transformer layer index to execute.
            s (dict): Per-expert mutable state dict updated in place.
        """
        layer = e.layers[layer_idx]
        attn = layer.attn
        h = layer.norm1(s['x'])
        q, k, v, bias = self._expert_qkv(attn, h, s['pos_emb'])
        B, Hn, T, D = q.shape
        dtype = q.dtype
        causal = e.attn_mode == 'causal'
        no_pad = s['no_pad'] and self.sdpa_fastpath
        base = self._opt_additive_mask(s['length'], T, dtype, q.device, no_pad, causal)
        with sdpa_kernel(_SDPA_BACKENDS) if self.sdpa_fastpath else contextlib.nullcontext():
            if bias is not None:
                attn_mask = bias if base is None else bias + base
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            elif base is None:
                # Fully packed, no mask -> FlashAttention-2 eligible.
                out = F.scaled_dot_product_attention(q, k, v)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=base)  # (B, Hn, T, D)
        o = out.transpose(1, 2).contiguous().view(B, T, Hn * D)
        o = attn.out_proj(o)
        s['x'] = s['x'] + layer.drop(o)

    def forward_serial_sdpa(self, audio_signal, length, bypass_pre_encode: bool = False) -> Dict[str, object]:
        """Run every encoder serially with SDPA attention and native FFNs.

        Each complete encoder stack runs before the next starts.

        * ``forward_all`` to ``forward_serial_sdpa`` changes the attention backend.
        * ``forward_serial_sdpa`` to ``forward_packed`` adds head packing and
          grouped FFN launch reduction.

        Unlike lockstep execution, this path does not require equal ``n_layers``.
        Its output is numerically equivalent to :meth:`forward_all` in eval mode,
        but is not bitwise identical because the attention backend differs.

        Args:
            audio_signal (torch.Tensor): Input features shared by every encoder.
                Shape: (B, C, T) mel/features, or (B, T, D) if ``bypass_pre_encode``.
            length (torch.Tensor): Valid frame counts per sample.
                Shape: (B,)
            bypass_pre_encode (bool): Passed through to each expert encoder.

        Returns:
            outputs (Dict[str, object]): Mapping from expert name to that encoder's
                encoded output and lengths.
        """
        encs = {name: self.experts[name] for name in self.expert_names}
        for name, e in encs.items():
            if not hasattr(e, 'layers') or not hasattr(e, 'n_layers'):
                raise TypeError(f"Expert '{name}' is not a flex TransformerEncoder-family module.")

        # Expert-major: run each encoder's full stack to completion before starting
        # the next. Nothing is shared across encoders; only the attention backend
        # differs from each encoder's native forward.
        out: Dict[str, object] = {}
        prepared = self._experts_pre(encs, audio_signal, length, bypass_pre_encode, build_block_mask=False)
        for name, e in encs.items():
            x, pos_emb, block_mask, ln = prepared[name]
            s = {
                'x': x,
                'pos_emb': pos_emb,
                'block_mask': block_mask,
                'length': ln,
                'no_pad': self._no_padding(ln, x.shape[1]),
            }
            for i in range(e.n_layers):
                self._sdpa_attention_single(e, i, s)
                layer = e.layers[i]
                s['x'] = s['x'] + layer.drop(layer.ffn(layer.norm2(s['x'])))
            out[name] = self._expert_post(e, s['x'], s['length'])

        return out

    @torch.no_grad()
    def verify_grouped_equivalence(
        self, audio_signal, length, bypass_pre_encode: bool = False, backend: str = 'baddbmm'
    ) -> Dict[str, float]:
        """Measure grouped-FFN error against the serial reference by encoder name.

        The comparison runs in eval mode so dropout is disabled and restores the
        previous mode afterward.

        Args:
            audio_signal (torch.Tensor): Input features shared by every encoder.
                Shape: (B, C, T) mel/features, or (B, T, D) if ``bypass_pre_encode``.
            length (torch.Tensor): Valid frame counts per sample.
                Shape: (B,)
            bypass_pre_encode (bool): Passed through to each expert encoder.
            backend (str): Grouped-GEMM backend (:data:`GROUPED_GEMM_BACKENDS`).

        Returns:
            errors (Dict[str, float]): Mapping ``name -> max(abs(reference - grouped))``.
        """
        was_training = self.training
        self.eval()
        try:
            ref = self.forward_all(audio_signal, length, bypass_pre_encode=bypass_pre_encode)
            grp = self.forward_grouped(audio_signal, length, bypass_pre_encode=bypass_pre_encode, backend=backend)
            return {name: (ref[name][0] - grp[name][0]).abs().max().item() for name in self.expert_names}
        finally:
            if was_training:
                self.train()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Namespace bare named-encoder keys under ``experts.<name>.``.

        A state dict may identify a registered encoder as
        ``<name>.layers.<layer>...`` instead of the container's canonical
        ``experts.<name>.layers.<layer>...`` path. This hook rewrites those keys before
        standard loading. Keys already below ``experts.`` remain unchanged.

        Args:
            state_dict (dict): Checkpoint state dict to rewrite in place.
            prefix (str): Module prefix for keys in ``state_dict``.
            local_metadata (dict): PyTorch load metadata (unused).
            strict (bool): Whether to enforce strict key matching.
            missing_keys (list): List populated with missing keys after load.
            unexpected_keys (list): List populated with unexpected keys after load.
            error_msgs (list): List populated with load error messages.
        """
        already = re.compile(r'^' + re.escape(prefix) + r'experts\.')
        bare_expert = re.compile(
            r'^' + re.escape(prefix) + r'(' + '|'.join(re.escape(n) for n in self.expert_names) + r')\.'
        )

        keys_to_add = {}
        keys_to_remove = []
        for key in list(state_dict.keys()):
            if already.match(key):
                continue
            m = bare_expert.match(key)
            if m:
                name = m.group(1)
                suffix = key[m.end() :]
                keys_to_add[f"{prefix}experts.{name}.{suffix}"] = state_dict[key]
                keys_to_remove.append(key)

        # Loading new weights invalidates any pre-packed grouped-FFN cache.
        self._packed_cache.clear()
        self._rope_cache.clear()

        for key in keys_to_remove:
            del state_dict[key]
        state_dict.update(keys_to_add)
        if keys_to_remove:
            print(
                f"GGEMMTransformerEncoder: Namespaced {len(keys_to_remove)} bare expert keys "
                f"under 'experts.<name>.' for experts {self.expert_names}."
            )

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
