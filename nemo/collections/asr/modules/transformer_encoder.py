# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import math
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import and_masks, create_block_mask, flex_attention

from nemo.collections.asr.models.configs import CacheAwareStreamingConfig
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    PositionalEncoding,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    RotaryPositionalEncoding,
)
from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking, StackingSubsampling
from nemo.core.classes.module import freeze, unfreeze
from nemo.utils import logging
from nemo.utils.decorators import experimental

flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


@dataclass
class TransformerEncoderConfig:
    """Configuration for ``TransformerEncoder`` and its sub-blocks.

    Args:
        feat_in: Input feature dimension (e.g. number of mel bins).
        d_model: Transformer encoder state dimension, i.e. the size of the residual stream that flows
            through every block (token/frame embedding size, attention input/output size, and
            feed-forward input/output size). Also known as ``hidden_size`` in HuggingFace
            ``transformers`` configs and ``embed_dim``/``d_model`` in PyTorch's
            ``nn.TransformerEncoderLayer``.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        drop_rate: Dropout probability applied inside attention and feed-forward sublayers.
        qkv_bias: If True, add a learnable bias to the fused Q/K/V projection. Many modern ASR/LM
            Transformers (e.g. HuggingFace Whisper) drop the bias on the K projection because a
            constant K bias adds the same scalar to every key and is wiped out by softmax's
            shift-invariance, making it a redundant parameter. Default ``False`` matches that style.
        qk_norm: If True, apply per-head ``LayerNorm`` to Q and K before the dot product. Stabilizes
            training by preventing exponential Q/K-norm growth and "attention entropy collapse"
            (Henry et al. 2020; used in OLMo 2, Gemma 3, Qwen 3). Cheap, ~no-op for inference.
        ff_expansion: Multiplier for the per-block FFN inner hidden size:
            ``ffn_hidden_size = int(ff_expansion * d_model)``. Only widens the intermediate FFN
            projection; FFN input/output stays at ``d_model``. Typical value ``4.0``; ``float``
            allows sub-1x experts for MoE. Equivalent to ``intermediate_size / hidden_size`` in
            HuggingFace and ``dim_feedforward / d_model`` in PyTorch's ``nn.TransformerEncoderLayer``.
        pre_block_norm: If True, apply ``LayerNorm`` to embeddings before the first Transformer block
            (BERT/ViT-style). Set False to match pre-norm Transformers such as Whisper or GPT-2.
        subsampling_factor: Frame-level subsampling factor performed by the pre-encoder.
        attn_mode: Attention pattern. Currently only ``"full"`` (bidirectional) is supported.
            Future modes: ``"causal"``, ``"lookahead"``, ``"local"``, ``"sliding_window"``.
        self_attention_model: Positional encoding / attention scoring scheme.

            - ``"rel_pos"`` (default): Transformer-XL relative positional encoding
              (https://arxiv.org/abs/1901.02860). The (b)+(d) cross/positional bias is computed
              from the relative-position embedding and injected into FlexAttention via a
              ``score_mod`` closure; the (c) global-content bias is folded into the query as
              ``Q + pos_bias_u``.
            - ``"abs_pos"``: sinusoidal absolute positional encoding added to embeddings
              before the first block; standard scaled dot-product attention.
            - ``"no_pos"`` (or ``None``): no positional encoding at all. The pre-encoder output
              is consumed directly by the Transformer blocks. ``xscaling``, ``pos_emb_max_len``,
              ``dropout_pre_encoder`` and ``dropout_emb`` are unused in this mode.
            - ``"rope"``: rotary position embedding applied to Q and K inside attention. No
              additive positional embedding is added to the embeddings; the standard scaled
              dot-product attention runs through FlexAttention with ``score_mod=None``.
        rope_base: Theta base for the rotary position embedding. Only used when
            ``self_attention_model='rope'``.
        rotary_fraction: Fraction of the per-head dim to rotate. Only used when
            ``self_attention_model='rope'``.
    """

    feat_in: int = 128
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 17
    drop_rate: float = 0.1
    qkv_bias: bool = False
    qk_norm: bool = False
    ff_expansion: float = 4.0
    pre_block_norm: bool = True
    subsampling_factor: int = 4
    # Attention mode: "full" (bidirectional) or "causal" (each token only attends to itself and earlier tokens).
    # Future: "lookahead", "local", "sliding_window".
    attn_mode: str = "full"
    self_attention_model: str = "rel_pos"
    rope_base: float = 10000.0
    rotary_fraction: float = 1.0


def _make_padding_mod(lengths):
    """Mask out padding positions based on per-sample lengths."""

    def pad_mask(b, h, q_idx, kv_idx):
        return kv_idx < lengths[b]

    return pad_mask


def _make_causal_mod():
    """Strictly causal — each query only attends to its own and earlier kv positions."""

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return causal


def _make_sliding_window_mod(left, right):
    """Sliding-window (local) attention.

    Query ``q_idx`` may attend only to keys within ``[q_idx - left, q_idx + right]``.
    A negative bound removes the limit on that side (``left < 0`` -> unlimited left
    context; ``right < 0`` -> unlimited right context / look-ahead). Callers are expected
    to pass at least one non-negative bound — when both are negative the window is
    unbounded (full attention) and this mod should be skipped entirely.

    Args:
        left (int): Maximum number of positions available to the left of each query.
        right (int): Maximum number of positions available to the right of each query.

    Returns:
        mask_mod (Callable): FlexAttention mask function for the requested sliding window.
    """

    def sliding_window(b, h, q_idx, kv_idx):
        # ``left`` / ``right`` are Python ints known at trace time, so these branches are
        # resolved once when FlexAttention traces the mask, not per element.
        if left >= 0 and right >= 0:
            return (kv_idx >= q_idx - left) & (kv_idx <= q_idx + right)
        if left >= 0:
            return kv_idx >= q_idx - left
        return kv_idx <= q_idx + right

    return sliding_window


def _make_chunked_limited_mod(left, right):
    """Chunk-aligned (``chunked_limited``) attention, matching ``ConformerEncoder``.

    Frames are grouped into non-overlapping chunks of ``chunk_size = right + 1``. A query
    attends to every frame of its own chunk (including in-chunk look-ahead) plus the
    ``left // chunk_size`` preceding chunks — whole chunks only, never a partial one.
    ``left < 0`` removes the left bound.

    Unlike :func:`_make_sliding_window_mod`, the look-ahead does **not** compound across
    layers: every layer stops at the same chunk boundary, so an N-layer stack still looks
    ahead at most to the end of the current chunk. That is what makes it streamable — see
    :class:`StreamingTransformerEncoder`.

    Args:
        left (int): Maximum left context in encoder frames; negative means unlimited.
        right (int): In-chunk right context. The attention chunk size is ``right + 1``.

    Returns:
        mask_mod (Callable): FlexAttention mask function for chunk-aligned limited context.
    """
    chunk = right + 1
    # ``left_chunks`` is a Python int (or None) known at trace time, so the branch below is
    # resolved once when FlexAttention traces the mask, not per element.
    left_chunks = (left // chunk) if left >= 0 else None

    def chunked_limited(b, h, q_idx, kv_idx):
        diff = (q_idx // chunk) - (kv_idx // chunk)
        if left_chunks is None:
            return diff >= 0
        return (diff >= 0) & (diff <= left_chunks)

    return chunked_limited


_SUPPORTED_ATTENTION_MODES = ("full", "causal")
_SUPPORTED_SELF_ATTENTION_MODELS = ("abs_pos", "rel_pos", "no_pos", "rope")
_SUPPORTED_ATT_CONTEXT_STYLES = ("sliding_window", "chunked_limited")


class FeedForward(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        ff_hidden = int(cfg.ff_expansion * cfg.d_model)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(cfg.drop_rate),
            nn.Linear(ff_hidden, cfg.d_model),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig, pos_enc=None):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.self_attention_model = cfg.self_attention_model
        self._uses_rel_pos = self.self_attention_model == "rel_pos"
        self._uses_rope = self.self_attention_model == "rope"
        if self.self_attention_model not in _SUPPORTED_SELF_ATTENTION_MODELS:
            raise ValueError(
                f"self_attention_model='{self.self_attention_model}' is not supported. "
                f"Supported modes: {_SUPPORTED_SELF_ATTENTION_MODELS}."
            )
        if self.head_dim < 16:
            raise ValueError(
                "PyTorch FlexAttention CUDA backend requires per-head embedding dimension >= 16, "
                f"but got head_dim={self.head_dim} from d_model={self.d_model}, n_heads={self.n_heads}."
            )

        # Rotary position embedding shared across layers; rotates Q/K before the attention
        # kernel. The shared module owns the cos/sin buffers (see ``TransformerEncoder``).
        if self._uses_rope:
            if pos_enc is None:
                raise ValueError("'rope' attention requires a RotaryPositionalEncoding via pos_enc.")
            self.rope = pos_enc
        else:
            self.rope = None

        self.w_qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        # Transformer-XL relative-position parameters (matrix b and matrix d from
        # https://arxiv.org/abs/1901.02860 Section 3.3). The "matrix c" term `u @ K^T` is
        # absorbed by passing `Q + pos_bias_u` as the query to FlexAttention.
        if self._uses_rel_pos:
            self.linear_pos = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.pos_bias_u = nn.Parameter(torch.zeros(self.n_heads, self.head_dim))
            self.pos_bias_v = nn.Parameter(torch.zeros(self.n_heads, self.head_dim))
        else:
            self.linear_pos = None
            self.pos_bias_u = None
            self.pos_bias_v = None

    def _rel_shift(self, x):
        """Transformer-XL relative-position shift.

        Delegates to ``RelPositionMultiHeadAttention.rel_shift`` (which does not reference
        ``self``) so the logic lives in a single place — NeMo's existing reference
        implementation in ``parts/submodules/multi_head_attention.py``.
        """
        return RelPositionMultiHeadAttention.rel_shift(None, x)

    def _build_rel_pos_score_mod(self, q, pos_emb):
        """Build the FlexAttention inputs that realize Transformer-XL relative attention.

        Implements the (b), (c), (d) terms of Transformer-XL Section 3.3
        (https://arxiv.org/abs/1901.02860) on top of FlexAttention:

        - Matrices (b) + (d) — the position-dependent score bias ``(Q + v) @ R^T`` rel-
          shifted into ``(q_idx, kv_idx)`` coordinates — are precomputed into a
          ``(B, H, T, T)`` tensor, scaled by ``1/sqrt(D)`` (to match FlexAttention's
          already-scaled ``QK^T`` scores), and captured by a local ``score_mod`` closure.
          FlexAttention's ``score_mod`` API fixes the callable signature, so the per-forward
          tensor is threaded in via closure capture rather than as an explicit argument.
          Keeping it local (instead of on ``self``) lets the ``(B, H, T, T)`` bias be freed
          as soon as the layer's attention call returns, so peak memory holds at most one
          layer's bias rather than all layers' biases at once.
        - Matrix (c) — the global-content bias ``u @ K^T`` — is folded into FlexAttention
          by rewriting the query as ``Q + pos_bias_u``, which is returned.

        Args:
            q: Query tensor with shape ``(B, H, T, D)``.
            pos_emb: Relative positional embedding ``(1, 2T - 1, d_model)`` produced by
                ``RelPositionalEncoding``.

        Returns:
            score_mod: Callable to pass as ``flex_attention(..., score_mod=...)``.
            q_with_bias_u: ``Q + pos_bias_u`` — the (c) "matrix c" query rewrite.
        """
        H, D = self.n_heads, self.head_dim
        T = q.size(-2)
        # pos_emb: (1, 2T - 1, d_model) -> p: (1, H, 2T - 1, D)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, H, D).transpose(1, 2)
        # pos_bias_{u,v}: (H, D) -> (1, H, 1, D) so they broadcast over the (B, H, T, D)
        # Q tensor against the head/depth axes rather than (incorrectly) against time.
        bias_u = self.pos_bias_u.view(1, H, 1, D).to(q.dtype)
        bias_v = self.pos_bias_v.view(1, H, 1, D).to(q.dtype)
        # Matrix b + d: ((Q + v) @ R^T) shifted into (q_idx, kv_idx) space, then scaled
        # by 1/sqrt(D) so it can be added directly to FlexAttention's already-scaled scores.
        q_with_bias_v = q + bias_v
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # (B, H, T, 2T - 1)
        # rel_shift converts absolute-relative-position columns into (query, key) columns;
        # keep the first T to land in (B, H, T, T) bias space.
        rel_pos_bias = self._rel_shift(matrix_bd)[..., :T] * (D**-0.5)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + rel_pos_bias[b, h, q_idx, kv_idx]

        # Matrix c: fold u @ K^T into FlexAttention by rewriting Q as (Q + u).
        return score_mod, q + bias_u

    def forward(self, x, block_mask=None, pos_emb=None):
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        qkv = self.w_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.qk_norm:
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)

        if self._uses_rope:
            # RoPE rotates Q/K in place; it is orthogonal to FlexAttention's score_mod.
            q, k = self.rope(q, k)

        score_mod = None
        if self._uses_rel_pos:
            score_mod, q = self._build_rel_pos_score_mod(q, pos_emb)

        attn_fn = flex_attention_compiled if q.is_cuda else flex_attention
        out = attn_fn(q, k, v, block_mask=block_mask, score_mod=score_mod)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)

    def _apply_streaming_position(self, q, k, pos_emb, num_cur, num_kv):
        """Positional handling for the cache-aware streaming path — the single extension point.

        Given the current-frame queries ``q`` ``(B, H, num_cur, D)`` and the concatenated
        ``[cache | current]`` keys ``k`` ``(B, H, num_kv, D)``, apply this attention's
        positional scheme and return ``(q, k, score_mod)`` for ``flex_attention``.

        The cache stores raw (position-agnostic) layer inputs, so each scheme applies its own
        transform here over the concatenated span — keeping the cache format scheme-agnostic.
        ``rope`` rotates ``q``/``k`` in place (its rotation depends only on the query/key
        position *difference*, so streaming needs no explicit position offset) and returns
        ``score_mod=None``; ``rel_pos`` leaves ``q``/``k`` in place (folding ``pos_bias_u`` into
        ``q``) and returns a relative-position ``score_mod``.

        Args:
            q (torch.Tensor): Current-chunk queries of shape ``(B, H, num_cur, D)``.
            k (torch.Tensor): Cached and current keys of shape ``(B, H, num_kv, D)``.
            pos_emb (Optional[torch.Tensor]): Positional embeddings for relative attention.
            num_cur (int): Number of query positions in the current chunk.
            num_kv (int): Total number of cached and current key/value positions.

        Returns:
            q (torch.Tensor): Position-adjusted queries.
            k_mod (Optional[torch.Tensor]): Position-adjusted keys, or ``None`` when unchanged.
            score_mod (Optional[Callable]): FlexAttention score modifier, if required.
        """
        if self.self_attention_model == "rel_pos":
            return self._build_streaming_rel_pos_score_mod(q, pos_emb, num_cur, num_kv)
        if self._uses_rope:
            # RoPE.forward handles ``num_kv > num_cur`` directly: keys rotate from position 0,
            # queries from ``num_kv - num_cur`` (= cache length), so the query/key position
            # difference matches the full-sequence forward exactly — streaming is exact.
            q, k = self.rope(q, k)
            return q, k, None
        if self.self_attention_model == "no_pos":
            return q, k, None
        # abs_pos bakes absolute positions into the embeddings before the blocks, which does
        # not compose with a running KV cache without a position offset; deferred.
        raise NotImplementedError(
            f"Cache-aware streaming is not implemented for self_attention_model="
            f"'{self.self_attention_model}'. Supported streaming schemes: 'rel_pos', 'rope', 'no_pos'."
        )

    def _build_streaming_rel_pos_score_mod(self, q, pos_emb, num_cur, num_kv):
        """Transformer-XL relative-position ``score_mod`` for the streaming (cached) attention.

        Same (b)+(d) bias and (c) query-rewrite as :meth:`_build_rel_pos_score_mod`, but for a
        rectangular ``num_cur`` (queries) x ``num_kv`` (keys) block instead of a square one.
        The concatenated ``[cache | current]`` sequence is positionally contiguous, so key
        index ``k`` sits at position ``k`` and current-query ``j`` sits at position
        ``(num_kv - num_cur) + j``; their relative distance maps into the length-``num_kv``
        ``pos_emb`` (``2*num_kv - 1`` entries, index 0 = most-positive rel-pos) at
        ``i = num_cur - 1 - j + k``. The bias is gathered directly (no square ``rel_shift``),
        which keeps it correct for ``num_cur != num_kv`` and easy to generalize.

        Args:
            q (torch.Tensor): Current-chunk queries of shape ``(B, H, num_cur, D)``.
            pos_emb (torch.Tensor): Relative-position embeddings for the full KV span.
            num_cur (int): Number of query positions in the current chunk.
            num_kv (int): Total number of cached and current key/value positions.

        Returns:
            q_with_bias_u (torch.Tensor): Queries with the Transformer-XL content bias applied.
            k_mod (None): ``None`` because relative attention does not transform the keys.
            score_mod (Callable): FlexAttention score modifier containing the relative-position bias.
        """
        H, D = self.n_heads, self.head_dim
        device = q.device
        # pos_emb: (1, 2*num_kv - 1, d_model) -> p: (1, H, 2*num_kv - 1, D)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, H, D).transpose(1, 2)
        bias_u = self.pos_bias_u.view(1, H, 1, D).to(dtype=q.dtype)
        bias_v = self.pos_bias_v.view(1, H, 1, D).to(dtype=q.dtype)
        # (b)+(d): (Q + v) @ R^T over all relative-position entries, then gather per (j, k).
        all_bd = torch.matmul(q + bias_v, p.transpose(-2, -1))  # (B, H, num_cur, 2*num_kv - 1)
        j = torch.arange(num_cur, device=device).view(num_cur, 1)
        k = torch.arange(num_kv, device=device).view(1, num_kv)
        rel_idx = num_cur - 1 - j + k  # (num_cur, num_kv), values in [0, num_cur + num_kv - 2]
        rows = torch.arange(num_cur, device=device).view(num_cur, 1).expand(num_cur, num_kv)
        rel_pos_bias = all_bd[:, :, rows, rel_idx] * (D**-0.5)  # (B, H, num_cur, num_kv)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + rel_pos_bias[b, h, q_idx, kv_idx]

        # Matrix c: fold u @ K^T into FlexAttention by rewriting Q as (Q + u).
        return q + bias_u, None, score_mod

    def forward_streaming(self, kv_normed, num_cur, block_mask, pos_emb):
        """Cache-aware attention: keys/values from the whole ``[cache | current]`` span,
        queries from the current frames only.

        Args:
            kv_normed: ``(B, num_kv, d_model)`` layer-normed ``[cache | current]`` inputs, with
                the current frames as the last ``num_cur`` positions.
            num_cur: number of current-chunk frames (the query count).
            block_mask: FlexAttention ``BlockMask`` over ``(num_cur, num_kv)``.
            pos_emb: relative-position embedding for length ``num_kv`` (``rel_pos`` only).

        Returns:
            ``(B, num_cur, d_model)`` attention output for the current frames.
        """
        B, num_kv, _ = kv_normed.shape
        H, D = self.n_heads, self.head_dim

        # One fused projection over the concatenated span; slice Q to the current frames.
        # ``w_qkv`` is linear, so ``q_all[:, :, -num_cur:]`` equals projecting only the current
        # frames, while K/V cover cache + current.
        qkv = self.w_qkv(kv_normed).view(B, num_kv, 3, H, D).permute(2, 0, 3, 1, 4)
        q_all, k, v = qkv.unbind(0)
        q = q_all[:, :, num_kv - num_cur :, :]

        if self.qk_norm:
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)

        q, k_mod, score_mod = self._apply_streaming_position(q, k, pos_emb, num_cur, num_kv)
        if k_mod is not None:
            k = k_mod

        attn_fn = flex_attention_compiled if q.is_cuda else flex_attention
        out = attn_fn(q, k, v, block_mask=block_mask, score_mod=score_mod)
        out = out.transpose(1, 2).contiguous().view(B, num_cur, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig, pos_enc=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg, pos_enc=pos_enc)
        self.drop = nn.Dropout(cfg.drop_rate)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x, block_mask=None, pos_emb=None):
        x = x + self.drop(self.attn(self.norm1(x), block_mask=block_mask, pos_emb=pos_emb))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

    def forward_streaming(self, x_cur, cache_in, block_mask, pos_emb, cache_size):
        """Cache-aware streaming forward for one chunk.

        Args:
            x_cur: ``(B, C, d_model)`` current-chunk layer input (pre-norm residual stream).
            cache_in: ``(B, L, d_model)`` cached layer inputs from previous chunks (padded to
                ``L``; validity is enforced by ``block_mask``).
            block_mask: FlexAttention ``BlockMask`` over ``(C, L + C)``.
            pos_emb: relative-position embedding for length ``L + C`` (``rel_pos`` only).
            cache_size: rolling-cache size in frames (``>= 0``), or ``< 0`` to grow the cache
                unbounded (full cache). Determines how many layer-input frames to retain.

        Returns:
            ``(x_out, new_cache)`` — the current-chunk output ``(B, C, d_model)`` and this
            layer's updated input cache.
        """
        C = x_cur.shape[1]
        # K/V source is the layer input over [cache | current]; Q is the current frames only.
        # LayerNorm is position-wise, so norm1([cache | cur]) == [norm1(cache) | norm1(cur)].
        kv_in = torch.cat([cache_in, x_cur], dim=1)
        attn_out = self.attn.forward_streaming(self.norm1(kv_in), C, block_mask, pos_emb)
        x = x_cur + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        # Update cache with this layer's *inputs* (kv_in), keeping the last ``cache_size`` frames
        # (rolling) or all of them (full cache when cache_size < 0). ``0`` keeps nothing.
        if cache_size < 0:
            new_cache = kv_in
        elif cache_size == 0:
            new_cache = kv_in[:, :0]
        else:
            new_cache = kv_in[:, -cache_size:]
        return x, new_cache


@experimental
class TransformerEncoder(nn.Module):
    """Pre-norm Transformer encoder for ASR.

    Architecture: PreEncode -> PositionalEncoding -> LayerNorm -> N x TransformerBlock -> FinalNorm

    Uses PyTorch FlexAttention for attention computation. On CUDA, mask functions
    are compiled into fused Triton kernels with block-sparse optimization. On CPU,
    FlexAttention falls back to an unfused implementation automatically.

    Args:
        feat_in: Input feature dimension (number of mel bins).
        d_model: Transformer encoder state dimension, i.e. the size of the residual stream that flows
            through every block (token/frame embedding size, attention input/output size, and
            feed-forward input/output size). Also known as ``hidden_size`` in HuggingFace
            ``transformers`` configs and ``embed_dim``/``d_model`` in PyTorch's
            ``nn.TransformerEncoderLayer``.

        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        feat_out: Output feature dimension. Defaults to ``d_model``.
        subsampling: Subsampling method. Supports ``feature_stacking`` for the
            Transformer-native ``FeatureStacking`` module, plus ``stacking`` and
            ``stacking_norm`` for linear frame stacking.
        subsampling_factor: Subsampling factor for the pre-encoder.
        drop_rate: Dropout probability.
        dropout_pre_encoder: Dropout probability after positional encoding. Defaults to ``drop_rate``.
        dropout_emb: Dropout probability for positional embeddings.
        qkv_bias: If True, add a learnable bias to the fused Q/K/V projection. Many modern ASR/LM
            Transformers (e.g. HuggingFace Whisper) drop the bias on the K projection because a
            constant K bias adds the same scalar to every key and is wiped out by softmax's
            shift-invariance, making it a redundant parameter. Default ``False`` matches that style.
        qk_norm: If True, apply per-head ``LayerNorm`` to Q and K before the dot product. Stabilizes
            training by preventing exponential Q/K-norm growth and "attention entropy collapse"
            (Henry et al. 2020; used in OLMo 2, Gemma 3, Qwen 3). Cheap, ~no-op for inference.
        ff_expansion: Multiplier for the per-block FFN inner hidden size:
            ``ffn_hidden_size = int(ff_expansion * d_model)``. Only widens the intermediate FFN
            projection; FFN input/output stays at ``d_model``. Typical value ``4.0``; ``float``
            allows sub-1x experts for MoE. Equivalent to ``intermediate_size / hidden_size`` in
            HuggingFace and ``dim_feedforward / d_model`` in PyTorch's ``nn.TransformerEncoderLayer``.
        pre_block_norm: If True (default), apply LayerNorm to embeddings before the first
            Transformer block (BERT/ViT-style). Set False to match pre-norm Transformers
            such as Whisper or GPT-2 — required when loading pretrained weights from those
            checkpoints.
        self_attention_model: Type of positional encoding and attention scoring scheme. Mirrors
            the Conformer encoder's ``self_attention_model`` choices, plus a ``"no_pos"`` option:

            - ``"rel_pos"`` (default): Transformer-XL relative positional encoding
              (https://arxiv.org/abs/1901.02860). The relative-position bias is computed in each
              layer and injected into FlexAttention via a ``score_mod`` closure (the (b)+(d)
              terms) plus a ``Q + pos_bias_u`` query rewrite (the (c) term), so the kernel stays
              FlexAttention.
            - ``"abs_pos"``: sinusoidal absolute positional encoding added to the embeddings
              before the first block; standard ``Q @ K^T`` attention via FlexAttention.
            - ``"no_pos"`` (or ``None``): no positional encoding at all — pre-encoder output
              flows straight into ``embed_norm`` and the Transformer blocks. ``xscaling``,
              ``pos_emb_max_len``, ``dropout_pre_encoder`` and ``dropout_emb`` have no effect
              in this mode. ``None`` is accepted as a YAML-friendly alias for ``"no_pos"``
              (an unset field in a config maps to ``None``).
            - ``"rope"``: rotary position embedding applied to Q/K inside attention. No additive
              positional embedding is added to the embeddings; ``dropout_pre_encoder`` (and
              ``xscaling`` if set) are still applied to the pre-encoder output, and ``dropout_emb``
              has no effect in this mode.

            ``"rel_pos_local_attn"`` is not implemented yet.
        rope_base: Theta base for the rotary position embedding. Only used when
            ``self_attention_model='rope'``. Defaults to 10000.0.
        rotary_fraction: Fraction of the per-head dim to rotate. Only used when
            ``self_attention_model='rope'``. Defaults to 1.0.
        pos_emb_max_len: Initial maximum length for sinusoidal positional embeddings.
        xscaling: If True, scale embeddings by ``sqrt(d_model)`` before adding positional encodings,
            following "Attention Is All You Need" article. Originally intended to balance the magnitude
            of small-variance token embeddings against unit-bounded sinusoidal positions and to keep
            tied input/pre-softmax logits well-scaled. With modern unit-variance ``nn.Linear``
            pre-encoders and the LayerNorm directly after the positional sum, this scaling is
            largely a no-op for activation magnitudes. Only meaningful when ``pre_block_norm=False``
            or when matching pretrained checkpoints that expect this scaling.
        attn_mode: Attention pattern — currently only "full" (bidirectional) is supported.
        sync_max_audio_length: When true, sync positional encoding allocation length across distributed ranks.
    """

    def __init__(
        self,
        feat_in: int = 128,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 17,
        feat_out: int = -1,
        subsampling: str = 'feature_stacking',
        subsampling_factor: int = 4,
        drop_rate: float = 0.1,
        dropout_pre_encoder: float = None,
        dropout_emb: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        ff_expansion: float = 4.0,
        pre_block_norm: bool = True,
        self_attention_model: Optional[str] = "rel_pos",
        rope_base: float = 10000.0,
        rotary_fraction: float = 1.0,
        pos_emb_max_len: int = 5000,
        xscaling: bool = False,
        attn_mode: str = "full",
        sync_max_audio_length: bool = True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        if attn_mode not in _SUPPORTED_ATTENTION_MODES:
            raise ValueError(
                f"attn_mode='{attn_mode}' is not yet supported. Supported modes: {_SUPPORTED_ATTENTION_MODES}."
            )
        # ``None`` is accepted as a YAML-friendly alias for ``"no_pos"`` (an unset field in a
        # config simply maps to None) — normalize here so the rest of the module only deals with
        # the string form.
        if self_attention_model is None:
            self_attention_model = "no_pos"
        if self_attention_model not in _SUPPORTED_SELF_ATTENTION_MODELS:
            raise ValueError(
                f"self_attention_model='{self_attention_model}' is not supported. "
                "Currently only 'abs_pos', 'rel_pos', 'rope', and 'no_pos' (or None) are available."
            )
        if dropout_pre_encoder is None:
            dropout_pre_encoder = drop_rate

        cfg = TransformerEncoderConfig(
            feat_in=feat_in,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            ff_expansion=ff_expansion,
            pre_block_norm=pre_block_norm,
            subsampling_factor=subsampling_factor,
            attn_mode=attn_mode,
            self_attention_model=self_attention_model,
            rope_base=rope_base,
            rotary_fraction=rotary_fraction,
        )
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.subsampling = subsampling
        self.subsampling_factor = subsampling_factor
        self.sync_max_audio_length = sync_max_audio_length
        self.self_attention_model = self_attention_model
        self.attn_mode = attn_mode

        if subsampling == 'feature_stacking':
            self.pre_encode = FeatureStacking(subsampling_factor, feat_in, d_model)
        elif subsampling and subsampling_factor > 1:
            if subsampling in ['stacking', 'stacking_norm']:
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    norm=True if subsampling == 'stacking_norm' else False,
                )
            else:
                raise ValueError(
                    f"subsampling='{subsampling}' is not supported. "
                    "Currently only 'feature_stacking', 'stacking', and 'stacking_norm' are available."
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        self._feat_out = d_model
        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None
        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            self.pos_enc = PositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "rope":
            # RoPE has no additive pos-emb step to host dropout, so pre-encoder
            # dropout is applied separately in forward_internal.
            self.dropout_pre_encoder = nn.Dropout(dropout_pre_encoder)
            self.pos_enc = RotaryPositionalEncoding(
                d_k=d_model // n_heads,
                rotary_fraction=rotary_fraction,
                rope_base=rope_base,
                max_len=pos_emb_max_len,
            )
        else:  # "no_pos"
            self.pos_enc = None
        self.embed_norm = nn.LayerNorm(d_model) if pre_block_norm else nn.Identity()
        # For 'rope', the shared RotaryPositionalEncoding is passed into each block so the
        # cos/sin buffers are computed once and reused across all attention modules.
        layer_pos_enc = self.pos_enc if self_attention_model == "rope" else None
        self.layers = nn.ModuleList([TransformerBlock(cfg, pos_enc=layer_pos_enc) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None

        self.set_max_audio_length(self.pos_emb_max_len)

    def forward(self, audio_signal, length, bypass_pre_encode=False):
        """
        Args:
            audio_signal: ``(B, C, T)`` mel spectrogram when ``bypass_pre_encode=False``,
                or ``(B, T, D)`` pre-encoded embeddings when ``bypass_pre_encode=True``.
            length: (B,) — valid frame counts per sample.
            bypass_pre_encode: If true, skip the pre-encoder and consume frame-level embeddings.
                               This option is used when pre-encoded embeddings are used as speaker cache
                               such as speaker diarization models.

        Returns:
            x: (B, D, T') — encoded representation (channels-first).
            length: (B,) — output lengths after subsampling.
        """
        if not bypass_pre_encode and audio_signal.shape[-2] != self._feat_in:
            raise ValueError(
                f"If bypass_pre_encode is False, audio_signal should have shape "
                f"(batch, {self._feat_in}, n_frame) but got last dimension {audio_signal.shape[-2]}."
            )
        if bypass_pre_encode and audio_signal.shape[-1] != self.d_model:
            raise ValueError(
                f"If bypass_pre_encode is True, audio_signal should have shape "
                f"(batch, n_frame, {self.d_model}) but got last dimension {audio_signal.shape[-1]}."
            )

        if bypass_pre_encode:
            self.update_max_seq_length(seq_length=audio_signal.size(1), device=audio_signal.device)
        else:
            self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_internal(audio_signal, length, bypass_pre_encode=bypass_pre_encode)

    def forward_internal(self, audio_signal, length, bypass_pre_encode=False):
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),),
                audio_signal.size(1) if bypass_pre_encode else audio_signal.size(-1),
                dtype=torch.int64,
                device=audio_signal.device,
            )

        if not bypass_pre_encode:
            # Unwrap activation-checkpointing (CheckpointWrapper) and match by name: both
            # the wrapper and duplicate module copies defeat isinstance(FeatureStacking).
            pre_encode_module = getattr(self.pre_encode, "_checkpoint_wrapped_module", self.pre_encode)
            is_feature_stacking = type(pre_encode_module).__name__ == "FeatureStacking"
            if is_feature_stacking:
                # FeatureStacking takes raw (B, C, T) and transposes internally.
                x, length = self.pre_encode(audio_signal, length)
            else:
                x = torch.transpose(audio_signal, 1, 2)
            if isinstance(pre_encode_module, nn.Linear):
                x = self.pre_encode(x)
            elif not is_feature_stacking:
                x, length = self.pre_encode(x=x, lengths=length)
            length = length.to(torch.int64)
        else:
            x = audio_signal
            length = length.to(torch.int64)

        if self.self_attention_model == "rope":
            # RoPE: no pos emb added; just apply xscale (if set) + pre-encoder dropout here.
            if self.xscale:
                x = x * self.xscale
            x = self.dropout_pre_encoder(x)
            pos_emb = None
        elif self.pos_enc is not None:
            x, pos_emb = self.pos_enc(x=x)
        else:  # "no_pos": pre-encoder output flows in unchanged
            pos_emb = None
        x = self.embed_norm(x)

        B, T, _ = x.shape
        mask_mod = self._build_mask_mod(length)
        block_mask = create_block_mask(mask_mod, B=B, H=1, Q_LEN=T, KV_LEN=T, device=x.device)
        # For ``abs_pos`` the positional information is already baked into ``x``, so we don't
        # need to thread ``pos_emb`` through each layer; only ``rel_pos`` consumes it.
        layer_pos_emb = pos_emb if self.self_attention_model == "rel_pos" else None
        for layer in self.layers:
            x = layer(x, block_mask=block_mask, pos_emb=layer_pos_emb)

        x = self.final_norm(x)
        if self.out_proj is not None:
            x = self.out_proj(x)
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        length = length.to(dtype=torch.int64)
        return x, length

    def _build_mask_mod(self, length):
        """Build the FlexAttention ``mask_mod`` shared by every layer in this forward.

        Returns a callable ``(b, h, q_idx, kv_idx) -> bool`` that selects which keys each
        query may attend to. The base encoder supports padding-only masking (``attn_mode
        == "full"``) and additionally causal masking (``attn_mode == "causal"``). Subclasses
        (e.g. :class:`StreamingTransformerEncoder`) override this to inject other attention
        patterns such as a sliding window; the single overridable hook keeps
        ``forward_internal`` agnostic to the masking scheme.

        Args:
            length (torch.Tensor): Valid sequence length for each batch element.

        Returns:
            mask_mod (Callable): FlexAttention mask function combining attention and padding constraints.
        """
        pad_mod = _make_padding_mod(length)
        if self.attn_mode == "causal":
            return and_masks(_make_causal_mod(), pad_mod)
        return pad_mod

    def update_max_seq_length(self, seq_length: int, device):
        """
        Updates the maximum sequence length for positional encodings.

        Args:
            seq_length: New maximum sequence length.
            device: Device to use for computations.
        """
        if self.sync_max_audio_length and torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)
            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def set_max_audio_length(self, max_audio_length):
        """Sets maximum input length and extends positional encodings if needed."""
        self.max_audio_length = max_audio_length
        if self.pos_enc is None:  # "no_pos" mode has no buffer to extend
            return
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)

    def freeze(self) -> None:
        freeze(self)

    def unfreeze(self, partial: bool = False) -> None:
        unfreeze(self, partial=partial)


@experimental
class StreamingTransformerEncoder(TransformerEncoder, StreamingEncoder):
    """``TransformerEncoder`` with bounded-context attention and cache-aware streaming.

    Adds two things on top of the base FlexAttention :class:`TransformerEncoder`:

    1. **Bounded-context attention** in the full-sequence (training) forward, in one of two
       styles selected by ``att_context_style``:

       - ``"sliding_window"`` (default): a per-query ``[left, right]`` window — frame ``t``
         attends to ``[t - left, t + right]``.
       - ``"chunked_limited"``: chunk-aligned attention matching ``ConformerEncoder`` — frames
         are grouped into chunks of ``chunk_size = right + 1`` and a query attends to its whole
         chunk plus the ``left // chunk_size`` preceding chunks.

    2. **Cache-aware streaming** — a rolling KV cache of the last :attr:`cache_size` layer-input
       frames, so chunked inference reproduces the full forward without re-encoding history.
       Implements the real :class:`StreamingEncoder` interface consumed by ``StreamingSTTModel`` /
       ``AudioPerceptionModule`` (``cache_aware_stream_step`` / ``get_initial_cache_state`` /
       ``setup_streaming_params``), reusing the Conformer ``cache_last_channel`` cache plumbing.

    **Look-ahead and exactness.** For ``sliding_window``, streaming is exact only with ``right ==
    0``: a sliding right context compounds across layers (an N-layer stack sees ``N * right``
    frames ahead), which chunk-by-chunk streaming cannot reproduce. ``chunked_limited`` exists
    precisely to lift that restriction — every layer stops at the same chunk boundary, so the
    look-ahead stays bounded by the current chunk no matter how deep the stack, and streaming is
    exact as long as the streaming chunk matches ``chunk_size = right + 1``. Use
    ``att_context_style="chunked_limited"`` with ``att_context_size=[left, chunk_size - 1]`` to get
    in-chunk look-ahead; use the default sliding window with ``[left, 0]`` for a strictly causal
    encoder.

    The cache stores raw (position-agnostic) per-layer inputs; the positional scheme is applied per
    step in :meth:`MultiHeadAttention._apply_streaming_position`, keeping the cache format
    scheme-agnostic. Streaming supports ``rel_pos``, ``rope``, and ``no_pos`` (``abs_pos`` is not
    supported for streaming).

    Args:
        att_context_size: ``[left, right]`` context in encoder (post-subsampling) frames, or a list
            of such pairs to train one model across several chunk sizes. A negative bound removes
            the limit on that side; ``left < 0`` means an unbounded (full) cache that grows with the
            stream (bounded memory only for finite ``left``). Under ``chunked_limited``,
            ``right + 1`` is the chunk size and the left context is quantized down to a whole number
            of chunks. Default ``(-1, -1)`` is full bidirectional attention (no streaming benefit —
            set a finite context to stream).

            Given a list (e.g. ``[[70, 1], [70, 6], [70, 13]]`` for chunk sizes 2/7/14), one context
            is sampled per training batch, matching ``ConformerEncoder``. Evaluation and streaming
            always use ``att_context_size_all[0]`` — or whatever
            :meth:`set_default_att_context_size` pinned — so validation loss and cache-aware
            inference stay deterministic. Pick the inference size with
            ``transcribe_speech.py att_context_size=[70,13]``.
        att_context_probs: Sampling weights over the entries of a multi-context ``att_context_size``.
            Defaults to uniform. Must be the same length as ``att_context_size`` and sum to 1.
        att_context_style: ``"sliding_window"`` (default) or ``"chunked_limited"``. Mirrors the
            ``ConformerEncoder`` argument of the same name so configs read alike across encoders.
        pre_encode_cache_size: Input-feature (mel) frames of look-back prepended to each streaming
            chunk before the pre-encoder; the resulting extra encoder frames are removed via
            ``streaming_cfg.drop_extra_pre_encoded``. Must be a multiple of ``subsampling_factor``
            so chunks stay aligned to the stacking grid. ``FeatureStacking`` itself is
            non-overlapping and needs **none** of this (default ``0``), but a streaming front-end
            that recomputes the mel spectrogram per chunk does: with no look-back the first ~2 mel
            frames of every chunk are built from reflect-padded audio instead of the true preceding
            samples. Set to ``subsampling_factor`` to give the STFT window its context back.
        *args, **kwargs: Forwarded to :class:`TransformerEncoder` (``attn_mode`` is managed
            internally and ignored).
    """

    def __init__(
        self,
        *args,
        att_context_size=(-1, -1),
        att_context_probs=None,
        att_context_style="sliding_window",
        pre_encode_cache_size: int = 0,
        **kwargs,
    ):
        # This encoder's attention pattern is governed by ``att_context_size`` /
        # ``att_context_style``; drop any caller-provided ``attn_mode`` so it never reaches (and
        # fails) the base's supported-mode validation, then run the base with a valid placeholder.
        kwargs.pop("attn_mode", None)
        super().__init__(*args, attn_mode="full", **kwargs)
        if att_context_style not in _SUPPORTED_ATT_CONTEXT_STYLES:
            raise ValueError(
                f"att_context_style='{att_context_style}' is not supported. "
                f"Supported styles: {_SUPPORTED_ATT_CONTEXT_STYLES}."
            )
        self.att_context_style = att_context_style
        if pre_encode_cache_size < 0 or pre_encode_cache_size % self.subsampling_factor != 0:
            raise ValueError(
                f"pre_encode_cache_size must be a non-negative multiple of subsampling_factor="
                f"{self.subsampling_factor} (so chunks stay aligned to the stacking grid), "
                f"but got {pre_encode_cache_size}."
            )
        self.pre_encode_cache_size = pre_encode_cache_size
        self.streaming_cfg = None
        self.att_context_size_all, self.att_context_probs = self._calc_context_sizes(
            att_context_size, att_context_probs
        )
        # Eval/inference default: the first entry, as in ConformerEncoder.
        self.att_context_size = self.att_context_size_all[0]
        self.setup_streaming_params()
        # Purely a label for introspection/logging: ``_build_mask_mod`` below fully
        # overrides the base's mask selection, so ``attn_mode`` is never consulted here.
        self.attn_mode = att_context_style

    def _calc_context_sizes(self, att_context_size, att_context_probs):
        """Normalize ``att_context_size`` to a list of ``[left, right]`` pairs plus sampling probs.

        Accepts a single pair or a list of pairs, mirroring ``ConformerEncoder._calc_context_sizes``
        so the two encoders read the same config.
        """
        sizes = list(att_context_size)
        if not sizes:
            raise ValueError("att_context_size must not be empty.")
        # A bare [left, right] pair -> single-context list.
        if isinstance(sizes[0], (int, float)):
            sizes = [sizes]
        normalized = []
        for i, pair in enumerate(sizes):
            pair = [int(v) for v in pair]
            if len(pair) != 2:
                raise ValueError(f"att_context_size entries must be [left, right] pairs, but entry {i} is {pair}.")
            left, right = pair
            if self.att_context_style == "chunked_limited" and left > 0 and right >= 0 and left % (right + 1):
                # ConformerEncoder rejects this outright. We quantize instead (``cache_size`` reports
                # the true value), because a config like [70, 3] is perfectly usable at 68 frames of
                # left context — but say so, since the number in the config is not what you get.
                chunk = right + 1
                logging.warning(
                    "att_context_size[%d]=%s: left context %d is not a whole multiple of the "
                    "chunked_limited chunk size %d, so only %d frames are visible (%d whole chunks).",
                    i,
                    pair,
                    left,
                    chunk,
                    (left // chunk) * chunk,
                    left // chunk,
                )
            normalized.append(pair)

        if att_context_probs is None:
            probs = [1.0 / len(normalized)] * len(normalized)
        else:
            probs = [float(p) for p in att_context_probs]
            if len(probs) != len(normalized):
                raise ValueError(
                    f"att_context_probs has {len(probs)} entries but att_context_size has "
                    f"{len(normalized)}; they must match."
                )
            if not math.isclose(sum(probs), 1.0, rel_tol=1e-6):
                raise ValueError(f"att_context_probs must sum to 1, but sums to {sum(probs)}.")
        return normalized, probs

    def _sample_att_context_size(self):
        """The ``[left, right]`` context for this forward.

        With several contexts configured, training samples one per batch so a single model learns
        every chunk size; everything else (validation, export, cache-aware streaming) uses the
        pinned :attr:`att_context_size` so it stays deterministic. Mirrors the sampling in
        ``ConformerEncoder.forward_internal``.
        """
        if self.training and len(self.att_context_size_all) > 1:
            return random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        return self.att_context_size

    @property
    def cache_size(self) -> int:
        """Rolling attention-cache size in encoder frames (``< 0`` -> unbounded/full cache).

        For ``sliding_window`` this is ``left`` — the furthest back any query may look. For
        ``chunked_limited`` the visible history is a whole number of chunks, so the cache only
        needs ``(left // chunk_size) * chunk_size`` frames; sizing it exactly is what keeps the
        streaming mask a plain "last N cache frames" test with no per-query term.
        """
        left, right = self.att_context_size
        if left < 0:
            return -1
        if self.att_context_style == "chunked_limited":
            if right < 0:
                # Unlimited right degenerates to a left-only sliding window (Conformer parity).
                return left
            chunk = right + 1
            return (left // chunk) * chunk
        return left

    def set_default_att_context_size(self, att_context_size) -> None:
        """Pin the ``[left, right]`` attention context (in encoder frames) and refresh streaming cfg.

        A negative bound means unlimited context on that side. Named to mirror
        ``ConformerEncoder.set_default_att_context_size`` so streaming inference code that calls this
        method works uniformly across encoder types. This is how you select one of a multi-context
        model's chunk sizes at inference.

        Note this pins the *eval and streaming* context only. Training keeps sampling over
        ``att_context_size_all`` — as ``ConformerEncoder`` does — so pinning a context does not
        narrow what a training run sees. To train at a single context, configure a single pair.
        """
        att_context_size = [int(v) for v in att_context_size]
        if len(att_context_size) != 2:
            raise ValueError(f"att_context_size must be a [left, right] pair, but got {att_context_size}.")
        if att_context_size not in self.att_context_size_all:
            logging.warning(
                "att_context_size=%s is not among the configured contexts %s; using it anyway, but "
                "the model was not trained at this setting.",
                att_context_size,
                self.att_context_size_all,
            )
        self.att_context_size = att_context_size
        self.setup_streaming_params()

    def _build_mask_mod(self, length):
        # Called exactly once per full-sequence forward, so sampling here gives one context per
        # batch. The cache-aware path never comes through here — it reads ``att_context_size``
        # directly — so streaming is unaffected by multi-context training.
        left, right = self._sample_att_context_size()
        pad_mod = _make_padding_mod(length)
        if self.att_context_style == "chunked_limited" and right >= 0:
            return and_masks(_make_chunked_limited_mod(left, right), pad_mod)
        # Sliding window, and the ``chunked_limited`` + unlimited-right degenerate case which
        # ConformerEncoder also collapses to a left-only bound.
        if left < 0 and right < 0:
            # Unbounded on both sides -> full (padding-only) attention.
            return pad_mod
        return and_masks(_make_sliding_window_mod(left, right), pad_mod)

    # ------------------------------------------------------------------ #
    # Cache-aware streaming interface (StreamingEncoder)
    # ------------------------------------------------------------------ #
    def setup_streaming_params(
        self,
        chunk_size: int = None,
        shift_size: int = None,
        left_chunks: int = None,
        att_context_size: list = None,
        max_context: int = 10000,
    ) -> None:
        """(Re)build ``streaming_cfg`` from the current attention context.

        Populates the full :class:`CacheAwareStreamingConfig` — the same dataclass
        ``ConformerEncoder`` uses — so the shared cache-aware tooling
        (``speech_to_text_cache_aware_streaming_infer.py``, ``CacheAwareStreamingAudioBuffer``,
        ``ASRModuleMixin.conformer_stream_step``) can drive this encoder unchanged. Sizes follow the
        Conformer's units: ``chunk_size`` / ``shift_size`` / ``pre_encode_cache_size`` are in input
        (mel) frames, everything else in encoder frames.

        Args:
            chunk_size: Encoder frames per streaming step. Defaults to the chunk implied by the
                attention context — ``right + 1`` — which is the true chunk size under
                ``chunked_limited`` and, under a causal ``sliding_window`` (``right == 0``), the
                minimum-latency single-frame step. A causal sliding window streams exactly at *any*
                chunk size, so overriding this is free there; under ``chunked_limited`` it must
                equal ``right + 1`` or streaming stops matching the offline forward.
            shift_size: Encoder frames the buffer advances per step. Defaults to ``chunk_size``.
                Only equal values are supported — overlapping chunks would recompute frames this
                encoder already emits exactly.
            left_chunks: Number of left chunks visible to each chunk. When given, it retunes the
                active evaluation/streaming attention window (``left = left_chunks * chunk_size``)
                rather than only the cache, so the mask and the cache cannot disagree. Configured
                training contexts in ``att_context_size_all`` are left unchanged.
            att_context_size: Optional ``[left, right]`` to apply before computing the config.
            max_context: Cache size to use when the left context is unbounded (``left < 0``).
        """
        if att_context_size is not None:
            if len(att_context_size) != 2:
                raise ValueError(f"att_context_size must be a [left, right] pair, but got {att_context_size}.")
            self.att_context_size = list(att_context_size)

        # Encoder frames per step. ``right + 1`` is the chunk under chunked_limited and the
        # minimum-latency step under a causal sliding window (right == 0 -> 1 frame).
        implied_chunk = self.att_context_size[1] + 1 if self.att_context_size[1] >= 0 else 1
        chunk = implied_chunk if chunk_size is None else int(chunk_size)
        if chunk < 1:
            raise ValueError(f"chunk_size must be >= 1 encoder frames, but got {chunk}.")
        if self.att_context_style == "chunked_limited" and chunk != implied_chunk:
            raise ValueError(
                f"chunk_size={chunk} must equal the chunked_limited attention chunk "
                f"(right + 1 = {implied_chunk}) so streaming matches the offline forward."
            )
        shift = chunk if shift_size is None else int(shift_size)
        if shift != chunk:
            raise ValueError(
                f"Overlapping/striding chunks are not supported: shift_size ({shift}) must equal "
                f"chunk_size ({chunk}). Every output frame of this encoder already equals the "
                f"offline forward, so there is nothing for an overlap to recompute."
            )

        if left_chunks is not None:
            # Retune the active window itself, not just the cache — otherwise the mask and rolling
            # cache would describe different amounts of history. Copy first because the initial
            # active context aliases att_context_size_all[0], which remains the training contract.
            self.att_context_size = list(self.att_context_size)
            self.att_context_size[0] = int(left_chunks) * chunk

        cache_size = self.cache_size
        S = self.subsampling_factor
        P = self.pre_encode_cache_size
        # ``[first_step, later_steps]`` when there is look-back at all: the first chunk has no
        # history to look back at, and ``CacheAwareStreamingAudioBuffer`` only special-cases step 0
        # when this field is a list. Emitting a scalar would make it prepend P zero frames at step 0
        # while ``calc_drop_extra_pre_encoded`` returns 0 for that step — P/S phantom output frames.
        # Stay a plain scalar when P == 0 so the default path is unchanged.
        self.streaming_cfg = CacheAwareStreamingConfig(
            chunk_size=S * chunk,
            shift_size=S * shift,
            # No look-ahead survives past a chunk boundary in either style, so nothing needs to be
            # dropped off the end of the cache to compensate for it.
            cache_drop_size=0,
            last_channel_cache_size=cache_size if cache_size >= 0 else max_context,
            # Every emitted frame already equals the offline forward (that is the design guarantee
            # validated by test_streaming_matches_full_forward), so all outputs of a step are valid.
            valid_out_len=shift,
            pre_encode_cache_size=[0, P] if P > 0 else 0,
            drop_extra_pre_encoded=P // S,
            last_channel_num=self.n_layers,
            last_time_num=0,  # convolution-free encoder — no time cache
        )

    def get_initial_cache_state(self, batch_size=1, dtype=None, device=None, max_dim=0):
        """Empty KV cache. ``cache_last_channel`` is ``(n_layers, B, cache_size, d_model)`` of
        per-layer inputs (padded to :attr:`cache_size` for a rolling cache; length ``0`` — grows —
        for a full cache or a zero-size one). ``cache_last_time`` is an unused zero-width
        placeholder (no conv)."""
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if device is None:
            device = next(self.parameters()).device
        cache_size = max(self.cache_size, 0)
        cache_last_channel = torch.zeros(
            self.n_layers, batch_size, cache_size, self.d_model, dtype=dtype, device=device
        )
        # Zero *width*, but the same RANK as ``ConformerEncoder``'s conv cache
        # ``(n_layers, B, d_model, conv_cache)`` -- the shared cache-aware tooling slices this
        # tensor positionally (e.g. ``StreamingContextManager.get_context`` does
        # ``cache_last_time[:, slot_ids, :, :]``), so a 3-D placeholder raises
        # "IndexError: too many indices" even though the encoder never reads it.
        cache_last_time = torch.zeros(self.n_layers, batch_size, self.d_model, 0, dtype=dtype, device=device)
        cache_last_channel_len = torch.zeros(batch_size, dtype=torch.int64, device=device)
        return cache_last_channel, cache_last_time, cache_last_channel_len

    def cache_aware_stream_step(
        self,
        processed_signal,
        processed_signal_length=None,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        keep_all_outputs=True,
        drop_extra_pre_encoded=None,
        bypass_pre_encode=False,
    ):
        """Encode one chunk with the rolling KV cache. Returns the Conformer-compatible 5-tuple
        ``(encoded, encoded_len, cache_last_channel, cache_last_time, cache_last_channel_len)``.

        Args:
            keep_all_outputs: Accepted for interface parity, but this encoder has nothing to drop.
                The Conformer trims to ``streaming_cfg.valid_out_len`` because under a *sliding*
                right context its trailing frames are computed without their full look-ahead and get
                recomputed on the next step. Neither of this encoder's masks has that property —
                every emitted frame already equals the offline forward — so all outputs are valid
                and are always returned.
            drop_extra_pre_encoded: Encoder frames to drop from the front of the pre-encoder output,
                removing the contribution of the ``pre_encode_cache_size`` look-back frames
                prepended to this chunk. ``None`` uses ``streaming_cfg.drop_extra_pre_encoded``;
                pass ``0`` explicitly for the first step, whose buffer has no real look-back to
                strip (this is what ``calc_drop_extra_pre_encoded`` in
                ``speech_to_text_cache_aware_streaming_infer.py`` does).
        """
        encoded, encoded_len, new_cache_last_channel, new_cache_last_channel_len = self._streaming_step(
            processed_signal,
            processed_signal_length,
            cache_last_channel,
            cache_last_channel_len,
            bypass_pre_encode,
            drop_extra_pre_encoded,
        )
        # This (convolution-free) encoder has no time cache; pass ``cache_last_time`` through as-is.
        return encoded, encoded_len, new_cache_last_channel, cache_last_time, new_cache_last_channel_len

    def _streaming_step(
        self,
        audio_signal,
        length,
        cache_last_channel,
        cache_last_channel_len,
        bypass_pre_encode,
        drop_extra_pre_encoded=None,
    ):
        # 1. Pre-encode (subsample) the current chunk. FeatureStacking is non-overlapping, so a
        #    chunk aligned to ``subsampling_factor`` subsamples independently and matches the full
        #    forward exactly. Any ``pre_encode_cache_size`` look-back frames the caller prepended
        #    are stripped back off here, after subsampling.
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),),
                audio_signal.size(1) if bypass_pre_encode else audio_signal.size(-1),
                dtype=torch.int64,
                device=audio_signal.device,
            )
        if not bypass_pre_encode:
            if isinstance(self.pre_encode, FeatureStacking):
                x, length = self.pre_encode(audio_signal, length)
            else:
                x = torch.transpose(audio_signal, 1, 2)
            if isinstance(self.pre_encode, nn.Linear):
                x = self.pre_encode(x)
            elif not isinstance(self.pre_encode, FeatureStacking):
                x, length = self.pre_encode(x=x, lengths=length)
        else:
            x = audio_signal
        length = length.to(torch.int64)

        if drop_extra_pre_encoded is None:
            # Only default to the configured drop when we are actually mid-stream. A caller with no
            # cache is passing a whole utterance, not a chunk with look-back prepended — e.g.
            # ``conformer_stream_step(processed_signal=<whole audio>)``, which
            # ``speech_to_text_cache_aware_streaming_infer.py`` uses for ``compare_vs_offline``.
            # Dropping there would silently truncate the offline reference. Mirrors the
            # ``cache_last_channel is not None`` guard in ``ConformerEncoder.forward_internal``.
            drop_extra_pre_encoded = self.streaming_cfg.drop_extra_pre_encoded if cache_last_channel is not None else 0
        if drop_extra_pre_encoded > 0:
            x = x[:, drop_extra_pre_encoded:, :]
            length = (length - drop_extra_pre_encoded).clamp(min=0)

        B, C, _ = x.shape
        if cache_last_channel is None:
            cache_last_channel, _, cache_last_channel_len = self.get_initial_cache_state(
                batch_size=B, dtype=x.dtype, device=x.device
            )
        right = self.att_context_size[1]
        cache_size = self.cache_size
        if self.att_context_style == "chunked_limited" and right >= 0 and C > right + 1:
            raise ValueError(
                f"chunked_limited streaming needs the streaming chunk to fit inside one attention "
                f"chunk, but got {C} encoder frames per step with chunk_size=right+1={right + 1}. "
                f"Feed {right + 1} frames per step (a shorter final chunk is fine)."
            )
        L = cache_last_channel.shape[2]  # padded cache size (== cache_size for rolling; grows for full)
        num_kv = L + C

        # 2. Positional prep (once). Extend positional buffers to cover the [cache | current]
        #    span, then apply this scheme's pre-block step (mirroring ``forward_internal``):
        #    rope rotates Q/K inside attention (only xscale + dropout here); rel_pos needs an
        #    additive pos_emb of length ``num_kv``; abs_pos bakes positions into ``x`` (and is
        #    rejected downstream for streaming); no_pos does nothing.
        if self.pos_enc is not None:
            self.update_max_seq_length(seq_length=num_kv, device=x.device)
        if self.self_attention_model == "rope":
            if self.xscale:
                x = x * self.xscale
            x = self.dropout_pre_encoder(x)
            pos_emb = None
        elif self.pos_enc is not None:
            x, _ = self.pos_enc(x=x)
            pos_emb = self._streaming_pos_emb(num_kv, x) if self.self_attention_model == "rel_pos" else None
        else:  # no_pos
            pos_emb = None
        x = self.embed_norm(x)

        # 3. Streaming layers with the shared block mask; collect each layer's updated cache.
        block_mask = self._build_streaming_block_mask(cache_last_channel_len, length, L, C, x.device)
        new_caches = []
        for i, layer in enumerate(self.layers):
            x, new_cache_l = layer.forward_streaming(x, cache_last_channel[i], block_mask, pos_emb, cache_size)
            new_caches.append(new_cache_l)
        new_cache_last_channel = torch.stack(new_caches, dim=0)

        x = self.final_norm(x)
        if self.out_proj is not None:
            x = self.out_proj(x)
        x = x.transpose(1, 2)  # (B, C, D) -> (B, D, C)

        new_cache_len = cache_last_channel_len + length
        if cache_size >= 0:
            new_cache_len = torch.clamp(new_cache_len, max=cache_size)
        return x, length, new_cache_last_channel, new_cache_len.to(torch.int64)

    def _streaming_pos_emb(self, num_kv, ref):
        """Relative-position embedding ``(1, 2*num_kv - 1, d_model)`` for the concatenated
        ``[cache | current]`` span. ``RelPositionalEncoding.forward`` returns ``(scaled_x, pos_emb)``;
        feed a zero dummy of length ``num_kv`` and keep only ``pos_emb`` (deterministic in eval)."""
        dummy = ref.new_zeros(1, num_kv, self.d_model)
        _, pos_emb = self.pos_enc(x=dummy)
        return pos_emb

    def _build_streaming_block_mask(self, cache_valid_len, cur_len, L, C, device):
        """FlexAttention block mask over ``(C, L + C)`` for the cache-aware step.

        The ``[cache | current]`` sequence is positionally contiguous: key index ``k`` sits at
        position ``k`` and current-query ``j`` at position ``L + j``. Valid keys are the last
        ``cache_valid_len`` cache frames (right-aligned in ``[0, L)``) plus the first ``cur_len``
        current frames.

        For ``sliding_window`` that set is further intersected with the ``[left, right]`` window
        around each query. For ``chunked_limited`` no per-query term is needed: the whole streaming
        chunk is one attention chunk, and the cache is sized to exactly the visible history
        (:attr:`cache_size` == ``left_chunks * chunk_size``), so "everything valid" *is* the mask.
        """
        left, right = self.att_context_size
        chunked = self.att_context_style == "chunked_limited" and right >= 0
        num_kv = L + C

        def mask_mod(b, h, q_idx, kv_idx):
            ok = (kv_idx >= L - cache_valid_len[b]) & (kv_idx < L + cur_len[b])
            if chunked:
                return ok
            if right >= 0:
                ok = ok & (kv_idx <= q_idx + L + right)
            if left >= 0:
                ok = ok & (kv_idx >= q_idx + L - left)
            return ok

        return create_block_mask(mask_mod, B=cache_valid_len.shape[0], H=1, Q_LEN=C, KV_LEN=num_kv, device=device)
