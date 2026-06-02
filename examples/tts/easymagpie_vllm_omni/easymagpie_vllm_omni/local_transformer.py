# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Autoregressive local transformer for EasyMagpieTTS on vLLM-Omni.

EasyMagpieTTS predicts the ``C * S`` stacked audio codebooks of one frame
*autoregressively* with a small causal transformer conditioned on the backbone's
per-frame hidden state. This module implements that local transformer so it can
run as a single compiled CUDA graph:

* :class:`EasyMagpieLocalTransformer` is a causal transformer stack with
  learnable positional embeddings, using ``scaled_dot_product_attention`` and no
  KV cache. It is decorated with ``@support_torch_compile`` so vLLM captures one
  CUDA graph for the fixed ``(num_tokens, num_stacked_codebooks, hidden)`` input
  shape. Its layer/weight layout matches the training checkpoint so weights load
  1:1.
* :class:`EasyMagpieCodePredictor` owns the persistent, address-stable scratch
  buffers and runs the per-frame autoregressive loop, re-invoking the compiled
  transformer once per codebook over the **same** buffer (replaying one
  fixed-shape graph N times is faster and simpler than capturing N separate
  graphs).

All sampling is CUDA-graph safe (Gumbel-max + ``topk`` + ``masked_fill`` only;
no host syncs, no ``multinomial`` on possibly-degenerate warmup data).
"""
from __future__ import annotations

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig

from easymagpie_vllm_omni.config import EasyMagpieOmniArch


def _gumbel_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Gumbel-max categorical draw — CUDA-graph safe.

    Equivalent to sampling from ``softmax(logits)`` but uses only
    ``uniform_`` + ``log`` + ``argmax`` (all legal inside a captured graph)
    and degrades gracefully on degenerate warmup logits instead of triggering
    a device-side assert the way ``multinomial`` does.
    """
    u = torch.empty_like(logits).uniform_(1e-20, 1.0 - 1e-20)
    return (logits - torch.log(-torch.log(u))).argmax(dim=-1)


def sample_codebook(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    forbidden_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Sample one codebook's tokens from logits (CUDA-graph safe).

    Args:
        logits: ``[num_tokens, vocab]`` raw codebook logits.
        temperature: Sampling temperature; ``<= 0`` falls back to argmax.
        top_k: Top-k truncation width (``<= 0`` disables truncation).
        forbidden_mask: Optional ``[vocab]`` bool mask; ``True`` entries are
            set to ``-inf`` before sampling (reserved/special tokens).

    Returns:
        ``[num_tokens]`` int64 sampled token ids.
    """
    if forbidden_mask is not None:
        logits = logits.masked_fill(forbidden_mask, float("-inf"))

    if temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        vals, idxs = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        sampled_in_k = _gumbel_argmax(vals)
        return idxs.gather(-1, sampled_in_k.unsqueeze(-1)).squeeze(-1)

    return _gumbel_argmax(logits)


class EasyMagpieLTSelfAttention(nn.Module):
    """Causal self-attention.

    Fused QKV projection (``qkv_net``) and output projection (``o_net``), both
    bias-free, with ``d_head ** -0.5`` scaling computed via
    ``scaled_dot_product_attention`` with ``is_causal=True``. No KV cache: the
    autoregressive loop re-runs the full (short, fixed-length) sequence each
    step, which is what makes the whole thing CUDA-graph capturable.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5
        self.qkv_net = nn.Linear(d_model, 3 * n_heads * self.d_head, bias=False)
        self.o_net = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv_net(x).reshape(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each [b, t, nh, dh]
        # [b, nh, t, dh]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.scale
        )
        attn = attn.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_net(attn)


class EasyMagpieLTFeedForward(nn.Module):
    """Positionwise feed-forward network.

    Uses ``Conv1d(kernel_size=1)`` layers named ``proj.conv`` and ``o_net.conv``
    (no bias). A kernel-1 conv is a plain linear over the channel dim, applied
    with a single transpose and GELU(tanh) in between. The ``Conv1d`` submodule
    names match the training checkpoint so weights load 1:1.
    """

    def __init__(self, d_model: int, d_ffn: int) -> None:
        super().__init__()
        self.proj = _Conv1dWrapper(d_model, d_ffn)
        self.o_net = _Conv1dWrapper(d_ffn, d_model)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, t, c] -> conv expects [b, c, t]
        h = x.transpose(1, 2)
        h = self.act(self.proj(h))
        h = self.o_net(h)
        return h.transpose(1, 2)


class _Conv1dWrapper(nn.Module):
    """Holds a kernel-1 ``Conv1d`` under attribute name ``conv`` (no bias)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EasyMagpieLTLayer(nn.Module):
    """One pre-norm transformer layer (self-attn + FFN), bias-free LayerNorms.

    Residual structure: ``x = x + attn(norm_self(x))`` then
    ``x = x + ff(norm_pos_ff(x))``.
    """

    def __init__(self, d_model: int, d_ffn: int, n_heads: int) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model, bias=False)
        self.self_attention = EasyMagpieLTSelfAttention(d_model, n_heads)
        self.norm_pos_ff = nn.LayerNorm(d_model, bias=False)
        self.pos_ff = EasyMagpieLTFeedForward(d_model, d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.norm_self(x))
        x = x + self.pos_ff(self.norm_pos_ff(x))
        return x


# NOTE: ``dynamic_arg_dims`` is passed explicitly rather than relying on
# vLLM's annotation-based inference. This file uses
# ``from __future__ import annotations`` (PEP 563), so ``forward``'s
# annotations are stored as strings (``"torch.Tensor"``) and vLLM's
# ``v.annotation in [torch.Tensor, ...]`` check would never match, raising
# "No dynamic dimensions found...". ``inputs_embeds`` is
# ``[num_tokens, num_codebooks, hidden]`` -> dim 0 (num_tokens) is dynamic.
@support_torch_compile(dynamic_arg_dims={"inputs_embeds": 0})
class EasyMagpieLocalTransformer(nn.Module):
    """Compiled causal transformer stack with learnable positional embeddings.

    Decorated with ``@support_torch_compile`` so vLLM captures a single CUDA
    graph for the fixed ``(num_tokens, num_stacked_codebooks, d_model)`` input
    shape. Holds learnable ``position_embeddings``, the stacked ``layers.{i}.*``
    and a no-op ``norm_out``.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        arch = EasyMagpieOmniArch.from_hf_config(vllm_config.model_config.hf_config)
        d_model = arch.local_transformer_hidden_dim
        n_heads = arch.local_transformer_n_heads
        n_layers = arch.local_transformer_n_layers
        d_ffn = d_model * 4
        # +2 of head-room over ``num_stacked_codebooks`` for the positional table.
        max_len = arch.num_stacked_codebooks + 2

        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(
            [EasyMagpieLTLayer(d_model, d_ffn, n_heads) for _ in range(n_layers)]
        )
        self.norm_out = nn.Identity()

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        seq_len = inputs_embeds.shape[1]
        positions = torch.arange(seq_len, device=inputs_embeds.device)
        x = inputs_embeds + self.position_embeddings(positions).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.norm_out(x)


class EasyMagpieCodePredictor(nn.Module):
    """Autoregressive intra-frame codebook predictor (the "local transformer").

    Given the backbone's per-frame hidden state, predicts all ``C * S`` stacked
    audio codebooks one at a time. Owns the codebook input embeddings (shared
    with the outer model for building decode-step input embeddings) and all the
    projection heads, plus the persistent scratch buffers required for
    CUDA-graph replay.

    Per frame (``generate_codes``):

    1. Position 0 of the input buffer holds ``in_proj(dec_hidden)``.
    2. For codebook ``k`` in ``0 .. N-1``: run the compiled transformer over the
       whole buffer, read row ``k`` of the output, project to codebook-``k``
       logits, sample, and (if ``k < N-1``) write ``in_proj(audio_emb_k(code))``
       into buffer row ``k + 1``.

    The buffer is zeroed once per frame and filled incrementally; because the
    transformer is causal, rows ``> k`` never influence row ``k``, so replaying
    the same fixed-shape graph N times yields the correct autoregressive result.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        arch = EasyMagpieOmniArch.from_hf_config(vllm_config.model_config.hf_config)
        self.arch = arch
        self.num_codebooks = arch.num_stacked_codebooks
        self.num_tokens_per_codebook = arch.num_all_tokens_per_codebook
        self.audio_embedding_dim = arch.audio_embedding_dim
        self.embedding_dim = arch.embedding_dim
        lt_hidden = arch.local_transformer_hidden_dim

        # Per-codebook audio token embeddings (shared with the outer model's
        # decode-step input-embedding assembly).
        self.audio_embeddings = nn.ModuleList(
            [nn.Embedding(self.num_tokens_per_codebook, self.audio_embedding_dim) for _ in range(self.num_codebooks)]
        )
        # audio_embedding_dim -> embedding_dim (Identity when equal).
        if self.audio_embedding_dim != self.embedding_dim:
            self.audio_in_projection = nn.Linear(self.audio_embedding_dim, self.embedding_dim)
        else:
            self.audio_in_projection = nn.Identity()

        # embedding_dim (== backbone hidden) -> local-transformer hidden.
        if lt_hidden != self.embedding_dim:
            self.local_transformer_in_projection = nn.Linear(self.embedding_dim, lt_hidden)
        else:
            self.local_transformer_in_projection = nn.Identity()

        self.local_transformer = EasyMagpieLocalTransformer(
            vllm_config=vllm_config, prefix=f"{prefix}.local_transformer"
        )

        # local-transformer hidden -> audio_embedding_dim (Identity when equal).
        if self.audio_embedding_dim != lt_hidden:
            self.local_transformer_audio_out_projection = nn.Linear(lt_hidden, self.audio_embedding_dim)
        else:
            self.local_transformer_audio_out_projection = nn.Identity()

        # Per-codebook output heads.
        self.local_transformer_out_projections = nn.ModuleList(
            [nn.Linear(self.audio_embedding_dim, self.num_tokens_per_codebook) for _ in range(self.num_codebooks)]
        )

        # Forbidden-token mask (reserved/special tokens, EOS kept reachable).
        # Populated by :meth:`init_forbidden_mask` once arch ids are known.
        self.register_buffer(
            "forbidden_mask",
            torch.zeros(self.num_tokens_per_codebook, dtype=torch.bool),
            persistent=False,
        )

        # Sampling knobs (overridable from the outer model / request).
        self.temperature: float = 0.7
        self.top_k: int = 80

        # ── Persistent address-stable scratch buffers ──────────────────
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        dtype = vllm_config.model_config.dtype
        self._buf_inputs = torch.zeros(max_num_tokens, self.num_codebooks, lt_hidden, dtype=dtype)
        self._out_codes = torch.zeros(max_num_tokens, self.num_codebooks, dtype=torch.long)

    @torch.no_grad()
    def init_forbidden_mask(self) -> None:
        """Forbid all trailing special tokens except audio EOS.

        Everything in the special-token block above ``codebook_size`` is blocked
        at sampling time, except ``audio_eos`` which must remain reachable to
        terminate.
        """
        mask = torch.zeros(self.num_tokens_per_codebook, dtype=torch.bool, device=self.forbidden_mask.device)
        mask[self.arch.codebook_size :] = True
        eos = self.arch.audio_eos_id
        if 0 <= eos < self.num_tokens_per_codebook:
            mask[eos] = False
        self.forbidden_mask.copy_(mask)

    def embed_codebook(self, codebook_idx: int, codes: torch.Tensor) -> torch.Tensor:
        """Embed a single codebook's tokens (``[num_tokens] -> [num_tokens, audio_dim]``)."""
        return self.audio_embeddings[codebook_idx](codes)

    def embed_audio_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed a full frame of stacked codes into the backbone embedding space.

        Averages the per-codebook embeddings then applies ``audio_in_projection``.
        Used by the outer model to build the decode input embedding from the
        previous frame's codes.

        Args:
            codes: ``[num_tokens, num_codebooks]`` int64 codes.

        Returns:
            ``[num_tokens, embedding_dim]`` float embedding.
        """
        acc = self.audio_embeddings[0](codes[:, 0])
        for c in range(1, self.num_codebooks):
            acc = acc + self.audio_embeddings[c](codes[:, c])
        acc = acc / self.num_codebooks
        return self.audio_in_projection(acc)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Run the compiled local transformer over the input buffer."""
        return self.local_transformer(inputs_embeds)

    @torch.no_grad()
    def generate_codes(self, dec_hidden: torch.Tensor) -> torch.Tensor:
        """Autoregressively sample all ``C * S`` codebooks for each frame.

        Args:
            dec_hidden: ``[num_tokens, hidden]`` backbone hidden state (one row
                per frame being decoded).

        Returns:
            ``[num_tokens, num_codebooks]`` int64 sampled codes.
        """
        num_tokens = dec_hidden.shape[0]
        buf = self._buf_inputs[:num_tokens]
        out = self._out_codes[:num_tokens]
        buf.zero_()

        # Row 0: projected backbone hidden state (the AR "prompt").
        buf[:, 0, :] = self.local_transformer_in_projection(dec_hidden)

        # Always pass the mask unconditionally. An all-False mask makes
        # ``masked_fill`` a no-op, so there's no need to guard with
        # ``forbidden_mask.any()`` — and that guard is a data-dependent
        # host sync that is illegal during CUDA-graph capture.
        forbidden = self.forbidden_mask
        for k in range(self.num_codebooks):
            hidden = self(buf)  # compiled transformer over the fixed buffer
            row = self.local_transformer_audio_out_projection(hidden[:, k, :])
            logits = self.local_transformer_out_projections[k](row)
            code_k = sample_codebook(
                logits,
                temperature=self.temperature,
                top_k=self.top_k,
                forbidden_mask=forbidden,
            )
            out[:, k] = code_k
            if k + 1 < self.num_codebooks:
                emb = self.audio_in_projection(self.audio_embeddings[k](code_k))
                buf[:, k + 1, :] = self.local_transformer_in_projection(emb)

        return out[:num_tokens]
