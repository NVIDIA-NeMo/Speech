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


# Default top-k width for audio-codebook sampling. Because ``torch.topk``'s ``k``
# shapes tensors inside the captured graph, this becomes a capture-time constant.
_DEFAULT_TOP_K = 80

# Minimum sampling temperature used inside the compiled graph. The old eager
# sampler special-cased ``temperature <= 0`` as exact argmax, but a
# data-dependent branch is illegal inside a captured graph, so we clamp to a tiny
# value (near-argmax) and always take the Gumbel-top-k path.
_MIN_SAMPLING_TEMPERATURE = 1e-4


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

    A ``Conv1d(kernel_size=1)`` over the channel dim is mathematically identical
    to an ``nn.Linear`` applied on the last dim, but the conv form forces a
    ``[b, t, c] -> [b, c, t]`` transpose on the way in and out (which torch
    cannot fuse away and which showed up as ``*_transpose_*`` /
    ``*_convolution_*`` triton kernels in profiling). We therefore use plain
    bias-free ``nn.Linear`` layers and operate directly on the ``[b, t, c]``
    layout. The ``conv`` submodule attribute is kept so the kernel-1 conv
    weights from the training checkpoint (shape ``[out, in, 1]``) still map 1:1;
    :meth:`EasyMagpieTTS.load_weights` squeezes the trailing singleton dim.
    """

    def __init__(self, d_model: int, d_ffn: int) -> None:
        super().__init__()
        self.proj = _LinearWrapper(d_model, d_ffn)
        self.o_net = _LinearWrapper(d_ffn, d_model)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, t, c]; no transpose needed for a kernel-1 conv == linear.
        return self.o_net(self.act(self.proj(x)))


class _LinearWrapper(nn.Module):
    """Holds a bias-free ``nn.Linear`` under attribute name ``conv``.

    The attribute is named ``conv`` purely so the parameter path matches the
    training checkpoint's kernel-1 ``Conv1d`` (``...proj.conv.weight``); the math
    is a plain dense projection on the channel dim.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Linear(in_ch, out_ch, bias=False)

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


class EasyMagpieLocalTransformer(nn.Module):
    """Causal transformer stack with learnable positional embeddings.

    Plain (uncompiled) module: it is invoked from inside
    :class:`EasyMagpieCodeLoop`'s compiled forward, so it gets *inlined* into
    that single captured graph rather than being compiled / replayed on its own.
    Holds learnable ``position_embeddings``, the stacked ``layers.{i}.*`` and a
    no-op ``norm_out`` (names match the training checkpoint).
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
        # Cache a constant ``arange`` so we don't re-materialize it (and re-run an
        # embedding gather) on every autoregressive step. The positional table is
        # tiny and fixed; gathering once per forward over a cached index avoids
        # the ``arange + embedding`` triton kernel seen in profiling.
        self.register_buffer("_positions", torch.arange(max_len), persistent=False)
        self.layers = nn.ModuleList(
            [EasyMagpieLTLayer(d_model, d_ffn, n_heads) for _ in range(n_layers)]
        )
        self.norm_out = nn.Identity()

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        seq_len = inputs_embeds.shape[1]
        pos_emb = self.position_embeddings(self._positions[:seq_len])
        x = inputs_embeds + pos_emb.unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.norm_out(x)


# NOTE: ``dynamic_arg_dims`` is passed explicitly rather than relying on vLLM's
# annotation-based inference. This file uses ``from __future__ import
# annotations`` (PEP 563), so ``forward``'s annotations are stored as strings
# (``"torch.Tensor"``) and vLLM's ``v.annotation in [torch.Tensor, ...]`` check
# would never match, raising "No dynamic dimensions found...". Both ``dec_hidden``
# and ``gumbel_noise`` are ``[num_tokens, ...]`` -> dim 0 (num_tokens) is dynamic.
@support_torch_compile(dynamic_arg_dims={"dec_hidden": 0, "gumbel_noise": 0})
class EasyMagpieCodeLoop(nn.Module):
    """Compiled single-graph autoregressive codebook loop.

    Runs the *entire* per-frame loop — transformer stack, per-codebook projection
    heads, and (graph-safe) sampling — under one ``@support_torch_compile`` graph,
    so vLLM captures a single CUDA graph replayed once per frame instead of
    replaying the transformer ``N`` times with eager projection / sampling in
    between. (Total FLOPs are unchanged — this removes per-step Python and
    kernel-launch overhead, which dominates at low concurrency.)

    It owns **no parameters**: the projection / embedding / out-projection modules
    and the forbidden mask live on the parent :class:`EasyMagpieCodePredictor` (so
    the checkpoint still loads 1:1) and are reached through a non-registered
    reference set by :meth:`bind_predictor`.

    Sampling is kept graph-safe by construction:

    * the Gumbel noise is drawn eagerly *outside* the graph and injected as
      ``gumbel_noise`` — running ``uniform_`` inside the capture would otherwise
      reuse the captured random numbers on every replay;
    * ``temperature`` is a runtime tensor, so per-request temperature works
      without recompiling;
    * ``top_k`` shapes the ``topk`` / noise tensors and is therefore a
      **capture-time constant** (per-request ``top_k`` changes are not honored
      once the graph is captured).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        arch = EasyMagpieOmniArch.from_hf_config(vllm_config.model_config.hf_config)
        self.num_codebooks = arch.num_stacked_codebooks
        self.lt_hidden = arch.local_transformer_hidden_dim
        self.top_k = min(_DEFAULT_TOP_K, arch.num_all_tokens_per_codebook)
        # Set by :meth:`bind_predictor`; held in a tuple so nn.Module does not
        # register the parent as a submodule (which would duplicate params).
        self._predictor_ref: tuple = ()

    def bind_predictor(self, predictor: "EasyMagpieCodePredictor") -> None:
        self._predictor_ref = (predictor,)
        self.top_k = predictor._sample_top_k

    def forward(
        self,
        dec_hidden: torch.Tensor,
        gumbel_noise: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Sample all ``num_codebooks`` codes for every frame in one graph.

        Args:
            dec_hidden: ``[num_tokens, embedding_dim]`` backbone hidden state.
            gumbel_noise: ``[num_tokens, num_codebooks, top_k]`` pre-drawn
                Gumbel noise (``-log(-log(u))``), one slice per codebook.
            temperature: ``[1]`` sampling temperature (already clamped > 0).

        Returns:
            ``[num_tokens, num_codebooks]`` int64 sampled codes.
        """
        cp = self._predictor_ref[0]
        num_tokens = dec_hidden.shape[0]
        n = self.num_codebooks

        buf = dec_hidden.new_zeros(num_tokens, n, self.lt_hidden)
        buf[:, 0, :] = cp.local_transformer_in_projection(dec_hidden)

        forbidden = cp.forbidden_mask
        codes: list[torch.Tensor] = []
        for k in range(n):
            hidden = cp.local_transformer(buf)
            row = cp.local_transformer_audio_out_projection(hidden[:, k, :])
            logits = cp.local_transformer_out_projections[k](row)
            logits = logits.masked_fill(forbidden, float("-inf")) / temperature
            vals, idxs = torch.topk(logits, self.top_k, dim=-1)
            picked = (vals + gumbel_noise[:, k, :]).argmax(dim=-1, keepdim=True)
            code_k = idxs.gather(-1, picked).squeeze(-1)
            codes.append(code_k)
            if k + 1 < n:
                emb = cp.audio_in_projection(cp.audio_embeddings[k](code_k))
                buf[:, k + 1, :] = cp.local_transformer_in_projection(emb)
        return torch.stack(codes, dim=1)


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

        # Sampling knobs (overridable from the outer model / request). ``top_k``
        # is captured into the compiled loop graph (see ``EasyMagpieCodeLoop``),
        # so per-request ``top_k`` changes are not honored once captured;
        # per-request ``temperature`` is, since it is fed as a runtime tensor.
        self.temperature: float = 0.7
        self.top_k: int = _DEFAULT_TOP_K
        self.lt_hidden = lt_hidden
        self._sample_top_k = min(self.top_k, self.num_tokens_per_codebook)

        # Compiled single-graph autoregressive loop (owns no params; reaches the
        # projection heads / embeddings / mask on ``self`` via a bound reference).
        self._code_loop = EasyMagpieCodeLoop(vllm_config=vllm_config, prefix=f"{prefix}.code_loop")
        self._code_loop.bind_predictor(self)

        # ── Persistent address-stable scratch buffers ──────────────────
        # (created on the CUDA default device that vLLM sets during model init).
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        dtype = vllm_config.model_config.dtype
        # Stable-address input for the captured loop graph.
        self._dec_hidden_buf = torch.zeros(max_num_tokens, self.embedding_dim, dtype=dtype)
        # Gumbel noise drawn eagerly each frame and injected into the graph; fp32
        # so the small ``-log(-log(u))`` values don't underflow in fp16.
        self._gumbel_buf = torch.zeros(max_num_tokens, self.num_codebooks, self._sample_top_k, dtype=torch.float32)
        self._temperature_buf = torch.zeros(1, dtype=torch.float32)

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

    @torch.no_grad()
    def generate_codes(self, dec_hidden: torch.Tensor) -> torch.Tensor:
        """Autoregressively sample all ``C * S`` codebooks for each frame.

        Draws this frame's Gumbel noise eagerly into a stable buffer (fresh
        randomness per frame, outside the captured graph) and stages the inputs
        at fixed addresses, then runs the whole loop as a single captured graph
        via :class:`EasyMagpieCodeLoop`.

        Args:
            dec_hidden: ``[num_tokens, hidden]`` backbone hidden state (one row
                per frame being decoded).

        Returns:
            ``[num_tokens, num_codebooks]`` int64 sampled codes.
        """
        num_tokens = dec_hidden.shape[0]
        in_buf = self._dec_hidden_buf[:num_tokens]
        in_buf.copy_(dec_hidden)

        # ``-log(-log(u))`` Gumbel noise, computed in place in fp32.
        noise = self._gumbel_buf[:num_tokens]
        noise.uniform_(1e-20, 1.0 - 1e-20)
        noise.log_().neg_().log_().neg_()

        self._temperature_buf.fill_(max(float(self.temperature), _MIN_SAMPLING_TEMPERATURE))
        return self._code_loop(in_buf, noise, self._temperature_buf)
