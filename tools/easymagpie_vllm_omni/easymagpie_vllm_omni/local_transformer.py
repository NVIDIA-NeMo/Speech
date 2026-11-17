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
"""Autoregressive intra-frame codebook predictor for EasyMagpieTTS."""
from __future__ import annotations

import os

import torch
from easymagpie_vllm_omni.cfg import CFGOptions, guided_logits
from easymagpie_vllm_omni.config import EasyMagpieOmniArch
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig

# Default top-k width for audio-codebook sampling. Because ``torch.topk``'s ``k``
# shapes tensors inside the captured graph, this becomes a capture-time constant.
_DEFAULT_TOP_K = 80

# A positive floor avoids data-dependent branches in the captured graph.
_MIN_SAMPLING_TEMPERATURE = 1e-4


class EasyMagpieLTSelfAttention(nn.Module):
    """Bias-free causal self-attention without a KV cache."""

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
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        attn = attn.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_net(attn)


class EasyMagpieLTFeedForward(nn.Module):
    """Positionwise feed-forward network.

    ``conv`` parameter names preserve checkpoint compatibility, while the
    kernel-1 operations use equivalent linear projections on ``[B, T, C]``.
    """

    def __init__(self, d_model: int, d_ffn: int) -> None:
        super().__init__()
        self.proj = _LinearWrapper(d_model, d_ffn)
        self.o_net = _LinearWrapper(d_ffn, d_model)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.register_buffer("_positions", torch.arange(max_len), persistent=False)
        self.layers = nn.ModuleList([EasyMagpieLTLayer(d_model, d_ffn, n_heads) for _ in range(n_layers)])
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
class EasyMagpieSanitizedCodeLoop(nn.Module):
    """Compiled per-frame codebook loop.

    Gumbel noise is supplied at runtime for fresh samples. Temperature is
    dynamic, while ``top_k`` is fixed when the graph is captured.
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

    def bind_predictor(self, predictor: "EasyMagpieLocalTransformerPredictor") -> None:
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
            gumbel_noise: ``[num_tokens, num_codebooks, vocabulary_size]``
                pre-drawn Gumbel noise, indexed by vocabulary ID.
            temperature: ``[1]`` temperature; nonpositive values use argmax.

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
            code_k = _sample_codebook_logits(logits, gumbel_noise[:, k, :], temperature, forbidden, self.top_k)
            codes.append(code_k)
            if k + 1 < n:
                emb = cp.audio_in_projection(cp.audio_embeddings[k](code_k))
                buf[:, k + 1, :] = cp.local_transformer_in_projection(emb)
        return torch.stack(codes, dim=1)


@support_torch_compile(dynamic_arg_dims={"conditional": 0, "unconditional": 0, "gumbel_noise": 0, "scales": 0})
class EasyMagpieSanitizedCFGCodeLoop(nn.Module):
    """Batched CFG sampling without diagnostic reductions in the compiled graph."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self._predictor_ref: tuple = ()

    def bind_predictor(self, predictor: 'EasyMagpieLocalTransformerPredictor') -> None:
        self._predictor_ref = (predictor,)

    def forward(self, conditional, unconditional, gumbel_noise, temperature, scales):
        return _generate_cfg_codes(
            self._predictor_ref[0], conditional, unconditional, gumbel_noise, temperature, scales, trace=False
        )


@support_torch_compile(dynamic_arg_dims={"conditional": 0, "unconditional": 0, "gumbel_noise": 0, "scales": 0})
class EasyMagpieSanitizedCFGTraceCodeLoop(nn.Module):
    """Diagnostic sampling with a separate AOT identity from the production loop."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self._predictor_ref: tuple = ()

    def bind_predictor(self, predictor: 'EasyMagpieLocalTransformerPredictor') -> None:
        self._predictor_ref = (predictor,)

    def forward(self, conditional, unconditional, gumbel_noise, temperature, scales):
        return _generate_cfg_codes(
            self._predictor_ref[0], conditional, unconditional, gumbel_noise, temperature, scales, trace=True
        )


class EasyMagpieLocalTransformerPredictor(nn.Module):
    """Predict all stacked audio codebooks from each backbone hidden state."""

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
        self._code_loop = EasyMagpieSanitizedCodeLoop(vllm_config=vllm_config, prefix=f"{prefix}.code_loop")
        self._code_loop.bind_predictor(self)

        # ── Persistent address-stable scratch buffers ──────────────────
        # (created on the CUDA default device that vLLM sets during model init).
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        dtype = vllm_config.model_config.dtype
        # Stable-address input for the captured loop graph.
        self._dec_hidden_buf = torch.zeros(max_num_tokens, self.embedding_dim, dtype=dtype)
        # Gumbel noise drawn eagerly each frame and injected into the graph; fp32
        # so the small ``-log(-log(u))`` values don't underflow in fp16.
        # Native top-k retains all ties at the cutoff. Noise is indexed over the
        # vocabulary so those ties remain eligible, including after clipping.
        self._gumbel_buf = torch.zeros(
            max_num_tokens, self.num_codebooks, self.num_tokens_per_codebook, dtype=torch.float32
        )
        self._temperature_buf = torch.zeros(1, dtype=torch.float32)
        self._cfg_unconditional_buf = torch.zeros(max_num_tokens, self.embedding_dim, dtype=dtype)
        self._cfg_scales_buf = torch.zeros(max_num_tokens, 1, dtype=torch.float32)
        # AOT keys include forward's qualified name, not mutable instance flags.
        # Independently compiled classes prevent trace/non-trace cache collisions.
        trace = bool(os.environ.get('EASYMAGPIE_CFG_TRACE_PATH'))
        cfg_loop = EasyMagpieSanitizedCFGTraceCodeLoop if trace else EasyMagpieSanitizedCFGCodeLoop
        with set_model_tag('local_transformer_cfg_trace' if trace else 'local_transformer_cfg'):
            self._cfg_code_loop = cfg_loop(vllm_config=vllm_config, prefix=f"{prefix}.cfg_code_loop")
        self._cfg_code_loop.bind_predictor(self)
        self._cfg_loop_forward = (self._cfg_code_loop,)
        mode = getattr(vllm_config.compilation_config, 'cudagraph_mode', None)
        if (
            CFGOptions.from_hf_config(vllm_config.model_config.hf_config).enabled
            and mode is not None
            and mode.has_full_cudagraphs()
        ):
            from vllm.compilation.cuda_graph import CUDAGraphWrapper
            from vllm.config import CUDAGraphMode

            self._cfg_loop_forward = (
                CUDAGraphWrapper(self._cfg_code_loop, vllm_config, runtime_mode=CUDAGraphMode.FULL),
            )

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

        self._temperature_buf.fill_(float(self.temperature))
        return self._code_loop(in_buf, noise, self._temperature_buf)

    @torch.no_grad()
    def generate_codes_cfg(
        self,
        conditional: torch.Tensor,
        unconditional: torch.Tensor,
        scales: torch.Tensor,
        *,
        padded_batch: int | None = None,
    ):
        """Run paired CFG without duplicating weights or sampling two audio streams."""
        batch = conditional.shape[0]
        padded_batch = batch if padded_batch is None else padded_batch
        if padded_batch < batch:
            raise ValueError('CFG graph batch cannot be smaller than the real logical batch')
        conditional_buf = self._dec_hidden_buf[:padded_batch]
        unconditional_buf = self._cfg_unconditional_buf[:padded_batch]
        conditional_buf[:batch].copy_(conditional)
        unconditional_buf[:batch].copy_(unconditional)
        scale_buf = self._cfg_scales_buf[:padded_batch]
        scale_buf[:batch].copy_(scales.reshape(batch, 1))
        if padded_batch > batch:
            conditional_buf[batch:].zero_()
            unconditional_buf[batch:].zero_()
            scale_buf[batch:] = 1.0
        noise = self._gumbel_buf[:padded_batch]
        noise.uniform_(1e-20, 1.0 - 1e-20)
        noise.log_().neg_().log_().neg_()
        self._temperature_buf.fill_(float(self.temperature))
        codes, diagnostics = self._cfg_loop_forward[0](
            conditional_buf, unconditional_buf, noise, self._temperature_buf, scale_buf
        )
        return codes[:batch], diagnostics[:batch]


def _generate_cfg_codes(cp, conditional, unconditional, gumbel_noise, temperature, scales, *, trace):
    batch = conditional.shape[0]
    n = cp.num_codebooks
    buf = conditional.new_zeros(2 * batch, n, cp.lt_hidden)
    buf[:, 0, :] = cp.local_transformer_in_projection(torch.cat((conditional, unconditional), dim=0))
    codes = []
    diagnostics = conditional.new_zeros(batch, 3, dtype=torch.float32)
    for k in range(n):
        # The expensive transformer and projection run ONCE over both branches.
        hidden = cp.local_transformer(buf)
        row = cp.local_transformer_audio_out_projection(hidden[:, k, :])
        logits = cp.local_transformer_out_projections[k](row)
        cond, uncond = logits[:batch], logits[batch:]
        guided = guided_logits(cond, uncond, scales)
        if trace and k == 0:
            diagnostics = torch.stack(
                (
                    (cond.float() - uncond.float()).abs().amax(-1),
                    (guided - cond.float()).abs().amax(-1),
                    (guided.argmax(-1) != cond.argmax(-1)).float(),
                ),
                dim=-1,
            )
        # The shared sampler sanitizes the mixed logits before special-token
        # masking, matching native NeMo without creating masked-token NaNs.
        code = _sample_codebook_logits(guided, gumbel_noise[:, k, :], temperature, cp.forbidden_mask, cp._sample_top_k)
        codes.append(code)
        if k + 1 < n:
            embedding = cp.local_transformer_in_projection(cp.audio_in_projection(cp.audio_embeddings[k](code)))
            # Both branches consume the SAME guided sample, never independently
            # sampled audio histories (the native NeMo CFG contract).
            buf[:batch, k + 1, :] = embedding
            buf[batch:, k + 1, :] = embedding
    return torch.stack(codes, dim=1), diagnostics


def _sample_codebook_logits(logits, gumbel_noise, temperature, forbidden_mask, top_k):
    """Native NeMo sampling policy, with graph-safe Gumbel sampling.

    Sanitize before masking, then retain every token tied at the top-k cutoff.
    All operations stay on device and compile into the existing codebook loop;
    CFG still runs both branches in one transformer forward per codebook.
    """
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
    logits = logits.masked_fill(forbidden_mask, float('-inf'))
    cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1:]
    logits = logits.masked_fill(logits < cutoff, float('-inf'))
    sampled = logits / temperature.clamp_min(_MIN_SAMPLING_TEMPERATURE) + gumbel_noise
    scores = torch.where(temperature <= 0, logits, sampled)
    return scores.argmax(dim=-1)


# Preserve imports while giving the corrected graphs distinct vLLM AOT keys.
# Qualified forward names are part of vLLM's compilation-cache identity.
EasyMagpieCodeLoop = EasyMagpieSanitizedCodeLoop
EasyMagpieCFGCodeLoop = EasyMagpieSanitizedCFGCodeLoop
EasyMagpieCFGTraceCodeLoop = EasyMagpieSanitizedCFGTraceCodeLoop
