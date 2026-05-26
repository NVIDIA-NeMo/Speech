# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fused Local Transformer module for single-call TRT export.

Wraps the full 16-codebook autoregressive sampling loop as an nn.Module
so torch.onnx.export can trace and unroll it into a flat ONNX graph.
The CategoricalSamplingPlugin handles stochastic multinomial sampling
inside the TRT engine without leaving the GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# ONNX symbolic for CategoricalSamplingPlugin
# ---------------------------------------------------------------------------

class CategoricalSamplingFn(torch.autograd.Function):
    """Bridge between PyTorch and the TRT CategoricalSamplingPlugin.

    In eager mode, falls back to torch.multinomial so tests can run
    without the plugin loaded.  During ONNX export, symbolic() emits
    the 'CategoricalSampling' node that matches the plugin's op_type.
    """

    @staticmethod
    def forward(ctx, probs: Tensor) -> Tensor:
        # Eager fallback: multinomial sampling (used in tests, not in TRT).
        # probs: (B, topk) — already softmaxed probabilities.
        return torch.multinomial(probs, num_samples=1).squeeze(1).to(torch.int32)

    @staticmethod
    def symbolic(g, probs):
        output = g.op("CategoricalSampling", probs)
        output.setType(probs.type().with_dtype(torch.int32).with_sizes([None]))
        return output

def categorical_sampling(probs: Tensor) -> Tensor:
    """Sample one token index per batch item from a probability distribution.

    Args:
        probs: (B, topk) softmax probabilities over the top-k logits.

    Returns:
        (B,) int32 index within [0, topk).
    """
    return CategoricalSamplingFn.apply(probs)

# ---------------------------------------------------------------------------
# Fused autoregressive loop module
# ---------------------------------------------------------------------------

class LocalTransformerFusedModule(nn.Module):
    """Full sample_autoregressive loop as a single traceable nn.Module.

    Exports cleanly to ONNX: the Python for-loop over n_codebooks is
    statically unrolled at trace time.  Per-codebook weights are stored
    as separate named sub-modules so they appear as constants in the graph.

    Scope: production path only (no CFG, no EOS masking, temperature > 0).

    Args:
        lt_helper: Object with the LT sub-modules (LocalTransformerHelper or
                   a compatible namespace for testing).
        temperature: Sampling temperature (baked into the engine at export).
        topk: Top-k filtering width (baked into the engine at export).
    """

    def __init__(self, lt_helper, temperature: float = 0.7, topk: int = 80):
        super().__init__()
        self.temperature = temperature
        self.topk = topk
        # support both real LocalTransformerHelper and test namespaces
        if hasattr(lt_helper, 'num_audio_codebooks'):
            self.n_codebooks = lt_helper.num_audio_codebooks * lt_helper.frame_stacking_factor
        else:
            self.n_codebooks = lt_helper.n_codebooks

        # Store sub-modules so they are tracked by nn.Module and appear as
        # constants in the exported ONNX graph.
        self.in_proj = lt_helper.local_transformer_in_projection
        self.transformer = lt_helper.local_transformer
        self.audio_out_proj = lt_helper.local_transformer_audio_out_projection
        self.out_projections = lt_helper.local_transformer_out_projections
        self.audio_embeddings = lt_helper.audio_embeddings
        self.audio_in_proj = lt_helper.audio_in_projection

    def forward(self, dec_output: Tensor, gumbel_noise: Tensor) -> Tensor:
        """Run the full n_codebooks autoregressive loop.

        Args:
            dec_output: Backbone hidden state, shape (B, H).
            gumbel_noise: Pre-generated Gumbel(0,1) noise per codebook,
                shape (n_codebooks, B, topk). Passing noise as input
                (instead of using the CategoricalSamplingPlugin's
                clock-seeded curand) makes sampling deterministic:
                identical (dec_output, gumbel_noise) inputs always
                produce identical outputs. The caller generates this
                noise from torch's RNG (seeded per request) so eager
                and TRT paths sample the same tokens.

        Returns:
            Sampled tokens, shape (B, n_codebooks), dtype int32.

        Sampling semantics: Gumbel-max is equivalent to multinomial
        sampling from softmax(logits/T): adding Gumbel(0,1) noise to
        log-probabilities and taking argmax gives the same distribution.
        Since softmax is monotonic, we can equivalently add noise to the
        scaled logits (logits/T) and skip the softmax+log step.
        """
        self.transformer.reset_cache(use_cache=False)  # clear any stale KV state
        # (B, H) -> (B, 1, H) -> (B, 1, D)
        x = self.in_proj(dec_output.unsqueeze(1))

        all_tokens = []

        for cb in range(self.n_codebooks):
            # Build causal mask: all-ones over current sequence length T.
            mask = torch.ones(x.size(0), x.size(1), device=x.device, dtype=x.dtype)

            # Local transformer forward: (B, T, D) -> (B, T, D)
            lt_out = self.transformer(x, mask)['output']

            # Project last position: (B, D) -> (B, vocab)
            last = self.audio_out_proj(lt_out[:, -1, :])
            logits = self.out_projections[cb](last)

            # Top-k: keep only top-k logits, drop the rest.
            topk_logits, topk_indices = torch.topk(logits, self.topk, dim=-1)

            # Gumbel-max sampling: argmax(logits/T + gumbel).
            # Cast gumbel to logits' dtype so fp16-built engines don't
            # need a Cast op around the add (TRT can fuse cleanly).
            scaled = topk_logits / self.temperature
            noisy = scaled + gumbel_noise[cb].to(scaled.dtype)
            sampled_pos = noisy.argmax(dim=-1)  # (B,)

            # Map back to real vocab index via gather.
            # topk_indices: (B, topk), sampled_pos: (B,) -> token: (B,)
            token = torch.gather(
                topk_indices, 1, sampled_pos.to(torch.int64).unsqueeze(1)
            ).squeeze(1).to(torch.int32)

            all_tokens.append(token.unsqueeze(1))  # (B, 1)

            # Embed predicted token and append to LT input sequence.
            emb = self.audio_embeddings[cb](token.to(torch.int64))   # (B, emb_dim)
            emb = self.audio_in_proj(emb.unsqueeze(1))               # (B, 1, D)
            emb = self.in_proj(emb)                                   # (B, 1, D)
            x = torch.cat([x, emb], dim=1)                           # (B, T+1, D)

        return torch.cat(all_tokens, dim=1)  # (B, n_codebooks)
