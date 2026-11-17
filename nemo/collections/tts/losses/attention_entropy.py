# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory-bounded entropy regularization for causal, right-padded attention."""

from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint


def attention_entropy(query, key, valid_mask, num_heads, num_key_value_heads, chunk_size=32):
    """Mean H(p)/log(N) over heads and valid queries with at least two keys.

    Inputs are the raw Q/K projection outputs [batch, time, heads * head_dim].
    ``valid_mask`` is [batch, time], True for real tokens. All causal attention
    rows are included; no query or key sampling is used. Recompute each chunk's
    scores during backward rather than retaining quadratic attention matrices.
    This helper is for full-sequence training, without a KV cache or window mask.
    """
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    batch, time, _ = query.shape
    head_dim = query.size(-1) // num_heads
    groups = num_heads // num_key_value_heads
    query = query.reshape(batch, time, num_key_value_heads, groups, head_dim).permute(0, 2, 3, 1, 4)
    key = key.reshape(batch, time, num_key_value_heads, head_dim).permute(0, 2, 1, 3)
    valid_mask = valid_mask.to(device=query.device, dtype=torch.bool)
    eligible = valid_mask & (valid_mask.long().cumsum(-1) > 1)
    denominator = (eligible.sum() * num_heads).clamp_min(1)
    total = query.new_zeros((), dtype=torch.float32)
    for start in range(0, time, chunk_size):
        args = (query[..., start : start + chunk_size, :], key, valid_mask, start)
        if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad):
            total = total + checkpoint(_entropy_chunk_sum, *args, use_reentrant=False)
        else:
            total = total + _entropy_chunk_sum(*args)
    return total / denominator


def mean_attention_entropy(projections, valid_mask):
    """Give each captured attention-layer invocation equal weight."""
    if not projections:
        return torch.zeros((), device=valid_mask.device, dtype=torch.float32)
    return torch.stack(
        [attention_entropy(q, k, valid_mask, heads, kv_heads) for q, k, heads, kv_heads in projections]
    ).mean()


@contextmanager
def capture_attention_projections(attention_modules, enabled=True):
    """Capture Q/K outputs without changing fused attention or checkpoint graphs.

    Hooks only retain existing tensors. Compute entropy after leaving the context,
    outside any decoder activation-checkpoint region. Hooks are removed even on
    failure, and are absent during checkpoint recomputation and inference.
    """
    projections = []
    pending = {}
    handles = []
    try:
        if enabled:
            for attention in attention_modules:

                def capture_query(module, inputs, output, attention=attention):
                    pending[id(attention)] = output

                def capture_key(module, inputs, output, attention=attention):
                    projections.append(
                        (
                            pending.pop(id(attention)),
                            output,
                            attention.num_heads,
                            attention.num_key_value_heads,
                        )
                    )

                handles.append(attention.q_proj.register_forward_hook(capture_query))
                handles.append(attention.k_proj.register_forward_hook(capture_key))
        yield projections
    finally:
        for handle in handles:
            handle.remove()


def _entropy_chunk_sum(query, key, valid_mask, start):
    # Explicitly disable autocast: softmax/log probabilities and QK scores use FP32.
    with torch.autocast(device_type=query.device.type, enabled=False):
        scores = torch.einsum('bhgqd,bhkd->bhgqk', query.float(), key.float()) * query.size(-1) ** -0.5
        positions = torch.arange(start, start + query.size(-2), device=query.device)
        allowed = valid_mask[:, None, :] & (
            torch.arange(key.size(-2), device=query.device)[None, None, :] <= positions[None, :, None]
        )
        num_keys = allowed.sum(-1)
        allowed = allowed[:, None, None]
        scores = scores.masked_fill(~allowed, -torch.inf)
        # Empty/padded rows must stay finite through softmax and backward.
        scores = torch.where((num_keys > 0)[:, None, None, :, None], scores, 0.0)
        log_probs = scores.log_softmax(-1)
        entropy = -(log_probs.exp() * log_probs.masked_fill(~allowed, 0.0)).sum(-1)
        eligible = valid_mask[:, start : start + query.size(-2)] & (num_keys > 1)
        normalized = entropy / num_keys.clamp_min(2).float().log()[:, None, None]
        return (normalized * eligible[:, None, None]).sum()
