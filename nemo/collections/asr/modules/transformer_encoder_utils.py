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

from functools import lru_cache

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


_flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


def _get_flex_attention(x):
    return _flex_attention_compiled if x.is_cuda else flex_attention


def _causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _apply_packed_rope(rope, q, k, position_ids):
    cos = rope.cos.index_select(0, position_ids).unsqueeze(1).to(q.dtype)
    sin = rope.sin.index_select(0, position_ids).unsqueeze(1).to(q.dtype)
    return rope._apply_rotary(q, cos, sin), rope._apply_rotary(k, cos.to(k.dtype), sin.to(k.dtype))


def _packed_flex_attention_reference(
    attn,
    q,
    k,
    v,
    *,
    lengths,
    pos_emb,
    padded_length,
    causal,
    sequence_offsets,
    use_math_reference=False,
):
    if sequence_offsets is None:
        sequence_offsets = tuple(torch.cat([lengths.new_zeros(1), lengths.cumsum(0)]).tolist())
    outputs = []
    attn_fn = _get_flex_attention(q)
    for offset, end in zip(sequence_offsets[:-1], sequence_offsets[1:]):
        length = end - offset
        if length == 0:
            continue
        qi = q[offset:end].transpose(0, 1).unsqueeze(0)
        ki = k[offset:end].transpose(0, 1).unsqueeze(0)
        vi = v[offset:end].transpose(0, 1).unsqueeze(0)
        score_mod = None
        if attn._uses_rel_pos:
            if pos_emb is None or padded_length is None:
                raise ValueError("Packed relative-position attention requires max-length positional metadata.")
            pos_i = pos_emb[:, padded_length - length : padded_length + length - 1]
            score_mod, qi = attn._build_rel_pos_score_mod(qi, pos_i)
        if use_math_reference:
            out = _packed_math_attention_reference(qi, ki, vi, causal=causal, score_mod=score_mod)
        else:
            block_mask = None
            if causal:
                block_mask = create_block_mask(_causal_mask, B=1, H=1, Q_LEN=length, KV_LEN=length, device=q.device)
            out = attn_fn(qi, ki, vi, block_mask=block_mask, score_mod=score_mod)
        outputs.append(out.squeeze(0).transpose(0, 1))
    if not outputs:
        # Keep every attention branch in the autograd graph even when a rank owns
        # no valid tokens. FSDP/DDP otherwise observes missing gradients for q/k
        # (and relative-position parameters), which can break collectives when
        # another rank in the same step has non-empty input.
        anchor = q.sum() + k.sum()
        if attn._uses_rel_pos:
            anchor = anchor + 0.0 * (attn.pos_bias_u.sum() + attn.pos_bias_v.sum())
            anchor = anchor + sum(0.0 * parameter.sum() for parameter in attn.linear_pos.parameters())
        return v + anchor.to(v.dtype)
    return torch.cat(outputs, dim=0)


def _packed_math_attention_reference(q, k, v, *, causal, score_mod):
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if score_mod is not None:
        scores = scores + score_mod._relative_position_bias
    if causal:
        causal_mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    return torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v)


def _select_flash_attention_varlen(x, *, static_eligible):
    """Return the varlen provider after cheap per-input checks and cached device probing."""
    if not static_eligible or not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16) or x.shape[0] == 0:
        return None
    return _get_flash_attention_varlen_for_device(x.device)


def _can_use_flash_attention_varlen(q):
    static_eligible = q.shape[-1] <= 256 and q.shape[-1] % 8 == 0
    return _select_flash_attention_varlen(q, static_eligible=static_eligible) is not None


def _can_use_flash_attention_varlen_layout(x, head_dim):
    static_eligible = head_dim <= 256 and head_dim % 8 == 0
    return _select_flash_attention_varlen(x, static_eligible=static_eligible) is not None


@lru_cache(maxsize=None)
def _get_flash_attention_varlen_for_device(device):
    if torch.version.cuda is None or torch.cuda.get_device_capability(device)[0] < 8:
        return None
    return _get_flash_attention_varlen()


@lru_cache(maxsize=1)
def _get_flash_attention_varlen():
    try:
        from flash_attn import flash_attn_varlen_func
    except (ImportError, ModuleNotFoundError):
        flash_forward = getattr(torch.ops.aten, "_flash_attention_forward", None)
        if flash_forward is None:
            return None

        def torch_flash_attention_varlen(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            *,
            dropout_p,
            softmax_scale,
            causal,
        ):
            return flash_forward(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                causal,
                False,
                scale=softmax_scale,
            )[0]

        torch_flash_attention_varlen._sequence_packed_provider = "aten"
        return torch_flash_attention_varlen
    return flash_attn_varlen_func


def _forward_sequence_packed_layer(layer, x, **kwargs):
    """Preserve packed execution through PyTorch's activation-checkpoint wrapper."""
    wrapped = getattr(layer, '_checkpoint_wrapped_module', None)
    if wrapped is None:
        return layer._forward_sequence_packed(x, **kwargs)
    packed_forward = getattr(wrapped, '_forward_sequence_packed', None)
    checkpoint_fn = getattr(layer, 'checkpoint_fn', None)
    if packed_forward is None or checkpoint_fn is None:
        raise TypeError(
            f"Activation-checkpoint wrapper around {type(wrapped).__name__} cannot execute sequence-packed layers."
        )
    return checkpoint_fn(packed_forward, x, **kwargs)
