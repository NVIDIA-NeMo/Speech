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
"""Numerics of Transformer Engine's packed (THD) causal attention when it dispatches to flash-attn-4.

Packed SALM training on Blackwell requires ``NVTE_FUSED_ATTN=0``, so TE routes its THD ``padding_causal`` attention
to flash-attn-4 (flash_attn.cute). TE releases before 2.21 passed the causal mask to flash-attn-4 as
``window_size=(-1, 0)``, which flash-attn-4 reads as a local window that admits no keys: the output and dQ/dK/dV are
all zero while training loss still decreases. This guards the TE pin in pyproject.toml.
"""
import importlib.util

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 10,
        reason="TE dispatches THD attention to flash-attn-4 only on Blackwell (sm_100+) in this setup",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("transformer_engine") is None, reason="Transformer Engine is not installed"
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("flash_attn") is None or importlib.util.find_spec("flash_attn.cute") is None,
        reason="flash-attn-4 (flash_attn.cute) is not installed",
    ),
]

NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 64
# bf16 kernels reach ~3e-3 relative error vs. the fp32 reference; the broken causal window gives 1.0 (all zeros).
REL_TOL = 2e-2


def test_te_thd_padding_causal_flash_attn_4_matches_reference(monkeypatch):
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends

    # TE reads these when it (re)selects a backend and caches the choice per attention-parameter set, so force a
    # fresh selection under this environment and restore the cache afterwards.
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
    for key in list(_attention_backends):
        monkeypatch.setitem(_attention_backends, key, _attention_backends[key])
    monkeypatch.setitem(_attention_backends, "attention_params", None)
    monkeypatch.setitem(_attention_backends, "backend_selection_requires_update", True)

    seqlens = [97, 65, 33]  # sequence boundaries not tile-aligned
    torch.manual_seed(0)
    total = sum(seqlens)
    q = torch.randn(total, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bounds = torch.tensor([0] + seqlens, device="cuda").cumsum(0).tolist()
    cu_seqlens = torch.tensor(bounds, device="cuda", dtype=torch.int32)

    attn = te.DotProductAttention(
        num_attention_heads=NUM_HEADS,
        kv_channels=HEAD_DIM,
        num_gqa_groups=NUM_KV_HEADS,
        attention_dropout=0.0,
        qkv_format="thd",
        attn_mask_type="padding_causal",
    )
    out = attn(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(seqlens),
        max_seqlen_kv=max(seqlens),
    )
    backend = _attention_backends["flash_attention_backend"]
    assert _attention_backends["use_flash_attention"] and backend is not None and backend.major == 4, (
        f"expected TE to dispatch to flash-attn-4, got use_flash_attention={_attention_backends['use_flash_attention']}"
        f", flash_attention_backend={backend}"
    )
    out = out.reshape(total, NUM_HEADS, HEAD_DIM)
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), grad_out)

    for start, end in zip(bounds[:-1], bounds[1:]):
        qs, ks, vs = (t[start:end].detach().float().transpose(0, 1).requires_grad_() for t in (q, k, v))
        ref = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True)
        ref_grads = torch.autograd.grad(ref, (qs, ks, vs), grad_out[start:end].float().transpose(0, 1))
        actual = [out[start:end]] + [g[start:end] for g in grads]
        expected = [ref] + list(ref_grads)
        for name, a, e in zip(["out", "dq", "dk", "dv"], actual, expected):
            e = e.transpose(0, 1)
            rel_err = ((a.float() - e).norm() / e.norm()).item()
            assert rel_err < REL_TOL, f"{name} of sequence [{start}, {end}) has relative error {rel_err:.3g}"
