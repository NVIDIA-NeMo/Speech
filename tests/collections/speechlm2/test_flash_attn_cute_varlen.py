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
"""Numerics of the container's FlashAttention 4 (flash_attn.cute) for packed (varlen) attention.

Transformer Engine falls back to flash_attn.cute for packed SALM training when its fused attention is disabled
(required on SM120). These checks guard the container's flash-attn-4 pin and patches in docker/Dockerfile.
"""
import importlib.util

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="flash_attn.cute requires CUDA"),
    pytest.mark.skipif(
        importlib.util.find_spec("flash_attn") is None or importlib.util.find_spec("flash_attn.cute") is None,
        reason="flash-attn-4 (flash_attn.cute) is not installed",
    ),
]

NUM_HEADS = 8
HEAD_DIM = 64
# bf16 kernels reach ~3e-3 relative error vs. the fp32 reference; the misaligned GQA dK/dV bug gives ~1.0.
REL_TOL = 2e-2


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize(
    "seqlens,num_kv_heads",
    [
        ([97, 65, 33], 2),  # GQA, sequence boundaries not tile-aligned
        ([128, 128], 2),  # GQA, tile-aligned boundaries
        ([200], 2),  # GQA, single sequence
        ([97, 65, 33], NUM_HEADS),  # MHA
        (None, 2),  # GQA, dense (non-varlen) batch of 2 x 128
    ],
    ids=["gqa_misaligned", "gqa_aligned", "gqa_single", "mha_misaligned", "gqa_dense"],
)
def test_flash_attn_cute_matches_reference_per_sequence(seqlens, num_kv_heads, causal):
    from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func

    dense = seqlens is None
    if dense:
        seqlens = [128, 128]
    torch.manual_seed(0)
    total = sum(seqlens)
    q = torch.randn(total, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, num_kv_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total, num_kv_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bounds = torch.tensor([0] + seqlens, device="cuda").cumsum(0).tolist()

    if dense:
        out = flash_attn_func(*(t.view(len(seqlens), seqlens[0], -1, HEAD_DIM) for t in (q, k, v)), causal=causal)
    else:
        cu_seqlens = torch.tensor(bounds, device="cuda", dtype=torch.int32)
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max(seqlens),
            max_seqlen_k=max(seqlens),
            causal=causal,
        )
    out = (out[0] if isinstance(out, tuple) else out).reshape(total, NUM_HEADS, HEAD_DIM)
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), grad_out)

    for start, end in zip(bounds[:-1], bounds[1:]):
        qs, ks, vs = (t[start:end].detach().float().transpose(0, 1).requires_grad_() for t in (q, k, v))
        ref = F.scaled_dot_product_attention(qs, ks, vs, is_causal=causal, enable_gqa=True)
        ref_grads = torch.autograd.grad(ref, (qs, ks, vs), grad_out[start:end].float().transpose(0, 1))
        actual = [out[start:end]] + [g[start:end] for g in grads]
        expected = [ref] + list(ref_grads)
        for name, a, e in zip(["out", "dq", "dk", "dv"], actual, expected):
            e = e.transpose(0, 1)
            rel_err = ((a.float() - e).norm() / e.norm()).item()
            assert rel_err < REL_TOL, f"{name} of sequence [{start}, {end}) has relative error {rel_err:.3g}"
