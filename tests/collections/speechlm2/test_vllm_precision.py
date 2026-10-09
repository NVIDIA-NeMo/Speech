# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from nemo.collections.speechlm2.vllm.salm.precision import _promote_mamba_parameters, _promote_router_precision


def test_mamba_fp32_checkpoint_values_survive_bf16_serving():
    mixer = nn.Module()
    for name in ("A", "D", "dt_bias"):
        mixer.register_parameter(name, nn.Parameter(torch.empty(3, dtype=torch.bfloat16)))
    mixer.in_proj = nn.Linear(3, 3, dtype=torch.bfloat16)
    source = torch.tensor([1.001, 0.1001, -0.03003], dtype=torch.float32)
    original = mixer.D
    original.weight_loader = object()
    loader = original.weight_loader
    _promote_mamba_parameters(mixer)
    assert mixer.D is original and mixer.D.weight_loader is loader
    for name in ("A", "D", "dt_bias"):
        param = getattr(mixer, name)
        param.data.copy_(source)
        assert param.dtype == torch.float32
        assert torch.equal(param, source)
    assert mixer.in_proj.weight.dtype == torch.bfloat16
    _promote_mamba_parameters(mixer)
    assert mixer.D is original


@pytest.mark.parametrize("specialized", [False, True])
def test_router_uses_fp32_operands_and_preserves_loader(specialized):
    gate = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    gate.out_dtype = torch.float32
    gate.allow_specialized_router_gemm = specialized
    gate.FP32_SUPPORTED_SHAPES = {(3, 2)}
    for name in (
        "allow_ll_bf16_gemm",
        "allow_dsv3_router_gemm",
        "allow_cublas_router_gemm",
        "allow_bf16x3_router_gemm",
    ):
        setattr(gate, name, True)
    original = gate.weight
    original.weight_loader = object()
    _promote_router_precision(gate)
    assert gate.weight is original and gate.weight.dtype == torch.float32
    assert gate.allow_fp32_router_gemm is specialized
    assert not any(
        getattr(gate, name)
        for name in (
            "allow_ll_bf16_gemm",
            "allow_dsv3_router_gemm",
            "allow_cublas_router_gemm",
            "allow_bf16x3_router_gemm",
        )
    )
    # GateLinear's final dispatch casts x to weight.dtype before F.linear.
    x = torch.tensor([[1.0, 0.25, -0.5]], dtype=torch.bfloat16)
    weight = torch.tensor([[1.001, 0.1001, -0.03003], [0.3333, 1.2345, 0.1234]])
    gate.weight.data.copy_(weight)
    actual = torch.nn.functional.linear(x.to(gate.weight.dtype), gate.weight)
    assert torch.equal(actual, torch.nn.functional.linear(x.float(), weight))


def test_router_does_not_enable_unsupported_fp32_kernel():
    gate = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    gate.allow_specialized_router_gemm = True
    gate.FP32_SUPPORTED_SHAPES = {(4096, 256)}
    _promote_router_precision(gate)
    assert gate.allow_fp32_router_gemm is False


def test_real_vllm_router_dispatch_matches_fp32_linear(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.linear import ReplicatedLinear

    gate = GateLinear.__new__(GateLinear)
    nn.Module.__init__(gate)
    gate.weight = nn.Parameter(torch.empty(2, 3, dtype=torch.bfloat16))
    gate.bias = None
    gate.allow_specialized_router_gemm = False
    _promote_router_precision(gate)
    observed = []

    def linear(module, x):
        observed.append((x.dtype, module.weight.dtype))
        return torch.nn.functional.linear(x, module.weight), None

    monkeypatch.setattr(ReplicatedLinear, "forward", linear)
    weight = torch.tensor([[1.001, 0.1001, -0.03003], [0.3333, 1.2345, 0.1234]])
    gate.weight.data.copy_(weight)
    x = torch.tensor([[1.0, 0.25, -0.5]], dtype=torch.bfloat16)
    actual, _ = gate(x)
    assert observed == [(torch.float32, torch.float32)]
    assert torch.equal(actual, torch.nn.functional.linear(x.float(), weight))
