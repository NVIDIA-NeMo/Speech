# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve the Nemotron training precision contract in composed vLLM models."""

import logging

import torch
from torch import nn

_LOGGER = logging.getLogger(__name__)


def preserve_hybrid_precision(model: nn.Module) -> None:
    """Guard target and draft weights before loading can round FP32 checkpoint values.

    Keep the parameter objects and their vLLM weight-loader metadata intact.
    Only the hybrid model passed by the SpeechLM wrapper is affected; the
    remaining language and perception parameters retain their serving dtype.
    """
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    mamba_count = router_count = 0
    for module in model.modules():
        if isinstance(module, MambaMixer2):
            _promote_mamba_parameters(module)
            mamba_count += 1
        elif isinstance(module, GateLinear):
            _promote_router_precision(module)
            router_count += 1
    _LOGGER.info("SpeechLM FP32 precision guards: %d Mamba mixers, %d routers", mamba_count, router_count)


def _promote_mamba_parameters(mixer: nn.Module) -> None:
    for name in ("A", "D", "dt_bias"):
        parameter = getattr(mixer, name)
        parameter.data = parameter.data.to(torch.float32)


def _promote_router_precision(gate: nn.Module) -> None:
    gate.weight.data = gate.weight.data.to(torch.float32)
    if getattr(gate, "bias", None) is not None:
        gate.bias.data = gate.bias.data.to(torch.float32)
    gate.out_dtype = torch.float32
    # BF16 GEMM with FP32 output is not the training FP32-operand contract.
    # Retain vLLM's specialized FP32 GEMM when the hardware and shape admit it;
    # otherwise GateLinear's existing linear path casts inputs to weight.dtype.
    for name in (
        "allow_ll_bf16_gemm",
        "allow_dsv3_router_gemm",
        "allow_cublas_router_gemm",
        "allow_bf16x3_router_gemm",
    ):
        setattr(gate, name, False)
    shape = (gate.weight.shape[1], gate.weight.shape[0])
    gate.allow_fp32_router_gemm = bool(getattr(gate, "allow_specialized_router_gemm", False)) and shape in getattr(
        gate, "FP32_SUPPORTED_SHAPES", set()
    )
