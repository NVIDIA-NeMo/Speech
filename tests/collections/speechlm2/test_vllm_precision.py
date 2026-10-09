# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo.collections.speechlm2.vllm.salm.precision import _promote_mamba_parameters, _promote_router_precision


def test_target_load_preserves_fp32_checkpoint_values(hybrid_checkpoint, monkeypatch):
    from nemo.collections.speechlm2.vllm.salm import model as model_module

    body, checkpoint, parameters = hybrid_checkpoint
    tower = nn.Module()
    tower.backbone = body
    tower.make_empty_intermediate_tensors = lambda: None
    config = SimpleNamespace(is_hybrid=True, text_config=SimpleNamespace(vocab_size=3), perception={})
    model_class = model_module.NeMoSpeechLMForConditionalGeneration
    monkeypatch.setattr(model_class, "_mark_language_model", lambda *args: nullcontext())
    monkeypatch.setattr(model_class, "_mark_tower_model", lambda *args: nullcontext())
    monkeypatch.setattr(model_module, "init_vllm_registered_model", lambda **kwargs: tower)
    monkeypatch.setattr(model_module, "_load_nemo_perception", lambda config: nn.Module())
    monkeypatch.setattr(model_module, "_maybe_mount_pe_encoder", lambda *args: None)
    monkeypatch.setattr(model_class, "_load_perception_weights", lambda self, weights: set())

    model = model_class(vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=config)))
    loaded = model.load_weights(("llm.model." + name, value) for name, value in checkpoint.items())

    assert loaded == {"language_model.backbone." + name for name in checkpoint}
    _assert_preserved_checkpoint(body, checkpoint, parameters)


def test_mtp_load_preserves_fp32_checkpoint_values(hybrid_checkpoint, monkeypatch):
    from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP
    from vllm.model_executor.models.utils import AutoWeightsLoader

    from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

    body, checkpoint, parameters = hybrid_checkpoint
    model = object.__new__(NeMoSpeechLMMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mtp_hybrid_override_pattern="*E", vocab_size=3)
    model.mtp = body
    # Avoid constructing a full draft; retain vLLM's real parameter-copy path.
    monkeypatch.setattr(
        NemotronHMTP, "load_weights", lambda self, weights: AutoWeightsLoader(self).load_weights(weights)
    )

    loaded = model.load_weights(("llm.mtp." + name, value) for name, value in checkpoint.items())

    assert loaded == {"mtp." + name for name in checkpoint}
    _assert_preserved_checkpoint(body, checkpoint, parameters)


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


@pytest.fixture
def hybrid_checkpoint():
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    mixer = object.__new__(MambaMixer2)
    nn.Module.__init__(mixer)
    for name in ("A", "D", "dt_bias"):
        mixer.register_parameter(name, nn.Parameter(torch.empty(3, dtype=torch.bfloat16)))
    mixer.in_proj = nn.Linear(3, 3, bias=False, dtype=torch.bfloat16)
    gate = object.__new__(GateLinear)
    nn.Module.__init__(gate)
    gate.weight = nn.Parameter(torch.empty(2, 3, dtype=torch.bfloat16))
    gate.bias = nn.Parameter(torch.empty(2, dtype=torch.bfloat16))
    gate.allow_specialized_router_gemm = False
    body = nn.Module()
    body.layers = nn.ModuleList([nn.Module(), nn.Module()])
    body.layers[0].mixer = mixer
    body.layers[1].mixer = nn.Module()
    body.layers[1].mixer.gate = gate

    source = torch.tensor([1.001, 0.1001, -0.03003], dtype=torch.float32)
    checkpoint = {"layers.0.mixer." + name: source.clone() for name in ("A", "D", "dt_bias")}
    checkpoint["layers.1.mixer.gate.weight"] = source.repeat(2, 1)
    checkpoint["layers.1.mixer.gate.bias"] = source[:2].clone()
    parameters = {name: parameter for name, parameter in body.named_parameters() if name in checkpoint}

    def load_fp32(parameter, tensor):
        # Check the destination at the copy, not just after load_weights returns.
        assert parameter.dtype == torch.float32
        default_weight_loader(parameter, tensor)

    for name, parameter in parameters.items():
        assert not torch.equal(checkpoint[name], checkpoint[name].bfloat16().float())
        parameter.weight_loader = load_fp32
    checkpoint["layers.0.mixer.in_proj.weight"] = source.repeat(3, 1)
    return body, checkpoint, parameters


def _assert_preserved_checkpoint(body, checkpoint, original_parameters):
    loaded_parameters = dict(body.named_parameters())
    for name, original in original_parameters.items():
        parameter = loaded_parameters[name]
        assert parameter is original
        assert parameter.dtype == torch.float32
        assert torch.equal(parameter, checkpoint[name])
    ordinary = body.layers[0].mixer.in_proj.weight
    assert ordinary.dtype == torch.bfloat16
    assert torch.equal(ordinary, checkpoint["layers.0.mixer.in_proj.weight"].bfloat16())
