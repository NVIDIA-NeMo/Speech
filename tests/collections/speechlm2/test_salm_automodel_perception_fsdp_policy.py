# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from nemo.collections.speechlm2.models import salm_automodel
from nemo.collections.speechlm2.parts.parallel import AutomodelParallelStrategy


class _PackedLayer(torch.nn.Linear):
    def _forward_sequence_packed(self, x, **_kwargs):
        return self(x)


class _Perception(torch.nn.Module):
    supports_sequence_packed_output = True

    def __init__(self, *, parallel_expert: bool):
        super().__init__()
        asr_encoder = torch.nn.Module()
        asr_encoder.layers = torch.nn.ModuleList([_PackedLayer(4, 4), _PackedLayer(4, 4)])
        self.encoder = SimpleNamespace(asr_encoder=asr_encoder) if parallel_expert else asr_encoder

    def forward_sequence_packed(self, **kwargs):
        return kwargs


@pytest.mark.parametrize("parallel_expert", [False, True])
def test_perception_finer_fsdp_wraps_each_asr_layer_then_root(monkeypatch, parallel_expert):
    perception = _Perception(parallel_expert=parallel_expert)
    asr_encoder = getattr(perception.encoder, "asr_encoder", perception.encoder)
    layers = list(asr_encoder.layers)
    mesh = object()
    shard_calls = []
    registered = []

    def fake_fully_shard(module, *, mesh):
        shard_calls.append((module, mesh))
        return module

    monkeypatch.setattr(salm_automodel, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(
        salm_automodel,
        "register_fsdp_forward_method",
        lambda module, method: registered.append((module, method)),
    )

    result = salm_automodel._fully_shard_perception(perception, mesh, wrap_asr_layers=True)

    assert result is perception
    assert shard_calls == [
        (layers[0], mesh),
        (layers[1], mesh),
        (perception, mesh),
    ]
    assert registered == [
        (layers[0], "_forward_sequence_packed"),
        (layers[1], "_forward_sequence_packed"),
        (perception, "forward_sequence_packed"),
    ]


def test_perception_finer_fsdp_registers_checkpointed_packed_entrypoint(monkeypatch):
    perception = _Perception(parallel_expert=True)
    perception.encoder.asr_encoder.layers[0] = checkpoint_wrapper(perception.encoder.asr_encoder.layers[0])
    layers = list(perception.encoder.asr_encoder.layers)
    registered = []

    monkeypatch.setattr(salm_automodel, "fully_shard", lambda module, *, mesh: module)
    monkeypatch.setattr(
        salm_automodel,
        "register_fsdp_forward_method",
        lambda module, method: registered.append((module, method)),
    )

    salm_automodel._fully_shard_perception(perception, object(), wrap_asr_layers=True)

    assert registered == [
        (layers[0], "checkpoint_fn"),
        (layers[1], "_forward_sequence_packed"),
        (perception, "forward_sequence_packed"),
    ]


def test_perception_default_keeps_single_root_fsdp_unit(monkeypatch):
    perception = _Perception(parallel_expert=True)
    mesh = object()
    shard_calls = []

    monkeypatch.setattr(
        salm_automodel,
        "fully_shard",
        lambda module, *, mesh: shard_calls.append((module, mesh)) or module,
    )
    monkeypatch.setattr(salm_automodel, "register_fsdp_forward_method", lambda *_args: None)

    salm_automodel._fully_shard_perception(perception, mesh)

    assert shard_calls == [(perception, mesh)]


@pytest.mark.parametrize(
    "encoder",
    [
        torch.nn.Linear(4, 4),
        SimpleNamespace(layers=torch.nn.ModuleList()),
        SimpleNamespace(layers=[torch.nn.Linear(4, 4)]),
    ],
)
def test_perception_finer_fsdp_fails_closed_without_nonempty_module_list(monkeypatch, encoder):
    perception = torch.nn.Module()
    perception.encoder = encoder
    monkeypatch.setattr(
        salm_automodel,
        "fully_shard",
        lambda *_args, **_kwargs: pytest.fail("must fail before sharding"),
    )

    with pytest.raises(ValueError, match="non-empty torch.nn.ModuleList"):
        salm_automodel._fully_shard_perception(perception, object(), wrap_asr_layers=True)


def test_perception_finer_fsdp_rejects_non_bool():
    with pytest.raises(TypeError, match="wrap_asr_layers must be a bool"):
        salm_automodel._fully_shard_perception(torch.nn.Module(), object(), wrap_asr_layers="true")


def test_strategy_exposes_finer_perception_fsdp_flag():
    assert AutomodelParallelStrategy().perception_fsdp_wrap_asr_layers is False
    assert AutomodelParallelStrategy(perception_fsdp_wrap_asr_layers=True).perception_fsdp_wrap_asr_layers is True
    with pytest.raises(TypeError, match="must be a bool"):
        AutomodelParallelStrategy(perception_fsdp_wrap_asr_layers="true")
