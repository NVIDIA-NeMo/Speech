# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from nemo.collections.speechlm2.parts.refit_verification import RefitWeightVerifier, append_refit_receipt


class ParameterModel:
    def __init__(self, name, parameter):
        self.parameters = {name: parameter}

    def get_parameter(self, name):
        return self.parameters[name]


@pytest.mark.parametrize(
    'name,axis,shape',
    [
        ('perception.proj.weight', None, (3, 4)),
        ('language_model.backbone.layers.0.norm.weight', None, (8,)),
        ('language_model.backbone.layers.0.mixer.D', 0, (4,)),
        ('language_model.backbone.layers.0.mixer.dt_bias', 0, (4,)),
        ('language_model.backbone.layers.0.mixer.out_proj.weight', 1, (3, 4)),
    ],
)
def test_received_bf16_matches_replicated_and_public_tp_layouts(name, axis, shape):
    local_shape = list(shape)
    if axis is not None:
        local_shape[axis] //= 2
    parameter = torch.nn.Parameter(torch.zeros(local_shape, dtype=torch.float32))
    if axis == 1:
        parameter.input_dim = 1
    received = (torch.arange(torch.tensor(shape).prod()).reshape(shape).float() / 7).bfloat16()
    model = ParameterModel(name, parameter)
    verifier = RefitWeightVerifier(model, {name: axis}, tp_rank=1, tp_size=2)
    verifier.capture(name, received)
    expected = received if axis is None else received.narrow(axis, local_shape[axis], local_shape[axis])
    with torch.no_grad():
        parameter.copy_(expected)
    receipt = verifier.verify()['samples'][name]
    assert receipt['actual'] == receipt['expected']
    assert receipt['transport_dtype'] == 'torch.bfloat16'
    assert receipt['destination_dtype'] == 'torch.float32'
    assert len(receipt['coordinates']) <= 4
    assert not any(isinstance(value, torch.Tensor) for value in verifier.samples[name].values())


@pytest.mark.parametrize('failure', ['values', 'nan', 'inf', 'dtype', 'shape'])
def test_changed_or_nonfinite_destination_fails(failure):
    name = 'perception.proj.weight'
    parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    model = ParameterModel(name, parameter)
    verifier = RefitWeightVerifier(model, {name: None}, tp_rank=0, tp_size=1)
    verifier.capture(name, torch.ones(3, dtype=torch.bfloat16))
    with torch.no_grad():
        parameter.fill_(1)
        if failure == 'values':
            parameter[0] = 2
        elif failure in ('nan', 'inf'):
            parameter[0] = float(failure)
        elif failure == 'dtype':
            model.parameters[name] = torch.nn.Parameter(parameter.float())
        else:
            model.parameters[name] = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        verifier.verify()


@pytest.mark.parametrize('failure', ['dtype', 'nan', 'shape', 'unsupported_axis', 'linear_metadata'])
def test_invalid_transport_or_unproven_mapping_fails(failure):
    name = 'language_model.backbone.layers.0.mixer.out_proj.weight'
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    parameter.input_dim = 1
    received = torch.ones(2, 4, dtype=torch.bfloat16)
    axis = 1
    if failure == 'dtype':
        received = received.float()
    elif failure == 'nan':
        received.fill_(float('nan'))
    elif failure == 'shape':
        received = torch.ones(2, 5, dtype=torch.bfloat16)
    elif failure == 'unsupported_axis':
        axis = 0
    else:
        del parameter.input_dim
    verifier = RefitWeightVerifier(ParameterModel(name, parameter), {name: axis}, tp_rank=0, tp_size=2)
    with pytest.raises(ValueError):
        verifier.capture(name, received)


def test_only_explicit_samples_are_retained_and_receipts_append(tmp_path):
    name = 'perception.proj.weight'
    model = ParameterModel(name, torch.nn.Parameter(torch.ones(10_000, dtype=torch.bfloat16)))
    verifier = RefitWeightVerifier(model, {name: None}, tp_rank=0, tp_size=1)
    verifier.capture('not.a.parameter', torch.ones(1))
    verifier.capture(name, torch.ones(10_000, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match='Duplicate'):
        verifier.capture(name, torch.ones(10_000, dtype=torch.bfloat16))
    receipt = verifier.verify()
    append_refit_receipt(str(tmp_path), receipt)
    append_refit_receipt(str(tmp_path), receipt)
    assert len(list(tmp_path.iterdir())) == 1
    lines = next(tmp_path.iterdir()).read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])['samples'][name]['verified']


@pytest.mark.parametrize("mapped", [True, False])
def test_receiver_mapper_resolves_destination_without_changing_sample_keys(mapped):
    source_name = "language_model.backbone.layers.0.mixer.D" if mapped else "language_model.model.layers.0.mixer.D"
    destination_name = "language_model.model.layers.0.mixer.D"
    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.model = torch.nn.Module()
    model.language_model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.language_model.model.layers[0].mixer = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    model.language_model.model.layers[0].mixer.register_parameter("D", parameter)
    calls = []
    if mapped:

        class ReceiverMapper:
            def apply_list(self, names):
                calls.append(names)
                return [name.replace("backbone", "model", 1) for name in names]

        model.language_model.hf_to_vllm_mapper = ReceiverMapper()
    verifier = RefitWeightVerifier(model, {source_name: 0}, tp_rank=1, tp_size=2)
    received = torch.arange(8, dtype=torch.bfloat16)
    verifier.capture(source_name, received)
    with torch.no_grad():
        parameter.copy_(received[4:])
    receipt = verifier.verify()["samples"][source_name]
    assert receipt["destination_name"] == destination_name
    assert receipt["actual"] == receipt["expected"] == [4.0, 5.0, 6.0, 7.0]
    assert set(verifier.samples) == {source_name}
    assert calls == ([["backbone.layers.0.mixer.D"]] if mapped else [])


@pytest.mark.parametrize("destinations", [[], ["model.D", "other.D"]])
def test_receiver_mapper_rejects_missing_or_ambiguous_sample(destinations):
    class ReceiverMapper:
        def apply_list(self, names):
            return destinations

    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.hf_to_vllm_mapper = ReceiverMapper()
    name = "language_model.backbone.layers.0.mixer.D"
    verifier = RefitWeightVerifier(model, {name: 0}, tp_rank=0, tp_size=1)
    with pytest.raises(ValueError, match="no unique receiver destination"):
        verifier.capture(name, torch.ones(4, dtype=torch.bfloat16))
