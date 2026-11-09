# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType

import pytest
import torch
from lightning.pytorch.strategies.model_parallel import ModelParallelStrategy

from nemo.collections.speechlm2.parts.parallel import AutomodelParallelStrategy


class _ConvertedState(torch.Tensor):
    def cpu(self):
        self.owner.live_buffers -= 1
        return self.as_subclass(torch.Tensor).clone()


class _FusedAdam(torch.optim.Optimizer):
    """Model TE's canonical conversion with a one-buffer accelerator budget."""

    def __init__(self):
        parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        super().__init__([parameter], {'lr': 0.1})
        self.state[parameter] = {
            'exp_avg': torch.arange(4, dtype=torch.bfloat16),
            'exp_avg_sq': torch.arange(4, dtype=torch.bfloat16) + 1,
            'master_param': torch.arange(4, dtype=torch.int16),
        }
        self.live_buffers = 0

    def get_unscaled_state(self, parameter, name):
        self.live_buffers += 1
        if self.live_buffers > 1:
            raise RuntimeError('retained converted accelerator buffers')
        value = self.state[parameter][name]
        if value.dtype != torch.int16:
            value = value.float()
        result = value.as_subclass(_ConvertedState)
        result.owner = self
        return result

    def state_dict(self):
        state = super().state_dict()
        parameter = self.param_groups[0]['params'][0]
        state['state'][0] = {name: self.get_unscaled_state(parameter, name) for name in self.state[parameter]}
        return state


@pytest.fixture
def fused_adam_module(monkeypatch):
    module = ModuleType('transformer_engine.pytorch.optimizers')
    module.FusedAdam = _FusedAdam
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(ModelParallelStrategy, 'optimizer_state', lambda self, optimizer: optimizer.state_dict())


def test_fused_adam_checkpoint_offloads_each_converted_state(fused_adam_module):
    optimizer = _FusedAdam()
    parameter = optimizer.param_groups[0]['params'][0]
    original = dict(optimizer.state[parameter])
    state = AutomodelParallelStrategy().optimizer_state(optimizer)
    saved = state['state'][0]
    assert optimizer.live_buffers == 0
    assert saved['exp_avg'].dtype == saved['exp_avg_sq'].dtype == torch.float32
    assert saved['master_param'].dtype == torch.int16
    for name, value in original.items():
        assert optimizer.state[parameter][name] is value
        assert torch.equal(saved[name], value.float() if value.dtype == torch.bfloat16 else value)
    assert 'state_dict' not in optimizer.__dict__


def test_fused_adam_checkpoint_restores_method_after_failure(fused_adam_module, monkeypatch):
    optimizer = _FusedAdam()

    def fail(*args):
        raise RuntimeError('serialization failed')

    monkeypatch.setattr(optimizer, 'get_unscaled_state', fail)
    with pytest.raises(RuntimeError, match='serialization failed'):
        AutomodelParallelStrategy().optimizer_state(optimizer)
    assert 'state_dict' not in optimizer.__dict__


def test_custom_state_dict_override_is_preserved(fused_adam_module):
    optimizer = _FusedAdam()
    custom = lambda: {'custom': True}
    optimizer.state_dict = custom
    assert AutomodelParallelStrategy().optimizer_state(optimizer) == {'custom': True}
    assert optimizer.state_dict is custom


def test_plain_optimizer_does_not_import_transformer_engine(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith('transformer_engine'):
            raise AssertionError('optional optimizer imported for plain SGD')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(ModelParallelStrategy, 'optimizer_state', lambda self, optimizer: optimizer.state_dict())
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    assert AutomodelParallelStrategy().optimizer_state(optimizer)['state'] == {}


@pytest.mark.run_only_on('GPU')
def test_fused_adam_gpu_checkpoint_values_and_peak_memory():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for the real optimizer integration')
    fused_adam = pytest.importorskip('transformer_engine.pytorch.optimizers').FusedAdam
    from nemo.collections.speechlm2.parts.optimizer_checkpoint import _cpu_optimizer_checkpoint

    parameters = [torch.nn.Parameter(torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)) for _ in range(3)]
    optimizer = fused_adam(
        parameters,
        lr=0.01,
        master_weights=True,
        store_param_remainders=True,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
    )
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    native = optimizer.state_dict()
    native_peak = torch.cuda.max_memory_allocated() - baseline
    expected = {key: {name: value.cpu() for name, value in state.items()} for key, state in native['state'].items()}
    del native
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with _cpu_optimizer_checkpoint(optimizer):
        saved = optimizer.state_dict()
    incremental_peak = torch.cuda.max_memory_allocated() - baseline
    for key, state in saved['state'].items():
        for name, value in state.items():
            assert value.device.type == 'cpu'
            assert torch.equal(value, expected[key][name])
    assert incremental_peak < native_peak
    assert 'state_dict' not in optimizer.__dict__
