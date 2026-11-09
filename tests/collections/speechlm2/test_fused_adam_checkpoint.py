# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nemo.collections.speechlm2.parts.parallel import AutomodelParallelStrategy


class _FusedAdam(torch.optim.Optimizer):
    """Model TE's FP32 checkpoint moments and compact live state."""

    def __init__(self, parameter=None):
        if parameter is None:
            parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        super().__init__([parameter], {'lr': 0.1})
        self.name_to_dtype_map = {
            'exp_avg': torch.bfloat16,
            'exp_avg_sq': torch.bfloat16,
            'master_param': torch.float32,
        }
        self.master_weights = True
        self.store_param_remainders = True
        self.state[parameter] = {
            'exp_avg': torch.arange(parameter.numel(), dtype=torch.bfloat16).reshape(parameter.shape),
            'exp_avg_sq': torch.arange(parameter.numel(), dtype=torch.bfloat16).reshape(parameter.shape) + 1,
            'master_param': torch.arange(
                parameter.numel(), dtype=torch.int16 if parameter.dtype == torch.bfloat16 else torch.float32
            ).reshape(parameter.shape),
        }

    def get_unscaled_state(self, parameter, name):
        value = self.state[parameter][name]
        return value.float() if value.dtype != torch.int16 else value

    def state_dict(self):
        state = super().state_dict()
        parameter = self.param_groups[0]['params'][0]
        state['state'][0] = {name: self.get_unscaled_state(parameter, name) for name in self.state[parameter]}
        return state

    def initialize_state(self, parameter, store_param_remainders):
        self.state[parameter] = {
            'exp_avg': torch.zeros_like(parameter, dtype=torch.bfloat16),
            'exp_avg_sq': torch.zeros_like(parameter, dtype=torch.bfloat16),
            'master_param': torch.zeros_like(
                parameter, dtype=torch.int16 if store_param_remainders else torch.float32
            ),
        }


@pytest.fixture
def fused_adam_module(monkeypatch):
    module = ModuleType('transformer_engine.pytorch.optimizers')
    module.FusedAdam = _FusedAdam
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_plain_optimizer_does_not_import_transformer_engine(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith('transformer_engine'):
            raise AssertionError('optional optimizer imported for plain SGD')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    from nemo.collections.speechlm2.parts.fused_adam_checkpoint import get_fused_adam_restore_optimizer

    assert get_fused_adam_restore_optimizer(optimizer) is None


def test_strategy_restores_fused_adam_into_live_buffers(fused_adam_module, monkeypatch, tmp_path):
    import lightning.pytorch.strategies.model_parallel as lightning_parallel
    import torch.distributed.checkpoint as dcp

    from nemo.collections.speechlm2.parts import parallel

    model = torch.nn.Linear(4, 1, bias=False, dtype=torch.bfloat16)
    optimizer = _FusedAdam(model.weight)
    original = {name: value for name, value in optimizer.state[model.weight].items()}
    strategy = AutomodelParallelStrategy()
    strategy._model = model
    strategy._lightning_module = SimpleNamespace(strict_loading=True)
    strategy._optimizers = [optimizer]
    strategy.broadcast = lambda path: path
    monkeypatch.setattr(lightning_parallel, '_is_sharded_checkpoint', lambda path: True)
    monkeypatch.setattr(dcp, 'FileSystemReader', lambda path: SimpleNamespace(read_metadata=lambda: object()))
    monkeypatch.setattr(parallel, '_optimizer_load_planner', lambda *args: (None, []))
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {})

    def load(state, checkpoint_id, **kwargs):
        if 'optimizer_0' not in state:
            return
        saved = state['optimizer_0']
        values = saved['state']['weight']
        for name, tensor in values.items():
            assert tensor is original[name], f'{name}: restore allocated a checkpoint copy'
            tensor.fill_(7)
        saved['param_groups'][0]['step'] = 13
        saved['param_groups'][0]['lr'] = 0.02

    monkeypatch.setattr(dcp, 'load', load)
    strategy.load_checkpoint(tmp_path)
    assert optimizer.param_groups[0]['step'] == 13
    assert optimizer.param_groups[0]['lr'] == 0.02
    for name, value in optimizer.state[model.weight].items():
        assert value is original[name]
        assert torch.equal(value, torch.full_like(value, 7))


@pytest.mark.parametrize(
    'unsupported',
    ['moment_dtype', 'remainders', 'master_weights', 'save_hook', 'load_hook', 'save_override', 'load_override'],
)
def test_in_place_restore_preserves_unsupported_optimizer_paths(fused_adam_module, unsupported):
    from nemo.collections.speechlm2.parts.fused_adam_checkpoint import get_fused_adam_restore_optimizer

    optimizer = _FusedAdam()
    if unsupported == 'moment_dtype':
        optimizer.name_to_dtype_map['exp_avg'] = torch.float16
    elif unsupported == 'remainders':
        optimizer.store_param_remainders = False
    elif unsupported == 'master_weights':
        optimizer.master_weights = False
    elif unsupported == 'save_hook':
        optimizer.register_state_dict_post_hook(lambda *args: None)
    elif unsupported == 'load_hook':
        optimizer.register_load_state_dict_post_hook(lambda *args: None)
    elif unsupported == 'save_override':
        optimizer.state_dict = lambda: {'custom': True}
    else:
        optimizer.load_state_dict = lambda state: None
    assert get_fused_adam_restore_optimizer(optimizer) is None


@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('param_dtype', [torch.bfloat16, torch.float32])
def test_in_place_restore_reads_canonical_dcp(fused_adam_module, tmp_path, partial, param_dtype):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict

    from nemo.collections.speechlm2.parts.fused_adam_checkpoint import get_fused_adam_restore_optimizer
    from nemo.collections.speechlm2.parts.parallel import _optimizer_load_planner

    model = torch.nn.Linear(4, 1, bias=False, dtype=param_dtype)
    source = _FusedAdam(model.weight)
    source.param_groups[0].update(step=17, lr=0.02)
    expected = {k: v.clone() for k, v in source.state[model.weight].items()}
    saved = get_optimizer_state_dict(model, source)
    if partial:
        del saved['state']['weight']['exp_avg_sq']
    dcp.save({'optimizer_0': saved}, checkpoint_id=tmp_path)

    optimizer = _FusedAdam(model.weight)
    optimizer.state.clear()
    adapter = get_fused_adam_restore_optimizer(optimizer)
    live = dict(optimizer.state[model.weight])
    target = {'optimizer_0': get_optimizer_state_dict(model, adapter)}
    metadata = dcp.FileSystemReader(tmp_path).read_metadata()
    if partial:
        with pytest.raises(RuntimeError, match='partial optimizer state'):
            _optimizer_load_planner(target, metadata, 'optimizer_0')
        return
    planner, missing = _optimizer_load_planner(target, metadata, 'optimizer_0')
    assert missing == []
    dcp.load(target, checkpoint_id=tmp_path, planner=planner)
    set_optimizer_state_dict(model, adapter, optim_state_dict=target['optimizer_0'])
    assert optimizer.param_groups[0]['step'] == 17
    assert optimizer.param_groups[0]['lr'] == 0.02
    for key, value in optimizer.state[model.weight].items():
        assert value is live[key]
        assert value.dtype == expected[key].dtype
        assert torch.equal(value, expected[key])


@pytest.mark.gpu
@pytest.mark.run_only_on('GPU')
@pytest.mark.parametrize('mixed_parameters', [False, True])
def test_fused_adam_gpu_in_place_dcp_resume(tmp_path, monkeypatch, mixed_parameters):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for the real optimizer integration')
    fused_adam = pytest.importorskip('transformer_engine.pytorch.optimizers').FusedAdam
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict

    from nemo.collections.speechlm2.parts.fused_adam_checkpoint import get_fused_adam_restore_optimizer

    def make_optimizer(model):
        return fused_adam(
            model.parameters(),
            lr=0.01,
            master_weights=True,
            store_param_remainders=True,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.bfloat16,
        )

    model = torch.nn.Linear(2048, 2048, bias=False, device='cuda', dtype=torch.bfloat16)
    if mixed_parameters:
        model.register_parameter('extra', torch.nn.Parameter(torch.ones(16, dtype=torch.float32, device='cuda')))
    original = make_optimizer(model)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    original.step()
    original.zero_grad(set_to_none=True)
    saved = {'optimizer_0': get_optimizer_state_dict(model, original)}
    dcp.save(saved, checkpoint_id=tmp_path)
    del saved

    restored = torch.nn.Linear(2048, 2048, bias=False, device='cuda', dtype=torch.bfloat16)
    if mixed_parameters:
        restored.register_parameter('extra', torch.nn.Parameter(torch.zeros(16, dtype=torch.float32, device='cuda')))
    restored.load_state_dict(model.state_dict())
    optimizer = make_optimizer(restored)
    adapter = get_fused_adam_restore_optimizer(optimizer)
    pointers = {
        id(parameter): {name: value.data_ptr() for name, value in state.items()}
        for parameter, state in optimizer.state.items()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError('In-place restore called a bulk TE checkpoint method')

    with monkeypatch.context() as context:
        context.setattr(fused_adam, 'state_dict', forbidden)
        context.setattr(fused_adam, 'load_state_dict', forbidden)
        target = {'optimizer_0': get_optimizer_state_dict(restored, adapter)}
        dcp.load(target, checkpoint_id=tmp_path)
        set_optimizer_state_dict(restored, adapter, optim_state_dict=target['optimizer_0'])
    assert optimizer.param_groups[0]['step'] == original.param_groups[0]['step']
    for source, target in zip(model.parameters(), restored.parameters()):
        for name, value in optimizer.state[target].items():
            assert value.data_ptr() == pointers[id(target)][name]
            assert value.dtype == original.state[source][name].dtype
            torch.testing.assert_close(value, original.state[source][name], rtol=0, atol=0)

    for module, opt in [(model, original), (restored, optimizer)]:
        for parameter in module.parameters():
            parameter.grad = torch.full_like(parameter, 0.5)
        opt.step()
    for source, target in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(target, source, rtol=0, atol=0)
        for name, value in optimizer.state[target].items():
            torch.testing.assert_close(value, original.state[source][name], rtol=0, atol=0)
