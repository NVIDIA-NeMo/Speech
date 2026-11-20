# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import operator
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import OmegaConf

from nemo.collections.common.callbacks.ema import EMA, EMAOptimizer


@pytest.mark.unit
@pytest.mark.parametrize('optimizer_cls', [torch.optim.SGD, torch.optim.Adam])
@pytest.mark.parametrize('schema', ['plain', 'ctc', 'multitask'])
def test_ema_resume_restores_optimizer_state(tmp_path, optimizer_cls, schema):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = EMAOptimizer(optimizer_cls([parameter], lr=0.1), device=torch.device('cpu'))
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state_dict()
    config = OmegaConf.create({'layers': [1, 2], 'resource': Path('model-resource')})
    if schema == 'ctc':
        from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig

        config.decoding = OmegaConf.structured(CTCDecodingConfig)
    elif schema == 'multitask':
        from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecodingConfig

        config.decoding = OmegaConf.structured(MultiTaskDecodingConfig)
    torch.save(
        {'optimizer_states': [state], 'epoch': 1, 'hyper_parameters': {'cfg': config}},
        tmp_path / 'model-EMA.ckpt',
    )
    checkpoint = {}
    EMA(decay=0.9).on_load_checkpoint(_trainer(tmp_path), None, checkpoint)
    restored = EMAOptimizer(optimizer_cls([torch.nn.Parameter(torch.zeros(1))], lr=0.2), device=torch.device('cpu'))
    restored.load_state_dict(checkpoint['optimizer_states'][0])
    assert restored.current_step == optimizer.current_step
    torch.testing.assert_close(restored.ema_params, optimizer.ema_params)
    assert [{k: v for k, v in g.items() if k != 'params'} for g in restored.param_groups] == [
        {k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups
    ]
    if optimizer_cls is torch.optim.Adam:
        for key in ('exp_avg', 'exp_avg_sq', 'step'):
            torch.testing.assert_close(
                next(iter(restored.optimizer.state.values()))[key], next(iter(optimizer.optimizer.state.values()))[key]
            )


@pytest.mark.unit
@pytest.mark.parametrize('phase', [TrainerFn.VALIDATING, TrainerFn.TESTING, TrainerFn.PREDICTING])
def test_ema_evaluation_does_not_restore_companion_state(tmp_path, phase):
    trainer = _trainer(tmp_path)
    trainer.state = SimpleNamespace(fn=phase)
    checkpoint = {'optimizer_states': ['original']}
    EMA(decay=0.9).on_load_checkpoint(trainer, None, checkpoint)
    assert checkpoint['optimizer_states'] == ['original']


@pytest.mark.unit
def test_ema_resume_rejects_pickle_code_execution(tmp_path):
    marker = tmp_path / 'unpickled'
    torch.save(
        {'optimizer_states': [], 'payload': _PickleMarker(str(marker)), 'cfg': OmegaConf.create({'name': 'model'})},
        tmp_path / 'model-EMA.ckpt',
    )
    checkpoint = {'optimizer_states': ['original']}
    try:
        with pytest.raises(pickle.UnpicklingError):
            EMA(decay=0.9).on_load_checkpoint(_trainer(tmp_path), None, checkpoint)
    finally:
        assert not marker.exists(), 'EMA resume executed a pickle reducer'
    assert checkpoint['optimizer_states'] == ['original']


@pytest.mark.unit
def test_ema_annotation_loading_does_not_resolve_config(tmp_path):
    calls = []
    OmegaConf.register_new_resolver('ema_checkpoint_probe', lambda: calls.append('resolved'))
    checkpoint = {'optimizer_states': ['original']}
    try:
        cfg = OmegaConf.create({'value': '${ema_checkpoint_probe:}'})
        torch.save({'payload': _ItemAccess(cfg)}, tmp_path / 'model-EMA.ckpt')
        with pytest.raises(pickle.UnpicklingError, match='typing annotations'):
            EMA(decay=0.9).on_load_checkpoint(_trainer(tmp_path), None, checkpoint)
        assert calls == []
        assert checkpoint['optimizer_states'] == ['original']
    finally:
        OmegaConf.clear_resolver('ema_checkpoint_probe')


@pytest.mark.unit
def test_ema_resume_preserves_existing_configuration_registrations(tmp_path):
    from nemo.utils.serialization import checkpoint_config_types

    previous = torch.serialization.get_safe_globals()
    try:
        types = checkpoint_config_types()
        torch.serialization.add_safe_globals(types)
        expected = set(torch.serialization.get_safe_globals())
        torch.save({'optimizer_states': [], 'cfg': OmegaConf.create({'name': 'model'})}, tmp_path / 'model-EMA.ckpt')
        EMA(decay=0.9).on_load_checkpoint(_trainer(tmp_path), None, {})
        assert expected <= set(torch.serialization.get_safe_globals())
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(previous)


class _ItemAccess:
    def __init__(self, cfg):
        self.cfg = cfg

    def __reduce__(self):
        return operator.getitem, (self.cfg, 'value')


def _trainer(tmp_path):
    callback = type('NeMoCheckpointStub', (), {'FILE_EXTENSION': '.ckpt'})()
    return SimpleNamespace(ckpt_path=str(tmp_path / 'model.ckpt'), checkpoint_callback=callback)


def _write_marker(path):
    Path(path).touch()


class _PickleMarker:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return _write_marker, (self.path,)
