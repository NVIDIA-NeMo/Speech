# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.common.callbacks.ema import EMA, EMAOptimizer


@pytest.mark.unit
@pytest.mark.parametrize('optimizer_cls', [torch.optim.SGD, torch.optim.Adam])
def test_ema_resume_restores_optimizer_state(tmp_path, optimizer_cls):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = EMAOptimizer(optimizer_cls([parameter], lr=0.1), device=torch.device('cpu'))
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state_dict()
    torch.save(
        {'optimizer_states': [state], 'epoch': 1, 'hyper_parameters': {'cfg': OmegaConf.create({'layers': [1, 2]})}},
        tmp_path / 'model-EMA.ckpt',
    )
    checkpoint = {}
    EMA(decay=0.9).on_load_checkpoint(_trainer(tmp_path), None, checkpoint)
    restored = EMAOptimizer(optimizer_cls([torch.nn.Parameter(torch.zeros(1))], lr=0.2), device=torch.device('cpu'))
    restored.load_state_dict(checkpoint['optimizer_states'][0])
    assert restored.current_step == optimizer.current_step
    torch.testing.assert_close(restored.ema_params, optimizer.ema_params)
    assert restored.param_groups[0]['lr'] == 0.1


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
