# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility coverage for dependency imports and checkpoint configuration."""

import io
import pickle
import subprocess
import sys
from importlib import import_module
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf


@pytest.mark.unit
@pytest.mark.parametrize('module', ['nemo.core', 'nemo.utils.exp_manager'])
def test_dependency_imports(module):
    result = subprocess.run([sys.executable, '-c', f'import {module}'], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
@pytest.mark.parametrize('namespace', ['lightning.pytorch', 'pytorch_lightning'])
def test_checkpoint_configuration_loads_with_weights_only(tmp_path, namespace):
    lightning = import_module(namespace)

    class Minimal(lightning.LightningModule):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.weight = torch.nn.Parameter(torch.zeros(2))

    config = OmegaConf.create({'name': 'model', 'layers': [2, 4], 'enabled': True, 'dropout': 0.1, 'optional': None})
    path = tmp_path / 'model.ckpt'
    torch.save(
        {
            'pytorch-lightning_version': lightning.__version__,
            'hyper_parameters': {'cfg': config},
            'state_dict': {'weight': torch.ones(2)},
        },
        path,
    )
    import nemo.utils  # noqa: F401

    restored = Minimal.load_from_checkpoint(path, weights_only=True)
    assert OmegaConf.to_container(restored.cfg) == OmegaConf.to_container(config)
    assert torch.equal(restored.weight, torch.ones(2))


@pytest.mark.unit
def test_weights_only_still_rejects_unregistered_objects():
    stream = io.BytesIO()
    torch.save(_UnregisteredObject(), stream)
    stream.seek(0)
    import nemo.utils  # noqa: F401

    with pytest.raises(pickle.UnpicklingError):
        torch.load(stream, weights_only=True)


@pytest.mark.unit
@pytest.mark.parametrize('namespace', ['lightning.pytorch', 'pytorch_lightning'])
def test_checkpoint_instantiators_keep_the_lightning_policy(tmp_path, namespace):
    lightning = import_module(namespace)
    cli = import_module(f'{namespace}.cli')

    class Minimal(lightning.LightningModule):
        pass

    checkpoint = {
        'state_dict': {},
        'hyper_parameters': {'_instantiator': 'builtins.print'},
        'pytorch-lightning_version': lightning.__version__,
    }
    path = tmp_path / 'model.ckpt'
    torch.save(checkpoint, path)
    with patch('builtins.print') as instantiate:
        with pytest.raises(ValueError, match='allowlist'):
            Minimal.load_from_checkpoint(path, weights_only=True)
        instantiate.assert_not_called()
    checkpoint['hyper_parameters']['_instantiator'] = f'{namespace}.cli.instantiate_module'
    torch.save(checkpoint, path)
    with patch.object(cli, 'instantiate_module', side_effect=lambda cls, kwargs: cls()) as instantiate:
        assert isinstance(Minimal.load_from_checkpoint(path, weights_only=True), Minimal)
        instantiate.assert_called_once_with(Minimal, {})


@pytest.mark.unit
def test_default_lifecycle_callbacks_do_not_import_telemetry():
    script = r"""
import importlib.abc
import sys

class BlockTelemetry(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('nv_one_logger'):
            raise AssertionError('telemetry imported')

sys.meta_path.insert(0, BlockTelemetry())
import nemo.core
import nemo.utils.exp_manager
from nemo.lightning.base_callback import BaseCallback
from nemo.lightning.callback_group import CallbackGroup

class CustomCallback(BaseCallback):
    calls = 0
    def on_model_init_start(self):
        self.calls += 1

group = CallbackGroup()
assert group.callbacks == []
callback = CustomCallback()
group.register(callback)
group.on_model_init_start()
assert callback.calls == 1
assert not any(name.startswith('nv_one_logger') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


class _UnregisteredObject:
    pass
