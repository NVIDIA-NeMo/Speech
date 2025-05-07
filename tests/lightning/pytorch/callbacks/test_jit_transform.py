# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
import torch

from nemo.lightning.pytorch.callbacks.jit_transform import (
    JitConfig,
    JitTransform,
    compile_module,
    extract_module_attr_name,
    get_modules_from_selector,
    listify,
)


def test_extract_module_attr_name_with_module():
    mock_pl_module = MagicMock(spec=[])
    mock_pl_module.module = MagicMock()
    assert extract_module_attr_name(mock_pl_module) == 'module', mock_pl_module


def test_extract_module_attr_name_with_model():
    mock_pl_module = MagicMock(spec=[])
    mock_pl_module.model = MagicMock()
    assert extract_module_attr_name(mock_pl_module) == 'model', mock_pl_module


def test_extract_module_attr_name_raises():
    mock_pl_module = MagicMock(spec=[])
    # no 'module' or 'model'
    with pytest.raises(ValueError, match="Expected lightning_module to have a .model or .module"):
        extract_module_attr_name(mock_pl_module)


def test_listify_non_list():
    assert listify("test") == ["test"]


def test_listify_list():
    assert listify(["test"]) == ["test"]


def test_get_modules_from_selector_none_selector():
    model = MagicMock()
    collected = list(get_modules_from_selector(model, None))
    assert collected == [model]


def test_get_modules_from_selector_empty_string():
    model = MagicMock()
    collected = list(get_modules_from_selector(model, ""))
    assert collected == [model]


def test_get_modules_from_selector_star():
    model = MagicMock()
    collected = list(get_modules_from_selector(model, "*"))
    assert collected == [model]


def test_get_modules_from_selector_exact_path():
    # Example: model.encoder.layer
    child_module = torch.nn.Linear(3, 3)
    parent_module = torch.nn.Module()
    parent_module.encoder = torch.nn.Module()
    parent_module.encoder.layer = child_module

    collected = list(get_modules_from_selector(parent_module, "encoder.layer"))
    assert collected == [child_module]


def test_get_modules_from_selector_non_existent_attr():
    parent_module = torch.nn.Module()
    parent_module.encoder = torch.nn.Module()
    with pytest.raises(AttributeError, match="has no attribute"):
        list(get_modules_from_selector(parent_module, "decoder"))


def test_get_modules_from_selector_attr_is_not_module():
    parent_module = torch.nn.Module()
    parent_module.something = "I am not a module"
    with pytest.raises(AttributeError, match="is not an nn.Module"):
        list(get_modules_from_selector(parent_module, "something"))


def test_get_modules_from_selector_wildcard_children():
    parent_module = torch.nn.Module()
    parent_module.block1 = torch.nn.Linear(3, 3)
    parent_module.block2 = torch.nn.Linear(3, 3)
    collected = list(get_modules_from_selector(parent_module, "block*"))
    assert len(collected) == 2


def test_jit_config_assertion():
    # Should raise if both use_torch and use_thunder
    with pytest.raises(AssertionError):
        JitConfig(use_torch=True, use_thunder=True)


def test_compile_module_torch():
    mock_module = MagicMock()
    config = JitConfig(use_torch=True, torch_kwargs={"some_arg": 123})
    compiled = compile_module(config, mock_module)
    mock_module.compile.assert_called_once_with(some_arg=123)
    assert compiled


# Disabling due to issue with 25.03  https://github.com/pytorch/pytorch/issues/144567
# def test_compile_module_thunder():
#    mock_module = MagicMock()
#    config = JitConfig(use_thunder=True)
#    compiled = compile_module(config, mock_module)
#    mock_module.compile.assert_called_once()
#    assert compiled


def test_compile_module_none():
    mock_module = MagicMock()
    config = JitConfig()
    compiled = compile_module(config, mock_module)
    mock_module.compile.assert_not_called()
    assert not compiled


def test_jit_transform_no_config():
    # If config is None, on_train_epoch_start returns early
    transform = JitTransform(JitConfig(use_thunder=False, use_torch=False))
    trainer_mock = MagicMock()
    pl_module = MagicMock(spec=[])
    transform.on_train_epoch_start(trainer_mock, pl_module)
    assert not getattr(pl_module, '_compiled', False)


def test_jit_transform_already_compiled():
    transform = JitTransform(JitConfig(use_torch=True))
    trainer_mock = MagicMock()
    pl_module = MagicMock(spec=[])
    pl_module._compiled = True
    pl_module.module = True
    transform.on_train_epoch_start(trainer_mock, pl_module)
    # Should remain True, and compile should not be called again
    assert pl_module._compiled is True
    assert pl_module.module == True


def test_jit_transform_compile_once():
    # simulate successful compile (torch or thunder)
    transform = JitTransform(JitConfig(use_torch=True))
    trainer_mock = MagicMock()

    # pl_module with the 'module' attribute (matching whatever name you expect inside transform)
    pl_module = MagicMock()
    pl_module.module = MagicMock()

    transform.on_train_epoch_start(trainer_mock, pl_module)
    assert pl_module._compiled is True
