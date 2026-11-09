# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import functools
import sys
import textwrap
from unittest.mock import patch

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer
from nemo.collections.tts.g2p.models.base import BaseG2p
from nemo.core.classes import common
from nemo.core.classes.common import (
    UnsafeTargetError,
    _is_target_allowed,
    add_allowed_target_prefixes,
    safe_instantiate,
)
from nemo.utils.decorators import experimental


class MockDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return index


def get_class_path(cls):
    return f"{cls.__module__}.{cls.__name__}"


class MockG2p(BaseG2p):
    def __call__(self, text: str) -> str:
        return text


class MockTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__(tokens=["a"])

    def encode(self, text: str) -> list[int]:
        return [0]


@experimental
class MockExperimentalModule(torch.nn.Module):
    """nn.Module wrapped by @experimental (wrapt), like asr's TransformerEncoder."""

    def __init__(self, value: int = 0):
        super().__init__()
        self.value = value


@pytest.mark.unit
@pytest.mark.parametrize(
    "config,expected_type",
    [
        ({"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1}, torch.nn.Linear),
        ({"_target_": get_class_path(MockDataset)}, MockDataset),
        ({"_target_": "torch.distributed.fsdp.MixedPrecisionPolicy"}, MixedPrecisionPolicy),
        ({"_target_": "lightning.pytorch.callbacks.ModelCheckpoint"}, ModelCheckpoint),
        ({"_target_": "lightning.pytorch.strategies.DDPStrategy"}, DDPStrategy),
    ],
)
def test_safe_instantiate_allows_approved_targets(config, expected_type):
    obj = safe_instantiate(DictConfig(config))
    assert isinstance(obj, expected_type)


@pytest.mark.unit
@pytest.mark.parametrize(
    "target,target_type",
    [
        ("nemo_text_processing.text_normalization.normalize.Normalizer", type("MockNormalizer", (), {})),
        ("nemo.collections.tts.torch.g2ps.EnglishG2p", MockG2p),
        ("nemo.collections.tts.torch.tts_tokenizers.EnglishPhonemesTokenizer", MockTokenizer),
    ],
)
def test_safe_instantiate_allows_exact_exception_targets(target, target_type):
    sentinel = object()
    config = DictConfig({"_target_": target})

    with (
        patch("hydra.utils.get_class", return_value=target_type),
        patch("hydra.utils.instantiate", return_value=sentinel) as instantiate_mock,
    ):
        obj = safe_instantiate(config)

    assert obj is sentinel
    instantiate_mock.assert_called_once_with(config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "target",
    [
        "subprocess.Popen",
        "builtins.open",
        "os.system",
    ],
)
def test_safe_instantiate_blocks_unsafe_targets_before_hydra(target):
    config = DictConfig({"_target_": target})
    with patch("hydra.utils.instantiate") as instantiate_mock:
        with pytest.raises(ValueError, match=f"Instantiation of unsafe target '{target}' is blocked"):
            safe_instantiate(config)

    instantiate_mock.assert_not_called()


@pytest.mark.unit
def test_safe_instantiate_validates_nested_targets_before_hydra():
    config = DictConfig(
        {
            "_target_": "torch.nn.ModuleList",
            "modules": [
                {"_target_": "torch.nn.Linear", "in_features": 1, "out_features": 1},
                {"_target_": "subprocess.Popen"},
            ],
        }
    )

    with patch("hydra.utils.instantiate") as instantiate_mock:
        with pytest.raises(ValueError, match="Instantiation of unsafe target 'subprocess.Popen' is blocked"):
            safe_instantiate(config)

    instantiate_mock.assert_not_called()


@pytest.mark.unit
def test_safe_instantiate_allows_wrapt_decorated_module():
    """Regression: a wrapt-decorated (@experimental) nn.Module must not be blocked."""
    target = get_class_path(MockExperimentalModule)

    assert _is_target_allowed(target) is True

    obj = safe_instantiate(DictConfig({"_target_": target, "value": 7}))
    assert isinstance(obj, torch.nn.Module)
    assert obj.value == 7


@pytest.mark.unit
def test_safe_instantiate_allows_experimental_asr_transformer_encoder():
    """The originally reported target: an @experimental nn.Module in asr.modules."""
    pytest.importorskip("nemo.collections.asr.modules.transformer_encoder")

    assert _is_target_allowed("nemo.collections.asr.modules.transformer_encoder.TransformerEncoder") is True
    assert _is_target_allowed("nemo.collections.asr.modules.TransformerEncoder") is True


@pytest.fixture
def external_package(tmp_path, monkeypatch):
    """Two packages outside NeMo's allow-list, importable for the test: `trusted_blocks` and `trusted_blocks_other`.

    `trusted_blocks` defines an `nn.Module`, a function returning one and a plain class in `trusted_blocks.layers`,
    re-exports the module and the function at the top level, and imports `os` and `torch` like any real package.
    No prefix is added when the fixture starts; the test adds what it needs.
    """
    (tmp_path / "trusted_blocks").mkdir()
    (tmp_path / "trusted_blocks" / "__init__.py").write_text(
        textwrap.dedent(
            """
            import os
            import torch

            from trusted_blocks.layers import Scale, scale
            """
        )
    )
    (tmp_path / "trusted_blocks" / "layers.py").write_text(
        textwrap.dedent(
            """
            import torch


            class Scale(torch.nn.Module):
                def __init__(self, factor: float = 2.0):
                    super().__init__()
                    self.factor = factor

                def forward(self, x):
                    return x * self.factor


            def scale(factor: float = 2.0):
                return Scale(factor)


            class Plain:
                pass
            """
        )
    )
    (tmp_path / "trusted_blocks_other.py").write_text(
        textwrap.dedent(
            """
            import torch


            class Scale(torch.nn.Module):
                pass
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(common, "_ADDED_TARGET_PREFIXES", set())
    monkeypatch.delenv(common.ALLOWED_TARGET_PREFIXES_ENV_VAR, raising=False)
    yield "trusted_blocks"
    for name in [name for name in sys.modules if name.startswith("trusted_blocks")]:
        del sys.modules[name]


@pytest.mark.unit
def test_external_target_is_blocked_by_default(external_package):
    assert _is_target_allowed("trusted_blocks.Scale") is False
    assert _is_target_allowed("trusted_blocks.layers.scale") is False

    with pytest.raises(UnsafeTargetError, match="NEMO_ALLOWED_TARGET_PREFIXES") as excinfo:
        safe_instantiate(DictConfig({"_target_": "trusted_blocks.Scale"}))
    assert "Instantiation of unsafe target 'trusted_blocks.Scale' is blocked" in str(excinfo.value)
    assert "add_allowed_target_prefixes" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "target",
    [
        "trusted_blocks.Scale",
        "trusted_blocks.layers.Scale",
        "trusted_blocks.scale",
        "trusted_blocks.layers.scale",
    ],
)
def test_added_prefix_allows_classes_and_functions(external_package, target):
    add_allowed_target_prefixes(external_package)

    assert _is_target_allowed(target) is True
    obj = safe_instantiate(DictConfig({"_target_": target, "factor": 3.0}))
    assert type(obj).__name__ == "Scale"
    assert obj.factor == 3.0


@pytest.mark.unit
def test_added_prefix_allows_a_partial_section(external_package):
    """The form a parent module builds a configurable block from: `_partial_: true`, the parent fills the rest."""
    add_allowed_target_prefixes("trusted_blocks.")

    partial = safe_instantiate(DictConfig({"_target_": "trusted_blocks.Scale", "_partial_": True}))
    assert isinstance(partial, functools.partial)
    assert partial(factor=5.0).factor == 5.0


@pytest.mark.unit
def test_added_prefix_allows_plain_classes_defined_in_the_package(external_package):
    add_allowed_target_prefixes(external_package)

    assert _is_target_allowed("trusted_blocks.layers.Plain") is True


@pytest.mark.unit
def test_added_prefix_can_be_a_submodule(external_package):
    add_allowed_target_prefixes("trusted_blocks.layers")

    assert _is_target_allowed("trusted_blocks.layers.Scale") is True
    # the target itself must lie under the prefix, not only the object it resolves to
    assert _is_target_allowed("trusted_blocks.Scale") is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "target",
    [
        "trusted_blocks.os.system",
        "trusted_blocks.torch.nn.Linear",
        "trusted_blocks.layers.torch.save",
        "trusted_blocks",
        "trusted_blocks.layers",
        "trusted_blocks.missing",
        "trusted_blocks_other.Scale",
    ],
)
def test_added_prefix_does_not_leak_out_of_the_package(external_package, target):
    """An attribute path out of the package, a module, a missing name and a sibling package stay blocked."""
    add_allowed_target_prefixes(external_package)

    assert _is_target_allowed(target) is False
    with pytest.raises(UnsafeTargetError):
        safe_instantiate(DictConfig({"_target_": target}))


@pytest.mark.unit
def test_added_prefixes_from_the_environment(external_package, monkeypatch):
    monkeypatch.setenv(common.ALLOWED_TARGET_PREFIXES_ENV_VAR, " some_other_package, trusted_blocks. ")

    assert _is_target_allowed("trusted_blocks.Scale") is True
    assert _is_target_allowed("trusted_blocks_other.Scale") is False


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["", " ", ".", "my-blocks", "my..blocks", "os.system()"])
def test_add_allowed_target_prefixes_rejects_invalid_prefixes(external_package, prefix):
    with pytest.raises(ValueError, match="Invalid module prefix"):
        add_allowed_target_prefixes(prefix)
    assert not common._ADDED_TARGET_PREFIXES


@pytest.mark.unit
def test_invalid_prefix_in_the_environment_fails_the_check(external_package, monkeypatch):
    monkeypatch.setenv(common.ALLOWED_TARGET_PREFIXES_ENV_VAR, "my-blocks")

    with pytest.raises(ValueError, match="Invalid module prefix"):
        _is_target_allowed("trusted_blocks.Scale")


@pytest.mark.unit
def test_added_prefix_leaves_the_builtin_allow_list_unchanged(external_package):
    add_allowed_target_prefixes(external_package)

    assert _is_target_allowed("torch.nn.Linear") is True
    assert _is_target_allowed("subprocess.Popen") is False
