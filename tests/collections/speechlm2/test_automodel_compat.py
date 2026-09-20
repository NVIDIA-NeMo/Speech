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

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

_COMPAT_PATH = Path(__file__).resolve().parents[3] / "nemo/collections/speechlm2/parts/automodel_compat.py"
_COMPAT_SPEC = importlib.util.spec_from_file_location("_speechlm2_automodel_compat", _COMPAT_PATH)
_COMPAT_MODULE = importlib.util.module_from_spec(_COMPAT_SPEC)
_COMPAT_SPEC.loader.exec_module(_COMPAT_MODULE)


def _install_fake_model_selection(monkeypatch, *, is_hf_model):
    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, model_path_or_name, **kwargs):
            return object()

    transformers = ModuleType("transformers")
    transformers.AutoConfig = FakeAutoConfig

    model_init = ModuleType("nemo_automodel._transformers.model_init")
    model_init.get_is_hf_model = lambda config, force_hf: force_hf or is_hf_model
    automodel_transformers = ModuleType("nemo_automodel._transformers")
    automodel_transformers.model_init = model_init
    automodel = ModuleType("nemo_automodel")
    automodel._transformers = automodel_transformers

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "nemo_automodel", automodel)
    monkeypatch.setitem(sys.modules, "nemo_automodel._transformers", automodel_transformers)
    monkeypatch.setitem(sys.modules, "nemo_automodel._transformers.model_init", model_init)


def test_hf_fallback_drops_automodel_backend(monkeypatch):
    _install_fake_model_selection(monkeypatch, is_hf_model=True)

    class Qwen3ForCausalLM:
        def __init__(self, config):
            self.config = config

    backend = object()
    kwargs = {"backend": backend}
    with pytest.raises(TypeError, match="unexpected keyword argument 'backend'"):
        Qwen3ForCausalLM(object(), **kwargs)

    assert _COMPAT_MODULE.remove_automodel_backend_for_hf_fallback("Qwen/Qwen3-1.7B", kwargs) is True
    assert "backend" not in kwargs
    Qwen3ForCausalLM(object(), **kwargs)


def test_native_automodel_preserves_backend(monkeypatch):
    _install_fake_model_selection(monkeypatch, is_hf_model=False)

    backend = object()
    kwargs = {"backend": backend}

    assert (
        _COMPAT_MODULE.remove_automodel_backend_for_hf_fallback("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", kwargs)
        is False
    )
    assert kwargs["backend"] is backend


def test_native_automodel_with_bnb_quantization_drops_backend(monkeypatch):
    _install_fake_model_selection(monkeypatch, is_hf_model=False)

    backend = object()
    quantization_config = object()
    kwargs = {"backend": backend, "quantization_config": quantization_config}

    assert _COMPAT_MODULE.remove_automodel_backend_for_hf_fallback("native-model", kwargs) is True
    assert "backend" not in kwargs
    assert kwargs["quantization_config"] is quantization_config


def test_model_selection_failure_preserves_backend(monkeypatch, caplog):
    _install_fake_model_selection(monkeypatch, is_hf_model=True)

    class BrokenAutoConfig:
        @classmethod
        def from_pretrained(cls, model_path_or_name, **kwargs):
            raise RuntimeError("config resolution failed")

    monkeypatch.setattr(sys.modules["transformers"], "AutoConfig", BrokenAutoConfig)
    backend = object()
    kwargs = {"backend": backend}

    with caplog.at_level(logging.WARNING):
        assert _COMPAT_MODULE.remove_automodel_backend_for_hf_fallback("unavailable-model", kwargs) is False

    assert kwargs["backend"] is backend
    assert "Could not determine Automodel implementation" in caplog.text


def test_installed_automodel_model_selection_accepts_force_hf_keyword():
    model_init = pytest.importorskip("nemo_automodel._transformers.model_init")

    assert model_init.get_is_hf_model(object(), force_hf=True) is True
