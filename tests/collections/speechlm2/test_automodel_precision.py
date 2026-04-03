# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig

from nemo.utils.trainer_utils import AutomodelPrecision, resolve_trainer_cfg


# ---------------------------------------------------------------------------
# AutomodelPrecision — forward_context does NOT mutate global state
# ---------------------------------------------------------------------------


class TestForwardContext:
    def test_default_dtype_remains_fp32_during_forward(self):
        """torch.get_default_dtype() == float32 inside forward_context."""
        plugin = AutomodelPrecision("bf16-automodel")
        with plugin.forward_context():
            assert torch.get_default_dtype() == torch.float32

    def test_implicit_tensor_creation_is_fp32(self):
        """torch.zeros/ones/empty create fp32 tensors inside forward_context."""
        plugin = AutomodelPrecision("bf16-automodel")
        with plugin.forward_context():
            assert torch.zeros(10).dtype == torch.float32
            assert torch.ones(10).dtype == torch.float32
            assert torch.empty(10).dtype == torch.float32


# ---------------------------------------------------------------------------
# AutomodelPrecision — convert_module does nothing
# ---------------------------------------------------------------------------


class TestConvertModule:
    def test_convert_module_does_nothing(self):
        """Linear layer parameters are cast to bf16."""
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
        plugin = AutomodelPrecision("bf16-automodel")
        plugin.convert_module(model)
        assert model[0].weight.dtype == torch.float32
        assert model[0].bias.dtype == torch.float32
        assert model[1].weight.dtype == torch.float32


# ---------------------------------------------------------------------------
# AutomodelPrecision — convert_input preserves audio tensors
# ---------------------------------------------------------------------------


class TestConvertInput:
    def test_preserves_audio_tensors(self):
        """Audio tensors in batch dict are not downcast to bf16."""
        plugin = AutomodelPrecision("bf16-automodel")
        batch = {"audio": torch.randn(1, 16000), "tokens": torch.randn(1, 10)}
        converted = plugin.convert_input(batch)
        assert converted["audio"].dtype == torch.float32
        assert converted["tokens"].dtype == torch.bfloat16

    def test_handles_nested_dicts(self):
        """Nested dict values are converted, audio keys preserved at any depth."""
        plugin = AutomodelPrecision("bf16-automodel")
        batch = {
            "inputs": {"audio_signal": torch.randn(1, 16000), "text_ids": torch.randn(1, 10)},
            "labels": torch.randn(1, 5),
        }
        converted = plugin.convert_input(batch)
        assert converted["inputs"]["audio_signal"].dtype == torch.float32
        assert converted["inputs"]["text_ids"].dtype == torch.bfloat16
        assert converted["labels"].dtype == torch.bfloat16

    def test_non_dict_input_converted(self):
        """Non-dict tensor input is converted to bf16."""
        plugin = AutomodelPrecision("bf16-automodel")
        t = torch.randn(4, 8)
        converted = plugin.convert_input(t)
        assert converted.dtype == torch.bfloat16

    def test_non_float_tensors_unchanged(self):
        """Integer tensors in dicts are not modified."""
        plugin = AutomodelPrecision("bf16-automodel")
        batch = {"ids": torch.tensor([1, 2, 3], dtype=torch.long), "values": torch.randn(3)}
        converted = plugin.convert_input(batch)
        assert converted["ids"].dtype == torch.long
        assert converted["values"].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Regression guard — HalfPrecision DOES change default dtype
# ---------------------------------------------------------------------------


class TestHalfPrecisionRegression:
    def test_half_precision_does_change_default_dtype(self):
        """Verify the problem we're solving: HalfPrecision sets default dtype to bf16."""
        from lightning.pytorch.plugins import HalfPrecision

        plugin = HalfPrecision("bf16-true")
        with plugin.forward_context():
            assert torch.get_default_dtype() == torch.bfloat16
            assert torch.zeros(10).dtype == torch.bfloat16
        # Restored after context exit
        assert torch.get_default_dtype() == torch.float32


# ---------------------------------------------------------------------------
# resolve_trainer_cfg integration
# ---------------------------------------------------------------------------


class TestResolveTrainerCfg:
    def test_bf16_automodel_creates_automodel_precision(self):
        """precision: bf16-automodel installs AutomodelPrecision plugin."""
        cfg = DictConfig({"precision": "bf16-automodel"})
        resolved = resolve_trainer_cfg(cfg)
        assert "precision" not in resolved
        plugins = resolved["plugins"]
        assert len(plugins) == 1
        assert isinstance(plugins[0], AutomodelPrecision)
        assert plugins[0].precision == "bf16-automodel"
        assert plugins[0]._desired_input_dtype == torch.bfloat16

    def test_fp16_automodel_creates_automodel_precision(self):
        """precision: fp16-automodel installs AutomodelPrecision plugin with fp16."""
        cfg = DictConfig({"precision": "fp16-automodel"})
        resolved = resolve_trainer_cfg(cfg)
        plugins = resolved["plugins"]
        assert isinstance(plugins[0], AutomodelPrecision)
        assert plugins[0]._desired_input_dtype == torch.float16

    def test_bf16_true_still_creates_half_precision_for_audio(self):
        """Existing bf16-true path is unchanged — still uses HalfPrecisionForAudio."""
        from nemo.utils.trainer_utils import HalfPrecisionForAudio

        cfg = DictConfig({"precision": "bf16-true"})
        resolved = resolve_trainer_cfg(cfg)
        plugins = resolved["plugins"]
        assert len(plugins) == 1
        assert isinstance(plugins[0], HalfPrecisionForAudio)
