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

import torch
import torch.nn as nn
from lightning.pytorch.plugins import HalfPrecision
from omegaconf import DictConfig

from nemo.utils.trainer_utils import FlashPrecision, HalfPrecisionForAudio, resolve_trainer_cfg


class TestForwardContext:
    def test_default_dtype_remains_fp32_during_forward(self):
        plugin = FlashPrecision("bf16-flash")
        with plugin.forward_context():
            assert torch.get_default_dtype() == torch.float32

    def test_implicit_tensor_creation_is_fp32(self):
        plugin = FlashPrecision("bf16-flash")
        with plugin.forward_context():
            assert torch.zeros(10).dtype == torch.float32
            assert torch.ones(10).dtype == torch.float32
            assert torch.empty(10).dtype == torch.float32


class TestConvertModule:
    def test_convert_module_casts_plain_fp32_module(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
        plugin = FlashPrecision("bf16-flash")
        plugin.convert_module(model)
        assert model[0].weight.dtype == torch.bfloat16
        assert model[0].bias.dtype == torch.bfloat16
        assert model[1].weight.dtype == torch.bfloat16

    def test_convert_module_skips_models_with_existing_dtype_policy(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
        model[0].to(dtype=torch.bfloat16)
        plugin = FlashPrecision("bf16-flash")
        plugin.convert_module(model)
        assert model[0].weight.dtype == torch.bfloat16
        assert model[1].weight.dtype == torch.float32


class TestConvertInput:
    def test_preserves_audio_tensors(self):
        plugin = FlashPrecision("bf16-flash")
        batch = {"audio": torch.randn(1, 16000), "tokens": torch.randn(1, 10)}
        converted = plugin.convert_input(batch)
        assert converted["audio"].dtype == torch.float32
        assert converted["tokens"].dtype == torch.bfloat16

    def test_handles_nested_dicts(self):
        plugin = FlashPrecision("bf16-flash")
        batch = {
            "inputs": {"audio_signal": torch.randn(1, 16000), "text_ids": torch.randn(1, 10)},
            "labels": torch.randn(1, 5),
        }
        converted = plugin.convert_input(batch)
        assert converted["inputs"]["audio_signal"].dtype == torch.float32
        assert converted["inputs"]["text_ids"].dtype == torch.bfloat16
        assert converted["labels"].dtype == torch.bfloat16

    def test_non_dict_input_converted(self):
        plugin = FlashPrecision("bf16-flash")
        t = torch.randn(4, 8)
        converted = plugin.convert_input(t)
        assert converted.dtype == torch.bfloat16

    def test_non_float_tensors_unchanged(self):
        plugin = FlashPrecision("bf16-flash")
        batch = {"ids": torch.tensor([1, 2, 3], dtype=torch.long), "values": torch.randn(3)}
        converted = plugin.convert_input(batch)
        assert converted["ids"].dtype == torch.long
        assert converted["values"].dtype == torch.bfloat16


class TestHalfPrecisionRegression:
    def test_half_precision_does_change_default_dtype(self):
        plugin = HalfPrecision("bf16-true")
        with plugin.forward_context():
            assert torch.get_default_dtype() == torch.bfloat16
            assert torch.zeros(10).dtype == torch.bfloat16

        assert torch.get_default_dtype() == torch.float32


class TestResolveTrainerCfg:
    def test_bf16_flash_creates_flash_precision(self):
        cfg = DictConfig({"precision": "bf16-flash"})
        resolved = resolve_trainer_cfg(cfg)
        plugins = resolved["plugins"]
        assert "precision" not in resolved
        assert len(plugins) == 1
        assert isinstance(plugins[0], FlashPrecision)
        assert plugins[0].precision == "bf16-flash"
        assert plugins[0]._desired_input_dtype == torch.bfloat16

    def test_fp16_flash_creates_flash_precision(self):
        cfg = DictConfig({"precision": "fp16-flash"})
        resolved = resolve_trainer_cfg(cfg)
        plugins = resolved["plugins"]
        assert isinstance(plugins[0], FlashPrecision)
        assert plugins[0].precision == "fp16-flash"
        assert plugins[0]._desired_input_dtype == torch.float16

    def test_legacy_automodel_aliases_resolve_to_flash_precision(self):
        cfg = DictConfig({"precision": "bf16-automodel"})
        resolved = resolve_trainer_cfg(cfg)
        plugins = resolved["plugins"]
        assert isinstance(plugins[0], FlashPrecision)
        assert plugins[0].precision == "bf16-flash"

        cfg = DictConfig({"precision": "fp16-automodel"})
        resolved = resolve_trainer_cfg(cfg)
        plugins = resolved["plugins"]
        assert isinstance(plugins[0], FlashPrecision)
        assert plugins[0].precision == "fp16-flash"

    def test_bf16_true_still_creates_half_precision_for_audio(self):
        cfg = DictConfig({"precision": "bf16-true"})
        resolved = resolve_trainer_cfg(cfg)
        plugins = resolved["plugins"]
        assert len(plugins) == 1
        assert isinstance(plugins[0], HalfPrecisionForAudio)
