# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Cross-environment numerical contract with the canonical Speech decoder."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from easymagpie_vllm_omni.codec.config import EasyMagpieCodecConfig
from easymagpie_vllm_omni.codec.packed import PackedEasyMagpieCodec
from safetensors.torch import load_file
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context

CONTRACT_ENV = "EASYMAGPIE_CODEC_CONTRACT"


def contract_dir() -> Path:
    value = os.environ.get(CONTRACT_ENV)
    if not value:
        pytest.skip(f"{CONTRACT_ENV} is set only by the cross-environment CI job")
    path = Path(value)
    required = (path / "config.json", path / "model.safetensors", path / "contract.safetensors")
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        pytest.fail(f"incomplete codec contract; missing {missing}")
    return path


def test_packed_decoder_matches_speech_contract() -> None:
    path = contract_dir()
    config = EasyMagpieCodecConfig.from_pretrained(path)
    weights = load_file(path / "model.safetensors", device="cpu")
    contract = load_file(path / "contract.safetensors", device="cpu")

    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        packed = PackedEasyMagpieCodec(config, dtype=torch.float32).eval()
    packed.load_state_dict(weights, strict=True)

    with torch.no_grad(), set_forward_context(None, vllm_config):
        actual = packed.audio_decoder(contract["latent"])

    expected = contract["expected_audio"]
    assert actual.shape == expected.shape
    assert actual.numel() == int(contract["expected_audio_len"].item())
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
