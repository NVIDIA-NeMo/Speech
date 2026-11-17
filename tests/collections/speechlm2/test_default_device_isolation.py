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

import importlib

import pytest
import torch

# Test modules that run on CUDA through the `cuda_default_device` fixture (see conftest.py).
CUDA_DEFAULT_MODULES = [
    "tests.collections.speechlm2.test_duplex_eartts",
    "tests.collections.speechlm2.test_duplex_s2s",
    "tests.collections.speechlm2.test_duplex_s2s_speech_decoder",
    "tests.collections.speechlm2.test_duplex_stt",
    "tests.collections.speechlm2.test_nemotron_voicechat",
    "tests.collections.speechlm2.test_salm",
    "tests.collections.speechlm2.test_salm_asr_decoder",
    "tests.collections.speechlm2.test_salm_asr_decoder_multilayerproj",
    "tests.collections.speechlm2.test_salm_automodel",
    "tests.collections.speechlm2.test_salm_automodel_lora",
]


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="The default device can only leak to CUDA when it is available"
)
def test_importing_cuda_test_modules_keeps_the_default_device():
    # Pytest imports every test module before running any test, so an import-time `torch.set_default_device`
    # changes the device of every test collected in the same session.
    before = torch.get_default_device()

    for module in CUDA_DEFAULT_MODULES:
        importlib.import_module(module)

    assert torch.get_default_device() == before
