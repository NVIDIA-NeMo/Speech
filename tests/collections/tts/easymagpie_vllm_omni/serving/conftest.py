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
"""Fixtures and dependency boundary for EasyMagpie vLLM serving tests."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SERVING_DEPENDENCIES = ("vllm", "vllm_omni")
_HAS_SERVING_DEPENDENCIES = all(importlib.util.find_spec(name) is not None for name in _SERVING_DEPENDENCIES)

# Ordinary Speech test environments do not install the pinned serving stack.
# The dedicated serving job provides both dependencies and collects this suite.
collect_ignore_glob = [] if _HAS_SERVING_DEPENDENCIES else ["test_*.py"]

EASYMAGPIE_ROOT = Path(__file__).resolve().parents[5] / "tools" / "easymagpie_vllm_omni"
SCRIPTS_DIR = EASYMAGPIE_ROOT / "scripts"

# Load the standalone serving package from this checkout without installing Speech.
sys.path.insert(0, str(EASYMAGPIE_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

# Equal dimensions exercise the identity projection path.
_DEFAULT_ARCH: dict = dict(
    hidden_dim=64,
    embedding_dim=64,
    audio_embedding_dim=64,
    num_audio_codebooks=2,
    codebook_size=32,
    frame_stacking_factor=2,
    local_transformer_n_layers=2,
    local_transformer_n_heads=4,
    local_transformer_hidden_dim=64,
)


def build_vllm_config(**arch_overrides):
    """Build a minimal eager ``VllmConfig`` for the code predictor."""
    import torch
    from vllm.config import CompilationMode

    arch = {**_DEFAULT_ARCH, **arch_overrides}
    hf_config = types.SimpleNamespace(**arch)
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_config=hf_config, dtype=torch.float32),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=128),
        compilation_config=types.SimpleNamespace(mode=CompilationMode.NONE),
    )


@pytest.fixture
def vllm_config_factory():
    """Fixture returning the :func:`build_vllm_config` factory."""
    return build_vllm_config
