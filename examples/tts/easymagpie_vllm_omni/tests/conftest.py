# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Shared pytest fixtures for the EasyMagpieTTS vLLM-Omni tests.

The model definition (``easymagpie_vllm_omni.local_transformer`` etc.) is plain
PyTorch: the ``@support_torch_compile`` decorator short-circuits to eager when
``compilation_config.mode == CompilationMode.NONE``, and the modules only read a
handful of scalars off the ``VllmConfig``. So the whole stack can be exercised as
ordinary PyTorch with a tiny stand-in config — **no model directory, no engine,
no GPU required** — which is what these fixtures provide.

All heavy imports (torch / vllm) are done lazily inside the fixture so test
collection never fails on machines where those packages are absent; the
dependent tests ``importorskip`` them and are skipped instead.
"""
from __future__ import annotations

import types

import pytest

# A deliberately tiny architecture so the tests run fast on CPU. Dimensions are
# kept equal by default (so the in/out projections collapse to ``nn.Identity``,
# matching the reference SmallMamba checkpoint where everything is 1536-wide).
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
    """Build a minimal stand-in ``VllmConfig`` for the code predictor.

    Returns a ``types.SimpleNamespace`` exposing exactly the attributes the
    EasyMagpie modules touch at construction time:

    * ``model_config.hf_config`` — arch scalars (read via ``from_hf_config``);
    * ``model_config.dtype`` — buffer dtype;
    * ``scheduler_config.max_num_batched_tokens`` — scratch-buffer length;
    * ``compilation_config.mode`` — ``CompilationMode.NONE`` so the
      ``@support_torch_compile`` wrapper stays in eager mode.

    Any keyword overrides are merged into the default tiny arch profile.
    """
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
