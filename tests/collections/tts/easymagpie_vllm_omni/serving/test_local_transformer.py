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
"""Tests for local-transformer sampling contracts."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from conftest import build_vllm_config  # noqa: E402
from easymagpie_vllm_omni.config import EasyMagpieOmniArch  # noqa: E402
from easymagpie_vllm_omni.local_transformer import EasyMagpieCodePredictor  # noqa: E402

# Cover identity and linear projection paths.
ARCH_PROFILES = {
    "equal_dims": dict(
        hidden_dim=64,
        embedding_dim=64,
        audio_embedding_dim=64,
        local_transformer_hidden_dim=64,
        local_transformer_n_heads=4,
    ),
    "mixed_dims": dict(
        hidden_dim=64,
        embedding_dim=64,
        audio_embedding_dim=48,
        local_transformer_hidden_dim=80,
        local_transformer_n_heads=4,
    ),
}


def _build_predictor(profile_kwargs: dict):
    """Build an initialized code predictor and its derived architecture."""
    cfg = build_vllm_config(**profile_kwargs)
    arch = EasyMagpieOmniArch.from_hf_config(cfg.model_config.hf_config)

    cp = EasyMagpieCodePredictor(vllm_config=cfg, prefix="code_predictor").eval()
    cp.init_forbidden_mask()
    return cp, arch


@pytest.mark.unit
def test_generate_codes_shape_dtype_and_range():
    """``generate_codes`` returns valid (num_tokens, num_codebooks) int64 codes within vocab."""
    cp, arch = _build_predictor(ARCH_PROFILES["equal_dims"])
    num_tokens = 5

    torch.manual_seed(0)
    codes = cp.generate_codes(torch.randn(num_tokens, arch.hidden_dim))

    assert codes.shape == (num_tokens, arch.num_stacked_codebooks)
    assert codes.dtype == torch.long
    assert codes.min().item() >= 0
    assert codes.max().item() < arch.num_all_tokens_per_codebook


@pytest.mark.unit
def test_generate_codes_respects_forbidden_mask():
    """With argmax sampling, forbidden special tokens are never emitted (only EOS stays reachable)."""
    cp, arch = _build_predictor(ARCH_PROFILES["equal_dims"])
    cp.temperature = 0.0  # argmax over masked logits

    torch.manual_seed(0)
    codes = cp.generate_codes(torch.randn(7, arch.hidden_dim))

    # Allowed = real codebook tokens [0, codebook_size) plus the audio EOS id.
    allowed = (codes < arch.codebook_size) | (codes == arch.audio_eos_id)
    assert allowed.all(), f"sampled forbidden tokens: {sorted(set(codes[~allowed].tolist()))}"


@pytest.mark.unit
def test_generate_codes_deterministic_with_seed():
    """Same seed + same input ⇒ identical sampled codes (sampler is RNG-driven, no host state)."""
    cp, arch = _build_predictor(ARCH_PROFILES["equal_dims"])
    dec_hidden = torch.randn(4, arch.hidden_dim)

    torch.manual_seed(7)
    first = cp.generate_codes(dec_hidden)
    torch.manual_seed(7)
    second = cp.generate_codes(dec_hidden)

    assert torch.equal(first, second)
