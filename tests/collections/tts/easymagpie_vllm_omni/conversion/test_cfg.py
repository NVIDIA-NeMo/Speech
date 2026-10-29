# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CFG artifact and request validation without serving dependencies."""
from types import SimpleNamespace

import convert_to_vllm as converter
import pytest
import torch
from easymagpie_vllm_omni.cfg import AcousticSamplingSettings, validate_cfg_context_embedding
from easymagpie_vllm_omni.config import EasyMagpieOmniArch


@pytest.mark.parametrize("disable_cas", [False, True])
def test_cfg_context_embedding_uses_exact_one_token_context_semantics(disable_cas):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.cfg = SimpleNamespace(embedding_dim=3)
            self.cfg_unk_token_id = 12
            self.disable_cas_for_context_text = disable_cas

        def embed_text_tokens(self, ids, text_lens, disable_cas_embedding):
            assert ids.tolist() == [[12]]
            assert text_lens.tolist() == [1]
            assert disable_cas_embedding is disable_cas
            return torch.full((1, 1, 3), 2.0 if disable_cas_embedding else 7.0)

    embedding = converter.precompute_cfg_context_embedding(Model())
    torch.testing.assert_close(embedding, torch.full((1, 3), 2.0 if disable_cas else 7.0))


@pytest.mark.parametrize(
    "tensor",
    [torch.ones(3), torch.ones(2, 3), torch.ones(1, 3, dtype=torch.long), torch.full((1, 3), float("nan"))],
)
def test_cfg_embedding_rejects_malformed_weights(tensor):
    with pytest.raises(ValueError, match="cfg_context_embedding"):
        validate_cfg_context_embedding(tensor, 3)


@pytest.mark.parametrize(
    "capability,token_id",
    [
        (True, 12),
        ({"version": True, "context_cas_disabled": False}, 12),
        ({"version": 2, "context_cas_disabled": False}, 12),
        ({"version": 1}, 12),
        ({"version": 1, "context_cas_disabled": 0}, 12),
        ({"version": 1, "context_cas_disabled": False}, None),
        ({"version": 1, "context_cas_disabled": False}, 32),
    ],
)
def test_cfg_metadata_is_strict_and_versioned(capability, token_id):
    arch = _arch(cfg_capability=capability, cfg_unk_token_id=token_id)
    with pytest.raises(ValueError, match="CFG capability"):
        arch.validate(text_vocab_size=32)


def test_old_artifact_and_new_cfg_artifact_are_valid():
    _arch().validate(text_vocab_size=32)
    _arch(cfg_capability={"version": 1, "context_cas_disabled": False}, cfg_unk_token_id=12).validate(
        text_vocab_size=32
    )


def test_hf_config_preserves_cfg_capability():
    arch = _arch(cfg_capability={"version": 1, "context_cas_disabled": True}, cfg_unk_token_id=12)
    restored = EasyMagpieOmniArch.from_hf_config(SimpleNamespace(**vars(arch), text_vocab_size=32))
    assert restored.cfg_capability == arch.cfg_capability
    assert restored.cfg_unk_token_id == 12


@pytest.mark.parametrize(
    "info",
    [
        {"use_cfg": "true"},
        {"use_cfg": 1},
        {"cfg_scale": float("inf")},
        {"cfg_scale": True},
        {"temperature": float("nan")},
        {"temperature": "0"},
        {"top_k": 0},
        {"top_k": 2.5},
        {"top_k": True},
    ],
)
def test_request_rejects_invalid_settings(info):
    with pytest.raises(ValueError):
        AcousticSamplingSettings.from_info(info)


def test_request_defaults_and_scale_identities():
    assert AcousticSamplingSettings.from_info({}) == AcousticSamplingSettings()
    for scale in (0, 1, 2.5):
        settings = AcousticSamplingSettings.from_info({"use_cfg": True, "cfg_scale": scale, "temperature": 0})
        assert settings.use_cfg and settings.cfg_scale == scale and settings.temperature == 0


def _arch(**kwargs):
    return EasyMagpieOmniArch(
        acoustic_predictor_type="acoustic_codes_predictor",
        use_codec_latent_audio_embedding=True,
        codec_num_levels_per_group=[8, 8, 16],
        **kwargs,
    )
