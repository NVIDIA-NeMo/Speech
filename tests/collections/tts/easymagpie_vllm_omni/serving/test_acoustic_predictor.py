# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the NemotronTTS v2 architecture adapter."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from easymagpie_vllm_omni.acoustic_predictor import CodecLatentAudioEmbedder  # noqa: E402
from easymagpie_vllm_omni.config import EasyMagpieOmniArch  # noqa: E402


def _v2_arch(**overrides) -> EasyMagpieOmniArch:
    values = dict(
        hidden_dim=4,
        embedding_dim=4,
        audio_embedding_dim=4,
        num_audio_codebooks=2,
        codebook_size=32,
        frame_stacking_factor=1,
        acoustic_predictor_type="acoustic_codes_predictor",
        use_codec_latent_audio_embedding=True,
        codec_num_levels_per_group=[2, 2, 2, 2, 2],
        phoneme_stacking_factor=0,
        phoneme_vocab_size=0,
    )
    values.update(overrides)
    arch = EasyMagpieOmniArch(**values)
    arch.validate()
    return arch


@pytest.mark.unit
def test_codec_latent_audio_embedder_matches_fsq_projection_and_special_rows():
    arch = _v2_arch()
    embedder = CodecLatentAudioEmbedder(arch)
    with torch.no_grad():
        embedder.audio_code_projection.weight.copy_(
            torch.arange(arch.embedding_dim * 10, dtype=torch.float32).reshape(arch.embedding_dim, 10) / 10
        )
        embedder.audio_code_projection.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        embedder.audio_special_embeddings.weight.zero_()
        embedder.audio_special_embeddings.weight[1].copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    regular_codes = torch.tensor([[0, 31]])
    regular_latent = embedder._decode(regular_codes).flatten(start_dim=1)
    expected_regular = torch.nn.functional.linear(
        regular_latent,
        embedder.audio_code_projection.weight,
        embedder.audio_code_projection.bias,
    )
    torch.testing.assert_close(embedder.embed_audio_frame(regular_codes), expected_regular)

    special_codes = torch.tensor([[arch.audio_eos_id, 31]])
    special_latent = embedder._decode(special_codes.clamp(max=arch.codebook_size - 1))
    special_latent[:, 0].zero_()
    expected_special = (
        torch.nn.functional.linear(
            special_latent.flatten(start_dim=1),
            embedder.audio_code_projection.weight,
            embedder.audio_code_projection.bias,
        )
        + embedder.audio_special_embeddings.weight[1]
    )
    torch.testing.assert_close(embedder.embed_audio_frame(special_codes), expected_special)
