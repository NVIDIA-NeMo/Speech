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
"""Focused tests for the NemotronTTS v2 architecture adapter."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from easymagpie_vllm_omni.acoustic_predictor import CodecLatentAudioEmbedder  # noqa: E402
from easymagpie_vllm_omni.config import EasyMagpieOmniArch  # noqa: E402


def _synthetic_predictor(logits):
    from easymagpie_vllm_omni.acoustic_predictor import EasyMagpieAcousticCodesPredictor

    class Block(torch.nn.Module):
        num_codes = 1
        codebook_indices = (0,)

        def forward(self, hidden, positions):
            return hidden

        def compute_logits(self, hidden):
            return logits

    predictor = EasyMagpieAcousticCodesPredictor.__new__(EasyMagpieAcousticCodesPredictor)
    torch.nn.Module.__init__(predictor)
    predictor.arch = _v2_arch()
    predictor.num_codebooks = 1
    predictor.num_output_tokens = 33
    predictor.blocks = torch.nn.ModuleList([Block()])
    predictor.temperature = 0.7
    predictor.top_k = 80
    predictor._sample_top_k = 33
    predictor._gumbel_buf = torch.zeros(logits.shape[0], 1, 33)
    return predictor


def test_zero_temperature_is_greedy_without_random_draws(monkeypatch):
    logits = torch.zeros(2, 1, 33)
    logits[:, :, 7] = 0.000001
    predictor = _synthetic_predictor(logits)
    predictor.temperature = 0.0

    def unexpected_noise(*args, **kwargs):
        raise AssertionError("greedy acoustic decoding must not consume random draws")

    monkeypatch.setattr(torch.Tensor, "uniform_", unexpected_noise)
    codes = predictor.generate_codes(torch.zeros(2, 4), positions=torch.arange(2))
    assert codes.tolist() == [[7], [7]]


def test_request_top_k_changes_cutoff():
    logits = torch.zeros(64, 1, 33)
    logits[:, :, 7] = 0.001
    predictor = _synthetic_predictor(logits)
    predictor.top_k = 1
    codes = predictor.generate_codes(torch.zeros(64, 4), positions=torch.arange(64))
    assert codes.eq(7).all()


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
