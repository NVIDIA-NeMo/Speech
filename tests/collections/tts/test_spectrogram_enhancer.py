# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from einops import rearrange
from omegaconf import DictConfig

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.collections.tts.models import SpectrogramEnhancerModel


@pytest.fixture
def enhancer_config():
    n_bands = 80
    latent_dim = 192
    style_depth = 4
    network_capacity = 16
    fmap_max = 192

    config = {
        "model": {
            "n_bands": n_bands,
            "latent_dim": latent_dim,
            "style_depth": style_depth,
            "network_capacity": network_capacity,
            "mixed_prob": 0.9,
            "fmap_max": fmap_max,
            "generator": {
                "_target_": "nemo.collections.tts.modules.spectrogram_enhancer.Generator",
                "n_bands": n_bands,
                "latent_dim": latent_dim,
                "network_capacity": network_capacity,
                "style_depth": style_depth,
                "fmap_max": fmap_max,
            },
            "discriminator": {
                "_target_": "nemo.collections.tts.modules.spectrogram_enhancer.Discriminator",
                "n_bands": n_bands,
                "network_capacity": network_capacity,
                "fmap_max": fmap_max,
            },
            "spectrogram_min_value": -13.18,
            "spectrogram_max_value": 4.78,
            "consistency_loss_weight": 10.0,
            "gradient_penalty_loss_weight": 10.0,
            "gradient_penalty_loss_every_n_steps": 4,
            "spectrogram_predictor_path": None,
        },
        "generator_opt": {"_target_": "torch.optim.Adam", "lr": 2e-4, "betas": [0.5, 0.9]},
        "discriminator_opt": {"_target_": "torch.optim.Adam", "lr": 2e-4, "betas": [0.5, 0.9]},
    }

    return DictConfig(config)


@pytest.fixture
def enhancer(enhancer_config):
    return SpectrogramEnhancerModel(cfg=enhancer_config.model)


@pytest.fixture
def enhancer_with_fastpitch(enhancer_config_with_fastpitch):
    return SpectrogramEnhancerModel(cfg=enhancer_config_with_fastpitch.model)


@pytest.fixture
def sample_input(batch_size=15, max_length=1000):
    generator = torch.Generator()
    generator.manual_seed(0)

    lengths = torch.randint(max_length // 4, max_length - 7, (batch_size,), generator=generator)
    input_spectrograms = torch.randn((batch_size, 80, 1000), generator=generator)

    input_spectrograms = mask_sequence_tensor(input_spectrograms, lengths)

    return input_spectrograms, lengths


@pytest.mark.unit
def test_pad_spectrograms(enhancer: SpectrogramEnhancerModel, sample_input):
    input_spectrograms, lengths = sample_input
    output = enhancer.pad_spectrograms(input_spectrograms)

    assert output.size(-1) >= input_spectrograms.size(-1)


@pytest.mark.unit
def test_spectrogram_norm_unnorm(enhancer: SpectrogramEnhancerModel, sample_input):
    input_spectrograms, lengths = sample_input
    same_input_spectrograms = enhancer.unnormalize_spectrograms(
        enhancer.normalize_spectrograms(input_spectrograms, lengths), lengths
    )
    assert torch.allclose(input_spectrograms, same_input_spectrograms, atol=1e-5)


@pytest.mark.unit
def test_spectrogram_unnorm_norm(enhancer: SpectrogramEnhancerModel, sample_input):
    input_spectrograms, lengths = sample_input
    same_input_spectrograms = enhancer.normalize_spectrograms(
        enhancer.unnormalize_spectrograms(input_spectrograms, lengths), lengths
    )
    assert torch.allclose(input_spectrograms, same_input_spectrograms, atol=1e-5)


@pytest.mark.unit
def test_spectrogram_norm_unnorm_dont_look_at_padding(enhancer: SpectrogramEnhancerModel, sample_input):
    input_spectrograms, lengths = sample_input
    same_input_spectrograms = enhancer.unnormalize_spectrograms(
        enhancer.normalize_spectrograms(input_spectrograms, lengths), lengths
    )
    for i, length in enumerate(lengths.tolist()):
        assert torch.allclose(input_spectrograms[i, :, :length], same_input_spectrograms[i, :, :length], atol=1e-5)


@pytest.mark.unit
def test_spectrogram_unnorm_norm_dont_look_at_padding(enhancer: SpectrogramEnhancerModel, sample_input):
    input_spectrograms, lengths = sample_input
    same_input_spectrograms = enhancer.normalize_spectrograms(
        enhancer.unnormalize_spectrograms(input_spectrograms, lengths), lengths
    )
    for i, length in enumerate(lengths.tolist()):
        assert torch.allclose(input_spectrograms[i, :, :length], same_input_spectrograms[i, :, :length], atol=1e-5)


@pytest.mark.unit
def test_generator_pass_keeps_size(enhancer: SpectrogramEnhancerModel, sample_input):
    input_spectrograms, lengths = sample_input
    output = enhancer.forward(input_spectrograms=input_spectrograms, lengths=lengths)

    assert output.shape == input_spectrograms.shape


@pytest.mark.unit
def test_discriminator_pass(enhancer: SpectrogramEnhancerModel, sample_input):
    input_spectrograms, lengths = sample_input
    input_spectrograms = rearrange(input_spectrograms, "b c l -> b 1 c l")
    logits = enhancer.discriminator(x=input_spectrograms, condition=input_spectrograms, lengths=lengths)

    assert logits.shape == lengths.shape


@pytest.mark.unit
def test_nemo_save_load(enhancer: SpectrogramEnhancerModel, tmp_path):
    path = tmp_path / "test-enhancer-save-load.nemo"

    enhancer.save_to(path)
    SpectrogramEnhancerModel.restore_from(path)
