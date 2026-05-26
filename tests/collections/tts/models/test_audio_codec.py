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


import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.tts.models import AudioCodecModel


def create_codec_config():
    audio_encoder = {
        'cls': 'nemo.collections.tts.modules.audio_codec_modules.MultiResolutionSTFTEncoder',
        'params': {
            'out_dim': 40,
            'resolutions': [[960, 240, 960], [1920, 480, 1920]],
            'resolution_filter_list': [256, 512],
        },
    }
    audio_decoder = {
        'cls': 'nemo.collections.tts.modules.audio_codec_modules.ResNetDecoder',
        'params': {
            'input_dim': 40,
            'input_filters': 512,
            'n_hidden_layers': 6,
            'hidden_filters': 512,
            'pre_up_sample_rates': [],
            'pre_up_sample_filters': [],
            'resblock_up_sample_rates': [10, 8, 6],
            'resblock_up_sample_filters': [256, 128, 32],
        },
    }
    vector_quantizer = {
        'cls': 'nemo.collections.tts.modules.audio_codec_modules.GroupFiniteScalarQuantizer',
        'params': {
            'num_groups': 8,
            'num_levels_per_group': [4, 4, 4, 4, 4],
        },
    }
    generator_loss = {
        'cls': 'nemo.collections.tts.losses.audio_codec_loss.GeneratorSquaredLoss',
    }
    discriminator_loss = {
        'cls': 'nemo.collections.tts.losses.audio_codec_loss.DiscriminatorSquaredLoss',
    }

    model_cfg = DictConfig(
        {
            'sample_rate': 24000,
            'samples_per_frame': 480,
            'loss_resolutions': [[960, 240, 960], [1920, 480, 1920]],
            'mel_loss_dims': [160, 320],
            'commit_loss_scale': 0.0,
            'audio_encoder': DictConfig(audio_encoder),
            'audio_decoder': DictConfig(audio_decoder),
            'vector_quantizer': DictConfig(vector_quantizer),
            'generator_loss': DictConfig(generator_loss),
            'discriminator_loss': DictConfig(discriminator_loss),
        }
    )
    return model_cfg


@pytest.fixture()
def codec_model():
    model_cfg = create_codec_config()
    codec_model = AudioCodecModel(cfg=model_cfg)
    return codec_model


@pytest.fixture()
def acoustic_codec_model():
    semantic_model_cfg = create_codec_config()
    semantic_model_cfg.vector_quantizer.params.num_groups = 1
    semantic_model_cfg.audio_encoder.params.out_dim = 5
    semantic_model_cfg.audio_decoder.params.input_dim = 5

    acoustic_model_cfg = create_codec_config()
    acoustic_model_cfg.semantic_codec = semantic_model_cfg
    acoustic_model_cfg.audio_encoder.params.out_dim = 35
    acoustic_codec_model = AudioCodecModel(cfg=acoustic_model_cfg)

    return acoustic_codec_model


class TestAudioCodecModel:
    @pytest.mark.unit
    def test_forward(self, codec_model):
        batch_size = 2
        audio = torch.randn(size=(batch_size, 20000))
        audio_len = torch.randint(size=[batch_size], low=10000, high=20000)
        output_audio, output_audio_len = codec_model.forward(
            audio=audio, audio_len=audio_len, sample_rate=codec_model.sample_rate
        )
        assert output_audio.shape[0] == batch_size
        assert output_audio.shape[1] == output_audio_len.max()

    @pytest.mark.unit
    def test_forward_with_acoustic_codec(self, acoustic_codec_model):
        batch_size = 3
        audio = torch.randn(size=(batch_size, 20000))
        audio_len = torch.randint(size=[batch_size], low=10000, high=20000)
        output_audio, output_audio_len = acoustic_codec_model.forward(
            audio=audio, audio_len=audio_len, sample_rate=acoustic_codec_model.sample_rate
        )
        assert output_audio.shape[0] == batch_size
        assert output_audio.shape[1] == output_audio_len.max()

    @pytest.mark.unit
    def test_encode_and_decode(self, codec_model):
        batch_size = 4
        audio = torch.randn(size=(batch_size, 20000))
        audio_len = torch.randint(size=[batch_size], low=10000, high=20000)

        tokens, tokens_len = codec_model.encode(audio=audio, audio_len=audio_len, sample_rate=codec_model.sample_rate)
        assert tokens.shape[0] == batch_size
        assert tokens.shape[2] == tokens_len.max()

        output_audio, output_audio_len = codec_model.decode(tokens=tokens, tokens_len=tokens_len)
        assert output_audio.shape[0] == batch_size
        assert output_audio.shape[1] == output_audio_len.max()

    @pytest.mark.unit
    def test_encode_and_decode_with_acoustic_codec(self, acoustic_codec_model):
        batch_size = 5
        audio = torch.randn(size=(batch_size, 20000))
        audio_len = torch.randint(size=[batch_size], low=10000, high=20000)

        tokens, tokens_len = acoustic_codec_model.encode(
            audio=audio, audio_len=audio_len, sample_rate=acoustic_codec_model.sample_rate
        )
        assert tokens.shape[0] == batch_size
        assert tokens.shape[2] == tokens_len.max()

        output_audio, output_audio_len = acoustic_codec_model.decode(tokens=tokens, tokens_len=tokens_len)
        assert output_audio.shape[0] == batch_size
        assert output_audio.shape[1] == output_audio_len.max()
