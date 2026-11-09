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


from unittest.mock import Mock

import pytest
import torch
from omegaconf import DictConfig

import nemo.collections.tts.models.audio_codec as audio_codec_module
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


@pytest.fixture(params=[True, False, None], ids=['frozen', 'trainable', 'default'])
def freeze_test_model(request):
    # Small modules exercise dropout and mutable BatchNorm buffers without audio data or pretrained weights.
    cfg = DictConfig(
        {
            'sample_rate': 8000,
            'samples_per_frame': 1,
            'loss_resolutions': [[16, 4, 16]],
            'mel_loss_dims': [4],
            'commit_loss_scale': 0.0,
            'audio_encoder': {
                '_target_': 'torch.nn.Sequential',
                '_args_': [
                    {'_target_': 'torch.nn.Linear', 'in_features': 4, 'out_features': 4},
                    {'_target_': 'torch.nn.BatchNorm1d', 'num_features': 4},
                    {'_target_': 'torch.nn.Dropout', 'p': 0.5},
                ],
            },
            'audio_decoder': {'_target_': 'torch.nn.Linear', 'in_features': 4, 'out_features': 4},
            'generator_loss': {'_target_': 'nemo.collections.tts.losses.audio_codec_loss.GeneratorSquaredLoss'},
            'discriminator_loss': {
                '_target_': 'nemo.collections.tts.losses.audio_codec_loss.DiscriminatorSquaredLoss'
            },
            'optim': {'_target_': 'torch.optim.Adam', 'lr': 0.01},
        }
    )
    if request.param is not None:
        cfg.freeze_encoder = request.param
    torch.manual_seed(42)
    return AudioCodecModel(cfg=cfg), request.param is True


@pytest.mark.unit
def test_freeze_encoder_modes(freeze_test_model):
    model, frozen = freeze_test_model
    # Freezing must take effect at construction, before optimizer setup.
    assert all(p.requires_grad == (not frozen) for p in model.audio_encoder.parameters())
    assert all(m.training == (not frozen) for m in model.audio_encoder.modules())
    inputs = torch.randn(8, 4)
    buffers_before = {name: buf.clone() for name, buf in model.audio_encoder.named_buffers()}

    for mode in (True, False, True, False, True):
        assert model.train(mode) is model
        assert model.training == mode
        assert model.audio_decoder.training == mode
        assert all(m.training == (mode and not frozen) for m in model.audio_encoder.modules())
        first = model.audio_encoder(inputs)
        second = model.audio_encoder(inputs)
        if frozen or not mode:
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    for name, buf in model.audio_encoder.named_buffers():
        if frozen:
            torch.testing.assert_close(buf, buffers_before[name], rtol=0, atol=0)
    if not frozen:
        assert model.audio_encoder[1].num_batches_tracked.item() == 6


@pytest.mark.unit
def test_freeze_encoder_optimizer_step(freeze_test_model):
    model, frozen = freeze_test_model
    optimizer = model.configure_optimizers()
    optimizer_params = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all((id(p) in optimizer_params) == (not frozen) for p in model.audio_encoder.parameters())
    assert all(id(p) in optimizer_params for p in model.audio_decoder.parameters())

    model.eval().train()
    encoder_before = {name: value.clone() for name, value in model.audio_encoder.state_dict().items()}
    decoder_before = {name: value.clone() for name, value in model.audio_decoder.named_parameters()}
    inputs, target = torch.randn(8, 4), torch.randn(8, 4)
    optimizer.zero_grad()
    prediction = model.audio_decoder(model.audio_encoder(inputs))
    (prediction - target).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.audio_decoder.parameters())
    if frozen:
        assert all(p.grad is None for p in model.audio_encoder.parameters())
    else:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.audio_encoder.parameters())
    optimizer.step()

    if frozen:
        for name, value in model.audio_encoder.state_dict().items():
            torch.testing.assert_close(value, encoder_before[name], rtol=0, atol=0)
    else:
        assert any(not torch.equal(p, encoder_before[name]) for name, p in model.audio_encoder.named_parameters())
    assert any(not torch.equal(p, decoder_before[name]) for name, p in model.audio_decoder.named_parameters())


class TestAudioCodecModel:
    @pytest.mark.unit
    def test_speaker_encoder_checkpoint_uses_hf_cache(self, monkeypatch):
        model_cfg = create_codec_config()
        model_cfg.use_scl_loss = True
        checkpoint_path = '/cache/Edresson/Speaker_Encoder_H_ASP/pytorch_model.bin'
        mock_hf_hub_download = Mock(return_value=checkpoint_path)
        speaker_encoder = torch.nn.Module()
        speaker_encoder.audio_config = {'sample_rate': model_cfg.sample_rate}
        speaker_encoder.load_checkpoint = Mock()
        speaker_encoder.freeze = Mock()
        monkeypatch.setattr(audio_codec_module, 'hf_hub_download', mock_hf_hub_download, raising=False)
        monkeypatch.setattr(audio_codec_module, 'ResNetSpeakerEncoder', Mock(return_value=speaker_encoder))

        AudioCodecModel(cfg=model_cfg)

        mock_hf_hub_download.assert_called_once_with(
            repo_id='Edresson/Speaker_Encoder_H_ASP', filename='pytorch_model.bin'
        )
        speaker_encoder.load_checkpoint.assert_called_once_with(checkpoint_path, strict=False)

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

    @pytest.mark.unit
    def test_encode_with_dropout(self, codec_model):
        batch_size = 4
        audio = torch.randn(size=(batch_size, 20000))
        audio_len = torch.randint(size=[batch_size], low=10000, high=20000)

        tokens, tokens_len = codec_model.encode(
            audio=audio,
            audio_len=audio_len,
            sample_rate=codec_model.sample_rate,
        )

        # Setting num_codebooks to the total number of codebooks should not modify the output
        tokens_without_dropout, _ = codec_model.encode(
            audio=audio,
            audio_len=audio_len,
            sample_rate=codec_model.sample_rate,
            num_codebooks=codec_model.num_codebooks,
        )
        torch.testing.assert_close(actual=tokens_without_dropout, expected=tokens)

        for i in range(1, codec_model.num_codebooks):
            tokens_with_dropout, _ = codec_model.encode(
                audio=audio, audio_len=audio_len, sample_rate=codec_model.sample_rate, num_codebooks=i
            )
            encoded = codec_model.dequantize(tokens=tokens_with_dropout, tokens_len=tokens_len)  # (B, D, T)
            # Validate appropriate dimensions are zero
            dropout_start_i = i * codec_model.vector_quantizer.codebook_dim
            dropped_codes = encoded[:, dropout_start_i:, :]
            torch.testing.assert_close(actual=dropped_codes, expected=torch.zeros_like(dropped_codes))

    @pytest.mark.unit
    def test_process_batch_gates_mmd_gradients_with_codebook_dropout(self, codec_model, monkeypatch):
        batch_size = 2
        num_frames = 2
        audio = torch.zeros(batch_size, 960)
        audio_len = torch.full((batch_size,), 960, dtype=torch.long)
        encoded_len = torch.full((batch_size,), num_frames, dtype=torch.long)
        encoder_output = torch.zeros(batch_size, 40, num_frames)
        full_codes = torch.ones_like(encoder_output, requires_grad=True)
        keep_mask = torch.ones_like(full_codes)
        keep_mask[:, full_codes.shape[1] // 2 :] = 0
        dropped_codes = full_codes * keep_mask

        # Stub each codec stage so full MMD values can be distinguished from dropped decoder inputs.
        monkeypatch.setattr(codec_model, 'encode_audio', Mock(return_value=(encoder_output, encoded_len)))
        monkeypatch.setattr(
            codec_model.vector_quantizer,
            'forward',
            Mock(return_value=(full_codes, torch.zeros(codec_model.num_codebooks, batch_size, num_frames))),
        )
        dropout_codebooks = Mock(return_value=dropped_codes)
        monkeypatch.setattr(codec_model, '_dropout_random_codebooks', dropout_codebooks)
        decoder_forward = Mock(return_value=(torch.zeros_like(audio), audio_len))
        monkeypatch.setattr(codec_model.audio_decoder, 'forward', decoder_forward)

        codec_model.codebook_dropout_rate = 1.0
        codec_model.train()
        batch = {'audio': audio, 'audio_lens': audio_len}

        _, _, _, _, mmd_inputs, _, _ = codec_model._process_batch(batch)

        # MMD observes the complete representation, but its gradients follow the codebook dropout mask.
        torch.testing.assert_close(mmd_inputs, full_codes)
        torch.testing.assert_close(decoder_forward.call_args.kwargs['inputs'], dropped_codes)
        dropout_codebooks.assert_called_once()
        mmd_inputs.sum().backward()
        torch.testing.assert_close(full_codes.grad, keep_mask)

        decoder_forward.reset_mock()
        dropout_codebooks.reset_mock()
        full_codes.grad = None
        codec_model.eval()

        _, _, _, _, mmd_inputs, _, _ = codec_model._process_batch(batch)

        # Without training-time dropout, MMD retains the ordinary identity gradient.
        torch.testing.assert_close(mmd_inputs, full_codes)
        torch.testing.assert_close(decoder_forward.call_args.kwargs['inputs'], full_codes)
        dropout_codebooks.assert_not_called()
        mmd_inputs.sum().backward()
        torch.testing.assert_close(full_codes.grad, torch.ones_like(full_codes))
