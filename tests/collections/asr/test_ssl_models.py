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
from omegaconf import DictConfig

from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel, SpeechEncDecSelfSupervisedModel
from nemo.core.classes.common import typecheck


@pytest.fixture()
def ssl_model():
    preprocessor = {
        'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
        'params': dict({'pad_to': 16, 'dither': 0}),
    }

    model_defaults = {'enc_hidden': 32, 'dec_out': 128}

    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                },
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                },
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                },
            ],
        },
    }

    spec_augment = {
        '_target_': 'nemo.collections.asr.modules.MaskedPatchAugmentation',
        'freq_masks': 3,
        'freq_width': 20,
        'patch_size': 16,
        'mask_patches': 0.5,
    }

    loss_list_contr_mlm = {
        'contr': {
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoderReconstruction',
                'feat_in': model_defaults['enc_hidden'],
                'feat_hidden': 128,
                'feat_out': model_defaults['dec_out'],
                'stride_layers': 0,
                'non_stride_layers': 0,
                'stride_transpose': False,
            },
            'loss': {
                '_target_': 'nemo.collections.asr.losses.ContrastiveLoss',
                'in_dim': 64,
                'proj_dim': model_defaults['dec_out'],
                'combine_time_steps': 1,
                'quantized_targets': True,
                'codebook_size': 64,
                'sample_from_same_utterance_only': True,
                'sample_from_non_masked': False,
                'num_negatives': 3,
            },
        },
        'mlm': {
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                'feat_in': model_defaults['enc_hidden'],
                'num_classes': 4096,
            },
            'loss': {'_target_': 'nemo.collections.asr.losses.MLMLoss', 'combine_time_steps': 1},
            'targets_from_loss': "contr",
        },
    }

    modelConfig_contr_mlm = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'spec_augment': DictConfig(spec_augment),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'loss_list': DictConfig(loss_list_contr_mlm),
        }
    )
    ssl_model = SpeechEncDecSelfSupervisedModel(cfg=modelConfig_contr_mlm)
    return ssl_model


@pytest.fixture()
def denoise_mlm_ssl_model():

    model_defaults = {
        "subsampling_factor": 1,
        'enc_hidden': 32,
        'dec_out': 128,
        "sample_rate": 16000,
        "num_classes": 32,
        "num_books": 1,
        "code_dim": 16,
        "squeeze_single": False,
        "mask_position": "pre_conv",  # position to apply masking, before or after conv subsampling, choices in ['pre_conv', 'post_conv']
    }

    preprocessor = {
        "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
        "sample_rate": model_defaults["sample_rate"],
        "normalize": "per_feature",
        "window_size": 0.025,
        "window_stride": 0.01,
        "window": "hann",
        "features": 80,
        "n_fft": 512,
        "log": True,
        "frame_splicing": 1,
        "dither": 0.00001,
        "pad_to": 16,
        "pad_value": 0.0,
    }

    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': preprocessor["features"],
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                },
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                },
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                },
            ],
        },
    }

    spec_augment = {
        '_target_': 'nemo.collections.asr.modules.SpectrogramAugmentation',
        'freq_masks': 0,
        'time_masks': 0,
        'freq_width': 16,
        'time_width': 0.05,
    }

    masking = {
        "_target_": "nemo.collections.asr.modules.RandomBlockMasking",
        "block_size": 40,  # for pre_conv masking, 10ms per frame, 400ms per block with block_size=40
        "mask_prob": 0.01,  # for allow_overlap=True, this means the mask prob for each frame; otherwise it means the overall masked proportion
        "feat_in": preprocessor["features"],
        "freeze": True,
        "allow_overlap": True,
    }

    quantizer = {
        "_target_": "nemo.collections.asr.modules.RandomProjectionVectorQuantizer",
        "feat_in": preprocessor["features"],
        "code_dim": model_defaults["code_dim"],
        "num_books": model_defaults["num_books"],
        "num_classes": model_defaults["num_classes"],
        "dist_fn": "l2",  # choices=["l2", "cosine"]
        "freeze": True,
        "squeeze_single": model_defaults["squeeze_single"],
        "combine_time_steps": model_defaults["subsampling_factor"],  # conformer sub-sampling ratio
    }

    decoder = {
        "_target_": "nemo.collections.asr.modules.MultiSoftmaxDecoder",
        "feat_in": model_defaults["enc_hidden"],
        "num_classes": model_defaults["num_classes"],
        "num_decoders": model_defaults["num_books"],
        "squeeze_single": model_defaults["squeeze_single"],
        "use_bias": True,
    }

    loss = {
        "_target_": "nemo.collections.asr.losses.MultiMLMLoss",
        "combine_time_steps": model_defaults[
            "subsampling_factor"
        ],  # conformer sub-sampling ratio for 'pre_conv', 1 for 'post_conv'
        "mask_threshold": 0.8,
        "num_decoders": model_defaults["num_books"],
        "squeeze_single": model_defaults["squeeze_single"],
    }

    optim = {
        "name": "adamw",
        "lr": 5.0,
        # optimizer arguments
        "betas": [0.9, 0.98],
        "weight_decay": 1e-3,
    }

    model_config = DictConfig(
        {
            "preprocessor": DictConfig(preprocessor),
            "spec_augment": DictConfig(spec_augment),
            'model_defaults': DictConfig(model_defaults),
            "masking": DictConfig(masking),
            "quantizer": DictConfig(quantizer),
            "encoder": DictConfig(encoder),
            "decoder": DictConfig(decoder),
            "loss": DictConfig(loss),
            "optim": DictConfig(optim),
        }
    )
    ssl_model = EncDecDenoiseMaskedTokenPredModel(cfg=model_config)
    return ssl_model


class TestSSLModel:
    @pytest.mark.unit
    def test_constructor(self, ssl_model):
        confdict = ssl_model.to_config_dict()
        instance2 = SpeechEncDecSelfSupervisedModel.from_config_dict(confdict)
        assert isinstance(instance2, SpeechEncDecSelfSupervisedModel)

    @pytest.mark.unit
    def test_contr_nonquant(self, ssl_model):
        modelConfig_contr_nonquant = ssl_model.to_config_dict()

        loss_list_contr_nonquant = dict(modelConfig_contr_nonquant['loss_list'])
        del loss_list_contr_nonquant['mlm']

        loss_list_contr_nonquant['contr']['loss']['quantized_targets'] = False

        modelConfig_contr_nonquant['loss_list'] = DictConfig(loss_list_contr_nonquant)

        ssl_model = SpeechEncDecSelfSupervisedModel(cfg=modelConfig_contr_nonquant)

        input_signal = torch.randn(size=(4, 64000))
        length = torch.randint(low=48000, high=64000, size=[4])

        with torch.no_grad():
            spectrograms, spec_masks, encoded, encoded_len = ssl_model.forward(
                input_signal=input_signal, input_signal_length=length
            )

            loss_value, loss_val_dict = ssl_model.decoder_loss_step(spectrograms, spec_masks, encoded, encoded_len)

        assert len(loss_val_dict) == 1

    @pytest.mark.unit
    def test_contr_mlm(self, ssl_model):

        input_signal = torch.randn(size=(4, 64000))
        length = torch.randint(low=48000, high=64000, size=[4])

        with torch.no_grad():
            spectrograms, spec_masks, encoded, encoded_len = ssl_model.forward(
                input_signal=input_signal, input_signal_length=length
            )

        loss_value, loss_val_dict = ssl_model.decoder_loss_step(spectrograms, spec_masks, encoded, encoded_len)

        assert len(loss_val_dict) == 2

    @pytest.mark.unit
    def test_contr_mlm_multi(self, ssl_model):
        modelConfig_contr_mlm_multi = ssl_model.to_config_dict()

        model_defaults = modelConfig_contr_mlm_multi['model_defaults']

        loss_list_contr_mlm_multi = dict(modelConfig_contr_mlm_multi['loss_list'])
        loss_list_contr_mlm_multi['mlm_2'] = {
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                'feat_in': model_defaults['enc_hidden'],
                'num_classes': 4096,
            },
            'loss': {'_target_': 'nemo.collections.asr.losses.MLMLoss', 'combine_time_steps': 1},
            'output_from_layer': "encoder.0",
            'targets_from_loss': "contr",
        }
        loss_list_contr_mlm_multi['mlm_3'] = {
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                'feat_in': model_defaults['enc_hidden'],
                'num_classes': 4096,
            },
            'loss': {'_target_': 'nemo.collections.asr.losses.MLMLoss', 'combine_time_steps': 1},
            'output_from_layer': "encoder.1",
            'targets_from_loss': "contr",
        }
        modelConfig_contr_mlm_multi['loss_list'] = DictConfig(loss_list_contr_mlm_multi)

        ssl_model = SpeechEncDecSelfSupervisedModel(cfg=modelConfig_contr_mlm_multi)

        input_signal = torch.randn(size=(4, 64000))
        length = torch.randint(low=48000, high=64000, size=[4])

        with torch.no_grad():
            spectrograms, spec_masks, encoded, encoded_len = ssl_model.forward(
                input_signal=input_signal, input_signal_length=length
            )

            loss_value, loss_val_dict = ssl_model.decoder_loss_step(spectrograms, spec_masks, encoded, encoded_len)

        assert len(loss_val_dict) == 4


class TestDenoiseMLMSSLModel:
    @pytest.mark.unit
    def test_forward(self, denoise_mlm_ssl_model):
        input_signal = torch.randn(size=(4, 64000))
        input_length = torch.randint(low=48000, high=64000, size=[4])
        noise = 0.1 * torch.ones_like(input_signal)
        noisy_input_signal = input_signal + noise
        noisy_input_length = input_length
        with torch.no_grad():
            with typecheck.disable_checks():
                log_probs, encoded_len, masks, tokens = denoise_mlm_ssl_model.forward(
                    input_signal=input_signal,
                    input_signal_length=input_length,
                    noisy_input_signal=noisy_input_signal,
                    noisy_input_signal_length=noisy_input_length,
                )

        assert log_probs.size(0) == 4
        assert log_probs.size(2) == denoise_mlm_ssl_model.cfg.model_defaults.num_classes
        assert encoded_len.size(0) == 4
        assert masks.size(0) == 4
        assert tokens.size(0) == 4
        assert masks.sum() == 0.0  # no mask should be applied to the input by default

    @pytest.mark.unit
    def test_forward_masked(self, denoise_mlm_ssl_model: EncDecDenoiseMaskedTokenPredModel):
        input_signal = torch.randn(size=(4, 64000))
        input_length = torch.randint(low=48000, high=64000, size=[4])
        noise = 0.1 * torch.ones_like(input_signal)
        noisy_input_signal = input_signal + noise
        noisy_input_length = input_length

        with torch.no_grad():
            with typecheck.disable_checks():
                log_probs, encoded_len, masks, tokens = denoise_mlm_ssl_model.forward(
                    input_signal=input_signal,
                    input_signal_length=input_length,
                    noisy_input_signal=noisy_input_signal,
                    noisy_input_signal_length=noisy_input_length,
                    apply_mask=True,
                )

        loss_value = denoise_mlm_ssl_model.loss(
            masks=masks, decoder_outputs=log_probs, targets=tokens, decoder_lengths=encoded_len
        )

        assert log_probs.size(0) == 4
        assert log_probs.size(2) == denoise_mlm_ssl_model.cfg.model_defaults.num_classes
        assert encoded_len.size(0) == 4
        assert masks.size(0) == 4
        assert tokens.size(0) == 4
        assert masks.sum() > 0.0  # mask should be applied to the input
        assert not torch.isnan(loss_value)
