# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.audio.models import EncMaskDecAudioToAudioModel

try:
    importlib.import_module('torchaudio')

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False


@pytest.fixture()
def mask_model_rnn():

    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 512,
        'hop_length': 256,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
    }
    mask_estimator = {
        '_target_': 'nemo.collections.audio.modules.masking.MaskEstimatorRNN',
        'num_outputs': model['num_outputs'],
        'num_subbands': encoder['fft_length'] // 2 + 1,
        'num_features': 256,
        'num_layers': 3,
        'bidirectional': True,
    }
    mask_processor = {
        '_target_': 'nemo.collections.audio.modules.masking.MaskReferenceChannel',
        'ref_channel': 0,
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.SDRLoss',
        'scale_invariant': True,
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'mask_estimator': DictConfig(mask_estimator),
            'mask_processor': DictConfig(mask_processor),
            'loss': DictConfig(loss),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    model = EncMaskDecAudioToAudioModel(cfg=model_config)

    return model


@pytest.fixture()
def mask_model_flexarray():

    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 512,
        'hop_length': 256,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
    }
    mask_estimator = {
        '_target_': 'nemo.collections.audio.modules.masking.MaskEstimatorFlexChannels',
        'num_outputs': model['num_outputs'],
        'num_subbands': encoder['fft_length'] // 2 + 1,
        'num_blocks': 3,
        'channel_reduction_position': 3,
        'channel_reduction_type': 'average',
        'channel_block_type': 'transform_average_concatenate',
        'temporal_block_type': 'conformer_encoder',
        'temporal_block_num_layers': 5,
        'temporal_block_num_heads': 4,
        'temporal_block_dimension': 128,
        'mag_reduction': None,
        'mag_normalization': 'mean_var',
        'use_ipd': True,
        'ipd_normalization': 'mean',
    }
    mask_processor = {
        '_target_': 'nemo.collections.audio.modules.masking.MaskReferenceChannel',
        'ref_channel': 0,
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.SDRLoss',
        'scale_invariant': True,
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'mask_estimator': DictConfig(mask_estimator),
            'mask_processor': DictConfig(mask_processor),
            'loss': DictConfig(loss),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    model = EncMaskDecAudioToAudioModel(cfg=model_config)

    return model


@pytest.fixture()
def bf_model_flexarray(mask_model_flexarray):

    model_config = mask_model_flexarray.to_config_dict()
    # Switch processor to beamformer
    model_config['mask_processor'] = {
        '_target_': 'nemo.collections.audio.modules.masking.MaskBasedBeamformer',
        'filter_type': 'pmwf',
        'filter_beta': 0.0,
        'filter_rank': 'one',
        'ref_channel': 'max_snr',
        'ref_hard': 1,
        'ref_hard_use_grad': False,
        'ref_subband_weighting': False,
        'num_subbands': model_config['mask_estimator']['num_subbands'],
    }

    model = EncMaskDecAudioToAudioModel(cfg=model_config)

    return model


class TestMaskModelRNN:
    """Test masking model with RNN mask estimator."""

    @pytest.mark.unit
    def test_constructor(self, mask_model_rnn):
        """Test that the model can be constructed from a config dict."""
        model = mask_model_rnn.train()
        confdict = model.to_config_dict()
        instance2 = EncMaskDecAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, EncMaskDecAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),  # Example 1
            (2, 8),  # Example 2
            (1, 10),  # Example 3
        ],
    )
    def test_forward_infer(self, mask_model_rnn, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = mask_model_rnn.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 1e-5

        with torch.no_grad():
            # batch size 1
            output_list = []
            output_length_list = []
            for i in range(input_signal.size(0)):
                output, output_length = model.forward(
                    input_signal=input_signal[i : i + 1], input_length=input_signal_length[i : i + 1]
                )
                output_list.append(output)
                output_length_list.append(output_length)
            output_instance = torch.cat(output_list, 0)
            output_length_instance = torch.cat(output_length_list, 0)

            # batch size batch_size
            output_batch, output_length_batch = model.forward(
                input_signal=input_signal, input_length=input_signal_length
            )

        # Check that the output and output length are the same for the instance and batch
        assert output_instance.shape == output_batch.shape
        assert output_length_instance.shape == output_length_batch.shape

        diff = torch.max(torch.abs(output_instance - output_batch))
        assert diff <= abs_tol


class TestMaskModelFlexArray:
    """Test masking model with channel-flexible mask estimator."""

    @pytest.mark.unit
    def test_constructor(self, mask_model_flexarray):
        """Test that the model can be constructed from a config dict."""
        model = mask_model_flexarray.train()
        confdict = model.to_config_dict()
        instance2 = EncMaskDecAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, EncMaskDecAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, num_channels, sample_len",
        [
            (4, 1, 4),  # 1-channel, Example 1
            (2, 1, 8),  # 1-channel, Example 2
            (1, 1, 10),  # 1-channel, Example 3
            (4, 3, 4),  # 3-channel, Example 1
            (2, 3, 8),  # 3-channel, Example 2
            (1, 3, 10),  # 3-channel, Example 3
        ],
    )
    def test_forward_infer(self, mask_model_flexarray, batch_size, num_channels, sample_len):
        """Test that the model can run forward inference."""
        model = mask_model_flexarray.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, num_channels, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 1e-5

        with torch.no_grad():
            # batch size 1
            output_list = []
            output_length_list = []
            for i in range(input_signal.size(0)):
                output, output_length = model.forward(
                    input_signal=input_signal[i : i + 1], input_length=input_signal_length[i : i + 1]
                )
                output_list.append(output)
                output_length_list.append(output_length)
            output_instance = torch.cat(output_list, 0)
            output_length_instance = torch.cat(output_length_list, 0)

            # batch size batch_size
            output_batch, output_length_batch = model.forward(
                input_signal=input_signal, input_length=input_signal_length
            )

        # Check that the output and output length are the same for the instance and batch
        assert output_instance.shape == output_batch.shape
        assert output_length_instance.shape == output_length_batch.shape

        diff = torch.max(torch.abs(output_instance - output_batch))
        assert diff <= abs_tol


class TestBFModelFlexArray:
    """Test beamforming model with channel-flexible mask estimator."""

    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    def test_constructor(self, bf_model_flexarray):
        """Test that the model can be constructed from a config dict."""
        model = bf_model_flexarray.train()
        confdict = model.to_config_dict()
        instance2 = EncMaskDecAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, EncMaskDecAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    @pytest.mark.parametrize(
        "batch_size, num_channels, sample_len",
        [
            (4, 1, 4),  # 1-channel, Example 1
            (2, 1, 8),  # 1-channel, Example 2
            (1, 1, 10),  # 1-channel, Example 3
            (4, 3, 4),  # 3-channel, Example 1
            (2, 3, 8),  # 3-channel, Example 2
            (1, 3, 10),  # 3-channel, Example 3
        ],
    )
    def test_forward_infer(self, bf_model_flexarray, batch_size, num_channels, sample_len):
        """Test that the model can run forward inference."""
        model = bf_model_flexarray.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, num_channels, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 1e-5

        with torch.no_grad():
            # batch size 1
            output_list = []
            output_length_list = []
            for i in range(input_signal.size(0)):
                output, output_length = model.forward(
                    input_signal=input_signal[i : i + 1], input_length=input_signal_length[i : i + 1]
                )
                output_list.append(output)
                output_length_list.append(output_length)
            output_instance = torch.cat(output_list, 0)
            output_length_instance = torch.cat(output_length_list, 0)

            # batch size batch_size
            output_batch, output_length_batch = model.forward(
                input_signal=input_signal, input_length=input_signal_length
            )

        # Check that the output and output length are the same for the instance and batch
        assert output_instance.shape == output_batch.shape
        assert output_length_instance.shape == output_length_batch.shape

        diff = torch.max(torch.abs(output_instance - output_batch))
        assert diff <= abs_tol
