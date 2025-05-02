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

from nemo.collections.audio.models import PredictiveAudioToAudioModel


@pytest.fixture()
def predictive_model_ncsn():

    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
        'max_utts_evaluation_metrics': 50,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 510,
        'hop_length': 128,
        'magnitude_power': 0.5,
        'scale': 0.33,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
        'magnitude_power': encoder['magnitude_power'],
        'scale': encoder['scale'],
    }
    estimator = {
        '_target_': 'nemo.collections.audio.parts.submodules.ncsnpp.SpectrogramNoiseConditionalScoreNetworkPlusPlus',
        'in_channels': 1,  # single-channel noisy input
        'out_channels': 1,  # single-channel estimate
        'num_res_blocks': 3,  # increased number of res blocks
        'pad_time_to': 64,  # pad to 64 frames for the time dimension
        'pad_dimension_to': 0,  # no padding in the frequency dimension
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.MSELoss',  # computed in the time domain
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'normalize_input': model['normalize_input'],
            'max_utts_evaluation_metrics': model['max_utts_evaluation_metrics'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'estimator': DictConfig(estimator),
            'loss': DictConfig(loss),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    model = PredictiveAudioToAudioModel(cfg=model_config)

    return model


@pytest.fixture()
def predictive_model_conformer():

    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
        'max_utts_evaluation_metrics': 50,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 510,
        'hop_length': 128,
        'magnitude_power': 0.5,
        'scale': 0.33,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
        'magnitude_power': encoder['magnitude_power'],
        'scale': encoder['scale'],
    }
    estimator = {
        '_target_': 'nemo.collections.audio.parts.submodules.conformer.SpectrogramConformer',
        'in_channels': 1,  # single-channel noisy input
        'out_channels': 1,  # single-channel estimate
        'feat_in': 256,  # input feature dimension = number of subbands
        'n_layers': 8,  # number of layers in the model
        'd_model': 512,  # the hidden size of the model
        'subsampling_factor': 1,  # subsampling factor for the model
        'self_attention_model': 'rel_pos',
        'n_heads': 8,  # number of heads for the model
        # streaming-related arguments
        # - this is a non-streaming config
        'conv_context_size': None,
        'conv_norm_type': 'layer_norm',
        'causal_downsampling': False,
        'att_context_size': [-1, -1],
        'att_context_style': 'regular',
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.MSELoss',  # computed in the time domain
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'normalize_input': model['normalize_input'],
            'max_utts_evaluation_metrics': model['max_utts_evaluation_metrics'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'estimator': DictConfig(estimator),
            'loss': DictConfig(loss),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    model = PredictiveAudioToAudioModel(cfg=model_config)

    return model


@pytest.fixture()
def predictive_model_streaming_conformer():

    model = {
        'sample_rate': 16000,
        'num_outputs': 1,
        'normalize_input': True,
        'max_utts_evaluation_metrics': 50,
    }
    encoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.AudioToSpectrogram',
        'fft_length': 510,
        'hop_length': 128,
        'magnitude_power': 0.5,
        'scale': 0.33,
    }
    decoder = {
        '_target_': 'nemo.collections.audio.modules.transforms.SpectrogramToAudio',
        'fft_length': encoder['fft_length'],
        'hop_length': encoder['hop_length'],
        'magnitude_power': encoder['magnitude_power'],
        'scale': encoder['scale'],
    }
    estimator = {
        '_target_': 'nemo.collections.audio.parts.submodules.conformer.SpectrogramConformer',
        'in_channels': 1,  # single-channel noisy input
        'out_channels': 1,  # single-channel estimate
        'feat_in': 256,  # input feature dimension = number of subbands
        'n_layers': 8,  # number of layers in the model
        'd_model': 512,  # the hidden size of the model
        'subsampling_factor': 1,  # subsampling factor for the model
        'self_attention_model': 'rel_pos',
        'n_heads': 8,  # number of heads for the model
        # streaming-related arguments
        # - streaming config with causal convolutions and limited attention context
        'conv_context_size': 'causal',
        'conv_norm_type': 'layer_norm',
        'causal_downsampling': True,
        'att_context_size': [102, 16],
        'att_context_style': 'chunked_limited',
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.MSELoss',  # computed in the time domain
    }

    model_config = DictConfig(
        {
            'sample_rate': model['sample_rate'],
            'num_outputs': model['num_outputs'],
            'normalize_input': model['normalize_input'],
            'max_utts_evaluation_metrics': model['max_utts_evaluation_metrics'],
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'estimator': DictConfig(estimator),
            'loss': DictConfig(loss),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    model = PredictiveAudioToAudioModel(cfg=model_config)

    return model


class TestPredictiveModelNCSN:
    """Test predictive model with NCSN estimator."""

    @pytest.mark.unit
    def test_constructor(self, predictive_model_ncsn):
        """Test that the model can be constructed from a config dict."""
        model = predictive_model_ncsn.train()
        confdict = model.to_config_dict()
        instance2 = PredictiveAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, PredictiveAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),  # Example 1
            (2, 8),  # Example 2
            (1, 10),  # Example 3
        ],
    )
    def test_forward_infer(self, predictive_model_ncsn, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = predictive_model_ncsn.eval()
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


class TestPredictiveModelConformer:
    """Test predictive model with conformer estimator."""

    @pytest.mark.unit
    def test_constructor(self, predictive_model_conformer):
        """Test that the model can be constructed from a config dict."""
        model = predictive_model_conformer.train()
        confdict = model.to_config_dict()
        instance2 = PredictiveAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, PredictiveAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),  # Example 1
            (2, 8),  # Example 2
            (1, 10),  # Example 3
        ],
    )
    def test_forward_infer(self, predictive_model_conformer, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = predictive_model_conformer.eval()
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


class TestPredictiveModelStreamingConformer:
    """Test predictive model with streaming conformer estimator."""

    @pytest.mark.unit
    def test_constructor(self, predictive_model_streaming_conformer):
        """Test that the model can be constructed from a config dict."""
        model = predictive_model_streaming_conformer.train()
        confdict = model.to_config_dict()
        instance2 = PredictiveAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, PredictiveAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),  # Example 1
            (2, 8),  # Example 2
            (1, 10),  # Example 3
        ],
    )
    def test_forward_infer(self, predictive_model_streaming_conformer, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = predictive_model_streaming_conformer.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

        abs_tol = 5e-5

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
