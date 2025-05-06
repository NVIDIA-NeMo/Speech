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

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.audio.models import SchroedingerBridgeAudioToAudioModel


@pytest.fixture()
def schroedinger_bridge_model_ncsn():

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
        'in_channels': 2,  # single-channel noisy input
        'out_channels': 1,  # single-channel estimate
        'conditioned_on_time': True,
        'channels': [8, 8, 8, 8, 8],
        'num_res_blocks': 3,  # increased number of res blocks
        'pad_time_to': 64,  # pad to 64 frames for the time dimension
        'pad_dimension_to': 0,  # no padding in the frequency dimension
    }

    loss_encoded = {'_target_': 'nemo.collections.audio.losses.MSELoss', 'ndim': 4}  # computed in the time domain

    loss_time = {'_target_': 'nemo.collections.audio.losses.MAELoss'}

    noise_schedule = {
        '_target_': 'nemo.collections.audio.parts.submodules.schroedinger_bridge.SBNoiseScheduleVE',
        'k': 2.6,
        'c': 0.4,
        'time_min': 1e-4,
        'time_max': 1.0,
        'num_steps': 1000,  # num steps for the forward process
    }

    sampler = {
        '_target_': 'nemo.collections.audio.parts.submodules.schroedinger_bridge.SBSampler',
        'time_min': 1e-4,
        'time_max': 1.0,
        'num_steps': 5,  # num steps for the reverse process
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
            'loss_encoded': DictConfig(loss_encoded),
            'loss_time': DictConfig(loss_time),
            'loss_time_weight': 0.001,
            'estimator_output': 'data_prediction',
            'noise_schedule': DictConfig(noise_schedule),
            'sampler': DictConfig(sampler),
            'optim': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'betas': (0.9, 0.98),
            },
        }
    )

    model = SchroedingerBridgeAudioToAudioModel(cfg=model_config)

    return model


class TestSchroedingerBridgeModelNCSN:
    """Test Schroedinger Bridge model with NCSN estimator."""

    @pytest.mark.unit
    def test_constructor(self, schroedinger_bridge_model_ncsn):
        """Test that the model can be constructed from a config dict."""
        model = schroedinger_bridge_model_ncsn.train()
        confdict = model.to_config_dict()
        instance2 = SchroedingerBridgeAudioToAudioModel.from_config_dict(confdict)
        assert isinstance(instance2, SchroedingerBridgeAudioToAudioModel)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            (4, 4),  # Example 1
            (2, 8),  # Example 2
            (1, 10),  # Example 3
        ],
    )
    def test_forward_infer(self, schroedinger_bridge_model_ncsn, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = schroedinger_bridge_model_ncsn.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        rng = torch.Generator()
        rng.manual_seed(0)
        input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate), generator=rng)
        input_signal_length = (sample_len * sampling_rate) * torch.ones(batch_size, dtype=torch.int)

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

        # It is generative model so we do not check the diffenence between output_instance and output_batch

        # Check that the output and output length are the same for the instance and batch
        assert output_instance.shape == output_batch.shape
        assert output_length_instance.shape == output_length_batch.shape
