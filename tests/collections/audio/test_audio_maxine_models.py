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

try:
    import importlib

    importlib.import_module('torchaudio')

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False

from nemo.collections.audio.models.maxine import BNR2


@pytest.fixture()
def maxine_model_fixture():
    sample_rate = 16000
    fft_length = 1920
    hop_length = 480
    num_mels = 320

    optim = {
        'name': 'adam',
        'lr': 0.0005,
        'sched': {
            'name': 'StepLR',
        },
        'gamma': 0.999,
        'step_size': 2,
    }

    loss = {
        '_target_': 'nemo.collections.audio.losses.maxine.CombinedLoss',
        'sample_rate': sample_rate,
        'fft_length': fft_length,
        'hop_length': hop_length,
        'num_mels': num_mels,
        'sisnr_loss_weight': 1,
        'spectral_loss_weight': 15,
        'asr_loss_weight': 1,
        'use_asr_loss': True,
        'use_mel_spec': True,
    }

    config = DictConfig(
        {
            'type': "bnr",
            'sample_rate': sample_rate,
            'fft_length': fft_length,
            'hop_length': hop_length,
            'num_mels': num_mels,
            'skip_nan_grad': False,
            'num_outputs': 1,
            'segment': 4,
            'loss': DictConfig(loss),
            'optim': DictConfig(optim),
        }
    )

    bnr = BNR2(cfg=config)
    return bnr


class TestBNR2Model:
    """Test BNR 2 model."""

    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    def test_constructor(self, maxine_model_fixture):
        """Test that the model can be constructed from a config dict."""
        model = maxine_model_fixture.train()
        confdict = model.to_config_dict()
        instance2 = BNR2.from_config_dict(confdict)
        assert isinstance(instance2, BNR2)

    @pytest.mark.unit
    @pytest.mark.skipif(not HAVE_TORCHAUDIO, reason="Modules in this test require torchaudio")
    @pytest.mark.parametrize(
        "batch_size, sample_len",
        [
            # Note: Must be a multiple of 10ms @ 16kkHz
            (4, 16),  # Example 1
            (2, 8),  # Example 2
            (1, 32),  # Example 3
        ],
    )
    def test_forward_infer(self, maxine_model_fixture, batch_size, sample_len):
        """Test that the model can run forward inference."""
        model = maxine_model_fixture.eval()
        confdict = model.to_config_dict()
        sampling_rate = confdict['sample_rate']
        input_signal = torch.randn(size=(batch_size, 1, sample_len * sampling_rate))

        abs_tol = 1e-5

        with torch.no_grad():
            # batch size 1
            output_list = []
            for i in range(input_signal.size(0)):
                output = model.forward(input_signal=input_signal[i : i + 1])
                output_list.append(output)
            output_instance = torch.cat(output_list, 0)

            # batch size batch_size
            output_batch = model.forward(input_signal=input_signal)

        # Check that the output is the same for the instance and batch
        assert output_instance.shape == output_batch.shape

        diff = torch.max(torch.abs(output_instance - output_batch))
        assert diff <= abs_tol
