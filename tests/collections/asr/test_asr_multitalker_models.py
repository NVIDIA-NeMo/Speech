# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models.multitalker_asr_models import EncDecMultiTalkerRNNTBPEModel
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cpu_is_supported(
    __NUMBA_MINIMUM_VERSION__
) or numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


@pytest.fixture()
def asr_model(test_data_dir):
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}

    model_defaults = {'enc_hidden': 1024, 'pred_hidden': 64}
    spk_kernel_type = "ff"
    spk_kernel_layers = [0]
    add_bg_spk_kernel = True

    encoder = {
        'cls': 'nemo.collections.asr.modules.ConformerEncoder',
        'params': {
            'feat_in': 64,
            'n_layers': 1,
            'd_model': model_defaults['enc_hidden'],  # Required by SpeakerKernelMixin
            'subsampling': 'dw_striding',
            'subsampling_factor': 2,
            'ff_expansion_factor': 4,
            'self_attention_model': 'rel_pos',
            'n_heads': 4,
            'conv_kernel_size': 7,
            'dropout': 0.1,
        },
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
        'prednet': {
            'pred_hidden': model_defaults['pred_hidden'],
            'pred_rnn_layers': 1,
        },
    }

    joint = {
        '_target_': 'nemo.collections.asr.modules.RNNTJoint',
        'jointnet': {
            'joint_hidden': 32,
            'activation': 'relu',
        },
    }

    decoding = {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 30}}

    tokenizer = {'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'}

    loss = {'loss_name': 'default', 'warprnnt_numba_kwargs': {'fastemit_lambda': 0.001}}

    modelConfig = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'joint': DictConfig(joint),
            'tokenizer': DictConfig(tokenizer),
            'decoding': DictConfig(decoding),
            'loss': DictConfig(loss),
            'spk_kernel_type': spk_kernel_type,
            'spk_kernel_layers': spk_kernel_layers,
            'add_bg_spk_kernel': add_bg_spk_kernel,
        }
    )

    model_instance = EncDecMultiTalkerRNNTBPEModel(cfg=modelConfig)
    return model_instance


class TestEncDecMultiTalkerRNNTBPEModel:
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_constructor(self, asr_model):
        """Test model constructor and speaker kernel initialization."""
        asr_model.train()

        # Check that it's the correct type
        assert isinstance(asr_model, EncDecMultiTalkerRNNTBPEModel)

        # Check speaker kernel configuration
        assert hasattr(asr_model, 'spk_kernel_type')
        assert hasattr(asr_model, 'spk_kernel_layers')
        assert hasattr(asr_model, 'add_bg_spk_kernel')

        # Check speaker kernel initialization
        assert asr_model.spk_kernel_type == "ff"
        assert asr_model.spk_kernel_layers == [0]
        assert asr_model.add_bg_spk_kernel is True

        # Check speaker kernels exist
        assert hasattr(asr_model, 'spk_kernels')
        if asr_model.add_bg_spk_kernel:
            assert hasattr(asr_model, 'bg_spk_kernels')

        # Test config dict conversion
        confdict = asr_model.to_config_dict()
        instance2 = EncDecMultiTalkerRNNTBPEModel.from_config_dict(confdict)
        assert isinstance(instance2, EncDecMultiTalkerRNNTBPEModel)

    @pytest.mark.with_downloads()
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_forward(self, asr_model):
        """Test forward pass functionality."""
        asr_model = asr_model.eval()

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.compute_eval_loss = False

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=321, high=500, size=[4])

        # Create mock speaker targets
        batch_size = input_signal.size(0)
        target_length = 32  # Typical encoder output length for test
        spk_targets = torch.randint(0, 2, (batch_size, target_length), dtype=torch.float32)
        bg_spk_targets = torch.randint(0, 2, (batch_size, target_length), dtype=torch.float32)

        # Set speaker targets
        asr_model.set_speaker_targets(spk_targets, bg_spk_targets)

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                # Set individual speaker targets for each sample
                asr_model.set_speaker_targets(spk_targets[i : i + 1], bg_spk_targets[i : i + 1])
                logprobs_ins, _ = asr_model.forward(
                    input_signal=input_signal[i : i + 1], input_signal_length=length[i : i + 1]
                )
                logprobs_instance.append(logprobs_ins)
            logits_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            asr_model.set_speaker_targets(spk_targets, bg_spk_targets)
            logprobs_batch, _ = asr_model.forward(input_signal=input_signal, input_signal_length=length)

        assert logits_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logits_instance - logprobs_batch))
        assert diff <= 1e-5  # Allow slightly higher tolerance for speaker processing
        diff = torch.max(torch.abs(logits_instance - logprobs_batch))
        assert diff <= 1e-5

    @pytest.mark.unit
    def test_speaker_target_setting(self, asr_model):
        """Test speaker target setting functionality."""
        batch_size = 2
        target_length = 32

        spk_targets = torch.randint(0, 2, (batch_size, target_length), dtype=torch.float32)
        bg_spk_targets = torch.randint(0, 2, (batch_size, target_length), dtype=torch.float32)

        # Test setting speaker targets
        asr_model.set_speaker_targets(spk_targets, bg_spk_targets)
        assert torch.equal(asr_model.spk_targets, spk_targets)
        if asr_model.add_bg_spk_kernel:
            assert torch.equal(asr_model.bg_spk_targets, bg_spk_targets)

        # Test clearing speaker targets
        asr_model.set_speaker_targets(None, None)
        assert asr_model.spk_targets is None
        if asr_model.add_bg_spk_kernel:
            assert asr_model.bg_spk_targets is None
