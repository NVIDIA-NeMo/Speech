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
import copy

import pytest
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.data import audio_to_text
from nemo.collections.asr.models import EncDecCTCModel, configs
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.utils.config_utils import assert_dataclass_signature_match, update_model_config


@pytest.fixture()
def asr_model():
    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
        'feat_in': 64,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': 1024,
                'repeat': 1,
                'kernel': [1],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.0,
                'residual': False,
                'separable': True,
                'se': True,
                'se_context_size': -1,
            }
        ],
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
        'feat_in': 1024,
        'num_classes': 28,
        'vocabulary': [
            ' ',
            'a',
            'b',
            'c',
            'd',
            'e',
            'f',
            'g',
            'h',
            'i',
            'j',
            'k',
            'l',
            'm',
            'n',
            'o',
            'p',
            'q',
            'r',
            's',
            't',
            'u',
            'v',
            'w',
            'x',
            'y',
            'z',
            "'",
        ],
    }
    modelConfig = DictConfig(
        {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
    )

    model_instance = EncDecCTCModel(cfg=modelConfig)
    return model_instance


class TestEncDecCTCModel:
    @pytest.mark.unit
    def test_constructor(self, asr_model):
        asr_model.train()
        # TODO: make proper config and assert correct number of weights
        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        instance2 = EncDecCTCModel.from_config_dict(confdict)
        assert isinstance(instance2, EncDecCTCModel)

    @pytest.mark.unit
    def test_forward(self, asr_model):
        asr_model = asr_model.eval()

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                logprobs_ins, _, _ = asr_model.forward(
                    input_signal=input_signal[i : i + 1], input_signal_length=length[i : i + 1]
                )
                logprobs_instance.append(logprobs_ins)
                print(len(logprobs_ins))
            logprobs_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            logprobs_batch, _, _ = asr_model.forward(input_signal=input_signal, input_signal_length=length)

        assert logprobs_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_vocab_change(self, asr_model):
        old_vocab = copy.deepcopy(asr_model.decoder.vocabulary)
        nw1 = asr_model.num_weights
        asr_model.change_vocabulary(new_vocabulary=old_vocab)
        # No change
        assert nw1 == asr_model.num_weights
        new_vocab = copy.deepcopy(old_vocab)
        new_vocab.append('!')
        new_vocab.append('$')
        new_vocab.append('@')
        asr_model.change_vocabulary(new_vocabulary=new_vocab)
        # fully connected + bias
        assert asr_model.num_weights == nw1 + 3 * (asr_model.decoder._feat_in + 1)

    @pytest.mark.unit
    def test_decoding_change(self, asr_model):
        assert asr_model.decoding is not None
        assert isinstance(asr_model.decoding, CTCDecoding)
        assert asr_model.decoding.cfg.strategy == "greedy_batch"
        assert asr_model.decoding.preserve_alignments is False
        assert asr_model.decoding.compute_timestamps is False

        cfg = CTCDecodingConfig(preserve_alignments=True, compute_timestamps=True)
        asr_model.change_decoding_strategy(cfg)

        assert asr_model.decoding.preserve_alignments is True
        assert asr_model.decoding.compute_timestamps is True

    @pytest.mark.unit
    def test_change_conv_asr_se_context_window(self, asr_model):
        old_cfg = copy.deepcopy(asr_model.cfg)
        asr_model.change_conv_asr_se_context_window(context_window=32)  # 32 * 0.01s context
        new_config = asr_model.cfg

        assert old_cfg.encoder.jasper[0].se_context_size == -1
        assert new_config.encoder.jasper[0].se_context_size == 32

        for name, m in asr_model.encoder.named_modules():
            if type(m).__class__.__name__ == 'SqueezeExcite':
                assert m.context_window == 32

    @pytest.mark.unit
    def test_change_conv_asr_se_context_window_no_config_update(self, asr_model):
        old_cfg = copy.deepcopy(asr_model.cfg)
        asr_model.change_conv_asr_se_context_window(context_window=32, update_config=False)  # 32 * 0.01s context
        new_config = asr_model.cfg

        assert old_cfg.encoder.jasper[0].se_context_size == -1
        assert new_config.encoder.jasper[0].se_context_size == -1  # no change

        for name, m in asr_model.encoder.named_modules():
            if type(m).__class__.__name__ == 'SqueezeExcite':
                assert m.context_window == 32

    @pytest.mark.unit
    def test_dataclass_instantiation(self, asr_model):
        model_cfg = configs.EncDecCTCModelConfig()

        # Update mandatory values
        vocabulary = asr_model.decoder.vocabulary
        model_cfg.model.labels = vocabulary

        # Update encoder
        model_cfg.model.encoder.activation = 'relu'
        model_cfg.model.encoder.feat_in = 64
        model_cfg.model.encoder.jasper = [
            nemo_asr.modules.conv_asr.JasperEncoderConfig(
                filters=1024,
                repeat=1,
                kernel=[1],
                stride=[1],
                dilation=[1],
                dropout=0.0,
                residual=False,
                se=True,
                se_context_size=-1,
            )
        ]

        # Update decoder
        model_cfg.model.decoder.feat_in = 1024
        model_cfg.model.decoder.num_classes = 28
        model_cfg.model.decoder.vocabulary = vocabulary

        # Construct the model
        asr_cfg = OmegaConf.create({'model': asr_model.cfg})
        model_cfg_v1 = update_model_config(model_cfg, asr_cfg)
        new_model = EncDecCTCModel(cfg=model_cfg_v1.model)

        assert new_model.num_weights == asr_model.num_weights
        # trainer and exp manager should be there
        # assert 'trainer' in model_cfg_v1
        # assert 'exp_manager' in model_cfg_v1
        # datasets and optim/sched should not be there after ModelPT.update_model_dataclass()
        assert 'train_ds' not in model_cfg_v1.model
        assert 'validation_ds' not in model_cfg_v1.model
        assert 'test_ds' not in model_cfg_v1.model
        assert 'optim' not in model_cfg_v1.model

        # Construct the model, without dropping additional keys
        asr_cfg = OmegaConf.create({'model': asr_model.cfg})
        model_cfg_v2 = update_model_config(model_cfg, asr_cfg, drop_missing_subconfigs=False)

        # Assert all components are in config
        # assert 'trainer' in model_cfg_v2
        # assert 'exp_manager' in model_cfg_v2
        assert 'train_ds' in model_cfg_v2.model
        assert 'validation_ds' in model_cfg_v2.model
        assert 'test_ds' in model_cfg_v2.model
        assert 'optim' in model_cfg_v2.model

        # Remove extra components (optim and sched can be kept without issue)
        with open_dict(model_cfg_v2.model):
            model_cfg_v2.model.pop('train_ds')
            model_cfg_v2.model.pop('validation_ds')
            model_cfg_v2.model.pop('test_ds')

        new_model = EncDecCTCModel(cfg=model_cfg_v2.model)

        assert new_model.num_weights == asr_model.num_weights
        # trainer and exp manager should be there

    @pytest.mark.unit
    def test_ASRDatasetConfig_for_AudioToCharDataset(self):
        # ignore some additional arguments as dataclass is generic
        IGNORE_ARGS = [
            'is_tarred',
            'num_workers',
            'batch_size',
            'tarred_audio_filepaths',
            'shuffle',
            'pin_memory',
            'drop_last',
            'tarred_shard_strategy',
            'shard_manifests',
            'shuffle_n',
            'use_start_end_token',
            'use_start_end_token',
            'bucketing_batch_size',
            'bucketing_strategy',
            'bucketing_weights',
            'channel_selector',
        ]

        REMAP_ARGS = {'trim_silence': 'trim'}

        result = assert_dataclass_signature_match(
            audio_to_text.AudioToCharDataset,
            configs.ASRDatasetConfig,
            ignore_args=IGNORE_ARGS,
            remap_args=REMAP_ARGS,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_ASRDatasetConfig_for_TarredAudioToCharDataset(self):
        # ignore some additional arguments as dataclass is generic
        IGNORE_ARGS = [
            'is_tarred',
            'num_workers',
            'batch_size',
            'shuffle',
            'pin_memory',
            'drop_last',
            'global_rank',
            'world_size',
            'use_start_end_token',
            'bucketing_batch_size',
            'bucketing_strategy',
            'bucketing_weights',
            'max_utts',
        ]

        REMAP_ARGS = {
            'trim_silence': 'trim',
            'tarred_audio_filepaths': 'audio_tar_filepaths',
            'tarred_shard_strategy': 'shard_strategy',
            'shuffle_n': 'shuffle',
        }

        result = assert_dataclass_signature_match(
            audio_to_text.TarredAudioToCharDataset,
            configs.ASRDatasetConfig,
            ignore_args=IGNORE_ARGS,
            remap_args=REMAP_ARGS,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None
