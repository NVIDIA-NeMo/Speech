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
import os

import pytest
import torch
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.data import audio_to_label
from nemo.collections.asr.models import EncDecClassificationModel, EncDecFrameClassificationModel, configs
from nemo.utils.config_utils import assert_dataclass_signature_match


@pytest.fixture()
def speech_classification_model():
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': 32,
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
        },
    }

    decoder = {
        'cls': 'nemo.collections.asr.modules.ConvASRDecoderClassification',
        'params': {'feat_in': 32, 'num_classes': 30,},
    }

    modelConfig = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'labels': ListConfig(["dummy_cls_{}".format(i + 1) for i in range(30)]),
        }
    )
    model = EncDecClassificationModel(cfg=modelConfig)
    return model


@pytest.fixture()
def frame_classification_model():
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}
    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': 32,
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
        },
    }

    decoder = {
        'cls': 'nemo.collections.common.parts.MultiLayerPerceptron',
        'params': {'hidden_size': 32, 'num_classes': 5,},
    }

    modelConfig = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'labels': ListConfig(["dummy_cls_{}".format(i + 1) for i in range(5)]),
        }
    )
    model = EncDecFrameClassificationModel(cfg=modelConfig)
    return model


class TestEncDecClassificationModel:
    @pytest.mark.unit
    def test_constructor(self, speech_classification_model):
        asr_model = speech_classification_model.train()

        conv_cnt = (64 * 32 * 1 + 32) + (64 * 1 * 1 + 32)  # separable kernel + bias + pointwise kernel + bias
        bn_cnt = (4 * 32) * 2  # 2 * moving averages
        dec_cnt = 32 * 30 + 30  # fc + bias

        param_count = conv_cnt + bn_cnt + dec_cnt
        assert asr_model.num_weights == param_count

        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        instance2 = EncDecClassificationModel.from_config_dict(confdict)

        assert isinstance(instance2, EncDecClassificationModel)

    @pytest.mark.unit
    def test_forward(self, speech_classification_model):
        asr_model = speech_classification_model.eval()

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                logprobs_ins = asr_model.forward(
                    input_signal=input_signal[i : i + 1], input_signal_length=length[i : i + 1]
                )
                logprobs_instance.append(logprobs_ins)
            logprobs_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            logprobs_batch = asr_model.forward(input_signal=input_signal, input_signal_length=length)

        assert logprobs_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6
        diff = torch.max(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_vocab_change(self, speech_classification_model):
        asr_model = speech_classification_model.train()

        old_labels = copy.deepcopy(asr_model._cfg.labels)
        nw1 = asr_model.num_weights
        asr_model.change_labels(new_labels=old_labels)
        # No change
        assert nw1 == asr_model.num_weights
        new_labels = copy.deepcopy(old_labels)
        new_labels.append('dummy_cls_31')
        new_labels.append('dummy_cls_32')
        new_labels.append('dummy_cls_33')
        asr_model.change_labels(new_labels=new_labels)
        # fully connected + bias
        assert asr_model.num_weights == nw1 + 3 * (asr_model.decoder._feat_in + 1)

    @pytest.mark.unit
    def test_transcription(self, speech_classification_model, test_data_dir):
        # Ground truth labels = ["yes", "no"]
        audio_filenames = ['an22-flrp-b.wav', 'an90-fbbh-b.wav']
        audio_paths = [os.path.join(test_data_dir, "asr", "train", "an4", "wav", fp) for fp in audio_filenames]

        model = speech_classification_model.eval()

        # Test Top 1 classification transcription
        results = model.transcribe(audio_paths, batch_size=2)
        assert len(results) == 2
        assert results[0].shape == torch.Size([1])

        # Test Top 5 classification transcription
        model._accuracy.top_k = [5]  # set top k to 5 for accuracy calculation
        results = model.transcribe(audio_paths, batch_size=2)
        assert len(results) == 2
        assert results[0].shape == torch.Size([5])

        # Test Top 1 and Top 5 classification transcription
        model._accuracy.top_k = [1, 5]
        results = model.transcribe(audio_paths, batch_size=2)
        assert len(results) == 2
        assert results[0].shape == torch.Size([2, 1])
        assert results[1].shape == torch.Size([2, 5])
        assert model._accuracy.top_k == [1, 5]

        # Test log probs extraction
        model._accuracy.top_k = [1]
        results = model.transcribe(audio_paths, batch_size=2, logprobs=True)
        assert len(results) == 2
        assert results[0].shape == torch.Size([len(model.cfg.labels)])

        # Test log probs extraction remains same for any top_k
        model._accuracy.top_k = [5]
        results = model.transcribe(audio_paths, batch_size=2, logprobs=True)
        assert len(results) == 2
        assert results[0].shape == torch.Size([len(model.cfg.labels)])

    @pytest.mark.unit
    def test_EncDecClassificationDatasetConfig_for_AudioToSpeechLabelDataset(self):
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
            'shuffle_n',
            # `featurizer` is supplied at runtime
            'featurizer',
            # additional ignored arguments
            'vad_stream',
            'int_values',
            'sample_rate',
            'normalize_audio',
            'augmentor',
            'bucketing_batch_size',
            'bucketing_strategy',
            'bucketing_weights',
        ]

        REMAP_ARGS = {'trim_silence': 'trim'}

        result = assert_dataclass_signature_match(
            audio_to_label.AudioToSpeechLabelDataset,
            configs.EncDecClassificationDatasetConfig,
            ignore_args=IGNORE_ARGS,
            remap_args=REMAP_ARGS,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None


class TestEncDecFrameClassificationModel(TestEncDecClassificationModel):
    @pytest.mark.parametrize(["logits_len", "labels_len"], [(20, 10), (21, 10), (19, 10), (20, 9), (20, 11)])
    @pytest.mark.unit
    def test_reshape_labels(self, frame_classification_model, logits_len, labels_len):
        model = frame_classification_model.eval()

        logits = torch.ones(4, logits_len, 2)
        labels = torch.ones(4, labels_len)
        logits_len = torch.tensor([6, 7, 8, 9])
        labels_len = torch.tensor([5, 6, 7, 8])
        labels_new, labels_len_new = model.reshape_labels(
            logits=logits, labels=labels, logits_len=logits_len, labels_len=labels_len
        )
        assert labels_new.size(1) == logits.size(1)
        assert torch.equal(labels_len_new, torch.tensor([6, 7, 8, 9]))

    @pytest.mark.unit
    def test_EncDecClassificationDatasetConfig_for_AudioToMultiSpeechLabelDataset(self):
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
            'shuffle_n',
            # `featurizer` is supplied at runtime
            'featurizer',
            # additional ignored arguments
            'vad_stream',
            'int_values',
            'sample_rate',
            'normalize_audio',
            'augmentor',
            'bucketing_batch_size',
            'bucketing_strategy',
            'bucketing_weights',
            'delimiter',
            'normalize_audio_db',
            'normalize_audio_db_target',
            'window_length_in_sec',
            'shift_length_in_sec',
        ]

        REMAP_ARGS = {'trim_silence': 'trim'}

        result = assert_dataclass_signature_match(
            audio_to_label.AudioToMultiLabelDataset,
            configs.EncDecClassificationDatasetConfig,
            ignore_args=IGNORE_ARGS,
            remap_args=REMAP_ARGS,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None
