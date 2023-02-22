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

import json
import os
import tempfile
from unittest import TestCase

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models import EncDecSpeakerLabelModel


class EncDecSpeechLabelModelTest(TestCase):
    @pytest.mark.unit
    def test_constructor(self):
        preprocessor = {
            '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
        }
        encoder = {
            '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': 512,
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': False,
                }
            ],
        }

        decoder = {
            '_target_': 'nemo.collections.asr.modules.SpeakerDecoder',
            'feat_in': 512,
            'num_classes': 2,
            'pool_mode': 'xvector',
            'emb_sizes': [1024],
        }

        modelConfig = DictConfig(
            {
                'preprocessor': DictConfig(preprocessor),
                'encoder': DictConfig(encoder),
                'decoder': DictConfig(decoder),
            },
        )
        speaker_model = EncDecSpeakerLabelModel(cfg=modelConfig)
        speaker_model.train()
        # TODO: make proper config and assert correct number of weights

        # Check to/from config_dict:
        confdict = speaker_model.to_config_dict()
        instance2 = EncDecSpeakerLabelModel.from_config_dict(confdict)
        self.assertTrue(isinstance(instance2, EncDecSpeakerLabelModel))

    @pytest.mark.unit
    def test_ecapa_enc_dec(self):
        preprocessor = {
            '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
        }
        encoder = {
            '_target_': 'nemo.collections.asr.modules.ECAPAEncoder',
            'feat_in': 80,
            'filters': [4, 4, 4, 4, 3],
            'kernel_sizes': [5, 3, 3, 3, 1],
            'dilations': [1, 1, 1, 1, 1],
            'scale': 2,
        }

        decoder = {
            '_target_': 'nemo.collections.asr.modules.SpeakerDecoder',
            'feat_in': 3,
            'num_classes': 2,
            'pool_mode': 'attention',
            'emb_sizes': 192,
        }

        modelConfig = DictConfig(
            {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder),}
        )
        speaker_model = EncDecSpeakerLabelModel(cfg=modelConfig)
        speaker_model.train()
        # TODO: make proper config and assert correct number of weights

        # Check to/from config_dict:
        confdict = speaker_model.to_config_dict()
        instance2 = EncDecSpeakerLabelModel.from_config_dict(confdict)
        self.assertTrue(isinstance(instance2, EncDecSpeakerLabelModel))

    @pytest.mark.unit
    def test_titanet_enc_dec(self):
        preprocessor = {
            '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
        }
        encoder = {
            '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': 256,
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
            '_target_': 'nemo.collections.asr.modules.SpeakerDecoder',
            'feat_in': 256,
            'num_classes': 2,
            'pool_mode': 'attention',
            'emb_sizes': [1024],
        }

        modelConfig = DictConfig(
            {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder),}
        )
        speaker_model = EncDecSpeakerLabelModel(cfg=modelConfig)
        speaker_model.train()
        # TODO: make proper config and assert correct number of weights

        # Check to/from config_dict:
        confdict = speaker_model.to_config_dict()
        instance2 = EncDecSpeakerLabelModel.from_config_dict(confdict)
        self.assertTrue(isinstance(instance2, EncDecSpeakerLabelModel))


class TestEncDecSpeechLabelModel:
    @pytest.mark.unit
    def test_pretrained_titanet_embeddings(self, test_data_dir):
        model_name = 'titanet_large'
        speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name)
        assert isinstance(speaker_model, EncDecSpeakerLabelModel)
        relative_filepath = "an4_speaker/an4/wav/an4_clstk/fash/an251-fash-b.wav"
        filename = os.path.join(test_data_dir, relative_filepath)

        emb, logits = speaker_model.infer_file(filename)

        class_id = logits.argmax(axis=-1)
        emb_sum = emb.sum()

        assert 11144 == class_id
        assert (emb_sum + 0.2575) <= 1e-2

    @pytest.mark.unit
    def test_pretrained_ambernet_logits(self, test_data_dir):
        model_name = 'langid_ambernet'
        lang_model = EncDecSpeakerLabelModel.from_pretrained(model_name)
        assert isinstance(lang_model, EncDecSpeakerLabelModel)
        relative_filepath = "an4_speaker/an4/wav/an4_clstk/fash/an255-fash-b.wav"
        filename = os.path.join(test_data_dir, relative_filepath)

        label = lang_model.get_label(filename)

        assert label == "en"

    @pytest.mark.unit
    def test_pretrained_ambernet_logits_batched(self, test_data_dir):
        model_name = 'langid_ambernet'
        lang_model = EncDecSpeakerLabelModel.from_pretrained(model_name)
        relative_filepath = "an4_speaker/an4/wav/an4_clstk/fash/an255-fash-b.wav"
        filename = os.path.join(test_data_dir, relative_filepath)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_manifest = os.path.join(tmpdir, 'manifest.json')
            with open(temp_manifest, 'w', encoding='utf-8') as fp:
                entry = {"audio_filepath": filename, "duration": 4.5, "label": 'en'}
                fp.write(json.dumps(entry) + '\n')
                entry = {
                    "audio_filepath": filename,
                    "duration": 4.5,
                    "label": 'test',
                }  # test sample outside of training set
                fp.write(json.dumps(entry) + '\n')

            embs, logits, gt_labels, trained_labels = lang_model.batch_inference(temp_manifest, device=device)
            pred_label = trained_labels[logits.argmax(axis=-1)[0]]
            true_label = gt_labels[0]

            assert pred_label == true_label
            assert gt_labels[1] == 'test'
