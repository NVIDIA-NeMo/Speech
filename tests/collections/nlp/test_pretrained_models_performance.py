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
from pathlib import Path
from shutil import rmtree
from unittest import TestCase

import lightning.pytorch as pl
import pytest
from omegaconf import OmegaConf

import nemo.collections.nlp.models as models


def get_metrics(data_dir, model):
    trainer = pl.Trainer(devices=[0], accelerator='gpu')

    model.set_trainer(trainer)
    model.update_data_dir(data_dir)

    test_ds = OmegaConf.create(
        {
            'text_file': 'text_dev.txt',
            'labels_file': 'labels_dev.txt',
            'shuffle': False,
            'num_samples': -1,
            'batch_size': 8,
        }
    )

    model._cfg.dataset.use_cache = False
    model.setup_test_data(test_data_config=test_ds)
    metrics = trainer.test(model)[0]

    return metrics


def get_metrics_new_format(data_dir, model):
    trainer = pl.Trainer(devices=[0], accelerator='gpu')

    model.set_trainer(trainer)

    test_ds = OmegaConf.create(
        {
            'use_tarred_dataset': False,
            'ds_item': data_dir,
            'text_file': 'text_dev.txt',
            'labels_file': 'labels_dev.txt',
            'shuffle': False,
            'num_samples': -1,
            'tokens_in_batch': 512,
            'use_cache': False,
        }
    )
    model.setup_test_data(test_data_config=test_ds)
    metrics = trainer.test(model)[0]

    return metrics


def data_exists(data_dir):
    return os.path.exists(data_dir)


class TestPretrainedModelPerformance:
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.skipif(
        not data_exists('/home/TestData/nlp/token_classification_punctuation/fisher'), reason='Not a Jenkins machine'
    )
    def test_punct_capit_with_bert(self):
        data_dir = '/home/TestData/nlp/token_classification_punctuation/fisher'
        model = models.PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")
        metrics = get_metrics_new_format(data_dir, model)

        assert abs(metrics['test_punct_precision'] - 52.3024) < 0.001
        assert abs(metrics['test_punct_recall'] - 58.9220) < 0.001
        assert abs(metrics['test_punct_f1'] - 53.2976) < 0.001
        assert abs(metrics['test_capit_precision'] - 87.0707) < 0.001
        assert abs(metrics['test_capit_recall'] - 87.0707) < 0.001
        assert abs(metrics['test_capit_f1'] - 87.0707) < 0.001
        assert int(model.metrics['test']['punct_class_report'][0].total_examples) == 128

        preds_512 = model.add_punctuation_capitalization(['what can i do for you today'], max_seq_length=512)[0]
        assert preds_512 == 'What can I do for you today?'
        preds_5 = model.add_punctuation_capitalization(['what can i do for you today'], max_seq_length=5, margin=0)[0]
        assert preds_5 == 'What can I? Do for you. Today.'
        preds_5_step_1 = model.add_punctuation_capitalization(
            ['what can i do for you today'], max_seq_length=5, margin=0, step=1
        )[0]
        assert preds_5_step_1 == 'What Can I do for you today.'
        preds_6_step_1_margin_6 = model.add_punctuation_capitalization(
            ['what can i do for you today'], max_seq_length=6, margin=1, step=1
        )[0]
        assert preds_6_step_1_margin_6 == 'What can I do for you today.'

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.skipif(
        not data_exists('/home/TestData/nlp/token_classification_punctuation/fisher'), reason='Not a Jenkins machine'
    )
    def test_punct_capit_with_distilbert(self):
        data_dir = '/home/TestData/nlp/token_classification_punctuation/fisher'
        model = models.PunctuationCapitalizationModel.from_pretrained("punctuation_en_distilbert")
        metrics = get_metrics_new_format(data_dir, model)

        assert abs(metrics['test_punct_precision'] - 53.0826) < 0.001
        assert abs(metrics['test_punct_recall'] - 56.2905) < 0.001
        assert abs(metrics['test_punct_f1'] - 52.4225) < 0.001
        assert int(model.metrics['test']['punct_class_report'][0].total_examples) == 128

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.skipif(
        not data_exists('/home/TestData/nlp/token_classification_punctuation/gmb'), reason='Not a Jenkins machine'
    )
    def test_ner_model(self):
        data_dir = '/home/TestData/nlp/token_classification_punctuation/gmb'
        model = models.TokenClassificationModel.from_pretrained("ner_en_bert")
        metrics = get_metrics(data_dir, model)

        assert abs(metrics['precision'] - 96.0937) < 0.001
        assert abs(metrics['recall'] - 96.0146) < 0.001
        assert abs(metrics['f1'] - 95.6076) < 0.001
        assert int(model.classification_report.total_examples) == 202
