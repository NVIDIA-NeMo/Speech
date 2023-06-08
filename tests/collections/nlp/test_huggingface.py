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
import tempfile

import pytest
import torch
from omegaconf import OmegaConf

import nemo.collections.nlp as nemo_nlp
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer


def do_export(model, name: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate filename in the temporary directory.
        tmp_file_name = os.path.join(tmpdir, name + '.onnx')
        # Test export.
        if torch.cuda.is_available():
            model = model.cuda()
        model.export(tmp_file_name)


class TestHuggingFace:
    config = {"language_model": {"pretrained_model_name": ""}, "tokenizer": {}}
    omega_conf = OmegaConf.create(config)

    @pytest.mark.unit
    def test_list_pretrained_models(self):
        pretrained_lm_models = nemo_nlp.modules.get_pretrained_lm_models_list()
        assert len(pretrained_lm_models) > 0

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_get_pretrained_bert_model(self):
        self.omega_conf.language_model.pretrained_model_name = 'bert-base-uncased'
        model = nemo_nlp.modules.get_lm_model(cfg=self.omega_conf)
        assert isinstance(model, nemo_nlp.modules.BertEncoder)
        do_export(model, "bert-base-uncased")

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_get_pretrained_distilbert_model(self):
        self.omega_conf.language_model.pretrained_model_name = 'distilbert-base-uncased'
        model = nemo_nlp.modules.get_lm_model(cfg=self.omega_conf)
        assert isinstance(model, nemo_nlp.modules.DistilBertEncoder)
        do_export(model, "distilbert-base-uncased")

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_get_pretrained_roberta_model(self):
        self.omega_conf.language_model.pretrained_model_name = 'roberta-base'
        model = nemo_nlp.modules.get_lm_model(cfg=self.omega_conf)
        assert isinstance(model, nemo_nlp.modules.RobertaEncoder)
        do_export(model, "roberta-base-uncased")

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_get_pretrained_albert_model(self):
        self.omega_conf.language_model.pretrained_model_name = 'albert-base-v1'
        model = nemo_nlp.modules.get_lm_model(cfg=self.omega_conf)
        assert isinstance(model, nemo_nlp.modules.AlbertEncoder)
        do_export(model, "albert-base-v1")

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_get_pretrained_chinese_bert_wwm_model(self):
        model_name = 'hfl/chinese-bert-wwm'
        self.omega_conf.language_model.pretrained_model_name = model_name
        model = nemo_nlp.modules.get_lm_model(cfg=self.omega_conf)
        assert isinstance(model, nemo_nlp.modules.BertModule)
        tokenizer = get_tokenizer(tokenizer_name=model_name)
        assert isinstance(tokenizer, AutoTokenizer)

    # model is not on HF anymore
    # @pytest.mark.with_downloads()
    # @pytest.mark.unit
    # def test_get_pretrained_arabic_model(self):
    #     model_name = 'asafaya/bert-base-arabic'
    #     self.omega_conf.language_model.pretrained_model_name = model_name
    #     model = nemo_nlp.modules.get_lm_model(cfg=self.omega_conf)
    #     assert isinstance(model, nemo_nlp.modules.BertModule)
    #     tokenizer = get_tokenizer(tokenizer_name=model_name)
    #     assert isinstance(tokenizer, AutoTokenizer)
