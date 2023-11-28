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

try:
    import apex

    apex_available = True
except Exception:
    apex_available = False

import os
import tempfile

import onnx
import pytest
import torch
from omegaconf import OmegaConf

import nemo.collections.nlp as nemo_nlp
from nemo.core.classes import typecheck


def get_pretrained_bert_345m_uncased_model():
    model_name = "megatron-bert-345m-uncased"
    config = {"language_model": {"pretrained_model_name": model_name}, "tokenizer": {}}
    omega_conf = OmegaConf.create(config)
    model = nemo_nlp.modules.get_lm_model(cfg=omega_conf)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


class TestMegatron:
    @pytest.mark.skip("This test was written for megatron-lm")
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_list_pretrained_models(self):
        pretrained_lm_models = nemo_nlp.modules.get_pretrained_lm_models_list()
        assert len(pretrained_lm_models) > 0

    @pytest.mark.with_downloads()
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.skip("Only one Megatron model is allowed")
    def test_get_model(self):
        model = get_pretrained_bert_345m_uncased_model()
        assert isinstance(model, nemo_nlp.modules.MegatronBertEncoder)

        typecheck.set_typecheck_enabled(enabled=False)
        inp = model.input_example()
        out = model.forward(*inp)
        typecheck.set_typecheck_enabled(enabled=True)

    @pytest.mark.skipif(not os.path.exists('/home/TestData'), reason='Not a Jenkins machine')
    @pytest.mark.with_downloads()
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.skip("Megatron-LM BERT support deprecated. Supported in NeMo < 1.5")
    def test_onnx_export(self):
        model = get_pretrained_bert_345m_uncased_model()
        assert model
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate filename in the temporary directory.
            # Test export.
            model.export(os.path.join(".", "megatron.onnx"))


if __name__ == "__main__":
    t = TestMegatron()
    t.test_onnx_export()
