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

import os

import numpy as np
import pytest
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy


class TestGPTEval:
    @pytest.mark.run_only_on('GPU')
    def setup_method(self, test_method):
        trainer_config = {
            "devices": 1,
            "num_nodes": 1,
            "accelerator": "gpu",
            "logger": False,
            "precision": 16,
        }
        tensor_model_parallel_size = 1
        pipeline_model_parallel_size = 1
        model_file = '/home/TestData/nlp/megatron_gpt/125M/megatron_gpt.nemo'

        # trainer required for restoring model parallel models
        trainer = Trainer(strategy=NLPDDPStrategy(), **trainer_config)
        assert (
            trainer_config["devices"] * trainer_config['num_nodes']
            == tensor_model_parallel_size * pipeline_model_parallel_size
        ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

        model = MegatronGPTModel.restore_from(restore_path=model_file, trainer=trainer)
        model.freeze()

        # has to turn off activations_checkpoint_method for inference
        try:
            model.model.language_model.encoder.activations_checkpoint_method = None
        except AttributeError:
            pass

        self.model = model

    # @pytest.mark.skipif(not os.path.exists('/home/TestData/nlp'), reason='Not a Jenkins machine')
    # skip this unit test for now. need to investigate the numerical issue
    @pytest.mark.skipif(True, reason='skip')
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    @pytest.mark.skip()
    # TODO renable the test
    def test_gpt_eval(self):
        # test greedy
        length_params: LengthParam = {
            "max_length": 30,
            "min_length": 0,
        }

        sampling_params: SamplingParam = {
            "use_greedy": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "add_BOS": True,
            "all_probs": False,
            "compute_logprob": False,
            "end_strings": ["<|endoftext|>"],
        }

        # test logprob
        sampling_params["compute_logprob"] = True
        sentence = 'run gpt in inference mode'
        response = self.model.generate(inputs=[sentence], length_params=length_params, sampling_params=sampling_params)
        assert response["sentences"][0] == sentence
        gt_token_ids = [5143, 308, 457, 287, 32278, 4235]
        assert np.array_equal(np.array(response['token_ids'][0]), gt_token_ids)
        assert len(response['full_logprob'][0]) == 5
        gt_log_prob = [
            -7.9579081535339355,
            -7.195970058441162,
            -5.269130706787109,
            -12.75404167175293,
            -4.631799697875977,
        ]
        assert np.allclose(np.array(response['logprob'][0]), gt_log_prob, atol=1e-4)
        gt_offsets = [0, 3, 5, 7, 10, 20]
        assert np.array_equal(np.array(response['offsets'][0]), gt_offsets)

        # # test top_p
        sampling_params["compute_logprob"] = False
        sampling_params["use_greedy"] = False
        sampling_params["top_p"] = 0.8
        sampling_params["repetition_penalty"] = 1.2

        gt_token_ids = [
            50256,
            15,
            59,
            198,
            59,
            2,
            16,
            59,
            2,
            17,
            58,
            57,
            59,
            62,
            37,
            7,
            39,
            15437,
            90,
            92,
            357,
            2481,
            8,
            3467,
            2,
            18,
            30109,
            9,
            43215,
            13,
            5416,
        ]
        gt_text = '0\\\n\\#1\\#2[Z\\_F(H)]{} (21) \\#3[[*Phys. Rev'
        response = self.model.generate(inputs=[''], length_params=length_params, sampling_params=sampling_params)
        assert np.array_equal(np.array(response['token_ids'][0]), gt_token_ids)
        assert response['sentences'][0] == gt_text
