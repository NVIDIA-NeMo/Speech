# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.asr.models import ASRModel, EncDecCTCModelBPE, EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.ctc_greedy_decoding import GreedyCTCInferConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInferConfig
from nemo.collections.asr.parts.utils.asr_confidence_benchmarking_utils import run_confidence_benchmark
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig

# both models recognize the test data without errors, thus every metric except ece return default values
ECE_VALUES = {("token", "ctc"): 0.87, ("token", "rnnt"): 0.82, ("word", "ctc"): 0.91, ("word", "rnnt"): 0.88}

TOL_DEGREE = 2
TOL = 1 / math.pow(10, TOL_DEGREE)


@pytest.fixture(scope="module")
def conformer_ctc_bpe_model():
    model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
    model.set_trainer(Trainer(devices=1, accelerator="cpu"))
    model = model.eval()
    return model


@pytest.fixture(scope="module")
def conformer_rnnt_bpe_model():
    model = EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_small")
    model.set_trainer(Trainer(devices=1, accelerator="cpu"))
    model = model.eval()
    return model


@pytest.mark.with_downloads
@pytest.fixture(scope="module")
# @pytest.fixture
def audio_and_texts(test_data_dir):
    # get filenames and reference texts from manifest
    filepaths = []
    reference_texts = []
    manifest = Path(test_data_dir) / Path("asr/an4_val.json")
    with open(manifest, 'r') as f:
        for line in f:
            item = json.loads(line)
            # alaptev: maybe fix those paths in the manifest?
            audio_file = Path(item['audio_filepath'].replace("/data/", "/.data/"))
            filepaths.append(str(audio_file.absolute()))
            reference_texts.append(item['text'])
    return filepaths, reference_texts


class TestASRConfidenceBenchmark:
    @pytest.mark.pleasefixme
    @pytest.mark.integration
    @pytest.mark.with_downloads
    @pytest.mark.parametrize('model_name', ("ctc", "rnnt"))
    @pytest.mark.parametrize('target_level', ("token", "word"))
    def test_run_confidence_benchmark(
        self, model_name, target_level, audio_and_texts, conformer_ctc_bpe_model, conformer_rnnt_bpe_model
    ):
        model = conformer_ctc_bpe_model if model_name == "ctc" else conformer_rnnt_bpe_model
        assert isinstance(model, ASRModel)
        filepaths, reference_texts = audio_and_texts
        confidence_cfg = (
            ConfidenceConfig(preserve_token_confidence=True)
            if target_level == "token"
            else ConfidenceConfig(preserve_word_confidence=True)
        )
        model.change_decoding_strategy(
            RNNTDecodingConfig(fused_batch_size=-1, strategy="greedy_batch", confidence_cfg=confidence_cfg)
            if model_name == "rnnt"
            else CTCDecodingConfig(confidence_cfg=confidence_cfg)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            assert np.allclose(
                np.array(
                    run_confidence_benchmark(model, target_level, filepaths, reference_texts, plot_dir=tmpdir)[
                        target_level
                    ]
                ),
                np.array([0.5, 1.0, 0.0, -math.inf, ECE_VALUES[(target_level, model_name)], 0.0, 0.0, 0.0]),
                atol=TOL,
            )

    @pytest.mark.pleasefixme
    @pytest.mark.integration
    @pytest.mark.with_downloads
    @pytest.mark.parametrize('model_name', ("ctc", "rnnt"))
    def test_deprecated_config_args(self, model_name, conformer_ctc_bpe_model, conformer_rnnt_bpe_model):
        assert ConfidenceConfig().method_cfg.alpha == 0.33, "default `alpha` is supposed to be 0.33"
        model = conformer_ctc_bpe_model if model_name == "ctc" else conformer_rnnt_bpe_model
        assert isinstance(model, ASRModel)

        conf = OmegaConf.create({"temperature": 0.5})
        test_args_main = {"method_cfg": conf}
        test_args_greedy = {"confidence_method_cfg": conf}
        confidence_cfg = ConfidenceConfig(preserve_word_confidence=True, **test_args_main)
        model.change_decoding_strategy(
            RNNTDecodingConfig(fused_batch_size=-1, strategy="greedy", confidence_cfg=confidence_cfg)
            if model_name == "rnnt"
            else CTCDecodingConfig(confidence_cfg=confidence_cfg)
        )
        assert model.cfg.decoding.confidence_cfg.method_cfg.alpha == 0.5
        model.change_decoding_strategy(
            RNNTDecodingConfig(
                fused_batch_size=-1,
                strategy="greedy",
                greedy=GreedyBatchedRNNTInferConfig(preserve_frame_confidence=True, **test_args_greedy),
            )
            if model_name == "rnnt"
            else CTCDecodingConfig(greedy=GreedyCTCInferConfig(preserve_frame_confidence=True, **test_args_greedy))
        )
        assert model.cfg.decoding.greedy.confidence_method_cfg.alpha == 0.5
