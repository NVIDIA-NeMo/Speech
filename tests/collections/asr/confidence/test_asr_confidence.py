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

from nemo.collections.asr.models import ASRModel, EncDecMultiTaskModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.ctc_greedy_decoding import GreedyCTCInferConfig
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecodingConfig
from nemo.collections.asr.parts.submodules.multitask_greedy_decoding import AEDGreedyInferConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInferConfig
from nemo.collections.asr.parts.utils.asr_confidence_benchmarking_utils import run_confidence_benchmark
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig

# both models recognize the test data without errors, thus every metric except ece return default values
# ECE values for fast conformer models (stt_en_fastconformer_ctc_large and stt_en_fastconformer_transducer_large)
ECE_VALUES = {("token", "ctc"): 0.86, ("token", "rnnt"): 0.75, ("word", "ctc"): 0.89, ("word", "rnnt"): 0.80}

TOL_DEGREE = 2
TOL = 2 / math.pow(10, TOL_DEGREE)


@pytest.fixture(scope="module")
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
    @pytest.mark.integration
    @pytest.mark.with_downloads
    @pytest.mark.parametrize('model_name', ("ctc", "rnnt"))
    @pytest.mark.parametrize('target_level', ("token", "word"))
    def test_run_confidence_benchmark(
        self, model_name, target_level, audio_and_texts, fast_conformer_ctc_model, fast_conformer_transducer_model
    ):
        model = fast_conformer_ctc_model if model_name == "ctc" else fast_conformer_transducer_model
        assert isinstance(model, ASRModel)
        filepaths, reference_texts = audio_and_texts
        confidence_cfg = (
            ConfidenceConfig(preserve_frame_confidence=True, preserve_token_confidence=True)
            if target_level == "token"
            else ConfidenceConfig(preserve_frame_confidence=True, preserve_word_confidence=True)
        )
        model.change_decoding_strategy(
            RNNTDecodingConfig(
                fused_batch_size=-1,
                strategy="greedy_batch",
                confidence_cfg=confidence_cfg,
                greedy=GreedyBatchedRNNTInferConfig(loop_labels=False),
            )
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

    @pytest.mark.integration
    @pytest.mark.with_downloads
    @pytest.mark.parametrize('model_name', ("ctc", "rnnt"))
    def test_deprecated_config_args(self, model_name, fast_conformer_ctc_model, fast_conformer_transducer_model):
        assert ConfidenceConfig().method_cfg.alpha == 0.33, "default `alpha` is supposed to be 0.33"
        model = fast_conformer_ctc_model if model_name == "ctc" else fast_conformer_transducer_model
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

    @pytest.mark.unit
    def test_aed_multitask_model_confidence(self, canary_1b_v2, test_data_dir):
        """Test token and word confidence for AED multitask models (Canary)."""
        model = canary_1b_v2
        assert isinstance(model, EncDecMultiTaskModel)

        audio_file = Path(test_data_dir) / "asr" / "train" / "an4" / "wav" / "an46-mmap-b.wav"

        # Configure decoding with confidence
        decode_cfg = MultiTaskDecodingConfig(
            strategy='greedy',
            greedy=AEDGreedyInferConfig(preserve_token_confidence=True),
            confidence_cfg=ConfidenceConfig(preserve_token_confidence=True, preserve_word_confidence=True),
        )
        model.change_decoding_strategy(decode_cfg)

        hypotheses = model.transcribe(
            audio=str(audio_file),
            batch_size=1,
            return_hypotheses=True,
        )

        assert len(hypotheses) == 1
        hyp = hypotheses[0]

        # Verify text is present
        assert isinstance(hyp.text, str)
        assert len(hyp.text) > 0

        # Verify y_sequence is present
        assert hyp.y_sequence is not None
        assert len(hyp.y_sequence) > 0

        # Verify token confidence is present and has correct length
        assert hyp.token_confidence is not None
        assert len(hyp.token_confidence) == len(hyp.y_sequence)

        # Verify word confidence is present
        assert hyp.word_confidence is not None
        assert len(hyp.word_confidence) > 0

        # Verify confidence values are in valid range [0, 1]
        for conf in hyp.token_confidence:
            assert 0.0 <= conf <= 1.0, f"Token confidence {conf} not in valid range [0, 1]"
        for conf in hyp.word_confidence:
            assert 0.0 <= conf <= 1.0, f"Word confidence {conf} not in valid range [0, 1]"
