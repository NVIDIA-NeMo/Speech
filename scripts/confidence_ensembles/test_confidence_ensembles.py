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

# these tests are not included in CI, since they take moderate amount of time
# they are supposed to be run in the nightly pipeline instead

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nemo.collections.asr.parts.utils.transcribe_utils import TextProcessingConfig

sys.path.append(str(Path(__file__).parents[2] / 'examples' / 'asr'))
import speech_to_text_eval


@pytest.mark.parametrize(
    'build_args',
    [
        "ensemble.0.model=stt_es_conformer_ctc_large ensemble.1.model=stt_it_conformer_ctc_large",
        "ensemble.0.model=stt_es_conformer_transducer_large ensemble.1.model=stt_it_conformer_transducer_large",
        (
            "ensemble.0.model=stt_es_fastconformer_hybrid_large_pc ensemble.1.model=stt_it_fastconformer_hybrid_large_pc "
            "confidence.method_cfg.alpha=0.33 confidence.method_cfg.entropy_norm=exp "
        ),
        (
            "ensemble.0.model=stt_es_fastconformer_hybrid_large_pc "
            "ensemble.1.model=stt_it_fastconformer_hybrid_large_pc "
            "transcription.decoder_type=ctc "
        ),
        "ensemble.0.model=stt_es_conformer_ctc_large ensemble.1.model=stt_it_conformer_transducer_large",
        (
            "ensemble.0.model=stt_es_conformer_ctc_large "
            "ensemble.1.model=stt_it_conformer_ctc_large "
            f"ensemble.0.dev_manifest={Path(os.getenv('TEST_DATA_PATH', '')) / 'es' / 'dev_manifest.json'} "
            f"ensemble.1.dev_manifest={Path(os.getenv('TEST_DATA_PATH', '')) / 'it' / 'dev_manifest.json'} "
            "tune_confidence=True "
        ),
        (
            "ensemble.0.model=stt_es_conformer_transducer_large "
            "ensemble.1.model=stt_it_conformer_transducer_large "
            f"ensemble.0.dev_manifest={Path(os.getenv('TEST_DATA_PATH', '')) / 'es' / 'dev_manifest.json'} "
            f"ensemble.1.dev_manifest={Path(os.getenv('TEST_DATA_PATH', '')) / 'it' / 'dev_manifest.json'} "
            "tune_confidence=True "
        ),
    ],
    ids=(
        [
            "CTC models",
            "Transducer models",
            "Hybrid models (Transducer mode)",
            "Hybrid models (CTC mode)",
            "CTC + Transducer",
            "CTC models + confidence tuning",
            "Transducer models + confidence tuning",
        ]
    ),
)
def test_confidence_ensemble(tmp_path, build_args):
    """Integration tests for confidence-ensembles.

    Tests building ensemble and running inference with the model.
    To use, make sure to define TEST_DATA_PATH env variable with path to
    the test data. The following structure is assumed:

        $TEST_DATA_PATH
        ├── es
        │   ├── dev
        │   ├── dev_manifest.json
        │   ├── test
        │   ├── train
        │   └── train_manifest.json
        ├── it
        │   ├── dev
        │   ├── dev_manifest.json
        │   ├── test
        │   ├── test_manifest.json
        │   ├── train
        │   └── train_manifest.json
        └── test_manifest.json

    """
    # checking for test data and failing right away if not defined
    if not os.getenv("TEST_DATA_PATH"):
        raise ValueError("TEST_DATA_PATH env variable has to be defined!")

    test_data_path = Path(os.environ['TEST_DATA_PATH'])

    build_ensemble_cmd = f"""
        python {Path(__file__).parent / 'build_ensemble.py'} \
            --config-name=ensemble_config.yaml \
            output_path={tmp_path / 'ensemble.nemo'} \
            {build_args}
    """
    subprocess.run(build_ensemble_cmd, check=True, shell=True)

    eval_cfg = speech_to_text_eval.EvaluationConfig(
        dataset_manifest=str(test_data_path / 'test_manifest.json'),
        output_filename=str(tmp_path / 'output.json'),
        model_path=str(tmp_path / 'ensemble.nemo'),
        text_processing=TextProcessingConfig(punctuation_marks=".,?", do_lowercase=True, rm_punctuation=True),
    )

    results = speech_to_text_eval.main(eval_cfg)
    assert results.metric_value < 0.20  # relaxed check for better than 20% WER
