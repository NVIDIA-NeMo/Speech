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

"""
Script to compute the Word or Character Error Rate of a given ASR model for a given manifest file for some dataset.
The manifest file must conform to standard ASR definition - containing `audio_filepath` and `text` as the ground truth.

Note: This script depends on the `transcribe_speech.py` script, and therefore both scripts should be located in the
same directory during execution.

# Arguments

<< All arguments of `transcribe_speech.py` are inherited by this script, so please refer to `transcribe_speech.py`
for full list of arguments >>

    dataset_manifest: Required - path to dataset JSON manifest file (in NeMo format)
    output_filename: Optional - output filename where the transcriptions will be written. (if scores_per_sample=True, 
    metrics per sample will be written there too)

    use_cer: Bool, whether to compute CER or WER
    use_punct_er: Bool, compute dataset Punctuation Error Rate (set the punctuation marks for metrics computation with 
    "text_processing.punctuation_marks")
     
    tolerance: Float, minimum WER/CER required to pass some arbitrary tolerance.

    only_score_manifest: Bool, when set will skip audio transcription and just calculate WER of provided manifest.
    scores_per_sample: Bool, compute metrics for each sample separately (if only_score_manifest=True, scores per sample
    will be added to the manifest at the dataset_manifest path)

# Usage

## To score a dataset with a manifest file that does not contain previously transcribed `pred_text`.

python speech_to_text_eval.py \
    model_path=null \
    pretrained_name=null \
    dataset_manifest=<Mandatory: Path to an ASR dataset manifest file> \
    output_filename=<Optional: Some output filename which will hold the transcribed text as a manifest> \
    batch_size=32 \
    amp=True \
    use_cer=False

## To score a manifest file which has been previously augmented with transcribed text as `pred_text`
This is useful when one uses `transcribe_speech_parallel.py` to transcribe larger datasets, and results are written
to a manifest which has the two keys `text` (for ground truth) and `pred_text` (for model's transcription)

python speech_to_text_eval.py \
    dataset_manifest=<Mandatory: Path to an ASR dataset manifest file> \
    use_cer=False \
    only_score_manifest=True

"""

import json
import os
from dataclasses import dataclass, is_dataclass
from typing import Optional

import torch
import transcribe_speech
from omegaconf import MISSING, OmegaConf, open_dict

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.transcribe_utils import (
    PunctuationCapitalization,
    TextProcessingConfig,
    compute_metrics_per_sample,
)
from nemo.collections.common.metrics.punct_er import DatasetPunctuationErrorRate
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class EvaluationConfig(transcribe_speech.TranscriptionConfig):
    dataset_manifest: str = MISSING
    output_filename: Optional[str] = "evaluation_transcripts.json"

    # decoder type: ctc or rnnt, can be used to switch between CTC and RNNT decoder for Joint RNNT/CTC models
    decoder_type: Optional[str] = None
    # att_context_size can be set for cache-aware streaming models with multiple look-aheads
    att_context_size: Optional[list] = None

    use_cer: bool = False
    use_punct_er: bool = False
    tolerance: Optional[float] = None

    only_score_manifest: bool = False
    scores_per_sample: bool = False

    text_processing: Optional[TextProcessingConfig] = TextProcessingConfig(
        punctuation_marks=".,?", separate_punctuation=False, do_lowercase=False, rm_punctuation=False,
    )


@hydra_runner(config_name="EvaluationConfig", schema=EvaluationConfig)
def main(cfg: EvaluationConfig):
    torch.set_grad_enabled(False)

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.audio_dir is not None:
        raise RuntimeError(
            "Evaluation script requires ground truth labels to be passed via a manifest file. "
            "If manifest file is available, submit it via `dataset_manifest` argument."
        )

    if not os.path.exists(cfg.dataset_manifest):
        raise FileNotFoundError(f"The dataset manifest file could not be found at path : {cfg.dataset_manifest}")

    if not cfg.only_score_manifest:
        # Transcribe speech into an output directory
        transcription_cfg = transcribe_speech.main(cfg)  # type: EvaluationConfig

        # Release GPU memory if it was used during transcription
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info("Finished transcribing speech dataset. Computing ASR metrics..")

    else:
        cfg.output_filename = cfg.dataset_manifest
        transcription_cfg = cfg

    ground_truth_text = []
    predicted_text = []
    invalid_manifest = False
    with open(transcription_cfg.output_filename, 'r') as f:
        for line in f:
            data = json.loads(line)

            if 'pred_text' not in data:
                invalid_manifest = True
                break

            ground_truth_text.append(data['text'])

            predicted_text.append(data['pred_text'])

    pc = PunctuationCapitalization(cfg.text_processing.punctuation_marks)
    if cfg.text_processing.separate_punctuation:
        ground_truth_text = pc.separate_punctuation(ground_truth_text)
        predicted_text = pc.separate_punctuation(predicted_text)
    if cfg.text_processing.do_lowercase:
        ground_truth_text = pc.do_lowercase(ground_truth_text)
        predicted_text = pc.do_lowercase(predicted_text)
    if cfg.text_processing.rm_punctuation:
        ground_truth_text = pc.rm_punctuation(ground_truth_text)
        predicted_text = pc.rm_punctuation(predicted_text)

    # Test for invalid manifest supplied
    if invalid_manifest:
        raise ValueError(
            f"Invalid manifest provided: {transcription_cfg.output_filename} does not "
            f"contain value for `pred_text`."
        )

    if cfg.use_punct_er:
        dper_obj = DatasetPunctuationErrorRate(
            hypotheses=predicted_text,
            references=ground_truth_text,
            punctuation_marks=list(cfg.text_processing.punctuation_marks),
        )
        dper_obj.compute()

    if cfg.scores_per_sample:
        metrics_to_compute = ["wer", "cer"]

        if cfg.use_punct_er:
            metrics_to_compute.append("punct_er")

        samples_with_metrics = compute_metrics_per_sample(
            manifest_path=cfg.dataset_manifest,
            reference_field="text",
            hypothesis_field="pred_text",
            metrics=metrics_to_compute,
            punctuation_marks=cfg.text_processing.punctuation_marks,
            output_manifest_path=cfg.output_filename,
        )

    # Compute the WER
    cer = word_error_rate(hypotheses=predicted_text, references=ground_truth_text, use_cer=True)
    wer = word_error_rate(hypotheses=predicted_text, references=ground_truth_text, use_cer=False)

    if cfg.use_cer:
        metric_name = 'CER'
        metric_value = cer
    else:
        metric_name = 'WER'
        metric_value = wer

    if cfg.tolerance is not None:
        if metric_value > cfg.tolerance:
            raise ValueError(f"Got {metric_name} of {metric_value}, which was higher than tolerance={cfg.tolerance}")

        logging.info(f'Got {metric_name} of {metric_value}. Tolerance was {cfg.tolerance}')

    logging.info(f'Dataset WER/CER ' + str(round(100 * wer, 2)) + "%/" + str(round(100 * cer, 2)) + "%")

    if cfg.use_punct_er:
        dper_obj.print()
        dper_obj.reset()

    # Inject the metric name and score into the config, and return the entire config
    with open_dict(cfg):
        cfg.metric_name = metric_name
        cfg.metric_value = metric_value

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
