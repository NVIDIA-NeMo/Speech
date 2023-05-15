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
#
# Run ``python build_ensemble.py --help`` for usage examples.

import atexit

# using default logging to be able to silence unnecessary messages from nemo
import logging
import os
import random
import sys
import tempfile
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nemo.collections.asr.models.confidence_ensemble import ConfidenceEnsembleModel
from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig,
    ConfidenceMethodConfig,
    get_confidence_aggregation_bank,
)
from nemo.core.config import hydra_runner

LOG = logging.getLogger(__file__)

# adding Python path. If not found, asking user to get the file
try:
    sys.path.append(str(Path(__file__).parents[2] / "examples" / "asr"))
    import transcribe_speech
except ImportError:
    # if users run script normally from nemo repo, this shouldn't be triggered as
    # we modify the path above. But if they downloaded the build_ensemble.py as
    # an isolated script, we'd ask them to also download corresponding version
    # of the transcribe_speech.py
    print(
        "Current script depends on 'examples/asr/transcribe_speech.py', but can't find it. "
        "If it's not present, download it from the NeMo github manually and put inside this folder."
    )


@dataclass
class EnsembleConfig:
    # .nemo path or pretrained name
    model: str
    # path to the training data manifest (non-tarred)
    training_manifest: str
    # specify to limit the number of training samples
    # 100 is most likely enough, but setting higher default just in case
    max_training_samples: int = 1000
    # specify to provide dev data manifest for HP tuning
    # dev_manifest: Optional[str] = None


@dataclass
class BuildEnsembleConfig:
    # where to save the resulting ensemble model
    output_path: str

    # each model specification
    ensemble: List[EnsembleConfig]

    random_seed: int = 0  # for reproducibility

    # default confidence, can override
    confidence: ConfidenceConfig = ConfidenceConfig(
        # we keep frame confidences and apply aggregation manually to get full-utterance confidence
        preserve_frame_confidence=True,
        exclude_blank=True,
        aggregation="mean",
        method_cfg=ConfidenceMethodConfig(
            name="entropy",
            entropy_type="renui",
            temperature=0.25,  # this is not really temperature, but alpha, see https://arxiv.org/abs/2212.08703
            entropy_norm="lin",
        ),
    )
    temperature: float = 1.0  # this is a real temperature that will be applied to logits

    # this is optional, but can be used to change any aspect of the transcription
    # config, such as batch size or amp usage. Note that model, data and confidence
    # will be overriden by this script
    transcription: transcribe_speech.TranscriptionConfig = transcribe_speech.TranscriptionConfig()


def calculate_score(features, labels, pipe):
    """Score is always calculated as mean of the per-class scores.

    This is done to account for possible class imbalances.
    """
    predictions = pipe.predict(features)
    conf_m = confusion_matrix(labels, predictions)
    score = np.diag(conf_m).sum() / conf_m.sum()
    return score, conf_m


def train_model_selection(
    training_features,
    training_labels,
    multi_class="multinomial",
    C=10000.0,  # disabling regularization by default as overfitting is likely not an issue
    class_weight="balanced",  # in case training data is imbalanced
    max_iter=1000,
):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(multi_class=multi_class, C=C, max_iter=max_iter, class_weight=class_weight),
    )
    pipe.fit(training_features, training_labels)

    accuracy, confusion = calculate_score(training_features, training_labels, pipe)

    LOG.info("Training fit accuracy: %.4f", accuracy * 100.0)
    LOG.info("Training confusion matrix:\n%s", str(confusion))
    return pipe


def subsample_manifest(manifest_file, max_samples):
    """Will save a subsampled version of the manifest to the same folder.

    Have to save to the same folder to support relative paths.
    """
    with open(manifest_file, "rt", encoding="utf-8") as fin:
        lines = fin.readlines()
    if max_samples < len(lines):
        lines = random.sample(lines, max_samples)
    output_file = manifest_file + "-subsampled"
    with open(output_file, "wt", encoding="utf-8") as fout:
        fout.write("".join(lines))
    return output_file


def cleanup_subsampled_manifests(subsampled_manifests):
    for manifest in subsampled_manifests:
        os.remove(manifest)


@hydra_runner(schema=BuildEnsembleConfig)
def main(cfg: BuildEnsembleConfig):
    # silencing all messages from nemo/ptl to avoid dumping tons of configs to the stdout
    logging.getLogger('pytorch_lightning').setLevel(logging.CRITICAL)
    logging.getLogger('nemo_logger').setLevel(logging.CRITICAL)
    LOG.info(f'Build ensemble config:\n{OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    # no matter what's in the config, frame confidence is required
    cfg.confidence.preserve_frame_confidence = True

    pl.seed_everything(cfg.random_seed)
    cfg.transcription.random_seed = None  # seed is already applied
    cfg.transcription.return_transcriptions = True
    cfg.transcription.ctc_decoding.confidence_cfg = cfg.confidence
    cfg.transcription.rnnt_decoding.confidence_cfg = cfg.confidence
    cfg.transcription.ctc_decoding.temperature = cfg.temperature
    cfg.transcription.rnnt_decoding.temperature = cfg.temperature

    aggregations = get_confidence_aggregation_bank()
    aggr_func = aggregations[cfg.confidence.aggregation]

    confidences = []
    labels = []

    # registering clean-up function that will hold on to this list and
    # should clean up even if there is partial error in some of the transcribe
    # calls
    subsampled_manifests = []
    atexit.register(cleanup_subsampled_manifests, subsampled_manifests)

    # note that we loop over the same config.
    # This is intentional, as we need to run all models on all datasets
    for model_idx, model_cfg in enumerate(cfg.ensemble):
        model_confidences = []
        for data_idx, data_cfg in enumerate(cfg.ensemble):
            if model_idx == 0:  # generating subsampled manifests only one time
                subsampled_manifests.append(
                    subsample_manifest(data_cfg.training_manifest, data_cfg.max_training_samples)
                )
            subsampled_manifest = subsampled_manifests[data_idx]

            if model_cfg.model.endswith(".nemo"):
                cfg.transcription.model_path = model_cfg.model
            else:  # assuming pretrained model
                cfg.transcription.pretrained_name = model_cfg.model

            cfg.transcription.dataset_manifest = subsampled_manifest

            with tempfile.NamedTemporaryFile() as output_file:
                cfg.transcription.output_filename = output_file.name
                LOG.info("Transcribing dataset %d with model %d", data_idx, model_idx)
                transcriptions = transcribe_speech.main(cfg.transcription.copy())

                for transcription in transcriptions:
                    if isinstance(transcription.frame_confidence[0], list):
                        # NeMo Transducer API returns list of lists for confidences
                        conf_values = [conf_value for confs in transcription.frame_confidence for conf_value in confs]
                    else:
                        conf_values = transcription.frame_confidence
                    model_confidences.append(aggr_func(conf_values))
                    if model_idx == 0:  # labels are the same for all models
                        labels.append(data_idx)

        confidences.append(model_confidences)

    # transposing with zip(*list)
    training_features = np.array(list(zip(*confidences)))
    training_labels = np.array(labels)
    model_selection_block = train_model_selection(training_features, training_labels)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_selection_block_path = os.path.join(tmpdir, 'model_selection_block.pkl')
        joblib.dump(model_selection_block, model_selection_block_path)

        # creating ensemble checkpoint
        ensemble_model = ConfidenceEnsembleModel(
            cfg=DictConfig(
                {
                    'model_selection_block': model_selection_block_path,
                    'confidence': cfg.confidence,
                    'temperature': cfg.temperature,
                    'load_models': [model_cfg.model for model_cfg in cfg.ensemble],
                }
            ),
            trainer=None,
        )
        ensemble_model.save_to(cfg.output_path)


if __name__ == '__main__':
    main()
