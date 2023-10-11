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

"""
This script provides a functionality to create confidence-based ensembles
from a collection of pretrained models.

For more details see the paper https://arxiv.org/abs/2306.15824
or tutorial in tutorials/asr/Confidence_Ensembles.ipynb

You would typically use this script by providing a yaml config file or overriding
default options from command line.

Usage examples:

1. Building an ensemble of two monolingual models with default settings (no confidence tuning).

    python build_ensemble.py --config-path=. --config-name=ensemble_config.yaml
        ensemble.0.model=stt_it_conformer_ctc_large
        ensemble.0.training_manifest=<path to the Italian data of 100+ utterances (no transcription required)>
        ensemble.1.model=stt_es_conformer_ctc_large
        ensemble.1.training_manifest=<path to the Spanish data of 100+ utterances (no transcription required)>
        output_path=<path to the desired location of the .nemo checkpoint>

    You can have more than 2 models and can control transcription settings (e.g., batch size)
    with ``transcription.<any argument of examples/asr/transcribe_speech.py>`` parameters.

2. If you want to get improved results, you can enable tuning of the confidence and logistic regression (LR) parameters.
   E.g.

   python build_ensemble.py
        <all arguments like in the previous example>
        ensemble.0.dev_manifest=<path to the dev data that's required for tuning>
        ...
        # IMPORTANT: see the note below if you use > 2 models!
        ensemble.N.dev_manifest=<path to the dev data that's required for tuning>
        tune_confidence=True  # to allow confidence tuning. LR is tuned by default

    As with any tuning, it is recommended to have reasonably large validation set for each model,
    otherwise you might overfit to the validation data.

    Note that if you add additional models (> 2) you will need to modify ensemble_config.yaml
    or create a new one with added models in there. While it's theoretically possible to
    fully override such parameters from commandline, hydra is very unfriendly for such
    use-cases, so it's strongly recommended to be creating new configs.

3. If you want to precisely control tuning grid search, you can do that with

    python build_ensemble.py
        <all arguments as in the previous examples>
        tune_confidence_config.confidence_type='[entropy_renyi_exp,entropy_tsallis_exp]'  # only tune over this set
        tune_confidence_config.alpha='[0.1,0.5,1.0]'  # only tune over this set

You can check the dataclasses in this file for the full list of supported
arguments and their default values.
"""

import atexit

# using default logging to be able to silence unnecessary messages from nemo
import logging
import os
import random
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pytorch_lightning as pl
from omegaconf import MISSING, DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from nemo.collections.asr.models.confidence_ensemble import (
    ConfidenceEnsembleModel,
    ConfidenceSpec,
    compute_confidence,
    get_filtered_logprobs,
)
from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig,
    ConfidenceMethodConfig,
    get_confidence_aggregation_bank,
    get_confidence_measure_bank,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
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
    model: str = MISSING
    # path to the training data manifest (non-tarred)
    training_manifest: str = MISSING
    # specify to limit the number of training samples
    # 100 is most likely enough, but setting higher default just in case
    max_training_samples: int = 1000
    # specify to provide dev data manifest for HP tuning
    dev_manifest: Optional[str] = None


@dataclass
class TuneConfidenceConfig:
    # important parameter, so should always be tuned
    exclude_blank: Tuple[bool] = (True, False)
    # prod is pretty much always worse, so not including by default
    aggregation: Tuple[str] = ("mean", "min", "max")
    # not including max prob, as there is always an entropy-based metric
    # that's better but otherwise including everything
    confidence_type: Tuple[str] = (
        "entropy_renyi_exp",
        "entropy_renyi_lin",
        "entropy_tsallis_exp",
        "entropy_tsallis_lin",
        "entropy_gibbs_lin",
        "entropy_gibbs_exp",
    )

    # TODO: currently it's not possible to efficiently tune temperature, as we always
    #    apply log-softmax in the decoder, so to try different values it will be required
    #    to rerun the decoding, which is very slow. To support this for one-off experiments
    #    it's possible to modify the code of CTC decoder / Transducer joint to
    #    remove log-softmax and then apply it directly in this script with the temperature
    #
    #    Alternatively, one can run this script multiple times with different values of
    #    temperature and pick the best performing ensemble. Note that this will increase
    #    tuning time by the number of temperature values tried. On the other hand,
    #    the above approach is a lot more efficient and will only slightly increase
    #    the total tuning runtime.

    # very important to tune for max prob, but for entropy metrics 1.0 is almost always best
    # temperature: Tuple[float] = (1.0,)

    # not that important, but can sometimes make a small difference
    alpha: Tuple[float] = (0.25, 0.33, 0.5, 1.0)

    def get_grid_size(self) -> int:
        """Returns the total number of points in the search space."""
        if "max_prob" in self.confidence_type:
            return (
                len(self.exclude_blank)
                * len(self.aggregation)
                * ((len(self.confidence_type) - 1) * len(self.alpha) + 1)
            )
        return len(self.exclude_blank) * len(self.aggregation) * len(self.confidence_type) * len(self.alpha)


@dataclass
class TuneLogisticRegressionConfig:
    # will have log-uniform grid over this range with that many points
    # note that a value of 10000.0 (not regularization) is always added
    C_num_points: int = 10
    C_min: float = 0.0001
    C_max: float = 10.0

    # not too important
    multi_class: Tuple[str] = ("ovr", "multinomial")

    # should try to include weights directly if the data is too imbalanced
    class_weight: Tuple = (None, "balanced")

    # increase if getting many warnings that algorithm didn't converge
    max_iter: int = 1000


@dataclass
class BuildEnsembleConfig:
    # where to save the resulting ensemble model
    output_path: str = MISSING

    # each model specification
    ensemble: List[EnsembleConfig] = MISSING

    random_seed: int = 0  # for reproducibility

    # default confidence, can override
    confidence: ConfidenceConfig = field(
        default_factory=lambda: ConfidenceConfig(
            # we keep frame confidences and apply aggregation manually to get full-utterance confidence
            preserve_frame_confidence=True,
            exclude_blank=True,
            aggregation="mean",
            method_cfg=ConfidenceMethodConfig(name="entropy", entropy_type="renyi", alpha=0.25, entropy_norm="lin",),
        )
    )
    temperature: float = 1.0

    # this is optional, but can be used to change any aspect of the transcription
    # config, such as batch size or amp usage. Note that model, data and confidence
    # will be overriden by this script
    transcription: transcribe_speech.TranscriptionConfig = field(
        default_factory=lambda: transcribe_speech.TranscriptionConfig()
    )

    # set to True to tune the confidence.
    # requires dev manifests to be specified for each model
    tune_confidence: bool = False
    # used to specify what to tune over. By default runs tuning over some
    # reasonalbe grid, so that it does not take forever.
    # Can be changed as needed
    tune_confidence_config: TuneConfidenceConfig = field(default_factory=lambda: TuneConfidenceConfig())

    # very fast to tune and can be important in case of imbalanced datasets
    # will automatically set to False if dev data is not available
    tune_logistic_regression: bool = True
    tune_logistic_regression_config: TuneLogisticRegressionConfig = field(
        default_factory=lambda: TuneLogisticRegressionConfig()
    )

    def __post_init__(self):
        """Checking that if any dev data is provided, all are provided.

        Will also auto-set tune_logistic_regression to False if no dev data
        is available.

        If tune_confidence is set to True (user choice) and no dev data is
        provided, will raise an error.
        """
        num_dev_data = 0
        for ensemble_cfg in self.ensemble:
            num_dev_data += ensemble_cfg.dev_manifest is not None
        if num_dev_data == 0:
            if self.tune_confidence:
                raise ValueError("tune_confidence is set to True, but no dev data is provided")
            LOG.info("Setting tune_logistic_regression = False since no dev data is provided")
            self.tune_logistic_regression = False
            return

        if num_dev_data < len(self.ensemble):
            raise ValueError(
                "Some ensemble configs specify dev data, but some don't. Either all have to specify it or none!"
            )


def calculate_score(features: np.ndarray, labels: np.ndarray, pipe: Pipeline) -> Tuple[float, np.ndarray]:
    """Score is always calculated as mean of the per-class scores.

    This is done to account for possible class imbalances.

    Args:
        features: numpy array of features of shape [N x D], where N is the
            number of objects (typically a total number of utterances in
            all datasets) and D is the total number of confidence scores
            used to train the model (typically = number of models).
        labels: numpy array of shape [N] contatining ground-truth model indices.
        pipe: classification pipeline (currently, standardization + logistic
            regression).

    Returns:
        tuple: score value in [0, 1] and full classification confusion matrix.
    """
    predictions = pipe.predict(features)
    conf_m = confusion_matrix(labels, predictions)
    score = np.diag(conf_m).sum() / conf_m.sum()
    return score, conf_m


def train_model_selection(
    training_features: np.ndarray,
    training_labels: np.ndarray,
    dev_features: Optional[np.ndarray] = None,
    dev_labels: Optional[np.ndarray] = None,
    tune_lr: bool = False,
    tune_lr_cfg: Optional[TuneLogisticRegressionConfig] = None,
    verbose: bool = False,
) -> Tuple[Pipeline, float]:
    """Trains model selection block with an (optional) tuning of the parameters.

    Returns a pipeline consisting of feature standardization and logistic
    regression. If tune_lr is set to True, dev features/labels will be used
    to tune the hyperparameters of the logistic regression with the grid
    search that's defined via ``tune_lr_cfg``.

    If no tuning is requested, uses the following parameters::

        best_pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                multi_class="multinomial",
                C=10000.0,
                max_iter=1000,
                class_weight="balanced",
            ),
        )

    Args:
        training_features: numpy array of features of shape [N x D], where N is
            the number of objects (typically a total number of utterances in
            all training datasets) and D is the total number of confidence
            scores used to train the model (typically = number of models).
        training_labels: numpy array of shape [N] contatining ground-truth
            model indices.
        dev_features: same as training, but for the validation subset.
        dev_labels: same as training, but for the validation subset.
        tune_lr: controls whether tuning of LR hyperparameters is performed.
            If set to True, it's required to also provide dev features/labels.
        tune_lr_cfg: specifies what values of LR hyperparameters to try.
        verbose: if True, will output final training/dev scores.

    Returns:
        tuple: trained model selection pipeline, best score (or -1 if no tuning
        was done).
    """
    if not tune_lr:
        # default parameters: C=10000.0 disables regularization
        best_pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(multi_class="multinomial", C=10000.0, max_iter=1000, class_weight="balanced"),
        )
        max_score = -1
    else:
        C_pms = np.append(
            np.exp(np.linspace(np.log(tune_lr_cfg.C_min), np.log(tune_lr_cfg.C_max), tune_lr_cfg.C_num_points)),
            10000.0,
        )
        max_score = 0
        best_pipe = None
        for class_weight in tune_lr_cfg.class_weight:
            for multi_class in tune_lr_cfg.multi_class:
                for C in C_pms:
                    pipe = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            multi_class=multi_class, C=C, max_iter=tune_lr_cfg.max_iter, class_weight=class_weight
                        ),
                    )
                    pipe.fit(training_features, training_labels)
                    score, confusion = calculate_score(dev_features, dev_labels, pipe)
                    if score > max_score:
                        max_score = score
                        best_pipe = pipe

    best_pipe.fit(training_features, training_labels)
    if verbose:
        accuracy, confusion = calculate_score(training_features, training_labels, best_pipe)
        LOG.info("Training fit accuracy: %.4f", accuracy * 100.0)
        LOG.info("Training confusion matrix:\n%s", str(confusion))
    if dev_features is not None and verbose:
        accuracy, confusion = calculate_score(dev_features, dev_labels, best_pipe)
        LOG.info("Dev fit accuracy: %.4f", accuracy * 100.0)
        LOG.info("Dev confusion matrix:\n%s", str(confusion))

    return best_pipe, max_score


def subsample_manifest(manifest_file: str, max_samples: int) -> str:
    """Will save a subsampled version of the manifest to the same folder.

    Have to save to the same folder to support relative paths.

    Args:
        manifest_file: path to the manifest file that needs subsampling.
        max_samples: how many samples to retain. Will randomly select that
            many lines from the manifest.

    Returns:
        str: the path to the subsampled manifest file.
    """
    with open(manifest_file, "rt", encoding="utf-8") as fin:
        lines = fin.readlines()
    if max_samples < len(lines):
        lines = random.sample(lines, max_samples)
    output_file = manifest_file + "-subsampled"
    with open(output_file, "wt", encoding="utf-8") as fout:
        fout.write("".join(lines))
    return output_file


def cleanup_subsampled_manifests(subsampled_manifests: List[str]):
    """Removes all generated subsamples manifests."""
    for manifest in subsampled_manifests:
        os.remove(manifest)


def compute_all_confidences(
    hypothesis: Hypothesis, tune_confidence_cfg: TuneConfidenceConfig
) -> Dict[ConfidenceSpec, float]:
    """Computes a set of confidence scores from a given hypothesis.

    Works with the output of both CTC and Transducer decoding.

    Args:
        hypothesis: generated hypothesis as returned from the transcribe
            method of the ASR model.
        tune_confidence_cfg: config specifying what confidence scores to
            compute.

    Returns:
        dict: dictionary with confidenct spec -> confidence score mapping.
    """
    conf_values = {}

    for exclude_blank in tune_confidence_cfg.exclude_blank:
        filtered_logprobs = get_filtered_logprobs(hypothesis, exclude_blank)
        vocab_size = filtered_logprobs.shape[1]
        for aggregation in tune_confidence_cfg.aggregation:
            aggr_func = get_confidence_aggregation_bank()[aggregation]
            for conf_type in tune_confidence_cfg.confidence_type:
                conf_func = get_confidence_measure_bank()[conf_type]
                if conf_type == "max_prob":  # skipping alpha in this case
                    conf_value = aggr_func(conf_func(filtered_logprobs, v=vocab_size, t=1.0)).cpu().item()
                    conf_values[ConfidenceSpec(exclude_blank, aggregation, conf_type, 1.0)] = conf_value
                else:
                    for alpha in tune_confidence_cfg.alpha:
                        conf_value = aggr_func(conf_func(filtered_logprobs, v=vocab_size, t=alpha)).cpu().item()
                        conf_values[ConfidenceSpec(exclude_blank, aggregation, conf_type, alpha)] = conf_value

    return conf_values


def find_best_confidence(
    train_confidences: List[List[Dict[ConfidenceSpec, float]]],
    train_labels: List[int],
    dev_confidences: List[List[Dict[ConfidenceSpec, float]]],
    dev_labels: List[int],
    tune_lr: bool,
    tune_lr_config: TuneConfidenceConfig,
) -> Tuple[ConfidenceConfig, Pipeline]:
    """Finds the best confidence configuration for model selection.

    Will loop over all values in the confidence dictionary and fit the LR
    model (optionally tuning its HPs). The best performing confidence (on the
    dev set) will be used for the final LR model.

    Args:
        train_confidences: this is an object of type
            ``List[List[Dict[ConfidenceSpec, float]]]``. The shape of this
            object is [M, N, S], where
                M: number of models
                N: number of utterances in all training sets
                S: number of confidence scores to try

            This argument will be used to construct np.array objects for each
            of the confidence scores with the shape [M, N]

        train_labels: ground-truth labels of the correct model for each data
            points. This is a list of size [N]
        dev_confidences: same as training, but for the validation subset.
        dev_labels: same as training, but for the validation subset.
        tune_lr: controls whether tuning of LR hyperparameters is performed.
        tune_lr_cfg: specifies what values of LR hyperparameters to try.

    Returns:
        tuple: best confidence config, best model selection pipeline
    """
    max_score = 0
    best_pipe = None
    best_conf_spec = None
    LOG.info("Evaluation all confidences. Total grid size: %d", len(train_confidences[0][0].keys()))
    for conf_spec in tqdm(train_confidences[0][0].keys()):
        cur_train_confidences = []
        for model_confs in train_confidences:
            cur_train_confidences.append([])
            for model_conf in model_confs:
                cur_train_confidences[-1].append(model_conf[conf_spec])
        cur_dev_confidences = []
        for model_confs in dev_confidences:
            cur_dev_confidences.append([])
            for model_conf in model_confs:
                cur_dev_confidences[-1].append(model_conf[conf_spec])
        # transposing with zip(*list)
        training_features = np.array(list(zip(*cur_train_confidences)))
        training_labels = np.array(train_labels)
        dev_features = np.array(list(zip(*cur_dev_confidences)))
        dev_labels = np.array(dev_labels)
        pipe, score = train_model_selection(
            training_features, training_labels, dev_features, dev_labels, tune_lr, tune_lr_config,
        )
        if max_score < score:
            max_score = score
            best_pipe = pipe
            best_conf_spec = conf_spec
            LOG.info("Found better parameters: %s. New score: %.4f", str(conf_spec), max_score)

    return best_conf_spec.to_confidence_config(), best_pipe


@hydra_runner(config_name="BuildEnsembleConfig", schema=BuildEnsembleConfig)
def main(cfg: BuildEnsembleConfig):
    # silencing all messages from nemo/ptl to avoid dumping tons of configs to the stdout
    logging.getLogger('pytorch_lightning').setLevel(logging.CRITICAL)
    logging.getLogger('nemo_logger').setLevel(logging.CRITICAL)
    LOG.info(f'Build ensemble config:\n{OmegaConf.to_yaml(cfg)}')

    # to ensure post init is called
    cfg = BuildEnsembleConfig(**cfg)

    pl.seed_everything(cfg.random_seed)
    cfg.transcription.random_seed = None  # seed is already applied
    cfg.transcription.return_transcriptions = True
    cfg.transcription.preserve_alignment = True
    cfg.transcription.ctc_decoding.temperature = cfg.temperature
    cfg.transcription.rnnt_decoding.temperature = cfg.temperature
    # this ensures that generated output is after log-softmax for consistency with CTC

    train_confidences = []
    dev_confidences = []
    train_labels = []
    dev_labels = []

    # registering clean-up function that will hold on to this list and
    # should clean up even if there is partial error in some of the transcribe
    # calls
    subsampled_manifests = []
    atexit.register(cleanup_subsampled_manifests, subsampled_manifests)

    # note that we loop over the same config.
    # This is intentional, as we need to run all models on all datasets
    # this loop will do the following things:
    # 1. Goes through each model X each training dataset
    # 2. Computes predictions by directly calling transcribe_speech.main
    # 3. Converts transcription to the confidence score(s) as specified in the config
    # 4. If dev sets are provided, computes the same for them
    # 5. Creates a list of ground-truth model indices by mapping each model
    #    to its own training dataset as specified in the config.
    # 6. After the loop, we either run tuning over all confidence scores or
    #    directly use a single score to fit logistic regression and save the
    #    final ensemble model.
    for model_idx, model_cfg in enumerate(cfg.ensemble):
        train_model_confidences = []
        dev_model_confidences = []
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

            # training
            with tempfile.NamedTemporaryFile() as output_file:
                cfg.transcription.output_filename = output_file.name
                LOG.info("Transcribing training dataset %d with model %d", data_idx, model_idx)
                transcriptions = transcribe_speech.main(deepcopy(cfg.transcription))
                LOG.info("Generating confidence scores")
                # TODO: parallelize this loop?
                for transcription in tqdm(transcriptions):
                    if cfg.tune_confidence:
                        train_model_confidences.append(
                            compute_all_confidences(transcription, cfg.tune_confidence_config)
                        )
                    else:
                        train_model_confidences.append(compute_confidence(transcription, cfg.confidence))
                    if model_idx == 0:  # labels are the same for all models
                        train_labels.append(data_idx)

            # optional dev
            if data_cfg.dev_manifest is not None:
                cfg.transcription.dataset_manifest = data_cfg.dev_manifest
                with tempfile.NamedTemporaryFile() as output_file:
                    cfg.transcription.output_filename = output_file.name
                    LOG.info("Transcribing dev dataset %d with model %d", data_idx, model_idx)
                    transcriptions = transcribe_speech.main(deepcopy(cfg.transcription))
                    LOG.info("Generating confidence scores")
                    for transcription in tqdm(transcriptions):
                        if cfg.tune_confidence:
                            dev_model_confidences.append(
                                compute_all_confidences(transcription, cfg.tune_confidence_config)
                            )
                        else:
                            dev_model_confidences.append(compute_confidence(transcription, cfg.confidence))
                        if model_idx == 0:  # labels are the same for all models
                            dev_labels.append(data_idx)

        train_confidences.append(train_model_confidences)
        if dev_model_confidences:
            dev_confidences.append(dev_model_confidences)

    if cfg.tune_confidence:
        best_confidence, model_selection_block = find_best_confidence(
            train_confidences,
            train_labels,
            dev_confidences,
            dev_labels,
            cfg.tune_logistic_regression,
            cfg.tune_logistic_regression_config,
        )
    else:
        best_confidence = cfg.confidence
        # transposing with zip(*list)
        training_features = np.array(list(zip(*train_confidences)))
        training_labels = np.array(train_labels)
        if dev_confidences:
            dev_features = np.array(list(zip(*dev_confidences)))
            dev_labels = np.array(dev_labels)
        else:
            dev_features = None
            dev_labels = None
        model_selection_block, _ = train_model_selection(
            training_features,
            training_labels,
            dev_features,
            dev_labels,
            cfg.tune_logistic_regression,
            cfg.tune_logistic_regression_config,
            verbose=True,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_selection_block_path = os.path.join(tmpdir, 'model_selection_block.pkl')
        joblib.dump(model_selection_block, model_selection_block_path)

        # creating ensemble checkpoint
        ensemble_model = ConfidenceEnsembleModel(
            cfg=DictConfig(
                {
                    'model_selection_block': model_selection_block_path,
                    'confidence': best_confidence,
                    'temperature': cfg.temperature,
                    'load_models': [model_cfg.model for model_cfg in cfg.ensemble],
                }
            ),
            trainer=None,
        )
        ensemble_model.save_to(cfg.output_path)


if __name__ == '__main__':
    main()
