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

import json
import os
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import MISSING, OmegaConf
from sklearn.model_selection import ParameterGrid

from nemo.collections.asr.models import ASRModel, EncDecRNNTModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.asr_confidence_benchmarking_utils import (
    apply_confidence_parameters,
    run_confidence_benchmark,
)
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils

"""
Get confidence metrics and curve plots for a given model, dataset, and confidence parameters.

# Arguments
  model_path: Path to .nemo ASR checkpoint
  pretrained_name: Name of pretrained ASR model (from NGC registry)
  dataset_manifest: Path to dataset JSON manifest file (in NeMo format)
  output_dir: Output directory to store a report and curve plot directories
  
  batch_size: batch size during inference
  num_workers: number of workers during inference
  
  cuda: Optional int to enable or disable execution of model on certain CUDA device
  amp: Bool to decide if Automatic Mixed Precision should be used during inference
  audio_type: Str filetype of the audio. Supported = wav, flac, mp3
  
  target_level: Word- or token-level confidence. Supported = word, token, auto (for computing both word and token)
  confidence_cfg: Config with confidence parameters
  grid_params: Dictionary with lists of parameters to iteratively benchmark on

# Usage
ASR model can be specified by either "model_path" or "pretrained_name".
Data for transcription are defined with "dataset_manifest".
Results are returned as a benchmark report and curve plots.

python benchmark_asr_confidence.py \
    model_path=null \
    pretrained_name=null \
    dataset_manifest="" \
    output_dir="" \
    batch_size=64 \
    num_workers=8 \
    cuda=0 \
    amp=True \
    target_level="word" \
    confidence_cfg.exclude_blank=False \
    'grid_params="{\"aggregation\": [\"min\", \"prod\"], \"alpha\": [0.33, 0.5]}"'
"""


def get_experiment_params(cfg):
    """Get experiment parameters from a confidence config and generate the experiment name.

    Returns:
        List of experiment parameters.
        String with the experiment name.
    """
    blank = "no_blank" if cfg.exclude_blank else "blank"
    duration = "duration" if cfg.tdt_include_duration else "no_duration"
    aggregation = cfg.aggregation
    method_name = cfg.method_cfg.name
    alpha = cfg.method_cfg.alpha
    if method_name == "entropy":
        entropy_type = cfg.method_cfg.entropy_type
        entropy_norm = cfg.method_cfg.entropy_norm
        experiment_param_list = [
            aggregation,
            str(cfg.exclude_blank),
            str(cfg.tdt_include_duration),
            method_name,
            entropy_type,
            entropy_norm,
            str(alpha),
        ]
        experiment_str = "-".join([aggregation, blank, duration, method_name, entropy_type, entropy_norm, str(alpha)])
    else:
        experiment_param_list = [
            aggregation,
            str(cfg.exclude_blank),
            str(cfg.tdt_include_duration),
            method_name,
            "-",
            "-",
            str(alpha),
        ]
        experiment_str = "-".join([aggregation, blank, duration, method_name, str(alpha)])
    return experiment_param_list, experiment_str


@dataclass
class ConfidenceBenchmarkingConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    dataset_manifest: str = MISSING
    output_dir: str = MISSING

    # General configs
    batch_size: int = 32
    num_workers: int = 4

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    audio_type: str = "wav"

    # Confidence configs
    target_level: str = "auto"  # Choices: "word", "token", "auto" (for both word- and token-level confidence)
    confidence_cfg: ConfidenceConfig = field(
        default_factory=lambda: ConfidenceConfig(preserve_word_confidence=True, preserve_token_confidence=True)
    )
    grid_params: Optional[str] = None  # a dictionary with lists of parameters to iteratively benchmark on


@hydra_runner(config_name="ConfidenceBenchmarkingConfig", schema=ConfidenceBenchmarkingConfig)
def main(cfg: ConfidenceBenchmarkingConfig):
    torch.set_grad_enabled(False)

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
        else:
            device = 1
            accelerator = 'cpu'
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'

    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')

    # setup model
    if cfg.model_path is not None:
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        asr_model = imported_class.restore_from(
            restore_path=cfg.model_path, map_location=map_location
        )  # type: ASRModel
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(
            model_name=cfg.pretrained_name, map_location=map_location
        )  # type: ASRModel

    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    asr_model.set_trainer(trainer)
    asr_model = asr_model.eval()

    # Check if ctc or rnnt model
    is_rnnt = isinstance(asr_model, EncDecRNNTModel)

    # Check that the model has the `change_decoding_strategy` method
    if not hasattr(asr_model, 'change_decoding_strategy'):
        raise RuntimeError("The asr_model you are using must have the `change_decoding_strategy` method.")

    # get filenames and reference texts from manifest
    filepaths = []
    reference_texts = []
    if os.stat(cfg.dataset_manifest).st_size == 0:
        logging.error(f"The input dataset_manifest {cfg.dataset_manifest} is empty. Exiting!")
        return None
    manifest_dir = Path(cfg.dataset_manifest).parent
    with open(cfg.dataset_manifest, 'r') as f:
        for line in f:
            item = json.loads(line)
            audio_file = Path(item['audio_filepath'])
            if not audio_file.is_file() and not audio_file.is_absolute():
                audio_file = manifest_dir / audio_file
            filepaths.append(str(audio_file.absolute()))
            reference_texts.append(item['text'])

    # setup AMP (optional)
    autocast = None
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast

    # do grid-based benchmarking if grid_params is provided, otherwise a regular one
    work_dir = Path(cfg.output_dir)
    os.makedirs(work_dir, exist_ok=True)
    report_legend = (
        ",".join(
            [
                "model_type",
                "aggregation",
                "blank",
                "duration",
                "method_name",
                "entropy_type",
                "entropy_norm",
                "alpha",
                "target_level",
                "auc_roc",
                "auc_pr",
                "auc_nt",
                "nce",
                "ece",
                "auc_yc",
                "std_yc",
                "max_yc",
            ]
        )
        + "\n"
    )
    model_typename = "RNNT" if is_rnnt else "CTC"
    report_file = work_dir / Path("report.csv")
    if cfg.grid_params:
        asr_model.change_decoding_strategy(
            RNNTDecodingConfig(fused_batch_size=-1, strategy="greedy_batch", confidence_cfg=cfg.confidence_cfg)
            if is_rnnt
            else CTCDecodingConfig(confidence_cfg=cfg.confidence_cfg)
        )
        params = json.loads(cfg.grid_params)
        hp_grid = ParameterGrid(params)
        hp_grid = list(hp_grid)

        logging.info(f"==============================Running a benchmarking with grid search=========================")
        logging.info(f"Grid search size: {len(hp_grid)}")
        logging.info(f"Results will be written to:\nreport file `{report_file}`\nand plot directories near the file")
        logging.info(f"==============================================================================================")

        with open(report_file, "tw", encoding="utf-8") as f:
            f.write(report_legend)
            f.flush()
            for i, hp in enumerate(hp_grid):
                logging.info(f"Run # {i + 1}, grid: `{hp}`")
                asr_model.change_decoding_strategy(apply_confidence_parameters(asr_model.cfg.decoding, hp))
                param_list, experiment_name = get_experiment_params(asr_model.cfg.decoding.confidence_cfg)
                plot_dir = work_dir / Path(experiment_name)
                results = run_confidence_benchmark(
                    asr_model,
                    cfg.target_level,
                    filepaths,
                    reference_texts,
                    cfg.batch_size,
                    cfg.num_workers,
                    plot_dir,
                    autocast,
                )
                for level, result in results.items():
                    f.write(f"{model_typename},{','.join(param_list)},{level},{','.join([str(r) for r in result])}\n")
                    f.flush()
    else:
        asr_model.change_decoding_strategy(
            RNNTDecodingConfig(fused_batch_size=-1, strategy="greedy_batch", confidence_cfg=cfg.confidence_cfg)
            if is_rnnt
            else CTCDecodingConfig(confidence_cfg=cfg.confidence_cfg)
        )
        param_list, experiment_name = get_experiment_params(asr_model.cfg.decoding.confidence_cfg)
        plot_dir = work_dir / Path(experiment_name)

        logging.info(f"==============================Running a single benchmarking===================================")
        logging.info(f"Results will be written to:\nreport file `{report_file}`\nand plot directory `{plot_dir}`")

        with open(report_file, "tw", encoding="utf-8") as f:
            f.write(report_legend)
            f.flush()
            results = run_confidence_benchmark(
                asr_model,
                cfg.batch_size,
                cfg.num_workers,
                cfg.target_level,
                filepaths,
                reference_texts,
                plot_dir,
                autocast,
            )
            for level, result in results.items():
                f.write(f"{model_typename},{','.join(param_list)},{level},{','.join([str(r) for r in result])}\n")
    logging.info(f"===========================================Done===============================================")


if __name__ == '__main__':
    main()
