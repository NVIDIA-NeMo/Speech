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

import contextlib
import json
import os
from dataclasses import dataclass, is_dataclass
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.modules.conformer_encoder import ConformerChangeConfig
from nemo.collections.asr.parts.utils.transcribe_utils import compute_output_filename, prepare_audio_data, setup_model
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
Translate audio file on a single CPU/GPU. Useful for translations of moderate amounts of audio data.

# Arguments
  model_path: path to .nemo ST checkpoint
  pretrained_name: name of pretrained ST model (from NGC registry)
  audio_dir: path to directory with audio files
  dataset_manifest: path to dataset JSON manifest file (in NeMo format)

  output_filename: Output filename where the translations will be written
  batch_size: batch size during inference

  cuda: Optional int to enable or disable execution of model on certain CUDA device.
  allow_mps: Bool to allow using MPS (Apple Silicon M-series GPU) device if available
  amp: Bool to decide if Automatic Mixed Precision should be used during inference
  audio_type: Str filetype of the audio. Supported = wav, flac, mp3

  overwrite_translations: Bool which when set allows repeated translations to overwrite previous results.

# Usage
ST model can be specified by either "model_path" or "pretrained_name".
Data for translation can be defined with either "audio_dir" or "dataset_manifest".
Results are returned in a JSON manifest file.

python translate_speech.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="<remove or path to folder of audio files>" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    batch_size=32 \
    cuda=0 \
    amp=True \
"""


@dataclass
class ModelChangeConfig:

    # Sub-config for changes specific to the Conformer Encoder
    conformer: ConformerChangeConfig = ConformerChangeConfig()


@dataclass
class TranslationConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    audio_key: str = 'audio_filepath'  # Used to override the default audio key in dataset_manifest
    eval_config_yaml: Optional[str] = None  # Path to a yaml file of config of evaluation

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    amp: bool = False
    audio_type: str = "wav"

    # Recompute model translation, even if the output folder exists with scores.
    overwrite_translations: bool = True

    # can be set to True to return list of translations instead of the config
    # if True, will also skip writing anything to the output file
    return_translations: bool = False


@hydra_runner(config_name="TranslationConfig", schema=TranslationConfig)
def main(cfg: TranslationConfig) -> Union[TranslationConfig, List[str]]:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    # Load augmentor from exteranl yaml file which contains eval info, could be extend to other feature such VAD, P&C
    augmentor = None
    if cfg.eval_config_yaml:
        eval_config = OmegaConf.load(cfg.eval_config_yaml)
        augmentor = eval_config.test_ds.get("augmentor")
        logging.info(f"Will apply on-the-fly augmentation on samples during translation: {augmentor} ")

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.warning(
                "MPS device (Apple Silicon M-series GPU) support is experimental."
                " Env variable `PYTORCH_ENABLE_MPS_FALLBACK=1` should be set in most cases to avoid failures."
            )
            device = [0]
            accelerator = 'mps'
            map_location = torch.device('mps')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    logging.info(f"Inference will be done on device: {map_location}")

    asr_model, model_name = setup_model(cfg, map_location)
    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    asr_model.set_trainer(trainer)
    asr_model = asr_model.eval()

    # collect additional translation information
    return_hypotheses = False

    # prepare audio filepaths and decide wether it's partial audio
    filepaths, partial_audio = prepare_audio_data(cfg)

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    # Compute output filename
    cfg = compute_output_filename(cfg, model_name)

    # if translations should not be overwritten, and already exists, skip re-translation step and return
    if not cfg.return_translations and not cfg.overwrite_translations and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous translations found at {cfg.output_filename}, and flag `overwrite_translations`"
            f"is {cfg.overwrite_translations}. Returning without re-translating text."
        )
        return cfg

    # translate audio
    with autocast():
        with torch.no_grad():
            translations = asr_model.translate(
                paths2audio_files=filepaths, batch_size=cfg.batch_size, return_hypotheses=return_hypotheses,
            )

    logging.info(f"Finished translating {len(filepaths)} files !")
    logging.info(f"Writing translations into file: {cfg.output_filename}")

    if cfg.return_translations:
        return translations

    # write audio translations
    with open(cfg.output_filename, 'w', encoding='utf-8', newline='\n') as f:
        for filepath, translation in zip(filepaths, translations):
            item = {'audio_filepath': filepath, 'pred_translation': translation}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logging.info(f"Finished writing predictions to {cfg.output_filename}!")

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
