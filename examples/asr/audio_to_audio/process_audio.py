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

import contextlib
import glob
import json
import os
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import AudioToAudioModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils


"""
Process audio file on a single CPU/GPU. Useful for processing of moderate amounts of audio data.

# Arguments
    model_path: path to .nemo checkpoint for an AudioToAudioModel
    pretrained_name: name of a pretrained AudioToAudioModel model (from NGC registry)
    audio_dir: path to directory with audio files
    dataset_manifest: path to dataset JSON manifest file (in NeMo format)

    input_channel_selector: list of channels to take from audio files, defaults to `None` and takes all available channels
    input_key: key for audio filepath in the manifest file, defaults to `audio_filepath`

    output_dir: Directory where processed files will be saved
    output_filename: Output filename where manifest pointing to processed files will be written
    batch_size: batch size during inference

    cuda: Optional int to enable or disable execution of model on certain CUDA device.
    amp: Bool to decide if Automatic Mixed Precision should be used during inference
    audio_type: Str filetype of the audio. Supported = wav, flac, mp3

    overwrite_output: Bool which when set allowes repeated processing runs to overwrite previous results.

# Usage
AudioToAudioModel can be specified by either `model_path` or `pretrained_name`.
Data for processing can be defined with either `audio_dir` or `dataset_manifest`.
Processed audio is saved in `output_dir`, and a manifest for processed files is saved
in `output_filename`.

```
python process_audio.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="" \
    dataset_manifest="" \
    input_channel_selector=[] \
    output_dir="" \
    output_filename="" \
    batch_size=1 \
    cuda=0 \
    amp=True
```
"""


@dataclass
class ProcessConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    # Audio configs
    input_channel_selector: Optional[List] = None  # Union types not supported Optional[Union[List, int]]
    input_key: Optional[str] = None  # Can be used with a manifest

    # General configs
    output_dir: Optional[str] = None
    output_filename: Optional[str] = None
    batch_size: int = 1
    num_workers: int = 0

    # Override model config
    override_config_path: Optional[str] = None  # path to a yaml config that will override the internal config file

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    audio_type: str = "wav"

    # Recompute model predictions, even if the output folder exists.
    overwrite_output: bool = False


@hydra_runner(config_name="ProcessConfig", schema=ProcessConfig)
def main(cfg: ProcessConfig) -> ProcessConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

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
        model_cfg = AudioToAudioModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: AudioToAudioModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        audio_to_audio_model = imported_class.restore_from(
            restore_path=cfg.model_path, override_config_path=cfg.override_config_path, map_location=map_location
        )  # type: AudioToAudioModel
        model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    else:
        # restore model by name
        audio_to_audio_model = AudioToAudioModel.from_pretrained(
            model_name=cfg.pretrained_name, map_location=map_location
        )  # type: AudioToAudioModel
        model_name = cfg.pretrained_name

    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    audio_to_audio_model.set_trainer(trainer)
    audio_to_audio_model = audio_to_audio_model.eval()

    if cfg.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True))
    else:
        # get filenames from manifest
        filepaths = []
        if os.stat(cfg.dataset_manifest).st_size == 0:
            raise RuntimeError(f"The input dataset_manifest {cfg.dataset_manifest} is empty.")

        input_key = 'audio_filepath' if cfg.input_key is None else cfg.input_key
        manifest_dir = Path(cfg.dataset_manifest).parent
        with open(cfg.dataset_manifest, 'r') as f:
            for line in f:
                item = json.loads(line)
                audio_file = Path(item[input_key])
                if not audio_file.is_file() and not audio_file.is_absolute():
                    audio_file = manifest_dir / audio_file
                filepaths.append(str(audio_file.absolute()))

    logging.info(f"\nProcessing {len(filepaths)} files...\n")

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    # Compute output filename
    if cfg.output_dir is None:
        # create default output filename
        if cfg.audio_dir is not None:
            cfg.output_dir = os.path.dirname(os.path.join(cfg.audio_dir, '.')) + f'_processed_{model_name}'
        else:
            cfg.output_dir = os.path.dirname(cfg.dataset_manifest) + f'_processed_{model_name}'

    # Compute output filename
    if cfg.output_filename is None:
        # create default output filename
        cfg.output_filename = cfg.output_dir.rstrip('/') + '_manifest.json'

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.overwrite_output and os.path.exists(cfg.output_dir):
        raise RuntimeError(
            f"Previous output found at {cfg.output_dir}, and flag `overwrite_output`"
            f"is {cfg.overwrite_output}. Returning without processing."
        )

    # Process audio
    with autocast():
        with torch.no_grad():
            paths2processed_files = audio_to_audio_model.process(
                paths2audio_files=filepaths,
                output_dir=cfg.output_dir,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                input_channel_selector=cfg.input_channel_selector,
            )

    logging.info(f"Finished processing {len(filepaths)} files!")
    logging.info(f"Processed audio is available in the output directory: {cfg.output_dir}")

    # Prepare new/updated manifest with a new key for processed audio
    with open(cfg.output_filename, 'w', encoding='utf-8') as f:
        if cfg.dataset_manifest is not None:
            with open(cfg.dataset_manifest, 'r') as fr:
                for idx, line in enumerate(fr):
                    item = json.loads(line)
                    item['processed_audio_filepath'] = paths2processed_files[idx]
                    f.write(json.dumps(item) + "\n")
        else:
            for idx, processed_file in enumerate(paths2processed_files):
                item = {'processed_audio_filepath': processed_file}
                f.write(json.dumps(item) + "\n")

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
