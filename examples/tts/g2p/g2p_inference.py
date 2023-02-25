# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from dataclasses import dataclass, is_dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from utils import get_metrics

from nemo.collections.tts.models.base import G2PModel
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
python g2p_inference.py \
    pretrained_model=<Path to .nemo file or pretrained model name for G2PModel from list_available_models()>" \
    manifest_filepath="<Path to .json manifest>" \
    output_file="<Path to .json manifest to save prediction>" \
    batch_size=32 \
    num_workers=4 \
    pred_field=pred_text
"""


@dataclass
class TranscriptionConfig:
    # Required configs
    pretrained_model: str  # Path to a .nemo file or Name of a pretrained model
    manifest_filepath: str  # Path to .json manifest file
    phoneme_field: Optional[
        str
    ] = None  # name of the field in manifest_filepath for ground truth phonemes, default during training "text"
    grapheme_field: Optional[str] = "text_graphemes"  # name of the field in manifest_filepath for input grapheme text

    # General configs
    output_file: Optional[
        str
    ] = None  # Path to .json manifest file to save predictions, will be saved in "target_field"
    pred_field: Optional[str] = "pred_text"  # name of the field in the output_file to save predictions
    batch_size: int = 32  # Batch size to use for inference
    num_workers: int = 0  # Number of workers to use for DataLoader during inference

    # Config for heteronyms correction
    pretrained_heteronyms_model: Optional[
        str
    ] = None  # Path to a .nemo file or a Name of a pretrained model to disambiguate heteronyms (Optional)


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> TranscriptionConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if not cfg.pretrained_model:
        raise ValueError(
            'To run evaluation and inference script a pre-trained model or .nemo file must be provided.'
            f'Choose from {G2PModel.list_available_models()} or "pretrained_model"="your_model.nemo"'
        )

    logging.info(
        'During evaluation/testing, it is currently advisable to construct a new Trainer with single GPU and \
            no DDP to obtain accurate results'
    )

    # setup GPU
    if torch.cuda.is_available():
        device = [0]  # use 0th CUDA device
        accelerator = 'gpu'
    else:
        device = 1
        accelerator = 'cpu'

    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')
    trainer = pl.Trainer(devices=device, accelerator=accelerator, logger=False, enable_checkpointing=False)

    if os.path.exists(cfg.pretrained_model):
        model = G2PModel.restore_from(cfg.pretrained_model, map_location=map_location)
    elif cfg.pretrained_model in G2PModel.get_available_model_names():
        model = G2PModel.from_pretrained(cfg.pretrained_model, map_location=map_location)
    else:
        raise ValueError(
            f'Provide path to the pre-trained .nemo checkpoint or choose from {G2PModel.list_available_models()}'
        )
    model._cfg.max_source_len = 512
    model.set_trainer(trainer)
    model = model.eval()

    if cfg.output_file is None:
        cfg.output_file = cfg.manifest_filepath.replace(".json", "_phonemes.json")

    with torch.no_grad():
        model.convert_graphemes_to_phonemes(
            manifest_filepath=cfg.manifest_filepath,
            output_manifest_filepath=cfg.output_file,
            grapheme_field=cfg.grapheme_field,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pred_field=cfg.pred_field,
        )
        print(f"IPA predictions saved in {cfg.output_file}")

        if cfg.phoneme_field is not None:
            get_metrics(cfg.output_file, phoneme_field=cfg.phoneme_field, grapheme_field=cfg.grapheme_field)


if __name__ == '__main__':
    main()
