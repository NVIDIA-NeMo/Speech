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


import json
import os
from dataclasses import dataclass, is_dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.g2p.models.heteronym_classification import HeteronymClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
This script runs inference with HeteronymClassificationModel
If the input manifest contains target "word_id", evaluation will be also performed.

To prepare dataset, see NeMo/scripts/dataset_processing/g2p/export_wikihomograph_data_to_manifest.py

Inference form manifest:

python g2p_heteronym_classification_inference.py \
    manifest="<Path to .json manifest>" \
    pretrained_model="<Path to .nemo file or pretrained model name from list_available_models()>" \
    output_manifest="<Path to .json manifest to save prediction>" \
    wordid_to_phonemes_file="<Path to a file with mapping from wordid predicted by the model to phonemes>"

Interactive inference:

python g2p_heteronym_classification_inference.py \
    pretrained_model="<Path to .nemo file or pretrained model name from list_available_models()>" \
    wordid_to_phonemes_file="<Path to a file with mapping from wordid predicted by the model to phonemes>" # Optional
        
"""


@dataclass
class TranscriptionConfig:
    # Required configs
    pretrained_model: str  # Path to a .nemo file or Name of a pretrained model

    # path to .json manifest inference, if not provided, interactive mode will be enabled
    manifest: Optional[str] = None  # Path to .json manifest
    output_manifest: Optional[
        str
    ] = "predictions.json"  # Path to .json manifest to save prediction, will be saved in "pred_text" field
    grapheme_field: str = "text_graphemes"  # name of the field in .json manifest for input grapheme text

    # mapping from wordid predicted by the model to phonemes, e.g.,
    # "../../../scripts/tts_dataset_files/wordid_to_ipa-0.7b_nv22.10.tsv"
    wordid_to_phonemes_file: Optional[str] = None

    # if "word_id" targets are present in the manifest, evaluation will be performed and errors will be saved in errors_file
    errors_file: Optional[str] = None  # path to a file to save prediction errors
    batch_size: int = 32
    num_workers: int = 0


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if not cfg.pretrained_model:
        raise ValueError(
            'To run evaluation and inference script a pre-trained model or .nemo file must be provided.'
            f'Choose from {HeteronymClassificationModel.list_available_models()} or "pretrained_model"="your_model.nemo"'
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
        model = HeteronymClassificationModel.restore_from(cfg.pretrained_model, map_location=map_location)
    elif cfg.pretrained_model in HeteronymClassificationModel.get_available_model_names():
        model = HeteronymClassificationModel.from_pretrained(cfg.pretrained_model, map_location=map_location)
    else:
        raise ValueError(
            f'Provide path to the pre-trained .nemo checkpoint or choose from {HeteronymClassificationModel.list_available_models()}'
        )
    model.set_trainer(trainer)
    model = model.eval()

    logging.info(f'Config Params: {model._cfg}')

    if cfg.manifest is not None:
        if not os.path.exists(cfg.manifest):
            raise ValueError(f"{cfg.manifest} not found.")
        with torch.no_grad():
            model.disambiguate_manifest(
                manifest=cfg.manifest,
                output_manifest=cfg.output_manifest,
                grapheme_field=cfg.grapheme_field,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )

        # save predictions to a file
        if cfg.errors_file is None:
            cfg.errors_file = cfg.output_manifest.replace(".json", "_errors.txt")

        save_errors = True
        correct = 0
        total = 0
        with open(cfg.output_manifest, "r", encoding="utf-8") as f_preds, open(
            cfg.errors_file, "w", encoding="utf-8"
        ) as f_errors:
            for line in f_preds:
                line = json.loads(line)
                predictions = line["pred_wordid"]
                # run evaluation if target word_id is available in the input manifest
                if "word_id" in line:
                    targets = line["word_id"]
                    if isinstance(targets, str):
                        targets = [targets]
                    for idx, target_ in enumerate(targets):
                        total += 1
                        if idx >= len(predictions) or target_ != predictions[idx]:
                            f_errors.write(f"INPUT: {line[cfg.grapheme_field]}\n")
                            f_errors.write(f"PRED : {predictions[idx]} -- GT: {target_}\n")
                            f_errors.write("===========================\n")
                        else:
                            correct += 1
                else:
                    save_errors = False
        if save_errors:
            logging.info(f"Accuracy: {round(correct / total * 100, 2)}% ({total - correct} errors out of {total})")
            logging.info(f"Errors saved at {cfg.errors_file}")
        else:
            logging.info("No 'word_id' values found, skipping evaluation.")
            if os.path.exists(cfg.errors_file):
                os.remove(cfg.errors_file)
    else:
        print('Entering interactive mode.')
        done = False
        while not done:
            print('Type "STOP" to exit.')
            test_input = input('Input a test input:')
            if test_input == "STOP":
                done = True
            if not done:
                with torch.no_grad():
                    _, sentences = model.disambiguate(
                        sentences=[test_input],
                        batch_size=1,
                        num_workers=cfg.num_workers,
                        wordid_to_phonemes_file=cfg.wordid_to_phonemes_file,
                    )
                    print(sentences[0])


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
