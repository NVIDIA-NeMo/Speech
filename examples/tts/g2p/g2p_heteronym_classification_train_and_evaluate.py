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

import pytorch_lightning as pl
import torch

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.g2p.models.heteronym_classification import HeteronymClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
This script runs training and evaluation of HeteronymClassificationModel

To prepare dataset, see NeMo/scripts/dataset_processing/g2p/export_wikihomograph_data_to_manifest.py

To run training:
python g2p_heteronym_classification_train_and_evaluate.py \
    train_manifest=<Path to train manifest file>" \
    validation_manifest=<Path to validation manifest file>" \
    model.wordids="<Path to wordids.tsv file>" \
    do_training=True
    
To run training and testing (once the training is complete):
python g2p_heteronym_classification_train_and_evaluate.py \
    train_manifest=<Path to train manifest file>" \
    validation_manifest=<Path to validation manifest file>" \
    model.test_ds.dataset.manifest=<Path to test manifest file>" \
    model.wordids="<Path to wordids.tsv file>" \
    do_training=True \
    do_testing=True
    
To run testing:
python g2p_heteronym_classification_train_and_evaluate.py \
    do_training=False \
    do_testing=True \
    model.test_ds.dataset.manifest=<Path to test manifest file>"  \
    pretrained_model=<Path to pretrained .nemo model or from list_available_models()>

    
See https://github.com/google-research-datasets/WikipediaHomographData/blob/master/data/wordids.tsv for wordids file
format example

See https://github.com/NVIDIA/NeMo/blob/main/scripts/dataset_processing/g2p/export_wikihomograph_data_to_manifest.py
on how to convert WikiHomograph data for HeteronymClassificationModel training/evaluation
"""


@hydra_runner(config_path="conf", config_name="g2p_heteronym_classification.yaml")
def main(cfg):
    # PTL 2.0 has find_unused_parameters as False by default, so its required to set it to True
    # when there are unused parameters like in this model
    if cfg.trainer.strategy == 'ddp':
        cfg.trainer.strategy = "ddp_find_unused_parameters_true"
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = None
    if cfg.do_training:
        model = HeteronymClassificationModel(cfg=cfg.model, trainer=trainer)
        lr_logger = pl.callbacks.LearningRateMonitor()
        epoch_time_logger = LogEpochTimeCallback()
        trainer.callbacks.extend([lr_logger, epoch_time_logger])
        trainer.fit(model)
        logging.info("Training is complete")

    if cfg.do_testing:
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

        if model is None:
            if os.path.exists(cfg.pretrained_model):
                # restore model from .nemo file path
                model = HeteronymClassificationModel.restore_from(restore_path=cfg.pretrained_model)
            elif cfg.pretrained_model in HeteronymClassificationModel.get_available_model_names():
                # restore model by name
                model = HeteronymClassificationModel.from_pretrained(cfg.pretrained_model, map_location=map_location)
            else:
                raise ValueError(
                    f'Provide path to the pre-trained .nemo checkpoint or choose from {HeteronymClassificationModel.list_available_models()}'
                )

        if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.dataset.manifest is not None:
            model.setup_test_data(cfg.model.test_ds)
            trainer.test(model)
        else:
            logging.info("test_ds not found, skipping evaluation")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
