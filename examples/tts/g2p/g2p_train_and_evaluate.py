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
from utils import get_model

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models.base import G2PModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager

"""
This script supports training of G2PModels 
(for T5G2PModel use g2p_t5.yaml, for CTCG2PModel use either g2p_conformer.yaml or g2p_t5_ctc.yaml)

# Training T5G2PModel and evaluation at the end of training:
    python examples/text_processing/g2p/g2p_train_and_evaluate.py \
        # (Optional: --config-path=<Path to dir of configs> --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<Path to manifest file>" \
        model.validation_ds.manifest_filepath="<Path to manifest file>" \
        model.test_ds.manifest_filepath="<Path to manifest file>" \
        trainer.devices=1 \
        do_training=True \
        do_testing=True
    
    Example of the config file: NeMo/examples/tts/g2p/conf/g2p_t5.yaml
        
# Training Conformer-G2P Model and evaluation at the end of training:
    python examples/text_processing/g2p/g2p_train_and_evaluate.py \
        # (Optional: --config-path=<Path to dir of configs> --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<Path to manifest file>" \
        model.validation_ds.manifest_filepath="<Path to manifest file>" \
        model.test_ds.manifest_filepath="<Path to manifest file>" \
        model.tokenizer.dir=<Path to pretrained tokenizer> \
        trainer.devices=1 \
        do_training=True \
        do_testing=True
        
    Example of the config file: NeMo/examples/text_processing/g2p/conf/g2p_conformer_ctc.yaml
        
# Run evaluation of the pretrained model:
    python examples/text_processing/g2p/g2p_train_and_evaluate.py \
        # (Optional: --config-path=<Path to dir of configs> --config-name=<name of config without .yaml>) \
        pretrained_model="<Path to .nemo file or pretrained model name from list_available_models()>" \
        model.test_ds.manifest_filepath="<Path to manifest file>" \
        trainer.devices=1 \
        do_training=False \
        do_testing=True
"""


@hydra_runner(config_path="conf", config_name="g2p_t5")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    g2p_model = None
    if cfg.do_training:
        g2p_model = get_model(cfg, trainer)
        lr_logger = pl.callbacks.LearningRateMonitor()
        epoch_time_logger = LogEpochTimeCallback()
        trainer.callbacks.extend([lr_logger, epoch_time_logger])
        trainer.fit(g2p_model)

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

        if g2p_model is None:
            if os.path.exists(cfg.pretrained_model):
                # restore g2p_model from .nemo file path
                model_cfg = G2PModel.restore_from(restore_path=cfg.pretrained_model, return_config=True)
                classpath = model_cfg.target  # original class path
                imported_class = model_utils.import_class_by_path(classpath)
                logging.info(f"Restoring g2p_model : {imported_class.__name__}")
                g2p_model = imported_class.restore_from(restore_path=cfg.pretrained_model, map_location=map_location)
                model_name = os.path.splitext(os.path.basename(cfg.pretrained_model))[0]
                logging.info(f"Restored {model_name} g2p_model from {cfg.pretrained_model}.")
            elif cfg.pretrained_model in G2PModel.get_available_model_names():
                # restore g2p_model by name
                g2p_model = G2PModel.from_pretrained(cfg.pretrained_model, map_location=map_location)
            else:
                raise ValueError(
                    f'Provide path to the pre-trained .nemo checkpoint or choose from {G2PModel.list_available_models()}'
                )

        if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.manifest_filepath is not None:
            g2p_model.setup_multiple_test_data(cfg.model.test_ds)
            if g2p_model.prepare_test(trainer):
                trainer.test(g2p_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
