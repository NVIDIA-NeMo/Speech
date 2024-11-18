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
from typing import Tuple

import lightning.pytorch as pl
from omegaconf import DictConfig

from nemo.collections.nlp.models import ThutmoseTaggerModel
from nemo.utils import logging

__all__ = ["ITN_MODEL", "MODEL_NAMES", "instantiate_model_and_trainer"]

ITN_MODEL = "itn"
MODEL_NAMES = [ITN_MODEL]


def instantiate_model_and_trainer(
    cfg: DictConfig, model_name: str, do_training: bool
) -> Tuple[pl.Trainer, ThutmoseTaggerModel]:
    """Function for instantiating a model and a trainer
    Args:
        cfg: The config used to instantiate the model and the trainer.
        model_name: A str indicates the model direction, currently only 'itn'.
        do_training: A boolean flag indicates whether the model will be trained or evaluated.

    Returns:
        trainer: A PyTorch Lightning trainer
        model: A ThutmoseTaggerModel
    """

    if model_name not in MODEL_NAMES:
        raise ValueError(f"{model_name} is unknown model type")

    # Get configs for the corresponding models
    trainer_cfg = cfg.get("trainer")
    model_cfg = cfg.get("model")
    pretrained_cfg = cfg.get("pretrained_model", None)
    trainer = pl.Trainer(**trainer_cfg)
    if not pretrained_cfg:
        logging.info(f"Initializing {model_name} model")
        if model_name == ITN_MODEL:
            model = ThutmoseTaggerModel(model_cfg, trainer=trainer)
        else:
            raise ValueError(f"{model_name} is unknown model type")
    elif os.path.exists(pretrained_cfg):
        logging.info(f"Restoring pretrained {model_name} model from {pretrained_cfg}")
        model = ThutmoseTaggerModel.restore_from(pretrained_cfg)
    else:
        logging.info(f"Loading pretrained model {pretrained_cfg}")
        if model_name == ITN_MODEL:
            if pretrained_cfg not in ThutmoseTaggerModel.get_available_model_names():
                raise (
                    ValueError(
                        f"{pretrained_cfg} not in the list of available Tagger models."
                        f"Select from {ThutmoseTaggerModel.list_available_models()}"
                    )
                )
            model = ThutmoseTaggerModel.from_pretrained(pretrained_cfg)
        else:
            raise ValueError(f"{model_name} is unknown model type")

    # Setup train and validation data
    if do_training:
        model.setup_training_data(train_data_config=cfg.data.train_ds)
        model.setup_validation_data(val_data_config=cfg.data.validation_ds)

    logging.info(f"Model {model_name} -- Device {model.device}")
    return trainer, model
