# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import lightning.pytorch as pl
from omegaconf import DictConfig

from nemo.collections.nlp.data.text_normalization import constants
from nemo.collections.nlp.models import DuplexDecoderModel, DuplexTaggerModel
from nemo.utils import logging

__all__ = ['TAGGER_MODEL', 'DECODER_MODEL', 'MODEL_NAMES', 'instantiate_model_and_trainer']

TAGGER_MODEL = 'tagger'
DECODER_MODEL = 'decoder'
MODEL_NAMES = [TAGGER_MODEL, DECODER_MODEL]


def instantiate_model_and_trainer(cfg: DictConfig, model_name: str, do_training: bool):
    """Function for instantiating a model and a trainer
    Args:
        cfg: The config used to instantiate the model and the trainer.
        model_name: A str indicates whether the model to be instantiated is a tagger or a decoder (i.e., model_name should be either TAGGER_MODEL or DECODER_MODEL).
        do_training: A boolean flag indicates whether the model will be trained or evaluated.

    Returns:
        trainer: A PyTorch Lightning trainer
        model: A NLPModel that can either be a DuplexTaggerModel or a DuplexDecoderModel
    """
    assert model_name in MODEL_NAMES

    # Get configs for the corresponding models
    trainer_cfg = cfg.get(f'{model_name}_trainer')
    model_cfg = cfg.get(f'{model_name}_model')
    pretrained_cfg = cfg.get(f'{model_name}_pretrained_model', None)

    trainer = pl.Trainer(**trainer_cfg)
    if not pretrained_cfg:
        logging.info(f'Initializing {model_name} model')
        if model_name == TAGGER_MODEL:
            model = DuplexTaggerModel(model_cfg, trainer=trainer)
        if model_name == DECODER_MODEL:
            model = DuplexDecoderModel(model_cfg, trainer=trainer)
    elif os.path.exists(pretrained_cfg):
        logging.info(f'Restoring pretrained {model_name} model from {pretrained_cfg}')
        if model_name == TAGGER_MODEL:
            model = DuplexTaggerModel.restore_from(pretrained_cfg)
        if model_name == DECODER_MODEL:
            model = DuplexDecoderModel.restore_from(pretrained_cfg)
    else:
        logging.info(f'Loading pretrained model {pretrained_cfg}')
        if model_name == TAGGER_MODEL:
            if pretrained_cfg not in DuplexTaggerModel.get_available_model_names():
                raise (
                    ValueError(
                        f'{pretrained_cfg} not in the list of available Tagger models. Select from {DuplexTaggerModel.list_available_models()}'
                    )
                )
            model = DuplexTaggerModel.from_pretrained(pretrained_cfg)
        if model_name == DECODER_MODEL:
            if pretrained_cfg not in DuplexDecoderModel.get_available_model_names():
                raise (
                    ValueError(
                        f'{pretrained_cfg} not in the list of available Decoder models. Select from {DuplexDecoderModel.list_available_models()}'
                    )
                )
            model = DuplexDecoderModel.from_pretrained(pretrained_cfg)

    # Set model.lang (if it is still None)
    if model.lang is None:
        model.lang = cfg.lang
    assert model.lang in constants.SUPPORTED_LANGS
    # Setup covering grammars (if enabled)
    # We only support integrating with English TN covering grammars at the moment
    if model_name == DECODER_MODEL and model_cfg.use_cg and cfg.lang == constants.ENGLISH:
        if model.cg_normalizer is None:
            model.setup_cgs(model_cfg)

    # Setup train and validation data
    if do_training:
        model.setup_training_data(train_data_config=cfg.data.train_ds)
        if model_name == DECODER_MODEL:
            model.setup_multiple_validation_data(val_data_config=cfg.data.validation_ds)
        else:
            model.setup_validation_data(val_data_config=cfg.data.validation_ds)

    logging.info(f'Model {model_name} -- Device {model.device}')
    return trainer, model
