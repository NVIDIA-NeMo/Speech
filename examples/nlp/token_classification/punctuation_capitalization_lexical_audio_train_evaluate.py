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
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.token_classification.punctuation_capitalization_config import (
    PunctuationCapitalizationLexicalAudioConfig,
)
from nemo.collections.nlp.models.token_classification.punctuation_capitalization_lexical_audio_model import (
    PunctuationCapitalizationLexicalAudioModel,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""
This script show how to train a Punctuation and Capitalization Model with lexical and acoustic features.
More details on the task and data format could be found in tutorials/nlp/Punctuation_and_Capitalization.ipynb

*** Setting the configs ***

The model and the PT trainer are defined in a config file which declares multiple important sections.
The most important ones are:
    model: All arguments that are related to the Model - language model, audio encoder, tokenizer, token classifier, optimizer,
            schedulers, and datasets/data loaders.
    trainer: Any argument to be passed to PyTorch Lightning including number of epochs, number of GPUs,
            precision level, etc.
This script uses the `/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml` config file
by default. You may update the config file from the file directly. 
The other option is to set another config file via command line arguments by `--config-name=CONFIG_FILE_PATH'.

*** Model training ***

To run this script and train the model from scratch, use:
    python punctuation_capitalization_lexical_audio_train_evaluate.py \
        model.train_ds.ds_item=<PATH/TO/TRAIN/DATA> \
        model.train_ds.text_file=<NAME_OF_TRAIN_INPUT_TEXT_FILE> \
        model.train_ds.labels_file=<NAME_OF_TRAIN_LABELS_FILE> \
        model.train_ds.audio_file=<NAME_OF_TRAIN_AUDIO_FILE> \
        model.validation_ds.ds_item=<PATH/TO/DEV/DATA> \
        model.validation_ds.text_file=<NAME_OF_DEV_INPUT_TEXT_FILE> \
        model.validation_ds.labels_file=<NAME_OF_DEV_LABELS_FILE> \
        model.validation_ds.audio_file=<NAME_OF_DEV_AUDIO_FILE>

To use BERT-like pretrained P&C models' weights to initialize lexical encoder, use:
    python punctuation_capitalization_lexical_audio_train_evaluate.py \
        model.train_ds.ds_item=<PATH/TO/TRAIN/DATA> \
        model.train_ds.text_file=<NAME_OF_TRAIN_INPUT_TEXT_FILE> \
        model.train_ds.labels_file=<NAME_OF_TRAIN_LABELS_FILE> \
        model.train_ds.audio_file=<NAME_OF_TRAIN_AUDIO_FILE> \
        model.validation_ds.ds_item=<PATH/TO/DEV/DATA> \
        model.validation_ds.text_file=<NAME_OF_DEV_INPUT_TEXT_FILE> \
        model.validation_ds.labels_file=<NAME_OF_DEV_LABELS_FILE> \
        model.validation_ds.audio_file=<NAME_OF_DEV_AUDIO_FILE> \
        model.restore_lexical_encoder_from=<PATH/TO/CHECKPOINT.nemo>


If you wish to perform testing after training set `do_testing` to `true:
    python punctuation_capitalization_lexical_audio_train_evaluate.py \
        +do_testing=true \
        pretrained_model=<PATH/TO/CHECKPOINT.nemo> \
        model.train_ds.ds_item=<PATH/TO/TRAIN/DATA> \
        model.train_ds.text_file=<NAME_OF_TRAIN_INPUT_TEXT_FILE> \
        model.train_ds.labels_file=<NAME_OF_TRAIN_LABELS_FILE> \
        model.train_ds.audio_file=<NAME_OF_TRAIN_AUDIO_FILE> \
        model.validation_ds.ds_item=<PATH/TO/DEV/DATA> \
        model.validation_ds.text_file=<NAME_OF_DEV_INPUT_TEXT_FILE> \
        model.validation_ds.labels_file=<NAME_OF_DEV_LABELS_FILE> \
        model.validation_ds.audio_file=<NAME_OF_DEV_AUDIO_FILE> \
        model.test_ds.ds_item=<PATH/TO/TEST_DATA> \
        model.test_ds.text_file=<NAME_OF_TEST_INPUT_TEXT_FILE> \
        model.test_ds.labels_file=<NAME_OF_TEST_LABELS_FILE> \
        model.test_ds.audio_file=<NAME_OF_TEST_AUDIO_FILE>

Set `do_training` to `false` and `do_testing` to `true` to perform evaluation without training:
    python punctuation_capitalization_lexical_audio_train_evaluate.py \
        +do_testing=true \
        +do_training=false \
        pretrained_model==<PATH/TO/CHECKPOINT.nemo> \
        model.test_ds.ds_item=<PATH/TO/DEV/DATA> \
        model.test_ds.text_file=<NAME_OF_TEST_INPUT_TEXT_FILE> \
        model.test_ds.labels_file=<NAME_OF_TEST_LABELS_FILE> \
        model.test_ds.audio_file=<NAME_OF_TEST_AUDIO_FILE>

"""


@hydra_runner(config_path="conf", config_name="punctuation_capitalization_lexical_audio_config")
def main(cfg: DictConfig) -> None:
    # PTL 2.0 has find_unused_parameters as False by default, so its required to set it to True
    # when there are unused parameters like here
    if cfg.trainer.strategy == 'ddp':
        cfg.trainer.strategy = "ddp_find_unused_parameters_true"
    torch.manual_seed(42)
    cfg = OmegaConf.merge(OmegaConf.structured(PunctuationCapitalizationLexicalAudioConfig()), cfg)
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    if not cfg.do_training and not cfg.do_testing:
        raise ValueError("At least one of config parameters `do_training` and `do_testing` has to be `true`.")
    if cfg.do_training:
        if cfg.model.get('train_ds') is None:
            raise ValueError('`model.train_ds` config section is required if `do_training` config item is `True`.')
    if cfg.do_testing:
        if cfg.model.get('test_ds') is None:
            raise ValueError('`model.test_ds` config section is required if `do_testing` config item is `True`.')

    if not cfg.pretrained_model:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        model = PunctuationCapitalizationLexicalAudioModel(cfg.model, trainer=trainer)
    else:
        if os.path.exists(cfg.pretrained_model):
            model = PunctuationCapitalizationLexicalAudioModel.restore_from(cfg.pretrained_model)
        elif cfg.pretrained_model in PunctuationCapitalizationLexicalAudioModel.get_available_model_names():
            model = PunctuationCapitalizationLexicalAudioModel.from_pretrained(cfg.pretrained_model)
        else:
            raise ValueError(
                f'Provide path to the pre-trained .nemo file or choose from '
                f'{PunctuationCapitalizationLexicalAudioModel.list_available_models()}'
            )
        model.update_config_after_restoring_from_checkpoint(
            class_labels=cfg.model.class_labels,
            common_dataset_parameters=cfg.model.common_dataset_parameters,
            train_ds=cfg.model.get('train_ds') if cfg.do_training else None,
            validation_ds=cfg.model.get('validation_ds') if cfg.do_training else None,
            test_ds=cfg.model.get('test_ds') if cfg.do_testing else None,
            optim=cfg.model.get('optim') if cfg.do_training else None,
        )
        model.set_trainer(trainer)
        if cfg.do_training:
            model.setup_training_data()
            model.setup_multiple_validation_data(cfg.model.validation_ds)
            model.setup_optimization()
        else:
            model.setup_multiple_test_data(cfg.model.test_ds)
    if cfg.do_training:
        trainer.fit(model)
    if cfg.do_testing:
        trainer.test(model)


if __name__ == '__main__':
    main()
