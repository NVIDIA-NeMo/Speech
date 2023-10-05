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


"""
This script contains an example on how to train a ThutmoseTaggerModel for inverse text normalization(ITN).

This script uses the `/examples/nlp/text_normalization_as_tagging/conf/thutmose_tagger_itn_config.yaml`
config file by default. The other option is to set another config file via command
line arguments by `--config-name=CONFIG_FILE_PATH'. Probably it is worth looking
at the example config file to see the list of parameters used for training.

USAGE Example:
1. Obtain a processed dataset
2. Run:
    python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_train.py \
      lang=${LANG} \
      data.validation_ds.data_path=${DATA_PATH}/valid.tsv \
      data.train_ds.data_path=${DATA_PATH}/train.tsv \
      data.train_ds.batch_size=128 \
      data.train_ds.num_workers=8 \
      model.language_model.pretrained_model_name=${LANGUAGE_MODEL} \
      model.label_map=${DATA_PATH}/label_map.txt \
      model.semiotic_classes=${DATA_PATH}/semiotic_classes.txt \
      model.optim.lr=3e-5 \
      trainer.devices=[1] \
      trainer.num_nodes=1 \
      trainer.accelerator=gpu \
      trainer.strategy=ddp \
      trainer.max_epochs=5

Information on the arguments:

Most arguments in the example config file are quite self-explanatory (e.g.,
`model.optim.lr` refers to the learning rate for training the model).

Some arguments we want to mention are:

+ lang: The language of the dataset.
+ model.language_model.pretrained_model_name: This is the backbone BERT model (depends on the language)
e.g. bert-base-uncased (English), DeepPavlov/rubert-base-cased (Russian)
"""

from helpers import ITN_MODEL, instantiate_model_and_trainer
from omegaconf import DictConfig, OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="thutmose_tagger_itn_config")
def main(cfg: DictConfig) -> None:
    # PTL 2.0 has find_unused_parameters as False by default, so its required to set it to True
    # when there are unused parameters like here
    if cfg.trainer.strategy == 'ddp':
        cfg.trainer.strategy = "ddp_find_unused_parameters_true"
    logging.info(f'Config Params: {OmegaConf.to_yaml(cfg)}')

    # Train the model
    if cfg.model.do_training:
        logging.info(
            "================================================================================================"
        )
        logging.info('Start training...')
        trainer, model = instantiate_model_and_trainer(cfg, ITN_MODEL, True)
        thutmose_tagger_exp_manager = cfg.get('exp_manager', None)
        exp_manager(trainer, thutmose_tagger_exp_manager)
        trainer.fit(model)
        logging.info('Training finished!')


if __name__ == '__main__':
    main()
