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

from dataclasses import dataclass
from typing import Optional

from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from omegaconf.omegaconf import MISSING

from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTEncDecModelConfig
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.core.config.modelPT import NemoConfig
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging
from nemo.utils.config_utils import update_model_config
from nemo.utils.exp_manager import ExpManagerConfig, exp_manager

"""
Usage:
 python enc_dec_nmt_finetune.py \
      model_path=/raid/models/de_en_24x6.nemo \
      trainer.devices=2 \
      ~trainer.max_epochs \
      +trainer.max_steps=4500 \
      +trainer.val_check_interval=500 \
      model.train_ds.tgt_file_name=/raid/data/train_lang_filtered.en \
      model.train_ds.src_file_name=/raid/data/train_lang_filtered.de  \
      model.train_ds.tokens_in_batch=6000 \
      model.validation_ds.tgt_file_name=/raid/data/2015.norm.tok.en \
      model.validation_ds.src_file_name=/raid/data/2015.norm.tok.de \
      model.validation_ds.tokens_in_batch=4000 \
      model.test_ds.tgt_file_name=/raid/data/2015.en \
      model.test_ds.src_file_name=/raid/data/2015.de \
      +exp_manager.exp_dir=/raid/results/finetune-test \
      +exp_manager.create_checkpoint_callback=True \
      +exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
      +exp_manager.checkpoint_callback_params.mode=max \
      +exp_manager.checkpoint_callback_params.save_best_model=true
"""


@dataclass
class MTFineTuneConfig(NemoConfig):
    name: Optional[str] = 'MTEncDec'
    model_path: str = MISSING
    do_training: bool = True
    do_testing: bool = False
    model: MTEncDecModelConfig = MTEncDecModelConfig()
    trainer: Optional[TrainerConfig] = TrainerConfig()
    exp_manager: Optional[ExpManagerConfig] = ExpManagerConfig(name='MTEncDec', files_to_copy=[])


@hydra_runner(config_path="conf", config_name="aayn_finetune")
def main(cfg: MTFineTuneConfig) -> None:
    # merge default config with user specified config
    default_cfg = MTFineTuneConfig()
    default_cfg.model = MTEncDecModel.restore_from(restore_path=cfg.model_path, return_config=True)
    del default_cfg.model.optim, default_cfg.model.train_ds, default_cfg.model.validation_ds, default_cfg.model.test_ds
    cfg = update_model_config(default_cfg, cfg, drop_missing_subconfigs=False)
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')

    # training is managed by PyTorch Lightning
    trainer_cfg = OmegaConf.to_container(cfg.trainer)
    trainer_cfg.pop('strategy', None)
    trainer = Trainer(strategy=NLPDDPStrategy(), **trainer_cfg)

    # experiment logs, checkpoints, and auto-resume are managed by exp_manager and PyTorch Lightning
    exp_manager(trainer, cfg.exp_manager)

    # everything needed to train translation models is encapsulated in the NeMo MTEncdDecModel
    mt_model = MTEncDecModel.restore_from(restore_path=cfg.model_path, override_config_path=cfg.model, trainer=trainer)

    mt_model.setup_training_data(cfg.model.train_ds)
    mt_model.setup_multiple_validation_data(val_data_config=cfg.model.validation_ds)

    logging.info("\n\n************** Model parameters and their sizes ***********")
    for name, param in mt_model.named_parameters():
        print(name, param.size())
    logging.info("***********************************************************\n\n")

    if cfg.do_training:
        trainer.fit(mt_model)

    if cfg.do_testing:
        mt_model.setup_multiple_test_data(test_data_config=cfg.model.test_ds)
        trainer.test(mt_model)


if __name__ == '__main__':
    main()
