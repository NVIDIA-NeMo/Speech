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

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import ZeroShotIntentModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="zero_shot_intent_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params:\n {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # initialize the model using the config file
    model = ZeroShotIntentModel(cfg.model, trainer=trainer)

    # training
    logging.info("================================================================================================")
    logging.info('Starting training...')
    trainer.fit(model)
    logging.info('Training finished!')
    if cfg.model.nemo_path:
        model.save_to(cfg.model.nemo_path)


if __name__ == '__main__':
    main()
