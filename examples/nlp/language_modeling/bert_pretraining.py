# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.language_modeling import BERTLMModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="bert_pretraining_from_text_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config:\n {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(strategy=DDPStrategy(find_unused_parameters=True), **cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    bert_model = BERTLMModel(cfg.model, trainer=trainer)
    trainer.fit(bert_model)
    if cfg.model.nemo_path:
        bert_model.save_to(cfg.model.nemo_path)


if __name__ == '__main__':
    main()
