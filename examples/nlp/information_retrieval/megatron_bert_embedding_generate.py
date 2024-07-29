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

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.information_retrieval.megatron_bert_embedding_model import MegatronBertEmbeddingModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronBertTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="megatron_bert_embedding_config")
def main(cfg) -> None:
    if cfg.model.data.dataloader_type != "LDDL":
        mp.set_start_method("spawn", force=True)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronBertTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model_cfg = MegatronBertEmbeddingModel.merge_cfg_with(cfg.restore_from_path, cfg)

    OmegaConf.set_struct(model_cfg, True)
    with open_dict(model_cfg):
        model_cfg.precision = trainer.precision

    logging.info(f"Loading model from {cfg.restore_from_path}")
    model = MegatronBertEmbeddingModel.restore_from(
        restore_path=cfg.restore_from_path,
        trainer=trainer,
        save_restore_connector=NLPSaveRestoreConnector(),
        override_config_path=model_cfg,
        strict=True,
    )

    trainer.test(model)


if __name__ == '__main__':
    main()
