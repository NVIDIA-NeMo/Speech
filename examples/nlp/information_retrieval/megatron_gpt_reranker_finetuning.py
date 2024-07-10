# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from collections.abc import MutableMapping

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from nemo.collections.nlp.models.information_retrieval.megatron_gpt_reranker_model import MegatronGPTRerankerModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@hydra_runner(config_path="conf", config_name="megatron_gpt_reranker_tuning_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model_cfg = MegatronGPTRerankerModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
    if trainer.global_rank == 0:
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                fd = flatten_dict(dict(model_cfg), sep="/")
                logger.experiment.config.update(fd)
    model = MegatronGPTRerankerModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
    peft_cfg_cls_lst = [PEFT_CONFIG_MAP[s] for s in cfg.model.peft.peft_scheme.split(",")]
    peft_cfg_cls = [_peft_cfg(model_cfg) for _peft_cfg in peft_cfg_cls_lst]

    if cfg.model.peft.restore_from_path is not None:
        # initialize peft weights from a checkpoint instead of randomly
        # This is not the same as resume training because optimizer states are not restored.
        logging.info("PEFT Weights will be loaded from", cfg.model.peft.restore_from_path)
        model.load_adapters(cfg.model.peft.restore_from_path, peft_cfg_cls)
    elif peft_cfg_cls is not None:
        logging.info("Adding adapter weights to the model for PEFT")
        # model.add_adapter(peft_cfg_cls(model_cfg))
        model.add_adapter(peft_cfg_cls)
    else:
        logging.info(f"Running full finetuning since no peft scheme is given.\n{model.summarize()}")

    trainer.fit(model)


if __name__ == '__main__':
    main()
