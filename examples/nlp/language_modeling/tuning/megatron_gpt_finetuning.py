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

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)

"""
This is the script to finetuning a GPT Model with any PEFT method.
A base GPT Model is required as a starting point. This script will then insert
Adapters into each Transformer layer and will train/update only these adapters
during training. The base GPT Model weights will remain frozen.

During training this script will only save the newly trained Adapter weights
in checkpoints. At the end of training a .nemo file of Adapter weights will 
be saved.

Usage:
    Assuming the base model is a 125m GPT Model, with TP=1, PP=1:
    a. run a training run for a base gpt nemo file:
        python megatron_gpt_finetuning.py \
            "model.data.train_ds.file_names=[PATH TO TRAINING JSONL FILE]",
            "model.data.train_ds.concat_sampling_probabilities=[SAMPLING VAL]",
            "model.data.validation_ds.file_names=[PATH TO VALIDATION JSONL FILE]",
            "model.data.validation_ds.names=[NAME FOR METRIC LOGGING]",
            model.restore_from_path="PATH TO BASE GPT MODEL .nemo FILE"
            model.peft.peft_scheme='lora'  # lora, ptuning, adapter, ia3, or none for full fineutning
            name="NAME OF TRAINING RUN"
            exp_manager.exp_dir="DIR TO SAVE CHECKPOINTS and .nemo FILE",
Please see lora.ipynb for a step-by-step guide.
"""


@hydra_runner(config_path="conf", config_name="megatron_gpt_finetuning_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # cfg.trainer.precision becomes None in TrainerBuilder if precision_plugins exist since both precision plugins and precision
    # can't exist in PTL >= 2.1, hence storing precision value from cfg.trainer.precision as its used for future steps like in merge_cfg_with func.
    precision = cfg.trainer.precision
    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()
    # Restore the precision value after Trainer is built.
    cfg.trainer.precision = precision
    exp_manager(trainer, cfg.exp_manager)

    model_cfg = MegatronGPTSFTModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
    model = MegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
    peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]

    if cfg.model.peft.restore_from_path is not None:
        # initialize peft weights from a checkpoint instead of randomly
        # This is not the same as resume training because optimizer states are not restored.
        logging.info("PEFT Weights will be loaded from", cfg.model.peft.restore_from_path)
        model.load_adapters(cfg.model.peft.restore_from_path, peft_cfg_cls(model_cfg))
    elif peft_cfg_cls is not None:
        logging.info("Adding adapter weights to the model for PEFT")
        model.add_adapter(peft_cfg_cls(model_cfg))
    else:
        logging.info(f"Running full finetuning since no peft scheme is given.\n{model.summarize()}")

    trainer.fit(model)


if __name__ == '__main__':
    main()
