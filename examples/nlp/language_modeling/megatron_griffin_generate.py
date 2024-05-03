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


import os
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf
from nemo.collections.nlp.models.language_modeling.megatron_griffin_sft_model import MegatronGriffinSFTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.model_utils import inject_model_parallel_rank


mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="megatron_griffin_generate_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()

    if cfg.model.peft.restore_from_path:
        model_cfg = MegatronGriffinSFTModel.merge_inference_cfg(cfg.model.peft.restore_from_path, cfg)
    else:
        model_cfg = MegatronGriffinSFTModel.merge_inference_cfg(cfg.model.restore_from_path, cfg)

    model = MegatronGriffinSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)

    if cfg.model.peft.restore_from_path:
        model.load_adapters(cfg.model.peft.restore_from_path)
    elif cfg.model.peft.restore_from_ckpt.checkpoint_dir and cfg.model.peft.restore_from_ckpt.checkpoint_name:
        peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]
        checkpoint_path = os.path.join(
            cfg.model.peft.restore_from_ckpt.checkpoint_dir, cfg.model.peft.restore_from_ckpt.checkpoint_name
        )
        # checkpoint_path is a dir in case of distributed checkpointing
        if not os.path.isdir(checkpoint_path):
            # legacy checkpoint needs model parallel rank injection
            checkpoint_path = inject_model_parallel_rank(
                os.path.join(
                    cfg.model.peft.restore_from_ckpt.checkpoint_dir, cfg.model.peft.restore_from_ckpt.checkpoint_name
                )
            )
            model.load_adapters(checkpoint_path, peft_cfgs=peft_cfg_cls(model_cfg))
        else:
            raise NotImplementedError("distributed checkpointing of PEFT weights is not supported")

    model.freeze()
    logging.info(f"Freezing parameters for PEFT eval:\n{model.summarize()}")

    trainer.test(model)


if __name__ == "__main__":
    main()
