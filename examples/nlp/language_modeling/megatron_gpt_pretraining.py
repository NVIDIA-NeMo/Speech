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


from pathlib import Path

# To suppress BF16 compile related issue in the CI runs with turing/V100
import torch._dynamo
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

torch._dynamo.config.suppress_errors = True

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    # Continual training
    if cfg.model.get("restore_from_path") is not None:
        # Option 1: Restore only the model weights from a .nemo file
        logging.info(f"Continual training: loading weights from {cfg.model.restore_from_path}")
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel

        model_cfg = MegatronGPTSFTModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.model.restore_from_path,
            override_config_path=model_cfg,
            trainer=trainer,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
    elif cfg.model.get("restore_from_ckpt") is not None:
        # Option 2: Restore both model weights and optimizer states from a PTL checkpoint
        logging.info(f"Continual training: loading weights and optimizer states from {cfg.model.restore_from_ckpt}")
        trainer.ckpt_path = Path(cfg.model.restore_from_ckpt)
        model = MegatronGPTModel(cfg.model, trainer)

    # Start new pretraining or resume from a checkpoint if it exists
    else:
        model = MegatronGPTModel(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
