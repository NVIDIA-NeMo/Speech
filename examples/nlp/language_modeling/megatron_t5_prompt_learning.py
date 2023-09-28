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
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.nlp.models.language_modeling.megatron_t5_prompt_learning_model import (
    MegatronT5PromptLearningModel,
)
from nemo.collections.nlp.parts.nlp_overrides import (
    CustomProgressBar,
    GradScaler,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.decorators import deprecated
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)


"""
This is an example of how to ptune/prompt-tune a pretrained T5 model.
Be sure to use a .nemo T5 model with this code. If you've downloaded 
a model from NGC or are otherwise using a MegatronLM model, please use
either megatron_ckpt_to_nemo.py or megatron_lm_ckpt_to_nemo.py found
within this examples directory to convert your model to .nemo format.
"""


@deprecated(
    explanation=f"{__file__} is deprecated. Please use MegatronT5SFTModel.add_adapter() for PEFT features."
    "See updated scripts `megatron_t5_peft_tuning.py` and `megatron_t5_peft_eval.py` for examples."
)
@hydra_runner(config_path="conf", config_name="megatron_t5_prompt_learning.yaml")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    plugins = []
    strategy = NLPDDPStrategy(no_ddp_communication_hook=True, find_unused_parameters=False,)
    if cfg.trainer.precision == 16 or cfg.trainer.precision == '16-mixed':
        scaler = GradScaler(
            init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
            growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
            hysteresis=cfg.model.get('hysteresis', 2),
            enabled=False
            if cfg.model.pipeline_model_parallel_size > 1
            else True,  # turn off the grad scale for pipeline parallel LM model
        )
        # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
        plugins.append(PipelineMixedPrecisionPlugin(precision='16-mixed', device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer, callbacks=[CustomProgressBar()])
    exp_manager(trainer, cfg.exp_manager)

    # load existing or init new soft prompt T5 model
    if cfg.model.get("restore_path", None):
        model = MegatronT5PromptLearningModel.restore_from(
            cfg.model.restore_path, cfg.model, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
        )

    else:
        model = MegatronT5PromptLearningModel(cfg.model, trainer=trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
