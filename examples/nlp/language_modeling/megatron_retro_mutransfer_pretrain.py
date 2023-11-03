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

from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.modules.common.megatron.mup.optim import MuAdam, MuAdamW
from nemo.collections.nlp.parts.nlp_overrides import (
    CustomProgressBar,
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
)
from nemo.core.config import hydra_runner
from nemo.core.config.optimizers import AdamParams, AdamWParams
from nemo.core.optim.optimizers import register_optimizer
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="megatron_retro_mutransfer")
def main(cfg) -> None:
    register_optimizer("muadamw", MuAdamW, AdamWParams())
    register_optimizer("muadam", MuAdam, AdamParams())
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True if megatron_amp_O2 else False,
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )

    if cfg.trainer.precision in [16, '16', 'bf16', '16-mixed', 'bf16-mixed']:
        scaler = None
        if cfg.trainer.precision in [16, '16', '16-mixed']:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'
        if megatron_amp_O2:
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(MixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer, callbacks=[CustomProgressBar()])

    exp_manager(trainer, cfg.exp_manager)

    # resume_from_checkpoint = uninject_model_parallel_rank(resume_from_checkpoint)
    logging.info(f'Resuming training from checkpoint: {trainer.ckpt_path}')

    model = MegatronRetrievalModel(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
