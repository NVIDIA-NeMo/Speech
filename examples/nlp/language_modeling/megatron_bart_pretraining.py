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


from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_bart_model import MegatronBARTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    CustomProgressBar,
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="megatron_bart_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,
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
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'
        if megatron_amp_O2:
            plugins.append(MegatronHalfPrecisionPlugin(plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(plugin_precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(
        plugins=plugins, strategy=strategy, **cfg.trainer, callbacks=[ModelSummary(max_depth=3), CustomProgressBar()]
    )

    exp_manager(trainer, cfg.exp_manager)

    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        trainer.ckpt_path = cfg.model.resume_from_checkpoint

    logging.info(f'Resuming training from checkpoint: {trainer.ckpt_path}')

    model = MegatronBARTModel(cfg.model, trainer)
    trainer.fit(model)


if __name__ == '__main__':
    main()
