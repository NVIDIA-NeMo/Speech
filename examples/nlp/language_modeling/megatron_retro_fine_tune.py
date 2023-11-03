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

import datetime
import os

from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_retro_fine_tune_model import MegatronRetroFinetuneModel
from nemo.collections.nlp.parts.nlp_overrides import (
    CustomProgressBar,
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager


def _modify_config(retro_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original retro pre-training config with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(retro_cfg, True)
    with open_dict(retro_cfg):
        retro_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
        retro_cfg.data = cfg.model.data
        retro_cfg.precision = cfg.trainer.precision
        retro_cfg.optim = cfg.model.optim
        retro_cfg.micro_batch_size = cfg.model.micro_batch_size
        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(retro_cfg)
            retro_cfg.cfg = retro_cfg
    return retro_cfg


def load_from_nemo(cls, cfg, trainer, retro_cfg, modify_confg_fn, save_restore_connector):
    retro_cfg = modify_confg_fn(retro_cfg, cfg, add_cfg_to_tree=False)
    model = cls.restore_from(
        restore_path=cfg.model.restore_path,
        trainer=trainer,
        override_config_path=retro_cfg,
        save_restore_connector=save_restore_connector,
    )
    return model


@hydra_runner(config_path="conf", config_name="megatron_retro_finetune_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    ###### following is the workaround for num_workers=0 issue #####
    # import torch.multiprocessing as mp
    # mp.set_start_method("spawn", force=True)
    #####################################################
    megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True if megatron_amp_O2 else False,
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
        timeout=datetime.timedelta(seconds=18000),
    )

    if cfg.trainer.precision in [16, '16', '16-mixed', 'bf16', 'bf16-mixed']:
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
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(MixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer, callbacks=[CustomProgressBar()])
    exp_manager(trainer, cfg.exp_manager)

    logging.info(f'Resuming training from checkpoint: {trainer.ckpt_path}')

    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    # load existing or init new soft prompt GPT model
    if cfg.model.get("restore_path", None):
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.model.restore_path):
            save_restore_connector.model_extracted_dir = cfg.model.restore_path

        model_cfg = MegatronRetroFinetuneModel.restore_from(
            restore_path=cfg.model.restore_path,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
        model = load_from_nemo(
            MegatronRetroFinetuneModel,
            cfg,
            trainer,
            model_cfg,
            modify_confg_fn=_modify_config,
            save_restore_connector=save_restore_connector,
        )
    else:
        model = MegatronRetroFinetuneModel(cfg.model, trainer=trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
