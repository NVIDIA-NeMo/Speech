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

from megatron_t5_seq2seq_finetune import load_from_checkpoint_dir, load_from_nemo, validate_checkpoint_loading_args
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin

from nemo.collections.nlp.models.language_modeling.megatron_glue_model import MegatronT5GLUEModel
from nemo.collections.nlp.models.language_modeling.megatron_t0_model import MegatronT0Model
from nemo.collections.nlp.models.language_modeling.megatron_t5_sft_model import MegatronT5SFTModel
from nemo.collections.nlp.parts.nlp_overrides import GradScaler, MegatronHalfPrecisionPlugin, NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def _modify_config(t5_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original t5 pre-training config (t5_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(t5_cfg, True)
    with open_dict(t5_cfg):
        t5_cfg.precision = cfg.trainer.precision
        # Overwrite data configs
        if cfg.model.data.validation_ds.get('src_file_name', None) is not None:
            logging.info(
                'Found validation_ds.src_file_name in the config file. Overriding the finetuned model config file with the values from the new config file.'
            )
            t5_cfg.data.validation_ds.src_file_name = cfg.model.data.validation_ds.src_file_name
        if cfg.model.data.validation_ds.get('tgt_file_name', None) is not None:
            logging.info(
                'Found validation_ds.tgt_file_name in the config file. Overriding the finetuned model config file with the values from the new config file.'
            )
            t5_cfg.data.validation_ds.tgt_file_name = cfg.model.data.validation_ds.tgt_file_name

        if "write_predictions_to_file" in cfg.model.data.validation_ds:
            t5_cfg.data.validation_ds.write_predictions_to_file = (
                cfg.model.data.validation_ds.write_predictions_to_file
            )
        if "output_file_path_prefix" in cfg.model.data.validation_ds:
            t5_cfg.data.validation_ds.output_file_path_prefix = cfg.model.data.validation_ds.output_file_path_prefix

        t5_cfg.data.validation_ds.micro_batch_size = cfg.model.data.validation_ds.micro_batch_size
        t5_cfg.data.validation_ds.global_batch_size = cfg.model.data.validation_ds.global_batch_size

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(t5_cfg)
            t5_cfg.cfg = t5_cfg

    return t5_cfg


@hydra_runner(config_path="conf", config_name="megatron_t5_config_finetune_glue_eval")
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

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

    exp_manager(trainer, cfg.exp_manager)

    if hasattr(cfg.model.data.validation_ds, 'task_name'):
        if cfg.model.restore_from_path:
            t5_cfg = MegatronT5GLUEModel.restore_from(
                restore_path=cfg.model.restore_from_path, trainer=trainer, return_config=True
            )
            model = load_from_nemo(MegatronT5GLUEModel, cfg, trainer, t5_cfg, modify_confg_fn=_modify_config)
        else:
            validate_checkpoint_loading_args(cfg.model.pretrained_checkpoint)
            model = load_from_checkpoint_dir(MegatronT5GLUEModel, cfg, trainer, modify_confg_fn=_modify_config)
    elif hasattr(cfg.model.data.validation_ds, 'file_names'):
        if cfg.model.restore_from_path:
            t5_cfg = MegatronT0Model.restore_from(
                restore_path=cfg.model.restore_from_path, trainer=trainer, return_config=True
            )
            model = load_from_nemo(MegatronT0Model, cfg, trainer, t5_cfg, modify_confg_fn=_modify_config)
        else:
            validate_checkpoint_loading_args(cfg.model.pretrained_checkpoint)
            model = load_from_checkpoint_dir(MegatronT0Model, cfg, trainer, modify_confg_fn=_modify_config)
    else:
        if cfg.model.restore_from_path:
            t5_cfg = MegatronT5SFTModel.restore_from(
                restore_path=cfg.model.restore_from_path, trainer=trainer, return_config=True
            )
            model = load_from_nemo(MegatronT5SFTModel, cfg, trainer, t5_cfg, modify_confg_fn=_modify_config)
        else:
            validate_checkpoint_loading_args(cfg.model.pretrained_checkpoint)
            model = load_from_checkpoint_dir(MegatronT5SFTModel, cfg, trainer, modify_confg_fn=_modify_config)

    model.freeze()
    trainer.validate(model)
    if hasattr(cfg.model.data, 'test_ds'):
        trainer.test(model)


if __name__ == '__main__':
    main()
