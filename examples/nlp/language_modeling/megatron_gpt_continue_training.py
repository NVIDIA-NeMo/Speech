# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile

from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import (
    CustomProgressBar,
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import AppState, logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.model_utils import inject_model_parallel_rank


def _modify_config(gpt_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt pre-training config (t5_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(gpt_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(gpt_cfg):
        gpt_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
        gpt_cfg.micro_batch_size = cfg.model.micro_batch_size
        gpt_cfg.global_batch_size = cfg.model.global_batch_size
        gpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        gpt_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
        gpt_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
        gpt_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
        gpt_cfg.data = cfg.model.data
        gpt_cfg.optim = cfg.model.optim
        gpt_cfg.precision = cfg.trainer.precision
        gpt_cfg.restore_from_path = cfg.restore_from_path
        gpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        gpt_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
        gpt_cfg.encoder_seq_length = cfg.model.encoder_seq_length
        gpt_cfg.max_position_embeddings = cfg.model.max_position_embeddings
        gpt_cfg.seq_len_interpolation_factor = cfg.model.seq_len_interpolation_factor
        gpt_cfg.use_flash_attention = cfg.model.use_flash_attention
        gpt_cfg.tensor_model_parallel_size = cfg.model.get('tensor_model_parallel_size', 1)
        gpt_cfg.pipeline_model_parallel_size = cfg.model.get('pipeline_model_parallel_size', 1)
        gpt_cfg.pipeline_model_parallel_split_rank = cfg.model.get('pipeline_model_parallel_split_rank', 0)

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(gpt_cfg)
            gpt_cfg.cfg = gpt_cfg

    return gpt_cfg


def load_from_nemo(cls, cfg, trainer, gpt_cfg, modify_confg_fn):
    gpt_cfg = modify_confg_fn(gpt_cfg, cfg, add_cfg_to_tree=False)
    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.restore_from_path):
        save_restore_connector.model_extracted_dir = cfg.restore_from_path
    model = cls.restore_from(
        restore_path=cfg.restore_from_path,
        trainer=trainer,
        override_config_path=gpt_cfg,
        save_restore_connector=save_restore_connector,
    )
    return model


def load_from_checkpoint_dir(cls, cfg, trainer, modify_confg_fn):
    app_state = AppState()
    if cfg.model.tensor_model_parallel_size > 1 or cfg.model.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size
        app_state.tensor_model_parallel_size = cfg.model.tensor_model_parallel_size
        app_state.pipeline_model_parallel_size = cfg.model.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            app_state.data_parallel_size,
            app_state.pipeline_model_parallel_split_rank,
            app_state.virtual_pipeline_model_parallel_rank,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=cfg.model.tensor_model_parallel_size,
            pipeline_model_parallel_size_=cfg.model.pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank_=cfg.model.pipeline_model_parallel_split_rank,
        )
    checkpoint_path = inject_model_parallel_rank(
        os.path.join(cfg.model.pretrained_checkpoint.checkpoint_dir, cfg.model.pretrained_checkpoint.checkpoint_name)
    )
    hparams_file = OmegaConf.load(cfg.model.pretrained_checkpoint.hparams_file)
    gpt_cfg = modify_confg_fn(hparams_file.cfg, cfg, add_cfg_to_tree=True)
    with tempfile.NamedTemporaryFile(suffix='.yaml') as f:
        OmegaConf.save(config=gpt_cfg, f=f.name)
        model = cls.load_from_checkpoint(checkpoint_path=checkpoint_path, trainer=trainer, hparams_file=f.name,)
        return model


def validate_checkpoint_loading_args(cfg):
    if cfg.checkpoint_dir is None or not os.path.isdir(cfg.checkpoint_dir):
        raise ValueError(f'Checkpoint directory {cfg.checkpoint_dir} does not exist or is not a directory.')
    if cfg.checkpoint_name is None:
        raise ValueError(f'Checkpoint name {cfg.checkpoint_name} is not valid.')
    if cfg.hparams_file is None or not os.path.isfile(cfg.hparams_file):
        raise ValueError(f'Hparams file {cfg.hparams_file} does not exist or is not a file.')


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name', 'fused_adam') == 'distributed_fused_adam'
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
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'
        if megatron_amp_O2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer, callbacks=[CustomProgressBar()])

    exp_manager(trainer, cfg.exp_manager)

    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        trainer.ckpt_path = cfg.model.resume_from_checkpoint

    logging.info(f'Resuming training from checkpoint: {trainer.ckpt_path}')

    if cfg.restore_from_path:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.restore_from_path):
            save_restore_connector.model_extracted_dir = cfg.restore_from_path
        gpt_cfg = MegatronGPTModel.restore_from(
            restore_path=cfg.restore_from_path,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        model = load_from_nemo(MegatronGPTModel, cfg, trainer, gpt_cfg, modify_confg_fn=_modify_config)
    elif cfg.model.get("pretrained_checkpoint", None) is not None:
        validate_checkpoint_loading_args(cfg.model.pretrained_checkpoint)
        model = load_from_checkpoint_dir(MegatronGPTModel, cfg, trainer, modify_confg_fn=_modify_config)
    else:
        print(' > WARNING: No checkpoint provided. Starting from scratch.')
        model = MegatronGPTModel(cfg.model, trainer)
    trainer.fit(model)


if __name__ == '__main__':
    main()
