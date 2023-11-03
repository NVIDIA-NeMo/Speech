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

import os
import tempfile

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_glue_model import MegatronT5GLUEModel
from nemo.collections.nlp.models.language_modeling.megatron_t0_model import MegatronT0Model
from nemo.collections.nlp.models.language_modeling.megatron_t5_sft_model import MegatronT5SFTModel
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

mp.set_start_method("spawn", force=True)


def _modify_config(t5_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original t5 pre-training config (t5_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(t5_cfg, True)
    with open_dict(t5_cfg):
        t5_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
        if hasattr(t5_cfg, 'encoder') and hasattr(t5_cfg, 'decoder'):
            t5_cfg.encoder.masked_softmax_fusion = False
            t5_cfg.decoder.masked_softmax_fusion = False
            t5_cfg.encoder.hidden_dropout = cfg.model.get('hidden_dropout', 0.1)
            t5_cfg.decoder.hidden_dropout = cfg.model.get('hidden_dropout', 0.1)
            if hasattr(t5_cfg.encoder, 'ffn_dropout'):
                t5_cfg.encoder.ffn_dropout = cfg.model.get('ffn_dropout', 0.1)
            if hasattr(t5_cfg.decoder, 'ffn_dropout'):
                t5_cfg.decoder.ffn_dropout = cfg.model.get('ffn_dropout', 0.1)

            if hasattr(cfg.model, 'encoder'):
                if hasattr(cfg.model.encoder, 'position_embedding_type'):
                    t5_cfg.encoder.position_embedding_type = cfg.model.encoder.position_embedding_type
                if hasattr(cfg.model.encoder, 'use_flash_attention'):
                    t5_cfg.encoder.use_flash_attention = cfg.model.encoder.use_flash_attention
                if hasattr(cfg.model.encoder, 'attention_dropout'):
                    t5_cfg.encoder.attention_dropout = cfg.model.encoder.attention_dropout
            if hasattr(cfg.model, 'decoder'):
                if hasattr(cfg.model.decoder, 'position_embedding_type'):
                    t5_cfg.decoder.position_embedding_type = cfg.model.decoder.position_embedding_type
                if hasattr(cfg.model.decoder, 'use_flash_attention'):
                    t5_cfg.decoder.use_flash_attention = cfg.model.decoder.use_flash_attention
                if hasattr(cfg.model.decoder, 'attention_dropout'):
                    t5_cfg.decoder.attention_dropout = cfg.model.decoder.attention_dropout
        else:
            t5_cfg.hidden_dropout = cfg.model.get('hidden_dropout', 0.1)
            t5_cfg.attention_dropout = cfg.model.get('attention_dropout', 0.1)
            t5_cfg.masked_softmax_fusion = False
        t5_cfg.data = cfg.model.data
        t5_cfg.precision = cfg.trainer.precision
        t5_cfg.optim = cfg.model.optim
        t5_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
        t5_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
        # XNLI has eval languages in the yaml config.
        if hasattr(cfg.model, 'eval_languages'):
            t5_cfg.eval_languages = cfg.model.eval_languages

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(t5_cfg)
            t5_cfg.cfg = t5_cfg

    return t5_cfg


def load_from_nemo(cls, cfg, trainer, t5_cfg, modify_confg_fn):
    t5_cfg = modify_confg_fn(t5_cfg, cfg, add_cfg_to_tree=False)
    model = cls.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=t5_cfg,
        save_restore_connector=NLPSaveRestoreConnector(),
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
    t5_cfg = modify_confg_fn(hparams_file.cfg, cfg, add_cfg_to_tree=True)
    with tempfile.NamedTemporaryFile(suffix='.yaml') as f:
        OmegaConf.save(config=t5_cfg, f=f.name)
        model = cls.load_from_checkpoint(checkpoint_path=checkpoint_path, trainer=trainer, hparams_file=f.name,)
        return model


def validate_checkpoint_loading_args(cfg):
    if cfg.checkpoint_dir is None or not os.path.isdir(cfg.checkpoint_dir):
        raise ValueError(f'Checkpoint directory {cfg.checkpoint_dir} does not exist or is not a directory.')
    if cfg.checkpoint_name is None:
        raise ValueError(f'Checkpoint name {cfg.checkpoint_name} is not valid.')
    if cfg.hparams_file is None or not os.path.isfile(cfg.hparams_file):
        raise ValueError(f'Hparams file {cfg.hparams_file} does not exist or is not a file.')


@hydra_runner(config_path="conf", config_name="megatron_t5_config_finetune_glue_mnli")
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

    if hasattr(cfg.model.data.train_ds, 'task_name'):
        if cfg.model.restore_from_path:
            t5_cfg = MegatronT5GLUEModel.restore_from(
                restore_path=cfg.model.restore_from_path, trainer=trainer, return_config=True
            )
            model = load_from_nemo(MegatronT5GLUEModel, cfg, trainer, t5_cfg, modify_confg_fn=_modify_config)
        else:
            validate_checkpoint_loading_args(cfg.model.pretrained_checkpoint)
            model = load_from_checkpoint_dir(MegatronT5GLUEModel, cfg, trainer, modify_confg_fn=_modify_config)
    elif hasattr(cfg.model.data.train_ds, 'file_names'):
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

    trainer.fit(model)
    trainer.validate(model)
    if hasattr(cfg.model.data, 'test_ds'):
        trainer.test(model)


if __name__ == '__main__':
    main()
