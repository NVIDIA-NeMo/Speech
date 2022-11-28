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


from lightning_lite.plugins.environments import TorchElasticEnvironment
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager


@hydra_runner(config_path="conf", config_name="megatron_t5_lm_adaptation_finetune")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )
    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_o2:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer, callbacks=[ModelSummary(max_depth=3)])
    exp_manager(trainer, cfg.exp_manager)

    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path
    logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    if hasattr(cfg.model, 'pretrained_model_path') and cfg.model.pretrained_model_path is not None:
        pretrained_cfg = MegatronT5Model.restore_from(
            cfg.model.pretrained_model_path, trainer=trainer, return_config=True
        )
        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):

            # Override data from T5 to Prefix-LM
            encoder_seq_length = pretrained_cfg.data.seq_length
            decoder_seq_length = (
                pretrained_cfg.data.seq_length
            )  # Set decoder seq length to be enoder seq length for prefix-lm
            pretrained_cfg.data = cfg.model.data
            pretrained_cfg.data.seq_length = encoder_seq_length
            pretrained_cfg.data.seq_length_dec = (
                decoder_seq_length - 1
            )  # -1 is to account for the addition of <bos> and <eos> and right shifting to create targets.

            # Override fusion params.
            pretrained_cfg.masked_softmax_fusion = cfg.model.masked_softmax_fusion
            pretrained_cfg.bias_dropout_add_fusion = cfg.model.bias_dropout_add_fusion
            pretrained_cfg.bias_gelu_fusion = cfg.model.bias_gelu_fusion

            # Override dropout
            if cfg.model.hidden_dropout is not None:
                pretrained_cfg.hidden_dropout = cfg.model.hidden_dropout

            if cfg.model.attention_dropout is not None:
                pretrained_cfg.attention_dropout = cfg.model.attention_dropout

            # Override precision
            pretrained_cfg.precision = trainer.precision  # Set above from trainer.precision

            # Override micro/global batch
            pretrained_cfg.micro_batch_size = cfg.model.micro_batch_size
            pretrained_cfg.global_batch_size = cfg.model.global_batch_size

            # O2 AMP
            pretrained_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)

            # Optimizer overrides.
            pretrained_cfg.optim = cfg.model.optim

        model = MegatronT5Model.restore_from(
            cfg.model.pretrained_model_path,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
    else:
        raise ValueError(f'No pretrained model path specified or does not exist {cfg.model.pretrained_model_path}')

    trainer.fit(model)


if __name__ == '__main__':
    main()
