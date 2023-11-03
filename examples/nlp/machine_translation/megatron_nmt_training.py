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


import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_bart_model import MegatronBARTModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.machine_translation.megatron_nmt_model import MegatronNMTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="aayn_base_megatron")
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

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer, callbacks=[ModelSummary(max_depth=3)])

    exp_manager(trainer, cfg.exp_manager)

    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        trainer.ckpt_path = cfg.model.resume_from_checkpoint
    logging.info(f'Resuming training from checkpoint: {trainer.ckpt_path}')

    trainer._checkpoint_connector = _CheckpointConnector(trainer)

    if hasattr(cfg.model, 'pretrained_model_path') and cfg.model.pretrained_model_path is not None:
        if not hasattr(cfg.model, 'pretrained_model_type'):
            raise ValueError(f"Pretrained model type must be in [T5, BART].")

        assert cfg.model.pretrained_model_type in ['T5', 'BART']
        if cfg.model.pretrained_model_type == 'T5':
            pretrained_cfg = MegatronT5Model.restore_from(
                cfg.model.pretrained_model_path, trainer=trainer, return_config=True
            )
        else:
            pretrained_cfg = MegatronBARTModel.restore_from(
                cfg.model.pretrained_model_path, trainer=trainer, return_config=True
            )
        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.masked_softmax_fusion = False
            # Set source and target language/multilingual
            pretrained_cfg.src_language = cfg.model.src_language
            pretrained_cfg.tgt_language = cfg.model.tgt_language
            pretrained_cfg.multilingual = cfg.model.multilingual
            pretrained_cfg.shared_tokenizer = True

            # Max generation delta
            pretrained_cfg.max_generation_delta = cfg.model.max_generation_delta

            # Set label smoothing
            pretrained_cfg.label_smoothing = cfg.model.label_smoothing

            # Set tokenizer paths:
            pretrained_cfg.encoder_tokenizer = pretrained_cfg.tokenizer
            pretrained_cfg.decoder_tokenizer = pretrained_cfg.tokenizer

            # Pre-trained models should use the legacy sentencepiece tokenizer ex: mT5
            pretrained_cfg.encoder_tokenizer.sentencepiece_legacy = True
            pretrained_cfg.decoder_tokenizer.sentencepiece_legacy = True

            # Override dropout

            # Old pre-trained checkpoints do not have separate encoder/decoder configurations, so replicate the config to encoder/decoder.
            if not hasattr(pretrained_cfg, 'encoder'):
                assert not hasattr(pretrained_cfg, 'decoder')
                logging.warning(
                    "No separate configuration for encoder, found in pretrained model, using encoder dropout settings everywhere."
                )
                pretrained_cfg.hidden_dropout = cfg.model.encoder.hidden_dropout
                pretrained_cfg.attention_dropout = cfg.model.encoder.attention_dropout
            else:
                assert hasattr(pretrained_cfg, 'decoder') and hasattr(pretrained_cfg, 'encoder')
                pretrained_cfg.encoder.hidden_dropout = cfg.model.encoder.hidden_dropout
                pretrained_cfg.encoder.attention_dropout = cfg.model.encoder.attention_dropout
                pretrained_cfg.decoder.hidden_dropout = cfg.model.decoder.hidden_dropout
                pretrained_cfg.decoder.attention_dropout = cfg.model.decoder.attention_dropout

            # Override precision
            pretrained_cfg.precision = trainer.precision  # Set above from trainer.precision

            # Override micro/global batch
            pretrained_cfg.micro_batch_size = cfg.model.micro_batch_size
            pretrained_cfg.global_batch_size = cfg.model.global_batch_size

            # O2 AMP
            pretrained_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)

            # Override data and global/micro batch size.
            pretrained_cfg.train_ds = cfg.model.train_ds
            pretrained_cfg.train_ds.micro_batch_size = cfg.model.micro_batch_size
            pretrained_cfg.train_ds.global_batch_size = cfg.model.global_batch_size
            if hasattr(cfg.model, 'validation_ds'):
                pretrained_cfg.validation_ds = cfg.model.validation_ds
            else:
                raise AttributeError(f"No validation dataset found in config.")
            if hasattr(cfg.model, 'test_ds'):
                pretrained_cfg.test_ds = cfg.model.test_ds

            # Class target for the new class being restored.
            pretrained_cfg.target = (
                "nemo.collections.nlp.models.machine_translation.megatron_nmt_model.MegatronNMTModel"
            )

            # Optimizer overrides.
            pretrained_cfg.optim = cfg.model.optim

        model = MegatronNMTModel.restore_from(
            cfg.model.pretrained_model_path,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
    else:
        model = MegatronNMTModel(cfg.model, trainer)

    trainer.fit(model)
    trainer.validate(model)


if __name__ == '__main__':
    main()
