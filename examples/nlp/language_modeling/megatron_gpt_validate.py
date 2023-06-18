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

from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import (
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank

""" Example script showing how to run validation on a MegatronGPT model.

    Sample usage:

    From nemo model:

    python megatron_gpt_validate.py \
        trainer.devices=4 \
        trainer.num_nodes=1 \
        trainer.limit_val_batches=10 \
        trainer.max_steps=100 \
        tensor_model_parallel_size=1 \
        pipeline_model_parallel_size=4 \
        trainer.precision=bf16 \
        gpt_model_file=/path/to/megatron_gpt_tp_1_pp4.nemo
    
    from PTL checkpoint:
    python megatron_gpt_validate.py \
        trainer.devices=4 \
        trainer.num_nodes=1 \
        trainer.limit_val_batches=10 \
        trainer.max_steps=100 \
        tensor_model_parallel_size=1 \
        pipeline_model_parallel_size=4 \
        virtual_pipeline_model_parallel_size=4 \
        trainer.precision=bf16 \
        checkpoint_dir='/path/to/experiment/checkpoints' \
        checkpoint_name='megatron_gpt--val_loss=7.78-step=100-consumed_samples=6336.0-last.ckpt' \
        hparams_file='/path/to/experiment/hparams.yaml

"""


def modify_pretrained_cfg(pretrained_cfg, trainer, cfg):
    with open_dict(pretrained_cfg):
        OmegaConf.set_struct(pretrained_cfg, True)
        pretrained_cfg.sequence_parallel = False
        pretrained_cfg.activations_checkpoint_granularity = None
        pretrained_cfg.activations_checkpoint_method = None
        pretrained_cfg.precision = trainer.precision
        if cfg.micro_batch_size is not None:
            pretrained_cfg.micro_batch_size = cfg.micro_batch_size
        if cfg.global_batch_size is not None:
            pretrained_cfg.global_batch_size = cfg.global_batch_size
        if trainer.precision == "16":
            pretrained_cfg.megatron_amp_O2 = False
    return pretrained_cfg


@hydra_runner(config_path="conf", config_name="megatron_gpt_validate_config")
def main(cfg) -> None:

    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    if cfg.gpt_model_file:
        logging.info(f"Restoring model from {cfg.gpt_model_file}")
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.gpt_model_file

        pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        pretrained_cfg = modify_pretrained_cfg(pretrained_cfg, trainer, cfg)
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
            map_location=f'cuda:{trainer.local_rank}',  # map_location is needed for converted models
        )
    elif cfg.checkpoint_dir:
        logging.info(
            f"Restoring model from checkpoint_dir: {cfg.checkpoint_dir} with checkpoint name: {cfg.checkpoint_name}"
        )
        app_state = AppState()
        if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
            app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            app_state.virtual_pipeline_model_parallel_size = cfg.virtual_pipeline_model_parallel_size
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
                tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
                pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size_=cfg.virtual_pipeline_model_parallel_size,
            )
        checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
        pretrained_cfg = OmegaConf.load(cfg.hparams_file)
        pretrained_cfg = modify_pretrained_cfg(pretrained_cfg.cfg, trainer, cfg)
        with tempfile.NamedTemporaryFile(suffix='.yaml') as f:
            OmegaConf.save(config=pretrained_cfg, f=f.name)
            model = MegatronGPTModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path, trainer=trainer, hparams_file=f.name,
            )
    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    logging.info("\n\n**************  Model configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(model.cfg)}')

    trainer.validate(model=model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
