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

from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_prompt_learning_model import (
    MegatronT5PromptLearningModel,
)
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

"""
This is the script to convert the p-tuning PTL checkpoint file to nemo file for evaluation. 

Example usage:
    Assume the model has TP=2, PP=2 in the following use cases.
    ```
    python scripts/nlp_language_modeling/convert_prompt_learning_ckpt_to_nemo.py \
                trainer.devices=4 \ 
                trainer.num_nodes=1 \
                trainer.precision=bf16 \
                tensor_model_parallel_size=2 \
                pipeline_model_parallel_size=2 \
                checkpoint_dir=/results/ptune_squad/checkpoints \
                checkpoint_name='megatron_gpt_prompt_tune--val_loss=3.401-step=500.ckpt' \
                hparams_file=/results/ptune_squad/version_1/hparams.yaml
    ```
    Note, the hparam file can be found under the pytorch lightning experiment result directory. The filename is `hparams.yaml`

"""


@hydra_runner(config_path="conf", config_name="prompt_learning_ckpt_to_nemo")
def main(cfg) -> None:
    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    if cfg.checkpoint_dir:
        app_state = AppState()
        if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
            app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
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
                pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
            )
        app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
        app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
        checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))

        # check model type
        if cfg.model_type.lower() == 't5':
            model: MegatronT5PromptLearningModel = MegatronT5PromptLearningModel.load_from_checkpoint(
                checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer
            )
        elif cfg.model_type.lower() == 'gpt':
            model: MegatronGPTPromptLearningModel = MegatronGPTPromptLearningModel.load_from_checkpoint(
                checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer
            )
        else:
            raise ValueError("Model Type Not Supported!")
    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    # check whether the DDP is initialized
    if parallel_state.is_unitialized():

        def dummy():
            return

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(dummy, trainer=trainer)
        trainer.strategy.setup_environment()
    model = model.cuda()
    model.on_train_end()


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
