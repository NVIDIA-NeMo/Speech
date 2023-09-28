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


import torch
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_t5_prompt_learning_model import (
    MegatronT5PromptLearningModel,
)
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.utils.decorators import deprecated

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


@deprecated(
    explanation=f"{__file__} is deprecated. Please use MegatronT5SFTModel.add_adapter() for PEFT features."
    "See updated scripts `megatron_t5_peft_tuning.py` and `megatron_t5_peft_eval.py` for examples."
)
@hydra_runner(config_path="conf", config_name="megatron_t5_prompt_learning_inference")
def main(cfg) -> None:

    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    if (
        cfg.tensor_model_parallel_size < 0
        or cfg.pipeline_model_parallel_size < 0
        or cfg.get('pipeline_model_parallel_split_rank', -1) < 0
    ):
        model_config = MegatronT5PromptLearningModel.restore_from(
            restore_path=cfg.language_model_path, trainer=trainer, return_config=True,
        )

        with open_dict(cfg):
            cfg.tensor_model_parallel_size = model_config.get('tensor_model_parallel_size', 1)
            cfg.pipeline_model_parallel_size = model_config.get('pipeline_model_parallel_size', 1)
            cfg.pipeline_model_parallel_split_rank = model_config.get('pipeline_model_parallel_split_rank', 0)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

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

    # Load prompt tuned model, virtual_prompt_model_file and language_model_path must be provided in config
    if cfg.get('virtual_prompt_model_file', None) is not None and cfg.get('language_model_path', None) is not None:

        # Update frozen T5 model path in case it has changed
        prompt_learning_cfg = MegatronT5PromptLearningModel.restore_from(
            cfg.virtual_prompt_model_file, trainer=trainer, return_config=True
        )
        with open_dict(prompt_learning_cfg):
            if cfg.get("language_model_path"):
                # This is for backward compatibility with old checkpoints that used `pretrained_language_model_path` instead of `language_model_path`.
                if hasattr(prompt_learning_cfg, 'pretrained_language_model_path'):
                    prompt_learning_cfg.pretrained_language_model_path = cfg.language_model_path
                else:
                    prompt_learning_cfg.language_model_path = cfg.language_model_path
            prompt_learning_cfg.micro_batch_size = cfg.data.get('micro_batch_size', 4)
            prompt_learning_cfg.global_batch_size = cfg.data.get('global_batch_size', 4)

        # Now load prompt learning model with frozen T5 model base
        model = MegatronT5PromptLearningModel.restore_from(
            restore_path=cfg.virtual_prompt_model_file, trainer=trainer, override_config_path=prompt_learning_cfg
        )

    else:
        raise ValueError("virtual_prompt_model_file and pretrained_language_model_file must be provided in config")

    # check whether the DDP is initialized
    if parallel_state.is_unitialized():

        def dummy():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

    model.freeze()

    _, test_dl = model.build_virtual_prompt_dataset(
        dataset_paths=cfg.data.test_ds,
        batch_size=cfg.data.global_batch_size,
        for_train=False,
        drop_last=False,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    outputs = trainer.predict(model, test_dl)
    with open(cfg.pred_file_path, "w", encoding="utf-8") as pred_file:
        for batch in outputs:
            preds = batch["preds_text"]
            for pred in preds:
                pred = pred.strip().replace("\n", " ")
                pred_file.write(pred + "\n")
    print('test finish---------------------------------')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
