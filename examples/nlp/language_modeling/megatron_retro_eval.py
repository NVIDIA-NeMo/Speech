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

import datetime
import os

import torch
from lightning.pytorch.trainer.trainer import Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.models.language_modeling.megatron_retro_model import MegatronRetroModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank

"""
This is the script to run Retro text generation.

Usage:
    Currently, Mcore-based RETRO only support batch-size of 1.
    Example running greedy inference from a distributed checkpoint dir:
        python megatron_retro_eval.py \
            checkpoint_dir=PATH_TO_CHECKPOINT \
            checkpoint_name=CHECKPOINT_NAME \
            inference.greedy=True \
            inference.add_BOS=False \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            prompt="sample prompt" \
            inference.retro_inference.retro_num_neighbors=2 \
            neighbors=["neighbor text 1", "neighbor text 2"]


        ```
"""

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RequestDataSet(Dataset):
    def __init__(self, sentences, neighbors):
        super().__init__()
        self.sentences = sentences
        self.neighbors = neighbors

    def __len__(
        self,
    ):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {'prompts': self.sentences[idx], 'neighbors': self.neighbors[idx]}


@hydra_runner(config_path="conf", config_name="megatron_retro_inference")
def main(cfg) -> None:

    # trainer required for restoring model parallel models
    trainer = Trainer(
        strategy=NLPDDPStrategy(timeout=datetime.timedelta(seconds=18000)),
        **cfg.trainer,
        callbacks=[CustomProgressBar()],
    )

    if cfg.checkpoint_dir:
        app_state = AppState()
        if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
            app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
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
        checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        # checkpoint_path is a dir in case of distributed checkpointing
        if not os.path.isdir(checkpoint_path):
            # legacy checkpoint needs model parallel rank injection
            checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
        model = MegatronRetroModel.load_from_checkpoint(
            checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer
        )
    else:
        raise ValueError("Requiring distributed checkpoint dir for loading Mcore RETRO.")

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    prompt = [cfg.prompt]
    neighbors = [cfg.neighbors]
    ds = RequestDataSet(prompt, neighbors)
    bs = 1
    request_dl = DataLoader(dataset=ds, batch_size=bs)
    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)

    response = trainer.predict(model, request_dl)

    print("***************************")
    print(response)
    print("***************************")


if __name__ == '__main__':
    main()
