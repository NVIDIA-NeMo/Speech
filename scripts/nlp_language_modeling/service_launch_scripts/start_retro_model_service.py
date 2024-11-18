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

import torch
from lightning.pytorch import Trainer
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

"""
This is the script to launch RETRO Model text generation server.

Usage:
    Assume the model has TP=1, PP=1
    run greedy inference from a nemo file:
        python megatron_retro_eval.py \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            trainer.accelerator=gpu \
            trainer.precision=16 \
            inference.tokens_to_generate=128 \
            inference.greedy=True \
            retro_model_file=path_to_retro_nemo_file \
            tensor_model_parallel_size=1 \
            pipeline_model_parallel_size=1 \
            retrieval_service.faiss_devices='0' \
            retrieval_service.faiss_index=path_to_faiss_index \
            retrieval_service.retrieval_index=path_to_retrieval_dataset \
            retrieval_service.neighbors=20
"""


@hydra_runner(config_path="conf", config_name="retro_text_generation_server")
def main(cfg) -> None:
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    model_path = cfg.retro_model_file

    save_restore_connector = NLPSaveRestoreConnector()

    if os.path.isdir(model_path):
        save_restore_connector.model_extracted_dir = model_path

    model_cfg = MegatronRetrievalModel.restore_from(
        model_path,
        trainer=trainer,
        return_config=True,
        save_restore_connector=save_restore_connector,
    )

    with open_dict(model_cfg):
        model_cfg.precision = trainer.precision
        model_cfg.sequence_parallel = False
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.activations_checkpoint_method = None

    model = MegatronRetrievalModel.restore_from(
        model_path,
        trainer=trainer,
        save_restore_connector=save_restore_connector,
        override_config_path=model_cfg,
    )

    # check whether the DDP is initialized
    if not parallel_state.is_initialized():

        def dummy():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

    retrieval_service = OmegaConf.to_container(cfg.retrieval_service)
    model.set_inference_config(None, retrieval_service)

    # running text generation, use inference server
    if parallel_state.is_pipeline_first_stage() and parallel_state.get_tensor_model_parallel_rank() == 0:
        server = MegatronServer(model.cuda(), inference_strategy=model.inference_strategy)
        server.run("0.0.0.0", port=cfg.port)

    while True:
        choice = torch.cuda.LongTensor(1)
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 0:
            generate(model.cuda(), strategy=model.inference_strategy)


if __name__ == '__main__':
    main()
