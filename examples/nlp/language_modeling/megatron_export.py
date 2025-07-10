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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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

from lightning.pytorch import Trainer
from omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_bart_model import MegatronBARTModel
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_retrieval_model import MegatronRetrievalModel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model

try:
    from nemo.collections.nlp.models.machine_translation.megatron_nmt_model import MegatronNMTModel
except ModuleNotFoundError:
    from abc import ABC

    MegatronNMTModel = ABC
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core import ModelPT
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank


def get_model_class(cfg):
    if cfg.model_type == 'gpt':
        return MegatronGPTModel
    elif cfg.model_type == 'bert':
        return MegatronBertModel
    elif cfg.model_type == 't5':
        return MegatronT5Model
    elif cfg.model_type == 'bart':
        return MegatronBARTModel
    elif cfg.model_type == 'nmt':
        return MegatronNMTModel
    elif cfg.model_type == 'retro':
        return MegatronRetrievalModel
    else:
        raise ValueError("Invalid Model Type")


@hydra_runner(config_path="conf", config_name="megatron_gpt_export")
def nemo_export(cfg):
    """Convert a nemo model into .onnx ONNX format."""
    nemo_in = None
    if cfg.gpt_model_file:
        nemo_in = cfg.gpt_model_file
    elif cfg.checkpoint_dir:
        nemo_in = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
    assert nemo_in is not None, "NeMo model not provided. Please provide the path to the .nemo or .ckpt file"

    onnx_out = cfg.onnx_model_file

    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    logging.info("Restoring NeMo model from '{}'".format(nemo_in))
    try:
        if cfg.gpt_model_file:
            save_restore_connector = NLPSaveRestoreConnector()
            if os.path.isdir(cfg.gpt_model_file):
                save_restore_connector.model_extracted_dir = cfg.gpt_model_file

            pretrained_cfg = ModelPT.restore_from(
                restore_path=cfg.gpt_model_file,
                trainer=trainer,
                return_config=True,
                save_restore_connector=save_restore_connector,
            )
            OmegaConf.set_struct(pretrained_cfg, True)
            with open_dict(pretrained_cfg):
                pretrained_cfg.sequence_parallel = False
                pretrained_cfg.activations_checkpoint_granularity = None
                pretrained_cfg.activations_checkpoint_method = None
                pretrained_cfg.precision = trainer.precision
                if trainer.precision == "16":
                    pretrained_cfg.megatron_amp_O2 = False
            model = ModelPT.restore_from(
                restore_path=cfg.gpt_model_file,
                trainer=trainer,
                override_config_path=pretrained_cfg,
                save_restore_connector=save_restore_connector,
            )
        elif cfg.checkpoint_dir:
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
            checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
            model_cls = get_model_class(cfg)
            model = model_cls.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
        else:
            raise ValueError("need at least a nemo file or checkpoint dir")
    except Exception as e:
        logging.error(
            "Failed to restore model from NeMo file : {}. Please make sure you have the latest NeMo package installed with [all] dependencies.".format(
                nemo_in
            )
        )
        raise e

    logging.info("Model {} restored from '{}'".format(model.__class__.__name__, nemo_in))

    # Export
    check_trace = cfg.export_options.runtime_check

    try:
        model.to(device=cfg.export_options.device).freeze()
        model.eval()
        model.export(
            onnx_out,
            onnx_opset_version=cfg.export_options.onnx_opset,
            do_constant_folding=cfg.export_options.do_constant_folding,
            dynamic_axes={
                'input_ids': {0: "sequence", 1: "batch"},
                'position_ids': {0: "sequence", 1: "batch"},
                'logits': {0: "sequence", 1: "batch"},
            },
            check_trace=check_trace,
            check_tolerance=cfg.export_options.check_tolerance,
            verbose=cfg.export_options.verbose,
        )
    except Exception as e:
        logging.error(
            "Export failed. Please make sure your NeMo model class ({}) has working export() and that you have the latest NeMo package installed with [all] dependencies.".format(
                model.__class__
            )
        )
        raise e


if __name__ == '__main__':
    nemo_export()
