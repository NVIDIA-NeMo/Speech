# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

import torch

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.utils import logging


def quantizable_model_config(model_cfg: llm.GPTConfig) -> llm.GPTConfig:
    """Modify model config for TensorRT Model Optimizer"""

    from nemo.collections.nlp.models.language_modeling.megatron.gpt_layer_modelopt_spec import (
        get_gpt_layer_modelopt_spec,
    )

    model_cfg.transformer_layer_spec = get_gpt_layer_modelopt_spec()
    if model_cfg.sequence_parallel:
        logging.warning("Disabling sequence parallelism for quantization...")
        model_cfg.sequence_parallel = False
    # Only custom ModelOpt spec is supported for Quantization: this custom spec is largely based on local Megatron-LM
    # layer definitions to avoid Transformer Engine implementations that are currently not supported.
    # This layer spec also requires RoPE fusion to be disabled for tensor view operations in attention
    # layer implementation from megatron/core/transformer/dot_product_attention.py to be functional.
    model_cfg.name = "modelopt"
    model_cfg.apply_rope_fusion = False
    return model_cfg


def load_with_modelopt_layer_spec(nemo_checkpoint_path: str, calib_tp: int = 1, calib_pp: int = 1) -> llm.GPTModel:
    trainer = nl.Trainer(
        devices=calib_tp,
        num_nodes=calib_pp,
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=calib_tp, pipeline_model_parallel_size=calib_pp, pipeline_dtype=torch.bfloat16
        ),
        plugins=nl.MegatronMixedPrecision(precision='bf16', pipeline_dtype=torch.bfloat16, autocast_enabled=True),
    )
    fabric = trainer.to_fabric()
    fabric.launch()

    model_path = Path(nemo_checkpoint_path)
    model = nl.io.load_context(ckpt_to_context_subdir(model_path)).model
    model.config = quantizable_model_config(model.config)
    return fabric.load_model(nemo_checkpoint_path, model=model)


def get_unwrapped_mcore_model(model: llm.GPTModel):
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel

    unwrapped_model = model
    while not isinstance(unwrapped_model, MCoreGPTModel):
        unwrapped_model = unwrapped_model.module

    return unwrapped_model
