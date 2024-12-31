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
from nemo.collections.llm.inference.base import _setup_trainer_and_restore_model
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.utils import logging


def get_modelopt_decoder_type(model: llm.GPTModel) -> str:
    """Infers the modelopt decoder type from GPTModel subclass."""
    mapping = [
        (llm.Baichuan2Model, "baichuan"),
        (llm.ChatGLMModel, "chatglm"),
        (llm.Gemma2Model, "gemma2"),
        (llm.GemmaModel, "gemma"),
        (llm.LlamaModel, "llama"),
        (llm.MistralModel, "llama"),
        (llm.MixtralModel, "llama"),
        (llm.NemotronModel, "gpt"),
        (llm.Qwen2Model, "qwen"),
        (llm.StarcoderModel, "gpt"),
        (llm.Starcoder2Model, "gpt"),
        (llm.Phi3Model, "phi3"),
    ]

    for config_class, decoder_type in mapping:
        if isinstance(model, config_class):
            return decoder_type

    logging.warning("Could not infer the decoder type")
    return None


def quantizable_model_config(model_cfg: llm.GPTConfig) -> llm.GPTConfig:
    """Modify model config for TensorRT-Model-Optimizer quantization"""

    from nemo.collections.nlp.models.language_modeling.megatron.gpt_layer_modelopt_spec import (
        get_gpt_layer_modelopt_spec,
    )

    model_cfg.transformer_layer_spec = get_gpt_layer_modelopt_spec(num_experts=model_cfg.num_moe_experts)
    if model_cfg.sequence_parallel:
        logging.warning("Disabling sequence parallelism for quantization...")
        model_cfg.sequence_parallel = False
    # Only custom ModelOpt spec is supported for quantization: this custom spec is largely based on local
    # Megatron-LM layer definitions to avoid Transformer Engine implementations that are currently not supported.
    # This layer spec also requires RoPE fusion to be disabled for tensor view operations in attention
    # layer implementation from megatron/core/transformer/dot_product_attention.py to be functional.
    model_cfg.name = "modelopt"
    model_cfg.apply_rope_fusion = False
    return model_cfg


def load_with_modelopt_layer_spec(
    nemo_checkpoint_path: str,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    inference_only: bool = True,
):
    """Loads a model from a NeMo 2.0 checkpoint using modelopt layer spec."""
    # TODO: setting ddp="pytorch" and deleting model.optim is a hackish way to disable DDP initialization.
    # Needs a systematic solution.
    if inference_only:
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,
            ckpt_parallel_save_optim=False,
            setup_optimizers=False,
            lazy_init=True,
            ddp="pytorch",
        )
    else:
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            pipeline_dtype=torch.bfloat16,
        )

    trainer = nl.Trainer(
        devices=tensor_model_parallel_size,
        num_nodes=pipeline_model_parallel_size,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision='bf16', params_dtype=torch.bfloat16, autocast_enabled=True),
    )
    model_path = Path(nemo_checkpoint_path)
    model = nl.io.load_context(path=ckpt_to_context_subdir(model_path), subpath="model")
    model.config = quantizable_model_config(model.config)

    if inference_only:
        del model.optim

    _setup_trainer_and_restore_model(nemo_checkpoint_path, trainer, model)
    return model


def get_unwrapped_mcore_model(model):
    """Unwraps NeMo 2.0 to base MCore model."""
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel

    unwrapped_model = model
    while not isinstance(unwrapped_model, MCoreGPTModel):
        unwrapped_model = unwrapped_model.module

    return unwrapped_model
