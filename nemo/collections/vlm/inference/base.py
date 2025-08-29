# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional, Union

import lightning.pytorch as pl
import torch
import torch.distributed
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from PIL.Image import Image
from transformers import AutoProcessor

import nemo.lightning as nl
from nemo.collections import vlm

from .llava_inference_wrapper import LlavaInferenceWrapper
from .mllama_inference_wrapper import MllamaInferenceWrapper
from .qwenvl_inference_wrapper import QwenVLInferenceWrapper
from .vlm_engine import VLMEngine
from .vlm_inference_controller import QwenVLTextGenerationController, VLMTextGenerationController


def _setup_trainer_and_restore_model(path: str, trainer: nl.Trainer, model: pl.LightningModule):
    """Setup trainer and restore model from path"""

    # [ModelOpt]: If modelopt_state exists, overwrite transformer_layer_spec to modelopt spec
    from nemo.collections.vlm.modelopt import set_modelopt_spec_if_exists_in_ckpt

    set_modelopt_spec_if_exists_in_ckpt(model, path)

    fabric = trainer.to_fabric()
    model = fabric.load_model(path, model)
    return model


def setup_inference_wrapper(
    model,
    tokenizer,
    params_dtype: torch.dtype = torch.bfloat16,
    inference_batch_times_seqlen_threshold: int = 1000,
):
    """Set up inference wrapper for the model"""
    config = model.config

    mcore_model = model.module.module.cuda()
    mcore_model = mcore_model.to(params_dtype)
    mcore_model.eval()

    if isinstance(config, vlm.MLlamaModelConfig):
        wrapper_cls = MllamaInferenceWrapper
        hidden_size = config.language_model_config.hidden_size
    elif isinstance(config, vlm.LlavaConfig):
        wrapper_cls = LlavaInferenceWrapper
        hidden_size = config.language_transformer_config.hidden_size
    elif isinstance(config, vlm.Qwen2VLConfig):
        wrapper_cls = QwenVLInferenceWrapper
        hidden_size = config.language_transformer_config.hidden_size
    else:
        raise ValueError(f"Unknown model config: {config}")

    inference_wrapped_model = wrapper_cls(
        mcore_model,
        InferenceWrapperConfig(
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=tokenizer.vocab_size,
        ),
    )

    return inference_wrapped_model


def setup_model_and_tokenizer(
    path: str,
    trainer: Optional[nl.Trainer] = None,
    tp_size: int = 1,
    pp_size: int = 1,
    params_dtype: torch.dtype = torch.bfloat16,
    inference_batch_times_seqlen_threshold: int = 1000,
):
    """Set up model and tokenizer"""
    model_context = nl.io.load_context(path=nl.ckpt_utils.ckpt_to_context_subdir(path), subpath="model")
    model_config = model_context.config

    if isinstance(model_config, vlm.MLlamaModelConfig):
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        processor = AutoProcessor.from_pretrained(model_id)
        model = vlm.MLlamaModel(model_config, tokenizer=processor.tokenizer)
    elif isinstance(model_config, vlm.LlavaConfig):
        model_id = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(model_id)
        model = vlm.LlavaModel(model_config, tokenizer=processor.tokenizer)
    elif isinstance(model_config, vlm.Qwen2VLConfig):
        if model_config.vision_projection_config.projector_type != "mcore_mlp":
            raise ValueError("Only support Qwen2.5-VL with mcore_mlp projector type")
        if model_config.vision_projection_config.hidden_size == 2048:
            model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        elif model_config.vision_projection_config.hidden_size == 3584:
            model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        elif model_config.vision_projection_config.hidden_size == 5120:
            model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
        elif model_config.vision_projection_config.hidden_size == 8192:
            model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
        else:
            raise ValueError(f"Unknown model size: {model_config}")
        min_pixels = 16 * 28 * 28
        max_pixels = 64 * 28 * 28
        processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        model = vlm.Qwen2VLModel(model_config, tokenizer=processor.tokenizer, model_version="qwen25-vl")
    else:
        raise ValueError(f"Unknown model config: {model_config}")

    if trainer is None:
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            ckpt_include_optimizer=False,
        )
        trainer = nl.Trainer(
            devices=tp_size * pp_size,
            max_steps=1000,
            accelerator="gpu",
            strategy=strategy,
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            val_check_interval=1000,
            limit_val_batches=50,
        )

    _setup_trainer_and_restore_model(path=path, trainer=trainer, model=model)

    inference_wrapped_model = setup_inference_wrapper(
        model, processor.tokenizer, params_dtype, inference_batch_times_seqlen_threshold
    )

    return inference_wrapped_model, processor


def generate(
    wrapped_model: AbstractModelInferenceWrapper,
    tokenizer,
    image_processor,
    prompts: List[str],
    images: List[Union[Image, List[Image]]],
    processor=None,
    max_batch_size: int = 4,
    random_seed: Optional[int] = None,
    inference_params: Optional[CommonInferenceParams] = None,
) -> dict:
    """
    Generates text using a NeMo VLM model.
    Args:
        wrapped_model (AbstractModelInferenceWrapper): The model inference wrapper.
        tokenizer: tokenizer for the input text,
        image_processor: image processor for the input image,
        prompts (list[str]): The list of prompts to generate text for.
        images (list): The list of images to generate text for.
        max_batch_size (int, optional): The maximum batch size. Defaults to 4.
        random_seed (Optional[int], optional): The random seed. Defaults to None.
        inference_params (Optional["CommonInferenceParams"], optional): The inference parameters defined in
            Mcore's CommonInferenceParams. Defaults to None.

    Returns:
        list[Union["InferenceRequest", str]]: A list of generated text,
            either as a string or as an InferenceRequest object.
    """
    if isinstance(wrapped_model, QwenVLInferenceWrapper):
        text_generation_controller = QwenVLTextGenerationController(
            inference_wrapped_model=wrapped_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            processor=processor,
        )
    else:
        text_generation_controller = VLMTextGenerationController(
            inference_wrapped_model=wrapped_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
    mcore_engine = VLMEngine(
        text_generation_controller=text_generation_controller, max_batch_size=max_batch_size, random_seed=random_seed
    )

    common_inference_params = inference_params or CommonInferenceParams(num_tokens_to_generate=50)

    results = mcore_engine.generate(
        prompts=prompts,
        images=images,
        common_inference_params=common_inference_params,
    )

    return results
