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


from typing import Optional

import lightning.pytorch as pl
import nemo_run as run
import torch

from nemo import lightning as nl
from nemo.collections import avlm, llm
from nemo.collections.avlm import AVLMMockDataModule
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.peft import LoRA
from nemo.collections.llm.recipes.finetune_default import nemo_resume
from nemo.collections.llm.recipes.log.default import default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.utils.exp_manager import TimingCallback

NAME = "avlm_8b"


def create_image_processor():
    """
    Create an image processor for the AVLM 8B model.
    This helps by pass fiddle.config
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")


@run.cli.factory(name=NAME)
def model(config=run.Config(avlm.AVLMConfig8B)) -> run.Config[pl.LightningModule]:
    """
    Factory function to create a AVLM 8B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the AVLM 8B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=avlm_8b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(avlm.AVLMModel, config=config)


@run.cli.factory(target=llm.finetune, name=NAME)
def finetune_recipe(
    checkpoint_path: str = "",
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'none',
    freeze_modules: Optional[dict] = {
        "freeze_language_model": False,
        "freeze_vision_model": True,
        "freeze_audio_model": True,
        "freeze_vision_projection": False,
        "freeze_audio_projection": False,
    },
) -> run.Partial:
    """
    Create a fine-tuning recipe for AVLM 8B model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory llava_next_7b

        Python API usage:
            >>> recipe = finetune_recipe(name="llava_next_7b_finetune", num_nodes=1)
            >>> print(recipe)
    """

    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        encoder_pipeline_model_parallel_size=0,
        pipeline_dtype=torch.bfloat16,
        ckpt_async_save=False,
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        devices=num_gpus_per_node,
        limit_val_batches=10,
        log_every_n_steps=1,
        max_steps=5190,
        num_nodes=num_nodes,
        plugins=bf16_mixed(),
        strategy=strategy,
        val_check_interval=1000,
        callbacks=[run.Config(TimingCallback)],
    )

    recipe = run.Partial(
        llm.finetune,
        model=model(
            config=run.Config(
                avlm.AVLMConfig8B,
                freeze_language_model=freeze_modules["freeze_language_model"],
                freeze_vision_model=freeze_modules["freeze_vision_model"],
                freeze_audio_model=freeze_modules["freeze_audio_model"],
                freeze_vision_projection=freeze_modules["freeze_vision_projection"],
                freeze_audio_projection=freeze_modules["freeze_audio_projection"],
            )
        ),
        trainer=trainer,
        data=run.Config(
            AVLMMockDataModule,
            seq_length=8192,
            global_batch_size=8,
            micro_batch_size=2,
            tokenizer=run.Config(AutoTokenizer, "meta-llama/Meta-Llama-3-8B"),
            image_processor=run.Config(create_image_processor),
            num_workers=4,
        ),
        log=llm.default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=2.0e-05, min_lr=2.0e-07, warmup_steps=150),
        resume=nemo_resume(checkpoint_path),
    )

    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.tensor_model_parallel_size = 2
        recipe.optim.config.lr = 2e-05
    elif peft_scheme.lower() == 'lora':
        recipe.peft = run.Config(
            LoRA,
            target_modules=[
                "*.language_model.*.linear_qkv",
                "*.language_model.*.linear_q",
                "*.language_model.*.linear_kv",
                "*.language_model.*.linear_proj",
                "*.language_model.*.linear_fc1",
                "*.language_model.*.linear_fc2",
            ],
        )
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")

    return recipe


@run.cli.factory(target=llm.pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    language_model_from_pretrained: Optional[str] = None,
    freeze_modules: Optional[dict] = {
        "freeze_language_model": True,
        "freeze_vision_model": True,
        "freeze_audio_model": True,
        "freeze_vision_projection": False,
        "freeze_audio_projection": False,
    },
) -> run.Partial:
    """
    Create a Pre-training recipe for AVLM 8B model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory llava_next_7b

        Python API usage:
            >>> recipe = finetune_recipe(name="llava_next_7b_pretrain", num_nodes=1)
            >>> print(recipe)
    """

    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        encoder_pipeline_model_parallel_size=0,
        pipeline_dtype=torch.bfloat16,
        ckpt_async_save=False,
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        devices=num_gpus_per_node,
        limit_val_batches=10,
        log_every_n_steps=1,
        max_steps=5190,
        num_nodes=num_nodes,
        plugins=bf16_mixed(),
        strategy=strategy,
        val_check_interval=1000,
        callbacks=[run.Config(TimingCallback)],
    )

    recipe = run.Partial(
        llm.pretrain,
        model=model(
            config=run.Config(
                avlm.AVLMConfig8B,
                language_model_from_pretrained=language_model_from_pretrained,
                freeze_language_model=freeze_modules["freeze_language_model"],
                freeze_vision_model=freeze_modules["freeze_vision_model"],
                freeze_audio_model=freeze_modules["freeze_audio_model"],
                freeze_vision_projection=freeze_modules["freeze_vision_projection"],
                freeze_audio_projection=freeze_modules["freeze_audio_projection"],
            )
        ),
        trainer=trainer,
        data=run.Config(
            AVLMMockDataModule,
            seq_length=8192,
            global_batch_size=8,
            micro_batch_size=1,
            tokenizer=run.Config(AutoTokenizer, "meta-llama/Meta-Llama-3-8B"),
            image_processor=run.Config(create_image_processor),
            num_workers=4,
        ),
        log=llm.default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=0.001, min_lr=2.0e-05, warmup_steps=150),
        resume=default_resume(),
    )

    return recipe
