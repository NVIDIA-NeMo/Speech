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


from typing import Optional

import lightning.pytorch as pl
import nemo_run as run
import torch

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.llm.recipes.finetune_default import nemo_resume
from nemo.collections.llm.recipes.log.default import tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.collections.vlm.mllama.data.mock import MockDataModule
from nemo.utils.exp_manager import TimingCallback

NAME = "mllama_11b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Llama-3.2-Vision 11B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Llama-3.2-Vision 11B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=mllama_11b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(vlm.MLlamaModel, config=run.Config(vlm.MLlamaConfig11BInstruct))


@run.cli.factory(target=llm.finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'lora',
) -> run.Partial:
    """
    Create a fine-tuning recipe for Llama3.2 11B model.

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
            $ nemo llm finetune --factory mllama_11b

        Python API usage:
            >>> recipe = finetune_recipe(name="mllama_11b_finetune", num_nodes=1)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning. For more information
        on fine-tuning LLMs with NeMo, see the fine-tuning guide in the
        `examples/llm/finetune/` directory.
    """

    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        encoder_pipeline_model_parallel_size=0,
        pipeline_dtype=torch.bfloat16,
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        devices=num_gpus_per_node,
        limit_val_batches=2,
        log_every_n_steps=10,
        max_steps=5190,
        num_nodes=num_nodes,
        plugins=bf16_mixed(),
        strategy=strategy,
        val_check_interval=100,
        callbacks=[run.Config(TimingCallback)],
    )

    recipe = run.Partial(
        llm.finetune,
        model=model(),
        trainer=trainer,
        data=run.Config(
            MockDataModule,
            seq_length=6404,  # encoder (vision) seq length
            decoder_seq_length=2048,  # decoder (llm) seq length
            global_batch_size=2,
            micro_batch_size=1,
            vocab_size=128256,
            crop_size=(560, 560),
            num_workers=0,
        ),
        log=llm.default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=1e-4, min_lr=2.0e-07, warmup_steps=150),
        resume=nemo_resume("meta-llama/Llama-3.2-11B-Vision-Instruct"),
    )

    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.tensor_model_parallel_size = 2
        recipe.optim.config.lr = 2e-05
    elif peft_scheme.lower() == 'lora':
        # pylint: disable=line-too-long
        """Adapted from https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/configs/peft.py"""
        recipe.peft = run.Config(
            vlm.LoRA,
            freeze_vision_model=True,
            target_modules=[
                "linear_qkv",
                "linear_q",
                "linear_kv",
            ],
            dim=8,
            alpha=32,
            dropout=0.05,
            dropout_position="pre",
        )
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")

    return recipe
