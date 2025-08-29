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

from nemo.collections.llm import GPTOSSConfig120B, GPTOSSModel
from nemo.collections.llm.api import finetune
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe

NAME = "gpt_oss_120b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create an GPT-OSS 120B model configuration.
    This is a MoE (Mixture of Experts) model with 128 experts.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the GPT-OSS 120B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=gpt_oss_120b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    conf = run.Config(GPTOSSConfig120B)
    return run.Config(GPTOSSModel, config=conf)


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    resume_path: str = "openai/gpt-oss-120b",
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'lora',
    packed_sequence: bool = False,
) -> run.Partial:
    """
    Create a fine-tuning recipe for GPT-OSS 120B model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.
    This model uses Mixture of Experts (MoE) architecture with 128 experts.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        resume_path (str): Path to the NeMo checkpoint
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning.
            Allowed values: 'lora'/'dora'/'none'/None.
        packed_sequence (Optional[bool]): Packing multiple training sequences into one long sequence for training
            efficiency. Default sequence length is 2048.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory gpt_oss_120b

        Python API usage:
            >>> recipe = finetune_recipe(name="gpt_oss_120b_finetune", num_nodes=8)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning.
    """
    recipe = default_finetune_recipe(model(), resume_path, dir, name, num_nodes, num_gpus_per_node, packed_sequence)
    recipe.trainer.strategy.expert_tensor_parallel_size = 1
    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.tensor_model_parallel_size = 1
        recipe.trainer.strategy.expert_model_parallel_size = 8
        recipe.trainer.strategy.pipeline_model_parallel_size = 4
        recipe.optim.config.lr = 5e-6
        recipe.trainer.strategy.ckpt_async_save = False
    elif peft_scheme.lower() in ['lora', 'dora']:
        recipe.trainer.strategy.tensor_model_parallel_size = 1
        recipe.trainer.strategy.expert_model_parallel_size = 8
        recipe.trainer.strategy.pipeline_model_parallel_size = 1
        recipe.peft = run.Config(PEFT_STR2CLS[peft_scheme.lower()])
        recipe.peft.target_modules = ['linear_qkv', 'linear_proj']
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")
    return recipe
