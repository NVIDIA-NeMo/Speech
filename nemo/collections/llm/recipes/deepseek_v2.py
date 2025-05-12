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


from typing import Callable, Optional

import lightning.pytorch as pl
import nemo_run as run

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
from nemo.collections.llm.gpt.model.deepseek import DeepSeekModel, DeepSeekV2Config
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.llm.recipes.deepseek import trainer
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.utils.exp_manager import TimingCallback

NAME = "deepseek_v2"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a DeepSeek-V2 (236B) model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the DeepSeek V2 model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=deepseek_v2 ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    conf = run.Config(DeepSeekV2Config)
    return run.Config(DeepSeekModel, config=conf)


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 16,
    num_gpus_per_node: int = 8,
    fn: Callable = pretrain,
) -> run.Partial:
    """
    Create a pre-training recipe for DeepSeek-V2 (236B) model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        performance_mode (bool): If true, enables optimizations for maximum performance.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for pre-training.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory deepseek_v2
            $ nemo llm pretrain --factory "deepseek_v2(num_nodes=16, name='my_deepseek_v2')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="deepseek_v2_pretrain", num_nodes=16)
            >>> print(recipe)

    """
    recipe = run.Partial(
        fn,
        model=model(),
        trainer=trainer(
            tensor_parallelism=1,
            pipeline_parallelism=4,
            expert_parallelism=32,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(MockDataModule, seq_length=4096, global_batch_size=512, micro_batch_size=1),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )

    recipe.model.config.recompute_granularity = "full"
    recipe.model.config.recompute_method = "uniform"
    recipe.model.config.recompute_num_layers = 1

    # Use DeepEP
    recipe.model.config.moe_token_dispatcher_type = "flex"
    recipe.model.config.moe_enable_deepep = True
    recipe.model.config.moe_shared_expert_overlap = False

    return recipe


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    resume_path: str = "deepseek-ai/DeepSeek-V2",
    name: str = "default",
    num_nodes: int = 2,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'lora',
    seq_length: Optional[int] = None,
    packed_sequence: Optional[bool] = None,
) -> run.Partial:
    """
    Create a fine-tuning recipe for DeepSeek-V2 (236B) model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        resume_path (str): Path to the NeMo checkpoint
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning.
            Allowed values: 'lora'/'dora'/'none'/None.
        seq_length (int): Maximum number of tokens per microbatch.
        packed_sequence (Optional[bool]): If true, fine-tuning sequences will be packed into batches up to the given
            maximum seq_length for better efficiency. By default, this value equals performance_mode.
    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory deepseek_v2
            $ nemo llm finetune --factory "deepseek_v2(num_nodes=2, name='my_deepseek_v2_finetune')"

        Python API usage:
            >>> recipe = finetune_recipe(name="deepseek_v2_finetune", num_nodes=2)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning. Be aware that fine-tuning the DeepSeek-V2 model
        requires substantial computational resources.
    """

    if seq_length is None:
        seq_length = 2048

    if num_nodes is None:
        if peft_scheme is None or peft_scheme.lower() == 'none':
            num_nodes = 16
        elif peft_scheme.lower() in ['lora', 'dora']:
            num_nodes = 2

    recipe = default_finetune_recipe(model(), resume_path, dir, name, num_nodes, num_gpus_per_node, packed_sequence)
    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.pipeline_model_parallel_size = 4
        recipe.trainer.strategy.expert_model_parallel_size = 32
        recipe.optim.config.lr = 5e-6
    elif peft_scheme.lower() in ['lora', 'dora']:
        recipe.peft = run.Config(PEFT_STR2CLS[peft_scheme.lower()])
        recipe.peft.target_modules = [
            'linear_q_down_proj',
            'linear_q_up_proj',
            'linear_kv_down_proj',
            'linear_kv_up_proj',
            'linear_proj',
        ]
        recipe.optim.config.use_distributed_optimizer = False
        recipe.model.config.cross_entropy_loss_fusion = False
        recipe.trainer.strategy.tensor_model_parallel_size = 1
        recipe.trainer.strategy.pipeline_model_parallel_size = 1
        recipe.trainer.strategy.expert_model_parallel_size = 16
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")

    # Sequence length settings in the model and dataset must agree
    recipe.model.config.seq_length = seq_length
    recipe.data.seq_length = seq_length
    if packed_sequence:
        recipe.data.dataset_kwargs = {'pad_to_max_length': True}
        recipe.data.packed_sequence_specs = run.Config(PackedSequenceSpecs, packed_sequence_size=seq_length)

    return recipe
