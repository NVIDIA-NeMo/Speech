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

import nemo_run as run
import pytorch_lightning as pl
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from pytorch_lightning.callbacks.callback import Callback

from nemo import lightning as nl
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model.mixtral import MixtralConfig8x22B, MixtralModel
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.callbacks.moe_token_drop import MegatronTokenDropCallback
from nemo.utils.exp_manager import TimingCallback

NAME = "mixtral_8x22b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Mixtral 8x22B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Mixtral 8x22B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=mixtral_8x22b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(MixtralModel, config=run.Config(MixtralConfig8x22B))


def trainer(
    tensor_parallelism: int = 2,
    pipeline_parallelism: int = 4,
    pipeline_parallelism_type: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = 14,
    context_parallelism: int = 2,
    sequence_parallelism: bool = True,
    expert_parallelism: int = 8,
    num_nodes: int = 16,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for Mixtral 8x22B model.

    This function sets up the distributed training strategy optimized for the large Mixtral 8x22B model.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_type (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        expert_parallelism (int): Degree of expert parallelism.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps.
        callbacks (Optional[list[run.Config[Callback]]]): List of callback configurations.

    Returns:
        run.Config[nl.Trainer]: Configuration for the NeMo Lightning Trainer.

    Examples:
        CLI usage:
            $ nemo llm pretrain trainer=mixtral_8x22b ...

        Python API usage:
            >>> trainer_config = trainer(num_nodes=16, num_gpus_per_node=8)
            >>> print(trainer_config)

    Note:
        This configuration uses extensive parallelism to handle the large model size efficiently.
    """
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        expert_model_parallel_size=expert_parallelism,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
        ),
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        callbacks=callbacks,
        devices=num_gpus_per_node,
        limit_test_batches=50,
        limit_val_batches=32,
        log_every_n_steps=10,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=run.Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=2000,
    )

    return trainer


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None, name: str = "default", num_nodes: int = 16, num_gpus_per_node: int = 8, fn=pretrain
) -> run.Partial:
    """
    Create a pre-training recipe for Mixtral 8x22B model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for pre-training.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory mixtral_8x22b
            $ nemo llm pretrain --factory "mixtral_8x22b(num_nodes=16, name='my_mixtral_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="mixtral_pretrain", num_nodes=16)
            >>> print(recipe)
    """
    return run.Partial(
        fn,
        model=model(),
        trainer=trainer(
            num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, callbacks=[run.Config(TimingCallback)]
        ),
        data=run.Config(MockDataModule, seq_length=4096, global_batch_size=512, micro_batch_size=1),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )


@run.cli.factory(target=pretrain, name=NAME + "_performance")
def pretrain_recipe_performance(
    dir: Optional[str] = None, name: str = "default", num_nodes: int = 8, num_gpus_per_node: int = 8, fn=pretrain
) -> run.Partial:
    """
    Create a performance-optimized pre-training recipe for Mixtral 8x22B model.

    This recipe enables performance optimizations that may not be suitable for all use cases.
    It builds upon the standard pre-training recipe and adds additional performance enhancements.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for performance-optimized pre-training.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory "mixtral_8x22b.pretrain_recipe_performance(num_nodes=8, name='perf_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe_performance(name="mixtral_8x22b_perf", num_nodes=8)
            >>> print(recipe)

    Note:
        Use this recipe with caution and only when you need maximum performance.
        It may not be suitable for all hardware configurations or use cases.
    """
    recipe = pretrain_recipe(name=name, dir=dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, fn=fn)
    recipe.trainer.callbacks.extend(
        [
            run.Config(MegatronTokenDropCallback),
            run.Config(MegatronCommOverlapCallback),
        ]
    )

    return recipe


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 8,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'lora',
) -> run.Partial:
    """
    Create a fine-tuning recipe for Mixtral 8x22B model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning. Allowed values: 'lora', 'none'/None.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory mixtral_8x22b
            $ nemo llm finetune --factory "mixtral_8x22b(num_nodes=2, name='my_mixtral_finetune')"

        Python API usage:
            >>> recipe = finetune_recipe(name="mixtral_finetune", num_nodes=2)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning.
    """
    recipe = default_finetune_recipe(
        model(), "mistralai/Mixtral-8x22B-v0.1mistralai/Mixtral-8x22B-v0.1", dir, name, num_nodes, num_gpus_per_node
    )
    recipe.trainer.strategy.expert_model_parallel_size = 8
    recipe.trainer.strategy.tensor_model_parallel_size = 8
    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.pipeline_model_parallel_size = 4
        recipe.trainer.strategy.virtual_pipeline_model_parallel_size = 14
        recipe.optim.config.lr = 5e-6
    elif peft_scheme.lower() == 'lora':
        recipe.peft = run.Config(LoRA, target_modules=['linear_qkv', 'linear_proj'], dim=32)
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")
    return recipe
