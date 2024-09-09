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
from nemo.collections.llm.gpt.model.mixtral import MixtralConfig8x7B, MixtralModel
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.utils.exp_manager import TimingCallback

NAME = "mixtral_8x7b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Mixtral 8x7B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Mixtral 8x7B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=mixtral_8x7b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(MixtralModel, config=run.Config(MixtralConfig8x7B))


def trainer(
    tensor_parallelism: int = 8,
    pipeline_parallelism: int = 2,
    pipeline_parallelism_type: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = True,
    expert_parallelism: int = 1,
    num_nodes: int = 2,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for Mixtral 8x7B model.

    This function sets up the distributed training strategy optimized for the Mixtral 8x7B model.

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
            $ nemo llm pretrain trainer=mixtral_8x7b ...

        Python API usage:
            >>> trainer_config = trainer(num_nodes=2, num_gpus_per_node=8)
            >>> print(trainer_config)
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
    dir: Optional[str] = None, name: str = "default", num_nodes: int = 2, num_gpus_per_node: int = 8, fn=pretrain
) -> run.Partial:
    """
    Create a pre-training recipe for Mixtral 8x7B model.

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
            $ nemo llm pretrain --factory mixtral_8x7b
            $ nemo llm pretrain --factory "mixtral_8x7b(num_nodes=2, name='my_mixtral_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="mixtral_8x7b_pretrain", num_nodes=2)
            >>> print(recipe)
    """
    return run.Partial(
        fn,
        model=model(),
        trainer=trainer(
            num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, callbacks=[run.Config(TimingCallback)]
        ),
        data=run.Config(MockDataModule, seq_length=8192, global_batch_size=512, micro_batch_size=1),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )


def hf_resume() -> run.Config[nl.AutoResume]:
    """
    Configure automatic resumption from a Hugging Face checkpoint for Mixtral 8x7B model.

    This function sets up the configuration to resume training from a pre-trained
    Hugging Face model checkpoint.

    More info about the model can be found at: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1

    Returns:
        run.Config[nl.AutoResume]: Configuration for resuming from HuggingFace checkpoint.

    Note:
        This is particularly useful for fine-tuning scenarios where you want to
        start from the pre-trained Mixtral 8x7B model.
    """
    return run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig, path="hf://mistralai/Mixtral-8x7B-v0.1"),
    )


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 2,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    """
    Create a fine-tuning recipe for Mixtral 8x7B model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    It uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory mixtral_8x7b
            $ nemo llm finetune --factory "mixtral_8x7b(num_nodes=2, name='my_mixtral_finetune')"

        Python API usage:
            >>> recipe = finetune_recipe(name="mixtral_8x7b_finetune", num_nodes=2)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning.
    """
    recipe = pretrain_recipe(name=name, dir=dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, fn=finetune)
    recipe.resume = hf_resume()
    recipe.peft = run.Config(LoRA, target_modules=['linear_qkv', 'linear_proj'], dim=32)
    recipe.data = run.Config(SquadDataModule, seq_length=8192, global_batch_size=512, micro_batch_size=1)
    return recipe
