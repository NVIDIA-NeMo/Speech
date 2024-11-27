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
from lightning.pytorch.callbacks.callback import Callback
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.llm.recipes.finetune_default import default_finetune_trainer, nemo_resume
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.collections.llm.t5.data.mock import MockDataModule
from nemo.collections.llm.t5.data.squad import SquadDataModule
from nemo.collections.llm.t5.model.t5 import T5Config3B, T5Model
from nemo.lightning.pytorch.optim.lr_scheduler import WarmupAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback

NAME = "t5_3b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a T5 3B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the T5 3B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=t5_3b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(T5Model, config=run.Config(T5Config3B))


def trainer(
    tensor_parallelism: int = 2,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    num_nodes: int = 20,
    num_gpus_per_node: int = 8,
    max_steps: int = 1000000,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for T5 model.

    This function sets up the distributed training strategy and other training parameters.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_type (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps.
        callbacks (Optional[list[run.Config[Callback]]]): List of callback configurations.

    Returns:
        run.Config[nl.Trainer]: Configuration for the NeMo Lightning Trainer.

    Examples:
        CLI usage:
            $ nemo llm pretrain trainer=t5_3b ...

        Python API usage:
            >>> trainer_config = trainer(num_nodes=2, num_gpus_per_node=8)
            >>> print(trainer_config)

    Note:
        For more information on distributed training strategies, refer to the
        NeMo documentation on multi-GPU and multi-node training.
    """
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
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
        plugins=bf16_mixed(),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=2000,
    )

    return trainer


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None, name: str = "default", num_nodes: int = 20, num_gpus_per_node: int = 8, fn=pretrain
) -> run.Partial:
    """
    Create a pre-training recipe for T5 3b model.

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
            $ nemo llm pretrain --factory t5_3b
            $ nemo llm pretrain --factory "t5_3b(num_nodes=2, name='my_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="t5_3b_pretrain", num_nodes=2)
            >>> print(recipe)

    Note:
        For more details on pre-training LLMs with NeMo, see the pre-training
        guide in the `examples/llm/pretrain/` directory.
    """

    opt_config = run.Config(
        OptimizerConfig,
        optimizer='adam',
        lr=0.0001,
        use_distributed_optimizer=True,
        bf16=True,
        weight_decay=0.01,
    )

    lr_scheduler = run.Config(
        WarmupAnnealingScheduler,
        warmup_steps=None,
        warmup_ratio=0.01,
        max_steps=1000000,
        min_lr=0.00001,
    )

    return run.Partial(
        fn,
        model=model(),
        trainer=trainer(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(
            MockDataModule, seq_length=512, seq_length_dec=128, global_batch_size=1920, micro_batch_size=24
        ),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=run.Config(MegatronOptimizerModule, config=opt_config, lr_scheduler=lr_scheduler),
        resume=default_resume(),
    )


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    checkpoint_path: str = "",
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'lora',
) -> run.Partial:
    """
    Create a fine-tuning recipe for T5 3B model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        checkpoint_path (str): Path to pretrained checkpoint
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning.
            Allowed values: 'lora'/'dora'/'none'/None.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory t5_3b

        Python API usage:
            >>> recipe = finetune_recipe(name="t5_3b_finetune", num_nodes=1)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning. For more information
        on fine-tuning LLMs with NeMo, see the fine-tuning guide in the
        `examples/llm/finetune/` directory.
    """
    opt_config = run.Config(
        OptimizerConfig,
        optimizer='adam',
        lr=0.0001,
        use_distributed_optimizer=True,
        bf16=True,
        weight_decay=0.01,
    )

    lr_scheduler = run.Config(
        WarmupAnnealingScheduler,
        warmup_steps=50,
        max_steps=2000,
        min_lr=0.00001,
    )

    recipe = run.Partial(
        finetune,
        model=model(),
        trainer=default_finetune_trainer(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
        ),
        data=run.Config(
            SquadDataModule, seq_length=512, seq_length_dec=128, global_batch_size=128, micro_batch_size=1
        ),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=run.Config(MegatronOptimizerModule, config=opt_config, lr_scheduler=lr_scheduler),
        resume=nemo_resume(checkpoint_path),
    )

    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.tensor_model_parallel_size = 2
        recipe.optim.config.lr = 5e-6
    elif peft_scheme.lower() in ['lora', 'dora']:
        recipe.peft = run.Config(PEFT_STR2CLS[peft_scheme.lower()])
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")
    return recipe
