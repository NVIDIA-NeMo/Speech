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

from typing import Callable, Optional

import lightning.pytorch as pl
import nemo_run as run
import torch
from lightning.pytorch.callbacks.callback import Callback
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections.diffusion.data.diffusion_mock_datamodule import MockDataModule
from nemo.collections.diffusion.models.flux.model import FluxModelParams, MegatronFluxModel
from nemo.collections.llm.api import pretrain
from nemo.collections.llm.recipes.log.default import default_log, tensorboard_logger
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.utils.exp_manager import TimingCallback

NAME = "flux_12b"


@run.cli.factory
@run.autoconvert
def flux_mock_datamodule() -> pl.LightningDataModule:
    """Mock Datamodule Initialization"""
    data_module = MockDataModule(
        image_h=1024,
        image_w=1024,
        micro_batch_size=1,
        global_batch_size=2,
        image_precached=True,
        text_precached=True,
    )
    return data_module


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a FLUX 12B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the FLUX 12B model.

    """
    return run.Config(MegatronFluxModel, flux_params=run.Config(FluxModelParams))


def trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for FLUX 12B model.

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
            $ nemo llm pretrain trainer=flux_12b ...

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
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        gradient_accumulation_fusion=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            use_custom_fsdp=True,
            data_parallel_sharding_strategy='optim_grads_params',
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
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    fn: Callable = pretrain,
) -> run.Partial:
    """
    Create a pre-training recipe for FLUX 12B model.

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
            $ nemo llm pretrain --factory flux_12b

        Python API usage:
            >>> recipe = pretrain_recipe(name="flux_12b_pretrain", num_nodes=1)
            >>> print(recipe)

    Note:
        For more details on pre-training LLMs with NeMo, see the pre-training
        guide in the `examples/llm/pretrain/` directory.
    """
    recipe = run.Partial(
        fn,
        model=model(),
        trainer=trainer(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=flux_mock_datamodule(),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                lr=1e-4,
                bf16=True,
                use_distributed_optimizer=True,
                weight_decay=0,
            ),
        ),
        resume=run.Config(
            nl.AutoResume,
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=True,
        ),
        model_transform=None,
    )

    return recipe
