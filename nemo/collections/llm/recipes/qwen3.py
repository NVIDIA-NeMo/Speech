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
from lightning.pytorch.callbacks.callback import Callback
from megatron.core.distributed import DistributedDataParallelConfig

from nemo import lightning as nl
from nemo.collections.llm.gpt.model.qwen3 import (
    Qwen3Config1P7B,
    Qwen3Config4B,
    Qwen3Config8B,
    Qwen3Config14B,
    Qwen3Config30B_A3B,
    Qwen3Config32B,
    Qwen3Config235B_A22B,
    Qwen3Config600M,
    Qwen3Model,
)
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed, fp16_mixed


def qwen3_model(version: str) -> run.Config[pl.LightningModule]:
    """
    A function to create a qwen3 models.

    Args:
        version (str): The version of the qwen3 model to create. One of ["qwen3_600m", "qwen3_1p7b",
            "qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen3_32b", "qwen3_30b_a3b", "qwen3_235b_a22b"].

    Returns:
        run.Config[pl.LightningModule]: Configuration for the qwen3 model.
    """
    config = None
    if version == "qwen3_600m":
        config = run.Config(Qwen3Config600M)
    elif version == "qwen3_1p7b":
        config = run.Config(Qwen3Config1P7B)
    elif version == "qwen3_4b":
        config = run.Config(Qwen3Config4B)
    elif version == "qwen3_8b":
        config = run.Config(Qwen3Config8B)
    elif version == "qwen3_14b":
        config = run.Config(Qwen3Config14B)
    elif version == "qwen3_32b":
        config = run.Config(Qwen3Config32B)
    elif version == "qwen3_30b_a3b":
        config = run.Config(Qwen3Config30B_A3B)
    elif version == "qwen3_235b_a22b":
        config = run.Config(Qwen3Config235B_A22B)

    assert config is not None, f"Invalid version: {version}"
    return run.Config(Qwen3Model, config=config)


def qwen3_trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    expert_parallelism: int = 1,
    account_for_embedding_in_pipeline_split: bool = False,
    account_for_loss_in_pipeline_split: bool = False,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    precision: str = "bf16-mixed",
    accumulate_grad_batches: int = 1,
    limit_test_batches: int = 32,
    limit_val_batches: int = 32,
    log_every_n_steps: int = 10,
    val_check_interval: int = 2000,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for qwen3 models.

    This function sets up the distributed training strategy and other training parameters.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_type (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        expert_parallelism (Optional[int]): Degree of expert parallelism.
        account_for_embedding_in_pipeline_split (bool): Whether to treat input embedding layer as a standard
            transformer layer in the context of partition and placement for pipeline parallelism.
        account_for_loss_in_pipeline_split (bool): Whether to treat loss layer as a standard transformer
            layer in the context of partition and placement for pipeline parallelism.
        account_for_loss_in_pipeline_split (bool): = False,
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps.
        precision (str): Precision configuration, one of fp32, 16-mixed or bf16-mixed.
        accumulate_grad_batches (int): Number of steps per gradient accumulation.
        limit_test_batches (int): Limit the number of test batches.
        limit_val_batches (int): Limit the number of validation batches.
        log_every_n_steps (int): Log every n steps.
        val_check_interval (int): Run validation every N steps.
        callbacks (Optional[list[run.Config[Callback]]]): List of callback configurations.

    Returns:
        run.Config[nl.Trainer]: Configuration for the NeMo Lightning Trainer.
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
        expert_tensor_parallel_size=1,
        account_for_embedding_in_pipeline_split=account_for_embedding_in_pipeline_split,
        account_for_loss_in_pipeline_split=account_for_loss_in_pipeline_split,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,  # Not supported for custom FSDP for now, need to be set to False if using FSDP
            data_parallel_sharding_strategy="optim_grads_params",  # For custom FSDP only
        ),
    )

    precision_plugin = None
    if precision == "16-mixed":
        precision_plugin = fp16_mixed()
    elif precision == "bf16-mixed":
        precision_plugin = bf16_mixed()

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        callbacks=callbacks,
        devices=num_gpus_per_node,
        accumulate_grad_batches=accumulate_grad_batches,
        limit_test_batches=limit_test_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=log_every_n_steps,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=precision_plugin,
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=val_check_interval,
    )

    return trainer
