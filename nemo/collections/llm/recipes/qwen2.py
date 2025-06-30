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

from nemo import lightning as nl
from nemo.collections.llm.gpt.model.qwen2 import (
    Qwen2Config1P5B,
    Qwen2Config7B,
    Qwen2Config72B,
    Qwen2Config500M,
    Qwen2Model,
    Qwen25Config1P5B,
    Qwen25Config3B,
    Qwen25Config7B,
    Qwen25Config14B,
    Qwen25Config32B,
    Qwen25Config72B,
    Qwen25Config500M,
)
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed, fp16_mixed


def qwen2_model(version: str) -> run.Config[pl.LightningModule]:
    """
    A function to create a qwen2 models.

    Args:
        version (str): The version of the qwen2 model to create. one of ["qwen2_500m", "qwen2_1p5b",
            "qwen2_7b", "qwen2_72b"].

    Returns:
        run.Config[pl.LightningModule]: Configuration for the qwen2 model.
    """
    config = None
    if version == "qwen2_500m":
        config = run.Config(Qwen2Config500M)
    elif version == "qwen25_500m":
        config = run.Config(Qwen25Config500M)
    elif version == "qwen2_1p5b":
        config = run.Config(Qwen2Config1P5B)
    elif version == "qwen25_1p5b":
        config = run.Config(Qwen25Config1P5B)
    elif version == "qwen25_3b":
        config = run.Config(Qwen25Config3B)
    elif version == "qwen2_7b":
        config = run.Config(Qwen2Config7B)
    elif version == "qwen25_7b":
        config = run.Config(Qwen25Config7B)
    elif version == "qwen25_14b":
        config = run.Config(Qwen25Config14B)
    elif version == "qwen25_32b":
        config = run.Config(Qwen25Config32B)
    elif version == "qwen2_72b":
        config = run.Config(Qwen2Config72B)
    elif version == "qwen25_72b":
        config = run.Config(Qwen25Config72B)

    assert config is not None, f"Invalid version: {version}"
    return run.Config(Qwen2Model, config=config)


def qwen2_trainer(
    tensor_parallelism: int = 2,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
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
    Configure the NeMo Lightning Trainer for qwen2 models.

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
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
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
