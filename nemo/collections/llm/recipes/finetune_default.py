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

import nemo.lightning as nl
from nemo.collections import llm
from nemo.collections.llm.recipes.log.default import tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed


def default_finetune_recipe(
    model: run.Config[pl.LightningModule],
    resume_path: str,
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    """
    Create a default fine-tuning recipe for any model.

    This function sets up a template for a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        model (run.Config[pl.LightningModule]): Configuration for a NeMo model.
        resume_path (str): Path to the Huggingface model.
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    See usages of this recipe for further details.
    """
    recipe = run.Partial(
        llm.finetune,
        model=model,
        trainer=default_finetune_trainer(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
        ),
        data=run.Config(llm.SquadDataModule, seq_length=2048, global_batch_size=128, micro_batch_size=1),
        log=llm.default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=1e-4, min_lr=0, warmup_steps=50),
        resume=nemo_resume(resume_path),
    )

    return recipe


def default_finetune_trainer(
    tensor_parallelism=1,
    pipeline_parallelism=1,
    pipeline_parallelism_type=None,
    virtual_pipeline_parallelism=None,
    context_parallelism=1,
    sequence_parallelism=False,
    num_nodes=1,
    num_gpus_per_node=8,
    max_steps=1000,
    limit_test_batches=None,
    limit_val_batches=None,
    val_check_interval=5,
):
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        gradient_as_bucket_view=True,
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        devices=num_gpus_per_node,
        limit_test_batches=limit_test_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=10,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=bf16_mixed(),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=val_check_interval,
    )

    return trainer


def nemo_resume(model_id: str) -> run.Config[nl.AutoResume]:
    """
    Configure automatic resumption from a NeMo checkpoint converted from Huggingface for https://huggingface.co/{model_id}.

    This NeMo checkpoint should be converted from Huggingface beforehand, using nemo.collections.llm.import_ckpt.
    When converting the checkpoint, the NeMo checkpoint will be saved in NEMO_HOME (set to ~/.cache/nemo by default).

    This function sets up the configuration to resume training from path nemo://{model_id}.
    This translates to the full path {NEMO_HOME}/models/{model_id}.

    Args:
        model_id (str): The Huggingface model to resume.

    Returns:
        run.Config[nl.AutoResume]: Configuration for resuming from NeMo checkpoint.
    """
    return run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig, path=f"nemo://{model_id}"),
    )
