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

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.llm.recipes.log.default import tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.utils.exp_manager import TimingCallback

NAME = "clip_b32"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Clip B32 model configuration.

    Returns:
        run.Co  nfig[pl.LightningModule]: Configuration for the Clip B32 model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=clip_b32 ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(vlm.CLIPModel, config=run.Config(vlm.CLIPConfigB32))


@run.cli.factory(target=llm.pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    """
    Create a fine-tuning recipe for Clip B32 model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory clip_b32

        Python API usage:
            >>> recipe = finetune_recipe(name="clip_b32", num_nodes=1)
            >>> print(recipe)

    Note:
        This recipe uses the Mock dataset for fine-tuning. For more information
        on fine-tuning LLMs with NeMo, see the fine-tuning guide in the
        `scripts/vlm/` directory.
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
        devices=num_gpus_per_node,
        limit_val_batches=10,
        log_every_n_steps=1,
        max_steps=5000,
        num_nodes=num_nodes,
        plugins=bf16_mixed(),
        strategy=strategy,
        val_check_interval=1000,
        callbacks=[
            run.Config(TimingCallback),
        ],
    )

    recipe = run.Partial(
        llm.pretrain,
        model=model(),
        trainer=trainer,
        data=run.Config(
            vlm.ClipMockDataModule,
            seq_length=80,
            global_batch_size=128,
            micro_batch_size=2,
            tokenizer=None,
            image_processor=None,
            num_workers=4,
        ),
        log=llm.default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(
            max_lr=1e-3,
            min_lr=1e-5,
            warmup_steps=2000,
            adam_beta1=0.9,
            adam_beta2=0.98,
        ),
        resume=run.Config(
            nl.AutoResume,
            resume_if_exists=False,
            resume_ignore_no_checkpoint=True,
        ),
    )

    return recipe
