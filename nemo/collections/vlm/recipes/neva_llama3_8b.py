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

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import lightning.pytorch as pl
import nemo_run as run
import torch
from lightning.pytorch.callbacks.callback import Callback
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo import lightning as nl
from nemo.collections import vlm
from nemo.collections.llm.api import pretrain
from nemo.collections.llm.gpt.model.llama import Llama3Config8B
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.collections.vlm.neva.data.mock import MockDataModule
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.utils.exp_manager import TimingCallback

NAME = "neva_llama3_8b"


@dataclass
class NevaConfig8B(vlm.NevaConfig):
    """NeVA (CLIP-ViT-L + LLaMa38B) Config"""

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Llama3Config8B(seq_length=8192))
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: vlm.HFCLIPVisionConfig(
            pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"
        )
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: vlm.MultimodalProjectorConfig(input_size=1024, hidden_size=4096, ffn_hidden_size=4096)
    )

    freeze_language_model: bool = False
    freeze_vision_model: bool = True
    freeze_vision_projection: bool = False


@run.cli.factory(name=NAME)
def model(
    seq_length: Optional[int] = 8192,
) -> run.Config[pl.LightningModule]:
    """
    Factory function to create a NeVA (CLIP-ViT-L + LLaMa3 8B) model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the NeVA (CLIP-ViT-L + LLaMa3 8B) model.
    """
    model_config = run.Config(vlm.NevaModel, config=run.Config(NevaConfig8B))
    return model_config


def trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 2,
    sequence_parallelism: bool = False,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for NeVA (CLIP-ViT-L + LLaMa3 8B) model.

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
            $ nemo llm finetune trainer=neva_llama3_8b ...

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
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=False,
            overlap_param_gather=True,
            average_in_collective=True,
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
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    seq_length: int = 8192,
    performance_mode: bool = False,
    packed_sequence: bool = False,
    fn: Callable = pretrain,
) -> run.Partial:
    """
    Create a finetuning recipe for NeVA (CLIP-ViT-L + LLaMa3 8B) model.
    Using pre-train fn targets to use appropriate llm api call.

    This function sets up a complete configuration for finetuning, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the finetuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        performance_mode (bool): If true, enables optimizations for maximum performance.
        fn (Callable): The finetuning function to use.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory neva_llama3_8b

        Python API usage:
            >>> recipe = finetune_recipe(name="neva_llama3_8b_finetune", num_nodes=2)
            >>> print(recipe)

    """
    recipe = run.Partial(
        fn,
        model=model(seq_length=seq_length),
        trainer=trainer(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(
            MockDataModule,
            seq_length=seq_length,
            global_batch_size=512,
            micro_batch_size=1,
            num_workers=4,
            packed_sequence=packed_sequence,
        ),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )

    if performance_mode:
        recipe = finetune_performance_optimizations(recipe)

    return recipe


def finetune_performance_optimizations(recipe: run.Partial) -> run.Partial:
    """
    Create a performance-optimized finetuning recipe for Llama3 8B model.

    This method enables performance optimizations that may not be suitable for all use cases.
    It builds upon the standard finetuning recipe and adds additional performance enhancements.

    Args:
        recipe (run.Partial): Base finetune recipe to which performance optimizations will be added

    Returns:
        run.Partial: Partial configuration for performance-optimized finetuning.

    Note:
        Use this method with caution and only when you need maximum performance.
        It may not be suitable for all hardware configurations or use cases.
    """
    if not hasattr(recipe.trainer, "callbacks") or recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []

    if recipe.trainer.strategy.tensor_model_parallel_size > 1:
        recipe.trainer.callbacks.append(
            run.Config(
                MegatronCommOverlapCallback,
                tp_comm_overlap=True,
            )
        )
    else:
        recipe.trainer.callbacks.append(
            run.Config(
                MegatronCommOverlapCallback,
                tp_comm_overlap=False,
            )
        )

    recipe.trainer.callbacks.append(run.Config(TimingCallback))
    recipe.trainer.callbacks.append(
        run.Config(
            GarbageCollectionCallback,
            100,
            100,
        )
    )

    recipe.trainer.plugins.grad_reduce_in_fp32 = False

    return recipe
