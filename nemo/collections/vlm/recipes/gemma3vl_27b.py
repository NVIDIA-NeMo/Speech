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
import torch
from lightning.pytorch.callbacks.callback import Callback
from megatron.core.distributed import DistributedDataParallelConfig

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.llm.api import pretrain
from nemo.collections.llm.recipes.finetune_default import nemo_resume
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.collections.vlm.gemma3vl.data.mock import Gemma3VLMockDataModule
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.callbacks.moe_token_drop import MegatronTokenDropCallback
from nemo.utils.exp_manager import TimingCallback

NAME = "gemma3vl_27b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Gemma3 VL 27B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Gemma3 VL 27B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=gemma3vl_27b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(
        vlm.Gemma3VLModel,
        config=run.Config(
            vlm.Gemma3VLConfig27B,
            language_transformer_config=run.Config(llm.Gemma3Config27B),
            vision_transformer_config=run.Config(vlm.Gemma3VLVisionConfig),
            vision_projection_config=run.Config(
                vlm.Gemma3VLMultimodalProjectorConfig,
                input_size=1152,
                hidden_size=5376,
            ),
        ),
    )


def trainer(
    tensor_parallelism: int = 8,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = True,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for Gemma3 VL 27B model.

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
            $ nemo llm pretrain trainer=gemma3vl_27b ...

        Python API usage:
            >>> trainer_config = trainer(num_nodes=1, num_gpus_per_node=8)
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
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            data_parallel_sharding_strategy="optim_grads_params",
        ),
        fsdp=None,
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
    performance_mode: bool = False,
    fn: Callable = pretrain,
) -> run.Partial:
    """
    Create a pre-training recipe for Gemma3 VL 27B model.

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
            $ nemo llm pretrain --factory gemma3vl_27b ...
            $ nemo llm pretrain --factory "gemma3vl_27b(num_nodes=2, name='my_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="gemma3vl_27b_pretrain", num_nodes=2)
            >>> print(recipe)
    """
    recipe = run.Partial(
        fn,
        model=model(),
        trainer=trainer(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(Gemma3VLMockDataModule, seq_length=512, global_batch_size=512, micro_batch_size=1),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )

    if performance_mode:
        recipe = pretrain_performance_optimizations(recipe)

    return recipe


def pretrain_performance_optimizations(recipe: run.Partial) -> run.Partial:
    """
    Create a performance-optimized pre-training recipe for Gemma3 VL 27B model.

    This method enables performance optimizations that may not be suitable for all use cases.
    It builds upon the standard pre-training recipe and adds additional performance enhancements.

    Args:
        recipe (run.Partial): Base pre-train recipe to which performance optimizations will be added

    Returns:
        run.Partial: Partial configuration for performance-optimized pre-training.

    Note:
        Use this method with caution and only when you need maximum performance.
        It may not be suitable for all hardware configurations or use cases.
    """
    if not recipe.trainer.callbacks:
        recipe.trainer.callbacks = []

    garbage_collection_callback = run.Config(
        GarbageCollectionCallback,
        gc_interval_train=100,
        gc_interval_val=100,
    )
    mcomm_overlap_callback = run.Config(
        MegatronCommOverlapCallback,
        tp_comm_overlap=True,
    )
    token_drop_callback = run.Config(
        MegatronTokenDropCallback,
    )
    recipe.trainer.callbacks.extend(
        [
            garbage_collection_callback,
            mcomm_overlap_callback,
            token_drop_callback,
        ]
    )

    recipe.trainer.plugins.grad_reduce_in_fp32 = False

    return recipe


@run.cli.factory(target=llm.finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    resume_path: Optional[str] = 'google/gemma-3-27b-it',
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'none',
) -> run.Partial:
    """
    Create a fine-tuning recipe for Gemma3 VL 27B model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        resume_path (str): Path to the NeMo checkpoint
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Parameter efficient fine-tuning scheme to use.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory gemma3vl_27b

        Python API usage:
            >>> recipe = finetune_recipe(name="gemma3vl_27b_finetune", num_nodes=1)
            >>> print(recipe)
    """

    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        encoder_tensor_model_parallel_size=0,
        encoder_pipeline_model_parallel_size=0,
        context_parallel_size=1,
        sequence_parallel=True,
        ckpt_async_save=True,
        ckpt_parallel_save=True,
        ckpt_parallel_load=True,
        ckpt_parallel_save_optim=True,
        ckpt_load_strictness="log_all",
        gradient_as_bucket_view=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        devices=num_gpus_per_node,
        limit_val_batches=10,
        log_every_n_steps=1,
        max_steps=5190,
        num_nodes=num_nodes,
        plugins=bf16_mixed(),
        strategy=strategy,
        val_check_interval=1000,
        callbacks=[
            run.Config(TimingCallback),
        ],
    )

    recipe = run.Partial(
        llm.finetune,
        model=model(),
        trainer=trainer,
        data=run.Config(
            Gemma3VLMockDataModule,
            seq_length=512,
            global_batch_size=32,
            micro_batch_size=1,
            tokenizer=run.Config(AutoTokenizer, pretrained_model_name='google/gemma-3-27b-it'),
            num_workers=4,
        ),
        log=llm.default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=2.0e-05, min_lr=2.0e-07, warmup_steps=150),
        resume=nemo_resume(resume_path),
    )

    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.tensor_model_parallel_size = 8
        recipe.optim.config.lr = 2e-05
    elif peft_scheme.lower() == 'lora':
        recipe.trainer.strategy.tensor_model_parallel_size = 4
        recipe.peft = run.Config(
            vlm.LoRA,
            freeze_vision_model=False,
            target_modules=[
                "*.language_model.*.linear_qkv",
                "*.language_model.*.linear_q",
                "*.language_model.*.linear_kv",
                "*.language_model.*.linear_proj",
                "*.language_model.*.linear_fc1",
                "*.language_model.*.linear_fc2",
            ],
        )
        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")

    return recipe
