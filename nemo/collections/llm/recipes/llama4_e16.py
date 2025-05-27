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
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
from nemo.collections.llm.gpt.model.llama import Llama4Experts16Config, LlamaModel
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.callbacks.moe_token_drop import MegatronTokenDropCallback
from nemo.utils.exp_manager import TimingCallback

NAME = "llama4_e16"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Llama4 16-Experts (Scout) model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Llama4 16-Experts (Scout) model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=llama4_e16 ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(LlamaModel, config=run.Config(Llama4Experts16Config))


def trainer(
    tensor_parallelism: int = 4,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = True,
    expert_tensor_parallelism: int = 4,
    expert_model_parallelism: int = 16,
    num_nodes: int = 32,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for Llama4 16-Experts (Scout) model.

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
            $ nemo llm pretrain trainer=llama4_e16 ...

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
        expert_tensor_parallel_size=expert_tensor_parallelism,
        expert_model_parallel_size=expert_model_parallelism,
        gradient_as_bucket_view=True,
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
        fsdp=None,  # Set to 'megatron' to use Megatron FSDP, 'pytorch' to use PyTorch FSDP 2 (WIP)
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
    Create a pre-training recipe for Llama4 16-Experts (Scout) model.

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
            $ nemo llm pretrain --factory llama4_e16 ...
            $ nemo llm pretrain --factory "llama4_e16(num_nodes=2, name='my_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="llama4_e16_pretrain", num_nodes=2)
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
        data=run.Config(MockDataModule, seq_length=8192, global_batch_size=512, micro_batch_size=1),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )

    if performance_mode:
        recipe = pretrain_performance_optimizations(recipe)

    return recipe


def pretrain_performance_optimizations(recipe: run.Partial) -> run.Partial:
    """
    Create a performance-optimized pre-training recipe for Llama4 16-Experts (Scout) model.

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


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    resume_path: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'lora',
    seq_length: Optional[int] = None,
    packed_sequence: Optional[bool] = None,
    performance_mode: bool = False,
) -> run.Partial:
    """
    Create a fine-tuning recipe for Llama4 16-Experts (Scout) model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        resume_path (str): Path to the NeMo checkpoint
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning.
            Allowed values: 'lora'/'dora'/'none'/None.
        seq_length (int): Maximum number of tokens per microbatch.
        packed_sequence (Optional[bool]): If true, fine-tuning sequences will be packed into batches up to the given
            maximum seq_length for better efficiency. By default, this value equals performance_mode.
        performance_mode (bool): If true, enables optimizations for maximum performance.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory llama4_e16

        Python API usage:
            >>> recipe = finetune_recipe(name="llama4_e16_finetune", num_nodes=2)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning. For more information
        on fine-tuning LLMs with NeMo, see the fine-tuning guide in the
        `examples/llm/finetune/` directory.
    """
    # Default to unpacked data in normal mode and packed data in performance mode
    # once packing recipe is well tested, change this default to true
    if packed_sequence is None:
        packed_sequence = performance_mode

    # For unpacked sequence, most samples in SQuAD dataset are shorter than 2K
    if seq_length is None:
        seq_length = 4096 if packed_sequence else 2048

    recipe = default_finetune_recipe(model(), resume_path, dir, name, num_nodes, num_gpus_per_node, packed_sequence)
    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.sequence_parallel = True
        recipe.trainer.strategy.tensor_model_parallel_size = 8
        recipe.trainer.strategy.expert_tensor_model_parallel_size = 8
        recipe.trainer.strategy.pipeline_model_parallel_size = 4

        recipe.optim.config.lr = 5e-6
    elif peft_scheme.lower() in ['lora', 'dora']:
        recipe.trainer.strategy.sequence_parallel = True
        recipe.trainer.strategy.tensor_model_parallel_size = 8
        recipe.trainer.strategy.expert_tensor_model_parallel_size = 8
        recipe.peft = run.Config(PEFT_STR2CLS[peft_scheme.lower()])
        recipe.peft.dim = 8
        recipe.peft.alpha = 16
        recipe.optim.config.use_distributed_optimizer = False

        # some settings currently do not function correctly with LoRA
        recipe.model.config.cross_entropy_loss_fusion = False

        recipe.optim.config.lr = 1e-4
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")

    # Sequence length settings in the model and dataset must agree
    recipe.model.config.seq_length = seq_length
    recipe.data.seq_length = seq_length
    if packed_sequence:
        recipe.data.dataset_kwargs = {'pad_to_max_length': True}
        recipe.data.packed_sequence_specs = run.Config(PackedSequenceSpecs, packed_sequence_size=seq_length)

    if performance_mode:
        recipe = finetune_performance_optimizations(recipe, peft_scheme)

    return recipe


def finetune_performance_optimizations(
    recipe: run.Partial,
    peft_scheme: str,
) -> run.Partial:
    """
    Modify the given recipe to optimize settings for performance.

    This method enables performance optimizations that may not be suitable for all use cases.
    Intended to build upon the standard fine-tuning recipe.

    Args:
        recipe (run.Partial): Base fine-tuning recipe to which performance optimizations will be added
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning.
            Allowed values: 'lora'/'dora'/'none'/None.

    Returns:
        run.Partial: Partial configuration for performance-optimized fine-tuning.

    Note:
        Use this method with caution and only when you need maximum performance.
        It may not be suitable for all hardware configurations or use cases.
    """
    recipe.trainer.strategy.tensor_model_parallel_size = 1

    if not hasattr(recipe.trainer, "callbacks") or recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []

    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.ddp = run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        )
    else:
        recipe.peft.target_modules = ['linear_qkv']

    recipe.trainer.plugins.grad_reduce_in_fp32 = False

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

    return recipe
