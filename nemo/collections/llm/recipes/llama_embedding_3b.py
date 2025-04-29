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

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm import Llama32EmbeddingConfig3B, LlamaEmbeddingModel
from nemo.collections.llm.api import finetune
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.utils.exp_manager import TimingCallback

NAME = "nvembed_llama_3b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a NVEmbed Llama3.2 3b model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the NVEmbed Llama3.2 3B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=nvembed_llama_3b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(LlamaEmbeddingModel, config=run.Config(Llama32EmbeddingConfig3B))


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
    Configure the NeMo Lightning Trainer for NVEmbed Llama3.2 3B model.

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
            $ nemo llm pretrain trainer=nvembed_llama_3b ...

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


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    resume_path: str = "meta-llama/Llama-3.2-3B",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    micro_batch_size: int = 4,
    global_batch_size: int = 64,
    peft_scheme: Optional[str] = 'lora',
    seq_length: Optional[int] = None,
    packed_sequence: Optional[bool] = None,
) -> run.Partial:
    """
    Create a fine-tuning recipe for NVEmbed Llama3.2 3B model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        resume_path (str): Path to the Huggingface model or pretrained distributed checkpoint for resume
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        micro_batch_size (int): Size of micro batch.
        global_batch_size (int): Size of global batch.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning.
            Allowed values: 'lora'/'dora'/'none'/None.
        seq_length (int): Maximum number of tokens per microbatch.
        packed_sequence (Optional[bool]): If true, fine-tuning sequences will be packed into batches up to the given
            maximum seq_length for better efficiency. pack sequence is not supported for embedding model training.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory nvembed_llama_3b

        Python API usage:
            >>> recipe = finetune_recipe(name="nvembed_llama_3b_finetune", num_nodes=2)
            >>> print(recipe)

    Note:
        This recipe uses the SPECTER dataset for fine-tuning. For more information
        on fine-tuning LLMs with NeMo, see the fine-tuning guide in the
        `examples/llm/finetune/` directory.
    """
    if seq_length is None:
        seq_length = 512

    assert packed_sequence is None, 'pack_sequence is not supported for Embedding model finetuning.'
    recipe = default_finetune_recipe(model(), resume_path, dir, name, num_nodes, num_gpus_per_node, packed_sequence)
    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.strategy.tensor_model_parallel_size = 1
        recipe.optim.config.lr = 5e-6
    elif peft_scheme.lower() in ['lora', 'dora']:
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
    # Use Specter Dataset as the default for finetuning
    recipe.data = run.Config(
        llm.SpecterDataModule,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        dataset_kwargs={
            'num_hard_negatives': recipe.model.config.num_hard_negatives,
            'negative_sample_strategy': recipe.model.config.negative_sample_strategy,
            'add_bos': recipe.model.config.add_bos,
            'add_eos': recipe.model.config.add_eos,
        },
    )

    return recipe


def finetune_performance_optimizations(
    recipe: run.Partial,
    peft_scheme: Optional[str],
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

    if not hasattr(recipe.trainer, "callbacks"):
        recipe.trainer.callbacks = []

    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.trainer.plugins.grad_reduce_in_fp32 = False
        recipe.trainer.strategy.ddp = run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        )
        recipe.trainer.callbacks.append(
            run.Config(
                MegatronCommOverlapCallback,
                tp_comm_overlap=False,
            )
        )
    else:
        recipe.peft.target_modules = ['linear_qkv']

    recipe.trainer.callbacks.append(run.Config(TimingCallback))
    recipe.trainer.callbacks.append(
        run.Config(
            GarbageCollectionCallback,
            100,
            100,
        )
    )

    return recipe
