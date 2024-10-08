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
import torch

from nemo import lightning as nl
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.nemotron import nemotron_model, nemotron_trainer
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.utils.exp_manager import TimingCallback

NAME = "nemotron4_340b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Nemotron4 340B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Nemotron4 340B model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=nemotron4_340 ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """

    return nemotron_model(version=NAME)


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    # General
    dir: Optional[str] = None,
    name: str = "default",
    # Trainer
    tensor_parallelism: int = 8,
    pipeline_parallelism: int = 12,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = 8,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    num_nodes: int = 768,
    num_gpus_per_node: int = 8,
    max_steps: int = 100000,
    precision: str = "bf16-mixed",
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float = 1.0,
    limit_test_batches: int = 32,
    limit_val_batches: int = 32,
    log_every_n_steps: int = 10,
    val_check_interval: int = 2000,
    # Data
    global_batch_size=2304,
    micro_batch_size=1,
    seq_length=4096,
    # Optimizer
    warmup_steps=500,
    constant_steps=0,
    min_lr=1.0e-5,
    max_lr=1.0e-4,
    # Training function
    fn=pretrain,
) -> run.Partial:
    """
    Create a pre-training recipe for Nemotron4 340B model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
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
        gradient_clip_val (float): Value for gradient clipping.
        limit_test_batches (int): Limit the number of test batches.
        limit_val_batches (int): Limit the number of validation batches.
        log_every_n_steps (int): Log every n steps.
        val_check_interval (int): Run validation every N steps.
        global_batch_size (int): Global batch size.
        micro_batch_size (int): Micro batch size.
        seq_length (int): Sequence length.
        warmup_steps (int): Number of warmup steps.
        constant_steps (int): Number of constant steps.
        min_lr (float): Minimum learning rate.
        max_lr (float): Maximum learning rate.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for pre-training.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory nemotron4_340
            $ nemo llm pretrain --factory "nemotron4_340(num_nodes=1, name='my_nemotron_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="nemotron_pretrain", num_nodes=1)
            >>> print(recipe)

    Note:
        This recipe uses a mock dataset, look for the finetune examples to see how to change the dataset.
    """
    return run.Partial(
        fn,
        model=model(),
        trainer=nemotron_trainer(
            tensor_parallelism=tensor_parallelism,
            pipeline_parallelism=pipeline_parallelism,
            pipeline_parallelism_type=pipeline_parallelism_type,
            virtual_pipeline_parallelism=virtual_pipeline_parallelism,
            context_parallelism=context_parallelism,
            sequence_parallelism=sequence_parallelism,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            max_steps=max_steps,
            precision=precision,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_test_batches=limit_test_batches,
            limit_val_batches=limit_val_batches,
            log_every_n_steps=log_every_n_steps,
            val_check_interval=val_check_interval,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(
            MockDataModule,
            seq_length=seq_length,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(
            precision=precision,
            warmup_steps=warmup_steps,
            constant_steps=constant_steps,
            min_lr=min_lr,
            max_lr=max_lr,
            clip_grad=gradient_clip_val,
        ),
        resume=default_resume(),
    )


@run.cli.factory(name=NAME + "_nemo")
def nemo_resume() -> run.Config[nl.AutoResume]:
    """
    Configure automatic resumption from a NeMo checkpoint converted from Huggingface for Nemotron4 340B model.

    More info about the Huggingface model can be found at: https://huggingface.co/nvidia/Nemotron-4-340B-Base.

    This NeMo checkpoint should be converted from Huggingface beforehand, using nemo.collections.llm.import_ckpt.
    When converting the checkpoint, the NeMo checkpoint will be saved in NEMO_HOME (set to ~/.cache/nemo by default).

    This function sets up the configuration to resume training from path nemo://nvidia/Nemotron-4-340B-Base.
    This translates to the full path {NEMO_HOME}/models/nvidia/Nemotron-4-340B-Base.

    Returns:
        run.Config[nl.AutoResume]: Configuration for resuming from NeMo checkpoint.

    Note:
        This is particularly useful for fine-tuning scenarios where you want to
        start from the pre-trained Nemotron4 340B model.
    """
    return run.Config(
        nl.AutoResume, restore_config=run.Config(nl.RestoreConfig, path="nemo://nvidia/Nemotron-4-340B-Base")
    )


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    # General
    dir: Optional[str] = None,
    name: str = "default",
    # Trainer
    tensor_parallelism: int = 8,
    pipeline_parallelism: int = 12,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = 8,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    num_nodes: int = 768,
    num_gpus_per_node: int = 8,
    max_steps: int = 100000,
    precision: str = "bf16-mixed",
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float = 1.0,
    limit_test_batches: int = 32,
    limit_val_batches: int = 32,
    log_every_n_steps: int = 10,
    val_check_interval: int = 2000,
    # Data
    global_batch_size=2304,
    micro_batch_size=1,
    seq_length=4096,
    # Optimizer
    warmup_steps=500,
    constant_steps=0,
    min_lr=1.0e-5,
    max_lr=1.0e-4,
    # Training function
    fn=finetune,
) -> run.Partial:
    """
    Create a fine-tuning recipe for Nemotron4 340B model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, and data settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
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
        gradient_clip_val (float): Value for gradient clipping.
        limit_test_batches (int): Limit the number of test batches.
        limit_val_batches (int): Limit the number of validation batches.
        log_every_n_steps (int): Log every n steps.
        val_check_interval (int): Run validation every N steps.
        global_batch_size (int): Global batch size.
        micro_batch_size (int): Micro batch size.
        seq_length (int): Sequence length.
        warmup_steps (int): Number of warmup steps.
        constant_steps (int): Number of constant steps.
        min_lr (float): Minimum learning rate.
        max_lr (float): Maximum learning rate.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory nemotron4_340
            $ nemo llm finetune --factory "nemotron4_340(name='my_nemotron4_340_finetune', num_nodes=4)"

        Python API usage:
            >>> recipe = finetune_recipe(name="my_nemotron4_340_finetune", num_nodes=4)
            >>> print(recipe)

    Note:
        This recipe is optimized for fine-tuning Nemotron4 8b model.
        This recipe uses the SQuAD dataset.
    """
    recipe = pretrain_recipe(
        dir=dir,
        name=name,
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        pipeline_parallelism_type=pipeline_parallelism_type,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
        context_parallelism=context_parallelism,
        sequence_parallelism=sequence_parallelism,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        max_steps=max_steps,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        limit_test_batches=limit_test_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        seq_length=seq_length,
        warmup_steps=warmup_steps,
        constant_steps=constant_steps,
        min_lr=min_lr,
        max_lr=max_lr,
        fn=fn,
    )
    recipe.resume = nemo_resume()
    recipe.peft = run.Config(LoRA)
    recipe.data = run.Config(
        SquadDataModule, seq_length=seq_length, global_batch_size=global_batch_size, micro_batch_size=micro_batch_size
    )
    return recipe
