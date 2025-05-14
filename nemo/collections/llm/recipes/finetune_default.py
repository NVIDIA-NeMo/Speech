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

from typing import TYPE_CHECKING, Any, Optional

import lightning.pytorch as pl
import nemo_run as run
import torch

import nemo.lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
from nemo.collections.llm.peft import DoRA, LoRA
from nemo.collections.llm.recipes.log.default import tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.lightning.pytorch.callbacks import PEFT
from nemo.utils.exp_manager import TimingCallback

if TYPE_CHECKING:
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

TokenizerType = Any


def default_finetune_recipe(
    model: run.Config[pl.LightningModule],
    resume_path: str,
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    packed_sequence: bool = False,  # once packing recipe is well tested, change this default to true
    tokenizer: Optional[TokenizerType] = "model",
) -> run.Partial:
    """
    Create a default fine-tuning recipe for any model.

    This function sets up a template for a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        model (run.Config[pl.LightningModule]): Configuration for a NeMo model.
        resume_path (str): Path to the Huggingface model or pretrained distributed checkpoint for resume
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        packed_sequence (bool): Whether to use packed sequence.
        tokenizer (Optional[TokenizerType]): Tokenizer setting to be applied. Can be 'data' or 'model'
            or an instance of TokenizerSpec.

    Returns:
        run.Partial: Partial configuration for fine-tuning.

    See usages of this recipe for further details.
    """
    if packed_sequence:
        datamodule = run.Config(
            llm.SquadDataModule,
            seq_length=2048,
            global_batch_size=8,
            micro_batch_size=1,
            packed_sequence_specs=PackedSequenceSpecs(packed_sequence_size=2048),
        )
    else:
        datamodule = run.Config(llm.SquadDataModule, seq_length=2048, global_batch_size=128, micro_batch_size=1)
    recipe = run.Partial(
        llm.finetune,
        model=model,
        trainer=default_finetune_trainer(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
        ),
        data=datamodule,
        log=default_finetune_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=1e-4, min_lr=0, warmup_steps=50, adam_beta2=0.98),
        resume=nemo_resume(resume_path),
        tokenizer=tokenizer,
    )

    return recipe


def default_finetune_trainer(
    tensor_parallelism=1,
    pipeline_parallelism=1,
    pipeline_parallelism_type=torch.bfloat16,
    virtual_pipeline_parallelism=None,
    context_parallelism=1,
    sequence_parallelism=False,
    num_nodes=1,
    num_gpus_per_node=8,
    max_steps=1000,
    limit_test_batches=None,
    limit_val_batches=None,
    val_check_interval=30,
):
    """
    Create a default fine-tuning trainer for any model.

    This function sets up a template for strategy and trainer.

    Args:
        See docstrings of MegatronStrategy and Trainer.

    Returns:
        run.Config: Config for a finetuning trainer.

    See usages of this in recipes for further details.
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
        ckpt_load_strictness="log_all",
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        devices=num_gpus_per_node,
        limit_test_batches=limit_test_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=1,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=bf16_mixed(),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=val_check_interval,
        callbacks=[run.Config(TimingCallback)],
    )

    return trainer


def default_finetune_log(
    dir: Optional[str] = None,
    name: str = "default",
    tensorboard_logger: Optional[run.Config['TensorBoardLogger']] = None,
    wandb_logger: Optional[run.Config['WandbLogger']] = None,
) -> run.Config[nl.NeMoLogger]:
    """
    Create a default fine-tuning logger for any model.

    This function sets up a template for ModelCheckpoint and NeMoLogger.

    Args:
        See docstrings of ModelCheckpoint and NeMoLogger.

    Returns:
        run.Config: Config for a finetuning NeMoLogger.

    See usages of this in recipes for further details.
    """

    ckpt = run.Config(
        nl.ModelCheckpoint,
        save_last="link",
        save_top_k=2,
        every_n_train_steps=50,
        filename="{model_name}--{val_loss:.2f}-{step}-{consumed_samples}",
    )

    return run.Config(
        nl.NeMoLogger,
        ckpt=ckpt,
        name=name,
        tensorboard=tensorboard_logger,
        wandb=wandb_logger,
        log_dir=dir,
    )


def nemo_resume(model_id: str) -> run.Config[nl.AutoResume]:
    """
    Configure automatic resumption from a NeMo checkpoint converted from Huggingface for
    https://huggingface.co/{model_id}.

    This NeMo checkpoint should be converted from Huggingface beforehand, using nemo.collections.llm.import_ckpt.
    When converting the checkpoint, the NeMo checkpoint will be saved in NEMO_HOME (set to ~/.cache/nemo by default).

    This function sets up the configuration to resume training from path nemo://{model_id}.
    This translates to the full path {NEMO_HOME}/models/{model_id}.

    Args:
        model_id (str): Path to the Huggingface model or pretrained distributed checkpoint for resume

    Returns:
        run.Config[nl.AutoResume]: Configuration for resuming from NeMo checkpoint.
    """
    return run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig, path=f"nemo://{model_id}"),
    )


@run.cli.factory(name='lora')
def lora() -> run.Config[PEFT]:
    """
    Factory function to create a LoRA configuration.

    Returns:
        run.Config[PEFT]: Configuration for the LoRA class.

    Examples:
        CLI usage:
            $ nemo llm finetune -f llama3_8b peft=lora

        Python API usage:
            >>> lora_config = lora()
            >>> print(lora_config)
    """
    return run.Config(LoRA)


@run.cli.factory(name='dora')
def dora() -> run.Config[PEFT]:
    """
    Factory function to create a DoRA configuration.

    Returns:
        run.Config[PEFT]: Configuration for the DoRA class.

    Examples:
        CLI usage:
            $ nemo llm finetune -f llama3_8b peft=dora

        Python API usage:
            >>> dora_config = dora()
            >>> print(dora_config)
    """
    return run.Config(DoRA)
