# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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


from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional

import lightning.pytorch as pl
import nemo_run as run
import torch
from lightning.pytorch.callbacks.callback import Callback
from megatron.core.distributed import DistributedDataParallelConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.common.tokenizers.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.llm.gpt.data.megatron.hyena import Evo2Dataset, parse_dataset_config
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.recipes.log.default import default_log, tensorboard_logger, wandb_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed, bf16_with_fp8_mixed
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.utils.exp_manager import TimingCallback

NAME = "hyena_test"


def tokenizer_recipe():
    """
    Creates and returns a configuration for initializing a tokenizer.

    The configuration is set up to use the `get_nmt_tokenizer` function with
    the specified library type as 'byte-level'.

    Returns:
        run.Config: A configuration object for the tokenizer setup.
    """
    return run.Config(
        get_nmt_tokenizer,
        library='byte-level',
    )


def model_recipe(model_size: str, tp_comm_overlap: bool, seq_length: int) -> run.Config[pl.LightningModule]:
    """
    Factory function to create a striped hyena model

    Returns:
        run.Config[pl.LightningModule]: Configuration for the striped hyena model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=hyena_test ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    if model_size == 'test':
        cfg_cls = llm.HyenaTestConfig
    elif model_size == '1b':
        cfg_cls = llm.Hyena1bConfig
    elif model_size == '7b':
        cfg_cls = llm.Hyena7bConfig
    elif model_size == 'nv-7b':
        cfg_cls = llm.HyenaNV7bConfig
    elif model_size == '40b':
        cfg_cls = llm.Hyena40bConfig
    elif model_size == 'nv-40b':
        cfg_cls = llm.HyenaNV40bConfig
    elif model_size == '7b_arc_longcontext':
        cfg_cls = llm.Hyena7bARCLongContextConfig
    elif model_size == '40b_arc_longcontext':
        cfg_cls = llm.Hyena40bARCLongContextConfig
    else:
        raise NotImplementedError(f"Unsupported model size: {model_size}")

    return run.Config(
        llm.HyenaModel,
        config=run.Config(
            cfg_cls,
            seq_length=seq_length,
            tp_comm_overlap=tp_comm_overlap,
        ),
        tokenizer=tokenizer(),
    )


def trainer_recipe(
    tensor_parallelism: int = 8,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 100,
    val_check_interval: int = 100,
    limit_test_batches: int = 50,
    limit_val_batches: int = 32,
    log_every_n_steps: int = 10,
    callbacks: Optional[list[run.Config[Callback]]] = None,
    fp8: bool = False,
    grad_reduce_in_fp32: bool = True,
    align_param_gather: bool = True,
    no_aligned_megatron_ddp: bool = False,
    ckpt_async_save: bool = True,
    save_ckpt_format: Literal['torch_dist', 'zarr'] = 'torch_dist',
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for Striped Hyena model.

    This function sets up the distributed training strategy and other training parameters.

    Args:
        tensor_parallelism (int): Number of tensor replicas for vector parallelism.
        pipeline_parallelism (int): Number of pipeline segments for model pipeline parallelism.
        pipeline_parallelism_type ([type]): Type of pipeline parallelism to apply. Support 'interleaved','split'.
        virtual_pipelien_parallelism (int): Number of virtual pipeline stages for interleaving smaller sub-microbatches
            to reduce the computational graph bubbles caused by pipeline parallel (if pipeline parallel>1 is in use)
        context_parallelism (int): Number of context parallel blocks for processing sub-attention matrices in a block
            parallel fassion.
        sequence_parallelism (bool): Whether to use the sequence_parallelism improvement
            on the base tensor parallelism.
            This will allow for more layers to be parallelized when using tensor parallelism.
        num_nodes (int): Number of nodes for the distributed setting.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps before training terminates.
        val_check_interval (int): Interval between val check runs.
        limit_test_batches (int): Maximum number of batches over which to run test check runs on the dev set data.
        limit_val_batches (int): Maximum number of batches over which to run val check runs on the dev set data.
        log_every_n_steps (int): Log to the endpoint every n steps.
        callbacks (list[run.Config[Callback]]): A list of nemo/lightning callbacks to execute during training.
        fp8 (bool): Whether to use fp8 precision for computations.
        grad_reduce_in_fp32 (bool): Boolean indicating whether to reduce the gradient weight in the FP32 format rather
            than the default bf16.
        align_param_gather (bool): Optimization for faster train step timing potentially through aligning parameter
            gather operations.
        no_aligned_megatron_ddp (bool): Skip the aligned megatron DDP optimizations.
        ckpt_async_save (bool): Bool indicating whether to use asynchronous checkpoint saving.
        save_ckpt_format (Literal['torch_dist', 'zarr']): The checkpoint save method. The default torch_dist may result
            in larger checkpoints currently, but is the preferred option in the long run.
    Returns:
        run.Config[nl.Trainer]: Configuration for the NeMo Lightning Trainer.

    """
    if no_aligned_megatron_ddp:
        ddp: str | DistributedDataParallelConfig = run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            align_param_gather=align_param_gather,
        )
    else:
        ddp = run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            align_param_gather=align_param_gather,
            use_distributed_optimizer=True,  # this should inherit from the optimizer config, but just in case...
        )

    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        gradient_as_bucket_view=True,
        ckpt_async_save=ckpt_async_save,
        save_ckpt_format=save_ckpt_format,
        ckpt_parallel_load=True,
        ddp=ddp,
    )
    if fp8:
        mixed_precision_cfg = bf16_with_fp8_mixed()
        mixed_precision_cfg.fp8_amax_history_len = 16  # from Arc's training setup
    else:
        mixed_precision_cfg = bf16_mixed()

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        callbacks=callbacks,
        devices=num_gpus_per_node,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=mixed_precision_cfg,
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=val_check_interval,
        limit_test_batches=limit_test_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=log_every_n_steps,
    )

    return trainer


def blended_dataset_config_recipe(config_path: Path | str | None = None):
    """
    Creates a configuration for a blended dataset by utilizing the `run.Config` function.

    Args:
        config_path (Path | str | None, optional): The path to the dataset configuration file.
            Can be a `Path` object, a string, or `None`. Defaults to `None`.

    Returns:
        run.Config: A configuration object initialized with the dataset parsing function
        and the provided dataset configuration path.
    """
    return run.Config(
        parse_dataset_config,
        dataset_config_path=config_path,
    )


def pretrain_recipe_creater(
    dataset_config: str = None,
    global_batch_size: int = 8,
    micro_batch_size: int = 1,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    grad_acc_batches: int = 1,
    tensor_parallel_size: int = 8,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    seq_length: int = 8192,
    seed: int = 1234,
    model_size: str = '7b',
    use_megatron_comm_overlap_llama3_8k: bool = False,
    workers: int = 10,
    val_check_interval: int = 100,
    dir: str = None,
    enable_preemption: bool = True,
    align_param_gather: bool = True,
    tflops_callback: bool = None,
    gc_interval: int = 0,
    nsys_profiling: bool = False,
    max_steps: int = 100,
    nsys_start_step: int = 1,
    nsys_end_step: int = None,
    nsys_ranks: list[int] = [0],
    no_aligned_megatron_ddp: bool = False,
    grad_reduce_in_fp32: bool = False,
    fp8: bool = True,
    ckpt_async_save: bool = True,
    sequence_parallel: bool = True,
    ckpt_format: Literal["zarr", "torch_dist"] = "torch_dist",
    limit_val_batches: int = 10,
    restore_optimizer_from_ckpt: bool = False,
    resume_path: str = None,
    wandb_project: str = None,
    wandb_name: str = None,
    name: str = "default",
    fn=pretrain,
    **kwargs,
) -> run.Partial:
    """
    Create a pre-training recipe for a striped hyena model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dataset_config (str): a string specifying the path to the dataset config file (json schema)
        global_batch_size (int | None): global training batch size. If left None this will be inferred from cluster and
            model parallelism configurations.
        micro_batch_size (int): micro batch size per active device.
        num_nodes (int): number of nodes to use
        num_gpus_per_node (int): number of num_gpus_per_node per node to use.
        grad_acc_batches (int): number of training batches the gradients will be accumulated before performing each
            optimizer update.
        tensor_parallel_size (int): Number of tensor parallel splits. Typically between 1 and 8 to keep parallelism
            within a single node.
        pipeline_model_parallel_size (int): Pipeline model parallel size, splits the model by layer.
        context_parallel_size (int): Context model parallel size, how many splits to make across the sequence dimension
            to be processed in parallel similar to the strategy described in the ring attention paper.
        seq_length (int): The desired sequence length to train this model on.
        seed (int): Random seed to use for initialization
        model_size (str): model size to load
        use_megatron_comm_overlap_llama3_8k (bool): If using TP, this controls advanced overlap communications
            which can improve performance during pretraining.
        workers (int): Number of workers to use for per-device batch creation.
        val_check_interval (int): How often the model evaluates during training.
        dir (str): Directory to save logs and checkpoints
        enable_preemption (bool): Enable preemption when training on slurm, captures timeout signals and attempts to
            save a final checkpoint.
        align_param_gather (bool): Optimization for faster train step timing potentially through aligning parameter
            gather operations.
        tflops_callback (bool): Enable tflops callbacks for reporting training speed and device utilization.
        gc_interval (int): How often to run GC operations throughout training (default is auto)
        nsys_profiling (bool): Enable nsys profiling from  NeMo repo.
        max_steps (int): Maximum number of steps the training model should take.
        nsys_start_step (int): Step for when NSYS will start collecting logs
        nsys_end_step (int): Step for when NSYS will stop collecting logs.
        nsys_ranks (list[int]): Ranks for processing nsys logs. Defaults to [0] if not specified.
        no_aligned_megatron_ddp (bool): Disables aligned megatron ddp optimizations.
        fp8 (bool): Whether to use fp8 precision for computations.
        grad_reduce_in_fp32 (bool): Boolean indicating whether to reduce the gradient weight in the FP32 format rather
            than the default bf16.
        ckpt_async_save (bool): Bool indicating whether to use asynchronous checkpoint saving.
        save_ckpt_format (Literal['torch_dist', 'zarr']): The checkpoint save method. The default torch_dist may result
            in larger checkpoints currently, but is the preferred option in the long run.
        resume_path (str): If specified starting weights will be loaded from this checkpoint rather than being
            randomly initialized.
        restore_optimizer_from_ckpt (bool): when loading checkpoint, try to load the optimizer.
        wandb_project (str): if set, logging to wandb will happen
        wandb_name (str): override default name for the wandb log.
    Returns:
        run.Partial: Partial configuration for pre-training.

    """
    model_run_cfg = model_recipe(
        model_size=model_size, seq_length=seq_length, tp_comm_overlap=use_megatron_comm_overlap_llama3_8k
    )
    if not dataset_config:
        data_run_cfg = run.Config(
            MockDataModule,
            seq_length=seq_length,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            tokenizer=tokenizer_recipe(),
        )
    else:
        data_run_cfg = run.Config(
            PreTrainingDataModule,
            paths=blended_dataset_config_recipe(dataset_config),
            dataset_cls=Evo2Dataset,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            seed=seed,
            num_workers=workers,
            tokenizer=tokenizer_recipe(),
        )

    extra_loggers = {}
    if wandb_project is not None:
        if wandb_name is None:
            wandb_name = (
                f"heyna-size-{model_size}-TP{tensor_parallel_size}-"
                f"PP{pipeline_model_parallel_size}-CP{context_parallel_size}"
                f"-GBS{global_batch_size}-MBS{micro_batch_size}"
                f"-GRFP32{grad_reduce_in_fp32}-"
                f"ALIGN{not no_aligned_megatron_ddp}"
                f"-NODES{num_nodes}-FP8{fp8}"
            )
        extra_loggers['wandb_logger'] = wandb_logger(project=wandb_project, name=wandb_name)
    if resume_path:
        restore_cfg = run.Config(
            nl.RestoreConfig,
            path=resume_path,
            load_model_state=True,
            load_optim_state=restore_optimizer_from_ckpt,
        )
    else:
        restore_cfg = None

    nemo_resume = run.Config(
        nl.AutoResume,
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_past_end=False,
        resume_from_directory=dir,
        restore_config=restore_cfg,
    )

    callbacks: list[run.Config] = [
        run.Config(TimingCallback),
    ]
    if use_megatron_comm_overlap_llama3_8k:
        callbacks.append(
            run.Config(
                MegatronCommOverlapCallback,
                tp_comm_overlap=use_megatron_comm_overlap_llama3_8k,
                tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
                wgrad_deferral_limit=22,  # default from NeMo
                overlap_param_gather_with_optimizer_step=False,
                align_param_gather=align_param_gather,
            )
        )

    if gc_interval > 0:
        callbacks.append(
            run.Config(
                nl_callbacks.GarbageCollectionCallback,
                gc_interval_train=gc_interval,
                gc_interval_val=gc_interval,
            )
        )
    if nsys_profiling:
        if nsys_end_step is None:
            nsys_end_step = max_steps
        callbacks.append(
            run.Config(
                nl_callbacks.NsysCallback,
                start_step=nsys_start_step,
                end_step=nsys_end_step,
                ranks=nsys_ranks,
                gen_shape=True,
            )
        )
    if enable_preemption:
        callbacks.append(run.Config(nl_callbacks.PreemptionCallback))
    if tflops_callback:
        # Add callback that logs the tera-FLOPS per second per GPU during training.
        flop_meas_callback = run.Config(
            FLOPsMeasurementCallback,
            run.Config(asdict, model_run_cfg),
            data_run_cfg,
            "hyena",
        )
        callbacks.append(flop_meas_callback)

    return run.Partial(
        fn,
        model=model_run_cfg,
        trainer=trainer_recipe(
            max_steps=max_steps,
            num_nodes=num_nodes,
            tensor_parallelism=tensor_parallel_size,
            pipeline_parallelism=pipeline_model_parallel_size,
            context_parallelism=context_parallel_size,
            sequence_parallelism=sequence_parallel,  # TODO Turn it on by default if TP is on
            num_gpus_per_node=num_gpus_per_node,
            val_check_interval=val_check_interval,
            limit_test_batches=limit_val_batches,
            limit_val_batches=limit_val_batches,
            callbacks=callbacks,
            fp8=fp8,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            align_param_gather=align_param_gather,
            no_aligned_megatron_ddp=no_aligned_megatron_ddp,
            ckpt_async_save=ckpt_async_save,
            save_ckpt_format=ckpt_format,
        ),
        data=data_run_cfg,
        log=default_log(
            dir=dir,
            name=name,
            tensorboard_logger=tensorboard_logger(name=name),
            **extra_loggers,
        ),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4, min_lr=3e-5, warmup_steps=2500),
        resume=nemo_resume,
    )


@run.cli.factory(name=NAME)
def tokenizer() -> run.Config[TokenizerSpec]:
    """
    Creates and returns a tokenizer configuration.

    Returns:
        run.Config[TokenizerSpec]: A configuration object for the tokenizer.
    """
    return tokenizer_recipe()


@run.cli.factory(name=NAME)
def model(tp_comm_overlap: bool, seq_length: int) -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Hyena model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for a Hyena model.

    """
    return model_recipe('test', tp_comm_overlap, seq_length)


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    *args,
    fn=pretrain,
    **kwargs,
) -> run.Partial:
    """
    Create a pre-training recipe for a Hyena model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for pre-training.

    """
    return pretrain_recipe_creater(*args, **kwargs)


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    resume_path: str | None,
    *args,
    fn=finetune,
    **kwargs,
) -> run.Partial:
    """ """
    assert resume_path is not None, "resume_path None, invalid for finetune"
    return pretrain_recipe_creater(*args, resume_path=resume_path, fn=fn, **kwargs)
