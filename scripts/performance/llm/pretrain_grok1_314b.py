# *****************************************************************************
#  Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
#  This file is part of the NeMo project, which is licensed under the Apache
#  License, Version 2.0 (the "License"); you may not use this file except in
#  compliance with the License. You may obtain a copy of the License at:
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations
#  under the License.
# *****************************************************************************

""" This is a script to run grok 314b with TP+EP. """

# set OPENBLAS_NUM_THREADS to 1 before any import of nemo, to avoid resource contention
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import os
from os.path import basename, splitext
from typing import Callable, Optional

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import lightning.pytorch as pl
import nemo_run as run
import torch
from lightning.pytorch.callbacks.callback import Callback
from megatron.core.distributed import DistributedDataParallelConfig
from scripts.performance.argument_parser import parse_cli_args

from nemo import lightning as nl
from nemo.collections.llm.api import pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.base import GPTModel
from nemo.collections.llm.gpt.model.mixtral import MixtralConfig8x7B
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
)
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.callbacks.moe_token_drop import MegatronTokenDropCallback
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin
from nemo.utils.exp_manager import TimingCallback

from ..argument_parser import parse_additional_slurm_params, parse_cli_args
from ..executors import slurm_executor
from ..helpers import (
    args_sanity_check,
    build_perf_env_plugin,
    build_torch_profiler_plugin,
    get_comm_overlap_callback_idx,
    get_user_configs,
    set_exp_logging_configs,
    set_primary_perf_configs,
)
from ..utils import hf_tokenizer

NAME = "grok1_314b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Grok-1 314B model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Grok-1 314B model.

    Examples:
        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """
    return run.Config(
        GPTModel,
        config=run.Config(
            MixtralConfig8x7B,
            num_layers=64,
            hidden_size=6144,
            num_attention_heads=48,
            num_query_groups=8,
            ffn_hidden_size=32768,
            max_position_embeddings=32768,
            seq_length=8192,
            num_moe_experts=8,  # 8
            init_method_std=0.008,
        ),
    )


def trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 4,
    pipeline_parallelism_type: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = 8,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    expert_parallelism: int = 8,
    num_nodes: int = 8,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for Grok 314B model.

    This function sets up the distributed training strategy optimized for the Grok 314B model.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_type (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        expert_parallelism (int): Degree of expert parallelism.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps.
        callbacks (Optional[list[run.Config[Callback]]]): List of callback configurations.

    Returns:
        run.Config[nl.Trainer]: Configuration for the NeMo Lightning Trainer.

    Examples:
        CLI usage:
            $ nemo llm pretrain trainer=mixtral_8x7b ...

        Python API usage:
            >>> trainer_config = trainer(num_nodes=2, num_gpus_per_node=8)
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
        expert_model_parallel_size=expert_parallelism,
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
        plugins=run.Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=2000,
    )

    return trainer


def pretrain_performance_optimizations(recipe: run.Partial) -> run.Partial:
    """
    Create a performance-optimized pre-training recipe for Grok 314B model.

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

    garbage_collection_callback = run.Config(
        GarbageCollectionCallback,
        gc_interval_train=100,
        gc_interval_val=100,
    )
    mcomm_overlap_callback = run.Config(
        MegatronCommOverlapCallback,
        tp_comm_overlap=True,
        # 'overlap_param_gather_with_optimizer_step' is set automatically. Added here for user's knowledge
        overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to issue with checkpointing.
    )
    recipe.trainer.callbacks.extend(
        [
            run.Config(MegatronTokenDropCallback),
            garbage_collection_callback,
            mcomm_overlap_callback,
        ]
    )

    recipe.trainer.strategy.expert_model_parallel_size = 1
    recipe.trainer.strategy.tensor_model_parallel_size = 8
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.plugins.grad_reduce_in_fp32 = False

    return recipe


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 8,
    num_gpus_per_node: int = 8,
    performance_mode: bool = False,
    fn: Callable = pretrain,
) -> run.Partial:
    """
    Create a pre-training recipe for Grok 314B model.

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
            $ nemo llm pretrain --factory mixtral_8x7b
            $ nemo llm pretrain --factory "mixtral_8x7b(num_nodes=8, name='my_mixtral_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="mixtral_8x7b_pretrain", num_nodes=8)
            >>> print(recipe)
    """
    recipe = run.Partial(
        fn,
        model=model(),
        trainer=trainer(
            num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, callbacks=[run.Config(TimingCallback)]
        ),
        data=run.Config(MockDataModule, seq_length=8192, global_batch_size=512, micro_batch_size=1),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )

    if performance_mode:
        recipe = pretrain_performance_optimizations(recipe)

    return recipe


def override_recipe_configs(
    args: str,
    num_nodes: int,
    mbs: int,
    gbs: int,
    max_steps: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
    num_layers: int,
    hidden_size: int,
    etp_size: int = None,
    enable_cuda_graphs: bool = False,
    use_mcore_fsdp: bool = False,
    recompute_layers: int = 0,
    activation_offload_layers: int = 0,
    compute_dtype: str = None,
    fp8_recipe: str = None,
):
    recipe = pretrain_recipe(performance_mode=True)

    recipe = set_primary_perf_configs(
        recipe,
        "pre_train",
        num_nodes,
        args.gpus_per_node,
        mbs,
        gbs,
        max_steps,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        num_layers,
        hidden_size,
        etp_size,
        enable_cuda_graphs,
        use_mcore_fsdp,
        args.use_fsdp_double_buffer,
        args.use_user_buffer_registration,
        args.use_sharp,
        recompute_layers,
        activation_offload_layers,
        compute_dtype,
        fp8_recipe,
    )
    recipe = set_exp_logging_configs(
        recipe,
        "pre_train",
        "llm",
        "mixtral",
        args.tensorboard,
        args.wandb,
        args.wandb_prj_name,
        args.wandb_job_name,
    )

    # data module configs
    recipe.data.num_train_samples = max_steps * gbs * mbs  # ensure only 1 epoch for whole run
    recipe.data.tokenizer = hf_tokenizer("meta-llama/Meta-Llama-3-70B")

    # compute dtype configs
    if args.compute_dtype.lower() == "fp8":
        recipe.trainer.plugins = bf16_with_fp8_mixed()
        recipe.trainer.plugins.grad_reduce_in_fp32 = False

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    assert comm_overlap_callback_idx is not None, "MegatronCommOverlapCallback missing. Required for performance."

    enable_tp_comm_overlap = os.environ.get("TP_COMM_OVERLAP", "True").lower() in ("1", "true", "yes")
    recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap = enable_tp_comm_overlap

    tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
    # needed as tp_overlap_configs.userbuffers are dataclass objects which are unserializable
    tp_comm_overlap_cfg = fdl.cast(run.Config, fdl_dc.convert_dataclasses_to_configs(tp_comm_overlap_cfg))
    recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = tp_comm_overlap_cfg

    recipe.model.config.enable_cuda_graph = enable_cuda_graphs
    recipe.trainer.strategy.use_te_rng_tracker = enable_cuda_graphs

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)
    # Parse additional SLURM parameters if provided
    additional_slurm_params = None
    if hasattr(args, 'additional_slurm_params') and args.additional_slurm_params:
        additional_slurm_params = parse_additional_slurm_params(args.additional_slurm_params)

    kwargs = get_user_configs(args.gpu.lower(), "pre_train", "grok1", "314b", args)

    (
        num_nodes,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        num_layers,
        hidden_size,
        etp_size,
        enable_cuda_graphs,
        use_mcore_fsdp,
        recompute_layers,
        activation_offload_layers,
    ) = kwargs[:15]

    recipe = override_recipe_configs(
        args,
        num_nodes,
        mbs,
        gbs,
        args.max_steps,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        num_layers,
        hidden_size,
        etp_size,
        enable_cuda_graphs,
        use_mcore_fsdp,
        recompute_layers,
        activation_offload_layers,
        args.compute_dtype,
        args.fp8_recipe,
    )

    exp_config = f"gpus{args.num_gpus}_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_ep{ep_size}_etp{etp_size}_mbs{mbs}_gbs{gbs}"
    exp_name = f"{splitext(basename(__file__))[0]}_{args.compute_dtype}_{exp_config}"

    env_vars = args.custom_env_vars

    if args.gpu.lower() == 'gb200':
        env_vars |= {"NCCL_NET_GDR_LEVEL": "PHB"}

    plugins = [build_perf_env_plugin(args, pp_size=pp_size)]

    if args.enable_nsys:
        plugins.append(
            NsysPlugin(
                start_step=args.profiling_start_step,
                end_step=args.profiling_stop_step,
                ranks=list(range(num_nodes * args.gpus_per_node)),
                nsys_gpu_metrics=args.profiling_gpu_metrics,
            )
        )

    if torch_profiler_plugin := build_torch_profiler_plugin(args):
        plugins.append(torch_profiler_plugin)

    if args.enable_memory_profile:
        assert args.memory_profile_out_path is not None
        plugins.append(MemoryProfilePlugin(dir=args.memory_profile_out_path))

    executor = slurm_executor(
        args.gpu.lower(),
        args.account,
        args.partition,
        args.log_dir,
        num_nodes,
        args.gpus_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars=env_vars,
        custom_srun_args=[],
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
        wandb_key=args.wandb_key,
        network='sharp' if args.use_sharp else None,
        additional_slurm_params=additional_slurm_params,
    )

    with run.Experiment(exp_name) as exp:
        exp.add(
            recipe,
            executor=executor,
            name=exp_name,
            plugins=plugins,
        )

        if not args.dryrun:
            exp.run(sequential=False, detach=True)
        else:
            exp.dryrun()
