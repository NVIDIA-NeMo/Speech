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

import os
from pathlib import Path
from typing import List, Optional

import nemo_run as run
import pandas as pd
from numpy import nan

from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.recipes.precision.mixed_precision import (
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_mixed,
    bf16_with_fp8_subchannel_scaling_mixed,
    bf16_with_mxfp8_mixed,
)
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from nemo.utils import logging

from .utils import get_comm_overlap_callback_idx


def get_csv_configs(gpu: str, task: str, model_name: str, model_size: str, args) -> pd.DataFrame:
    """
    Get recommended configs tuned for performance from a csv file.
    User (command line) provided args override the recommended configs.
    """
    script_dir = str(Path(__file__).parent.absolute())
    recommended_configs_csv = os.path.join(script_dir, "recommended_model_configs", f"model_configs_{gpu}.csv")
    logging.info(f"Using {recommended_configs_csv} for loading default recommended model configs")

    config_df = pd.DataFrame()
    if os.path.isfile(recommended_configs_csv):
        df = pd.read_csv(recommended_configs_csv)
        config_df = df[
            (df["task"] == task)
            & (df["model"] == model_name)
            & (df["size"] == model_size)
            & (df["dtype"] == args.compute_dtype)
            & (args.num_gpus is None or df['num_gpus'] == args.num_gpus)
        ]
        config_df = config_df.replace({nan: None})
        if len(config_df) == 0:
            logging.warning(f"Missing performance configs for {task}-{model_name}-{model_size}-{args.compute_dtype}")
            logging.warning("Make sure you provide all necessary arguments in the command line")

    config = config_df.to_dict(orient='records')[0] if len(config_df) > 0 else {}

    return config


def get_user_configs(gpu: str, task: str, model_name: str, model_size: str, args) -> List[int]:
    """
    Choose recommended configs tuned for performance from a csv file if available.
    User (command line) provided args override the recommended configs.

    NOTE: pre-train and PEFT recommended configs available for H100 and B200.

    Args:
        gpu (str): target GPU machine for experiment. Options- ['h100', 'b200']
        task (str): experiment task. Options- ['pre_train', 'sft', 'lora']
        model_name (str): target model for experiment. E.g.: 'llama3', 'mixtral'
        model_size (str): size of target model. E.g.: '8b' (for llama3)
    """
    config = get_csv_configs(gpu.lower(), task, model_name, model_size, args)

    if gpu.lower() == "gb200" and args.gpus_per_node > 4:
        args.gpus_per_node = 4
        logging.warning("GB200 has 4 GPUs per node. Setting gpus_per_node to 4.")
    num_gpus = config.get("num_gpus") if args.num_gpus is None else args.num_gpus
    num_nodes = -(num_gpus // -args.gpus_per_node)  # ceil division
    mbs = config.get("mbs") if args.micro_batch_size is None else args.micro_batch_size
    gbs = config.get("gbs") if args.global_batch_size is None else args.global_batch_size
    tp_size = config.get("tp_size") if args.tensor_parallel_size is None else args.tensor_parallel_size
    pp_size = config.get("pp_size") if args.pipeline_parallel_size is None else args.pipeline_parallel_size
    cp_size = config.get("cp_size") if args.context_parallel_size is None else args.context_parallel_size
    ep_size = config.get("ep_size") if args.expert_parallel_size is None else args.expert_parallel_size
    vp_size = args.virtual_pipeline_parallel_size
    vp_size = config.get("vp_size") if vp_size is None else vp_size
    etp_size = args.expert_tensor_parallel_size
    etp_size = config.get("etp_size") if etp_size is None else etp_size

    enable_cuda_graphs = config.get("cuda_graphs") if args.cuda_graphs is None else args.cuda_graphs
    enable_cuda_graphs = False if enable_cuda_graphs is None else bool(int(enable_cuda_graphs))

    use_mcore_fsdp = config.get("use_mcore_fsdp") if args.use_mcore_fsdp is None else args.use_mcore_fsdp
    use_mcore_fsdp = False if use_mcore_fsdp is None else bool(int(use_mcore_fsdp))

    recompute_layers = config.get("recompute_layers") if args.recompute_layers is None else args.recompute_layers
    recompute_layers = 0 if recompute_layers is None else int(recompute_layers)
    activation_offload_layers = (
        config.get("activation_offload_layers")
        if args.activation_offload_layers is None
        else args.activation_offload_layers
    )
    activation_offload_layers = 0 if activation_offload_layers is None else int(activation_offload_layers)

    if args.recompute_modules is not None:
        recompute_modules = args.recompute_modules
        assert isinstance(recompute_modules, list), "recompute_modules must be a list"
    elif config.get("recompute_modules") is not None:
        recompute_modules = config.get("recompute_modules").split('/')
    else:
        recompute_modules = None

    keep_fsdp_fp8_transpose_cache = (
        config.get("keep_fsdp_fp8_transpose_cache")
        if args.keep_fsdp_fp8_transpose_cache is None
        else args.keep_fsdp_fp8_transpose_cache
    )
    keep_fsdp_fp8_transpose_cache = (
        False if keep_fsdp_fp8_transpose_cache is None else bool(int(keep_fsdp_fp8_transpose_cache))
    )

    use_user_buffer_registration = (
        config.get("use_user_buffer_registration")
        if args.use_user_buffer_registration is None
        else args.use_user_buffer_registration
    )
    use_user_buffer_registration = (
        False if use_user_buffer_registration is None else bool(int(use_user_buffer_registration))
    )

    use_sharp = config.get("use_sharp") if args.use_sharp is None else args.use_sharp
    use_sharp = False if use_sharp is None else bool(int(use_sharp))

    kwargs = num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, etp_size
    kwargs = [int(arg) if arg is not None else arg for arg in kwargs]
    kwargs += [
        enable_cuda_graphs,
        use_mcore_fsdp,
        recompute_layers,
        activation_offload_layers,
        recompute_modules,
        keep_fsdp_fp8_transpose_cache,
        use_user_buffer_registration,
        use_sharp,
    ]

    # print the received arguments for users to debug
    logging.info("Received model parallel configs: ")
    logging.info(f"{num_nodes=}")
    logging.info(f"num_gpus_per_node={args.gpus_per_node}")
    logging.info(f"{mbs=}")
    logging.info(f"{gbs=}")
    logging.info(f"{tp_size=}")
    logging.info(f"{pp_size=}")
    logging.info(f"{cp_size=}")
    logging.info(f"{vp_size=}")
    logging.info(f"{ep_size=}")
    logging.info(f"{etp_size=}")
    logging.info(f"{enable_cuda_graphs=}")
    logging.info(f"{use_mcore_fsdp=}")
    logging.info(f"{recompute_layers=}")
    logging.info(f"{activation_offload_layers=}")
    logging.info(f"{recompute_modules=}")
    logging.info(f"{keep_fsdp_fp8_transpose_cache=}")
    logging.info(f"{use_user_buffer_registration=}")
    logging.info(f"{use_sharp=}")

    return kwargs


def set_mcore_fsdp_configs(recipe, comm_overlap_callback_idx: int | None, tp_size: int | None):
    """
    Set Mcore FSDP related configs.
    """
    recipe.model.config.init_model_with_meta_device = True
    recipe.trainer.strategy.fsdp = "megatron"
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    # At fp32 gradient, `recipe.trainer.strategy.ddp.gradient_reduce_div_fusion` is used for fusion
    if recipe.trainer.plugins.grad_reduce_in_fp32:
        recipe.trainer.strategy.ddp.average_in_collective = False
    recipe.trainer.strategy.ddp.keep_fp8_transpose_cache_when_using_custom_fsdp = False
    recipe.model.config.gradient_accumulation_fusion = False
    if (
        comm_overlap_callback_idx is not None
        and recipe.trainer.callbacks[comm_overlap_callback_idx].defer_embedding_wgrad_compute
    ):
        logging.warning("Disabling deferring embedding wgrad compute because it cannot work with FSDP together.")
        recipe.trainer.callbacks[comm_overlap_callback_idx].defer_embedding_wgrad_compute = False

    return recipe


def set_precision_configs(recipe, compute_dtype: str, fp8_recipe: str | None = None):
    """
    Set precision related configs.
    """
    if compute_dtype is None:
        return recipe

    if compute_dtype.lower() == "bf16":
        recipe.optim.config.use_precision_aware_optimizer = True

    if compute_dtype is not None and compute_dtype.lower() == "fp8":
        if fp8_recipe is None:
            fp8_recipe = "ds"
        if fp8_recipe.lower() == "ds":
            recipe.trainer.plugins = bf16_with_fp8_mixed()
        elif fp8_recipe.lower() == "cs":
            recipe.trainer.plugins = bf16_with_fp8_current_scaling_mixed()
            # disable first/last layer bf16 for benchmarking
            recipe.trainer.plugins.first_last_layers_bf16 = False
        elif fp8_recipe.lower() == "mxfp8":
            recipe.trainer.plugins = bf16_with_mxfp8_mixed()
        elif fp8_recipe.lower() == "ss":
            recipe.trainer.plugins = bf16_with_fp8_subchannel_scaling_mixed()

    recipe.trainer.plugins.grad_reduce_in_fp32 = False

    # Enable reuse_grad_buf_for_mxfp8_param_ag for MXFP8 and disable AG overlap
    # because it is not supported with reuse_grad_buf_for_mxfp8_param_ag
    if compute_dtype.lower() == "fp8" and fp8_recipe.lower() == "mxfp8":
        recipe.trainer.strategy.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
        recipe.optim.config.reuse_grad_buf_for_mxfp8_param_ag = True
        comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
        if comm_overlap_callback_idx is not None:
            recipe.trainer.callbacks[comm_overlap_callback_idx].overlap_param_gather = False
        logging.warning(
            "When using MXFP8, to reduce memory usage, we use reuse_grad_buf_for_mxfp8_param_ag. "
            "Disabling AG overlap because it is not supported with reuse_grad_buf_for_mxfp8_param_ag."
        )

    return recipe


def set_recompute_configs(
    recipe,
    recompute_layers: int,
    activation_offload_layers: int,
    recompute_modules: Optional[List[str]],
):
    """
    Set activation recomputing and offloading related configs.
    """
    if recompute_layers > 0:
        recipe.model.config.recompute_granularity = "full"
        recipe.model.config.recompute_method = "block"
        recipe.model.config.recompute_num_layers = recompute_layers

    # Activation cpu offloading
    if activation_offload_layers > 0:
        recipe.model.config.cpu_offloading = True
        recipe.model.config.cpu_offloading_weights = False
        recipe.model.config.cpu_offloading_num_layers = activation_offload_layers

    # Activation recompute configs
    if recompute_modules is not None:
        recipe.model.config.recompute_modules = recompute_modules
        assert (
            recipe.model.config.recompute_granularity == "selective"
        ), "recompute_granularity must be selective when recompute_modules is provided"
        assert (
            recipe.model.config.recompute_num_layers is None
        ), "recompute_num_layers must be None when recompute_modules is provided"

    return recipe


def set_cuda_graph_configs(recipe, enable_cuda_graphs: bool, task: str):
    """
    Set CUDA graph related configs.
    """
    recipe.model.config.enable_cuda_graph = enable_cuda_graphs
    recipe.trainer.strategy.use_te_rng_tracker = enable_cuda_graphs
    if (
        task in ["none", "lora"]
        and hasattr(recipe.data, "packed_sequence_specs")
        and recipe.data.packed_sequence_specs is not None
    ):
        recipe.data.packed_sequence_specs.pad_cu_seqlens = enable_cuda_graphs

    return recipe


def set_perf_optimization_configs(
    recipe,
    use_mcore_fsdp: bool,
    enable_cuda_graphs: bool,
    task: str,
    tp_size: int | None,
    compute_dtype: str,
    fp8_recipe: str | None,
    recompute_layers: int,
    activation_offload_layers: int,
    recompute_modules: Optional[List[str]],
    use_fsdp_double_buffer: Optional[bool] = None,
    use_user_buffer_registration: Optional[bool] = None,
    use_sharp: Optional[bool] = None,
    keep_fsdp_fp8_transpose_cache: Optional[bool] = None,
):
    """
    Set performance optimization related configs.
    """
    # enable cross entropy fusion with TE kernel
    recipe.model.config.cross_entropy_fusion_impl = "te"

    if use_fsdp_double_buffer:
        assert use_mcore_fsdp == True, "use_fsdp_double_buffer requires use_mcore_fsdp to be True"

    if use_mcore_fsdp and enable_cuda_graphs:
        logging.warning("Currently, cuda graphs are not supported with FSDP. Disabling cuda graphs.")
        enable_cuda_graphs = False
    recipe = set_cuda_graph_configs(recipe, enable_cuda_graphs, task)

    if use_mcore_fsdp:
        comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
        recipe = set_mcore_fsdp_configs(recipe, comm_overlap_callback_idx, tp_size)

    recipe = set_precision_configs(recipe, compute_dtype, fp8_recipe)

    recipe = set_recompute_configs(recipe, recompute_layers, activation_offload_layers, recompute_modules)

    recipe.trainer.strategy.use_sharp = bool(use_sharp)

    is_ddp_obj = hasattr(recipe.trainer.strategy, "ddp") and not isinstance(recipe.trainer.strategy.ddp, str)
    if use_user_buffer_registration and not is_ddp_obj:
        logging.warning("DDP is not configured. Cannot use user buffer registration.")
    if is_ddp_obj:
        # Disable local gradient checker at non-debugging mode
        recipe.trainer.strategy.ddp.check_for_nan_in_grad = False
        recipe.trainer.strategy.ddp.check_for_large_grads = False
        recipe.trainer.strategy.ddp.nccl_ub = bool(use_user_buffer_registration)
        recipe.trainer.strategy.ddp.fsdp_double_buffer = bool(use_fsdp_double_buffer)
        recipe.trainer.strategy.ddp.keep_fp8_transpose_cache_when_using_custom_fsdp = bool(
            keep_fsdp_fp8_transpose_cache
        )

    return recipe


def set_primary_perf_configs(
    recipe,
    task: str,
    num_nodes: int,
    num_gpus_per_node: int,
    mbs: int,
    gbs: int,
    max_steps: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
    etp_size: Optional[int] = None,
    enable_cuda_graphs: bool = False,
    use_mcore_fsdp: bool = False,
    use_fsdp_double_buffer: Optional[bool] = None,
    use_user_buffer_registration: Optional[bool] = None,
    use_sharp: Optional[bool] = None,
    recompute_layers: int = 0,
    activation_offload_layers: int = 0,
    compute_dtype: str = None,
    fp8_recipe: str = None,
    recompute_modules: Optional[List[str]] = None,
    nccl_communicator_config_path: str = None,
    keep_fsdp_fp8_transpose_cache: Optional[bool] = None,
):
    """Set experiment configs we usually tune for performance of all models."""
    # nemo.lightning.Trainer configs
    recipe.trainer.num_nodes = num_nodes
    recipe.trainer.devices = num_gpus_per_node
    recipe.trainer.max_steps = max_steps

    recipe.trainer.val_check_interval = max_steps
    recipe.trainer.limit_val_batches = 0

    # lightning.pytorch.LightningDataModule configs
    recipe.data.micro_batch_size = mbs
    recipe.data.global_batch_size = gbs
    if recipe.data.__fn_or_cls__ == MockDataModule:
        recipe.data.num_train_samples = max_steps * gbs  # ensure only 1 epoch for whole run

    # parallelism configs
    recipe.trainer.strategy.tensor_model_parallel_size = tp_size
    recipe.trainer.strategy.pipeline_model_parallel_size = pp_size
    recipe.trainer.strategy.context_parallel_size = cp_size
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None if vp_size == 1 else vp_size
    recipe.trainer.strategy.expert_model_parallel_size = ep_size
    recipe.trainer.strategy.expert_tensor_parallel_size = etp_size
    recipe.trainer.strategy.sequence_parallel = bool(tp_size > 1)
    if nccl_communicator_config_path is not None:
        recipe.trainer.strategy.nccl_communicator_config_path = nccl_communicator_config_path

    # callback configs
    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    dp_size = (num_nodes * num_gpus_per_node) / (tp_size * pp_size * cp_size)
    if comm_overlap_callback_idx is not None:
        # WARNING: If True, checkpointing (if enabled) might not work
        recipe.trainer.callbacks[comm_overlap_callback_idx].overlap_param_gather_with_optimizer_step = bool(
            dp_size > 1 and pp_size > 1 and vp_size and vp_size > 1
        )

    recipe = set_perf_optimization_configs(
        recipe=recipe,
        use_mcore_fsdp=use_mcore_fsdp,
        enable_cuda_graphs=enable_cuda_graphs,
        task=task,
        tp_size=tp_size,
        compute_dtype=compute_dtype,
        fp8_recipe=fp8_recipe,
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        recompute_modules=recompute_modules,
        use_fsdp_double_buffer=use_fsdp_double_buffer,
        use_user_buffer_registration=use_user_buffer_registration,
        use_sharp=use_sharp,
        keep_fsdp_fp8_transpose_cache=keep_fsdp_fp8_transpose_cache,
    )

    return recipe


def set_exp_logging_configs(
    recipe,
    task: str,
    domain: str,
    model_name: str,
    enable_tb: bool,
    enable_wd: bool,
    wandb_prj_name: str,
    wandb_job_name: str,
):
    """Set experiment logging configs."""
    if task == "pre_train" and domain == "llm":
        recipe.trainer.callbacks.append(
            run.Config(
                FLOPsMeasurementCallback,
                model_config=recipe.model.config,
                data_config=recipe.data,
                model_name=model_name,
            )
        )

    if not enable_tb:  # tensorboard adds performance overhead.
        recipe.log.tensorboard = None
        recipe.trainer.logger = False
    else:
        # default path is NOT intuitive- `<log_dir>/code/nemo_experiments/tb_logs/default/<tfevents_file>`
        recipe.log.log_dir = "/nemo_run/lightning_logs"  # saves file at- `<log_dir>/lightning_logs/tb_logs
    if enable_wd:
        from nemo.collections.llm.recipes.log.default import wandb_logger

        recipe.log.wandb = wandb_logger(project=wandb_prj_name, name=wandb_job_name)

    # Misc. for overall faster experiment runtime
    recipe.log.ckpt = None

    # disable checkpointing if no ModelCheckpoint callback is found
    callbacks = recipe.trainer.callbacks
    checkpoint_callback_idx = None
    if callbacks:  # default is None in lightning
        for idx, callback in enumerate(callbacks):
            if callback.__fn_or_cls__ == ModelCheckpoint:
                checkpoint_callback_idx = idx
                break
    recipe.trainer.enable_checkpointing = checkpoint_callback_idx is not None
    recipe.trainer.log_every_n_steps = 1

    return recipe


def args_sanity_check(args: dict) -> None:
    """
    Check the sanity of argument settings
    """
    if args.wandb:
        assert args.wandb_key is not None, "wandb logger needs \"wandb_key\""
        assert args.wandb_prj_name is not None, "wandb logger needs \"wandb_prj_name\""
        assert args.wandb_job_name is not None, "wandb logger needs \"wandb_job_name\""
