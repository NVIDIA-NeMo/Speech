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
import sys
from pathlib import Path
from typing import Dict, List, Optional

import nemo_run as run
import pandas as pd
from lightning.pytorch.callbacks.callback import Callback
from nemo_run.config import get_nemorun_home
from numpy import nan

from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model import GPTModel
from nemo.collections.llm.recipes.llama3_8b import MegatronCommOverlapCallback
from nemo.collections.llm.recipes.precision.mixed_precision import (
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_mixed,
    bf16_with_mxfp8_mixed,
)
from nemo.lightning.base import DEFAULT_NEMO_CACHE_HOME
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from nemo.utils import logging

DEFAULT_NEMO_HOME = os.getenv('NEMO_HOME', DEFAULT_NEMO_CACHE_HOME)


def slurm_executor(
    account: str,
    partition: str,
    log_dir: str,
    nodes: int,
    num_gpus_per_node: int,
    time_limit: str = "00:30:00",
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    custom_mounts: List[str] = [],
    custom_env_vars: Dict[str, str] = {},
    custom_srun_args: List[str] = [],
    hf_token: str = None,
    nemo_home: str = DEFAULT_NEMO_HOME,
    wandb_key: str = None,
    network: str = None,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """
    err_msgs = []
    if log_dir != get_nemorun_home():
        err_msgs.append(f"\nRun `export NEMORUN_HOME={log_dir}` in your shell environment and rerun this script.")
    if len(err_msgs) > 0:
        logging.error("\n".join(err_msgs))
        sys.exit(1)

    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
        "TRANSFORMERS_OFFLINE": "1",  # Enable online downloads from HuggingFace
        "TOKENIZERS_PARALLELISM": "False",  # Restrict warning message prints
        "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
        "NVTE_FLASH_ATTN": "1",  # Enable Flash Attention, which is needed to enable cuDNN fused attention
        "NVTE_FUSED_ATTN": "1",  # Enable cuDNN fused attention
        "NEMO_LOG_MEMORY_USAGE": "1",  # Print memory allocation
        "NEMORUN_HOME": log_dir,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    # Assuming 4 GPU's per node as GB200, might need to revisit in future!
    if num_gpus_per_node == 4:
        env_vars["NCCL_NET_GDR_LEVEL"] = "PHB"  # For NCCL 2.25
        env_vars["NCCL_NET_GDR_C2C"] = 1  # For NCCL 2.26

    if wandb_key is not None:
        env_vars["WANDB_API_KEY"] = wandb_key
    mounts = []

    srun_args = custom_srun_args.copy()
    srun_args.extend(
        [
            "--mpi=pmix",
        ]
    )

    if num_gpus_per_node == 4:
        srun_args.append("numactl --cpunodebind=$((SLURM_LOCALID/2)) --membind=$((SLURM_LOCALID/2))")
    else:
        srun_args.append("numactl --cpunodebind=$((SLURM_LOCALID/4)) --membind=$((SLURM_LOCALID/4))")

    if nemo_home != DEFAULT_NEMO_CACHE_HOME:  # DO NOT change this to 'DEFAULT_NEMO_HOME'/'NEMO_HOME'
        env_vars.update({"NEMO_HOME": nemo_home})
        mounts.extend([f"{nemo_home}:{nemo_home}"])
    if hf_token is not None:
        env_vars.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})

    env_vars |= custom_env_vars
    mounts.extend(custom_mounts)

    # add --segment flag to sbatch if job uses GB200 and goes beyond one rack.
    segment = None
    if num_gpus_per_node == 4 and nodes > 18:
        for segment_candidate in range(18, 0, -1):
            if nodes % segment_candidate == 0:
                segment = segment_candidate
                break

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.LocalTunnel(
            job_dir=os.path.join(log_dir, "experiments"),
        ),
        nodes=nodes,
        ntasks_per_node=num_gpus_per_node,
        container_image=container_image,
        container_mounts=mounts,
        env_vars=env_vars,
        srun_args=srun_args,
        time=time_limit,
        mem="0",
        exclusive=True,
        packager=run.GitArchivePackager(),
        segment=segment,
        network=network,
    )

    return executor


def hf_tokenizer(model_name: str) -> run.Config[AutoTokenizer]:
    """
    HuggingFace tokenizer.

    Args:
        model_name (str): corresponds to HuggingFace-AutoTokenizer's 'pretrained_model_name_or_path' input argument.
                For more details please refer to-
                huggingface.co/docs/transformers/v4.47.1/en/model_doc/auto#transformers.AutoTokenizer
    """
    log_msg = [
        f"`AutoTokenizer` first searches for tokenizer files locally stored in {DEFAULT_NEMO_HOME}.",
        "(from env var `NEMO_HOME`- can be changed using '-nh/--nemo_home' CLI arg).",
        "If files are missing locally, `AutoTokenizer` will try downloading from HuggingFace. In this case-",
        "make sure env vars 'TRANSFORMERS_OFFLINE':'0' and 'HF_TOKEN':'<token_value>' are set in your sbatch script.",
        "Both of these will be set automatically if you provide '-hf/--hf_token' CLI arg.",
    ]
    logging.warning(" ".join(log_msg))

    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )


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

    kwargs = num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, etp_size
    kwargs = [int(arg) if arg is not None else arg for arg in kwargs] + [
        enable_cuda_graphs,
        use_mcore_fsdp,
        recompute_layers,
        activation_offload_layers,
        recompute_modules,
    ]

    return kwargs


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
    use_user_buffer_registration: bool = False,
    use_sharp: bool = False,
    recompute_layers: int = 0,
    activation_offload_layers: int = 0,
    compute_dtype: str = None,
    fp8_recipe: str = None,
    recompute_modules: Optional[List[str]] = None,
    nccl_communicator_config_path: str = None,
):
    """Set experiment configs we usually tune for performance of all models."""

    # print the received arguments for users to debug
    logging.info("Received model parallel configs: ")
    logging.info(f"num_nodes: {num_nodes}")
    logging.info(f"num_gpus_per_node: {num_gpus_per_node}")
    logging.info(f"mbs: {mbs}")
    logging.info(f"gbs: {gbs}")
    logging.info(f"tp_size: {tp_size}")
    logging.info(f"pp_size: {pp_size}")
    logging.info(f"cp_size: {cp_size}")
    logging.info(f"vp_size: {vp_size}")
    logging.info(f"ep_size: {ep_size}")
    logging.info(f"etp_size: {etp_size}")
    logging.info(f"enable_cuda_graphs: {enable_cuda_graphs}")
    logging.info(f"use_mcore_fsdp: {use_mcore_fsdp}")
    logging.info(f"use_user_buffer_registration: {use_user_buffer_registration}")
    logging.info(f"use_sharp: {use_sharp}")
    logging.info(f"recompute_layers: {recompute_layers}")
    logging.info(f"activation_offload_layers: {activation_offload_layers}")
    logging.info(f"compute_dtype: {compute_dtype}")
    logging.info(f"fp8_recipe: {fp8_recipe}")
    logging.info(f"recompute_modules: {recompute_modules}")

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

    # enable cross entropy fusion with TE kernel
    recipe.model.config.cross_entropy_fusion_impl = "te"

    # Cuda graph configs
    if use_mcore_fsdp and enable_cuda_graphs:
        logging.warning("Currently, cuda graphs are not supported with FSDP. Disabling cuda graphs.")
        enable_cuda_graphs = False
    recipe.model.config.enable_cuda_graph = enable_cuda_graphs
    recipe.trainer.strategy.use_te_rng_tracker = enable_cuda_graphs
    if (
        task in ["none", "lora"]
        and hasattr(recipe.data, "packed_sequence_specs")
        and recipe.data.packed_sequence_specs is not None
    ):
        recipe.data.packed_sequence_specs.pad_cu_seqlens = enable_cuda_graphs

    # FSDP configs
    if use_mcore_fsdp:
        recipe.model.config.init_model_with_meta_device = True
        recipe.trainer.strategy.fsdp = "megatron"
        recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "optim_grads_params"
        recipe.trainer.strategy.ddp.average_in_collective = False
        recipe.trainer.strategy.ddp.keep_fp8_transpose_cache_when_using_custom_fsdp = False
        recipe.model.config.gradient_accumulation_fusion = False
        if (
            comm_overlap_callback_idx is not None
            and recipe.trainer.callbacks[comm_overlap_callback_idx].defer_embedding_wgrad_compute
        ):
            logging.warning("Disabling deferring embedding wgrad compute because it cannot work with FSDP together.")
            recipe.trainer.callbacks[comm_overlap_callback_idx].defer_embedding_wgrad_compute = False
            if tp_size is not None and tp_size > 1:
                logging.warning(
                    "Currently, TP overlap performance is poor when FSDP is used because of jitters. "
                    "A fix is in progress. Disabling TP overlap."
                )
                recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap = False

    # User buffers configs
    if use_user_buffer_registration:
        recipe.trainer.strategy.ddp.nccl_ub = True

    # Sharp configs
    if use_sharp:
        recipe.trainer.strategy.use_sharp = True

    # Recompute configs
    if recompute_layers > 0:
        recipe.model.config.recompute_granularity = "full"
        recipe.model.config.recompute_method = "block"
        recipe.model.config.recompute_num_layers = recompute_layers

    # Activation cpu offloading
    if activation_offload_layers > 0:
        recipe.model.config.cpu_offloading = True
        recipe.model.config.cpu_offloading_weights = False
        recipe.model.config.cpu_offloading_num_layers = activation_offload_layers

    if compute_dtype.lower() == "bf16":
        recipe.optim.config.use_precision_aware_optimizer = True

    # low precision training configs
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
        recipe.trainer.plugins.grad_reduce_in_fp32 = False

    # Activation recompute configs
    if recompute_modules is not None:
        recipe.model.config.recompute_modules = recompute_modules
        assert (
            recipe.model.config.recompute_granularity == "selective"
        ), "recompute_granularity must be selective when recompute_modules is provided"
        assert (
            recipe.model.config.recompute_num_layers is None
        ), "recompute_num_layers must be None when recompute_modules is provided"

    # Disable local gradient checker at non-debugging mode
    recipe.trainer.strategy.ddp.check_for_nan_in_grad = False
    recipe.trainer.strategy.ddp.check_for_large_grads = False

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


def import_ckpt_experiment(executor: run.SlurmExecutor, model: run.Config[GPTModel], source: str):
    """
    Downloads/Acceses checkpoint to be used for fine-tuning. `import_ckpt` first tries find the nemo checkpoint in
    <NEMO_HOME>/models/. For eg: for llama3 8b, the path will look like- <NEMO_HOME>/models/meta-llama/Meta-Llama-3-8B
    If missing, tries to downloads at the same location from HuggingFace and converts it nemo format.

    Args:
        source (str): HuggingFace URL. For eg- hf://meta-llama/Meta-Llama-3-70B
    """
    from copy import deepcopy

    from nemo.collections.llm import import_ckpt

    import_executor = deepcopy(executor)
    import_executor.ntasks_per_node = 1
    import_executor.nodes = 1

    return run.Partial(import_ckpt, model=model, source=source, overwrite=False), import_executor, "import_ckpt_exp"


def isfile_train_pack_metadata(hf_model_uri: str, data_config: run.Config[SquadDataModule]) -> bool:
    """
    This method is used for fine-tuning. It checks if packed train data for a partiular
    sequence length exists locally. This is needed to set data flag (force_redownload=True)
    which avoids experiment crash in case files are missing.
    """
    datasets_dir = os.getenv("NEMO_DATASETS_CACHE", os.path.join(DEFAULT_NEMO_HOME, "datasets"))
    model_dir = hf_model_uri.replace("/", "--")
    metadata_filename = f"{data_config.seq_length}_metadata.jsonl"

    train_pack_metadata_filepath = os.path.join(datasets_dir, "squad", "packed", model_dir, metadata_filename)

    return os.path.exists(train_pack_metadata_filepath) and os.path.isfile(train_pack_metadata_filepath)


def get_comm_overlap_callback_idx(callbacks: List[Callback]) -> int | None:
    """
    nemo.lightning.Trainer has a list of callbacks defined. This method identifies index of MegatronCommOverlapCallback
    from the list defined in recipes in nemo.collections.llm.recipes. The index is needed to override ddp communication
    params
    """
    if callbacks:  # default is None in lightning
        for idx, callback in enumerate(callbacks):
            if callback.__fn_or_cls__ == MegatronCommOverlapCallback:
                return idx
    return None


def args_sanity_check(args: dict) -> None:
    """
    Check the sanity of argument settings
    """
    if args.wandb:
        assert args.wandb_key is not None, "wandb logger needs \"wandb_key\""
        assert args.wandb_prj_name is not None, "wandb logger needs \"wandb_prj_name\""
        assert args.wandb_job_name is not None, "wandb logger needs \"wandb_job_name\""
