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
from typing import Any, Dict, List

import nemo_run as run
from nemo_run.config import get_nemorun_home
from nemo_run.core.execution.launcher import SlurmTemplate

from nemo.lightning.base import DEFAULT_NEMO_CACHE_HOME
from nemo.utils import logging

DEFAULT_NEMO_HOME = os.getenv('NEMO_HOME', DEFAULT_NEMO_CACHE_HOME)

# WORKAROUND: Buffer time for srun timeout to ensure jobs exit before sbatch timeout.
# This allows proper exit code validation. Set to 5 minutes (300 seconds).
# TODO: Remove this workaround once the underlying job exit issue is resolved.
SRUN_TIMEOUT_BUFFER_SECONDS = 300  # 5 minutes

# NOTE: If you update this template,
# PLEASE test it by submitting a job to GPU/node/cluster and verifying the sbatch and bash scripts.
INLINE_TEMPLATE = r"""
#!/usr/bin/env bash
set -euo pipefail

# NOTE: DO NOT change the single quotes to double quotes.
bash -c '{{ pre_cmds }} {{ command }}'
"""


def parse_slurm_time_to_seconds(time_str: str) -> int:
    """
    Parse Slurm time format string to total seconds.

    Supports common Slurm time formats:
    - DD-HH:MM:SS (e.g., "2-04:30:00" = 2 days, 4 hours, 30 minutes)
    - HH:MM:SS (e.g., "04:30:00" = 4 hours, 30 minutes)
    - HH:MM (e.g., "04:30" = 4 hours, 30 minutes)
    - MM (e.g., "30" = 30 minutes)

    Args:
        time_str: Slurm time format string

    Returns:
        Total time in seconds

    Raises:
        ValueError: If time_str is not a valid Slurm time format
    """
    time_str = time_str.strip()

    # Try to match DD-HH:MM:SS format
    if '-' in time_str:
        parts = time_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid Slurm time format: {time_str}")
        days = int(parts[0])
        time_part = parts[1]
    else:
        days = 0
        time_part = time_str

    # Parse the time part (HH:MM:SS, HH:MM, or MM)
    time_components = time_part.split(':')

    if len(time_components) == 3:
        # HH:MM:SS
        hours, minutes, seconds = int(time_components[0]), int(time_components[1]), int(time_components[2])
    elif len(time_components) == 2:
        # HH:MM
        hours, minutes, seconds = int(time_components[0]), int(time_components[1]), 0
    elif len(time_components) == 1:
        # MM (just minutes)
        hours, minutes, seconds = 0, int(time_components[0]), 0
    else:
        raise ValueError(f"Invalid Slurm time format: {time_str}")

    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total_seconds


def format_seconds_to_slurm_time(total_seconds: int) -> str:
    """
    Convert seconds to Slurm time format string.

    Returns the most appropriate format based on the duration:
    - If >= 1 day: DD-HH:MM:SS format
    - Otherwise: HH:MM:SS format

    Args:
        total_seconds: Total time in seconds

    Returns:
        Slurm-formatted time string
    """
    days = total_seconds // 86400
    remaining = total_seconds % 86400
    hours = remaining // 3600
    remaining = remaining % 3600
    minutes = remaining // 60
    seconds = remaining % 60

    if days > 0:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def calculate_srun_timeout_with_buffer(sbatch_timeout: str) -> str:
    """
    WORKAROUND: Calculate srun timeout by subtracting buffer from sbatch timeout.

    This ensures the workload times out before the overall job, allowing proper
    exit code validation to distinguish actual failures from timeouts.

    Args:
        sbatch_timeout: The sbatch (overall job) time limit in Slurm format

    Returns:
        Adjusted time limit for srun in Slurm format

    Note:
        This is a temporary workaround. Remove when underlying job exit issue is fixed.
    """
    try:
        total_seconds = parse_slurm_time_to_seconds(sbatch_timeout)
        adjusted_seconds = total_seconds - SRUN_TIMEOUT_BUFFER_SECONDS

        if adjusted_seconds <= 0:
            logging.warning(
                f"Timeout buffer ({SRUN_TIMEOUT_BUFFER_SECONDS}s) is >= total job time ({total_seconds}s). "
                f"Using original timeout: {sbatch_timeout}"
            )
            return sbatch_timeout

        return format_seconds_to_slurm_time(adjusted_seconds)
    except ValueError as e:
        logging.warning(f"Failed to parse time limit '{sbatch_timeout}': {e}. Using original timeout.")
        return sbatch_timeout


def slurm_executor(
    gpu: str,
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
    custom_bash_cmds: List[str] = None,
    additional_slurm_params: Dict[str, Any] = None,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments

    Args:
        additional_slurm_params: Dict[str, Any], optional
            Additional SLURM parameters to pass to sbatch. These will be converted to #SBATCH directives.
            Example: {"nodelist": "node001,node002", "constraint": "gpu"} will generate:
                #SBATCH --nodelist=node001,node002
                #SBATCH --constraint=gpu
    """
    PERF_ENV_VARS = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
        "TRANSFORMERS_OFFLINE": "1",  # Enable online downloads from HuggingFace
        "TOKENIZERS_PARALLELISM": "False",  # Restrict warning message prints
        "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
        "NVTE_FLASH_ATTN": "1",  # Enable Flash Attention, which is needed to enable cuDNN fused attention
        "NVTE_FUSED_ATTN": "1",  # Enable cuDNN fused attention
        "NEMO_LOG_MEMORY_USAGE": "1",  # Print memory allocation
    }

    custom_bash_cmds = [] if custom_bash_cmds is None else custom_bash_cmds
    err_msgs = []
    mounts = []
    srun_args = custom_srun_args.copy() + ["--mpi=pmix", "--no-container-mount-home", "--container-writable"]

    # WORKAROUND: Set srun timeout to be less than sbatch timeout
    # This ensures jobs exit before hitting the sbatch timeout limit
    srun_timeout = calculate_srun_timeout_with_buffer(time_limit)
    srun_args.append(f"--time={srun_timeout}")

    if log_dir != get_nemorun_home():
        err_msgs.append(f"\nRun `export NEMORUN_HOME={log_dir}` in your shell environment and rerun this script.")
    if len(err_msgs) > 0:
        logging.error("\n".join(err_msgs))
        sys.exit(1)

    if gpu.lower() not in ['b200']:
        # TODO: we currently disable PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        # on B200 as it causes an unexpected error. Add back when issue is debugged and fixed.
        PERF_ENV_VARS["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    PERF_ENV_VARS["NEMORUN_HOME"] = log_dir
    if wandb_key is not None:
        PERF_ENV_VARS["WANDB_API_KEY"] = wandb_key

    if gpu.lower() == 'gb200':
        PERF_ENV_VARS["NCCL_NET_GDR_LEVEL"] = "PHB"  # For NCCL 2.25
        PERF_ENV_VARS["NCCL_NET_GDR_C2C"] = "1"  # For NCCL 2.26

    if nemo_home != DEFAULT_NEMO_CACHE_HOME:  # DO NOT change this to 'DEFAULT_NEMO_HOME'/'NEMO_HOME'
        PERF_ENV_VARS["NEMO_HOME"] = nemo_home
        mounts.extend([f"{nemo_home}:{nemo_home}"])
        # Extra location mount for checkpointing support
        NEMORUN_HOME = os.getenv('NEMORUN_HOME')
        mounts.extend([f"{NEMORUN_HOME}:{NEMORUN_HOME}"])
    if hf_token is not None:
        PERF_ENV_VARS.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})

    PERF_ENV_VARS |= custom_env_vars
    # add all environment variables to container environment
    container_env_args = ["--container-env=" + ",".join(list(PERF_ENV_VARS.keys()))]
    srun_args.extend(container_env_args)

    mounts.extend(custom_mounts)

    if additional_slurm_params is None:
        additional_slurm_params = {}

    # add --segment flag to sbatch if job uses GB200.
    segment = None
    if "segment" in additional_slurm_params:
        segment = additional_slurm_params.pop("segment")
    elif num_gpus_per_node == 4:
        if nodes <= 18:
            segment = nodes
        else:  # nodes > 18
            for segment_candidate in range(18, 0, -1):
                if nodes % segment_candidate == 0:
                    segment = segment_candidate
                    break

    numa_divisor = 2 if gpu.lower() == 'gb200' else 4
    numa_cmd = f"numactl --cpunodebind=$((SLURM_LOCALID/{numa_divisor})) --membind=$((SLURM_LOCALID/{numa_divisor}))"
    custom_bash_cmds.append(numa_cmd)

    launcher = SlurmTemplate(
        template_inline=INLINE_TEMPLATE,
        template_vars={"pre_cmds": " ; ".join(custom_bash_cmds)},
    )

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.LocalTunnel(job_dir=os.path.join(log_dir, "experiments")),
        nodes=nodes,
        ntasks_per_node=num_gpus_per_node,
        container_image=container_image,
        container_mounts=mounts,
        env_vars=PERF_ENV_VARS,
        srun_args=srun_args,
        time=time_limit,
        mem="0",
        exclusive=True,
        packager=run.Packager(),
        segment=segment,
        network=network,
        launcher=launcher,
        additional_parameters=additional_slurm_params,
    )

    return executor


def runai_executor(
    base_url: str,
    app_id: str,
    app_secret: str,
    project_name: str,
    pvc_nemo_run_dir: str,
    nodes: int,
    num_gpus_per_node: int,
    launched_from_cluster: bool = False,
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    custom_mounts: List[dict[str, any]] = [],
    custom_env_vars: Dict[str, str] = {},
    hf_token: str = None,
    wandb_key: str = None,
) -> run.DGXCloudExecutor:
    """
    DGXC Create cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
        "TRANSFORMERS_OFFLINE": "1",  # Enable online downloads from HuggingFace
        "TOKENIZERS_PARALLELISM": "False",  # Restrict warning message prints
        "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
        "NVTE_FLASH_ATTN": "1",  # Enable Flash Attention, which is needed to enable cuDNN fused attention
        "NVTE_FUSED_ATTN": "1",  # Enable cuDNN fused attention
        "NEMO_LOG_MEMORY_USAGE": "1",  # Print memory allocation
    }

    if wandb_key is not None:
        env_vars["WANDB_API_KEY"] = wandb_key
    if hf_token is not None:
        env_vars.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})
    env_vars |= custom_env_vars

    executor = run.DGXCloudExecutor(
        base_url=base_url,
        app_id=app_id,
        app_secret=app_secret,
        project_name=project_name,
        nodes=nodes,
        gpus_per_node=num_gpus_per_node,
        container_image=container_image,
        pvc_nemo_run_dir=pvc_nemo_run_dir,
        env_vars=env_vars,
        launcher="torchrun",  # Use torchrun to launch the processes
        launched_from_cluster=launched_from_cluster,
        pvcs=custom_mounts,
    )

    return executor
