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
from typing import Dict, List, Optional

import nemo_run as run
from nemo_run.config import get_nemorun_home
from nemo_run.core.execution.launcher import SlurmTemplate

from nemo.lightning.base import DEFAULT_NEMO_CACHE_HOME
from nemo.utils import logging

DEFAULT_NEMO_HOME = os.getenv('NEMO_HOME', DEFAULT_NEMO_CACHE_HOME)

# NOTE: If you update this template,
# PLEASE test it by submitting a job to GPU/node/cluster and verifying the sbatch and bash scripts.
INLINE_TEMPLATE = r"""
#!/usr/bin/env bash
set -euo pipefail

# NOTE: DO NOT change the single quotes to double quotes.
bash -c '{{ pre_cmds }} {{ command }}'
"""


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
    optional_gpus_per_node: Optional[int] = None,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
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
    if hf_token is not None:
        PERF_ENV_VARS.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})

    PERF_ENV_VARS |= custom_env_vars
    mounts.extend(custom_mounts)

    # add --segment flag to sbatch if job uses GB200 and goes beyond one rack.
    segment = None
    if num_gpus_per_node == 4 and nodes > 18:
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
        gpus_per_node=optional_gpus_per_node,
        container_image=container_image,
        container_mounts=mounts,
        env_vars=PERF_ENV_VARS,
        srun_args=srun_args,
        time=time_limit,
        mem="0",
        exclusive=True,
        packager=run.GitArchivePackager(),
        segment=segment,
        network=network,
        launcher=launcher,
    )

    return executor
