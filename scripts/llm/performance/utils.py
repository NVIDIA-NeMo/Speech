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

import argparse
import os
from typing import Dict, List, Optional

import nemo_run as run
from lightning.pytorch.callbacks.callback import Callback
from nemo_run.config import NEMORUN_HOME

from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.collections.llm.recipes.llama3_8b import MegatronCommOverlapCallback
from nemo.utils import logging


def slurm_executor(
    account: str,
    partition: str,
    log_dir: str,
    nodes: int,
    num_gpus_per_node: int,
    time_limit: str = "01:00:00",
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    custom_mounts: Optional[List[str]] = None,
    custom_env_vars: Optional[Dict[str, str]] = None,
    custom_srun_args: Optional[List[str]] = None,
    retries: int = 0,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """
    if not (log_dir and account and partition and nodes and num_gpus_per_node):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this ",
            "function.",
        )

    mounts = []
    if custom_mounts:
        mounts.extend(custom_mounts)

    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "False",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "1",
        "NVTE_FLASH_ATTN": "0",
        "NEMO_LOG_MEMORY_USAGE": "1",
        "NEMORUN_HOME": log_dir,
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    srun_args = ["--mpi=pmix"]
    if custom_srun_args:
        srun_args.extend(custom_srun_args)

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.LocalTunnel(
            job_dir=os.path.join(log_dir, "experiments"),
        ),
        nodes=nodes,
        ntasks_per_node=num_gpus_per_node,
        mem="0",
        exclusive=True,
        packager=run.GitArchivePackager(),
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.srun_args = srun_args
    executor.retries = retries
    executor.time = time_limit

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
        "AutoTokenizer first searches for tokenizer files locally in env var 'NEMO_HOME'.",
        "If files are missing locally, AutoTokenizer will try downloading from HuggingFace.",
        "Make sure 'TRANSFORMERS_OFFLINE=0' and 'HF_TOKEN:<token_value>'.",
        "You can set them as scripts.llm.performance.utils.slurm_executor(custom_env_vars=",
        "{'TRANSFORMERS_OFFLINE: 0', 'HF_TOKEN: <token_value>'}",
    ]
    logging.warning(" ".join(log_msg))

    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )


def get_comm_overlap_callback_idx(callbacks: List[Callback]):
    """
    nemo.lightning.Trainer has a list of callbacks defined. This method identifies index of MegatronCommOverlapCallback
    from the list defined in recipes in nemo.collections.llm.recipes. The index is needed to override ddp communication
    params
    """
    if callbacks:  # default is None in lightning
        for idx, callback in enumerate(callbacks):
            if isinstance(callback, MegatronCommOverlapCallback):
                return idx
    return -1


def parse_cli_args():
    """
    Command line arguments correspong to Slurm cluster and NeMo2.0 for running pre-training and
    fine-tuning experiments.
    """
    parser = argparse.ArgumentParser(description="NeMo2.0 Performance Pretraining and Fine-Tuning")

    parser.add_argument(
        "-a",
        "--account",
        type=str,
        help="Slurm account to use for experiment",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        help="Slurm partition to use for experiment",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        help=f"Directory for logging experiment results. Defaults to {NEMORUN_HOME}",
        required=False,
        default=NEMORUN_HOME,
    )
    parser.add_argument(
        "-t",
        "--time_limit",
        type=str,
        help="Maximum time limit to run experiment for. Defaults to 30 minutes (format- 'HH:MM:SS')",
        required=False,
        default="00:30:00",
    )
    parser.add_argument(
        "-i",
        "--container_image",
        type=str,
        help="NeMo container to use for experiment. Defaults to latest dev container- 'nvcr.io/nvidia/nemo:dev'\
            Make sure your NGC credentials are accessible in your environment.",
        required=False,
        default="nvcr.io/nvidia/nemo:dev",
    )
    parser.add_argument(
        "-c",
        "--compute_dtype",
        type=str,
        help="Compute precision. Options- bf16 or fp8. Defaults to bf16",
        required=False,
        default="bf16",
    )
    parser.add_argument(
        "-ep",
        "--enable_profiling",
        help="Enable Nsys profiling. Diabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-tb",
        "--tensorboard",
        help="Enable tensorboard logging. Disabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--dryrun",
        help="If true, prints sbatch script to terminal without launching experiment.",
        required=False,
        action="store_true",
    )

    return parser
