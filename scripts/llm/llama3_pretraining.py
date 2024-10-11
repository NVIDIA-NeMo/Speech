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

# NOTE: This script is only an example of using NeMo with NeMo-Run's APIs and is subject to change without notice.
# This script is used for pretraining a Llama3 model, specifically for the 8b or 70b model variants, on local and slurm executors.
# It uses NeMo 2.0 recipes (https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/llama3_8b.py#L74) and NeMo-Run (https://github.com/NVIDIA/NeMo-Run) to configure and execute the runs.

import argparse
from functools import partial
from typing import Any, Optional

import nemo_run as run

from nemo.collections import llm


def get_parser():
    parser = argparse.ArgumentParser(description="Llama3 Pretraining")
    parser.add_argument(
        "--size",
        type=str,
        default="8b",
        help="Choose llama3 model size 70b/8b",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Optional tag for your experiment title which will be appended after the model/exp name.",
        required=False,
        default="",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dryrun and exit",
        default=False,
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Run on slurm using run.SlurmExecutor",
        default=False,
    )
    return parser


def slurm_executor(
    user: str,
    host: str,
    remote_job_dir: str,
    account: str,
    partition: str,
    nodes: int,
    devices: int,
    time: str = "01:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    retries: int = 0,
) -> run.SlurmExecutor:
    if not (user and host and remote_job_dir and account and partition and nodes and devices):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this function."
        )

    mounts = []
    if custom_mounts:
        mounts.extend(custom_mounts)

    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir,
        ),
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,
        mem="0",
        exclusive=True,
        gres="gpu:8",
        packager=run.GitArchivePackager(subpath="examples/llm/run"),
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor


def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "0",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor


def main():
    args = get_parser().parse_args()
    if args.tag and not args.tag.startswith("-"):
        args.tag = "-" + args.tag

    MODEL_SIZE_MAPPING: dict[str, dict[str, Any]] = {
        "8b": {
            "exp_name": "llama3-8b",
            "nemo": {
                "pretrain": partial(llm.llama3_8b.pretrain_recipe, num_nodes=1, num_gpus_per_node=8),
            },
        },
        "70b": {
            "exp_name": "llama3-70b",
            "nemo": {
                "pretrain": partial(llm.llama3_70b.pretrain_recipe, num_nodes=128, num_gpus_per_node=8),
            },
        },
    }

    exp_name = MODEL_SIZE_MAPPING[args.size]["exp_name"]

    # Uses configs from NeMo directly
    pretrain = MODEL_SIZE_MAPPING[args.size]["nemo"]["pretrain"](
        name=exp_name,
        ckpt_dir="/nemo_run/checkpoints",
    )

    # Overwrite the dataloader in the recipe to use your custom dataloader.
    # dataloader = set_your_custom_dataloader
    # pretrain.data = dataloader

    pretrain.trainer.val_check_interval = 400
    pretrain.log.ckpt.save_top_k = -1
    pretrain.log.ckpt.every_n_train_steps = 400

    pretrain.trainer.max_steps = 1000

    executor: run.Executor

    if args.slurm:
        # TODO: Set your custom parameters for the Slurm Executor.
        executor = slurm_executor(
            user="",
            host="",
            remote_job_dir="",
            account="",
            partition="",
            nodes=pretrain.trainer.num_nodes,
            devices=pretrain.trainer.devices,
        )
    else:
        executor = local_executor_torchrun(nodes=pretrain.trainer.num_nodes, devices=pretrain.trainer.devices)

    with run.Experiment(f"{exp_name}{args.tag}") as exp:
        for i in range(1):
            exp.add(
                pretrain,
                executor=executor,
                name=exp_name,
                tail_logs=True if isinstance(executor, run.LocalExecutor) else False,
            )

        if args.dryrun:
            exp.dryrun()
        else:
            exp.run(sequential=True, detach=True)


if __name__ == "__main__":
    main()
