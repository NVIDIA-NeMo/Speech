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
# This script is used for pretraining on local and slurm executors.
# It uses NeMo 2.0 recipes (https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/) and
# NeMo-Run (https://github.com/NVIDIA/NeMo-Run) to configure and execute the runs.

import argparse
import os
from typing import Optional

import nemo_run as run

import nemo.lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.gpt.data.hf_dataset import SquadHFDataModule
from nemo.utils import logging

# TODO: Set your SQuaD dataset path, remember to add the path in custom_mounts if using slurm executor
DATA_PATH = ''


def get_parser():
    parser = argparse.ArgumentParser(description="NeMo2.0 Pretraining")
    parser.add_argument('--model', default='nvidia/Llama-3_3-Nemotron-Super-49B-v1')
    parser.add_argument('--nodes', type=int, default=4)
    parser.add_argument('--devices', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=200)
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
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Huggingface token for downloading models",
        required=False,
        default=None,
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
    time: str = "04:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = "nvcr.io/nvidia/nemo:25.02",
    retries: int = 0,
) -> run.SlurmExecutor:
    if not (user and host and remote_job_dir and account and partition and nodes and devices):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this ",
            "function.",
        )

    mounts = []
    if custom_mounts:
        mounts.extend(custom_mounts)

    env_vars = {
        "TRANSFORMERS_OFFLINE": "0",
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
        packager=run.GitArchivePackager(),
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor


def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    env_vars = {
        "TRANSFORMERS_OFFLINE": "0",
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

    exp_name = "HFAutoModelForCausalLM"

    # Uses configs from NeMo directly
    recipe = llm.hf_auto_model_for_causal_lm.finetune_recipe(
        model_name=args.model,
        name=exp_name,
        num_nodes=args.nodes,
        num_gpus_per_node=args.devices,
        peft_scheme='none',
        dir="/nemo_run/checkpoints",
        max_steps=args.max_steps,
        trust_remote_code=True,
        attn_implementation='eager',
    )

    recipe.trainer.val_check_interval = 50

    tokenizer = llm.HFAutoModelForCausalLM.configure_tokenizer(args.model)
    recipe.data = run.Config(
        SquadHFDataModule,
        path_or_dataset=DATA_PATH,
        split="train[:100]",
        pad_token_id=tokenizer.tokenizer.eos_token_id,
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name=args.model),
    )

    recipe.trainer.strategy = run.Config(
        nl.FSDP2Strategy,
        data_parallel_size=1,
        tensor_parallel_size=1,
        context_parallel_size=32,
    )
    recipe.trainer.plugins = None

    if args.hf_token is not None:
        os.environ["HF_TOKEN"] = args.hf_token

    executor: run.Executor

    if args.slurm:
        if args.hf_token:
            custom_env_vars = {
                "HF_TOKEN": args.hf_token,
            }
        elif os.environ.get("HF_TOKEN"):
            custom_env_vars = {
                "HF_TOKEN": os.environ["HF_TOKEN"],
            }
        else:
            custom_env_vars = {}
            logging.info("No HF_TOKEN provided, gated repos may be inaccessible.")

        # TODO: Set your custom parameters for the Slurm Executor.
        executor = slurm_executor(
            user="",
            host="",
            remote_job_dir="",
            account="",
            partition="",
            nodes=recipe.trainer.num_nodes,
            devices=recipe.trainer.devices,
            custom_mounts=[],
            custom_env_vars=custom_env_vars,
        )
    else:
        executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    with run.Experiment(f"{exp_name}{args.tag}") as exp:
        for i in range(1):
            exp.add(
                recipe,
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
