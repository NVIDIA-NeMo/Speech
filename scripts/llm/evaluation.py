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
# This script is used for evaluation on local and slurm executors using NeMo-Run.
# It uses deploy method from nemo/llm/collections/api.py to deploy nemo2.0 ckpt on PyTriton server and uses evaluate
# method from nemo/llm/collections/api.py to run evaluation on it.
# (https://github.com/NVIDIA/NeMo-Run) to configure and execute the runs.

import argparse
from typing import Optional

import nemo_run as run

from nemo.collections.llm import deploy, evaluate
from nemo.collections.llm.evaluation.api import ApiEndpoint, ConfigParams, EvaluationConfig, EvaluationTarget


ENDPOINT_TYPES = {"chat": "chat/completions/", "completions": "completions/"}

COMPLETIONS_TASKS = (
    "gsm8k",
    "mgsm",
    "mmlu",
    "mmlu_pro",
    "mmlu_redux",
)

CHAT_TASKS = (
    "gpqa_diamond_cot",
    "gsm8k_cot_instruct",
    "ifeval",
    "mgsm_cot",
    "mmlu_instruct",
    "mmlu_pro_instruct",
    "mmlu_redux_instruct",
    "wikilingua",
)

EVAL_TASKS = COMPLETIONS_TASKS + CHAT_TASKS


def get_parser():
    parser = argparse.ArgumentParser(description="NeMo2.0 Evaluation")
    parser.add_argument(
        "--nemo_checkpoint",
        type=str,
        required=True,
        help="NeMo 2.0 checkpoint to be evaluated",
    )
    parser.add_argument(
        "--triton_http_address", type=str, default="0.0.0.0", help="IP address at which PyTriton server is created"
    )
    parser.add_argument("--fastapi_port", type=int, default=8080, help="Port at which FastAPI server is created")
    parser.add_argument(
        "--endpoint_type",
        type=str,
        default="completions",
        help="Whether to use completions or chat endpoint",
        choices=list(ENDPOINT_TYPES),
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=4096,
        help="Max input length of the model",
    )
    parser.add_argument(
        "--tensor_parallelism_size",
        type=int,
        default=1,
        help="Tensor parallelism size to deploy the model",
    )
    parser.add_argument(
        "--pipeline_parallelism_size",
        type=int,
        default=1,
        help="Pipeline parallelism size to deploy the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for deployment and evaluation",
    )
    parser.add_argument(
        "--eval_task",
        type=str,
        default="mmlu",
        help="Evaluation benchmark to run.",
        choices=EVAL_TASKS,
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit evaluation to `limit` samples. Default: use all samples."
    )
    parser.add_argument(
        "--parallel_requests",
        type=int,
        default=1,
        help="Number of parallel requests to send to server. Default: use default for the task.",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=None,
        help="Request timeout for querying the server. Default: use default for the task.",
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
    parser.add_argument('--nodes', type=int, default=2, help="Num nodes for the executor")
    parser.add_argument('--devices', type=int, default=8, help="Num devices per node for the executor")
    parser.add_argument(
        '--container_image',
        type=str,
        default="nvcr.io/nvidia/nemo:dev",
        help="Container image for the run, only used in case of slurm runs."
        "Can be a path as well in case of .sqsh file.",
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
    container_image: str,
    time: str = "04:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
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
        # required for some eval benchmarks from lm-eval-harness
        "HF_DATASETS_TRUST_REMOTE_CODE": "1"
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
        ntasks_per_node=1,
        exclusive=True,
        packager=run.GitArchivePackager(),
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor


def local_executor_torchrun() -> run.LocalExecutor:
    env_vars = {
        # required for some eval benchmarks from lm-eval-harness
        "HF_DATASETS_TRUST_REMOTE_CODE": "1"
    }

    executor = run.LocalExecutor(env_vars=env_vars)

    return executor


def main():
    args = get_parser().parse_args()
    if args.tag and not args.tag.startswith("-"):
        args.tag = "-" + args.tag

    exp_name = "NeMoEvaluation"
    deploy_fn = run.Partial(
        deploy,
        nemo_checkpoint=args.nemo_checkpoint,
        fastapi_port=args.fastapi_port,
        triton_http_address=args.triton_http_address,
        max_input_len=args.max_input_len,
        tensor_parallelism_size=args.tensor_parallelism_size,
        pipeline_parallelism_size=args.pipeline_parallelism_size,
        max_batch_size=args.batch_size,
    )

    api_endpoint = run.Config(
        ApiEndpoint,
        url=f"http://{args.triton_http_address}:{args.fastapi_port}/v1/{ENDPOINT_TYPES[args.endpoint_type]}",
        type=args.endpoint_type,
    )
    eval_target = run.Config(EvaluationTarget, api_endpoint=api_endpoint)
    eval_params = run.Config(
        ConfigParams,
        limit_samples=args.limit,
        parallelism=args.parallel_requests,
        request_timeout=args.request_timeout,
    )
    eval_config = run.Config(EvaluationConfig, type=args.eval_task, params=eval_params)

    eval_fn = run.Partial(evaluate, target_cfg=eval_target, eval_cfg=eval_config)

    executor: run.Executor
    executor_eval: run.Executor
    if args.slurm:
        # TODO: Set your custom parameters for the Slurm Executor.
        executor = slurm_executor(
            user="",
            host="",
            remote_job_dir="",
            account="",
            partition="",
            nodes=args.nodes,
            devices=args.devices,
            container_image=args.container_image,
            custom_mounts=[],
        )
        executor.srun_args = ["--mpi=pmix", "--overlap", "--ntasks-per-node=1"]
        executor_eval = executor.clone()
    else:
        executor = local_executor_torchrun()
        executor_eval = None

    with run.Experiment(f"{exp_name}{args.tag}") as exp:
        if args.slurm:
            exp.add(
                [deploy_fn, eval_fn],
                executor=[executor, executor_eval],
                name=exp_name,
                tail_logs=True if isinstance(executor, run.LocalExecutor) else False,
            )
        else:
            exp.add(deploy_fn, executor=executor, name=f"{exp_name}_deploy")
            exp.add(eval_fn, executor=executor, name=f"{exp_name}_evaluate")

        if args.dryrun:
            exp.dryrun()
        else:
            exp.run()


if __name__ == "__main__":
    main()
