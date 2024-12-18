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

# NOTE: This script is just an example of using NeMo checkpoints
# for generating outputs and is subject to change without notice.

from argparse import ArgumentParser

import torch
import torch.distributed
from megatron.core.inference.common_inference_params import CommonInferenceParams

import nemo.lightning as nl
from nemo.collections.llm import api


def get_args():
    """
    Parse the command line arguments.
    """
    parser = ArgumentParser(description="""Run generation on a few sample prompts given the checkpoint path.""")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="""Path to NeMo 2 checkpoint""",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="""Tensor parallel size""",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="""Pipeline parallel size""",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="""Number of GPUs to use on a single node""",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="""Number of nodes to use""",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="""Temperature to be used in megatron.core.inference.common_inference_params.CommonInferenceParams""",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="""top_p to be used in megatron.core.inference.common_inference_params.CommonInferenceParams""",
    )
    parser.add_argument(
        "--top_k",
        type=float,
        default=0,
        help="""top_k to be used in megatron.core.inference.common_inference_params.CommonInferenceParams""",
    )
    parser.add_argument(
        "--num_tokens_to_generate",
        type=int,
        default=4,
        help="""Number of tokens to generate per prompt""",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=1,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.nodes,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
    )
    prompts = [
        "Hello, how are you?",
        "How many r's are in the word 'strawberry'?",
        "Which number is bigger? 10.119 or 10.19?",
    ]
    results = api.generate(
        path=args.model_path,
        prompts=prompts,
        trainer=trainer,
        inference_params=CommonInferenceParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_tokens_to_generate=args.num_tokens_to_generate,
        ),
        text_only=True,
    )
    if torch.distributed.get_rank() == 0:
        for i, r in enumerate(results):
            print(prompts[i])
            print("*" * 50)
            print(r)
            print("\n\n")
