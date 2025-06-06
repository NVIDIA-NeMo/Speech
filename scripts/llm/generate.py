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

"""
torchrun --nproc-per-node=8 /opt/NeMo/scripts/llm/generate.py \
    --model_path=<PATH_TO_NEMO2_MODEL> \
    --tp=8 \
    --devices=8 \
    --num_tokens_to_generate=40 \
    --temperature=0.001 \
    --top_p=0.0 \
    --top_k=1 \
    --fp8
"""


def get_args():
    """
    Parse the command line arguments.
    """
    parser = ArgumentParser(description="""Run generation on a few sample prompts given the checkpoint path.""")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "Q: How are you?",
            "Q: How big is the universe?",
            "Q: How is the weather?",
            "Q: How many stars are there?",
            "Paris is know for its ",
            "In a hot sunny day, you should ",
            "Q: How many planets are in the solar system?",
            "Q: How old are you?",
        ],
        help="List of prompt strings",
    )
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
        "--ep",
        type=int,
        default=1,
        help="""Expert parallel size""",
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
        "--add_BOS",
        action="store_true",
        help="""Whether to add BOS token to the prompt""",
    )
    parser.add_argument(
        "--num_tokens_to_generate",
        type=int,
        default=25,
        help="""Number of tokens to generate per prompt""",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="""Whether to run inference in FP8 precision""",
    )
    parser.add_argument(
        "--fp8_recipe",
        type=str,
        default="tensorwise",
        help="""fp8 recipe, can be 'tensorwise', 'delayed', or 'mxfp8'""",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="""Maximum batch size for inference""",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1234,
        help="""Random seed for generation""",
    )
    parser.add_argument(
        "--legacy_ckpt",
        action="store_true",
        help="""Load ckpt saved with TE < 1.14""",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if args.fp8:
        assert len(args.prompts) % 8 == 0, "Batch size should be divisible by 8 for FP8 inference"

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=1 if args.ep > 1 else None,
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
            fp8="hybrid" if args.fp8 else None,
            fp8_recipe=args.fp8_recipe if args.fp8 else None,
            fp8_amax_history_len=1,
            fp8_amax_compute_algo="max" if args.fp8 else "most_recent",
        ),
    )

    # Load ckpt saved with TE < 1.14
    if args.legacy_ckpt:
        trainer.strategy.ckpt_load_strictness = False

    prompts = args.prompts

    results = api.generate(
        path=args.model_path,
        prompts=prompts,
        trainer=trainer,
        add_BOS=args.add_BOS,
        inference_params=CommonInferenceParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_tokens_to_generate=args.num_tokens_to_generate,
        ),
        text_only=True,
        max_batch_size=args.max_batch_size,
        random_seed=args.random_seed,
    )
    if torch.distributed.get_rank() == 0:
        for i, r in enumerate(results):
            print(prompts[i])
            print("*" * 50)
            print(r)
            print("\n\n")
