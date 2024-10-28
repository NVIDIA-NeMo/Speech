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

# NOTE: This script is just an example of using NeMo checkpoints for generating outputs and is subject to change without notice.

import argparse
import torch
import torch.distributed
from megatron.core.inference.common_inference_params import CommonInferenceParams

import nemo.lightning as nl
from nemo.collections.llm import api


def get_args():
    parser = argparse.ArgumentParser(description='Train a small T5 model using NeMo 2.0')
    parser.add_argument('--devices', type=int, help="Number of devices to use for training.")
    parser.add_argument('--checkpoint-path', type=str, help="Path to trained model.")
    parser.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    parser.add_argument("--top_k", type=int, default=1, help='Top k sampling.')
    parser.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    parser.add_argument(
        "--num-tokens-to-generate", type=int, default=30, help='Number of tokens to generate for each prompt.'
    )
    parser.add_argument(
        "--prompts",
        metavar='N',
        type=str,
        nargs='+',
        help='Prompts with each prompt within quotes and seperated by space.',
    )
    parser.add_argument(
        "--encoder-prompts",
        metavar='N',
        type=str,
        nargs='+',
        help='Encoder input prompts with each prompt within quotes and seperated by space.',
    )
    parser.add_argument("--max-batch-size", type=int, default=1, help='Max number of prompts to process at once.')

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=1,
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
        "",
        "",
        "",
    ]
    encoder_prompts = [
        "Hello, how are <mask>?",
        "How many r's are in the <mask> 'strawberry'?",
        "Which number is <mask>? 10.119 <mask> 10.19?",
    ]

    results = api.generate(
        path=args.checkpoint_path,
        prompts=prompts,
        encoder_prompts=encoder_prompts,
        trainer=trainer,
        add_BOS=True,
        inference_params=CommonInferenceParams(
            temperature=args.temperature, top_k=args.top_k, num_tokens_to_generate=args.num_tokens_to_generate
        ),
        text_only=True,
    )
    if torch.distributed.get_rank() == 0:
        for i, r in enumerate(results):
            print(prompts[i])
            print("*" * 50)
            print(r)
            print("\n\n")
