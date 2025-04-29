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
from nemo.collections.llm.api import deploy

# NOTE: This script is an example script to deploy a nemo2 model in-framework (i.e wo converting the model to any
# other model) on PyTriton server by exposing the OpenAI API endpoints (v1/completions and v1/chat/completions).
# The intended use case of this script is to run evaluations with NVIDIA LM-Evaluation-Harness.


def get_parser():
    parser = argparse.ArgumentParser(description="NeMo2.0 Deployment")
    parser.add_argument(
        "--nemo_checkpoint",
        type=str,
        help="NeMo 2.0 checkpoint to be evaluated",
    ),
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Num of gpus per node",
    ),
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Num of nodes",
    ),
    parser.add_argument(
        "--tensor_parallelism_size",
        type=int,
        default=1,
        help="Tensor parallelism size to deploy the model",
    ),
    parser.add_argument(
        "--pipeline_parallelism_size",
        type=int,
        default=1,
        help="Pipeline parallelism size to deploy the model",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="context parallelism size to deploy the model",
    )
    parser.add_argument(
        "--expert_model_parallel_size",
        type=int,
        default=1,
        help="Expert model parallelism size to deploy the model",
    )
    parser.add_argument(
        "--expert_tensor_parallel_size",
        type=int,
        default=1,
        help="Expert tensor parallelism size to deploy the model",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Max batch size for the underlying Triton server",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    deploy(
        nemo_checkpoint=args.nemo_checkpoint,
        num_gpus=args.ngpus,
        num_nodes=args.nnodes,
        fastapi_port=8886,
        tensor_parallelism_size=args.tensor_parallelism_size,
        pipeline_parallelism_size=args.pipeline_parallelism_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        max_batch_size=args.max_batch_size,
    )
