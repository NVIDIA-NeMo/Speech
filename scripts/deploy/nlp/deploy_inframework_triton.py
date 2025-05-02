# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import sys
import torch

from nemo.deploy import DeployPyTriton

LOGGER = logging.getLogger("NeMo")

megatron_llm_supported = True
try:
    from nemo.deploy.nlp.megatronllm_deployable import MegatronLLMDeployableNemo2
except Exception as e:
    LOGGER.warning(f"Cannot import MegatronLLMDeployable, it will not be available. {type(e).__name__}: {e}")
    megatron_llm_supported = False


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton",
    )
    parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source .nemo file")
    parser.add_argument("-tmn", "--triton_model_name", required=True, type=str, help="Name for the service")
    parser.add_argument("-tmv", "--triton_model_version", default=1, type=int, help="Version for the service")
    parser.add_argument(
        "-trp", "--triton_port", default=8000, type=int, help="Port for the Triton server to listen for requests"
    )
    parser.add_argument(
        "-tha", "--triton_http_address", default="0.0.0.0", type=str, help="HTTP address for the Triton server"
    )
    parser.add_argument("-ng", "--num_gpus", default=1, type=int, help="Number of GPUs for the deployment")
    parser.add_argument("-nn", "--num_nodes", default=1, type=int, help="Number of GPUs for the deployment")
    parser.add_argument("-tps", "--tensor_parallelism_size", default=1, type=int, help="Tensor parallelism size")
    parser.add_argument("-pps", "--pipeline_parallelism_size", default=1, type=int, help="Pipeline parallelism size")
    parser.add_argument("-cps", "--context_parallel_size", default=1, type=int, help="Pipeline parallelism size")
    parser.add_argument(
        "-emps",
        "--expert_model_parallel_size",
        default=1,
        type=int,
        help="Distributes MoE Experts across sub data parallel dimension.",
    )
    parser.add_argument("-mbs", "--max_batch_size", default=8, type=int, help="Max batch size of the model")
    parser.add_argument("-dm", "--debug_mode", default=False, action='store_true', help="Enable debug mode")
    parser.add_argument(
        "-fd",
        '--enable_flash_decode',
        default=False,
        action='store_true',
        help='Enable flash decoding',
    )
    parser.add_argument("-lc", "--legacy_ckpt", action="store_true", help="Load checkpoint saved with TE < 1.14")
    args = parser.parse_args(argv)
    return args


def nemo_deploy(argv):
    args = get_args(argv)

    if args.debug_mode:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    LOGGER.setLevel(loglevel)
    LOGGER.info("Logging level set to {}".format(loglevel))
    LOGGER.info(args)

    if not megatron_llm_supported:
        raise ValueError("MegatronLLMDeployable is not supported in this environment.")

    if args.nemo_checkpoint is None:
        raise ValueError("In-Framework deployment requires a checkpoint folder.")

    model = MegatronLLMDeployableNemo2(
        nemo_checkpoint_filepath=args.nemo_checkpoint,
        num_devices=args.num_gpus,
        num_nodes=args.num_nodes,
        tensor_model_parallel_size=args.tensor_parallelism_size,
        pipeline_model_parallel_size=args.pipeline_parallelism_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        max_batch_size=args.max_batch_size,
        enable_flash_decode=args.enable_flash_decode,
        legacy_ckpt=args.legacy_ckpt,
    )

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            try:
                nm = DeployPyTriton(
                    model=model,
                    triton_model_name=args.triton_model_name,
                    triton_model_version=args.triton_model_version,
                    max_batch_size=args.max_batch_size,
                    http_port=args.triton_port,
                    address=args.triton_http_address,
                )

                LOGGER.info("Triton deploy function will be called.")
                nm.deploy()
            except Exception as error:
                LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
                return

            try:
                LOGGER.info("Model serving on Triton will be started.")
                nm.serve()
            except Exception as error:
                LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
                return

            torch.distributed.broadcast(torch.tensor([1], dtype=torch.long, device="cuda"), src=0)

            LOGGER.info("Model serving will be stopped.")
            nm.stop()
        elif torch.distributed.get_rank() > 0:
            model.generate_other_ranks()

    else:
        LOGGER.info("Torch distributed wasn't initialized.")


if __name__ == '__main__':
    nemo_deploy(sys.argv[1:])
