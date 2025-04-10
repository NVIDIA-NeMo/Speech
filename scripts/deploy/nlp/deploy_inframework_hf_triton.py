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
import os
import sys

import torch
import torch.distributed as dist

from nemo.deploy import DeployPyTriton
from nemo.deploy.nlp.hf_deployable import HuggingFaceLLMDeploy

LOGGER = logging.getLogger("NeMo")


def setup_torch_dist(rank, world_size):
    """Sets up PyTorch distributed training environment.

    Args:
        rank (int): The rank of the current process
        world_size (int): Total number of processes for distributed training
    """

    torch.cuda.set_device(rank)
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def get_args(argv):
    """Get command line arguments for deploying HuggingFace models to Triton.

    Returns:
        argparse.Namespace: Parsed command line arguments including:
            - hf_model_id_path: Path to HuggingFace model
            - task: Model task type (text-generation)
            - device_map: Device mapping strategy
            - tp_plan: Tensor parallelism plan
            - trust_remote_code: Whether to trust remote code
            - triton_model_name: Name for model in Triton
            - triton_model_version: Model version number
            - triton_port: Triton HTTP port
            - triton_http_address: Triton HTTP address
            - max_batch_size: Maximum inference batch size
            - debug_mode: Enable debug logging
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Deploy HuggingFace models to Triton Inference Server",
    )
    parser.add_argument(
        "-hp",
        "--hf_model_id_path",
        type=str,
        help="Path to local HuggingFace " "model directory or model ID from HuggingFace " "Hub",
    )
    parser.add_argument(
        "-t",
        "--task",
        nargs='?',
        choices=['text-generation'],
        default="text-generation",
        type=str,
        help="Task type for the HuggingFace model (currently only text-generation is supported)",
    )
    parser.add_argument(
        "-dvm",
        "--device_map",
        nargs='?',
        choices=['auto', 'balanced', 'balanced_low_0', 'sequential'],
        default=None,
        type=str,
        help="Device mapping " "strategy for model placement " "(e.g. 'auto', 'sequential', etc)",
    )
    parser.add_argument(
        "-tpp",
        "--tp_plan",
        nargs='?',
        choices=['auto'],
        default=None,
        type=str,
        help="Tensor parallelism plan for distributed inference",
    )
    parser.add_argument(
        "-trc",
        "--trust_remote_code",
        default=False,
        action='store_true',
        help="Allow loading " "remote code from HuggingFace " "Hub",
    )
    parser.add_argument(
        "-tmn", "--triton_model_name", required=True, type=str, help="Name to " "identify the model in " "Triton"
    )
    parser.add_argument(
        "-tmv", "--triton_model_version", default=1, type=int, help="Version " "number for the model " "in Triton"
    )
    parser.add_argument(
        "-trp", "--triton_port", default=8000, type=int, help="Port number for Triton server " "HTTP endpoint"
    )
    parser.add_argument(
        "-tha",
        "--triton_http_address",
        default="0.0.0.0",
        type=str,
        help="Network interface " "address for Triton HTTP endpoint",
    )
    parser.add_argument(
        "-mbs", "--max_batch_size", default=8, type=int, help="Maximum " "batch size for model inference"
    )
    parser.add_argument(
        "-dm", "--debug_mode", default=False, action='store_true', help="Enable " "verbose debug logging"
    )
    args = parser.parse_args(argv)
    return args


def hf_deploy(argv):
    """Deploy a HuggingFace model to Triton Inference Server.

    This function handles the deployment workflow including:
    - Parsing command line arguments
    - Setting up distributed training if needed
    - Initializing the HuggingFace model
    - Starting the Triton server

    Args:
        argv: Command line arguments

    Raises:
        ValueError: If required arguments are missing or invalid
    """

    args = get_args(argv)

    if args.debug_mode:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    LOGGER.setLevel(loglevel)
    LOGGER.info("Logging level set to {}".format(loglevel))
    LOGGER.info(args)

    if args.hf_model_id_path is None:
        raise ValueError("In-Framework deployment requires a Hugging Face model ID or path.")

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size > 1:
            setup_torch_dist(rank, world_size)
    else:
        if args.device_map == "auto":
            LOGGER.warning(
                "device_map is set to auto and it is recommended that the script"
                "is started with torchrun with a process per GPU. You might "
                "see unexpected issues during the inference otherwise."
            )

        if args.tp_plan is not None:
            raise ValueError("tp_plan is only available with torchrun.")

    hf_deployable = HuggingFaceLLMDeploy(
        hf_model_id_path=args.hf_model_id_path,
        task=args.task,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        tp_plan=args.tp_plan,
    )

    start_triton_server = True
    if dist.is_initialized():
        if dist.get_rank() > 0:
            start_triton_server = False

    if start_triton_server:
        try:
            nm = DeployPyTriton(
                model=hf_deployable,
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
            if dist.is_initialized():
                dist.barrier()
            return

        try:
            LOGGER.info("Model serving on Triton will be started.")
            nm.serve()
        except Exception as error:
            LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))

        if dist.is_initialized():
            if dist.get_world_size() > 1:
                torch.distributed.broadcast(torch.tensor([1], dtype=torch.long, device="cuda"), src=0)

        LOGGER.info("Model serving will be stopped.")
        nm.stop()
    else:
        if dist.is_initialized():
            if dist.get_rank() > 0:
                hf_deployable.generate_other_ranks()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    hf_deploy(sys.argv[1:])
