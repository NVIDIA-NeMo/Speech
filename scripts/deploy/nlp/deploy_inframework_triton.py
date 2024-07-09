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

from nemo.deploy import DeployPyTriton

LOGGER = logging.getLogger("NeMo")

megatron_llm_supported = True
try:
    from nemo.deploy.nlp import MegatronLLMDeployable
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
    parser.add_argument("-mbs", "--max_batch_size", default=8, type=int, help="Max batch size of the model")
    parser.add_argument("-dm", "--debug_mode", default=False, action='store_true', help="Enable debug mode")
    args = parser.parse_args(argv)
    return args


def get_nemo_deployable(args):
    if args.nemo_checkpoint is None:
        raise ValueError("In-Framework deployment requires a .nemo checkpoint")

    return MegatronLLMDeployable(args.nemo_checkpoint, args.num_gpus)


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
    triton_deployable = get_nemo_deployable(args)

    try:
        nm = DeployPyTriton(
            model=triton_deployable,
            triton_model_name=args.triton_model_name,
            triton_model_version=args.triton_model_version,
            max_batch_size=args.max_batch_size,
            port=args.triton_port,
            address=args.triton_http_address,
        )

        LOGGER.info("Triton deploy function will be called.")
        nm.deploy()
    except Exception as error:
        LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
        return

    try:
        LOGGER.info("Model serving on Triton is will be started.")
        nm.serve()
    except Exception as error:
        LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
        return

    LOGGER.info("Model serving will be stopped.")
    nm.stop()


if __name__ == '__main__':
    nemo_deploy(sys.argv[1:])
