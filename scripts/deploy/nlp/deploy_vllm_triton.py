# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile

from nemo.deploy import DeployPyTriton

# Configure the NeMo logger to look the same as vLLM
logging.basicConfig(format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s", datefmt="%m-%d %H:%M:%S")
LOGGER = logging.getLogger("NeMo")

try:
    from nemo.export.vllm_exporter import vLLMExporter
except Exception as e:
    LOGGER.error(f"Cannot import the vLLM exporter. {type(e).__name__}: {e}")
    sys.exit(1)


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Export NeMo models to vLLM and deploy them on Triton",
    )
    parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source .nemo file")
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        required=True,
        choices=["llama", "mistral", "mixtral", "starcoder2", "gemma"],
        help="Type of the model",
    )
    parser.add_argument("-tmn", "--triton_model_name", required=True, type=str, help="Name for the service")
    parser.add_argument("-tmv", "--triton_model_version", default=1, type=int, help="Version for the service")
    parser.add_argument(
        "-trp", "--triton_port", default=8000, type=int, help="Port for the Triton server to listen for requests"
    )
    parser.add_argument(
        "-tha", "--triton_http_address", default="0.0.0.0", type=str, help="HTTP address for the Triton server"
    )
    parser.add_argument(
        "-tmr", "--triton_model_repository", default=None, type=str, help="Folder for the vLLM conversion"
    )
    parser.add_argument("-tps", "--tensor_parallelism_size", default=1, type=int, help="Tensor parallelism size")
    parser.add_argument(
        "-dt",
        "--dtype",
        choices=["bfloat16", "float16", "fp8", "int8"],
        default="bfloat16",
        type=str,
        help="dtype of the model on vLLM",
    )
    parser.add_argument(
        "-mml", "--max_model_len", default=512, type=int, help="Max input + ouptut length of the model"
    )
    parser.add_argument("-mbs", "--max_batch_size", default=8, type=int, help="Max batch size of the model")
    parser.add_argument(
        "-lc", "--lora_ckpt", default=[], type=str, nargs="+", help="List of LoRA checkpoints in HF format"
    )
    parser.add_argument(
        "-es", '--enable_streaming', default=False, action='store_true', help="Enables streaming sentences."
    )
    parser.add_argument("-dm", "--debug_mode", default=False, action='store_true', help="Enable debug mode")
    parser.add_argument(
        '-ws',
        '--weight_storage',
        default='auto',
        choices=['auto', 'cache', 'file', 'memory'],
        help='Strategy for storing converted weights for vLLM: "file" - always write weights into a file, '
        '"memory" - always do an in-memory conversion, "cache" - reuse existing files if they are '
        'newer than the nemo checkpoint, "auto" - use "cache" for multi-GPU runs and "memory" '
        'for single-GPU runs.',
    )
    parser.add_argument(
        "-gmu",
        '--gpu_memory_utilization',
        default=0.9,
        type=float,
        help="GPU memory utilization percentage for vLLM.",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        choices=["fp8"],
        help="Quantization method for vLLM.",
    )
    args = parser.parse_args(argv)
    return args


def get_vllm_deployable(args, model_dir):
    exporter = vLLMExporter()
    exporter.export(
        nemo_checkpoint=args.nemo_checkpoint,
        model_dir=model_dir,
        model_type=args.model_type,
        tensor_parallel_size=args.tensor_parallelism_size,
        max_model_len=args.max_model_len,
        lora_checkpoints=args.lora_ckpt,
        dtype=args.dtype,
        weight_storage=args.weight_storage,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
    )
    return exporter


def nemo_deploy(argv):
    args = get_args(argv)

    if args.debug_mode:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    LOGGER.setLevel(loglevel)
    LOGGER.info("Logging level set to {}".format(loglevel))
    LOGGER.info(args)

    # If no model_dir was supplied, create a temporary directory.
    # This directory should persist while the model is being served, becaue it may contain
    # converted LoRA checkpoints, and those are accessed by vLLM at request time.
    tempdir = None
    model_dir = args.triton_model_repository
    if model_dir is None:
        tempdir = tempfile.TemporaryDirectory()
        model_dir = tempdir.name
        LOGGER.info(
            f"{model_dir} will be used for the vLLM intermediate folder. "
            + "Please set the --triton_model_repository parameter if you'd like to use a path that already "
            + "includes the vLLM model files."
        )
    elif not os.path.exists(model_dir):
        os.makedirs(model_dir)

    try:
        triton_deployable = get_vllm_deployable(args, model_dir=model_dir)

        nm = DeployPyTriton(
            model=triton_deployable,
            triton_model_name=args.triton_model_name,
            triton_model_version=args.triton_model_version,
            max_batch_size=args.max_batch_size,
            port=args.triton_port,
            address=args.triton_http_address,
            streaming=args.enable_streaming,
        )

        LOGGER.info("Starting the Triton server...")
        nm.deploy()
        nm.serve()

        LOGGER.info("Stopping the Triton server...")
        nm.stop()

    except Exception as error:
        LOGGER.error("An error has occurred while setting up or serving the model. Error message: " + str(error))
        return

    # Clean up the temporary directory
    finally:
        if tempdir is not None:
            tempdir.cleanup()


if __name__ == '__main__':
    nemo_deploy(sys.argv[1:])
