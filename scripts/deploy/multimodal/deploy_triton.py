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
import logging
import os
import sys
from pathlib import Path

from nemo.deploy import DeployPyTriton

LOGGER = logging.getLogger("NeMo")

multimodal_supported = True
try:
    from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter
except Exception as e:
    LOGGER.warning(f"Cannot import the TensorRTMMExporter exporter, it will not be available. {type(e).__name__}: {e}")
    multimodal_supported = False


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton",
    )
    parser.add_argument("-vc", "--visual_checkpoint", type=str, help="Source .nemo file for visual model")
    parser.add_argument(
        "-lc",
        "--llm_checkpoint",
        type=str,
        required=False,
        help="Source .nemo file for llm",
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        required=True,
        choices=["neva", "video-neva", "lita", "vila", "vita"],
        help="Type of the model that is supported.",
    )
    parser.add_argument(
        "-lmt",
        "--llm_model_type",
        type=str,
        required=True,
        choices=["gptnext", "gpt", "llama", "falcon", "starcoder", "mixtral", "gemma"],
        help="Type of LLM. gptnext, gpt, llama, falcon, and starcoder are only supported."
        " gptnext and gpt are the same and keeping it for backward compatibility",
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
        "-tmr", "--triton_model_repository", default=None, type=str, help="Folder for the trt-llm conversion"
    )
    parser.add_argument("-ng", "--num_gpus", default=1, type=int, help="Number of GPUs for the deployment")
    parser.add_argument(
        "-dt",
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        type=str,
        help="dtype of the model on TensorRT",
    )
    parser.add_argument("-mil", "--max_input_len", default=4096, type=int, help="Max input length of the model")
    parser.add_argument("-mol", "--max_output_len", default=256, type=int, help="Max output length of the model")
    parser.add_argument("-mbs", "--max_batch_size", default=1, type=int, help="Max batch size of the llm model")
    parser.add_argument("-mml", "--max_multimodal_len", default=3072, type=int, help="Max length of multimodal input")
    parser.add_argument(
        "-vmb",
        "--vision_max_batch_size",
        default=1,
        type=int,
        help="Max batch size of the visual inputs, for lita/vita model with video inference, this should be set to 256",
    )
    args = parser.parse_args(argv)
    return args


def get_trt_deployable(args):
    if args.triton_model_repository is None:
        trt_path = "/tmp/trt_model_dir/"
        LOGGER.info(
            "/tmp/trt_model_dir/ path will be used as the TensorRT folder. "
            "Please set the --triton_model_repository parameter if you'd like to use a path that already "
            "includes the TensorRT model files."
        )
        Path(trt_path).mkdir(parents=True, exist_ok=True)
    else:
        trt_path = args.triton_model_repository

    if args.visual_checkpoint is None and args.triton_model_repository is None:
        raise ValueError(
            "The provided model repository is not a valid TensorRT model "
            "directory. Please provide a --visual_checkpoint."
        )

    if args.visual_checkpoint is None and not os.path.isdir(args.triton_model_repository):
        raise ValueError(
            "The provided model repository is not a valid TensorRT model "
            "directory. Please provide a --visual_checkpoint."
        )

    if args.visual_checkpoint is not None and args.model_type is None:
        raise ValueError("Model type is required to be defined if a nemo checkpoint is provided.")

    exporter = TensorRTMMExporter(
        model_dir=trt_path,
        load_model=(args.visual_checkpoint is None),
    )

    if args.visual_checkpoint is not None:
        try:
            LOGGER.info("Export operation will be started to export the nemo checkpoint to TensorRT.")
            exporter.export(
                visual_checkpoint_path=args.visual_checkpoint,
                llm_checkpoint_path=args.llm_checkpoint,
                model_type=args.model_type,
                llm_model_type=args.llm_model_type,
                tensor_parallel_size=args.num_gpus,
                max_input_len=args.max_input_len,
                max_output_len=args.max_output_len,
                vision_max_batch_size=args.vision_max_batch_size,
                max_batch_size=args.max_batch_size,
                max_multimodal_len=args.max_multimodal_len,
                dtype=args.dtype,
            )
        except Exception as error:
            raise RuntimeError("An error has occurred during the model export. Error message: " + str(error))

    return exporter


def nemo_deploy(argv):
    args = get_args(argv)

    loglevel = logging.INFO

    LOGGER.setLevel(loglevel)
    LOGGER.info("Logging level set to {}".format(loglevel))
    LOGGER.info(args)

    triton_deployable = get_trt_deployable(args)

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
